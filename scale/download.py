"""Download the PPanGGOLiN file of every bacterial pangenome of PanGBank's GTDB_refseq (latest release).

PanGBank's rule is at most one HTTP request per 30 seconds, across every route, never in parallel:
this process is the only one that talks to PanGBank, and it leaves 31 s between the starts of two
requests. The list comes from /pangenomes/ (count first, pages of 100); archaea are left out. Files
are downloaded smallest first to raw/<id>.h5, checked against file_md5sum, and announced by raw/<id>.ok
for extract.py. At most BUFFER_GB of files wait for extraction at any time, so little is on disk.

    python download.py [--only ID ...] [--list-only]
"""
import os, sys, json, time, hashlib, argparse, urllib.request
ROOT = os.environ.get("PGG_ROOT", "."); API = "https://pangbank-api.genoscope.cns.fr"
GAP = 31.0; BUFFER_GB = float(os.environ.get("PGG_BUFFER_GB", 20)); _last = [0.0]
RAW, OUT, STATE = (os.path.join(ROOT, x) for x in ("raw", "out", "state"))


LAST = os.path.join(STATE, "pangbank.last")          # the previous request's time, kept across processes


def request(path, stream_to=None):
    """one request to PanGBank, never sooner than GAP seconds after the previous one, from any run"""
    try: _last[0] = max(_last[0], float(open(LAST).read()))
    except (OSError, ValueError): pass
    wait = _last[0] + GAP - time.time()
    if wait > 0: time.sleep(wait)
    _last[0] = time.time(); open(LAST, "w").write(str(_last[0]))
    r = urllib.request.urlopen(urllib.request.Request(API + path, headers={"User-Agent": "pangramgraph-batch"}), timeout=600)
    if stream_to is None: return json.loads(r.read())
    md5 = hashlib.md5()
    with open(stream_to, "wb") as f:
        while True:
            b = r.read(1 << 22)
            if not b: break
            f.write(b); md5.update(b)
    return md5.hexdigest()


def species_list():
    p = os.path.join(STATE, "species.json")
    if os.path.exists(p): return json.load(open(p))
    q = "collection_name=GTDB_refseq&only_latest_release=true"
    n = request(f"/pangenomes/count/?{q}"); rows = []
    for off in range(0, n, 100):
        rows += request(f"/pangenomes/?{q}&limit=100&offset={off}")
        print(f"  listed {len(rows)}/{n}", flush=True)
    keep = []
    for r in rows:
        taxa = (r.get("taxonomy") or {}).get("taxa") or []
        dom = next((t["name"] for t in taxa if t.get("rank") == "Domain"), "")
        if dom != "d__Bacteria": continue
        keep.append(dict(id=r["id"], name=r["name"], species=next((t["name"] for t in taxa if t.get("rank") == "Species"), r["name"]),
                         genomes=r["genome_count"], genes=r["gene_count"], families=r["family_count"], md5=r["file_md5sum"],
                         release=(r.get("collection_release") or {}).get("version")))
    keep.sort(key=lambda r: r["genes"])
    json.dump(dict(total=n, bacteria=len(keep), archaea=n - len(keep), species=keep), open(p, "w"))
    return json.load(open(p))


def buffered_gb():
    return sum(os.path.getsize(os.path.join(RAW, f)) for f in os.listdir(RAW) if f.endswith(".h5")) / 1e9


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*", type=int); ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args()
    for d in (RAW, OUT, STATE): os.makedirs(d, exist_ok=True)
    L = species_list(); sp = L["species"]
    print(f"GTDB_refseq latest: {L['total']} pangenomes, {L['bacteria']} bacterial, {L['archaea']} archaeal left out; "
          f"{sum(r['genomes'] for r in sp)} genomes, {sum(r['genes'] for r in sp) / 1e6:.0f} M genes", flush=True)
    if a.list_only: sys.exit(0)
    if a.only: sp = [r for r in sp if r["id"] in set(a.only)]
    sp = sorted(sp, key=lambda r: -r["genes"])          # the largest first: the GPU has work while the rest downloads
    MAX_GB = float(os.environ.get("PGG_MAX_GB", 180))   # all that is kept, beyond which downloading pauses
    kept = lambda: sum(e.stat().st_size for d in os.scandir(OUT) if d.is_dir() for e in os.scandir(d.path) if e.is_file()) / 1e9
    for k, r in enumerate(sp):
        if k % 20 == 0:
            while kept() + buffered_gb() > MAX_GB: print("disk cap reached, waiting", flush=True); time.sleep(600)
        pg = str(r["id"]); h5 = os.path.join(RAW, pg + ".h5")
        if os.path.exists(os.path.join(OUT, pg, "extract.ok")) or os.path.exists(os.path.join(OUT, pg, "gpu.ok")) \
           or any(f.startswith(pg + ".") for f in os.listdir(RAW)):
            continue                                       # extracted, done, or waiting / being extracted
        while buffered_gb() > BUFFER_GB: time.sleep(60)    # the disk holds little at a time
        for attempt in range(3):
            try:
                md5 = request(f"/pangenomes/{pg}/file", stream_to=h5 + ".part")
                if md5 != r["md5"]: raise IOError(f"md5 {md5} != {r['md5']}")
                os.rename(h5 + ".part", h5); open(os.path.join(RAW, pg + ".ok"), "w").write(md5)
                print(f"{pg} {r['species']}: {r['genomes']} genomes, {os.path.getsize(h5) / 1e6:.0f} MB", flush=True)
                break
            except Exception as e:
                print(f"{pg}: attempt {attempt + 1} failed: {e}", flush=True)
                if os.path.exists(h5 + ".part"): os.remove(h5 + ".part")
        else:
            open(os.path.join(STATE, "download.failed"), "a").write(pg + "\n")
    if not a.only: open(os.path.join(STATE, "download.finished"), "w").write(str(time.time()))
