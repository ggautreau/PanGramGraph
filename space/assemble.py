"""Put the Hugging Face Space together in space_build/: the files of space/ (Dockerfile,
README with the Space's settings, requirements, start.py), the model's server, the page, the
reading's skill and the readings written ahead, and, unless --no-data, the data the server reads
(about 670 MB, and 80 MB more for influence; with --no-data they are fetched at start from the
dataset repository PGG_DATA_REPO). The Space repository is capped at 1 GB.

    python3 build_page.py && python3 space/assemble.py [--no-data] [--dry-run] [--check-facts] [--out DIR]
    huggingface-cli upload ggautreau/PanGramGraph space_build . --repo-type space

--dry-run lists what would go and its size without copying; --out assembles elsewhere (a local test).

READINGS: the Findings tab's readings written ahead (pgb/readings/*.json, pg_reading.py pregen) are served on the
Space without a call to the LLM only if their cache key still matches: the prompt version (pg_reading.PROMPT_VERSION),
the skill document's sha (reading_skill.md, shipped too) and the facts of the window. The assembly stops when any
reading was written with another prompt version or another skill (regenerate them: python3 pg_reading.py pregen
--chain dnaA <anchors...>), since every visitor's Findings would otherwise call the LLM. --check-facts also rebuilds
every window and checks each reading's facts hash (a few minutes): a reading whose window's data changed stops it too.

INFLUENCE: what serve_live.py's /elements and /influence read (the Coupled regions tab: the model's
in-silico knockout of one accessory element, pg_knock.influence) on top of DATA: pg_knock.py, the two
link sets (long and close range), the compact lineage-CV decoder (pgb/knock/decoder_halves.npz,
pg_knock.export_decoder(), ~73 MB: the full base_top64_{f,p}.npy, ~0.9 GB, stay out) and the units cache
of each set (else rebuilt at the first request, ~10 s). Without them the Space runs as before and /health
says why influence is off. A warning when the decoder is older than the base_top64 it was exported from.
With --no-data, the dataset repository PGG_DATA_REPO must hold DATA and INFLUENCE (start.py fetches them).

build.json (the git commit, whether the tree had changes, the time of the assembly) goes with it: /health's build.
"""
import os, sys, json, time, glob, shutil, hashlib, subprocess
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJ); sys.path.insert(0, PROJ)
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "space_build"
CODE = ["serve_live.py", "pgb_region.py", "pg_knock.py", "pg_reading.py", "reading_skill.md", "fam_annot.json",
        "eco/manifest.json", "assets/favicon-32.png",
        "standalone/pangramgraph.html", "standalone/index.html", "standalone/origin-fork.html"]
DATA = ["pgb/window.json", "pgb/window_emb.npy", "pgb/model_calls.npz", "pgb/graph_pgb.json", "pgb/fam_info.json",
        "pgb/chrom.npz", "pgb/chrom_genomes.json", "pgb/calls_top15_f.i32", "pgb/calls_top15_p.f16",
        "pgb/calls_ent.f16", "pgb/chrom_emb.f16"]
INFLUENCE = ["pgb/region_sets.json", "pgb/region_sets_close.json", "pgb/knock/decoder_halves.npz"]

# the page: built from the template after its last change?
page = "standalone/pangramgraph.html"
if os.path.exists(page):
    newer = [f for f in ("page_template.html", "build_page.py") if os.path.exists(f) and os.path.getmtime(f) > os.path.getmtime(page)]
    if newer: print(f"WARNING: {' and '.join(newer)} changed after {page} was built: run python3 build_page.py first")

# the readings written ahead: the same prompt version and skill as the code shipped, else stop
import pg_reading as PR
skill_sha = PR.skill()[1]
if not PR._SKILL[2]: sys.exit(f"stop: the skill document {PR.SKILL_PATH} is missing or empty")
readings = sorted(f for f in glob.glob("pgb/readings/*.json") if not os.path.basename(f).startswith("_"))
stale, bad = [], []
for f in readings:
    try: r = json.load(open(f))
    except Exception as e: bad.append(f"{f} ({type(e).__name__})"); continue
    if r.get("prompt_version") != PR.PROMPT_VERSION or r.get("skill_sha") != skill_sha:
        stale.append(f"{os.path.basename(f)[:-5]} ({r.get('prompt_version')}, skill {r.get('skill_sha')})")
if bad or stale:
    sys.exit(f"stop: {len(stale) + len(bad)} of the {len(readings)} readings in pgb/readings/ would not be served "
             f"(the code ships prompt {PR.PROMPT_VERSION}, skill {skill_sha}): every visitor's Findings would call the LLM.\n  "
             + "\n  ".join((bad + stale)[:12]) + (f"\n  ... and {len(bad) + len(stale) - 12} more" if len(bad) + len(stale) > 12 else "")
             + "\nRegenerate them (python3 pg_reading.py pregen --chain dnaA <anchors...>), or remove the stale files.")
print(f"readings: {len(readings)} written ahead, all prompt {PR.PROMPT_VERSION} with skill {skill_sha}")
if "--check-facts" in sys.argv:                  # each reading's facts hash against its window rebuilt now
    import pgb_region as RG
    miss = []
    for f in readings:
        r = json.load(open(f)); wid = r.get("window") or os.path.basename(f)[:-5]
        if wid != PR.DNAA and RG.FIDX.get(wid) not in RG.BB_RANK: miss.append(f"{wid} (not a window anchor)"); continue
        _, h = PR.facts_of(wid, lambda w: RG.build(RG.FIDX[w]))
        if r.get("facts_hash") != h: miss.append(f"{wid} (facts {r.get('facts_hash')}, now {h})")
    if miss: sys.exit(f"stop: {len(miss)} readings do not match their window's facts any more:\n  " + "\n  ".join(miss[:12]))
    print(f"readings: the facts of all {len(readings)} windows match")

import pg_knock
for sets in (None, "pgb/region_sets_close.json"):
    uc = os.path.relpath(pg_knock.units_cache_file(sets), PROJ)
    if os.path.exists(uc): INFLUENCE.append(uc)
    else: print(f"note: no units cache {uc} yet (the Space will build it at its first request of that set, ~10 s)")
dec, top = "pgb/knock/decoder_halves.npz", ["pgb/knock/base_top64_f.npy", "pgb/knock/base_top64_p.npy"]
if os.path.exists(dec) and all(map(os.path.exists, top)) and max(map(os.path.getmtime, top)) > os.path.getmtime(dec):
    print(f"WARNING: {dec} is older than pgb/knock/base_top64_*.npy: re-export it "
          "(python3 -c 'import pg_knock as K; K.export_decoder()'), or the Space reads stale decoder rows")
# the files of space/ itself: regular files only (no __pycache__, no assemble.py)
files = [(f"space/{f}", f) for f in sorted(os.listdir("space"))
         if f != "assemble.py" and os.path.isfile(f"space/{f}") and not f.endswith(".pyc")] + [(f, f) for f in CODE]
if "--no-data" not in sys.argv: files += [(f, f) for f in DATA + INFLUENCE]
# the LLM readings already written here (pg_reading, the Findings tab): the Space serves them without a new call
files += [(f, f) for f in readings]
miss = [s for s, _ in files if not os.path.exists(s)]
if miss: sys.exit(f"missing: {', '.join(miss)}" + (" (python3 -c 'import pg_knock as K; K.export_decoder()')"
                                                     if "pgb/knock/decoder_halves.npz" in miss else ""))
size = sum(os.path.getsize(s) for s, _ in files)
infl = sum(os.path.getsize(f) for f in INFLUENCE if os.path.exists(f)) if "--no-data" not in sys.argv else 0


def git(*a):
    try: return subprocess.run(["git", *a], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception: return ""


sha = git("rev-parse", "--short=12", "HEAD") or None
build = dict(git_sha=sha, dirty=bool(git("status", "--porcelain", "--untracked-files=no")) if sha else None,
             assembled_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             page_md5=hashlib.md5(open(page, "rb").read()).hexdigest() if os.path.exists(page) else None)
if build["dirty"]: print(f"note: the working tree has uncommitted changes: /health will say git {sha} with dirty=true")
if "--dry-run" in sys.argv:
    for s, d in sorted(files, key=lambda x: -os.path.getsize(x[0])): print(f"{os.path.getsize(s) / 1e6:9.1f} MB  {d}")
    print(f"build.json: {json.dumps(build)}")
else:
    shutil.rmtree(OUT, ignore_errors=True)
    for src, dst in files:
        os.makedirs(os.path.join(OUT, os.path.dirname(dst)), exist_ok=True); shutil.copy2(src, os.path.join(OUT, dst))
    json.dump(build, open(os.path.join(OUT, "build.json"), "w"), indent=1)
print(f"{OUT}/: {len(files) + 1} files, {size / 1e6:.0f} MB (influence data {infl / 1e6:.0f} MB)"
      + (" -- over the Space's 1 GB: use --no-data and a dataset repository" if size > 1e9 else "")
      + (" [dry run: nothing copied]" if "--dry-run" in sys.argv else ""))
