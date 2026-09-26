"""PanGBank's own ids of the 2,002 genomes of pangenome 11587, for links from the page to
https://pangbank.genoscope.cns.fr/pangenome/11587/genome/<id>. PanGBank asks for at most one request
per 30 seconds from one place: the pages of 100 are fetched 31 s apart (about 11 minutes).

    python3 pgb_pangbank_ids.py        # -> pgb/pangbank_genome_ids.json {accession: genome id}
"""
import json, re, time, urllib.request
API, PG, OUT = "https://pangbank-api.genoscope.cns.fr", 11587, "pgb/pangbank_genome_ids.json"
get = lambda p: json.loads(urllib.request.urlopen(urllib.request.Request(API + p, headers={"User-Agent": "pangramgraph"}), timeout=120).read())
rows, off = [], 0
while True:
    page = get(f"/pangenomes/{PG}/genomes?limit=100&offset={off}")
    rows += page; print(f"{len(rows)} genomes", flush=True)
    if len(page) < 100: break
    off += 100; time.sleep(31)
json.dump(rows, open("pgb/pangbank_genomes_raw.json", "w"), separators=(",", ":"))     # every row, in case more is needed
ids = {}
for r in rows:
    m = re.search(r"GC[AF]_\d+\.\d+", r.get("Genome_name") or r.get("genome_file_name", ""))
    if m: ids[m.group(0)] = r["genome_id"]
json.dump(dict(pangenome=PG, source=f"{API}/pangenomes/{PG}/genomes", ids=ids), open(OUT, "w"), separators=(",", ":"))
print(f"{len(ids)} of {len(rows)} genomes mapped -> {OUT}; example {rows[0]}")
