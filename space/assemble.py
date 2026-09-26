"""Put the Hugging Face Space together in space_build/: the files of space/ (Dockerfile,
README with the Space's settings, requirements, start.py), the model's server, the page and,
unless --no-data, the data the server reads (about 720 MB; with --no-data they are fetched at
start from the dataset repository PGG_DATA_REPO). The Space repository is capped at 1 GB.

    python3 build_page.py && python3 space/assemble.py [--no-data] [--dry-run] [--out DIR]
    huggingface-cli upload ggautreau/PanGramGraph space_build . --repo-type space

--dry-run lists what would go and its size without copying; --out assembles elsewhere (a local test).

INFLUENCE: what serve_live.py's /elements and /influence read (the Coupled regions tab: the model's
in-silico knockout of one accessory element, pg_knock.influence) on top of DATA: pg_knock.py, the two
link sets (long and close range), the compact lineage-CV decoder (pgb/knock/decoder_halves.npz,
pg_knock.export_decoder(), ~73 MB: the full base_top64_{f,p}.npy, ~0.9 GB, stay out) and the units cache
of each set (else rebuilt at the first request, ~10 s). Without them the Space runs as before and /health
says why influence is off. A warning when the decoder is older than the base_top64 it was exported from.
With --no-data, the dataset repository PGG_DATA_REPO must hold DATA and INFLUENCE (start.py fetches them).
"""
import os, sys, shutil
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJ); sys.path.insert(0, PROJ)
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "space_build"
CODE = ["serve_live.py", "pgb_region.py", "pg_knock.py", "fam_annot.json", "eco/manifest.json",
        "standalone/pangramgraph.html", "standalone/index.html", "standalone/origin-fork.html"]
DATA = ["pgb/window.json", "pgb/window_emb.npy", "pgb/model_calls.npz", "pgb/graph_pgb.json", "pgb/fam_info.json",
        "pgb/chrom.npz", "pgb/chrom_genomes.json", "pgb/calls_top15_f.i32", "pgb/calls_top15_p.f16",
        "pgb/calls_ent.f16", "pgb/chrom_emb.f16"]
INFLUENCE = ["pgb/region_sets.json", "pgb/region_sets_close.json", "pgb/knock/decoder_halves.npz"]
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
miss = [s for s, _ in files if not os.path.exists(s)]
if miss: sys.exit(f"missing: {', '.join(miss)}" + (" (python3 -c 'import pg_knock as K; K.export_decoder()')"
                                                     if "pgb/knock/decoder_halves.npz" in miss else ""))
size = sum(os.path.getsize(s) for s, _ in files)
infl = sum(os.path.getsize(f) for f in INFLUENCE if os.path.exists(f)) if "--no-data" not in sys.argv else 0
if "--dry-run" in sys.argv:
    for s, d in sorted(files, key=lambda x: -os.path.getsize(x[0])): print(f"{os.path.getsize(s) / 1e6:9.1f} MB  {d}")
else:
    shutil.rmtree(OUT, ignore_errors=True)
    for src, dst in files:
        os.makedirs(os.path.join(OUT, os.path.dirname(dst)), exist_ok=True); shutil.copy2(src, os.path.join(OUT, dst))
print(f"{OUT}/: {len(files)} files, {size / 1e6:.0f} MB (influence data {infl / 1e6:.0f} MB)"
      + (" -- over the Space's 1 GB: use --no-data and a dataset repository" if size > 1e9 else "")
      + (" [dry run: nothing copied]" if "--dry-run" in sys.argv else ""))
