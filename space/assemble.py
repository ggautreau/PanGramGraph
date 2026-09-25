"""Put the Hugging Face Space together in space_build/: the files of space/ (Dockerfile,
README with the Space's settings, requirements, start.py), the model's server, the page and,
unless --no-data, the data the server reads (about 670 MB; with --no-data they are fetched at
start from the dataset repository PGG_DATA_REPO).

    python3 build_page.py && python3 space/assemble.py [--no-data]
    huggingface-cli upload ggautreau/PanGramGraph space_build . --repo-type space
"""
import os, sys, shutil
OUT = "space_build"
CODE = ["serve_live.py", "pgb_region.py", "fam_annot.json", "eco/manifest.json",
        "standalone/pangramgraph.html", "standalone/index.html", "standalone/origin-fork.html"]
DATA = ["pgb/window.json", "pgb/window_emb.npy", "pgb/model_calls.npz", "pgb/graph_pgb.json", "pgb/fam_info.json",
        "pgb/chrom.npz", "pgb/chrom_genomes.json", "pgb/calls_top15_f.i32", "pgb/calls_top15_p.f16",
        "pgb/calls_ent.f16", "pgb/chrom_emb.f16"]
shutil.rmtree(OUT, ignore_errors=True)
files = [(f"space/{f}", f) for f in os.listdir("space") if f != "assemble.py"] + [(f, f) for f in CODE]
if "--no-data" not in sys.argv: files += [(f, f) for f in DATA]
for src, dst in files:
    os.makedirs(os.path.join(OUT, os.path.dirname(dst)), exist_ok=True); shutil.copy2(src, os.path.join(OUT, dst))
size = sum(os.path.getsize(os.path.join(OUT, d)) for _, d in files)
print(f"{OUT}/: {len(files)} files, {size / 1e6:.0f} MB")
