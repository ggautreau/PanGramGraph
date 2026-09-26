"""Start the PanGramGraph Space: fetch the data if the Space does not carry them, then serve.

The data (pgb/*, about 720 MB) either sit in the Space itself, or in a dataset repository
named by the variable PGG_DATA_REPO; a private one is read with HF_TOKEN, a Space secret.
DATA are needed to start; INFLUENCE (the Coupled regions tab's knockout of one element,
serve_live /influence) are not: without them the Space starts and /health says influence is off.
"""
import os, sys
DATA = ["pgb/window.json", "pgb/window_emb.npy", "pgb/model_calls.npz", "pgb/graph_pgb.json", "pgb/fam_info.json",
        "pgb/chrom.npz", "pgb/chrom_genomes.json", "pgb/calls_top15_f.i32", "pgb/calls_top15_p.f16",
        "pgb/calls_ent.f16", "pgb/chrom_emb.f16"]
INFLUENCE = ["pgb/region_sets.json", "pgb/region_sets_close.json", "pgb/knock/decoder_halves.npz"]   # + the units caches, pgb/knock/units_cache_*.pkl
missing = [p for p in DATA if not os.path.exists(p)]
optional = [p for p in INFLUENCE if not os.path.exists(p)]
repo = os.environ.get("PGG_DATA_REPO")
if missing and not repo:
    sys.exit(f"missing data ({', '.join(missing[:3])}...) and no PGG_DATA_REPO to fetch them from")
if (missing or optional) and repo:
    from huggingface_hub import snapshot_download
    print(f"fetching {len(missing + optional)} data files from {repo}", flush=True)
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=".", token=os.environ.get("HF_TOKEN"),
                      allow_patterns=missing + optional + (["pgb/knock/units_cache_*.pkl"] if optional else []))
    missing = [p for p in DATA if not os.path.exists(p)]
    if missing: sys.exit(f"still missing after fetching from {repo}: {', '.join(missing)}")
optional = [p for p in INFLUENCE if not os.path.exists(p)]
if optional: print(f"influence off (missing {', '.join(optional)}): the rest of the page is served", flush=True)
os.execvp(sys.executable, [sys.executable, "serve_live.py"])
