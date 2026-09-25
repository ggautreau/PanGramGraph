"""Start the PanGramGraph Space: fetch the data if the Space does not carry them, then serve.

The data (pgb/*, about 670 MB) either sit in the Space itself, or in a dataset repository
named by the variable PGG_DATA_REPO; a private one is read with HF_TOKEN, a Space secret.
"""
import os, sys
DATA = ["pgb/window.json", "pgb/window_emb.npy", "pgb/model_calls.npz", "pgb/graph_pgb.json", "pgb/fam_info.json",
        "pgb/chrom.npz", "pgb/chrom_genomes.json", "pgb/calls_top15_f.i32", "pgb/calls_top15_p.f16",
        "pgb/calls_ent.f16", "pgb/chrom_emb.f16"]
missing = [p for p in DATA if not os.path.exists(p)]
if missing:
    repo = os.environ.get("PGG_DATA_REPO")
    if not repo: sys.exit(f"missing data ({', '.join(missing[:3])}...) and no PGG_DATA_REPO to fetch them from")
    from huggingface_hub import snapshot_download
    print(f"fetching {len(missing)} data files from {repo}", flush=True)
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=".", allow_patterns=["pgb/*"], token=os.environ.get("HF_TOKEN"))
os.execvp(sys.executable, [sys.executable, "serve_live.py"])
