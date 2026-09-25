"""Where the PanGBank-wide run stands: species listed, downloaded, extracted, embedded and called, failed,
disk used by what is kept, and a projection from the species done so far.

    python status.py
"""
import os, json, glob
ROOT = os.environ.get("PGG_ROOT", "."); L = json.load(open(f"{ROOT}/state/species.json")); sp = L["species"]
genes = {str(r["id"]): r["genes"] for r in sp}; tot = sum(genes.values())
raw = glob.glob(f"{ROOT}/raw/*.h5"); ext = glob.glob(f"{ROOT}/out/*/extract.ok") + glob.glob(f"{ROOT}/out/*/gpu.claim.*")
done = glob.glob(f"{ROOT}/out/*/gpu.ok"); failed = glob.glob(f"{ROOT}/raw/*.failed") + glob.glob(f"{ROOT}/out/*/gpu.failed")
size = lambda d: sum(e.stat().st_size for e in os.scandir(d) if e.is_file())
kept = sum(size(os.path.dirname(p)) for p in done); gdone = sum(genes.get(os.path.basename(os.path.dirname(p)), 0) for p in done)
print(f"{L['bacteria']} bacterial species ({L['archaea']} archaeal left out), {tot / 1e6:.0f} M genes")
print(f"waiting in raw/: {len(raw)} files, {sum(os.path.getsize(p) for p in raw) / 1e9:.1f} GB | extracted, not yet on GPU: {len(ext)}")
print(f"done: {len(done)} species, {gdone / 1e6:.1f} M genes ({gdone / max(tot, 1):.1%}), {kept / 1e9:.1f} GB kept"
      + (f" -> about {kept / max(gdone, 1) * tot / 1e9:.0f} GB for all" if gdone else ""))
if failed: print("failed:", [os.path.basename(p) for p in failed][:20])
