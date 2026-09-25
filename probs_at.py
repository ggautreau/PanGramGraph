#!/usr/bin/env python3
"""Full next-family distribution at one position of one E. coli chromosome.

    python3 probs_at.py --strain "O157:H7 Sakai" --locus 12 --top 20

Locus is counted in genes from the dnaA anchor, exactly as in the graph: locus 0
is dnaA, locus 1 the gene after it. The model is conditioned on the REAL protein
embeddings of loci 0..locus-1, so this is the distribution it actually saw.
Reports every family's probability, ranked, plus where the real gene falls.
"""
import json, argparse, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

P = argparse.ArgumentParser()
P.add_argument("--strain", default="O157:H7 Sakai", help="substring of the strain name")
P.add_argument("--locus", type=int, default=12)
P.add_argument("--top", type=int, default=20)
P.add_argument("--csv", help="write the full 50,000-family distribution here")
A = P.parse_args()

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
CLS, PROT = 2, 4
tok = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(torch.float16).eval().to(dev)
cm = AutoModelForCausalLM.from_pretrained("macwiatrak/bacformer-causal-complete-genomes",
        trust_remote_code=True).to(torch.bfloat16).eval().to(dev)

ex = np.load("fam_exemplars.npz")
ids, emb, offs = ex["ids"], ex["emb"], ex["offsets"]
E = torch.tensor(emb, device=dev); E = E / E.norm(dim=1, keepdim=True).clamp(min=1e-9)
FAM = torch.tensor(np.concatenate([[f] * (offs[i+1]-offs[i]) for i, f in enumerate(ids)]), device=dev)
annot = json.load(open("fam_annot.json"))

def label(f):
    a = annot.get(str(int(f)))
    if not a: return f"fam{int(f)}"
    if a["gene"]: return a["gene"][0][0]
    if a["product"]: return a["product"][0][0][:40]
    return f"fam{int(f)}"

@torch.no_grad()
def embed(seqs):
    t = tok([s[:1022] for s in seqs], return_tensors="pt", padding=True,
            truncation=True, max_length=1024).to(dev)
    h = esm(**t).last_hidden_state; m = t["attention_mask"].unsqueeze(-1).to(h.dtype); m[:, 0] = 0
    for j, L in enumerate(t["attention_mask"].sum(1)): m[j, L-1] = 0
    return ((h*m).sum(1) / m.sum(1).clamp(min=1)).float().cpu().numpy()

@torch.no_grad()
def next_dist(pe):
    n = pe.shape[0] + 1; x = np.zeros((1, n, 480), dtype=np.float32); x[0, 1:] = pe
    lg = cm(protein_embeddings=torch.tensor(x, device=dev).to(torch.bfloat16),
            special_tokens_mask=torch.tensor([[CLS] + [PROT]*pe.shape[0]], device=dev),
            token_type_ids=torch.zeros((1, n), dtype=torch.long, device=dev),
            return_dict=True).logits[0, -1].float()
    return torch.softmax(lg, -1).cpu().numpy()

@torch.no_grad()
def assign(pe):
    q = torch.tensor(pe, device=dev); q = q / q.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return FAM[(q @ E.T).argmax(1)].cpu().numpy()

G = json.load(open("eco_parsed.json"))
g = next((x for x in G if A.strain.lower() in x["desc"].lower()), None)
if g is None:
    raise SystemExit("strains: " + " | ".join(x["desc"].split(" (")[0] for x in G))
off = 1 if g.get("dnaA_pseudo") else 0
k = A.locus - off
if k < 1: raise SystemExit(f"locus {A.locus} is at or before this strain's anchor")

prots = g["proteins"][:k+1]
pe = embed(prots)
p = next_dist(pe[:k])                      # conditioned on loci 0..locus-1
fam = assign(pe)
truth = int(fam[k]) if k < len(fam) else None

order = np.argsort(-p)
ent = float(-(p[p > 1e-9] * np.log2(p[p > 1e-9])).sum())
print(f"\n{g['desc']}   locus {A.locus}  (prefix = {k} real proteins from dnaA)")
print(f"context: {' '.join(str(x) for x in g['genes'][:k] if x)[:88]}")
print(f"entropy {ent:.3f} bits   |   sum of top-{A.top} = {p[order[:A.top]].sum():.4f}\n")
print(f"{'rank':>5} {'family':>8} {'gene':22s} {'p':>10}  {'cum':>7}")
cum = 0.0
for r, f in enumerate(order[:A.top]):
    cum += p[f]
    mark = "  <- real" if truth is not None and int(f) == truth else ""
    print(f"{r+1:5d} {int(f):8d} {label(f)[:22]:22s} {p[f]:10.6f}  {cum:7.4f}{mark}")
if truth is not None:
    r = int(np.where(order == truth)[0][0])
    print(f"\nreal gene here: {g['genes'][k]}  family {truth}  "
          f"p={p[truth]:.3e}  rank {r+1} / 50000")
if A.csv:
    import csv
    with open(A.csv, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["rank", "family", "label", "p"])
        for r, f in enumerate(order):
            w.writerow([r+1, int(f), label(f), f"{p[f]:.8g}"])
    print(f"full distribution -> {A.csv}  ({len(order)} rows)")
