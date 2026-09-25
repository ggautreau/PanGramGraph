"""ESM-2 embeddings of every distinct protein of the complete chromosomes (pg_chrom.py).

Mean-pooled ESM-2 t12 35M, as everywhere in this project. Resumable: proteins are sorted
by length and cut into shards of 20,000; each finished shard is written into a float16
memory map and recorded, so an interrupted run restarts where it stopped.

    python3 pg_embed.py      # -> pgb/chrom_emb.f16 (N x 480), pgb/chrom_emb.done.json
"""
import os, json, time, numpy as np, torch
from transformers import AutoModel, AutoTokenizer
dev="cuda:0"; SH=20000; BUDGET=12000
P=[l.rstrip("\n") for l in open("pgb/chrom_prot.txt")]; N=len(P)
tok=AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm=AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(torch.float16).eval().to(dev)
M=np.memmap("pgb/chrom_emb.f16",dtype=np.float16,mode="r+" if os.path.exists("pgb/chrom_emb.f16") else "w+",shape=(N,480))
DONE="pgb/chrom_emb.done.json"; done=set(json.load(open(DONE))) if os.path.exists(DONE) else set()
order=np.argsort([len(s) for s in P],kind="stable"); t0=time.time()
@torch.no_grad()
def run(idx):
    i=0
    while i<len(idx):
        j=i
        while j<len(idx) and (j-i+1)*(min(len(P[idx[j]]),1022)+2)<=BUDGET: j+=1
        b=idx[i:max(j,i+1)]; i=max(j,i+1)
        t=tok([P[k][:1022] for k in b],return_tensors="pt",padding=True,truncation=True,max_length=1024).to(dev)
        h=esm(**t).last_hidden_state; m=t["attention_mask"].unsqueeze(-1).to(h.dtype); m[:,0]=0
        for r,L in enumerate(t["attention_mask"].sum(1)): m[r,L-1]=0
        M[b]=((h*m).sum(1)/m.sum(1).clamp(min=1)).float().cpu().numpy().astype(np.float16); del h,t
for s in range(0,N,SH):
    if s in done: continue
    run(order[s:s+SH]); M.flush(); done.add(s); json.dump(sorted(done),open(DONE,"w"))
    print(f"  {min(s+SH,N)}/{N} proteins, {time.time()-t0:.0f}s",flush=True)
print("done")
