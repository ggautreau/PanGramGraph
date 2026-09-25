"""Are the adjacencies Bacformer expects, but no strain of the 11 carries, real?

eco_graph.json holds 1,343 "predicted" edges (a -> x): at a locus whose previous
family is a, the model put x in its top-4 and none of the 11 strains has x there.
PanGBank pangenome 11587 (GTDB_refseq v2.0.0, 2,002 genomes) is a much larger
sample of the same species. If the model generalises synteny beyond what it saw,
its predicted edges should exist in that graph far more often than chance.

Both graphs are put in one vocabulary, the Bacformer family: every PanGBank family's
representative protein is embedded with ESM-2 and assigned to its nearest real
exemplar - the assignment that agreed across strains 90 % of the time. Families
whose nearest exemplar is not close enough stay unmapped rather than being forced
onto a wrong one.

Controls, read before the headline number:
  positive  the observed edges of the 11 strains must come out supported, or the
            mapping is broken.
  null      same source a, target drawn from the bank at random: the rate at which
            a random target is "supported" by chance given how dense a's
            neighbourhood becomes after projection.
"""
import json, collections, argparse, numpy as np, tables, torch
from transformers import AutoModel, AutoTokenizer

P=argparse.ArgumentParser()
P.add_argument("--h5",default="pgb/ecoli_11587.h5")
P.add_argument("--min_sim",type=float,default=0.95)
P.add_argument("--hops",type=int,default=1)
P.add_argument("--out",default="pgb/edges_vs_pgb.json")
A=P.parse_args()
dev="cuda:0"; rng=np.random.default_rng(0)

# --- PanGBank: families, representative proteins, family-level edges --------
h=tables.open_file(A.h5)
info=h.root.geneFamiliesInfo.read()
PART={"per":"persistent","she":"shell","clo":"cloud","P":"persistent","S":"shell","C":"cloud"}
fam_names=[x.decode() for x in info["name"]]
fam_part={n:PART.get(p.decode(),p.decode()) for n,p in zip(fam_names,info["partition"])}
fi={n:i for i,n in enumerate(fam_names)}
# representative protein: PPanGGOLiN 2.x keeps it in geneFamiliesInfo or in a
# separate sequence table depending on the version - take whichever exists
if "protein" in info.dtype.names:
    reps=[x.decode() for x in info["protein"]]
else:
    raise SystemExit(f"no representative protein column in geneFamiliesInfo: {info.dtype.names}")
print(f"PanGBank families {len(fam_names)}",flush=True)

import pandas as pd
gf=h.root.geneFamilies.read()
gidx=pd.Index(gf["gene"]); gfam=pd.Index(fam_names).get_indexer([x.decode() for x in gf["geneFam"]])
print(f"genes {len(gidx)}",flush=True)
ed=h.root.edges.read()
a_=gfam[gidx.get_indexer(ed["geneSource"])]; b_=gfam[gidx.get_indexer(ed["geneTarget"])]
ok=(a_>=0)&(b_>=0)&(a_!=b_); pairs=np.unique(np.sort(np.stack([a_[ok],b_[ok]],1),1),axis=0)
adj=collections.defaultdict(set)
for a,b in pairs: adj[int(a)].add(int(b)); adj[int(b)].add(int(a))
h.close()
print(f"family adjacencies {sum(len(v) for v in adj.values())//2}",flush=True)

# --- map every PanGBank family to its nearest Bacformer exemplar -------------
tok=AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm=AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(torch.float16).eval().to(dev)
ex=np.load("fam_exemplars.npz"); ids,emb,offs=ex["ids"],ex["emb"],ex["offsets"]
E=torch.tensor(emb,device=dev); E=E/E.norm(dim=1,keepdim=True).clamp(min=1e-9)
FAM=torch.tensor(np.concatenate([[f]*(offs[i+1]-offs[i]) for i,f in enumerate(ids)]),device=dev)

@torch.no_grad()
def assign(seqs,budget=12000):
    """Batches bounded by residues, not by count: the longest proteins go a few at a time."""
    order=np.argsort([len(s) for s in seqs]); fam=np.zeros(len(seqs),int); sim=np.zeros(len(seqs))
    i=0; done=0
    while i<len(order):
        L=min(len(seqs[order[i]]),1022)+2; j=i
        while j<len(order) and (j-i+1)*(min(len(seqs[order[j]]),1022)+2)<=budget: j+=1
        idx=order[i:max(j,i+1)]; i=max(j,i+1)
        t=tok([seqs[k].rstrip("*")[:1022] for k in idx],return_tensors="pt",padding=True,
              truncation=True,max_length=1024).to(dev)
        x=esm(**t).last_hidden_state; m=t["attention_mask"].unsqueeze(-1).to(x.dtype); m[:,0]=0
        for k,L in enumerate(t["attention_mask"].sum(1)): m[k,L-1]=0
        q=((x*m).sum(1)/m.sum(1).clamp(min=1)).float(); q=q/q.norm(dim=1,keepdim=True).clamp(min=1e-9)
        s=q@E.T; b=s.argmax(1)
        fam[idx]=FAM[b].cpu().numpy(); sim[idx]=s.gather(1,b[:,None])[:,0].cpu().numpy()
        del x,t,q,s
        if i//5000>done: done=i//5000; print(f"  embedded {i}/{len(seqs)}",flush=True)
    return fam,sim
import os
CACHE="pgb/pgb_fam_assign.npz"
if os.path.exists(CACHE):
    z=np.load(CACHE); pf,ps=z["fam"],z["sim"]
else:
    pf,ps=assign(reps); np.savez(CACHE,fam=pf,sim=ps)
mapped={i:int(pf[i]) for i in range(len(reps)) if ps[i]>=A.min_sim}
print(f"mapped {len(mapped)}/{len(reps)} PanGBank families at sim >= {A.min_sim}",flush=True)
by_part=collections.Counter(fam_part[fam_names[i]] for i in mapped)
print("  by partition", dict(by_part))

# --- project the PanGBank graph onto Bacformer families ----------------------
B=collections.defaultdict(set)          # bacformer family -> pangbank family indices
for i,f in mapped.items(): B[f].add(i)
def neigh(pgb_fams,hops):
    front=set(pgb_fams); seen=set(pgb_fams)
    for _ in range(hops):
        front={n for a in front for n in adj[a]}-seen; seen|=front
    return seen-set(pgb_fams) if hops else set()
def bac_neigh(a):
    return {mapped[n] for n in neigh(B[a],A.hops) if n in mapped}

G=json.load(open("eco_graph.json")); bank=set(ids.tolist())
local={n["family"] for n in G["nodes"]}      # families some strain carries within the 80 loci
rows=collections.defaultdict(list); cache={}
for e in G["edges"]:
    a=int(e["source"][1:]); x=int(e["target"][1:])
    if a not in B: rows[e["kind"]+"_src_unmapped"].append(e); continue
    if x not in bank: rows[e["kind"]+"_tgt_outside_bank"].append(e); continue
    if a not in cache: cache[a]=bac_neigh(a)
    N=cache[a]
    rnd=rng.choice(ids,size=200)
    rows[e["kind"]].append(dict(a=a,x=x,p=e["p"],hit=x in N,local=x in local,
        null=float(np.mean([r in N for r in rnd])),deg=len(N)))

def summ(k):
    r=rows[k]
    if not r: return None
    hit=np.mean([z["hit"] for z in r]); null=np.mean([z["null"] for z in r])
    return dict(n=len(r),supported=round(float(hit),3),null=round(float(null),4),
                enrichment=round(float(hit/null),1) if null>0 else None,
                median_degree=int(np.median([z["deg"] for z in r])))
out=dict(pangenome="11587 GTDB_refseq v2.0.0 (GTDB R232), 2,002 genomes",
         min_sim=A.min_sim,hops=A.hops,mapped=len(mapped),n_pgb=len(reps),
         observed=summ("observed"),predicted=summ("predicted"),
         excluded={k:len(v) for k,v in rows.items() if k not in("observed","predicted")})
# does model confidence rank the predicted edges?
pr=rows["predicted"]
if pr:
    q=np.quantile([z["p"] for z in pr],[0.5])
    out["predicted_by_p"]={
      "p above median":round(float(np.mean([z["hit"] for z in pr if z["p"]>q[0]])),3),
      "p below median":round(float(np.mean([z["hit"] for z in pr if z["p"]<=q[0]])),3)}
# target-matched null: keep x, swap a for another source of the same kind.
# Guards against targets that are simply hubs of the projected graph.
for k in ("observed","predicted"):
    r=rows[k]; srcs=[z["a"] for z in r]
    for z in r:
        alt=[b for b in rng.choice(srcs,size=60) if b!=z["a"]]
        z["null_tgt"]=float(np.mean([z["x"] in cache[b] for b in alt])) if alt else 0.0
    if r:
        out[k]["null_target_matched"]=round(float(np.mean([z["null_tgt"] for z in r])),4)
        out[k]["enrichment_target_matched"]=round(out[k]["supported"]/max(out[k]["null_target_matched"],1e-9),1)
# a supported prediction may just be a gene a few loci away in the same walk
pr=rows["predicted"]
for lab,sel in (("target seen in the 11 strains",[z for z in pr if z["local"]]),
                ("target never seen in the 11 strains",[z for z in pr if not z["local"]])):
    if sel: out.setdefault("predicted_by_target",{})[lab]=dict(n=len(sel),
        supported=round(float(np.mean([z["hit"] for z in sel])),3),
        null=round(float(np.mean([z["null"] for z in sel])),4))
json.dump(dict(summary=out,predicted=pr,observed=rows["observed"]),open(A.out,"w"))
print(json.dumps(out,indent=1,ensure_ascii=False))
