"""Bacformer's call at every gene of the complete chromosomes (pg_chrom.py, pg_embed.py).

A chromosome of ~4,700 proteins does not fit the 8 GB card in one pass (attention is
quadratic; ~900 proteins is the ceiling), so each is read in windows of 800 proteins
every 400: a gene's call is taken from the window where it has at least 400 proteins of
upstream context (all of it in the first window, from dnaA). One causal pass per window
gives every call in it.

Stored per gene: entropy and the top 15 (for the decoder), and at fork sites, the gene
after a family followed by >= 2 families each in >= 10 of the 540 genomes, the top 64
(for branch probabilities). Resumable, written genome by genome.

    python3 pg_calls.py      # -> pgb/calls_*.{i32,f16}, pgb/fork_sites.npy, pgb/calls.done.json
    python3 pg_calls.py --modal
        the same chromosomes with every protein replaced by the most common protein of its
        family in the 540 genomes: the model then sees only the order of families, what
        the pangenome graph holds. Fork sites only -> pgb/calls_fork_modal_*, calls_modal.done.json
"""
import sys
MODAL="--modal" in sys.argv
import os, json, time, collections, numpy as np, torch
from transformers import AutoModelForCausalLM
dev="cuda:0"; CLS,PROT=2,4; W=800; S=400; MINB=10
cm=AutoModelForCausalLM.from_pretrained("macwiatrak/bacformer-causal-complete-genomes",
     trust_remote_code=True).to(torch.bfloat16).eval().to(dev)
Z=np.load("pgb/chrom.npz"); off=Z["offsets"]; FAM=Z["fam"]; PID=Z["pid"]; NG=len(off)-1; NP=off[-1]
NPROT=sum(1 for _ in open("pgb/chrom_prot.txt"))
EMB=np.memmap("pgb/chrom_emb.f16",dtype=np.float16,mode="r",shape=(NPROT,480))
# fork families and fork sites (global index of the gene right after a fork family)
succ=collections.defaultdict(collections.Counter)
for g in range(NG):
    f=FAM[off[g]:off[g+1]]
    for a,b in set(zip(f[:-1].tolist(),f[1:].tolist())): succ[a][b]+=1         # genomes, not copies
forks={a for a,c in succ.items() if sum(1 for v in c.values() if v>=MINB)>=2}
is_site=np.zeros(NP,bool)
for g in range(NG):
    f=FAM[off[g]:off[g+1]]; is_site[off[g]+1:off[g+1]]=np.isin(f[:-1],list(forks))
sites=np.where(is_site)[0]; site_row=np.full(NP,-1,np.int64); site_row[sites]=np.arange(len(sites))
if MODAL:                                     # each family's most common protein, everywhere it occurs
    best={}
    for f,p in zip(FAM.tolist(),PID.tolist()): best.setdefault(f,collections.Counter())[p]+=1
    modal={f:c.most_common(1)[0][0] for f,c in best.items()}
    PID=np.array([modal[f] for f in FAM.tolist()],dtype=PID.dtype)
else:
    np.save("pgb/fork_sites.npy",sites); json.dump(sorted(int(x) for x in forks),open("pgb/fork_families.json","w"))
print(f"{NG} genomes, {NP} genes, {len(forks)} fork families, {len(sites)} fork sites",flush=True)
def mm(name,dt,shape):
    return np.memmap(name,dtype=dt,mode="r+" if os.path.exists(name) else "w+",shape=shape)
TAG="_modal" if MODAL else ""
if not MODAL:
    TF=mm("pgb/calls_top15_f.i32",np.int32,(NP,15)); TP=mm("pgb/calls_top15_p.f16",np.float16,(NP,15))
    EN=mm("pgb/calls_ent.f16",np.float16,(NP,))
FF=mm(f"pgb/calls_fork{TAG}_f.i32",np.int32,(len(sites),64)); FP=mm(f"pgb/calls_fork{TAG}_p.f16",np.float16,(len(sites),64))
DONE=f"pgb/calls{TAG}.done.json"; done=set(json.load(open(DONE))) if os.path.exists(DONE) else set(); t0=time.time()
@torch.no_grad()
def window(pids):
    n=len(pids); x=torch.zeros((1,n+1,480),device=dev,dtype=torch.bfloat16)
    x[0,1:]=torch.tensor(np.asarray(EMB[pids],dtype=np.float32),device=dev).to(torch.bfloat16)
    lg=cm(protein_embeddings=x,special_tokens_mask=torch.tensor([[CLS]+[PROT]*n],device=dev),
          token_type_ids=torch.zeros((1,n+1),dtype=torch.long,device=dev),return_dict=True).logits[0]
    return torch.softmax(lg.float(),-1)                # row i: the call for protein i (i >= 1)
for g in range(NG):
    if g in done: continue
    a,b=off[g],off[g+1]; n=b-a; pids=PID[a:b]; covered=1
    starts=list(range(0,max(n-W,0)+1,S))
    if starts[-1]+W<n: starts.append(n-W)
    for s in starts:
        lo=max(covered,s+(S if s else 1)) if s!=starts[-1] or s==0 else covered
        hi=min(s+W,n)
        if hi<=lo: continue
        p=window(pids[s:hi]); rows=torch.arange(lo-s,hi-s,device=dev)
        pr=p[rows]; tp,tf=pr.topk(64,1)
        ent=-(pr.clamp(min=1e-12)*pr.clamp(min=1e-12).log2()).sum(1)
        gi=np.arange(a+lo,a+hi)
        if not MODAL:
            TF[gi]=tf[:,:15].cpu().numpy(); TP[gi]=tp[:,:15].float().cpu().numpy().astype(np.float16); EN[gi]=ent.cpu().numpy().astype(np.float16)
        sr=site_row[gi]; m=sr>=0
        if m.any():
            mt=torch.tensor(np.where(m)[0],device=dev)
            FF[sr[m]]=tf[mt].cpu().numpy(); FP[sr[m]]=tp[mt].float().cpu().numpy().astype(np.float16)
        covered=hi; del p,pr
    for arr in ((FF,FP) if MODAL else (TF,TP,EN,FF,FP)): arr.flush()
    done.add(g); json.dump(sorted(done),open(DONE,"w"))
    if g%20==0: print(f"  genome {g+1}/{NG}, {time.time()-t0:.0f}s",flush=True)
print("done")
