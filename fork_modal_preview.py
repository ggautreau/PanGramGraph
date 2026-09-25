"""Preview, on the 80-gene dnaA windows: at the forks where Bacformer beats phylogeny and
the graph's paths (fork_rules.py), does it still win when every protein is replaced by
the most common protein of its family in the 2,002 windows? That context keeps only the
order of families, which is what the pangenome graph holds; what the model loses is
information carried by alleles.

    python3 fork_modal_preview.py
"""
import json, collections, numpy as np, torch
from scipy import sparse
from transformers import AutoModelForCausalLM
dev="cuda:0"; CLS,PROT=2,4; MINB=20
cm=AutoModelForCausalLM.from_pretrained("macwiatrak/bacformer-causal-complete-genomes",
     trust_remote_code=True).to(torch.bfloat16).eval().to(dev)
W=json.load(open("pgb/window.json")); EMB=np.load("pgb/window_emb.npy").astype(np.float32); G=W["genomes"]
R=json.load(open("pgb/fork_rules.json")); C={k:v for k,v in np.load("pgb/model_calls.npz").items()}
FI={int(f):i for i,f in enumerate(sorted(map(int,W["families"])))}; NF=len(FI)
fam_at=np.array([FI[G[int(g)]["genes"][int(l)-G[int(g)]["start"]][1]] for g,l in zip(C["genome"],C["locus"])])
def decoder(mask):
    A=sparse.csr_matrix((C["top_p"][mask].ravel(),(np.repeat(fam_at[mask],C["top_f"].shape[1]),C["top_f"][mask].ravel())),shape=(NF,50001))
    return A,np.asarray(A.sum(0)).ravel()
DEC={h:decoder(C["genome"]%2!=h) for h in (0,1)}
cnt=collections.defaultdict(collections.Counter)
for g in G:
    for y in g["genes"]: cnt[y[1]][y[3]]+=1
modal={f:c.most_common(1)[0][0] for f,c in cnt.items()}
succ=collections.defaultdict(collections.Counter)
for g in G:
    L=g["genes"]
    for k in range(len(L)-1): succ[L[k][1]][L[k+1][1]]+=1
NODES=json.load(open("pgb/graph_pgb.json"))["nodes"]
targets=[r for r in R if r["gain_vs_phylo"]>0 and r["gain_vs_path"]>0.05]
@torch.no_grad()
def q_branch(pids,br,h):
    n=len(pids); x=np.zeros((1,n+1,480),np.float32); x[0,1:]=EMB[pids]
    p=torch.softmax(cm(protein_embeddings=torch.tensor(x,device=dev).to(torch.bfloat16),
        special_tokens_mask=torch.tensor([[CLS]+[PROT]*n],device=dev),
        token_type_ids=torch.zeros((1,n+1),dtype=torch.long,device=dev),return_dict=True).logits[0,-1].float(),-1).cpu().numpy()
    A,t=DEC[h]; s=np.array([sum(p[c]*v/t[c] for c,v in zip(A.getrow(FI[b]).indices,A.getrow(FI[b]).data)) for b in br])+1e-12
    return s/s.sum()
print("fork (column)                     sites  bits: phylo  path5  model  model on family order only")
for r in targets:
    F=int(max((n for n in NODES if n["label"]==r["fork"] and n["locus"]==r["column"]),key=lambda n:n["n"])["id"][1:])
    br=[b for b,c in succ[F].most_common(6) if c>=MINB]
    lr=[];lm=[]
    for gi,g in enumerate(G):
        L=g["genes"]
        for k in range(len(L)-1):
            if L[k][1]!=F or L[k+1][1] not in br: continue
            t=br.index(L[k+1][1]); real=[y[3] for y in L[:k+1]]; fam=[modal[y[1]] for y in L[:k+1]]
            lr.append(-np.log2(q_branch(real,br,gi%2)[t])); lm.append(-np.log2(q_branch(fam,br,gi%2)[t]))
    print(f"{r['fork'][:30]:30s}({r['column']:2d}) {len(lr):5d}      {r['bits']['phylo-15']:5.2f}  {r['bits']['path-5']:5.2f}  {np.mean(lr):5.2f}  {np.mean(lm):5.2f}")
