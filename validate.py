"""Two gates that must pass before any generated graph means anything.

GATE 1 - prototype loop. The causal checkpoint eats pLM embeddings and emits a
family id, so an autoregressive walk must turn the emitted family back into an
embedding. We use the family's mean embedding ("prototype"). Does that
substitution change what the model predicts? Replay real genomes twice - on real
embeddings, and on prototypes of the very same families - and compare.

GATE 2 - nucleus coverage. Our prototype dictionary covers only part of the
50,000-family vocabulary. If most of the top_p mass lands on families we cannot
represent, the walk is steered by our dictionary, not by the model.
"""
import json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

dev="cuda:0"
model=AutoModelForCausalLM.from_pretrained("macwiatrak/bacformer-causal-complete-genomes",
        trust_remote_code=True).to(torch.bfloat16).eval().to(dev)
tok=AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm=AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(torch.float16).eval().to(dev)
CLS,PROT=2,4
pr=np.load("fam_proto.npz"); ids,embs=pr["ids"],pr["emb"]
idx={int(f):i for i,f in enumerate(ids)}
genomes=json.load(open("genomes.json"))
from datasets import load_dataset
ds=load_dataset("macwiatrak/bacbench-essential-genes-protein-sequences",split="train",streaming=True)

@torch.no_grad()
def embed(seqs):
    t=tok([s[:1022] for s in seqs],return_tensors="pt",padding=True,truncation=True,max_length=1024).to(dev)
    h=esm(**t).last_hidden_state; m=t["attention_mask"].unsqueeze(-1).to(h.dtype); m[:,0]=0
    for j,L in enumerate(t["attention_mask"].sum(1)): m[j,L-1]=0
    return ((h*m).sum(1)/m.sum(1).clamp(min=1)).float().cpu().numpy()

@torch.no_grad()
def dist(pe):
    n=pe.shape[0]+1; x=np.zeros((1,n,480),dtype=np.float32); x[0,1:]=pe
    lg=model(protein_embeddings=torch.tensor(x,device=dev).to(torch.bfloat16),
             special_tokens_mask=torch.tensor([[CLS]+[PROT]*pe.shape[0]],device=dev),
             token_type_ids=torch.zeros((1,n),dtype=torch.long,device=dev),
             return_dict=True).logits[0,-1].float()
    return torch.softmax(lg,-1).cpu().numpy()

def nucleus_cov(p,top_p=0.6):
    o=np.argsort(-p); c=np.cumsum(p[o]); k=int(np.searchsorted(c,top_p))+1
    sel=o[:k]; mass=p[sel].sum()
    cov=sum(p[f] for f in sel if int(f) in idx)
    return k, mass, cov/max(mass,1e-9)

L=25; NG=8
hit_r=hit_p=agree=tot=0; ks=[]; covs=[]
seen=0
for item in ds:
    if seen>=NG: break
    prots=[p for c in item["protein_sequence"] for p in c][:L]
    if len(prots)<L: continue
    gn=next((g for g in genomes if g["name"]==item["genome_name"]),None)
    if gn is None: continue
    seen+=1
    real=embed(prots); fams=gn["families"][:L]
    if any(f not in idx for f in fams): continue
    for k in range(3,L-1):
        truth=fams[k]
        pr_=dist(np.stack([embs[idx[f]] for f in fams[:k]]))
        rr_=dist(real[:k])
        tp,tr=int(pr_.argmax()),int(rr_.argmax())
        hit_p+=tp==truth; hit_r+=tr==truth; agree+=tp==tr; tot+=1
        kk,mass,cov=nucleus_cov(rr_); ks.append(kk); covs.append(cov)
print(f"\nGATE 1  steps={tot} over {seen} genomes")
print(f"  next-family top1, real embeddings : {hit_r/max(tot,1):.3f}")
print(f"  next-family top1, prototypes      : {hit_p/max(tot,1):.3f}")
print(f"  prototype/real top1 agreement     : {agree/max(tot,1):.3f}")
print(f"\nGATE 2  nucleus at top_p=0.6")
print(f"  median families in nucleus        : {np.median(ks):.0f}  (mean {np.mean(ks):.1f})")
print(f"  mass covered by our prototypes    : {np.mean(covs):.3f}  (median {np.median(covs):.3f})")
