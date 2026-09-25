"""Precompute the model's ranked next-family distribution at every position.

11 strains x 80 loci = 880 forward passes, each conditioned on the real protein
embeddings of that strain's preceding genes. Stored top-N so the page can show
the distribution on click without a model behind it.
"""
import json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
dev="cuda:0"; CLS,PROT=2,4; D=80; TOP=15
tok=AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm=AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(torch.float16).eval().to(dev)
cm=AutoModelForCausalLM.from_pretrained("macwiatrak/bacformer-causal-complete-genomes",
     trust_remote_code=True).to(torch.bfloat16).eval().to(dev)
ex=np.load("fam_exemplars.npz"); ids,emb,offs=ex["ids"],ex["emb"],ex["offsets"]
E=torch.tensor(emb,device=dev); E=E/E.norm(dim=1,keepdim=True).clamp(min=1e-9)
FAM=torch.tensor(np.concatenate([[f]*(offs[i+1]-offs[i]) for i,f in enumerate(ids)]),device=dev)
annot=json.load(open("fam_annot.json"))
def label(f):
    a=annot.get(str(int(f)))
    if not a: return f"fam{int(f)}"
    if a["gene"]: return a["gene"][0][0]
    if a["product"]: return a["product"][0][0][:38]
    return f"fam{int(f)}"
@torch.no_grad()
def embed(s,bs=24):
    o=[]
    for i in range(0,len(s),bs):
        t=tok([x[:1022] for x in s[i:i+bs]],return_tensors="pt",padding=True,truncation=True,max_length=1024).to(dev)
        h=esm(**t).last_hidden_state;m=t["attention_mask"].unsqueeze(-1).to(h.dtype);m[:,0]=0
        for j,L in enumerate(t["attention_mask"].sum(1)):m[j,L-1]=0
        o.append(((h*m).sum(1)/m.sum(1).clamp(min=1)).float().cpu().numpy());del h,t
    return np.concatenate(o,0)
@torch.no_grad()
def nxt(pe):
    n=pe.shape[0]+1;x=np.zeros((1,n,480),dtype=np.float32);x[0,1:]=pe
    return torch.softmax(cm(protein_embeddings=torch.tensor(x,device=dev).to(torch.bfloat16),
        special_tokens_mask=torch.tensor([[CLS]+[PROT]*pe.shape[0]],device=dev),
        token_type_ids=torch.zeros((1,n),dtype=torch.long,device=dev),
        return_dict=True).logits[0,-1].float(),-1).cpu().numpy()
@torch.no_grad()
def assign(pe):
    q=torch.tensor(pe,device=dev);q=q/q.norm(dim=1,keepdim=True).clamp(min=1e-9)
    return FAM[(q@E.T).argmax(1)].cpu().numpy()

G=json.load(open("eco_parsed.json"))
labels={}; out=[]
for gi,g in enumerate(G):
    off=1 if g.get("dnaA_pseudo") else 0
    prots=g["proteins"][:D-off]; pe=embed(prots); fam=assign(pe)
    steps={}
    for k in range(1,len(fam)):
        p=nxt(pe[:k]); o=np.argsort(-p)[:TOP]; t=int(fam[k])
        r=int(np.where(np.argsort(-p)==t)[0][0])
        ent=float(-(p[p>1e-9]*np.log2(p[p>1e-9])).sum())
        for f in o: labels.setdefault(int(f),label(f))
        labels.setdefault(t,label(t))
        steps[k+off]=dict(e=round(ent,3),
            t=[[int(f),round(float(p[f]),6)] for f in o],
            r=dict(f=t,k=r,p=round(float(p[t]),8)))
    out.append(dict(strain=g["desc"].split(" (")[0],steps=steps))
    print(f"[{gi+1}/{len(G)}] {g['desc'][:38]:40s} {len(steps)} loci",flush=True)
json.dump(dict(labels=labels,strains=out),open("dists.json","w"),separators=(",",":"))
import os; print("dists.json  %.0f KB"%(os.path.getsize("dists.json")/1024))
