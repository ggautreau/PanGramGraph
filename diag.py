"""Do orthologous proteins get the same family across strains?

Two ways of assigning a family are compared. The masked head predicts a family
from the protein AND its genomic context, so its answer can move with the
context. Nearest-exemplar assigns by embedding distance alone, which is what the
paper says the 50,000 clusters actually are - a clustering in embedding space.
"""
import json, numpy as np, torch, collections
from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer
dev="cuda:0"; CLS,PROT=2,4
tok=AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm=AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(torch.float16).eval().to(dev)
mm=AutoModelForMaskedLM.from_pretrained("macwiatrak/bacformer-masked-complete-genomes",
     trust_remote_code=True).to(torch.bfloat16).eval().to(dev)
ex=np.load("fam_exemplars.npz")
ids,emb,offs=ex["ids"],ex["emb"],ex["offsets"]
fam_of=np.concatenate([[f]*(offs[i+1]-offs[i]) for i,f in enumerate(ids)])
E=torch.tensor(emb,device=dev)
En=E/E.norm(dim=1,keepdim=True).clamp(min=1e-9)
FAM=torch.tensor(fam_of,device=dev)

@torch.no_grad()
def embed(seqs,bs=24):
    o=[]
    for i in range(0,len(seqs),bs):
        t=tok([s[:1022] for s in seqs[i:i+bs]],return_tensors="pt",padding=True,truncation=True,max_length=1024).to(dev)
        h=esm(**t).last_hidden_state;m=t["attention_mask"].unsqueeze(-1).to(h.dtype);m[:,0]=0
        for j,L in enumerate(t["attention_mask"].sum(1)): m[j,L-1]=0
        o.append(((h*m).sum(1)/m.sum(1).clamp(min=1)).float().cpu().numpy()); del h,t
    return np.concatenate(o,0)
@torch.no_grad()
def masked_fam(pe):
    n=pe.shape[0]+1;x=np.zeros((1,n,480),dtype=np.float32);x[0,1:]=pe
    return mm(protein_embeddings=torch.tensor(x,device=dev).to(torch.bfloat16),
        special_tokens_mask=torch.tensor([[CLS]+[PROT]*pe.shape[0]],device=dev),
        token_type_ids=torch.zeros((1,n),dtype=torch.long,device=dev),
        return_dict=True).logits[0,1:].argmax(-1).cpu().numpy()
@torch.no_grad()
def nn_fam(pe):
    q=torch.tensor(pe,device=dev); q=q/q.norm(dim=1,keepdim=True).clamp(min=1e-9)
    return FAM[(q@En.T).argmax(1)].cpu().numpy()

G=json.load(open("eco_parsed.json")); D=40
res={}
for g in G:
    pe=embed(g["proteins"][:D])
    res[g["desc"].split(" (")[0]]=dict(genes=g["genes"][:D],
        masked=[int(x) for x in masked_fam(pe)], nn=[int(x) for x in nn_fam(pe)])
# agreement on orthologues: same gene name across strains -> same family?
for scheme in ("masked","nn"):
    byg=collections.defaultdict(list)
    for st,r in res.items():
        for j,gn in enumerate(r["genes"]):
            if gn: byg[gn].append(r[scheme][j])
    multi={k:v for k,v in byg.items() if len(v)>=5}
    agree=[len(set(v))==1 for v in multi.values()]
    frac=[max(collections.Counter(v).values())/len(v) for v in multi.values()]
    print(f"{scheme:7s}  genes seen in >=5 strains: {len(multi):3d} | "
          f"identical family in all: {np.mean(agree):.3f} | mean majority share: {np.mean(frac):.3f}")
print("\n  gene     masked families across strains        | nn families")
for gn in ["dnaA","dnaN","recF","gyrB","yidB","yidA","yidC","rnpA","mnmE","atpB"]:
    m=[r["masked"][r["genes"].index(gn)] for r in res.values() if gn in r["genes"]]
    n=[r["nn"][r["genes"].index(gn)] for r in res.values() if gn in r["genes"]]
    if not m: continue
    print(f"  {gn:6s} n={len(m):2d} distinct={len(set(m)):2d} {sorted(set(m))[:5]}"
          f"  | distinct={len(set(n)):2d} {sorted(set(n))[:5]}")
