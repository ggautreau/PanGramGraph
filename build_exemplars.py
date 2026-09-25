"""Store REAL protein embeddings per family, not their mean.

The prototype loop failed because a mean embedding is not a protein. A genuine
member of the family is exactly the kind of input Bacformer was trained on, so
the autoregressive loop can be closed with real exemplars instead.
"""
import json, torch, numpy as np, collections, time, os, sys
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel

dev="cuda:0"; MAXP=int(os.environ.get("MAXP","1200")); KEEP=6
tok=AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm=AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(torch.float16).eval().to(dev)
mm=AutoModelForMaskedLM.from_pretrained("macwiatrak/bacformer-masked-complete-genomes",
     trust_remote_code=True).to(torch.bfloat16).eval().to(dev)
CLS,PROT=2,4
print("models loaded",flush=True)

@torch.no_grad()
def embed(seqs,bs=24):
    out=[]
    for i in range(0,len(seqs),bs):
        t=tok([s[:1022] for s in seqs[i:i+bs]],return_tensors="pt",padding=True,
              truncation=True,max_length=1024).to(dev)
        h=esm(**t).last_hidden_state;m=t["attention_mask"].unsqueeze(-1).to(h.dtype);m[:,0]=0
        for j,L in enumerate(t["attention_mask"].sum(1)): m[j,L-1]=0
        out.append(((h*m).sum(1)/m.sum(1).clamp(min=1)).float().cpu().numpy()); del h,t
    return np.concatenate(out,0)

@torch.no_grad()
def families(pe):
    n=pe.shape[0]+1;x=np.zeros((1,n,480),dtype=np.float32);x[0,1:]=pe
    return mm(protein_embeddings=torch.tensor(x,device=dev).to(torch.bfloat16),
        special_tokens_mask=torch.tensor([[CLS]+[PROT]*pe.shape[0]],device=dev),
        token_type_ids=torch.zeros((1,n),dtype=torch.long,device=dev),
        return_dict=True).logits[0,1:].argmax(-1).cpu().numpy()

ds=load_dataset("macwiatrak/bacbench-essential-genes-protein-sequences",split="train",streaming=True)
ex=collections.defaultdict(list); nm=collections.defaultdict(collections.Counter)
pd_=collections.defaultdict(collections.Counter); genomes=[]; t0=time.time()
for gi,item in enumerate(ds):
    prots=[p for c in item["protein_sequence"] for p in c][:MAXP]
    names=[n for c in item["gene_name"] for n in c][:MAXP]
    prods=[p for c in item["product"] for p in c][:MAXP]
    if len(prots)<60: continue
    try:
        pe=embed(prots); fam=families(pe); k=min(len(fam),len(prots))
        for j in range(k):
            f=int(fam[j])
            if f>=50000: continue
            if len(ex[f])<KEEP: ex[f].append(pe[j])
            if names[j]: nm[f][names[j]]+=1
            if prods[j]: pd_[f][prods[j]]+=1
        genomes.append(dict(name=item["genome_name"],strain=str(item["strain_name"]),
            species=item["species"],genus=item["genus"],
            families=[int(x) for x in fam[:k]],gene_names=names[:k],
            proteins=prots[:60]))
        print(f"[{gi}] {str(item['strain_name'])[:34]:36s} n={k:4d} fams={len(ex):6d} t={time.time()-t0:.0f}s",flush=True)
        torch.cuda.empty_cache()
    except Exception as e:
        print("skip",gi,str(e)[:80],flush=True); torch.cuda.empty_cache()

ids=sorted(ex)
counts=np.array([len(ex[f]) for f in ids])
flat=np.concatenate([np.stack(ex[f]) for f in ids]).astype(np.float32)
offs=np.concatenate([[0],np.cumsum(counts)])
np.savez_compressed("fam_exemplars.npz",ids=np.array(ids),emb=flat,offsets=offs,counts=counts)
json.dump({str(f):{"gene":nm[f].most_common(3),"product":pd_[f].most_common(2),"n":len(ex[f])} for f in ex},
          open("fam_annot.json","w"))
json.dump(genomes,open("genomes.json","w"))
print(f"DONE families={len(ids)} exemplars={flat.shape[0]} genomes={len(genomes)} t={time.time()-t0:.0f}s")
