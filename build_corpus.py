import json, torch, numpy as np, collections, time, os, sys
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel

dev="cuda:0"; MAXG=int(os.environ.get("MAXG","120")); MAXP=int(os.environ.get("MAXP","900"))
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
        b=[s[:1022] for s in seqs[i:i+bs]]
        t=tok(b,return_tensors="pt",padding=True,truncation=True,max_length=1024).to(dev)
        h=esm(**t).last_hidden_state
        m=t["attention_mask"].unsqueeze(-1).to(h.dtype)
        m[:,0]=0
        for j,L in enumerate(t["attention_mask"].sum(1)): m[j,L-1]=0
        out.append(((h*m).sum(1)/m.sum(1).clamp(min=1)).float().cpu().numpy())
        del h,t
    return np.concatenate(out,0)

@torch.no_grad()
def families(pe):
    n=pe.shape[0]+1
    x=np.zeros((1,n,pe.shape[1]),dtype=np.float32); x[0,1:]=pe
    stm=torch.tensor([[CLS]+[PROT]*pe.shape[0]],device=dev)
    tti=torch.zeros((1,n),dtype=torch.long,device=dev)
    lg=mm(protein_embeddings=torch.tensor(x,device=dev).to(torch.bfloat16),
          special_tokens_mask=stm,token_type_ids=tti,return_dict=True).logits[0,1:]
    return lg.argmax(-1).cpu().numpy()

ds=load_dataset("macwiatrak/bacbench-essential-genes-protein-sequences",split="train",streaming=True)
fam2emb=collections.defaultdict(list); fam2name=collections.defaultdict(collections.Counter)
fam2prod=collections.defaultdict(collections.Counter); genomes=[]; t0=time.time(); small=0
for gi,item in enumerate(ds):
    if gi>=MAXG: break
    prots=[p for c in item["protein_sequence"] for p in c][:MAXP]
    names=[n for c in item["gene_name"] for n in c][:MAXP]
    prods=[p for c in item["product"] for p in c][:MAXP]
    if len(prots)<60: small+=1; continue
    try:
        pe=embed(prots); fam=families(pe); k=min(len(fam),len(prots))
        for j in range(k):
            f=int(fam[j])
            if f>=50000: continue
            if len(fam2emb[f])<30: fam2emb[f].append(pe[j])
            if names[j]: fam2name[f][names[j]]+=1
            if prods[j]: fam2prod[f][prods[j]]+=1
        genomes.append({"name":item["genome_name"],"strain":item["strain_name"],
            "species":item["species"],"genus":item["genus"],
            "families":[int(x) for x in fam[:k]],"gene_names":names[:k]})
        print(f"[{gi}] {str(item['strain_name'])[:38]:40s} n={k:4d} fams={len(fam2emb):6d} t={time.time()-t0:.0f}s",flush=True)
        torch.cuda.empty_cache()
    except Exception as e:
        print("skip",gi,str(e)[:90],flush=True); torch.cuda.empty_cache()

if not fam2emb: sys.exit("no families collected (small=%d)"%small)
np.savez_compressed("fam_proto.npz", ids=np.array(sorted(fam2emb)),
    emb=np.stack([np.mean(fam2emb[f],0) for f in sorted(fam2emb)]).astype(np.float32))
json.dump({str(f):{"gene":fam2name[f].most_common(3),"product":fam2prod[f].most_common(2),
                   "n":len(fam2emb[f])} for f in fam2emb}, open("fam_annot.json","w"))
json.dump(genomes,open("genomes.json","w"))
print(f"DONE families={len(fam2emb)} genomes={len(genomes)} skipped_small={small} t={time.time()-t0:.0f}s")
