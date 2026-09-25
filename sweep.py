import json,numpy as np,torch,collections
from transformers import AutoModelForCausalLM,AutoModel,AutoTokenizer
from datasets import load_dataset
dev="cuda:0";CLS,PROT=2,4
model=AutoModelForCausalLM.from_pretrained("macwiatrak/bacformer-causal-complete-genomes",trust_remote_code=True).to(torch.bfloat16).eval().to(dev)
tok=AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm=AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(torch.float16).eval().to(dev)
gm={g["name"]:g for g in json.load(open("genomes.json"))}
@torch.no_grad()
def embed(s):
    t=tok([x[:1022] for x in s],return_tensors="pt",padding=True,truncation=True,max_length=1024).to(dev)
    h=esm(**t).last_hidden_state;m=t["attention_mask"].unsqueeze(-1).to(h.dtype);m[:,0]=0
    for j,L in enumerate(t["attention_mask"].sum(1)):m[j,L-1]=0
    return ((h*m).sum(1)/m.sum(1).clamp(min=1)).float().cpu().numpy()
@torch.no_grad()
def dist(pe):
    n=pe.shape[0]+1;x=np.zeros((1,n,480),dtype=np.float32);x[0,1:]=pe
    lg=model(protein_embeddings=torch.tensor(x,device=dev).to(torch.bfloat16),
        special_tokens_mask=torch.tensor([[CLS]+[PROT]*pe.shape[0]],device=dev),
        token_type_ids=torch.zeros((1,n),dtype=torch.long,device=dev),return_dict=True).logits[0,-1].float()
    return torch.softmax(lg,-1).cpu().numpy()
TP=[0.6,0.8,0.9,0.95,0.99]
size=collections.defaultdict(list);cov=collections.defaultdict(list)
ds=load_dataset("macwiatrak/bacbench-essential-genes-protein-sequences",split="train",streaming=True)
seen=set();ng=0;D=30
ents=[];depths=[];accs=[]
for item in ds:
    if ng>=14:break
    sp=item["species"]
    if sp in seen:continue
    pr=[p for c in item["protein_sequence"] for p in c][:D+1];g=gm.get(item["genome_name"])
    if g is None or len(pr)<D+1:continue
    seen.add(sp);ng+=1;f=g["families"][:D+1];re_=embed(pr)
    for k in range(1,D+1):
        p=dist(re_[:k]);o=np.argsort(-p);c=np.cumsum(p[o]);t=f[k]
        r=int(np.where(o==t)[0][0])
        e=float(-(p[p>1e-9]*np.log2(p[p>1e-9])).sum());ents.append(e);depths.append(k-1);accs.append(r==0)
        for tp in TP:
            kn=int(np.searchsorted(c,tp))+1
            size[tp].append(kn);cov[tp].append(r<kn)
print("\n top_p   median nucleus   mean nucleus   true family inside")
for tp in TP:
    print(f"  {tp:.2f}   {np.median(size[tp]):12.0f}   {np.mean(size[tp]):11.1f}   {np.mean(cov[tp]):16.3f}")
ents=np.array(ents);depths=np.array(depths);accs=np.array(accs)
print(f"\n corr(entropy, locus)      r = {np.corrcoef(ents,depths)[0,1]:+.3f}   n={len(ents)}")
print(f" mean entropy loci 0-9     {ents[depths<10].mean():.2f} bits   top1 {accs[depths<10].mean():.3f}")
print(f" mean entropy loci 20-29   {ents[depths>=20].mean():.2f} bits   top1 {accs[depths>=20].mean():.3f}")
