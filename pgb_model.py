"""Bacformer's call at every position of the 2,002 PanGBank dnaA windows.

Bacformer is causal, so one forward pass over a genome's window gives the call at
every position: the output after protein k is the distribution for protein k+1,
the same as a pass on the prefix alone (checked: top-1 identical at 14 of 14
positions, |dp| <= 0.006, bfloat16 noise). 2,002 passes instead of 130,000.

Each distinct protein (19,198) is embedded once with ESM-2 and assigned its
Bacformer family by nearest real exemplar, as in the 11-strain walk.

    python3 pgb_model.py     # -> pgb/window_emb.npy, pgb/model_calls.npz
"""
import json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
dev="cuda:0"; CLS,PROT=2,4; TOP=15
tok=AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm=AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(torch.float16).eval().to(dev)
cm=AutoModelForCausalLM.from_pretrained("macwiatrak/bacformer-causal-complete-genomes",
     trust_remote_code=True).to(torch.bfloat16).eval().to(dev)
ex=np.load("fam_exemplars.npz"); ids,emb,offs=ex["ids"],ex["emb"],ex["offsets"]
E=torch.tensor(emb,device=dev); E=E/E.norm(dim=1,keepdim=True).clamp(min=1e-9)
FAM=torch.tensor(np.concatenate([[f]*(offs[i+1]-offs[i]) for i,f in enumerate(ids)]),device=dev)

@torch.no_grad()
def embed(seqs,budget=12000):
    """Mean-pooled ESM-2; batches bounded by residues so long proteins go a few at a time."""
    order=np.argsort([len(s) for s in seqs]); out=np.zeros((len(seqs),480),dtype=np.float32); i=0
    while i<len(order):
        j=i
        while j<len(order) and (j-i+1)*(min(len(seqs[order[j]]),1022)+2)<=budget: j+=1
        idx=order[i:max(j,i+1)]; i=max(j,i+1)
        t=tok([seqs[k][:1022] for k in idx],return_tensors="pt",padding=True,truncation=True,max_length=1024).to(dev)
        h=esm(**t).last_hidden_state; m=t["attention_mask"].unsqueeze(-1).to(h.dtype); m[:,0]=0
        for k,L in enumerate(t["attention_mask"].sum(1)): m[k,L-1]=0
        out[idx]=((h*m).sum(1)/m.sum(1).clamp(min=1)).float().cpu().numpy(); del h,t
    return out

W=json.load(open("pgb/window.json")); P=json.load(open("pgb/window_prot.json"))
pe_all=embed(P); np.save("pgb/window_emb.npy",pe_all.astype(np.float16))
with torch.no_grad():
    q=torch.tensor(pe_all,device=dev); q=q/q.norm(dim=1,keepdim=True).clamp(min=1e-9)
    bf=np.concatenate([FAM[(q[i:i+4096]@E.T).argmax(1)].cpu().numpy() for i in range(0,len(q),4096)])
print(f"embedded {len(P)} proteins",flush=True)

G=[];K=[];ENT=[];RANK=[];PR=[];TF=[];TP=[]
with torch.no_grad():
    for gi,g in enumerate(W["genomes"]):
        pids=[x[3] for x in g["genes"]]; n=len(pids)
        if n<2: continue
        x=np.zeros((1,n+1,480),dtype=np.float32); x[0,1:]=pe_all[pids]
        lg=cm(protein_embeddings=torch.tensor(x,device=dev).to(torch.bfloat16),
              special_tokens_mask=torch.tensor([[CLS]+[PROT]*n],device=dev),
              token_type_ids=torch.zeros((1,n+1),dtype=torch.long,device=dev),return_dict=True).logits[0].float()
        p=torch.softmax(lg[1:n],-1)                   # row k-1: call for protein k, k=1..n-1
        t=torch.tensor(bf[pids[1:]],device=dev,dtype=torch.long)
        pt=p.gather(1,t[:,None])[:,0]
        ent=-(p.clamp(min=1e-12)*p.clamp(min=1e-12).log2()).sum(1)
        tp,tf=p.topk(TOP,1)
        G.append(np.full(n-1,gi,np.int16)); K.append(np.arange(1,n,dtype=np.int16)+g["start"])
        ENT.append(ent.cpu().numpy()); RANK.append(((p>pt[:,None]).sum(1)+1).cpu().numpy())
        PR.append(pt.cpu().numpy()); TF.append(tf.cpu().numpy().astype(np.int32)); TP.append(tp.cpu().numpy())
        if gi%250==0: print(f"  {gi}/{len(W['genomes'])}",flush=True)
np.savez_compressed("pgb/model_calls.npz",genome=np.concatenate(G),locus=np.concatenate(K),
    entropy=np.concatenate(ENT),rank=np.concatenate(RANK),p_real=np.concatenate(PR),
    top_f=np.concatenate(TF),top_p=np.concatenate(TP),bf_family=bf)
r=np.concatenate(RANK)
print(f"calls {len(r)}  top-1 {np.mean(r==1)*100:.1f}%  median rank {int(np.median(r))}")
