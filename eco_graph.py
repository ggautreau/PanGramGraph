"""An E. coli pangenome graph, with Bacformer's expectations layered on top.

Nodes are protein families, edges are adjacencies - the PPanGGOLiN construction.
Two edge layers:

  observed   an adjacency actually present in one or more of the 11 strains,
             weighted by how many carry it. Core synteny is the path every
             strain shares; accessory regions are where the strains part.

  predicted  a family the causal model puts in its top-k at that locus but which
             no strain carries there. These are the adjacencies the model thinks
             could be there, and they exist only because it is a generative model.

Every prefix fed to the model is real protein embeddings from a real genome, so
nothing here depends on the prototype loop that failed its control.

FAMILY ASSIGNMENT. Not by the masked head. Measured across these 11 strains, the
masked head gives orthologues the same family in 0% of cases (mean majority share
0.48) because it predicts from genomic context as well as from the protein, so its
answer moves with the context. Assigning by nearest exemplar in embedding space -
which is what the 50,000 clusters are - agrees across all strains for 90% of genes
(majority share 0.985), and puts dnaA, dnaN, recF and gyrB each in a single family.
"""
import json, argparse, collections, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoModel, AutoTokenizer

P=argparse.ArgumentParser()
P.add_argument("--loci",type=int,default=120)
P.add_argument("--topk",type=int,default=4)
P.add_argument("--out",default="eco_graph.json")
A=P.parse_args()
dev="cuda:0"; CLS,PROT=2,4

tok=AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm=AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(torch.float16).eval().to(dev)
cm=AutoModelForCausalLM.from_pretrained("macwiatrak/bacformer-causal-complete-genomes",
     trust_remote_code=True).to(torch.bfloat16).eval().to(dev)
_ex=np.load("fam_exemplars.npz")
_ids,_emb,_offs=_ex["ids"],_ex["emb"],_ex["offsets"]
_E=torch.tensor(_emb,device=dev); _E=_E/_E.norm(dim=1,keepdim=True).clamp(min=1e-9)
_FAM=torch.tensor(np.concatenate([[f]*(_offs[i+1]-_offs[i]) for i,f in enumerate(_ids)]),device=dev)
print(f"models loaded | exemplar bank: {len(_ids)} families, {_emb.shape[0]} proteins",flush=True)

@torch.no_grad()
def embed(seqs,bs=24):
    o=[]
    for i in range(0,len(seqs),bs):
        t=tok([s[:1022] for s in seqs[i:i+bs]],return_tensors="pt",padding=True,
              truncation=True,max_length=1024).to(dev)
        h=esm(**t).last_hidden_state;m=t["attention_mask"].unsqueeze(-1).to(h.dtype);m[:,0]=0
        for j,L in enumerate(t["attention_mask"].sum(1)): m[j,L-1]=0
        o.append(((h*m).sum(1)/m.sum(1).clamp(min=1)).float().cpu().numpy()); del h,t
    return np.concatenate(o,0)

def _inp(pe):
    n=pe.shape[0]+1; x=np.zeros((1,n,480),dtype=np.float32); x[0,1:]=pe
    return dict(protein_embeddings=torch.tensor(x,device=dev).to(torch.bfloat16),
        special_tokens_mask=torch.tensor([[CLS]+[PROT]*pe.shape[0]],device=dev),
        token_type_ids=torch.zeros((1,n),dtype=torch.long,device=dev),return_dict=True)

@torch.no_grad()
def assign(pe):
    """Family = nearest real exemplar in embedding space (context-free, consistent)."""
    q=torch.tensor(pe,device=dev); q=q/q.norm(dim=1,keepdim=True).clamp(min=1e-9)
    sim=q@_E.T; best=sim.argmax(1)
    return _FAM[best].cpu().numpy(), sim.gather(1,best[:,None])[:,0].cpu().numpy()
@torch.no_grad()
def nextp(pe):   return torch.softmax(cm(**_inp(pe)).logits[0,-1].float(),-1).cpu().numpy()

G=json.load(open("eco_parsed.json")); D=A.loci
fam_gene=collections.defaultdict(collections.Counter)
fam_prod=collections.defaultdict(collections.Counter)
fam_strains=collections.defaultdict(set); fam_loci=collections.defaultdict(list)
obs=collections.defaultdict(set)          # (a,b) -> strains
pred=collections.defaultdict(list)        # (a,b) -> [p]
pobs=collections.defaultdict(list)        # observed edge -> model p
lanes=[]; ent=collections.defaultdict(list); hits=[]; ranks=[]

for gi,g in enumerate(G):
    # a pseudogene dnaA leaves no protein, so that lane starts at dnaN: shift it
    # back by one so every lane's locus index means the same genomic position
    off = 1 if g.get("dnaA_pseudo") else 0
    prots=g["proteins"][:D-off]; genes=g["genes"][:D-off]; prods=g["products"][:D-off]
    pe=embed(prots); fam,sim=assign(pe); k=min(len(fam),len(prots))
    strain=g["desc"].split(" (")[0]
    steps=[]
    for j in range(k):
        f=int(fam[j]); loc=j+off
        if genes[j]: fam_gene[f][genes[j]]+=1
        if prods[j]: fam_prod[f][prods[j]]+=1
        fam_strains[f].add(strain); fam_loci[f].append(loc)
    for j in range(1,k):
        p=nextp(pe[:j]); a,b=int(fam[j-1]),int(fam[j])
        e=float(-(p[p>1e-9]*np.log2(p[p>1e-9])).sum()); ent[j-1].append(e)
        order=np.argsort(-p); r=int(np.where(order==b)[0][0])
        ranks.append(r); hits.append(r==0)
        obs[(a,b)].add(strain); pobs[(a,b)].append(float(p[b]))
        for t in [int(x) for x in order[:A.topk]]:
            if t==b or t>=50000: continue
            pred[(a,t)].append(float(p[t]))
        steps.append(dict(locus=j+off,fam=b,gene=genes[j],sim=round(float(sim[j]),3),p=round(float(p[b]),5),
                          rank=r,hit=bool(r==0),entropy=round(e,3),
                          expected=[dict(fam=int(x),p=round(float(p[int(x)]),5)) for x in order[:3]]))
    lanes.append(dict(strain=strain,acc=g["acc"],pseudo=g.get("dnaA_pseudo",False),offset=off,
                      seed=dict(fam=int(fam[0]),gene=genes[0]),steps=steps))
    print(f"[{gi+1}/{len(G)}] {strain:26s} loci={k}",flush=True)

NS=len(G)
def lab(f):
    if fam_gene[f]: return fam_gene[f].most_common(1)[0][0]
    if fam_prod[f]: return fam_prod[f].most_common(1)[0][0][:38]
    return f"fam{f}"
nodes=[dict(id=f"f{f}",family=int(f),label=lab(f),
            product=fam_prod[f].most_common(1)[0][0] if fam_prod[f] else "",
            strains=sorted(fam_strains[f]),n_strains=len(fam_strains[f]),
            locus=int(np.median(fam_loci[f])),
            partition="core" if len(fam_strains[f])==NS else
                      ("shell" if len(fam_strains[f])>1 else "cloud"))
       for f in fam_strains]
edges=[dict(source=f"f{a}",target=f"f{b}",kind="observed",n_strains=len(s),
            p=round(float(np.mean(pobs[(a,b)])),5)) for (a,b),s in obs.items()]
edges+=[dict(source=f"f{a}",target=f"f{b}",kind="predicted",n_strains=0,
             p=round(float(np.mean(v)),5)) for (a,b),v in pred.items() if (a,b) not in obs]
meta=dict(species="Escherichia coli",n_strains=NS,loci=D,topk=A.topk,
    model="macwiatrak/bacformer-causal-complete-genomes",
    anchor="dnaA locus, canonical orientation",
    n_nodes=len(nodes),n_observed=sum(1 for e in edges if e["kind"]=="observed"),
    n_predicted=sum(1 for e in edges if e["kind"]=="predicted"),
    core=sum(1 for n in nodes if n["partition"]=="core"),
    shell=sum(1 for n in nodes if n["partition"]=="shell"),
    cloud=sum(1 for n in nodes if n["partition"]=="cloud"),
    top1=round(float(np.mean(hits)),3),median_rank=int(np.median(ranks)),
    entropy_curve=[round(float(np.mean(ent[d])),3) for d in sorted(ent)],
    strains=[dict(strain=g["desc"].split(" (")[0],desc=g["desc"],acc=g["acc"]) for g in G])
json.dump(dict(meta=meta,nodes=nodes,edges=edges,lanes=lanes),open(A.out,"w"))
print(f"\n{NS} strains, {D} loci")
print(f"  families      {len(nodes)}   core {meta['core']} | shell {meta['shell']} | cloud {meta['cloud']}")
print(f"  observed edges {meta['n_observed']}   predicted-only edges {meta['n_predicted']}")
print(f"  next-family top1 {meta['top1']:.3f}   median rank {meta['median_rank']}")
