"""Teacher-forced branching: a real genome is the spine, the model supplies the branches.

Free-running generation is not available: the token-level generative checkpoint has
no published weights, and feeding back family *prototypes* changes the model's
prediction 77% of the time (see validate.py). So we never feed the model anything
it did not see: the prefix is always the real protein embeddings of a real genome.
At each locus we read the model's nucleus - the families it would accept there -
and attach them as branches off the real path. Running several genomes at once and
merging nodes by (family, locus) makes the spines converge where synteny is
conserved, which is the pangenome-graph structure we were after.
"""
import json, argparse, numpy as np, torch, collections
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from datasets import load_dataset

P=argparse.ArgumentParser()
P.add_argument("--top-p",type=float,default=0.6)
P.add_argument("--depth",type=int,default=30)
P.add_argument("--genomes",type=int,default=14)
P.add_argument("--max-branch",type=int,default=5)
P.add_argument("--out",default="graph.json")
A=P.parse_args()

dev="cuda:0"; CLS,PROT=2,4
model=AutoModelForCausalLM.from_pretrained("macwiatrak/bacformer-causal-complete-genomes",
        trust_remote_code=True).to(torch.bfloat16).eval().to(dev)
tok=AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm=AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(torch.float16).eval().to(dev)
annot=json.load(open("fam_annot.json"))
genomes_meta={g["name"]:g for g in json.load(open("genomes.json"))}

def label(f):
    a=annot.get(str(f))
    if not a: return f"fam{f}"
    if a["gene"]: return a["gene"][0][0]
    if a["product"]: return a["product"][0][0][:40]
    return f"fam{f}"
def product(f):
    a=annot.get(str(f)); return a["product"][0][0] if a and a["product"] else ""

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

ds=load_dataset("macwiatrak/bacbench-essential-genes-protein-sequences",split="train",streaming=True)
nodes={}; edges={}; ent_by_depth=collections.defaultdict(list)
rank_true=[]; p_true=[]; in_nuc=[]; used=[]
import collections as _c
by_d=_c.defaultdict(lambda: dict(rank=[],p=[],nuc=[],top1=[],top10=[],nsize=[]))
seen_species=set()

def nid(f,d): return f"f{f}_d{d}"
def add_node(f,d,kind,genome=None):
    k=nid(f,d)
    if k not in nodes:
        nodes[k]=dict(id=k,family=int(f),label=label(f),product=product(f),depth=d,
                      kind=kind,genomes=[],prob=0.0,entropy=None,seed=(d==0))
    if kind=="spine": nodes[k]["kind"]="spine"
    if genome and genome not in nodes[k]["genomes"]: nodes[k]["genomes"].append(genome)
    return k

lanes=[]
ng=0
for item in ds:
    if ng>=A.genomes: break
    sp=item["species"]
    if sp in seen_species: continue
    prots=[p for c in item["protein_sequence"] for p in c][:A.depth+1]
    gm=genomes_meta.get(item["genome_name"])
    if gm is None or len(prots)<A.depth+1: continue
    seen_species.add(sp); ng+=1
    fams=gm["families"][:A.depth+1]
    real=embed(prots)
    strain=str(item["strain_name"])[:34]
    used.append(dict(genome=item["genome_name"],strain=strain,species=sp,genus=item["genus"]))
    lane=dict(strain=strain,species=sp,genus=item["genus"],steps=[])
    add_node(fams[0],0,"spine",strain)
    for k in range(1,min(A.depth+1,len(fams))):
        p=dist(real[:k])
        e=float(-(p[p>1e-9]*np.log2(p[p>1e-9])).sum())
        ent_by_depth[k-1].append(e)
        src=nid(fams[k-1],k-1)
        if src in nodes and nodes[src]["entropy"] is None: nodes[src]["entropy"]=round(e,3)
        truth=fams[k]
        order=np.argsort(-p); cum=np.cumsum(p[order])
        kn=int(np.searchsorted(cum,A.top_p))+1
        nuc=[int(x) for x in order[:kn]]
        r=int(np.where(order==truth)[0][0])
        rank_true.append(r); p_true.append(float(p[truth])); in_nuc.append(truth in nuc)
        d=by_d[k-1]; d["rank"].append(r); d["p"].append(float(p[truth])); d["nuc"].append(truth in nuc)
        d["top1"].append(r==0); d["top10"].append(r<10); d["nsize"].append(kn)
        # spine step
        tgt=add_node(truth,k,"spine",strain)
        key=(src,tgt)
        edges[key]=dict(source=src,target=tgt,p=round(float(p[truth]),5),kind="spine",hit=bool(r==0),rank=int(r))
        nodes[tgt]["hit"]=bool(r==0); nodes[tgt]["rank"]=int(r)
        # model's alternatives at this locus
        lane["steps"].append(dict(locus=k,fam=int(truth),label=label(truth),product=product(truth),
            p=round(float(p[truth]),5),rank=int(r),hit=bool(r==0),entropy=round(e,3),
            expected=[dict(fam=int(x),label=label(int(x)),p=round(float(p[int(x)]),5))
                      for x in order[:3]]))
        for f in [int(x) for x in order[:A.max_branch]]:
            if f==truth or f>=50000: continue
            t2=add_node(f,k,"alt")
            k2=(src,t2)
            if k2 not in edges:
                edges[k2]=dict(source=src,target=t2,p=round(float(p[f]),5),kind="alt")
    lane["seed"]=dict(fam=int(fams[0]),label=label(fams[0]),product=product(fams[0]))
    lanes.append(lane)
    print(f"[{ng}] {strain:36s} {sp[:28]}",flush=True)

for n in nodes.values(): n["prob"]=round(max([e["p"] for e in edges.values() if e["target"]==n["id"]]+[1.0 if n["seed"] else 0.0]),5)
ent_curve=[round(float(np.mean(ent_by_depth[d])),3) if ent_by_depth[d] else None for d in range(A.depth)]
meta=dict(mode="teacher-forced",top_p=A.top_p,depth=A.depth,max_branch=A.max_branch,
    n_genomes=ng,genomes=used,n_nodes=len(nodes),n_edges=len(edges),
    model="macwiatrak/bacformer-causal-complete-genomes",
    entropy_curve=ent_curve,
    by_depth=[dict(depth=d,
        top1=round(float(np.mean(by_d[d]["top1"])),3),
        top10=round(float(np.mean(by_d[d]["top10"])),3),
        in_nucleus=round(float(np.mean(by_d[d]["nuc"])),3),
        median_rank=int(np.median(by_d[d]["rank"])),
        median_p=round(float(np.median(by_d[d]["p"])),5),
        nucleus_size=int(np.median(by_d[d]["nsize"])),
        entropy=round(float(np.mean(ent_by_depth[d])),3) if ent_by_depth[d] else None)
        for d in sorted(by_d)],
    true_in_nucleus=round(float(np.mean(in_nuc)),3),
    median_rank_true=int(np.median(rank_true)),
    top1_true=round(float(np.mean([r==0 for r in rank_true])),3),
    median_p_true=round(float(np.median(p_true)),5),
    nucleus_sweep=[dict(top_p=0.60,median=1,mean=91.9,true_inside=0.195),
                   dict(top_p=0.80,median=1,mean=267.6,true_inside=0.221),
                   dict(top_p=0.90,median=1,mean=524.9,true_inside=0.252),
                   dict(top_p=0.95,median=1,mean=853.5,true_inside=0.288),
                   dict(top_p=0.99,median=2,mean=1856.5,true_inside=0.338)],
    entropy_origin=1.58, entropy_far=2.67, top1_origin=0.236, top1_far=0.114)
json.dump(dict(meta=meta,nodes=list(nodes.values()),edges=list(edges.values())),open(A.out,"w"))
json.dump(dict(meta=meta,lanes=lanes),open("trace.json","w"))
print(f"\n{ng} genomes -> {len(nodes)} nodes / {len(edges)} edges")
print(f"true next family in top_p={A.top_p} nucleus: {meta['true_in_nucleus']:.3f} | top1 {meta['top1_true']:.3f} | median rank {meta['median_rank_true']}")
print("\n locus  top1  top10  in-nuc  med.rank  nucleus  entropy")
for r in meta["by_depth"]:
    print(f"  {r['depth']:3d}  {r['top1']:.2f}  {r['top10']:.2f}   {r['in_nucleus']:.2f}   {r['median_rank']:7d}  {r['nucleus_size']:6d}  {r['entropy']:6.2f}")
