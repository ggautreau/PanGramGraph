"""Where does the upstream context decide which branch a genome takes at a fork?

A fork is a family followed by at least two families, each in >= 20 of the 2,002
PanGBank genomes. The pangenome graph only knows how often each branch is taken. If
Bacformer, reading the genome's proteins from dnaA up to the fork, predicts the
branch better than those frequencies, something upstream carries information about
what comes next: a regularity that single adjacencies cannot show.

For each genome at each fork, the model's full softmax (one causal pass per genome)
gives each branch a score, sum over the branch family's model clusters of
p(cluster) x P(family | cluster), normalised over the branches. Compared, in bits
per genome (log loss) and accuracy, against two baselines:

  frequency   how often each branch is taken, across all genomes (the graph)
  same-11     leave-one-out majority branch among the other genomes whose 11 proteins
              up to and including the fork are identical: lineage lookup. If the model
              only matches it, it is recognising a lineage, not a rule.
  phylo-15    the branches taken by the 15 nearest genomes on core-genome alleles
              (pgb_cgmlst.py), shrunk to the frequencies: the lineage baseline proper.

    python3 fork_rules.py        # -> pgb/fork_rules.json, printed table
"""
import json, collections, numpy as np, torch
from transformers import AutoModelForCausalLM
dev="cuda:0"; CLS,PROT=2,4; MINB=20; HAP=11; K=15
cm=AutoModelForCausalLM.from_pretrained("macwiatrak/bacformer-causal-complete-genomes",
     trust_remote_code=True).to(torch.bfloat16).eval().to(dev)
W=json.load(open("pgb/window.json")); EMB=np.load("pgb/window_emb.npy").astype(np.float32)
P=json.load(open("pgb/graph_pgb.json")); node={int(n["id"][1:]):n for n in P["nodes"]}
G=W["genomes"]
D=np.load("pgb/cgmlst_dist.npy").astype(np.float32)
assert json.load(open("pgb/cgmlst_loci.json"))["genomes"]==[g["acc"] for g in G]

succ=collections.defaultdict(collections.Counter)
for g in G:
    L=g["genes"]
    for k in range(len(L)-1): succ[L[k][1]][L[k+1][1]]+=1
forks={f:[b for b,c in cnt.most_common(6) if c>=MINB] for f,cnt in succ.items()}
forks={f:b for f,b in forks.items() if len(b)>=2}
# per fork: a (clusters x branches) weight matrix, P(branch family | cluster)
WM={}
for f,br in forks.items():
    cl=sorted({c for b in br for c,_ in node[b]["bfw"]})
    M=np.zeros((len(cl),len(br)),np.float32)
    for j,b in enumerate(br):
        for c,w in node[b]["bfw"]: M[cl.index(c),j]=w
    WM[f]=(torch.tensor(cl,device=dev),torch.tensor(M,device=dev))

obs=collections.defaultdict(list)              # fork -> [(genome, truth index, model q over branches, haplotype)]
with torch.no_grad():
    for gi,g in enumerate(G):
        L=g["genes"]; n=len(L)
        sites=[k for k in range(n-1) if L[k][1] in forks and L[k+1][1] in forks[L[k][1]]]
        if not sites: continue
        x=np.zeros((1,n+1,480),np.float32); x[0,1:]=EMB[[y[3] for y in L]]
        lg=cm(protein_embeddings=torch.tensor(x,device=dev).to(torch.bfloat16),
              special_tokens_mask=torch.tensor([[CLS]+[PROT]*n],device=dev),
              token_type_ids=torch.zeros((1,n+1),dtype=torch.long,device=dev),return_dict=True).logits[0].float()
        for k in sites:
            f=L[k][1]; cl,M=WM[f]
            p=torch.softmax(lg[k+1],-1)                 # output after protein k: the call for protein k+1
            s=(p[cl][:,None]*M).sum(0).cpu().numpy()
            hap=tuple(y[3] for y in L[max(0,k-HAP+1):k+1])
            obs[f].append((gi,forks[f].index(L[k+1][1]),s,hap))
        if gi%500==0: print(f"  {gi}/{len(G)}",flush=True)

EPS=1e-3
def norm(v):
    v=np.asarray(v,float)+EPS*max(float(np.sum(v)),1e-12); return v/v.sum()
out=[]
for f,rows in obs.items():
    br=forks[f]; nb=len(br)
    freq=np.bincount([r[1] for r in rows],minlength=nb)+0.5; freq=freq/freq.sum()
    byhap=collections.defaultdict(list)
    for r in rows: byhap[r[3]].append(r[1])
    ll={"frequency":0.0,"same-11":0.0,"phylo-15":0.0,"model":0.0}; acc={k:0 for k in ll}; uniq=[0,0,0.0,0.0]   # n, model hits, ll model, ll freq
    gis=np.array([r[0] for r in rows]); tru=np.array([r[1] for r in rows])
    Ds=D[np.ix_(gis,gis)].copy(); Ds[gis[:,None]==gis[None,:]]=np.inf            # never a genome's own vote
    kk=min(K,len(rows)-1); nn=np.argpartition(Ds,kk-1,axis=1)[:,:kk]
    for i,(gi,t,s,hap) in enumerate(rows):
        qm=norm(s); others=list(byhap[hap]); others.remove(t)
        qh=norm(np.bincount(others,minlength=nb)+freq) if others else freq      # leave-one-out, shrunk to frequency
        qp=(np.bincount(tru[nn[i]],minlength=nb)+freq)/(kk+1)                   # nearest relatives, shrunk to frequency
        for k,q in (("frequency",freq),("same-11",qh),("phylo-15",qp),("model",qm)):
            ll[k]+=-np.log2(q[t]); acc[k]+=int(q.argmax()==t)
        if not others:                                      # a haplotype no other genome shares: lookup cannot help
            uniq[0]+=1; uniq[1]+=int(qm.argmax()==t); uniq[2]+=-np.log2(qm[t]); uniq[3]+=-np.log2(freq[t])
    n=len(rows)
    out.append(dict(fork=node[f]["label"],fam=node[f]["fam"],column=node[f]["locus"],genomes=n,
        branches=[[node[b]["label"],succ[f][b]] for b in br],
        bits={k:round(v/n,3) for k,v in ll.items()},acc={k:round(v/n,3) for k,v in acc.items()},
        gain_vs_frequency=round((ll["frequency"]-ll["model"])/n,3),gain_vs_same11=round((ll["same-11"]-ll["model"])/n,3),
        gain_vs_phylo=round((ll["phylo-15"]-ll["model"])/n,3),
        unique_haplotypes=dict(n=uniq[0],model_acc=round(uniq[1]/uniq[0],3) if uniq[0] else None,
                               gain_vs_frequency=round((uniq[3]-uniq[2])/uniq[0],3) if uniq[0] else None)))
out.sort(key=lambda r:-r["gain_vs_frequency"])
json.dump(out,open("pgb/fork_rules.json","w"),indent=1)
print(f"\n{len(out)} forks. Model beats frequency by > 0.1 bit/genome at {sum(r['gain_vs_frequency']>0.1 for r in out)}; "
      f"beats the 15 nearest relatives at {sum(r['gain_vs_phylo']>0 for r in out)}; relatives beat frequency by > 0.1 bit at "
      f"{sum((r['bits']['frequency']-r['bits']['phylo-15'])>0.1 for r in out)}")
print("fork (col)          genomes  bits: freq  same11 phylo  model | acc: freq phylo model | branches")
for r in out[:14]:
    print(f"{r['fork'][:17]:17s}({r['column']:2d}) {r['genomes']:6d}   {r['bits']['frequency']:5.2f} {r['bits']['same-11']:6.2f} {r['bits']['phylo-15']:5.2f} {r['bits']['model']:6.2f} |"
          f"  {r['acc']['frequency']:4.0%} {r['acc']['phylo-15']:5.0%} {r['acc']['model']:5.0%} | "
          +" / ".join(f"{a[:14]}:{b}" for a,b in r["branches"]))
if False:
    u=None
    print(f"{r['fork'][:17]:17s}({r['column']:2d}) {r['genomes']:6d}   {r['bits']['frequency']:5.2f} {r['bits']['same-11']:6.2f} {r['bits']['model']:6.2f} |"
          f"   {r['acc']['frequency']:4.0%} {r['acc']['model']:5.0%} | {u['n']:4d} {u['model_acc'] if u['model_acc'] is not None else '-'!s:5s} {u['gain_vs_frequency'] if u['gain_vs_frequency'] is not None else '-'!s:6s} | "
          +" / ".join(f"{a[:14]}:{b}" for a,b in r["branches"]))
