"""The dnaA-window pangenome graph of all 2,002 PanGBank genomes, with Bacformer's
calls aggregated on it, as data for the PanGramGraph page.

Nodes are PPanGGOLiN families, coloured by PPanGGOLiN's own partition, placed at
the locus where they most often sit. Edges are adjacencies weighted by the number
of genomes carrying them. For each family, the model's call after it is averaged
over every genome carrying it (mean softmax probability, from each call's top 15),
with how often the model named the gene that actually follows.

A dashed box is a family the model expects after the best-carried family of a
column but that follows it in none of the 2,002 genomes.

    python3 pgb_graph.py     # -> pgb/graph_pgb.json, and prints the summary tables
"""
import json, collections, numpy as np
from scipy import sparse
W=json.load(open("pgb/window.json"))
C={k:v for k,v in np.load("pgb/model_calls.npz").items()}      # load once: NpzFile re-reads on every access
annot=json.load(open("fam_annot.json")); bf=C["bf_family"]
GEN=W["genomes"]; FAMS={int(k):v for k,v in W["families"].items()}; NG=len(GEN)
def bflabel(f):
    a=annot.get(str(int(f)))
    if not a: return f"fam{int(f)}"
    if a["gene"]: return a["gene"][0][0]
    return a["product"][0][0] if a["product"] else f"fam{int(f)}"

# --- index the calls by (genome, locus) --------------------------------------
call={(int(g),int(l)):i for i,(g,l) in enumerate(zip(C["genome"],C["locus"]))}
occ=collections.defaultdict(list)                  # family -> [(genome, lane index)]
carriers=collections.defaultdict(set); loci=collections.defaultdict(collections.Counter)
names=collections.defaultdict(collections.Counter); prods=collections.defaultdict(collections.Counter)
rgp=collections.defaultdict(list); nxt_obs=collections.defaultdict(collections.Counter)
nxt_bf=collections.defaultdict(set); edges=collections.Counter(); present_bf=set()
pids_of=collections.defaultdict(collections.Counter)      # family -> its proteins in the windows
for gi,g in enumerate(GEN):
    L=g["genes"]; s=g["start"]
    for k,(gid,f,r,pid,nm,pr,st) in enumerate(L):
        occ[f].append((gi,k)); carriers[f].add(gi); loci[f][k+s]+=1; rgp[f].append(r); pids_of[f][pid]+=1
        if nm: names[f][nm]+=1
        if pr: prods[f][pr]+=1
        present_bf.add(int(bf[pid]))
        if k+1<len(L):
            f2=L[k+1][1]; nxt_obs[f][f2]+=1; edges[(f,f2)]+=1; nxt_bf[f].add(int(bf[L[k+1][3]]))

# --- the model's own names for the families ----------------------------------------
# Bacformer predicts among its 50,000 clusters; the exemplar bank holds 9,814 of them, so a
# family whose cluster is missing is given the nearest one the bank has (yidB: 13961, while
# the model predicts 44543 where yidB sits, in 1,969 of 1,976 genomes). The decoder reads a
# cluster as the families actually there when the model puts mass on it: P(family | cluster),
# from the top 15 of all 130,837 calls. For accuracy it is learned on one half of the
# genomes and applied to the other, so no call is read with a decoder that saw it.
FI={f:i for i,f in enumerate(sorted(FAMS))}; FR=np.array(sorted(FAMS)); NF=len(FI)
fam_at=np.array([FI[GEN[int(g)]["genes"][int(l)-GEN[int(g)]["start"]][1]] for g,l in zip(C["genome"],C["locus"])])
def decoder(mask):
    A=sparse.csr_matrix((C["top_p"][mask].ravel(),(np.repeat(fam_at[mask],C["top_f"].shape[1]),C["top_f"][mask].ravel())),shape=(NF,50001))
    return A,np.asarray(A.sum(0)).ravel()
A_all,tot_all=decoder(np.ones(len(fam_at),bool))
Ac=A_all.tocsc()
def reads(c,min_mass=1.0):
    """the family index a cluster reads as, and how clearly; None below one genome's worth of mass"""
    if tot_all[c]<min_mass: return None,0.0
    col=Ac.getcol(c); i=col.indices[col.data.argmax()]; return int(FR[i]),float(col.data.max()/tot_all[c])
def dec_weights(f):
    row=A_all.getrow(FI[f]); out=[[int(c),round(float(v/tot_all[c]),4)] for c,v in zip(row.indices,row.data) if v>=0.5 and v/tot_all[c]>=0.05]
    return sorted(out,key=lambda x:-x[1])[:12]
def dec_accuracy():
    """top-1 read through the decoder of the other half of the genomes, and the probability it gives the
    family actually there (the model's top 15 clusters read as families)"""
    hit=np.zeros(len(fam_at),bool); ptrue=np.zeros(len(fam_at)); half=C["genome"]%2
    for h in (0,1):
        A,t=decoder(half!=h); Pn=sparse.diags(1/np.maximum(t,1e-12))@A.T                      # 50001 x NF, columns sum to 1 where seen
        m=np.where(half==h)[0]
        Q=sparse.csr_matrix((C["top_p"][m].ravel(),(np.repeat(np.arange(len(m)),C["top_f"].shape[1]),C["top_f"][m].ravel())),shape=(len(m),50001))
        S=(Q@Pn).toarray(); hit[m]=(S.argmax(1)==fam_at[m])&(S.max(1)>0); ptrue[m]=S[np.arange(len(m)),fam_at[m]]
    return hit,ptrue
HIT,PTRUE=dec_accuracy()
# perplexity on the family actually there: 2 ** mean(-log2 P). A family none of the model's 15 best clusters
# reads as (1.5 % of the calls) is given the probability of the 15th, the most any one cluster outside the top 15
# can have: the perplexity errs on the low side
P_EFF=np.where(PTRUE>0,PTRUE,C["top_p"][:,-1])
ppl=lambda m: round(float(2**np.mean(-np.log2(np.maximum(P_EFF[m],1e-12)))),3)
print(f"top-1: exemplar bank {np.mean(C['rank']==1)*100:.1f}%  |  model's own names, cross-validated {HIT.mean()*100:.1f}%",flush=True)

# reference strains first when a card offers a genome (the 11 of the first walk that PanGBank keeps)
PREF={acc:i for i,acc in enumerate(json.load(open("eco/manifest.json")))}
# --- per family: the model's call after it, averaged over carriers -------------
agg={}
for f,oc in occ.items():
    acc=collections.defaultdict(float); n=0; hit=0; hitd=0; ranks=[]
    for gi,k in oc:
        i=call.get((gi,k+1+GEN[gi]["start"]))
        if i is None: continue
        n+=1; hit+=int(C["rank"][i]==1); ranks.append(int(C["rank"][i])); hitd+=int(HIT[i])
        for ff,pp in zip(C["top_f"][i],C["top_p"][i]): acc[int(ff)]+=float(pp)
    top=sorted(acc.items(),key=lambda x:-x[1])[:8]
    rd=[reads(ff) for ff,_ in top]
    agg[f]=dict(calls=n,top1=round(hit/n,3) if n else None,top1_dec=round(hitd/n,3) if n else None,med_rank=int(np.median(ranks)) if ranks else None,
                # [cluster, mean p, follows it in some genome, the family it reads as]
                call=[[ff,round(pp/n,5),int(r is not None and r in nxt_obs[f]),f"p{r}" if r is not None else ""]
                      for (ff,pp),(r,_) in zip(top,rd)] if n else [])

def bf_major(f):
    c=collections.Counter()
    for pid,n in pids_of[f].items(): c[int(bf[pid])]+=n
    return c.most_common(1)[0][0]
cl_count=collections.Counter()                 # model cluster -> window occurrences
for f,c in pids_of.items():
    for pid,n in c.items(): cl_count[int(bf[pid])]+=n
def bf_weights(f):
    """[cluster, P(family | cluster)]: a box's probability of coming next is the sum over its
    clusters of p(cluster) x this weight, so a cluster shared by two families is not counted twice"""
    c=collections.Counter()
    for pid,n in pids_of[f].items(): c[int(bf[pid])]+=n
    return [[b,round(n/cl_count[b],4)] for b,n in c.most_common()]
def bf_all(f):
    """every model cluster holding >= 10 % of the family's proteins: families split across clusters"""
    c=collections.Counter()
    for pid,n in pids_of[f].items(): c[int(bf[pid])]+=n
    t=sum(c.values()); return [b for b,n in c.most_common() if n>=0.1*t]
def label(f):
    return names[f].most_common(1)[0][0] if names[f] else (prods[f].most_common(1)[0][0] if prods[f] else FAMS[f][0])
nodes=[dict(id=f"p{f}",fam=FAMS[f][0],label=label(f),named=bool(names[f]),
            product=prods[f].most_common(1)[0][0] if prods[f] else "",partition=FAMS[f][1],
            n=len(carriers[f]),locus=loci[f].most_common(1)[0][0],rgp=round(float(np.mean(rgp[f])),3),
            pid=pids_of[f].most_common(1)[0][0],bf=bf_major(f),
            bfw=dec_weights(f) or bf_weights(f),                                  # clusters that read as it, P(it | cluster)
            bfs=[c for c,w in (dec_weights(f) or bf_weights(f)) if w>=0.5] or bf_all(f),
            next=[[f"p{f2}",c] for f2,c in nxt_obs[f].most_common(5)],
            ex=sorted(carriers[f],key=lambda gi:(PREF.get(GEN[gi]["acc"],99),GEN[gi].get("level")!="Complete Genome",gi))[:10],**agg[f])
       for f in occ]
E=[dict(s=f"p{a}",t=f"p{b}",n=c) for (a,b),c in edges.items()]

# --- ghosts: expected after the best-carried family of a column, never observed there
bycol=collections.defaultdict(list)
for nd in nodes: bycol[nd["locus"]].append(nd)
ghosts=[]
for col,nds in bycol.items():
    src=max(nds,key=lambda x:x["n"]); f=int(src["id"][1:])
    cand=[(ff,pp,r) for ff,pp,seen,r in src["call"] if not seen][:2]
    for i,(ff,pp,r) in enumerate(cand):                     # what the model expects, read as a family if it can be
        ghosts.append(dict(src=src["id"],locus=col+1,bf=ff,label=label(int(r[1:])) if r else bflabel(ff),p=pp,
                           reads=r,elsewhere=int(bool(r))))

# --- per-locus curve and the tables by partition / RGP --------------------------
part_of={}; rgp_of={}
for gi,g in enumerate(GEN):
    for k,x in enumerate(g["genes"]): part_of[(gi,k+g["start"])]=FAMS[x[1]][1]; rgp_of[(gi,k+g["start"])]=x[2]
pl=[]
for l in range(1,W["window"]):
    m=C["locus"]==l
    if m.sum(): pl.append(dict(locus=l,n=int(m.sum()),entropy=round(float(C["entropy"][m].mean()),3),ppl=ppl(m),
                               unread=round(float((PTRUE[m]==0).mean()),4),
                               top1=round(float((C["rank"][m]==1).mean()),3),top1_dec=round(float(HIT[m].mean()),3)))
def table(key):
    rows=collections.defaultdict(list)
    for i,(g,l) in enumerate(zip(C["genome"],C["locus"])): rows[key(int(g),int(l))].append(i)
    out={}
    for k,ix in rows.items():
        ix=np.array(ix); out[str(k)]=dict(n=len(ix),top1=round(float((C["rank"][ix]==1).mean()),3),top1_dec=round(float(HIT[ix].mean()),3),
            med_rank=int(np.median(C["rank"][ix])),entropy=round(float(C["entropy"][ix].mean()),3))
    return out
by_part=table(lambda g,l:part_of[(g,l)]); by_rgp=table(lambda g,l:"inside an RGP" if rgp_of[(g,l)] else "outside an RGP")
genomes=[dict(acc=g["acc"],strain=g.get("strain",""),org=g.get("org",""),level=g.get("level",""),n=len(g["genes"])) for g in GEN]
out=dict(meta=dict(pangenome=W["pangenome"],n_genomes=NG,window=W["window"],calls=int(len(C["rank"])),
                   top1=round(float((C["rank"]==1).mean()),3),top1_dec=round(float(HIT.mean()),3),median_rank=int(np.median(C["rank"])),
                   ppl=ppl(np.ones(len(P_EFF),bool)),unread=round(float((PTRUE==0).mean()),4),
                   full_windows=sum(len(g["genes"])+g["start"]>=W["window"] for g in GEN),
                   families=len(nodes),edges=len(E)),
         per_locus=pl,by_partition=by_part,by_rgp=by_rgp,nodes=nodes,edges=E,ghosts=ghosts,genomes=genomes,
         bflabels={str(ff):bflabel(ff) for nd in nodes for ff,*_ in nd["call"]})
json.dump(out,open("pgb/graph_pgb.json","w"),separators=(",",":"))
import os; print(f"graph_pgb.json {os.path.getsize('pgb/graph_pgb.json')/1e6:.1f} MB | {len(nodes)} families, {len(E)} edges, {len(ghosts)} ghosts")
print("families by genomes carrying:",{t:sum(nd['n']>=t for nd in nodes) for t in (1,2,5,20,100,500,1000,1900)})
print("\nby partition of the gene being predicted:");[print(f"  {k:10s}",v) for k,v in by_part.items()]
print("by RGP:");[print(f"  {k:15s}",v) for k,v in by_rgp.items()]
print("\nper locus (1-12) strict / decoded / entropy / perplexity:",[(r['locus'],r['top1'],r['top1_dec'],r['entropy'],r['ppl']) for r in pl[:12]])
print(f"perplexity on the family actually there, all calls: {out['meta']['ppl']} ({out['meta']['unread']:.2%} of calls read at the 15th cluster)")
