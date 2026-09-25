"""Allelic contingency without any language model.

The prediction stated in DISCOVERY.md, tested on the pangenome alone: at a fork, two
genomes carrying the SAME allele of the fork family should take the same branch more
often than two genomes carrying different alleles — matched on the upstream family path
(so the pangenome graph sees the same thing for both pairs) and stratified by
phylogenetic distance (so it is not lineage).

For every fork, every pair of sites from two different genomes whose upstream family
path over L genes is identical:
    same_allele   the two genomes have the identical protein of the fork family
                  (same sequence id in the HDF5: no embedding, no model, no threshold)
    same_branch   they go on to the same family
    d             their cgMLST distance, binned
Every comparison is made inside a stratum = (fork, exact upstream family path, distance
bin), so the two pairs being compared are identical for everything the pangenome graph
holds and comparable in phylogeny. Mantel-Haenszel odds ratio over strata, and a
permutation test that shuffles the allele label within each stratum.

    python3 pg_noLM.py [--L 5] [--bins 6]
"""
import json, argparse, collections, itertools, warnings, numpy as np
warnings.simplefilter("ignore")
ap=argparse.ArgumentParser(); ap.add_argument("--L",type=int,default=5); ap.add_argument("--bins",type=int,default=6)
ap.add_argument("--max_pairs",type=int,default=200000); ap.add_argument("--perm",type=int,default=1000); A=ap.parse_args()
rng=np.random.default_rng(0)
Z=np.load("pgb/chrom.npz"); off=Z["offsets"]; FAM=Z["fam"]; PID=Z["pid"]; RGP=Z["rgp"]; NG=len(off)-1
J=json.load(open("pgb/chrom_genomes.json")); FN=J["families"]
SITES=np.load("pgb/fork_sites.npy"); gen_of=np.repeat(np.arange(NG),np.diff(off))
D=np.load("pgb/cgmlst_dist.npy").astype(np.float32); acc=json.load(open("pgb/cgmlst_loci.json"))["genomes"]
ix={a:i for i,a in enumerate(acc)}; sel=np.array([ix[g["acc"]] for g in J["genomes"]]); D=D[np.ix_(sel,sel)]
AL=json.load(open("pgb/allelic.json")); cls={r["fork"]:r["cls"] for r in AL["per_fork"]}
gap={r["fork"]:r["bits"]["family"]-r["bits"]["model"] for r in AL["per_fork"]}
occ=collections.Counter(); succ=collections.defaultdict(collections.Counter)
for p in SITES: succ[int(FAM[p-1])][int(FAM[p])]+=1; occ[int(FAM[p-1])]+=1
by_fork=collections.defaultdict(list)
for p in SITES: by_fork[int(FAM[p-1])].append(int(p))
edges=np.quantile(D[np.triu_indices(NG,1)],np.linspace(0,1,A.bins+1)); edges[0]=-1; edges[-1]=2
L=A.L; rec=[]                                  # (fork, bin, same_allele, same_branch, rgp, class, gap)
for F,ps in by_fork.items():
    br=[b for b,c in succ[F].most_common(8) if c>=max(0.10*occ[F],20)]
    if len(br)<2: continue
    ps=[p for p in ps if FAM[p] in br and (p-off[gen_of[p]])>L]
    if len(ps)<30: continue
    byk=collections.defaultdict(list)
    for p in ps: byk[tuple(FAM[p-L:p].tolist())].append(p)
    name=FN[F]
    for k,grp in byk.items():
        if len(grp)<4: continue
        pr=list(itertools.combinations(grp,2))
        if len(pr)>4000: pr=[pr[i] for i in rng.choice(len(pr),4000,replace=False)]
        for a,b in pr:
            ga,gb=gen_of[a],gen_of[b]
            if ga==gb: continue
            rec.append((name,f"{name}|{hash(k)&0xffffff}|{int(np.searchsorted(edges,D[ga,gb],'right')-1)}",
                        int(np.searchsorted(edges,D[ga,gb],"right")-1),
                        int(PID[a-1]==PID[b-1]),int(FAM[a]==FAM[b]),int(RGP[a] or RGP[b])))
print(f"{len(rec)} pairs over {len({r[0] for r in rec})} forks, family path of {L} genes",flush=True)
if len(rec)>A.max_pairs: rec=[rec[i] for i in rng.choice(len(rec),A.max_pairs,replace=False)]
R=np.array([(r[2],r[3],r[4],r[5]) for r in rec],np.int64)
forks=np.array([r[0] for r in rec]); strat=np.array([r[1] for r in rec])
bn,sa,sb,rg=R[:,0],R[:,1],R[:,2],R[:,3]
print(f"\npairs: {len(R)}; same allele {sa.mean():.1%}; same branch {sb.mean():.1%}")
print(f"{'cgMLST distance bin':22s}{'pairs':>9s}{'same allele':>13s}{'P(same branch) same':>21s}{'diff':>8s}{'gap':>8s}")
for b in range(A.bins):
    m=bn==b
    if m.sum()<50: continue
    x=sb[m&(sa==1)]; y=sb[m&(sa==0)]
    if not len(x) or not len(y): continue
    # Mantel-Haenszel odds ratio accumulators
    print(f"[{edges[b]:.3f},{edges[b+1]:.3f}]{m.sum():>11d}{np.mean(sa[m]):>12.1%}{x.mean():>20.1%}{y.mean():>8.1%}{x.mean()-y.mean():>+8.1%}")
# --- everything below is stratified by (fork, exact family path, distance bin) ----------
order=np.argsort(strat); ks=strat[order]; bounds=np.flatnonzero(np.r_[True,ks[1:]!=ks[:-1],True])
SA,SB=sa[order],sb[order]
num=den=0.0; wd=wn=0.0
for i in range(len(bounds)-1):
    s_,e_=bounds[i],bounds[i+1]; x=SB[s_:e_][SA[s_:e_]==1]; y=SB[s_:e_][SA[s_:e_]==0]
    if not len(x) or not len(y): continue
    a_,b_,c_,d_=x.sum(),len(x)-x.sum(),y.sum(),len(y)-y.sum(); n=len(x)+len(y)
    num+=a_*d_/n; den+=b_*c_/n; w=len(x)*len(y)/n; wd+=w*(x.mean()-y.mean()); wn+=w
informative=sum(1 for i in range(len(bounds)-1)
                if len(set(SA[bounds[i]:bounds[i+1]]))==2)
print(f"\nstrata = (fork, exact family path, distance bin): {len(bounds)-1}, of which {informative} contain both same- and different-allele pairs")
print(f"Mantel-Haenszel odds ratio (same allele -> same branch): {num/max(den,1e-9):.2f}")
print(f"stratum-weighted difference in P(same branch): {wd/max(wn,1e-9):+.1%}")
obs=wd/max(wn,1e-9)
null=[]
for _ in range(A.perm):
    p=SA.copy(); num2=den2=0.0
    for i in range(len(bounds)-1):
        s_,e_=bounds[i],bounds[i+1]
        if e_-s_>1: p[s_:e_]=rng.permutation(p[s_:e_])
        x=SB[s_:e_][p[s_:e_]==1]; y=SB[s_:e_][p[s_:e_]==0]
        if not len(x) or not len(y): continue
        w=len(x)*len(y)/(len(x)+len(y)); num2+=w*(x.mean()-y.mean()); den2+=w
    null.append(num2/max(den2,1e-9))
null=np.array(null); pv=(np.sum(null>=obs)+1)/(A.perm+1)
print(f"permutation of the allele label within each stratum, {A.perm} draws: observed {obs:+.4f}, "
      f"null {null.mean():+.5f} +/- {null.std():.5f}, p = {pv:.4f}")
inr=sb[(sa==1)&(rg==1)].mean()-sb[(sa==0)&(rg==1)].mean(); out=sb[(sa==1)&(rg==0)].mean()-sb[(sa==0)&(rg==0)].mean()
print(f"gap in a region of plasticity {inr:+.1%} (n={int(((rg==1)).sum())}) vs outside {out:+.1%} (n={int(((rg==0)).sum())})")
byc=collections.defaultdict(lambda:[[],[]])
for (f,_st,_b,s,y,r) in rec: byc[cls.get(f,"other")][s].append(y)
print("\nby functional class:")
for c,(no,yes) in sorted(byc.items(),key=lambda kv:-(np.mean(kv[1][1] or [0])-np.mean(kv[1][0] or [0]))):
    if len(yes)<30 or len(no)<30: continue
    print(f"  {c:32s} same allele {np.mean(yes):.1%}  different {np.mean(no):.1%}  gap {np.mean(yes)-np.mean(no):+.1%}  ({len(yes)+len(no)} pairs)")
json.dump(dict(L=L,pairs=len(R),strata=int(len(bounds)-1),informative_strata=int(informative),
               mh_odds_ratio=float(num/max(den,1e-9)),observed_gap=float(obs),perm_p=float(pv)),
          open("pgb/noLM.json","w"))
