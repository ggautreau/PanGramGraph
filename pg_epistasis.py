"""Distant regions that are coupled, and that shared descent does not explain.

pg_longrange.py showed the allele-branch association dies by 50-100 genes: it is linkage
inside a transfer tract. This asks the other question, on presence/absence of accessory
families rather than on fork branches, which gives far more power:

  are there pairs of accessory families, far apart on the chromosome and in different
  insertion spots, that co-occur (or exclude each other) across genomes more than their
  phylogeny allows?

  families    present in 10-90 % of the 540 complete chromosomes (accessory, variable)
  distance    median |position difference| > --mindist genes the short way round, and
              different panPPanGGOLiN insertion spot
  strata      tight lineage clusters: average linkage on cgMLST cut at --cut, clusters of
              >= 4 genomes. Two genomes in one cluster are close relatives, so an
              association WITHIN clusters is not shared descent.
  statistic   Cochran-Mantel-Haenszel over clusters, one genome = one observation (no
              overlapping pairs, unlike the first attempt, which inflated Z enormously)
  null        genome labels permuted within clusters, which keeps every marginal and the
              lineage structure; the threshold is read off this null
  replication the clusters are split in two disjoint sets; a real coupling holds in both

    python3 pg_epistasis.py [--cut 0.4] [--mindist 500] [--perm 30]
"""
import json, argparse, collections, warnings, numpy as np, torch
warnings.simplefilter("ignore")
ap=argparse.ArgumentParser()
ap.add_argument("--cut",type=float,default=0.4); ap.add_argument("--mindist",type=int,default=500)
ap.add_argument("--perm",type=int,default=30); ap.add_argument("--lo",type=float,default=0.10)
ap.add_argument("--hi",type=float,default=0.90)
ap.add_argument("--ctrl_content",action="store_true",help="stratify also by tertile of accessory content")
A=ap.parse_args()
dev="cuda:0"; rng=np.random.default_rng(0)
Z=np.load("pgb/chrom.npz"); off=Z["offsets"]; FAM=Z["fam"]; RGP=Z["rgp"]; SPOT=Z["spot"]
NG=len(off)-1; J=json.load(open("pgb/chrom_genomes.json")); FN=J["families"]; GL=np.diff(off)
name_of=np.concatenate([np.array(n,dtype=object) for n in J["gene_names"]])
D2=np.load("pgb/cgmlst_dist.npy").astype(np.float32); acc=json.load(open("pgb/cgmlst_loci.json"))["genomes"]
ix={a:i for i,a in enumerate(acc)}; sel=np.array([ix[g["acc"]] for g in J["genomes"]]); DM=D2[np.ix_(sel,sel)]
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
cl=fcluster(linkage(squareform(np.maximum(DM,DM.T),checks=False),"average"),A.cut,"distance")
size=collections.Counter(cl); CL=np.array([c for c,n in size.items() if n>=4]); keep=np.isin(cl,CL)
print(f"lineage clusters at cut {A.cut}: {len(CL)} with >= 4 genomes, covering {keep.sum()} genomes "
      f"(sizes {sorted([size[c] for c in CL],reverse=True)[:8]}...)",flush=True)

gid=np.repeat(np.arange(NG),GL); pos_in=np.arange(len(FAM))-off[gid]
nf=FAM.max()+1
P=np.zeros((nf,NG),bool); P[FAM,gid]=True
posacc=np.full((nf,NG),np.nan,np.float32)
key=gid.astype(np.int64)*nf+FAM; _,fi=np.unique(key,return_index=True)
posacc[FAM[fi],gid[fi]]=pos_in[fi]
freq=P[:,keep].mean(1)
cand=np.where((freq>=A.lo)&(freq<=A.hi))[0]
print(f"{len(cand)} accessory families present in {A.lo:.0%}-{A.hi:.0%} of the genomes",flush=True)
spot=collections.defaultdict(collections.Counter)
for f,s in zip(FAM.tolist(),SPOT.tolist()): spot[f][s]+=1
spot_maj=np.array([spot[int(f)].most_common(1)[0][0] for f in cand])
inrgp=np.array([RGP[FAM==f].mean() for f in cand])
medpos=np.nanmedian(posacc[cand],1)

Pc=torch.tensor(P[cand][:,keep],device=dev,dtype=torch.float32); clk=cl[keep]; NC=len(cand)
if A.ctrl_content:                    # a genome rich in accessory genes has more of everything
    content=P[cand][:,keep].sum(0)
    ter=np.zeros(len(content),int)
    for c in CL:
        g=np.where(clk==c)[0]; q=np.quantile(content[g],[1/3,2/3])
        ter[g]=np.searchsorted(q,content[g],"right")
    clk=np.array([f"{a_}|{b_}" for a_,b_ in zip(clk,ter)])
    CL=np.array([k for k,n in collections.Counter(clk).items() if n>=4])
    print(f"  after stratifying by accessory-content tertile: {len(CL)} strata, {int(np.isin(clk,CL).sum())} genomes",flush=True)
def cmh(Pm,perm=None):
    num=torch.zeros((NC,NC),device=dev); var=torch.zeros((NC,NC),device=dev)
    for c in CL:
        g=np.where(clk==c)[0]; n=len(g)
        X=Pm[:,g]; Y=Pm[:,g] if perm is None else Pm[:,g[perm[c]]]
        a=X@Y.T; r1=X.sum(1,keepdim=True); c1=Y.sum(1,keepdim=True).T
        num+=a-r1*c1/n
        var+=r1*(n-r1)*c1*(n-c1)/(n*n*(n-1))
    return num/var.clamp(min=1e-9).sqrt(), var
Zo,Vo=cmh(Pc); Zo=Zo.cpu().numpy(); Vo=Vo.cpu().numpy()
print("observed CMH computed",flush=True)
nulls=[]
for _ in range(A.perm):
    perm={c:rng.permutation(int((clk==c).sum())) for c in CL}
    Zp,_=cmh(Pc,perm); nulls.append(Zp.cpu().numpy()[np.triu_indices(NC,1)])
NU=np.concatenate(nulls); print(f"null: {len(NU)} statistics, |Z| 99.9th pct {np.quantile(np.abs(NU),.999):.2f}, max {np.abs(NU).max():.2f}",flush=True)

iu=np.triu_indices(NC,1); zz=Zo[iu]; inf=Vo[iu]>0.5
d=np.abs(medpos[iu[0]]-medpos[iu[1]]); d=np.minimum(d,np.median(GL)-d)
diff_spot=spot_maj[iu[0]]!=spot_maj[iu[1]]
far=inf&(d>A.mindist)&diff_spot
thr=np.quantile(np.abs(NU),.999); tail=(np.abs(NU)>thr).mean()
print(f"\n{'distance':26s}{'pairs':>10s}{'median |Z|':>12s}{'|Z|>thr':>10s}{'expected':>10s}{'ratio':>8s}")
for lab,m in (("same spot",inf&~diff_spot),("< 100 genes, other spot",inf&(d<100)&diff_spot),
              ("100-500 genes",inf&(d>=100)&(d<=A.mindist)&diff_spot),(f"> {A.mindist} genes, other spot",far)):
    if m.sum(): print(f"{lab:26s}{m.sum():>10d}{np.median(np.abs(zz[m])):>12.2f}{(np.abs(zz[m])>thr).mean():>10.3%}{tail:>10.3%}{(np.abs(zz[m])>thr).mean()/max(tail,1e-12):>8.1f}")

# replication on two disjoint sets of lineages
h=rng.permutation(len(CL)); CA=CL[h[:len(CL)//2]]; CB=CL[h[len(CL)//2:]]
def cmh_sub(cls_):
    num=torch.zeros((NC,NC),device=dev); var=torch.zeros((NC,NC),device=dev)
    for c in cls_:
        g=np.where(clk==c)[0]; n=len(g)
        X=Pc[:,g]; a=X@X.T; r1=X.sum(1,keepdim=True); c1=r1.T
        num+=a-r1*c1/n; var+=r1*(n-r1)*c1*(n-c1)/(n*n*(n-1))
    return (num/var.clamp(min=1e-9).sqrt()).cpu().numpy()
ZA=cmh_sub(CA)[iu]; ZB=cmh_sub(CB)[iu]
rep=far&(np.abs(zz)>thr)&(np.abs(ZA)>2)&(np.abs(ZB)>2)&(np.sign(ZA)==np.sign(ZB))&(np.sign(ZA)==np.sign(zz))
print(f"\nlong-range pairs beyond the null and replicated in both lineage sets: {rep.sum()} of {far.sum()} "
      f"(expected by chance {tail*far.sum():.1f} before the replication filter)")
nm=collections.defaultdict(collections.Counter)
for f,n_ in zip(FAM.tolist(),name_of.tolist()):
    if n_: nm[f][n_]+=1
gn=lambda i: (nm[int(cand[i])].most_common(1)[0][0] if nm[int(cand[i])] else FN[int(cand[i])])
out=[]
for k in np.argsort(-np.abs(zz)*rep)[:int(rep.sum())]:
    i,j=iu[0][k],iu[1][k]
    out.append(dict(a=gn(i),a_family=FN[int(cand[i])],a_freq=round(float(freq[cand[i]]),3),a_rgp=round(float(inrgp[i]),2),
                    b=gn(j),b_family=FN[int(cand[j])],b_freq=round(float(freq[cand[j]]),3),b_rgp=round(float(inrgp[j]),2),
                    distance=int(d[k]),z=round(float(zz[k]),2),zA=round(float(ZA[k]),2),zB=round(float(ZB[k]),2),
                    sign="co-occurrence" if zz[k]>0 else "avoidance"))
print(f"\n{'family A':20s}{'freq':>6s}{'RGP':>5s}  {'family B':20s}{'freq':>6s}{'RGP':>5s}{'apart':>8s}{'Z':>8s}{'A':>7s}{'B':>7s}  type")
for r in out[:40]:
    print(f"{r['a'][:19]:20s}{r['a_freq']:>6.2f}{r['a_rgp']:>5.1f}  {r['b'][:19]:20s}{r['b_freq']:>6.2f}{r['b_rgp']:>5.1f}"
          f"{r['distance']:>8d}{r['z']:>8.1f}{r['zA']:>7.1f}{r['zB']:>7.1f}  {r['sign']}")
json.dump(dict(cut=A.cut,mindist=A.mindist,ctrl_content=bool(A.ctrl_content),perm=A.perm,clusters=int(len(CL)),genomes=int(keep.sum()),families=int(NC),
               null_999=float(thr),far=int(far.sum()),replicated=int(rep.sum()),hits=out),open("pgb/epistasis.json","w"))
