"""Long-range contingency between two regions of plasticity, with lineage controlled.

pg_longrange.py showed that the allele-branch association decays to nothing by 50-100
genes, the length of E. coli recombination tracts: the local contingency is linkage. The
open question is whether two DISTANT variable loci are coupled beyond what shared descent
explains. That is the co-occurrence question of Beavan et al. 2023 and Coinfinder, asked
here between the branches taken at two forks of regions of plasticity.

  RGP fork     a fork whose site is in a panRGP region of plasticity in >= 50 % of genomes
  y            per genome, 1 if it takes the fork's majority branch, 0 otherwise
  strata       tight lineages: average-linkage clusters of the cgMLST distance cut at
               --cut; only clusters of >= 3 genomes are used. Two genomes of one cluster
               are close relatives, so an association INSIDE clusters is not lineage.
  statistic    Cochran-Mantel-Haenszel over clusters, genome level (no overlapping pairs),
               signed Z; calibrated by permuting genome labels within clusters
  replication  clusters split at random into two halves; a real coupling holds in both

    python3 pg_rgp_pairs.py [--cut 0.5] [--perm 200]
"""
import json, argparse, collections, warnings, numpy as np
warnings.simplefilter("ignore")
ap=argparse.ArgumentParser(); ap.add_argument("--cut",type=float,default=0.5); ap.add_argument("--perm",type=int,default=200)
ap.add_argument("--minfork",type=int,default=150); A=ap.parse_args(); rng=np.random.default_rng(0)
Z=np.load("pgb/chrom.npz"); off=Z["offsets"]; FAM=Z["fam"]; PID=Z["pid"]; RGP=Z["rgp"]; SPOT=Z["spot"]
NG=len(off)-1; J=json.load(open("pgb/chrom_genomes.json")); FN=J["families"]; GL=np.diff(off)
name_of=np.concatenate([np.array(n,dtype=object) for n in J["gene_names"]])
D2=np.load("pgb/cgmlst_dist.npy").astype(np.float32); acc=json.load(open("pgb/cgmlst_loci.json"))["genomes"]
ix={a:i for i,a in enumerate(acc)}; sel=np.array([ix[g["acc"]] for g in J["genomes"]]); DM=D2[np.ix_(sel,sel)]
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
cl=fcluster(linkage(squareform(np.maximum(DM,DM.T),checks=False),"average"),A.cut,"distance")
size=collections.Counter(cl); keep=np.array([size[c]>=3 for c in cl]); CL=np.unique(cl[keep])
print(f"lineage clusters at cut {A.cut}: {len(size)} clusters, {len(CL)} with >= 3 genomes, "
      f"covering {keep.sum()} genomes (largest {max(size.values())})",flush=True)

# forks, first copy of each family per genome
gid=np.repeat(np.arange(NG),GL); key=gid.astype(np.int64)*(FAM.max()+1)+FAM; _,fi=np.unique(key,return_index=True)
last=np.zeros(len(FAM),bool); last[off[1:]-1]=True; nxt=np.full(len(FAM),-1,np.int64); nxt[~last]=FAM[1:][~last[:-1]]
succ=collections.defaultdict(collections.Counter); site=collections.defaultdict(dict)
for p in fi:
    if nxt[p]>=0: succ[int(FAM[p])][int(nxt[p])]+=1; site[int(FAM[p])][int(gid[p])]=int(p)
forks=[]
for F,c in succ.items():
    if sum(c.values())<A.minfork: continue
    br=[b for b,n in c.most_common(8) if n>=0.15*sum(c.values())]
    if len(br)<2: continue
    ps=np.array(list(site[F].values())); inr=RGP[np.minimum(ps+1,len(RGP)-1)].mean()
    if inr<0.5: continue
    forks.append((F,br[0],br))
NF=len(forks); Y=np.full((NF,NG),np.nan); POS=np.full((NF,NG),np.nan); SP=np.zeros(NF,np.int64)
for k,(F,maj,br) in enumerate(forks):
    for g,p in site[F].items():
        if nxt[p] in br: Y[k,g]=float(nxt[p]==maj); POS[k,g]=p-off[g]
    sp=collections.Counter(SPOT[list(site[F].values())].tolist()).most_common(1)[0][0]; SP[k]=sp
print(f"{NF} forks inside regions of plasticity",flush=True)

def cmh(Y1,Y2,clusters,genomes_mask):
    """signed CMH Z between every row of Y1 and every row of Y2, strata = clusters"""
    num=np.zeros((Y1.shape[0],Y2.shape[0])); var=np.zeros_like(num)
    for c in clusters:
        g=np.where((cl==c)&genomes_mask)[0]
        if len(g)<3: continue
        P1=~np.isnan(Y1[:,g]); P2=~np.isnan(Y2[:,g]); V1=np.nan_to_num(Y1[:,g]); V2=np.nan_to_num(Y2[:,g])
        P1f=P1.astype(float); P2f=P2.astype(float)
        n=P1f@P2f.T; a=V1@V2.T; r1=V1@P2f.T; c1=P1f@V2.T
        ok=n>1
        E=np.where(ok,r1*c1/np.maximum(n,1),0); Vv=np.where(ok,r1*(n-r1)*c1*(n-c1)/np.maximum(n*n*(n-1),1),0)
        num+=np.where(ok,a-E,0); var+=Vv
    return np.where(var>0,num/np.sqrt(np.maximum(var,1e-12)),0.0),var
Zo,Vo=cmh(Y,Y,CL,keep)
# distance between forks, genes, short way round, median over genomes
dist=np.full((NF,NF),np.nan)
for i in range(NF):
    d=np.abs(POS[i][None]-POS); d=np.minimum(d,GL[None]-d); dist[i]=np.nanmedian(d,1)
iu=np.triu_indices(NF,1); zz=Zo[iu]; dd=dist[iu]; same_spot=(SP[iu[0]]==SP[iu[1]]); inf=Vo[iu]>0.5
# null: permute genome labels within clusters on one side
null=[]
for _ in range(A.perm if A.perm<=50 else 50):
    perm=np.arange(NG)
    for c in CL:
        g=np.where((cl==c)&keep)[0]; perm[g]=rng.permutation(g)
    Zp,_=cmh(Y,Y[:,perm],CL,keep); null.append(Zp[iu][inf])
null=np.concatenate(null)
print(f"\ninformative fork pairs (enough variation inside lineages): {inf.sum()} of {len(zz)}")
print(f"null (within-lineage permutation): |Z| 99th pct {np.quantile(np.abs(null),.99):.2f}, 99.9th {np.quantile(np.abs(null),.999):.2f}, "
      f"max {np.abs(null).max():.2f}")
print(f"\n{'distance between forks':28s}{'pairs':>8s}{'median |Z|':>12s}{'|Z|>4':>9s}{'null |Z|>4':>12s}")
thr4=(np.abs(null)>4).mean()
for lab,m in (("same spot",same_spot),("< 50 genes, other spot",(dd<50)&~same_spot),("50-500 genes",(dd>=50)&(dd<500)&~same_spot),
              ("> 500 genes, other spot",(dd>=500)&~same_spot)):
    m=m&inf
    if m.sum(): print(f"{lab:28s}{m.sum():>8d}{np.median(np.abs(zz[m])):>12.2f}{(np.abs(zz[m])>4).mean():>9.2%}{thr4:>12.3%}")
# replication in two random halves of the lineage clusters
half=rng.permutation(CL)[:len(CL)//2]; hA=np.isin(cl,half)
ZA,_=cmh(Y,Y,CL,keep&hA); ZB,_=cmh(Y,Y,CL,keep&~hA)
za=ZA[iu]; zb=ZB[iu]
far=(dd>=500)&~same_spot&inf
t=np.quantile(np.abs(null),.999)
rep=far&(np.abs(za)>3)&(np.abs(zb)>3)&(np.sign(za)==np.sign(zb))&(np.abs(zz)>t)
nm=collections.defaultdict(collections.Counter)
for f,n_ in zip(FAM.tolist(),name_of.tolist()):
    if n_: nm[f][n_]+=1
gn=lambda f: nm[int(f)].most_common(1)[0][0] if nm[int(f)] else FN[int(f)]
print(f"\nlong-range pairs of RGP forks (> 500 genes, other spot) beyond the null 99.9th pct (|Z| > {t:.2f}) "
      f"AND same-sign |Z| > 3 in both halves of the lineages: {rep.sum()} of {far.sum()}")
out=[]
for k in np.argsort(-np.abs(zz)*rep)[:rep.sum()]:
    i,j=iu[0][k],iu[1][k]; (F1,m1,b1),(F2,m2,b2)=forks[i],forks[j]
    out.append(dict(fork1=gn(F1),branch1=gn(m1),alt1=[gn(x) for x in b1[1:3]],fork2=gn(F2),branch2=gn(m2),alt2=[gn(x) for x in b2[1:3]],
                    distance=int(dd[k]),z=round(float(zz[k]),2),zA=round(float(za[k]),2),zB=round(float(zb[k]),2)))
for r in out[:25]:
    print(f"  {r['fork1'][:14]:15s}-> {r['branch1'][:12]:13s} vs {r['fork2'][:14]:15s}-> {r['branch2'][:12]:13s} {r['distance']:>6d} genes  Z={r['z']:+.1f} (A {r['zA']:+.1f}, B {r['zB']:+.1f})")
tail=far&(np.abs(zz)>4)
print(f"\nall long-range pairs with |Z| > 4: {tail.sum()} (expected under the null {thr4*far.sum():.1f}); same sign in both halves: {(tail&(np.sign(za)==np.sign(zb))).sum()}")
for k in np.argsort(-np.abs(zz)*tail)[:tail.sum()]:
    i,j=iu[0][k],iu[1][k]; (F1,m1,b1),(F2,m2,b2)=forks[i],forks[j]
    print(f"  {gn(F1)[:14]:15s}-> {gn(m1)[:12]:13s} vs {gn(F2)[:14]:15s}-> {gn(m2)[:12]:13s} {int(dd[k]):>6d} genes  Z={zz[k]:+.1f} (A {za[k]:+.1f}, B {zb[k]:+.1f})")
json.dump(dict(cut=A.cut,clusters=int(len(CL)),genomes=int(keep.sum()),forks=NF,informative=int(inf.sum()),
               null_999=float(t),hits=out),open("pgb/rgp_pairs.json","w"))
