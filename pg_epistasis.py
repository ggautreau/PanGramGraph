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

  --maxdist D  keep only the pairs with --mindist < distance <= D (other spot), e.g. the close range
               where the model has range:
    python3 pg_epistasis.py --ctrl_content --mindist 50 --maxdist 500 --out pgb/epistasis_close.json
               the observed statistics, the null, its threshold and the replication halves are the same
               as without it (same seed, same order of draws); only the selection of pairs changes, and
               the JSON gains `maxdist` and `bands` (per distance band: pairs, beyond the null, expected,
               replicated, and the permutation null of those same pairs for both counts), `bands_carriers`, and per
               hit where the two families sit in the genomes carrying both (carriers, near10/50/100/500 = share of
               them with the nearest copies within t genes, same_spot = share with a copy of each in one panRGP
               spot). Without it the output is exactly as before (checked: byte-identical pgb/epistasis.json).
  --dist carriers  with --maxdist: select by the median nearest-copy distance in the genomes carrying both
               (>= 10 of them) and require a different spot there too (the median-position rule for pairs
               with fewer co-carriers): a prophage whose genes sit at different spots in different genomes
               no longer passes as two coupled regions
    python3 pg_epistasis.py --ctrl_content --mindist 50 --maxdist 500 --dist carriers --out pgb/epistasis_close_carriers.json
  --out FILE   default pgb/epistasis.json
  --halves lineage  the replication halves are two disjoint sets of LINEAGES. With --ctrl_content the list that
               is halved by default holds the lineage x content strata, so the two halves share lineages (6 of the
               8 lineage clusters that have a stratum have strata on both sides, 172 of the 185 genomes): the
               default "replicated in two disjoint lineage halves" is not lineage-disjoint. With `lineage` every
               stratum of a lineage goes to the same half, the lineages split so as to balance the genomes (largest
               first, to the half with fewer genomes). The default (`strata`) is unchanged, for the existing outputs;
               the JSON then gains `halves`, the two lineage sets, and `replicated_strata_halves` (the count the
               default rule gives on the same run, to show the loss of power).
  --sep K      with --dist carriers: two SEPARATE insertions in the genomes carrying both families, not one
               element: in >= half of the co-carriers at least K backbone genes (families single-copy in >= 95 %
               of the genomes) lie between every copy of one and every copy of the other (the short way round),
               and each family is found without the other in >= 3 genomes. Pairs with < 10 co-carriers keep the
               median-position rule (their geometry cannot be read) and are marked. Hits gain `sep` (share of
               co-carriers with >= K backbone genes between) and `alone_a`, `alone_b`.
    python3 pg_epistasis.py --ctrl_content --mindist 50 --maxdist 500 --dist carriers --halves lineage --sep 10 \
            --out pgb/epistasis_close_sep.json
"""
import json, argparse, collections, warnings, numpy as np, torch
warnings.simplefilter("ignore")
ap=argparse.ArgumentParser()
ap.add_argument("--cut",type=float,default=0.4); ap.add_argument("--mindist",type=int,default=500)
ap.add_argument("--perm",type=int,default=30); ap.add_argument("--lo",type=float,default=0.10)
ap.add_argument("--hi",type=float,default=0.90)
ap.add_argument("--ctrl_content",action="store_true",help="stratify also by tertile of accessory content")
ap.add_argument("--maxdist",type=int,default=None,help="keep pairs with mindist < distance <= maxdist (default: no upper bound)")
ap.add_argument("--out",default="pgb/epistasis.json")
ap.add_argument("--dist",choices=("median","carriers"),default="median",
                help="with --maxdist: 'carriers' measures the distance and the spot in the genomes carrying both families")
ap.add_argument("--halves",choices=("strata","lineage"),default="strata",
                help="replication halves: 'lineage' puts every stratum of a lineage on one side (default: halves of the strata list, as before)")
ap.add_argument("--sep",type=int,default=None,help="with --dist carriers: >= SEP backbone genes between the two in most co-carriers")
A=ap.parse_args()
if A.sep is not None and A.dist!="carriers": ap.error("--sep needs --maxdist and --dist carriers")
dev="cuda:0"; rng=np.random.default_rng(0)
torch.cuda.set_per_process_memory_fraction(0.35,0)     # share the card with serve_live.py and another run
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
nulls=[]; perms=[]
for _ in range(A.perm):
    perm={c:rng.permutation(int((clk==c).sum())) for c in CL}; perms.append(perm)
    Zp,_=cmh(Pc,perm); nulls.append(Zp.cpu().numpy()[np.triu_indices(NC,1)])
NU=np.concatenate(nulls); print(f"null: {len(NU)} statistics, |Z| 99.9th pct {np.quantile(np.abs(NU),.999):.2f}, max {np.abs(NU).max():.2f}",flush=True)

iu=np.triu_indices(NC,1); zz=Zo[iu]; inf=Vo[iu]>0.5
d=np.abs(medpos[iu[0]]-medpos[iu[1]]); d=np.minimum(d,np.median(GL)-d)
diff_spot=spot_maj[iu[0]]!=spot_maj[iu[1]]
far=inf&(d>A.mindist)&diff_spot
if A.maxdist is not None: far&=(d<=A.maxdist)
if A.maxdist is not None:
    # close range: the median positions over all genomes and the majority spot say little about a mobile element
    # that sits at different spots in different genomes. Where do the two families sit in the genomes carrying
    # BOTH (every copy, all chromosomes)? nearest-copy distance (shares within t genes), and whether a copy of
    # each sits in one panRGP spot there (then they are one element, or parts of one, not two regions).
    fidx=-np.ones(nf,np.int64); fidx[cand]=np.arange(NC); TH=sorted({10,50,100,500,A.mindist,A.maxdist})
    CNT={t:torch.zeros(NC*NC,device=dev) for t in TH}; SAME=torch.zeros(NC*NC,device=dev); NB=torch.zeros(NC*NC,device=dev)
    if A.sep is not None:             # backbone = single-copy in >= 95 % of the genomes (as pg_knock.Data.BB)
        uk_,cnt_=np.unique(gid.astype(np.int64)*nf+FAM,return_counts=True)
        BBf=np.bincount((uk_%nf)[cnt_==1],minlength=nf)>=0.95*NG; SEPN=torch.zeros(NC*NC,device=dev)
    for g_ in range(NG):
        a_,b_=int(off[g_]),int(off[g_+1]); Lg=b_-a_; ci=fidx[FAM[a_:b_]]; m_=ci>=0
        pt=torch.tensor(np.arange(Lg)[m_],device=dev,dtype=torch.float32); ct=torch.tensor(ci[m_],device=dev)
        st=torch.tensor(SPOT[a_:b_][m_],device=dev)
        dd=(pt[:,None]-pt[None]).abs(); dd=torch.minimum(dd,Lg-dd); flat=(ct[:,None]*NC+ct[None]).reshape(-1)
        M=torch.full((NC*NC,),1e9,device=dev); M.scatter_reduce_(0,flat,dd.reshape(-1),reduce="amin")
        S=torch.zeros(NC*NC,device=dev)
        S.scatter_reduce_(0,flat,((st[:,None]==st[None])&(st[:,None]>=0)&(dd<200)).float().reshape(-1),reduce="amax")
        pr=torch.zeros(NC,device=dev); pr[ct]=1; both=(pr[:,None]*pr[None]).reshape(-1)
        NB+=both; SAME+=S*both
        for t in TH: CNT[t]+=((M<=t)&(both>0)).float()
        if A.sep is not None:         # backbone genes strictly between two copies, the short way round; min over copy pairs
            bbg=BBf[FAM[a_:b_]].astype(np.float32); Cg=np.cumsum(bbg); tot=float(Cg[-1])
            Ci=torch.tensor(Cg[m_],device=dev); bi=torch.tensor(bbg[m_],device=dev)
            later=pt[:,None]>=pt[None]                                    # i after j
            Chi=torch.where(later,Ci[:,None],Ci[None]); bhi=torch.where(later,bi[:,None],bi[None])
            Clo=torch.where(later,Ci[None],Ci[:,None]); blo=torch.where(later,bi[None],bi[:,None])
            lin=Chi-bhi-Clo; wrap=tot-Chi+Clo-blo
            span=(pt[:,None]-pt[None]).abs(); bbn=torch.where(span<=Lg-span,lin,wrap).clamp(min=0)
            Bm=torch.full((NC*NC,),1e9,device=dev); Bm.scatter_reduce_(0,flat,bbn.reshape(-1),reduce="amin")
            SEPN+=((Bm>=A.sep)&(both>0)).float()
    ex=lambda T: T.view(NC,NC).cpu().numpy()[iu]
    nbv=ex(NB); shr={t:ex(CNT[t])/np.maximum(nbv,1) for t in TH}; same=ex(SAME)/np.maximum(nbv,1); enough=nbv>=10
    del CNT,SAME,NB
    if A.sep is not None:
        sep=ex(SEPN)/np.maximum(nbv,1); del SEPN
        Pa=torch.tensor(P[cand],device=dev,dtype=torch.float32); AL=(Pa@(1-Pa).T).cpu().numpy(); del Pa
        alone_a=AL[iu]; alone_b=AL.T[iu]; del AL                      # genomes (of 540) with one family and not the other
    cin=lambda lo,hi: enough&((shr[lo]<.5) if lo else True)&(shr[hi]>=.5)&(same<.5)   # median nearest-copy distance in (lo,hi], other spot
    print(f"carrier geometry: {int((inf&enough).sum())} informative pairs with >= 10 genomes carrying both",flush=True)
    if A.dist=="carriers":            # >= 10 co-carriers: distance and spot read in them; fewer: the median-position rule
        far=inf&diff_spot&((enough&cin(A.mindist,A.maxdist))|(~enough&(d>A.mindist)&(d<=A.maxdist)))
        if A.sep is not None:         # two separate insertions where they co-occur, each also found alone
            far0=far.copy()
            far&=((~enough)|(sep>=.5))&(np.minimum(alone_a,alone_b)>=3)
            print(f"--sep {A.sep}: {int(far0.sum())} selected pairs -> {int(far.sum())} (dropped: {int((far0&enough&(sep<.5)).sum())} "
                  f"with < {A.sep} backbone genes between them in most co-carriers, {int((far0&(np.minimum(alone_a,alone_b)<3)).sum())} "
                  f"with a family never found without the other in >= 3 genomes)",flush=True)
thr=np.quantile(np.abs(NU),.999); tail=(np.abs(NU)>thr).mean()
print(f"\n{'distance':26s}{'pairs':>10s}{'median |Z|':>12s}{'|Z|>thr':>10s}{'expected':>10s}{'ratio':>8s}")
if A.maxdist is None:
    BANDS=(("same spot",inf&~diff_spot),("< 100 genes, other spot",inf&(d<100)&diff_spot),
           ("100-500 genes",inf&(d>=100)&(d<=A.mindist)&diff_spot),(f"> {A.mindist} genes, other spot",far))
else:                                 # the selected range, cut at 100 genes when it spans it
    cuts=[A.mindist]+[c for c in (100,) if A.mindist<c<A.maxdist]+[A.maxdist]
    SEL=[(f"{lo_}-{hi_} genes, other spot",inf&diff_spot&((d>lo_) if k==0 else (d>=lo_))&((d<hi_) if k<len(cuts)-2 else (d<=hi_)))
         for k,(lo_,hi_) in enumerate(zip(cuts[:-1],cuts[1:]))]
    BANDS=(("same spot",inf&~diff_spot),(f"<= {A.mindist} genes, other spot",inf&(d<=A.mindist)&diff_spot),*SEL,
           (f"selected {A.mindist}-{A.maxdist}",far),(f"> {A.maxdist} genes, other spot",inf&(d>A.maxdist)&diff_spot))
for lab,m in BANDS:
    if m.sum(): print(f"{lab:26s}{m.sum():>10d}{np.median(np.abs(zz[m])):>12.2f}{(np.abs(zz[m])>thr).mean():>10.3%}{tail:>10.3%}{(np.abs(zz[m])>thr).mean()/max(tail,1e-12):>8.1f}")

# replication on two disjoint sets of lineages
h=rng.permutation(len(CL)); CA=CL[h[:len(CL)//2]]; CB=CL[h[len(CL)//2:]]
if A.halves=="lineage":               # every stratum of a lineage on one side, genomes balanced (largest lineage first)
    lin_of=lambda s_: str(s_).split("|")[0]
    nlin=collections.Counter()
    for c in CL: nlin[lin_of(c)]+=int((clk==c).sum())
    side={}; tot_=[0,0]
    for l_,n_ in sorted(nlin.items(),key=lambda t:(-t[1],t[0])):
        k_=0 if tot_[0]<=tot_[1] else 1; side[l_]=k_; tot_[k_]+=n_
    CA_s,CB_s=CA,CB
    CA=np.array([c for c in CL if side[lin_of(c)]==0]); CB=np.array([c for c in CL if side[lin_of(c)]==1])
    LIN_A=sorted({lin_of(c) for c in CA}); LIN_B=sorted({lin_of(c) for c in CB})
    assert not set(LIN_A)&set(LIN_B)
    print(f"lineage halves: A {LIN_A} ({tot_[0]} genomes, {len(CA)} strata), B {LIN_B} ({tot_[1]} genomes, {len(CB)} strata)",flush=True)
def cmh_sub(cls_):
    num=torch.zeros((NC,NC),device=dev); var=torch.zeros((NC,NC),device=dev)
    for c in cls_:
        g=np.where(clk==c)[0]; n=len(g)
        X=Pc[:,g]; a=X@X.T; r1=X.sum(1,keepdim=True); c1=r1.T
        num+=a-r1*c1/n; var+=r1*(n-r1)*c1*(n-c1)/(n*n*(n-1))
    return (num/var.clamp(min=1e-9).sqrt()).cpu().numpy()
ZA=cmh_sub(CA)[iu]; ZB=cmh_sub(CB)[iu]
rep=far&(np.abs(zz)>thr)&(np.abs(ZA)>2)&(np.abs(ZB)>2)&(np.sign(ZA)==np.sign(ZB))&(np.sign(ZA)==np.sign(zz))
if A.halves=="lineage":               # what the default (strata) halves give on the same pairs: the loss of power
    ZAs=cmh_sub(CA_s)[iu]; ZBs=cmh_sub(CB_s)[iu]
    rep_s=far&(np.abs(zz)>thr)&(np.abs(ZAs)>2)&(np.abs(ZBs)>2)&(np.sign(ZAs)==np.sign(ZBs))&(np.sign(ZAs)==np.sign(zz))
    print(f"replicated with lineage-disjoint halves: {int(rep.sum())}; with the default strata halves (not lineage-disjoint): "
          f"{int(rep_s.sum())}; both: {int((rep&rep_s).sum())}",flush=True)
print(f"\n{'long-range' if A.maxdist is None else f'{A.mindist}-{A.maxdist} genes'} pairs beyond the null and replicated in both lineage sets: {rep.sum()} of {far.sum()} "
      f"(expected by chance {tail*far.sum():.1f} before the replication filter)")
if A.maxdist is not None:
    # the permutation null of THESE pairs: the global 99.9th percentile is exceeded by 0.1 % of all pairs, but mostly
    # by the uninformative ones (variance <= 0.5, discarded above); the same perms also give the null of the
    # replication filter (each half read under the same permutation)
    def cmh_perm(cls_,perm):
        num=torch.zeros((NC,NC),device=dev); var=torch.zeros((NC,NC),device=dev)
        for c in cls_:
            g=np.where(clk==c)[0]; n=len(g); X=Pc[:,g]; Y=Pc[:,g[perm[c]]]
            a=X@Y.T; r1=X.sum(1,keepdim=True); c1=Y.sum(1,keepdim=True).T
            num+=a-r1*c1/n; var+=r1*(n-r1)*c1*(n-c1)/(n*n*(n-1))
        return (num/var.clamp(min=1e-9).sqrt()).cpu().numpy()[iu]
    BEYN=np.zeros(len(zz),np.int32); REPN=np.zeros(len(zz),np.int32)
    for k_,perm in enumerate(perms):
        Zp=nulls[k_]; ZAp=cmh_perm(CA,perm); ZBp=cmh_perm(CB,perm); bz=np.abs(Zp)>thr; BEYN+=bz
        REPN+=bz&(np.abs(ZAp)>2)&(np.abs(ZBp)>2)&(np.sign(ZAp)==np.sign(ZBp))&(np.sign(ZAp)==np.sign(Zp))
    rep_any=(np.abs(zz)>thr)&(np.abs(ZA)>2)&(np.abs(ZB)>2)&(np.sign(ZA)==np.sign(ZB))&(np.sign(ZA)==np.sign(zz))
    print(f"  null exceedance of the threshold: all pairs {BEYN.sum()/A.perm/len(zz):.4%}, informative pairs {BEYN[inf].sum()/A.perm/max(inf.sum(),1):.4%}")
    def band_row(lab,m):
        b_=dict(band=lab,pairs=int(m.sum()),beyond_null=int((np.abs(zz[m])>thr).sum()),expected=round(float(tail*m.sum()),1),
                null_beyond=round(float(BEYN[m].sum()/A.perm),1),replicated=int((rep_any&m).sum()),null_replicated=round(float(REPN[m].sum()/A.perm),2),
                co=int((rep_any&m&(zz>0)).sum()),avoid=int((rep_any&m&(zz<0)).sum()))
        b_["ratio"]=round(b_["beyond_null"]/max(b_["expected"],1e-12),1); b_["ratio_perm"]=round(b_["beyond_null"]/max(b_["null_beyond"],0.1),1)
        print(f"  {lab:34s} {b_['pairs']:>9d} pairs, {b_['beyond_null']:>6d} beyond the null (expected {b_['expected']} at the global tail, "
              f"{b_['null_beyond']} under the permutation null of these pairs), replicated {b_['replicated']} (null {b_['null_replicated']}; "
              f"{b_['co']} co-occurrence, {b_['avoid']} avoidance)")
        return b_
    print("\nselected bands (median positions, majority spots):" if A.dist=="median" else "\nmedian-position bands, for reference:")
    bands=[band_row(lab.split(" genes")[0],m) for lab,m in SEL]
    if A.dist=="carriers":
        print("selected (carrier distance and spot; median-position rule for < 10 co-carriers):")
        bands=[band_row(f"{lo_}-{hi_}",far&(((enough&cin(lo_,hi_))|(~enough&(d>lo_)&(d<=hi_)))))
               for lo_,hi_ in zip(cuts[:-1],cuts[1:])]
    print("in the genomes carrying both (>= 10 of them; different majority spot):")
    bands_carriers=[band_row(f"carriers {lo_}-{hi_}, other spot",inf&diff_spot&cin(lo_,hi_))
                    for lo_,hi_ in ((0,10),(10,50),(50,100),(100,500)) if lo_ in shr or lo_==0]
    bands_carriers.append(band_row("carriers > 500, other spot",inf&diff_spot&enough&(shr[500]<.5)&(same<.5)))
    bands_carriers.append(band_row("carriers: one spot",inf&diff_spot&enough&(same>=.5)))
    bands_carriers.append(band_row("fewer than 10 co-carriers",inf&diff_spot&~enough))
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
    if A.maxdist is not None:         # where the two sit in the genomes carrying both
        out[-1].update(carriers=int(nbv[k]),near10=round(float(shr[10][k]),2),near50=round(float(shr[50][k]),2),
                       near100=round(float(shr[100][k]),2),near500=round(float(shr[500][k]),2),same_spot=round(float(same[k]),2))
        if A.sep is not None:
            out[-1].update(sep=round(float(sep[k]),2),alone_a=int(alone_a[k]),alone_b=int(alone_b[k]),geometry="carriers" if enough[k] else "median")
print(f"\n{'family A':20s}{'freq':>6s}{'RGP':>5s}  {'family B':20s}{'freq':>6s}{'RGP':>5s}{'apart':>8s}{'Z':>8s}{'A':>7s}{'B':>7s}  type")
for r in out[:40]:
    print(f"{r['a'][:19]:20s}{r['a_freq']:>6.2f}{r['a_rgp']:>5.1f}  {r['b'][:19]:20s}{r['b_freq']:>6.2f}{r['b_rgp']:>5.1f}"
          f"{r['distance']:>8d}{r['z']:>8.1f}{r['zA']:>7.1f}{r['zB']:>7.1f}  {r['sign']}")
res=dict(cut=A.cut,mindist=A.mindist,ctrl_content=bool(A.ctrl_content),perm=A.perm,clusters=int(len(CL)),genomes=int(keep.sum()),families=int(NC),
         null_999=float(thr),far=int(far.sum()),replicated=int(rep.sum()),hits=out)
if A.maxdist is not None:             # keys added only with --maxdist: the default output is unchanged
    res=dict(res,maxdist=A.maxdist,dist=A.dist,tail=float(tail),null_exceed_informative=float(BEYN[inf].sum()/A.perm/max(inf.sum(),1)),
             bands=bands,bands_carriers=bands_carriers); res["hits"]=res.pop("hits")
if A.halves=="lineage":               # keys added only with --halves lineage
    res=dict(res,halves="lineage",lineages_A=LIN_A,lineages_B=LIN_B,replicated_strata_halves=int(rep_s.sum()),
             replicated_both_rules=int((rep&rep_s).sum())); res["hits"]=res.pop("hits")
if A.sep is not None:
    res=dict(res,sep=A.sep,far_before_sep=int(far0.sum())); res["hits"]=res.pop("hits")
json.dump(res,open(A.out,"w"))
print(f"-> {A.out}")
