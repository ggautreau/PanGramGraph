"""Is the allelic contingency just linkage disequilibrium inside one transfer tract?

The local result (DISCOVERY.md) reads the allele of the gene ADJACENT to a fork, so it is
compatible with allele and branch having arrived together in a single recombinant
transfer. In E. coli such tracts run 40-115 kb in the least-diverged genome pairs (Dixit
et al. 2015, PNAS), with hot regions above 100 kb. An association across a distance well
beyond that, between two different insertion spots, cannot be one transfer.

No language model here. For every fork F (a family followed by >= 2 families, each in
>= 15 % of the >= 150 genomes carrying it) and every locus D (a family in >= 85 % of
genomes with >= 5 distinct alleles), over all pairs of genomes:

  same_allele   the two genomes carry the identical protein of D
  same_branch   they go on to the same family after F
  stratum       cgMLST distance, 20 quantile bins: pairs are compared only with pairs
                of the same relatedness, so shared descent cannot produce the signal
  Z             signed Cochran-Mantel-Haenszel over strata (inflated by overlapping pairs:
                read it as a ranking, calibrated by the replication below)
  replication   Z computed separately in two DISJOINT sets of lineages (the largest clade
                vs all others, cgMLST average linkage); a real association holds in both

Then the question itself: how does the association decay with the distance between D
and F? Linkage inside transfer tracts predicts a signal that vanishes beyond ~100 genes.

    python3 pg_longrange.py      # -> pgb/longrange.json
"""
import json, collections, warnings, numpy as np, torch
warnings.simplefilter("ignore")
dev="cuda:0"; NB=20
Z=np.load("pgb/chrom.npz"); off=Z["offsets"]; FAM=Z["fam"]; PID=Z["pid"]; SPOT=Z["spot"]
NG=len(off)-1; J=json.load(open("pgb/chrom_genomes.json")); FN=J["families"]; GL=np.diff(off)
name_of=np.concatenate([np.array(n,dtype=object) for n in J["gene_names"]])
D2=np.load("pgb/cgmlst_dist.npy").astype(np.float32); acc=json.load(open("pgb/cgmlst_loci.json"))["genomes"]
ix={a:i for i,a in enumerate(acc)}; sel=np.array([ix[g["acc"]] for g in J["genomes"]]); DM=D2[np.ix_(sel,sel)]
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
cl=fcluster(linkage(squareform(np.maximum(DM,DM.T),checks=False),"average"),12,"maxclust")
big=collections.Counter(cl).most_common(1)[0][0]; HALF=(cl!=big).astype(int)        # 0: largest clade, 1: the rest
print(f"lineage halves: {int((HALF==0).sum())} / {int((HALF==1).sum())} genomes",flush=True)

# allele and position of every family, first copy, vectorised
first=np.ones(len(FAM),bool); gid=np.repeat(np.arange(NG),GL)
key=gid.astype(np.int64)*(FAM.max()+1)+FAM; _,firstidx=np.unique(key,return_index=True)
fams,finv=np.unique(FAM[firstidx],return_inverse=True)
ALLE=np.full((len(fams),NG),-1,np.int64); POS=np.full((len(fams),NG),-1,np.int64)
ALLE[finv,gid[firstidx]]=PID[firstidx]; POS[finv,gid[firstidx]]=firstidx-off[gid[firstidx]]
row={int(f):i for i,f in enumerate(fams)}
pres=(ALLE>=0).sum(1); nall=np.array([len(np.unique(a[a>=0])) for a in ALLE])
cand=np.where((pres>=0.85*NG)&(nall>=5))[0]
print(f"{len(fams)} families; {len(cand)} candidate loci (>= 85 % of genomes, >= 5 alleles)",flush=True)
spot_maj={}
for f,s in zip(FAM.tolist(),SPOT.tolist()): spot_maj.setdefault(f,collections.Counter())[s]+=1
spot_maj={f:c.most_common(1)[0][0] for f,c in spot_maj.items()}

# forks and branch per genome
succ=collections.defaultdict(collections.Counter); nxt=np.full(len(FAM),-1,np.int64)
last=np.zeros(len(FAM),bool); last[off[1:]-1]=True; nxt[~last]=FAM[1:][~last[:-1]]
for f,n_ in zip(FAM[firstidx].tolist(),nxt[firstidx].tolist()):
    if n_>=0: succ[f][n_]+=1
forks=[(F,[b for b,c in cnt.most_common(8) if c>=0.15*sum(cnt.values())]) for F,cnt in succ.items() if sum(cnt.values())>=150]
forks=[(F,br) for F,br in forks if len(br)>=2]
BR=np.full((len(forks),NG),-1,np.int64); PF=np.full((len(forks),NG),-1,np.int64)
for k,(F,br) in enumerate(forks):
    r=row[F]; g=np.where(ALLE[r]>=0)[0]; p=off[g]+POS[r,g]; nx=nxt[p]
    m=np.isin(nx,br); BR[k,g[m]]=[br.index(x) for x in nx[m]]; PF[k,g]=POS[r,g]
print(f"{len(forks)} forks",flush=True)

# all pairs of genomes, strata by relatedness
PI,PJ=np.triu_indices(NG,1); pdist=DM[PI,PJ]
edges=np.quantile(pdist,np.linspace(0,1,NB+1)); bn=np.clip(np.searchsorted(edges,pdist,"right")-1,0,NB-1)
BIN=torch.zeros((len(PI),NB),device=dev,dtype=torch.float32); BIN[torch.arange(len(PI)),torch.tensor(bn)]=1
tPI=torch.tensor(PI,device=dev); tPJ=torch.tensor(PJ,device=dev)
halfpair={"all":np.ones(len(PI),bool),"A":(HALF[PI]==0)&(HALF[PJ]==0),"B":(HALF[PI]==1)&(HALF[PJ]==1)}

def zscores(pmask):
    """signed CMH Z for every (locus, fork), pairs restricted to pmask"""
    pm=torch.tensor(pmask,device=dev)
    tBR=torch.tensor(BR,device=dev)
    ok=(tBR[:,tPI]>=0)&(tBR[:,tPJ]>=0)&pm[None]                       # F x P
    sb=(tBR[:,tPI]==tBR[:,tPJ])&ok
    n=ok.float()@BIN; c1=sb.float()@BIN                # F x NB
    sbT=sb.float().T.contiguous(); okT=ok.float().T.contiguous()
    Zs=np.zeros((len(cand),len(forks)),np.float32)
    for i in range(0,len(cand),256):
        A_=torch.tensor(ALLE[cand[i:i+256]],device=dev)
        sa=((A_[:,tPI]==A_[:,tPJ])&(A_[:,tPI]>=0)).float()                # L x P
        num=torch.zeros((sa.shape[0],len(forks)),device=dev); var=torch.zeros_like(num)
        for b in range(NB):
            w=BIN[:,b]
            a11=(sa*w)@sbT; r1=(sa*w)@okT                 # L x F
            a11=a11.float(); r1=r1.float(); nn_=n[:,b][None]; cc=c1[:,b][None]
            good=nn_>1
            num+=torch.where(good,a11-r1*cc/nn_.clamp(min=1),0)
            var+=torch.where(good,r1*(nn_-r1)*cc*(nn_-cc)/(nn_**2*(nn_-1)).clamp(min=1),0)
        Zs[i:i+256]=(num/var.clamp(min=1e-9).sqrt()).cpu().numpy()
    return Zs
Zall=zscores(halfpair["all"]); ZA=zscores(halfpair["A"]); ZB=zscores(halfpair["B"])
print("Z computed",flush=True)

# distance between locus and fork (genes, short way round), median over genomes
dist=np.zeros((len(cand),len(forks)),np.float32)
Pc=torch.tensor(POS[cand],device=dev,dtype=torch.float32); tGL=torch.tensor(GL,device=dev,dtype=torch.float32)
for k in range(len(forks)):
    pf=torch.tensor(PF[k],device=dev,dtype=torch.float32)
    d=(Pc-pf[None]).abs(); d=torch.minimum(d,tGL[None]-d)
    d[(Pc<0)|(pf[None]<0)]=float("nan"); dist[:,k]=torch.nanmedian(d,1).values.cpu().numpy()
samefam=np.array([[fams[c]==F for F,_ in forks] for c in cand])
samespot=np.array([[spot_maj.get(int(fams[c]),-1)==spot_maj.get(F,-2)!=-1 for F,_ in forks] for c in cand])

# --- the decay: how strong, and how often replicated, by distance -----------------------
T=4.0
cls=[("the fork family itself",samefam),("1-10 genes",(~samefam)&(dist>=1)&(dist<=10)),
     ("10-50 genes",(dist>10)&(dist<=50)),("50-100 genes",(dist>50)&(dist<=100)),
     ("100-500 genes",(dist>100)&(dist<=500)),("> 500 genes, other spot",(dist>500)&~samespot)]
print(f"\nassociation between the allele of a locus and the branch at a fork, by distance (signed Z, strata = 20 relatedness bins)")
print(f"{'distance':26s}{'tests':>9s}{'median Z':>10s}{'Z>4 all':>10s}{'Z>4 in A':>10s}{'Z>4 in B':>10s}{'both':>8s}{'expected':>10s}{'ratio':>8s}")
decay=[]
for lab,m in cls:
    if not m.any(): continue
    za,zb,z=ZA[m],ZB[m],Zall[m]; pa=(za>T).mean(); pb=(zb>T).mean(); both=((za>T)&(zb>T)).mean()
    decay.append(dict(cls=lab,tests=int(m.sum()),median_z=float(np.median(z)),frac_z4=float((z>T).mean()),
                      rep_both=float(both),rep_expected=float(pa*pb)))
    print(f"{lab:26s}{m.sum():>9d}{np.median(z):>10.2f}{(z>T).mean():>10.2%}{pa:>10.2%}{pb:>10.2%}{both:>8.2%}{pa*pb:>10.3%}{both/max(pa*pb,1e-9):>8.1f}")

# --- long-range hits replicated in both lineage halves ---------------------------------
far=(dist>500)&~samespot
hit=far&(ZA>T)&(ZB>T)
nm=collections.defaultdict(collections.Counter)
for f,n_ in zip(FAM.tolist(),name_of.tolist()):
    if n_: nm[f][n_]+=1
gname=lambda f: nm[int(f)].most_common(1)[0][0] if nm[int(f)] else FN[int(f)]
li,fi=np.where(hit); order=np.argsort(-(np.minimum(ZA[hit],ZB[hit])))
print(f"\nlong-range (> 500 genes, other spot) associations replicated in both lineage halves (Z > {T} in each): "
      f"{hit.sum()} of {far.sum()} tests, {len(set(fi.tolist()))} forks, {len(set(li.tolist()))} loci")
print(f"{'fork':22s}{'branches':34s}{'distant locus':20s}{'genes apart':>12s}{'Z (A)':>8s}{'Z (B)':>8s}")
out=[]
for k in order:
    l,f=li[k],fi[k]; F,br=forks[f]
    rec=dict(fork=gname(F),fork_family=FN[F],branches=[gname(b) for b in br],locus=gname(fams[cand[l]]),
             locus_family=FN[int(fams[cand[l]])],distance=int(dist[l,f]),zA=round(float(ZA[l,f]),2),zB=round(float(ZB[l,f]),2),
             z_all=round(float(Zall[l,f]),2))
    out.append(rec)
for r in out[:30]:
    print(f"{r['fork'][:21]:22s}{' / '.join(x[:10] for x in r['branches'][:3])[:33]:34s}{r['locus'][:19]:20s}{r['distance']:>12d}{r['zA']:>8.1f}{r['zB']:>8.1f}")
json.dump(dict(decay=decay,threshold=T,hits=out,tests_far=int(far.sum())),open("pgb/longrange.json","w"))
