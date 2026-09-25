"""Allelic contingency at the variable loci of the E. coli pangenome.

At a balanced fork (a family followed by >= 2 families, each after >= 10 % of the
family's occurrences and >= 20 of them; 540 complete chromosomes), is the branch a genome
takes written in the ALLELES of the genes just upstream, beyond what the genome's
phylogeny says and beyond what the pangenome graph (one node per family) can read?

Predictors of the branch, all leave-one-genome-out, with smoothing fitted on one half of
the genomes and evaluated on the other (so no predictor is tuned on what it is scored on):
  frequency        branch frequencies at the fork
  graph            the longest family path upstream (20, 10, 5, 3 or 2 families) shared
                   with >= 3 other genomes, back-off n-gram on the pangenome graph's paths
  phylo            the 15 nearest genomes on core-genome alleles (pgb_cgmlst.py)
  allele@d         the 15 sites whose protein d genes upstream of the fork (d = 0 is the
                   fork family itself) is most similar (ESM-2 cosine): model-free
  model            Bacformer, real proteins, read through the decoder of the other half
  model-family     Bacformer when every protein is its family's most common one: the
                   order of families only, what the graph holds (pg_calls.py --modal)
Count predictors are smoothed as (counts + a*freq) / (K + a); model ones as (1-l)*q + l*freq.

    python3 pg_allelic.py            # random halves of the genomes -> pgb/allelic.json
    python3 pg_allelic.py --clade    # halves = groups of whole clades -> pgb/allelic_clade.json
"""
import sys, json, collections, warnings, numpy as np, torch
from scipy import sparse
warnings.simplefilter("ignore")
CLADE="--clade" in sys.argv; dev="cuda:0"; K=15; DMAX=12; rng=np.random.default_rng(0)
Z=np.load("pgb/chrom.npz"); off=Z["offsets"]; FAM=Z["fam"]; PID=Z["pid"]; RGP=Z["rgp"]; NG=len(off)-1; NP=off[-1]
J=json.load(open("pgb/chrom_genomes.json")); GEN=J["genomes"]; FN=J["families"]
NPROT=sum(1 for _ in open("pgb/chrom_prot.txt"))
EMB=torch.tensor(np.asarray(np.memmap("pgb/chrom_emb.f16",dtype=np.float16,mode="r",shape=(NPROT,480))),device=dev)
EMB=EMB/EMB.float().norm(dim=1,keepdim=True).half()
TF=np.memmap("pgb/calls_top15_f.i32",dtype=np.int32,mode="r",shape=(NP,15)); TP=np.memmap("pgb/calls_top15_p.f16",dtype=np.float16,mode="r",shape=(NP,15))
SITES=np.load("pgb/fork_sites.npy"); NS=len(SITES); srow=np.full(NP,-1,np.int64); srow[SITES]=np.arange(NS)
FF=np.memmap("pgb/calls_fork_f.i32",dtype=np.int32,mode="r",shape=(NS,64)); FP=np.memmap("pgb/calls_fork_p.f16",dtype=np.float16,mode="r",shape=(NS,64))
FFm=np.memmap("pgb/calls_fork_modal_f.i32",dtype=np.int32,mode="r",shape=(NS,64)); FPm=np.memmap("pgb/calls_fork_modal_p.f16",dtype=np.float16,mode="r",shape=(NS,64))
gen_of=np.repeat(np.arange(NG),np.diff(off)); pos_in=np.arange(NP)-off[gen_of]; first=np.zeros(NP,bool); first[off[:-1]]=True
D2=np.load("pgb/cgmlst_dist.npy").astype(np.float32); acc2=json.load(open("pgb/cgmlst_loci.json"))["genomes"]
ix={a:i for i,a in enumerate(acc2)}; D=D2[np.ix_([ix[g["acc"]] for g in GEN],[ix[g["acc"]] for g in GEN])]
if CLADE:
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    cl=fcluster(linkage(squareform(np.maximum(D,D.T),checks=False),"average"),12,"maxclust")
    GH=np.zeros(NG,int); load=[0,0]
    for c,nm in collections.Counter(cl).most_common():
        s_=int(load[1]<load[0]); GH[cl==c]=s_; load[s_]+=nm
    print("clade halves:",load,flush=True)
else: GH=np.arange(NG)%2
# decoders P(family | cluster), learned on each half
fams=np.unique(FAM); FI=np.full(FAM.max()+1,-1); FI[fams]=np.arange(len(fams))
def decoder(hh):
    m=np.where((GH[gen_of]!=hh)&~first)[0]
    A=sparse.csr_matrix((np.asarray(TP[m],np.float32).ravel(),(np.repeat(FI[FAM[m]],15),np.asarray(TF[m]).ravel())),shape=(len(fams),50001))
    return A.tocsr(),np.asarray(A.sum(0)).ravel()
DEC={hh:decoder(hh) for hh in (0,1)}; print("decoders built",flush=True)
def model_q(rows,hs,br,Fc,Fp):
    C=np.asarray(Fc[rows]).astype(np.int64); Pv=np.asarray(Fp[rows],np.float32); q=np.zeros((len(rows),len(br)))
    for hh in (0,1):
        m=np.where(hs==hh)[0]
        if not len(m): continue
        A,tt=DEC[hh]; rws=[A.getrow(FI[b]) for b in br]
        cl=np.array(sorted({int(c) for r in rws for c in r.indices}),dtype=np.int64)
        if not len(cl): continue
        pos={c:i for i,c in enumerate(cl.tolist())}; M=np.zeros((len(cl)+1,len(br)))
        for j,r in enumerate(rws):
            for c,v in zip(r.indices.tolist(),r.data.tolist()): M[pos[c],j]=v/max(tt[c],1e-12)
        ii=np.clip(np.searchsorted(cl,C[m]),0,len(cl)-1); ii=np.where(cl[ii]==C[m],ii,len(cl))
        q[m]=(Pv[m][:,:,None]*M[ii]).sum(1)
    s=q.sum(1,keepdims=True); return np.where(s>0,q/np.maximum(s,1e-12),1.0/len(br))
# balanced forks
succ=collections.defaultdict(collections.Counter); occ=collections.Counter()
for p in SITES: succ[int(FAM[p-1])][int(FAM[p])]+=1; occ[int(FAM[p-1])]+=1
by_fork=collections.defaultdict(list)
for p in SITES: by_fork[int(FAM[p-1])].append(int(p))
forks=[]
for F,ps in by_fork.items():
    br=[b for b,c in succ[F].most_common(8) if c>=max(0.10*occ[F],20)]
    if len(br)>=2: forks.append((F,br))
print(f"{len(forks)} balanced forks",flush=True)

REC=[]            # per fork: dict of arrays
for fi,(F,br) in enumerate(forks):
    ps=np.array([p for p in by_fork[F] if FAM[p] in br and pos_in[p]>DMAX])
    if len(ps)>700: ps=np.sort(rng.choice(ps,700,replace=False))
    if len(ps)<30: continue
    nb=len(br); t=np.array([br.index(FAM[p]) for p in ps]); gs=gen_of[ps]; hs=GH[gs]; n=len(ps)
    same=gs[:,None]==gs[None,:]
    cnt=np.bincount(t,minlength=nb).astype(float)
    own=np.stack([np.bincount(t[gs==g],minlength=nb) for g in gs]); freq=(cnt[None]-own+0.5); freq/=freq.sum(1,keepdims=True)
    def knn_counts(S):                              # S: similarity (higher = closer), same genome excluded
        S=S.copy(); S[same]=-np.inf; kk=min(K,n-1); nn=np.argpartition(-S,kk-1,axis=1)[:,:kk]
        return np.stack([np.bincount(t[r],minlength=nb) for r in nn]).astype(float)
    counts={"phylo":knn_counts(-D[np.ix_(gs,gs)])}
    for d in range(DMAX):
        E=EMB[torch.tensor(PID[ps-1-d],device=dev)]; S=(E@E.T).float().cpu().numpy()
        counts[f"allele@{d}"]=knn_counts(S)
    # graph: back-off over family paths
    g_cnt=np.zeros((n,nb))
    keys={L:[tuple(FAM[p-L:p].tolist()) for p in ps] for L in (20,10,5,3,2)}
    grp={L:collections.defaultdict(list) for L in keys}
    for L in keys:
        for i,k in enumerate(keys[L]): grp[L][k].append(i)
    for i in range(n):
        for L in (20,10,5,3,2):
            o=[j for j in grp[L][keys[L][i]] if gs[j]!=gs[i]]
            if len(o)>=3: g_cnt[i]=np.bincount(t[o],minlength=nb); break
    counts["graph"]=g_cnt
    rows=srow[ps]
    REC.append(dict(F=F,br=br,t=t,hs=hs,gs=gs,freq=freq,counts=counts,rgp=float(RGP[ps].mean()),
                    model=model_q(rows,hs,br,FF,FP),family=model_q(rows,hs,br,FFm,FPm)))
    if fi%200==0: print(f"  fork {fi}/{len(forks)}",flush=True)
print(f"{len(REC)} forks kept, {sum(len(r['t']) for r in REC)} sites",flush=True)

# --- calibration on one half, evaluation on the other ---------------------------------
AG=[0.25,0.5,1,2,4,8,16,32]; LG=[0,0.005,0.01,0.02,0.05,0.1,0.2,0.4]
def bits_count(r,key,a,m):
    c=r["counts"][key][m]; q=(c+a*r["freq"][m])/(c.sum(1,keepdims=True)+a); return -np.log2(q[np.arange(m.sum()),r["t"][m]])
def bits_model(r,key,l,m):
    q=(1-l)*r[key][m]+l*r["freq"][m]; return -np.log2(np.maximum(q[np.arange(m.sum()),r["t"][m]],1e-12))
CK=["phylo","graph"]+[f"allele@{d}" for d in range(DMAX)]
fit={}
for hh in (0,1):                                  # parameters fitted on half hh, used on the other
    for key in CK:
        fit[(key,hh)]=min(AG,key=lambda a:sum(bits_count(r,key,a,r["hs"]==hh).sum() for r in REC))
    for key in ("model","family"):
        fit[(key,hh)]=min(LG,key=lambda l:sum(bits_model(r,key,l,r["hs"]==hh).sum() for r in REC))
PER={}              # fork index -> method -> (bits sum, sites) on the evaluation halves
for i,r in enumerate(REC):
    PER[i]=collections.defaultdict(lambda:[0.0,0])
    for hh in (0,1):
        m=r["hs"]==1-hh
        if not m.any(): continue
        PER[i]["frequency"][0]+=float(-np.log2(r["freq"][m][np.arange(m.sum()),r["t"][m]]).sum()); PER[i]["frequency"][1]+=int(m.sum())
        for key in CK:
            b=bits_count(r,key,fit[(key,hh)],m); PER[i][key][0]+=float(b.sum()); PER[i][key][1]+=int(m.sum())
        for key in ("model","family"):
            b=bits_model(r,key,fit[(key,hh)],m); PER[i][key][0]+=float(b.sum()); PER[i][key][1]+=int(m.sum())
METH=["frequency","graph","phylo","family","model"]+[f"allele@{d}" for d in range(DMAX)]
tot={k:sum(PER[i][k][0] for i in PER)/sum(PER[i][k][1] for i in PER) for k in METH}
fb=lambda i,k: PER[i][k][0]/max(PER[i][k][1],1)
# per fork, per evaluation half, for replication
def half_bits(i,k,hh):
    r=REC[i]; m=r["hs"]==1-hh
    if not m.any(): return None
    if k=="frequency": return float(-np.log2(r["freq"][m][np.arange(m.sum()),r["t"][m]]).mean())
    if k in ("model","family"): return float(bits_model(r,k,fit[(k,hh)],m).mean())
    return float(bits_count(r,k,fit[(k,hh)],m).mean())
def beats(i,a,bs):
    return all((x:=half_bits(i,a,hh)) is not None and all(x<half_bits(i,b,hh) for b in bs) for hh in (0,1))
allelic_free=[i for i in PER if beats(i,"allele@0",["phylo","graph"])]
allelic_model=[i for i in PER if beats(i,"model",["phylo","graph","family"])]
both=set(allelic_free)&set(allelic_model)
ga=np.array([fb(i,"phylo")-fb(i,"allele@0") for i in PER]); gm=np.array([fb(i,"family")-fb(i,"model") for i in PER])
from scipy.stats import spearmanr
rho=spearmanr(ga,gm)
cls={r["fork_idx"]:r["class"] for r in json.load(open("pgb/contingency.json"))}
out=dict(split="clade" if CLADE else "random",forks=len(REC),sites=int(sum(len(r["t"]) for r in REC)),
    bits_per_site={k:round(v,4) for k,v in tot.items()},fitted={f"{k}|{h}":v for (k,h),v in fit.items()},
    allelic_model_free=len(allelic_free),allelic_in_model=len(allelic_model),allelic_both=len(both),
    spearman_allele_vs_model=dict(rho=round(float(rho.correlation),3),p=float(rho.pvalue)),
    decay=[round(tot[f"allele@{d}"],4) for d in range(DMAX)],
    per_fork=[dict(fork=FN[REC[i]["F"]],fork_idx=int(REC[i]["F"]),branches=[FN[b] for b in REC[i]["br"]],sites=int(len(REC[i]["t"])),
                   rgp=round(REC[i]["rgp"],3),cls=cls.get(REC[i]["F"],"other"),
                   bits={k:round(fb(i,k),4) for k in METH},allelic_free=i in allelic_free,allelic_model=i in allelic_model) for i in PER])
json.dump(out,open("pgb/allelic_clade.json" if CLADE else "pgb/allelic.json","w"))
print(f"\n{out['split']} halves: {out['forks']} forks, {out['sites']} sites")
print("bits per site (lower is better):",{k:round(tot[k],3) for k in ["frequency","graph","phylo","family","model","allele@0","allele@1","allele@5"]})
print(f"forks where an upstream allele beats BOTH phylogeny and the graph, in both halves:")
print(f"   model-free (allele@0): {out['allelic_model_free']} of {out['forks']} ({out['allelic_model_free']/out['forks']:.0%})")
print(f"   in the model (beats family order too): {out['allelic_in_model']} ({out['allelic_in_model']/out['forks']:.0%}); both agree on {out['allelic_both']}")
print(f"   per-fork gains agree: Spearman rho={rho.correlation:.3f}, p={rho.pvalue:.1e}")
print("distance decay, bits per site by allele@d:",[round(tot[f"allele@{d}"],3) for d in range(DMAX)])
byc=collections.defaultdict(lambda:[0,0])
for r in out["per_fork"]:
    byc[r["cls"]][0]+=r["allelic_free"]; byc[r["cls"]][1]+=1
print("model-free allelic contingency by class:")
for c,(a,b) in sorted(byc.items(),key=lambda x:-x[1][1]): print(f"   {c:32s} {a:4d} / {b:4d}  ({a/b:.0%})")
hi=[r for r in out["per_fork"] if r["allelic_free"]]; lo=[r for r in out["per_fork"] if not r["allelic_free"]]
print(f"share of sites in a region of plasticity: allelic {np.mean([r['rgp'] for r in hi]):.2f} vs others {np.mean([r['rgp'] for r in lo]):.2f}")
top=sorted(hi,key=lambda r:r["bits"]["phylo"]-r["bits"]["allele@0"],reverse=True)[:12]
print("\nstrongest allelic forks (bits per site):")
print(f"{'fork':22s}{'class':17s}{'sites':>6s} {'freq':>6s}{'graph':>7s}{'phylo':>7s}{'family':>7s}{'model':>7s}{'allele':>8s}  branches")
for r in top:
    b=r["bits"]
    print(f"{r['fork'][:21]:22s}{r['cls'][:16]:17s}{r['sites']:6d} {b['frequency']:6.2f}{b['graph']:7.2f}{b['phylo']:7.2f}{b['family']:7.2f}{b['model']:7.2f}{b['allele@0']:8.2f}  "+" / ".join(x[:16] for x in r["branches"][:3]))
