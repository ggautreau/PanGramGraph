"""Positional contingency across the whole chromosome: where does the genome upstream of a
fork decide which branch comes next, beyond the genome's phylogeny and beyond what the
pangenome graph can read along its paths?

Forks: families followed by >= 2 families, each in >= 10 of the 540 complete genomes
(pg_calls.py). At each fork site, Bacformer's call (top 64) is read into branch
probabilities through the decoder P(family | cluster), learned on the other half of the
genomes (as in pgb_graph.py). Compared, in bits per site:
  frequency    branch frequencies at this fork (the graph's edges), leave-one-genome-out
  path-k       genomes with the same last k families, k = 5, 10, 20 (the graph read along
               paths), leave-one-genome-out
  phylo-15     branches of the 15 nearest genomes on core-genome alleles (pgb_cgmlst.py)
  model        Bacformer on the real proteins
  model-modal  Bacformer on the same chromosomes where every protein is its family's most
               common one (pg_calls.py --modal): only the order of families is left, which
               is what the graph holds. What the model loses here is allelic information.
A fork is contingent when the model beats phylo-15 and every path-k in each half of the
genomes separately: the effect must replicate on genomes the other half never saw.

    python3 pg_contingency.py            # -> pgb/contingency.json, printed summary
    python3 pg_contingency.py --clade    # halves are groups of whole clades instead of a random
        split: learned on some lineages, tested on others -> pgb/contingency_clade.json
"""
import sys
CLADE="--clade" in sys.argv
import json, os, re, collections, warnings, numpy as np, pandas as pd, tables
from scipy import sparse
warnings.simplefilter("ignore")
Z=np.load("pgb/chrom.npz"); off=Z["offsets"]; FAM=Z["fam"]; RGP=Z["rgp"]; SPOT=Z["spot"]; NG=len(off)-1; NP=off[-1]
J=json.load(open("pgb/chrom_genomes.json")); GEN=J["genomes"]; FN=J["families"]
name_of=np.concatenate([np.array(n,dtype=object) for n in J["gene_names"]])
TF=np.memmap("pgb/calls_top15_f.i32",dtype=np.int32,mode="r",shape=(NP,15)); TP=np.memmap("pgb/calls_top15_p.f16",dtype=np.float16,mode="r",shape=(NP,15))
SITES=np.load("pgb/fork_sites.npy"); NS=len(SITES)
FF=np.memmap("pgb/calls_fork_f.i32",dtype=np.int32,mode="r",shape=(NS,64)); FP=np.memmap("pgb/calls_fork_p.f16",dtype=np.float16,mode="r",shape=(NS,64))
HAVE_MODAL=os.path.exists("pgb/calls_fork_modal_p.f16")
if HAVE_MODAL:
    FFm=np.memmap("pgb/calls_fork_modal_f.i32",dtype=np.int32,mode="r",shape=(NS,64)); FPm=np.memmap("pgb/calls_fork_modal_p.f16",dtype=np.float16,mode="r",shape=(NS,64))
else: FFm=FPm=None
gen_of=np.repeat(np.arange(NG),np.diff(off)); first=np.zeros(NP,bool); first[off[:-1]]=True

# --- partitions, modules, products from the HDF5 -------------------------------------
h=tables.open_file("pgb/ecoli_11587.h5")
part=np.array([{b"P":"persistent",b"S":"shell",b"C":"cloud"}[x] for x in h.root.geneFamiliesInfo.read(field="partition")])
fam_idx={n:i for i,n in enumerate(FN)}
module_of={fam_idx[x.decode()]:int(m) for x,m in zip(h.root.modules.read(field="geneFam"),h.root.modules.read(field="module"))}
gf=h.root.geneFamilies.read(); gf_fam=pd.Index(FN).get_indexer([x.decode() for x in gf["geneFam"]])
first_gene=pd.Series(gf["gene"]).groupby(gf_fam).first()                  # one gene per family
g_id=h.root.annotations.genes.read(field="ID"); g_gd=h.root.annotations.genes.read(field="genedata_id")
def products(fams):
    fams=[f for f in fams if f in first_gene.index]
    rows=pd.Index(g_id).get_indexer([first_gene[f] for f in fams]); gd=g_gd[rows]; o=np.argsort(gd)
    pr=h.root.annotations.genedata.read_coordinates(gd[o],field="product")
    return {fams[i]:p.decode() for i,p in zip(o,pr)}

# --- decoder P(family | cluster), per half of the genomes ------------------------------
fams=np.unique(FAM); FI=np.full(FAM.max()+1,-1); FI[fams]=np.arange(len(fams)); NFm=len(fams)
def decoder(mask):
    m=np.where(mask&~first)[0]
    A=sparse.csr_matrix((np.asarray(TP[m],np.float32).ravel(),(np.repeat(FI[FAM[m]],15),np.asarray(TF[m]).ravel())),shape=(NFm,50001))
    return A.tocsr(),np.asarray(A.sum(0)).ravel()
# --- phylogeny proxy among the 540 ------------------------------------------------------
D2=np.load("pgb/cgmlst_dist.npy").astype(np.float32); acc2=json.load(open("pgb/cgmlst_loci.json"))["genomes"]
ix={a:i for i,a in enumerate(acc2)}; sel=np.array([ix[g["acc"]] for g in GEN]); D=D2[np.ix_(sel,sel)]
if CLADE:                                   # whole clades to one side: average linkage, 12 clades, balanced
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    cl=fcluster(linkage(squareform(np.maximum(D,D.T),checks=False),"average"),12,"maxclust")
    side=np.zeros(NG,int); load=[0,0]
    for c,nm in collections.Counter(cl).most_common():
        s_=int(load[1]<load[0]); side[cl==c]=s_; load[s_]+=nm
    GH=side; print(f"clade halves: {load} genomes",flush=True)
else: GH=np.arange(NG)%2
half=GH[gen_of]; DEC={hh:decoder(half!=hh) for hh in (0,1)}
print("decoders built",flush=True)

prev=FAM[SITES-1]; cur=FAM[SITES]
succ=collections.defaultdict(collections.Counter)
for g in range(NG):
    f=FAM[off[g]:off[g+1]]
    for a,b in set(zip(f[:-1].tolist(),f[1:].tolist())): succ[a][b]+=1
EPS=1e-3; KEYS=["frequency","path-5","path-10","path-20","phylo-15","model"]+(["model-modal"] if HAVE_MODAL else [])
def branch_weights(br,hh):
    """(sorted clusters, (clusters+1) x branches) P(branch | cluster) from the decoder that never saw half hh"""
    A,tt=DEC[hh]; rows=[A.getrow(FI[b]) for b in br]
    cl=np.array(sorted({int(c) for r in rows for c in r.indices}),dtype=np.int64); pos={c:i for i,c in enumerate(cl.tolist())}
    M=np.zeros((len(cl)+1,len(br)))                      # last row: clusters no branch reads as
    for j,r in enumerate(rows):
        for c,v in zip(r.indices.tolist(),r.data.tolist()): M[pos[c],j]=v/max(tt[c],1e-12)
    return cl,M
def branch_q(ks,gs,br,Fc,Fp):
    C=np.asarray(Fc[ks]).astype(np.int64); Pv=np.asarray(Fp[ks],np.float32); q=np.zeros((len(ks),len(br)))
    for hh in (0,1):
        m=np.where(GH[gs]==hh)[0]
        if not len(m): continue
        cl,M=branch_weights(br,hh)
        if len(cl):
            ix=np.clip(np.searchsorted(cl,C[m]),0,len(cl)-1); ix=np.where(cl[ix]==C[m],ix,len(cl))
            q[m]=(Pv[m][:,:,None]*M[ix]).sum(1)
    q=q+EPS*np.maximum(q.sum(1,keepdims=True),1e-12); return q/q.sum(1,keepdims=True)
by_fork=collections.defaultdict(list)
for k in range(NS): by_fork[int(prev[k])].append(k)
out=[]
for F,ks in by_fork.items():
    br=[b for b,c in succ[F].most_common(8) if c>=10]
    ks=[k for k in ks if cur[k] in br]
    if len(br)<2 or len(ks)<20: continue
    nb=len(br); ks=np.array(ks); t=np.array([br.index(cur[k]) for k in ks]); gs=gen_of[SITES[ks]]
    Q={"model":branch_q(ks,gs,br,FF,FP)}
    if HAVE_MODAL: Q["model-modal"]=branch_q(ks,gs,br,FFm,FPm)
    cnt=np.bincount(t,minlength=nb).astype(float)
    paths={L:[tuple(FAM[max(off[g],SITES[k]-L):SITES[k]].tolist()) for k,g in zip(ks,gs)] for L in (5,10,20)}
    bypath={L:collections.defaultdict(list) for L in paths}
    for L in paths:
        for i,pa in enumerate(paths[L]): bypath[L][pa].append(i)
    Ds=D[np.ix_(gs,gs)].copy(); Ds[gs[:,None]==gs[None,:]]=np.inf; kk=min(15,len(ks)-1)
    nn=np.argpartition(Ds,kk-1,axis=1)[:,:kk]
    ll={key:np.zeros(len(ks)) for key in KEYS}; hit={key:np.zeros(len(ks)) for key in KEYS}
    for i in range(len(ks)):
        own=np.bincount(t[gs==gs[i]],minlength=nb); fr=(cnt-own+0.5)/(cnt-own+0.5).sum()
        qs={"frequency":fr,"phylo-15":(np.bincount(t[nn[i]],minlength=nb)+fr)/(kk+1)}
        for L in paths:
            po=[j for j in bypath[L][paths[L][i]] if gs[j]!=gs[i]]
            qs[f"path-{L}"]=(np.bincount(t[po],minlength=nb)+fr)/(len(po)+1) if po else fr
        for key in Q: qs[key]=Q[key][i]
        for key in KEYS: ll[key][i]=-np.log2(qs[key][t[i]]); hit[key][i]=float(qs[key].argmax()==t[i])
    res={}
    for hh in ("0","1","all"):
        m=(GH[gs]==int(hh)) if hh!="all" else np.ones(len(ks),bool)
        res[hh]={key:dict(bits=round(float(ll[key][m].mean()),4),acc=round(float(hit[key][m].mean()),4)) for key in KEYS} if m.sum()>=5 else None
    graph_best=lambda r: min(r[k]["bits"] for k in ("phylo-15","path-5","path-10","path-20"))
    ok=all(res[hh] and res[hh]["model"]["bits"]<graph_best(res[hh]) for hh in ("0","1"))
    ok_modal=HAVE_MODAL and all(res[hh] and res[hh]["model-modal"]["bits"]<graph_best(res[hh]) for hh in ("0","1"))
    out.append(dict(fork=FN[F],fork_idx=int(F),branches=[FN[b] for b in br],branch_idx=[int(b) for b in br],
        fork_name=collections.Counter(name_of[SITES[k]-1] for k in ks).most_common(1)[0][0],
        branch_names=[collections.Counter(name_of[SITES[k]] for k in ks if cur[k]==b).most_common(1)[0][0] for b in br],
        fork_partition=part[F],branch_partitions=[part[b] for b in br],branch_counts=[int(c) for c in np.bincount(t,minlength=nb)],
        sites=int(len(ks)),genomes=int(len(set(gs.tolist()))),in_rgp=round(float(RGP[SITES[ks]].mean()),3),
        spot=int(collections.Counter(SPOT[SITES[ks]].tolist()).most_common(1)[0][0]),
        in_module=bool(any(b in module_of for b in br)),position=int(np.median(SITES[ks]-off[gs])),
        contingent=bool(ok),contingent_family_order=bool(ok_modal),halves=res))
# --- functional classes from gene names and products ----------------------------------
pr=products({x for r in out for x in [r["fork_idx"]]+r["branch_idx"]})
CLASSES=[("LPS, O-antigen, capsule",r"\b(waa|rfa|rfb|wb[a-z]|wz[a-z]|wca|kps|neu)|lipopolysaccharide|o-antigen|capsul|polysaccharide|glycosyltransferase|glycosyl transferase"),
         ("prophage",r"phage|integrase|excisionase|terminase|capsid|portal|tail|holin|lysozyme|endolysin|\bcro\b|repressor ci"),
         ("IS, transposase",r"transposase|\bins[a-z]\b|insertion element|\bis[0-9]"),
         ("toxin-antitoxin, defence",r"toxin|antitoxin|\bcbt|\bcbe|restriction|methyltransferase|crispr|abortive|\bhig[ab]|\brel[be]\b"),
         ("secretion, adhesion, motility",r"secretion|\besc[a-z]|\bsct|\bces|fimbri|\bfim[a-z]|pilus|pilin|\bpap|adhesin|autotransporter|agn43|flagell|\bfli[a-z]|\bflg"),
         ("transport",r"transport|permease|porin|abc |efflux|symporter|antiporter")]
def fclass(r):
    txt=" ".join([r["fork_name"]]+r["branch_names"]+[pr.get(x,"") for x in [r["fork_idx"]]+r["branch_idx"]]).lower()
    for c,pat in CLASSES:
        if re.search(pat,txt): return c
    return "other"
for r in out: r["class"]=fclass(r); r["products"]=[pr.get(x,"") for x in [r["fork_idx"]]+r["branch_idx"]]
json.dump(out,open("pgb/contingency_clade.json" if CLADE else "pgb/contingency.json","w"))
# --- summary -----------------------------------------------------------------------
n=len(out); con=[r for r in out if r["contingent"]]; oth=[r for r in out if not r["contingent"]]
print(f"\n{n} forks. Contingent (model beats phylogeny and every graph path, in both halves): {len(con)} ({len(con)/n:.1%})")
mb={k:np.average([r["halves"]["all"][k]["bits"] for r in out],weights=[r["sites"] for r in out]) for k in KEYS}
print("bits per site, weighted over forks:",{k:round(v,3) for k,v in mb.items()})
if HAVE_MODAL:
    fo=sum(r["contingent_family_order"] for r in con)
    print(f"of the contingent forks, {fo} stay contingent when the model sees only families (family order); "
          f"{len(con)-fo} need the real alleles (allelic contingency)")
print(f"sites in a region of plasticity: contingent {np.mean([r['in_rgp'] for r in con]):.2f} vs others {np.mean([r['in_rgp'] for r in oth]):.2f}")
print(f"branches in a panModule module: contingent {np.mean([r['in_module'] for r in con]):.2f} vs others {np.mean([r['in_module'] for r in oth]):.2f}")
cc=collections.Counter(r["class"] for r in con); co=collections.Counter(r["class"] for r in out)
print("functional class: contingent / all forks (share contingent)")
for c,_ in co.most_common(): print(f"  {c:32s} {cc[c]:4d} / {co[c]:4d}  ({cc[c]/co[c]:.0%})")
