"""How much does mean-pooling ESM-2 lose, for the questions this project asks?

Bacformer's input is one 480-dimensional vector per protein: the mean over residues of
ESM-2 t12 35M. Two measurements of what that costs.

A. COLLISIONS. Over the 397,521 distinct proteins of the 540 chromosomes, the nearest
   neighbour of each sampled protein in pooled space: how close, and is it even the same
   PPanGGOLiN family? Pairs the encoder cannot separate are pairs Bacformer cannot
   separate either, whatever it learned. For same-family pairs of equal length, pooled
   cosine is put against real sequence identity, so we see what a residue difference
   costs in the pooled space.

B. THE TASK. Branch prediction at forks, exact allele identity against pooled cosine:
     exact@0   leave-one-genome-out branches of the genomes carrying the IDENTICAL
               protein of the fork family (no embedding at all, no threshold)
     allele@0  the same but neighbours chosen by pooled ESM-2 cosine (what pg_allelic used)
   The gap between them is the information pooling destroys, in bits, on the one question
   where the alleles are known to matter.

    python3 pool_loss.py
"""
import json, collections, warnings, numpy as np, torch
warnings.simplefilter("ignore")
dev="cuda:0" if torch.cuda.is_available() else "cpu"; rng=np.random.default_rng(0); K=15
Z=np.load("pgb/chrom.npz"); off=Z["offsets"]; FAM=Z["fam"]; PID=Z["pid"]; NG=len(off)-1
J=json.load(open("pgb/chrom_genomes.json")); FN=J["families"]
P=[l.rstrip("\n") for l in open("pgb/chrom_prot.txt")]; NPROT=len(P)
E=np.asarray(np.memmap("pgb/chrom_emb.f16",dtype=np.float16,mode="r",shape=(NPROT,480)),dtype=np.float32)
En=torch.tensor(E/np.maximum(np.linalg.norm(E,axis=1,keepdims=True),1e-9),device=dev,dtype=torch.float16)
# a protein's family: the one it has most often in the chromosomes
pf=collections.defaultdict(collections.Counter)
for f,p in zip(FAM.tolist(),PID.tolist()): pf[p][f]+=1
prot_fam=np.full(NPROT,-1,np.int64)
for p,c in pf.items(): prot_fam[p]=c.most_common(1)[0][0]
seen=np.where(prot_fam>=0)[0]
print(f"A. {len(seen)} distinct proteins seen in the 540 chromosomes, {len(set(prot_fam[seen].tolist()))} families",flush=True)

samp=rng.choice(seen,20000,replace=False)
nn_cos=np.zeros(len(samp),np.float32); nn_idx=np.zeros(len(samp),np.int64)
Q=En[torch.tensor(samp,device=dev)]
for i in range(0,len(samp),512):
    S=(Q[i:i+512].float()@En[torch.tensor(seen,device=dev)].float().T)
    for r,g in enumerate(range(i,min(i+512,len(samp)))):
        S[r,(seen==samp[g]).nonzero()[0]]=-1                     # itself
    v,j=S.max(1); nn_cos[i:i+512]=v.cpu().numpy(); nn_idx[i:i+512]=seen[j.cpu().numpy()]
same=prot_fam[nn_idx]==prot_fam[samp]
print(f"   nearest neighbour in pooled space: cosine median {np.median(nn_cos):.4f}, "
      f"90th pct {np.quantile(nn_cos,.9):.4f}, 99th {np.quantile(nn_cos,.99):.4f}")
for thr in (0.99,0.999,0.9999):
    m=nn_cos>=thr
    print(f"   cosine >= {thr}: {m.mean()*100:5.2f}% of proteins; of those, {(~same[m]).mean()*100:5.1f}% have a DIFFERENT family as nearest neighbour")
print(f"   overall, nearest neighbour is a different family for {(~same).mean()*100:.1f}% of proteins")

# same-family pairs of equal length: pooled cosine vs true identity
byfam=collections.defaultdict(list)
for p in seen.tolist(): byfam[prot_fam[p]].append(p)
pairs=[]
for f,ps in byfam.items():
    if len(ps)<2: continue
    for _ in range(min(40,len(ps))):
        a,b=rng.choice(ps,2,replace=False)
        if a!=b and len(P[a])==len(P[b]): pairs.append((a,b))
    if len(pairs)>40000: break
a=np.array([x for x,_ in pairs]); b=np.array([y for _,y in pairs])
cos=(En[torch.tensor(a,device=dev)].float()*En[torch.tensor(b,device=dev)].float()).sum(1).cpu().numpy()
ident=np.array([np.mean([c==d for c,d in zip(P[x],P[y])]) for x,y in pairs])
print(f"\n   {len(pairs)} same-family pairs of equal length:")
for lo,hi in ((0.995,1.0),(0.98,0.995),(0.95,0.98),(0.90,0.95),(0.0,0.90)):
    m=(ident>=lo)&(ident<hi+1e-9)
    if m.sum()>30: print(f"     sequence identity {lo:.3f}-{hi:.3f}: {m.sum():6d} pairs, pooled cosine {np.median(cos[m]):.4f} (5th-95th {np.quantile(cos[m],.05):.4f}-{np.quantile(cos[m],.95):.4f})")
from scipy.stats import spearmanr
print(f"     Spearman(identity, pooled cosine) = {spearmanr(ident,cos).correlation:.3f}")
hi=cos[ident>0.99]; lo=cos[(ident<0.95)&(ident>0.8)]
if len(hi)>30 and len(lo)>30:
    print(f"     pairs >99% identical: pooled cosine {np.median(hi):.4f}; pairs 80-95% identical: {np.median(lo):.4f}")

# ---------------- B. the task ----------------
SITES=np.load("pgb/fork_sites.npy"); gen_of=np.repeat(np.arange(NG),np.diff(off))
occ=collections.Counter(); succ=collections.defaultdict(collections.Counter)
for p in SITES: succ[int(FAM[p-1])][int(FAM[p])]+=1; occ[int(FAM[p-1])]+=1
by_fork=collections.defaultdict(list)
for p in SITES: by_fork[int(FAM[p-1])].append(int(p))
AG=[0.25,0.5,1,2,4,8,16,32]
rows=[]
for F,ps in by_fork.items():
    br=[b for b,c in succ[F].most_common(8) if c>=max(0.10*occ[F],20)]
    if len(br)<2: continue
    ps=np.array([p for p in ps if FAM[p] in br and (p-off[gen_of[p]])>12])
    if len(ps)<30: continue
    if len(ps)>700: ps=np.sort(rng.choice(ps,700,replace=False))
    nb=len(br); t=np.array([br.index(FAM[p]) for p in ps]); gs=gen_of[ps]; n=len(ps); same_g=gs[:,None]==gs[None,:]
    cnt=np.bincount(t,minlength=nb).astype(float)
    own=np.stack([np.bincount(t[gs==g],minlength=nb) for g in gs]); freq=(cnt[None]-own+0.5); freq/=freq.sum(1,keepdims=True)
    pid=PID[ps-1]
    Ep=En[torch.tensor(pid,device=dev)].float(); S=(Ep@Ep.T).cpu().numpy(); S[same_g]=-np.inf
    kk=min(K,n-1); nn=np.argpartition(-S,kk-1,axis=1)[:,:kk]
    c_cos=np.stack([np.bincount(t[r],minlength=nb) for r in nn]).astype(float)
    c_exact=np.zeros((n,nb))
    for i in range(n):
        o=np.where((pid==pid[i])&(gs!=gs[i]))[0]
        if len(o): c_exact[i]=np.bincount(t[o],minlength=nb)
    rows.append((t,freq,c_cos,c_exact,gs%2))
print(f"\nB. {len(rows)} forks, {sum(len(r[0]) for r in rows)} sites",flush=True)
def bits(rs,which,alpha,half):
    o=[]
    for t,freq,cc,ce,hh in rs:
        m=hh==half
        if not m.any(): continue
        c=(cc if which=="cos" else ce)[m]
        q=(c+alpha*freq[m])/(c.sum(1,keepdims=True)+alpha); o.append(-np.log2(q[np.arange(m.sum()),t[m]]))
    return np.concatenate(o) if o else np.array([])
res={}
for which in ("cos","exact"):
    tot=[]
    for h in (0,1):
        a_=min(AG,key=lambda al: bits(rows,which,al,h).sum())          # fitted on this half
        tot.append(bits(rows,which,a_,1-h))                             # scored on the other
    res[which]=np.concatenate(tot).mean()
base=[]
for h in (0,1):
    for t,freq,cc,ce,hh in rows:
        m=hh==1-h
        if m.any(): base.append(-np.log2(freq[m][np.arange(m.sum()),t[m]]))
print(f"   frequency only                       {np.concatenate(base).mean():.3f} bits/site")
print(f"   neighbours by pooled ESM-2 cosine    {res['cos']:.3f} bits/site")
print(f"   neighbours by EXACT allele identity  {res['exact']:.3f} bits/site")
print(f"   -> pooling costs {res['cos']-res['exact']:+.3f} bits/site, "
      f"{(res['cos']-res['exact'])/max(np.concatenate(base).mean()-res['exact'],1e-9)*100:.0f}% of the information the exact allele carries")
json.dump(dict(nn_median=float(np.median(nn_cos)),diff_family_nn=float((~same).mean()),
               spearman_identity_cosine=float(spearmanr(ident,cos).correlation),
               bits_frequency=float(np.concatenate(base).mean()),bits_cosine=float(res["cos"]),
               bits_exact=float(res["exact"])),open("pgb/pool_loss.json","w"))
