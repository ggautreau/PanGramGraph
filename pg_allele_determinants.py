"""Model-free check of allelic contingency: at each contingent fork (pg_contingency.py),
does the ALLELE of a family shared upstream predict the branch better than phylogeny?

For every family present within 20 genes upstream in >= 80 % of the fork's sites, a
genome's branch is predicted from the branches of the 15 genomes whose protein of that
family is most similar to its own (cosine on ESM-2 embeddings), leave-one-genome-out and
shrunk to the branch frequencies, like the phylogenetic baseline. No Bacformer here: only
the sequences of the proteins. The best such family is the candidate determinant.

    python3 pg_allele_determinants.py   # -> pgb/allele_determinants.json
"""
import json, collections, numpy as np
Z=np.load("pgb/chrom.npz"); off=Z["offsets"]; FAM=Z["fam"]; PID=Z["pid"]; NG=len(off)-1; NP=off[-1]
J=json.load(open("pgb/chrom_genomes.json")); FN=J["families"]
name_of=np.concatenate([np.array(n,dtype=object) for n in J["gene_names"]])
NPROT=sum(1 for _ in open("pgb/chrom_prot.txt"))
EMB=np.memmap("pgb/chrom_emb.f16",dtype=np.float16,mode="r",shape=(NPROT,480))
R=json.load(open("pgb/contingency.json")); SITES=np.load("pgb/fork_sites.npy")
gen_of=np.repeat(np.arange(NG),np.diff(off)); U=20; K=15
def best_upstream(fork):
    F=fork["fork_idx"]; br=fork["branch_idx"]
    ks=[p for p in SITES if FAM[p-1]==F and FAM[p] in br]
    t=np.array([br.index(FAM[p]) for p in ks]); gs=gen_of[ks]; nb=len(br)
    cnt=np.bincount(t,minlength=nb).astype(float)
    ups=[dict() for _ in ks]                                  # family -> pid, nearest occurrence upstream
    for i,p in enumerate(ks):
        for q in range(p-1,max(off[gs[i]],p-U)-1,-1):
            ups[i].setdefault(int(FAM[q]),int(PID[q]))
    common=[f for f,c in collections.Counter(f for u in ups for f in u).items() if c>=0.8*len(ks) and f!=F]
    res=[]
    for f in common:
        idx=[i for i in range(len(ks)) if f in ups[i]]
        E=np.asarray(EMB[[ups[i][f] for i in idx]],np.float32); E/=np.linalg.norm(E,axis=1,keepdims=True)+1e-9
        S=E@E.T; g=gs[idx]; S[g[:,None]==g[None,:]]=-np.inf
        kk=min(K,len(idx)-1); nn=np.argpartition(-S,kk-1,axis=1)[:,:kk]; ll=0.0; hit=0
        for a,i in enumerate(idx):
            own=np.bincount(t[gs==gs[i]],minlength=nb); fr=(cnt-own+0.5)/(cnt-own+0.5).sum()
            q=(np.bincount(t[np.array(idx)[nn[a]]],minlength=nb)+fr)/(kk+1)
            ll+=-np.log2(q[t[i]]); hit+=int(q.argmax()==t[i])
        res.append(dict(family=FN[f],name="",fam_idx=int(f),bits=round(ll/len(idx),4),acc=round(hit/len(idx),4),sites=len(idx),
                        distinct_alleles=int(len({ups[i][f] for i in idx}))))
    # readable names for the upstream families
    nm=collections.defaultdict(collections.Counter)
    for i,p in enumerate(ks):
        for q in range(p-1,max(off[gs[i]],p-U)-1,-1): nm[int(FAM[q])][name_of[q]]+=1
    for r in res: r["name"]=nm[r["fam_idx"]].most_common(1)[0][0] or r["family"]
    return sorted(res,key=lambda r:r["bits"])
out=[]
top=sorted([r for r in R if r["contingent"]],key=lambda r:r["halves"]["all"]["model"]["bits"]-min(r["halves"]["all"][k]["bits"] for k in ("phylo-15","path-5","path-10","path-20")))[:25]
for r in top:
    b=best_upstream(r); h=r["halves"]["all"]
    out.append(dict(fork=r["fork_name"] or r["fork"],branches=r["branch_names"],cls=r["class"],
        model=h["model"]["bits"],modal=h.get("model-modal",{}).get("bits"),phylo=h["phylo-15"]["bits"],
        path=min(h[k]["bits"] for k in ("path-5","path-10","path-20")),freq=h["frequency"]["bits"],best_alleles=b[:3]))
    x=b[0] if b else None
    print(f"{(r['fork_name'] or r['fork'])[:18]:18s} [{r['class'][:14]:14s}] bits: freq {h['frequency']['bits']:.2f} phylo {h['phylo-15']['bits']:.2f} "
          f"model {h['model']['bits']:.2f}" + (f" | best upstream allele: {x['name'][:14]} {x['bits']:.2f} ({x['distinct_alleles']} alleles)" if x else ""))
json.dump(out,open("pgb/allele_determinants.json","w"),indent=1)
