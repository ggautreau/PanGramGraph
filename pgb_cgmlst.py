"""A phylogeny proxy for the 2,002 genomes of PanGBank pangenome 11587: core-genome
allele profiles, cgMLST-style, from the HDF5 alone.

Loci: PPanGGOLiN persistent families present exactly once in >= 95 % of genomes. The
allele of a genome at a locus is the identifier of its gene's DNA sequence: PanGBank
stores each distinct sequence once, so identical alleles share an identifier. The
distance between two genomes is the fraction of loci typed in both where their
alleles differ. Used as the lineage baseline for fork_rules.py: a genome's branch is
predicted from its nearest relatives.

    python3 pgb_cgmlst.py        # -> pgb/cgmlst_dist.npy (2,002 x 2,002, float16), pgb/cgmlst_loci.json
"""
import json, warnings, numpy as np, pandas as pd, tables, torch
warnings.simplefilter("ignore")
dev="cuda:0" if torch.cuda.is_available() else "cpu"
h=tables.open_file("pgb/ecoli_11587.h5"); A=h.root.annotations
g_id=A.genes.read(field="ID"); g_ctg=A.genes.read(field="contig")
ct=A.contigs.read(); gnames=[x.decode() for x in A.genomes.read(field="name")]; gidx={n:i for i,n in enumerate(gnames)}
ctg2gen=np.full(int(ct["ID"].max())+1,-1,np.int64); ctg2gen[ct["ID"]]=[gidx[x.decode()] for x in ct["genome"]]
g_gen=ctg2gen[g_ctg]
info=h.root.geneFamiliesInfo.read(); fam_names=[x.decode() for x in info["name"]]; part=info["partition"]
gf=h.root.geneFamilies.read()
g_fam=pd.Index(fam_names).get_indexer([x.decode() for x in gf["geneFam"]])[pd.Index(gf["gene"]).get_indexer(g_id)]
g_seq=A.geneSequences.read(field="seqid")[pd.Index(A.geneSequences.read(field="gene")).get_indexer(g_id)]
h.close()
NG=len(gnames); pers=np.where(part==b"P")[0]; pmap=np.full(len(fam_names),-1); pmap[pers]=np.arange(len(pers))
m=np.where(pmap[g_fam]>=0)[0]; gg=g_gen[m]; ff=pmap[g_fam[m]]
cnt=np.bincount(gg*len(pers)+ff,minlength=NG*len(pers)).reshape(NG,len(pers))
loci=np.where((cnt==1).mean(0)>=0.95)[0]
AL=np.full((NG,len(pers)),-1,np.int64); one=cnt[gg,ff]==1; AL[gg[one],ff[one]]=g_seq[m][one]; AL=AL[:,loci]
print(f"{len(pers)} persistent families; {len(loci)} single-copy loci in >= 95 % of genomes; "
      f"typed {np.mean(AL>=0)*100:.1f} % of genome x locus",flush=True)
X=torch.tensor(AL,device=dev); diff=torch.zeros((NG,NG),device=dev); both=torch.zeros((NG,NG),device=dev)
for j in range(X.shape[1]):
    a=X[:,j]; v=a>=0; vv=(v[:,None]&v[None,:]).float()
    diff+=(a[:,None]!=a[None,:]).float()*vv; both+=vv
D=(diff/both.clamp(min=1)).cpu().numpy(); np.fill_diagonal(D,0)
np.save("pgb/cgmlst_dist.npy",D.astype(np.float16))
json.dump(dict(genomes=gnames,loci=[fam_names[pers[i]] for i in loci]),open("pgb/cgmlst_loci.json","w"))
s=gidx["GCF_000008865.2"]; nn=np.argsort(D[s])[1:6]
W={g["acc"]:g for g in json.load(open("pgb/window.json"))["genomes"]}
print("median pairwise distance %.3f"%np.median(D[np.triu_indices(NG,1)]))
print("nearest to O157:H7 Sakai:",[(gnames[i],W[gnames[i]].get("strain",""),round(float(D[s,i]),4)) for i in nn])
