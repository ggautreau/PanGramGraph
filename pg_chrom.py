"""Whole chromosomes of the complete genomes of PanGBank pangenome 11587, for the
genome-wide contingency analysis.

The 540 assemblies at NCBI level "Complete Genome"; for each, its largest contig (the
chromosome), rotated to the dnaA-family gene and read in dnaA's direction of
transcription, as the 80-gene windows were (pgb_window.py). A chromosome without a
dnaA-family gene is anchored on dnaN at position 1. Each gene: PPanGGOLiN family,
whether panRGP puts it in a region of plasticity and in which insertion spot, and its
protein, deduplicated so that each distinct protein is embedded once.

    python3 pg_chrom.py      # -> pgb/chrom.npz, pgb/chrom_prot.txt, pgb/chrom_genomes.json
"""
import json, collections, warnings, numpy as np, pandas as pd, tables
from Bio.Seq import Seq
warnings.simplefilter("ignore")
h=tables.open_file("pgb/ecoli_11587.h5"); A=h.root.annotations
g_id=A.genes.read(field="ID"); g_ctg=A.genes.read(field="contig"); g_gd=A.genes.read(field="genedata_id")
gd=A.genedata; pos=gd.read(field="position")[g_gd]; strand=gd.read(field="strand")[g_gd]; gname=gd.read(field="name")[g_gd]
info=h.root.geneFamiliesInfo.read(); fam_names=[x.decode() for x in info["name"]]
gf=h.root.geneFamilies.read()
g_fam=pd.Index(fam_names).get_indexer([x.decode() for x in gf["geneFam"]])[pd.Index(gf["gene"]).get_indexer(g_id)]
def family_of(name): return collections.Counter(g_fam[gname==name]).most_common(1)[0][0]
FA,FN=family_of(b"dnaA"),family_of(b"dnaN")
mw=h.root.metadata.genomes.pangbank_wf.read()
complete={r["ID"].decode():dict(strain=r["ncbi_strain_identifiers"].decode(),org=r["ncbi_organism_name"].decode())
          for r in mw if r["ncbi_assembly_level"].decode()=="Complete Genome"}
ct=A.contigs.read(); chrom={}
for r in ct:
    g=r["genome"].decode()
    if g in complete and (g not in chrom or r["length"]>chrom[g][1]): chrom[g]=(int(r["ID"]),int(r["length"]),bool(r["is_circular"]))
print(f"{len(chrom)} complete genomes; dnaA family {fam_names[FA]}",flush=True)
# RGP and spot of every gene
rg=h.root.RGP.read(); sp=h.root.spots.read()
spot_of_rgp=dict(zip(sp["RGP"],sp["spot"]))
rgp_of=pd.Series(rg["RGP"],index=rg["gene"]); rgp_of=rgp_of[~rgp_of.index.duplicated()]
gi_rgp=rgp_of.index.get_indexer(g_id); in_rgp=gi_rgp>=0
spot=np.full(len(g_id),-1,np.int64); spot[in_rgp]=[spot_of_rgp.get(x,-1) for x in rgp_of.values[gi_rgp[in_rgp]]]
gs_seq=A.geneSequences.read(field="seqid")[pd.Index(A.geneSequences.read(field="gene")).get_indexer(g_id)]

ctg_of={v[0]:k for k,v in chrom.items()}
rows=np.where(np.isin(g_ctg,list(ctg_of)))[0]
by=collections.defaultdict(list)
for i in rows: by[int(g_ctg[i])].append(i)
order={}; skipped=[]
for c,L in by.items():
    L.sort(key=lambda i:pos[i]); gen=ctg_of[c]
    a=[j for j,i in enumerate(L) if g_fam[i]==FA]; start=0
    if not a: a=[j for j,i in enumerate(L) if g_fam[i]==FN]; start=1
    if not a: skipped.append(gen); continue
    k=a[0]; d=1 if strand[L[k]]==b"+" else -1
    order[gen]=(start,[L[(k+d*s)%len(L)] for s in range(len(L))])
print(f"rotated and oriented {len(order)}; skipped {len(skipped)}",flush=True)

# distinct proteins: translate each distinct sequence once, in chunks
allg=np.concatenate([np.array(v[1]) for v in order.values()])
useq=np.unique(gs_seq[allg]); prot_of={}; CH=10000
for s in range(0,len(useq),CH):
    ids=useq[s:s+CH]; dna=A.sequences.read_coordinates(ids,field="dna")
    for q,d in zip(ids,dna):
        p=str(Seq(d.decode()).translate(table=11)).rstrip("*"); prot_of[int(q)]=("M"+p[1:]) if p else "M"
    print(f"  translated {min(s+CH,len(useq))}/{len(useq)}",flush=True)
h.close()
pid={}; plist=[]
for q in useq:
    p=prot_of[int(q)]
    if p not in pid: pid[p]=len(plist); plist.append(p)
gens=sorted(order); off=[0]; F=[];R=[];S=[];P_=[];N=[]
for gen in gens:
    L=order[gen][1]
    F.append(g_fam[L]); R.append(in_rgp[L]); S.append(spot[L]); P_.append([pid[prot_of[int(gs_seq[i])]] for i in L])
    N.append([x.decode() for x in gname[L]]); off.append(off[-1]+len(L))
np.savez_compressed("pgb/chrom.npz",offsets=np.array(off),fam=np.concatenate(F).astype(np.int32),rgp=np.concatenate(R),
                    spot=np.concatenate(S).astype(np.int32),pid=np.concatenate(P_).astype(np.int32))
open("pgb/chrom_prot.txt","w").write("\n".join(plist)+"\n")
json.dump(dict(pangenome="11587 GTDB_refseq v2.0.0 (GTDB R232)",genomes=[dict(acc=g,start=order[g][0],**complete[g]) for g in gens],
               gene_names=N,families=fam_names),open("pgb/chrom_genomes.json","w"))
print(f"genomes {len(gens)}  genes {off[-1]}  distinct proteins {len(plist)}")
