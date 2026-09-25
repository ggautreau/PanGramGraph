"""The dnaA window of every genome in PanGBank pangenome 11587 (GTDB_refseq v2.0.0,
GTDB R232, 2,002 genomes): 80 genes from the replication initiator, oriented as in
the 11-strain walk, with PPanGGOLiN family, partition, RGP membership and protein.

Anchor: the gene of the dnaA family, the PPanGGOLiN family holding the genes named
dnaA. A genome without one (dnaA a pseudogene, or lost in assembly) is anchored on
the dnaN family at locus 1, as CFT073 and Nissle 1917 were in the first walk. The
walk follows the anchor's strand: dnaA and dnaN are co-transcribed, so it reads
dnaA, dnaN, recF, gyrB. It wraps on circular contigs and stops at a contig end, so
draft assemblies give shorter windows.

Sequences are stored in coding orientation for both strands. The first residue is
set to M, as in RefSeq translations of GTG/TTG starts.

    python3 pgb_window.py        # -> pgb/window.json, pgb/window_prot.json
"""
import json, collections, warnings, numpy as np, pandas as pd, tables
from Bio.Seq import Seq
warnings.simplefilter("ignore")
W=80
h=tables.open_file("pgb/ecoli_11587.h5"); A=h.root.annotations

# --- genes, their order on contigs, their families -------------------------
g_id=A.genes.read(field="ID"); g_ctg=A.genes.read(field="contig"); g_gd=A.genes.read(field="genedata_id")
gd=A.genedata; pos=gd.read(field="position")[g_gd]; strand=gd.read(field="strand")[g_gd]; gname=gd.read(field="name")[g_gd]
print(f"genes {len(g_id)}",flush=True)
info=h.root.geneFamiliesInfo.read(); fam_names=[x.decode() for x in info["name"]]
PART={"P":"persistent","S":"shell","C":"cloud"}; fam_part=[PART.get(x.decode(),x.decode()) for x in info["partition"]]
gf=h.root.geneFamilies.read()
g_fam=pd.Index(fam_names).get_indexer([x.decode() for x in gf["geneFam"]])[pd.Index(gf["gene"]).get_indexer(g_id)]
def family_of(name):
    c=collections.Counter(g_fam[gname==name]); return c.most_common(1)[0][0]
FA,FN=family_of(b"dnaA"),family_of(b"dnaN")
print(f"dnaA family {fam_names[FA]}  dnaN family {fam_names[FN]}",flush=True)

ct=A.contigs.read(); ctg_genome={int(i):g.decode() for i,g in zip(ct["ID"],ct["genome"])}
ctg_circ={int(i):bool(c) for i,c in zip(ct["ID"],ct["is_circular"])}
ctg_len={int(i):int(l) for i,l in zip(ct["ID"],ct["length"])}
genomes=[x.decode() for x in A.genomes.read(field="name")]

# anchor per genome: dnaA family, else dnaN family at locus 1; longest contig wins
anchor={}
for F,loc in ((FA,0),(FN,1)):
    for i in np.where(g_fam==F)[0]:
        gen=ctg_genome[int(g_ctg[i])]; cur=anchor.get(gen)
        if cur and cur[1]<loc: continue                   # a dnaA anchor beats any dnaN
        if cur is None or ctg_len[int(g_ctg[i])]>ctg_len[int(g_ctg[cur[0]])]:
            anchor[gen]=(i,loc)
print(f"anchored {len(anchor)}/{len(genomes)}: dnaA {sum(v[1]==0 for v in anchor.values())}, "
      f"dnaN {sum(v[1]==1 for v in anchor.values())}",flush=True)

# --- walk each anchor contig -------------------------------------------------
need=set(int(g_ctg[i]) for i,_ in anchor.values())
rows=np.where(np.isin(g_ctg,list(need)))[0]
by_ctg=collections.defaultdict(list)
for i in rows: by_ctg[int(g_ctg[i])].append(i)
for c in by_ctg: by_ctg[c].sort(key=lambda i:pos[i])
win={}
for gen,(i,loc) in anchor.items():
    c=int(g_ctg[i]); L=by_ctg[c]; k=L.index(i); d=1 if strand[i]==b"+" else -1
    out=[]
    for s in range(W-loc):
        j=k+d*s
        if not 0<=j<len(L):
            if not ctg_circ[c]: break
            j%=len(L)
        if s and L[j]==i: break                        # went all the way round
        out.append(L[j])
    win[gen]=(loc,out)
wg=np.array(sorted({j for _,o in win.values() for j in o}))
print(f"window genes {len(wg)}; windows of full length {sum(len(o)+l>=W for l,o in win.values())}",flush=True)

# --- RGP membership, names, products, proteins --------------------------------
rgp=set(h.root.RGP.read(field="gene")); in_rgp={int(j):g_id[j] in rgp for j in wg}
o=np.argsort(g_gd[wg]); pr=gd.read_coordinates(g_gd[wg][o],field="product")
prod={int(j):p.decode() for j,p in zip(wg[o],pr)}
gs_gene=A.geneSequences.read(field="gene"); gs_seq=A.geneSequences.read(field="seqid")
sid=gs_seq[pd.Index(gs_gene).get_indexer(g_id[wg])]; sid={int(j):int(s) for j,s in zip(wg,sid)}
useq=sorted(set(sid.values())); dna=A.sequences.read_coordinates(useq,field="dna")
prot={}
for s,d in zip(useq,dna):
    p=str(Seq(d.decode()).translate(table=11)).rstrip("*")
    prot[s]="M"+p[1:] if p else p
pid={}; plist=[]
for s in useq:
    if prot[s] not in pid: pid[prot[s]]=len(plist); plist.append(prot[s])
print(f"distinct proteins {len(plist)}",flush=True)

# --- genome labels ------------------------------------------------------------
mw=h.root.metadata.genomes.pangbank_wf.read(); ma=h.root.metadata.genomes.annotation_file.read()
lab={}
for r in mw:
    lab[r["ID"].decode()]=dict(org=r["ncbi_organism_name"].decode(),level=r["ncbi_assembly_level"].decode(),
                                strain=r["ncbi_strain_identifiers"].decode())
for r in ma:
    s=r["strain"].decode()
    if s and r["ID"].decode() in lab and not lab[r["ID"].decode()]["strain"]: lab[r["ID"].decode()]["strain"]=s
h.close()

fams=sorted({int(g_fam[j]) for j in wg})
out=dict(pangenome="11587 GTDB_refseq v2.0.0 (GTDB R232)",window=W,
    anchor=dict(dnaA=fam_names[FA],dnaN=fam_names[FN]),
    families={str(f):[fam_names[f],fam_part[f]] for f in fams},
    genomes=[dict(acc=gen,**lab.get(gen,{}),start=win[gen][0],circular=ctg_circ[int(g_ctg[win[gen][1][0]])] if win[gen][1] else False,
                  genes=[[g_id[j].decode(),int(g_fam[j]),int(in_rgp[int(j)]),pid[prot[sid[int(j)]]],
                          gname[j].decode(),prod[int(j)],strand[j].decode()] for j in win[gen][1]])
             for gen in genomes if gen in win],
    unanchored=[gen for gen in genomes if gen not in win])
json.dump(out,open("pgb/window.json","w"),separators=(",",":"))
json.dump(plist,open("pgb/window_prot.json","w"))
print(f"genomes {len(out['genomes'])}  unanchored {len(out['unanchored'])}  families {len(fams)}")
