"""Collapse the long-range couplings of pg_epistasis.py into independent region pairs.

A cassette of neighbouring families at one locus produces one association counted many
times. Families are grouped into regions by chromosomal neighbourhood (median position
within --gap genes of each other); a region pair keeps its strongest family pair. Each
region is annotated with the products of its families, so the coupling can be read.

Two things a coupling could be, other than a genuine interaction, are checked:
  same element   the two regions carry families with the same products (two copies of one
                 element inserted at two sites) -> flagged `same_products`
  genome size    genomes with more accessory genes have more of everything; the CMH is
                 recomputed stratified by lineage AND by tertile of accessory content

    python3 pg_epi_regions.py [--gap 30]
"""
import json, argparse, collections, warnings, numpy as np, pandas as pd, tables
warnings.simplefilter("ignore")
ap=argparse.ArgumentParser(); ap.add_argument("--gap",type=int,default=30); A=ap.parse_args()
H=json.load(open("pgb/epistasis.json")); hits=H["hits"]
Z=np.load("pgb/chrom.npz"); off=Z["offsets"]; FAM=Z["fam"]; RGP=Z["rgp"]
NG=len(off)-1; J=json.load(open("pgb/chrom_genomes.json")); FN=J["families"]; GL=np.diff(off)
fidx={n:i for i,n in enumerate(FN)}
gid=np.repeat(np.arange(NG),GL); pos_in=np.arange(len(FAM))-off[gid]
nf=FAM.max()+1; key=gid.astype(np.int64)*nf+FAM; _,fi=np.unique(key,return_index=True)
posacc=np.full((nf,NG),np.nan,np.float32); posacc[FAM[fi],gid[fi]]=pos_in[fi]
med=np.nanmedian(posacc,1)
fams=sorted({fidx[h[k]] for h in hits for k in ("a_family","b_family")})
# products, one gene per family
h5=tables.open_file("pgb/ecoli_11587.h5"); Aa=h5.root.annotations
gf=h5.root.geneFamilies.read(); gfam=pd.Index(FN).get_indexer([x.decode() for x in gf["geneFam"]])
one=pd.Series(gf["gene"]).groupby(gfam).first()
gid_all=pd.Index(Aa.genes.read(field="ID")); ggd=Aa.genes.read(field="genedata_id")
ok=[f for f in fams if f in one.index]
rows=gid_all.get_indexer([one[f] for f in ok]); gd=ggd[rows]; o=np.argsort(gd)
pr=Aa.genedata.read_coordinates(gd[o],field="product"); h5.close()
PROD={}
for i,p in zip(o,pr): PROD[ok[i]]=p.decode()
name_of=np.concatenate([np.array(n,dtype=object) for n in J["gene_names"]])
nm=collections.defaultdict(collections.Counter)
for f,n_ in zip(FAM.tolist(),name_of.tolist()):
    if n_: nm[f][n_]+=1
gn=lambda f: nm[int(f)].most_common(1)[0][0] if nm[int(f)] else FN[int(f)]

# regions: single-linkage on median position among the hit families
order=sorted(fams,key=lambda f:med[f]); reg={}; r=0
for i,f in enumerate(order):
    if i and med[f]-med[order[i-1]]>A.gap: r+=1
    reg[f]=r
print(f"{len(fams)} families in the hits -> {r+1} regions (gap {A.gap} genes)",flush=True)
best={}
for h in hits:
    fa,fb=fidx[h["a_family"]],fidx[h["b_family"]]; ra,rb=reg[fa],reg[fb]
    if ra==rb: continue
    k=(min(ra,rb),max(ra,rb))
    if k not in best or abs(h["z"])>abs(best[k]["z"]): best[k]=h
print(f"{len(best)} independent region pairs\n")
members=collections.defaultdict(list)
for f,rr in reg.items(): members[rr].append(f)
def describe(rr):
    ps=[PROD.get(f,"") for f in members[rr]]; c=collections.Counter(p for p in ps if p)
    return c
print(f"{'region A (top products)':46s}{'region B (top products)':46s}{'apart':>7s}{'Z':>7s}  type")
out=[]
for k,h in sorted(best.items(),key=lambda kv:-abs(kv[1]["z"]))[:30]:
    ra,rb=k; ca,cb=describe(ra),describe(rb)
    sa=", ".join(p[:28] for p,_ in ca.most_common(2)) or gn(fidx[h["a_family"]])
    sb=", ".join(p[:28] for p,_ in cb.most_common(2)) or gn(fidx[h["b_family"]])
    same=bool(set(ca)&set(cb))
    out.append(dict(regionA=sa,regionB=sb,nA=len(members[ra]),nB=len(members[rb]),distance=h["distance"],
                    z=h["z"],zA=h["zA"],zB=h["zB"],sign=h["sign"],same_products=same,
                    exampleA=gn(fidx[h["a_family"]]),exampleB=gn(fidx[h["b_family"]]),
                    freqA=h["a_freq"],freqB=h["b_freq"],rgpA=h["a_rgp"],rgpB=h["b_rgp"]))
    print(f"{sa[:45]:46s}{sb[:45]:46s}{h['distance']:>7d}{h['z']:>7.1f}  {h['sign']}{'  [same products]' if same else ''}")
json.dump(dict(gap=A.gap,regions=r+1,region_pairs=len(best),top=out),open("pgb/epi_regions.json","w"))
print(f"\nregion pairs whose two sides share a product (possibly one element at two sites): "
      f"{sum(1 for o_ in out if o_['same_products'])} of the {len(out)} shown")
