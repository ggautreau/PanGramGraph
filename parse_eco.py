"""Parse E. coli GBFF into an ordered protein list anchored on the dnaA locus.

Three corrections the earlier passes exposed.

ANCHOR. RefSeq numbers E. coli K-12 from thrL, not from the replication origin, so
walking from the record start walks from an arbitrary point. We anchor on dnaA.

PSEUDOGENES. Two of eleven RefSeq assemblies (CFT073, Nissle 1917) annotate dnaA as
a pseudogene, so those records carry no DnaA protein at all. We therefore locate the
anchor by the *gene* feature's coordinate - present either way - and start the walk
at the first translated CDS from there, which for those two is dnaN.

ORIENTATION. Assemblies are not deposited in a consistent strand orientation:
MG1655 and W3110 run opposite ways. We orient every genome so the walk reads
dnaA -> dnaN -> recF -> gyrB, the canonical origin cassette.
"""
import json, glob, os
from Bio import SeqIO

def anchor_coord(rec):
    best=None
    for f in rec.features:
        if f.type not in ("gene","CDS"): continue
        g=(f.qualifiers.get("gene") or [""])[0]
        p=(f.qualifiers.get("product") or [""])[0].lower()
        hit = g=="dnaA" or ("chromosomal replication init" in p
                            and "regulat" not in p and "inhibit" not in p)
        if hit:
            c=int(f.location.start)
            how="gene tag" if g=="dnaA" else "product"
            pseudo = "pseudo" in f.qualifiers
            if best is None or (best[2] and not pseudo): best=(c,how,pseudo)
    return best

out=[]; manifest=json.load(open("eco/manifest.json"))
for fp in sorted(glob.glob("eco/*.gbff")):
    acc=os.path.basename(fp).replace(".gbff","")
    recs=sorted(SeqIO.parse(fp,"genbank"),key=lambda r:-len(r.seq))
    chrom=recs[0]; cds=[]
    for f in chrom.features:
        if f.type!="CDS": continue
        t=f.qualifiers.get("translation",[None])[0]
        if not t: continue
        cds.append(dict(start=int(f.location.start),
            gene=(f.qualifiers.get("gene") or [None])[0],
            product=(f.qualifiers.get("product") or [""])[0],
            locus=(f.qualifiers.get("locus_tag") or [""])[0], prot=t))
    cds.sort(key=lambda c:c["start"])
    a=anchor_coord(chrom)
    if a is None: print(f"{acc}  NO dnaA locus - skipped",flush=True); continue
    coord,how,pseudo=a
    i=min(range(len(cds)),key=lambda j:abs(cds[j]["start"]-coord))
    fwd=cds[i:]+cds[:i]
    rev=[cds[i]]+cds[:i][::-1]+cds[i+1:][::-1]
    def score(seq):
        g=[c["gene"] for c in seq[:9]]
        return sum(w for w,n in ((3,"dnaN"),(2,"recF"),(2,"gyrB"),(1,"gyrA")) if n in g)
    sf,sr=score(fwd),score(rev)
    rot,orient=(fwd,"+") if sf>=sr else (rev,"-")
    out.append(dict(acc=acc,desc=manifest.get(acc,acc),replicon=chrom.id,n_cds=len(cds),
        anchor=how,dnaA_pseudo=bool(pseudo),orientation=orient,
        genes=[c["gene"] for c in rot[:400]],products=[c["product"] for c in rot[:400]],
        proteins=[c["prot"] for c in rot[:400]]))
    tag=" dnaA=PSEUDO" if pseudo else ""
    print(f"{acc} {len(cds):5d} CDS  anchor@{coord:7d} ({how}){tag}  strand {orient} "
          f"(+{sf}/-{sr})  {[c['gene'] for c in rot[:5]]}  {manifest.get(acc,'')}",flush=True)
json.dump(out,open("eco_parsed.json","w"))
print("\nparsed",len(out),"genomes")
