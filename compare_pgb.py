"""Do the model's confident loci coincide with PPanGGOLiN's persistent genome?

My walk is 80 genes from the dnaA locus of O157:H7 Sakai. PanGBank's E. coli
pangenome (11587, GTDB_refseq v2.0.0, 2,002 genomes) labels every gene of that
same chromosome persistent / shell / cloud, and marks the regions of genomic
plasticity. Joining the two on genomic coordinate tests the prediction made at
the outset: the model should be certain exactly where the pangenome is persistent,
and lost where it is not.
"""
import json, collections, numpy as np
from Bio import SeqIO

# --- PanGBank side ---------------------------------------------------------
cg=json.load(open("pgb/cgview_sakai.json"))["cgview"]
part={}; rgp=[]
for f in cg["features"]:
    if f.get("source")=="Gene" and f.get("legend") in ("persistent","shell","cloud"):
        part[int(f["start"])]=f["legend"]
    if f.get("source")=="RGP" or f.get("legend")=="RGP":
        rgp.append((int(f["start"]),int(f["stop"])))
print(f"PanGBank : {len(part)} gènes partitionnés, {len(rgp)} RGP sur le chromosome")

# --- my walk ---------------------------------------------------------------
rec=max(SeqIO.parse("eco/GCF_000008865.2.gbff","genbank"),key=lambda r:len(r.seq))
cds=[]
for f in rec.features:
    if f.type!="CDS" or not f.qualifiers.get("translation"): continue
    cds.append(dict(start=int(f.location.start),stop=int(f.location.end),
        gene=(f.qualifiers.get("gene") or [None])[0]))
cds.sort(key=lambda c:c["start"])
i=next(j for j,c in enumerate(cds) if c["gene"]=="dnaA")
rot=[cds[i]]+cds[:i][::-1]+cds[i+1:][::-1]        # canonical orientation (strand -)

G=json.load(open("eco_graph.json"))
lane=next(l for l in G["lanes"] if "Sakai" in l["strain"])
steps={s["locus"]:s for s in lane["steps"]}

def part_at(c):
    for d in (0,1,-1,2,-2):
        if c["start"]+d in part: return part[c["start"]+d]
    return None
def in_rgp(c):
    return any(a<=c["start"]<=b or a<=c["stop"]<=b for a,b in rgp)

rows=[]
for k in range(1,min(80,len(rot))):
    s=steps.get(k)
    if not s: continue
    c=rot[k]
    rows.append(dict(locus=k,gene=c["gene"],part=part_at(c),rgp=in_rgp(c),
                     hit=s["hit"],rank=s["rank"],ent=s.get("entropy")))
matched=[r for r in rows if r["part"]]
print(f"joints sur coordonnée : {len(matched)}/{len(rows)}\n")

print("=== justesse du modèle selon la partition PPanGGOLiN ===")
print(f"{'partition':12s} {'n':>4s} {'top-1':>7s} {'rang médian':>12s} {'entropie':>9s}")
for p in ("persistent","shell","cloud"):
    sel=[r for r in matched if r["part"]==p]
    if not sel: continue
    print(f"{p:12s} {len(sel):4d} {np.mean([r['hit'] for r in sel])*100:6.1f}% "
          f"{int(np.median([r['rank'] for r in sel])):12d} "
          f"{np.mean([r['ent'] for r in sel if r['ent'] is not None]):8.2f}")
print("\n=== dans une RGP ou non ===")
for lab,sel in (("hors RGP",[r for r in matched if not r["rgp"]]),
                ("dans RGP",[r for r in matched if r["rgp"]])):
    if not sel: continue
    print(f"{lab:12s} {len(sel):4d} {np.mean([r['hit'] for r in sel])*100:6.1f}% "
          f"{int(np.median([r['rank'] for r in sel])):12d} "
          f"{np.mean([r['ent'] for r in sel if r['ent'] is not None]):8.2f}")
print("\n=== les 26 premiers loci ===")
print(f"{'loc':>3s} {'gène':10s} {'PPanGGOLiN':11s} {'RGP':4s} {'modèle':7s} {'rang':>6s}")
for r in rows[:26]:
    print(f"{r['locus']:3d} {str(r['gene'])[:10]:10s} {str(r['part']):11s} "
          f"{'oui' if r['rgp'] else '-':4s} {'juste' if r['hit'] else 'raté':7s} {r['rank']:6d}")
json.dump(rows,open("pgb_compare.json","w"))
