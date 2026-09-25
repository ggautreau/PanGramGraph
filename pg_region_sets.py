"""Sets of distant regions whose accessory genes go together (or exclude each other) beyond
descent, as data for the "Coupled regions" card of the PanGramGraph page.

Input: pgb/epistasis.json from `pg_epistasis.py --ctrl_content`: pairs of accessory families,
more than 500 genes apart and in different insertion spots, whose presence is associated within
lineages AND within tertiles of accessory content (Cochran-Mantel-Haenszel, one genome = one
observation), beyond the 99.9th percentile of a within-stratum permutation null, and replicated
in two disjoint halves of the lineages.

  region    an element at an insertion spot: the families of one panRGP spot (where a family most often
            sits in the 540 complete chromosomes) split into cassettes that come and go together
            (Jaccard distance of their presence across genomes, average linkage, cut at 0.5), since
            a hotspot holds alternative elements. A family outside every spot joins those within 10
            genes of it, split the same way.
  link      two regions joined by at least one such family pair: its sign is the majority's, its
            strength the largest |Z| and the number of family pairs behind it.
  set       a community of the co-occurrence links (Louvain, weighted by family pairs): connected
            components chain nearly everything through weak links. Avoidance links are kept,
            within and between sets.
  shared    a link whose two regions carry families with the same specific product (transposase,
            phage tail, integrase...): possibly one mobile element inserted at two sites rather than
            two regions that depend on each other.
  presence  for each region and each of the 540 complete genomes, the share of the region's coupled
            families it carries; genomes in cgMLST order (average linkage), lineage clusters marked.

    python3 pg_region_sets.py        # -> pgb/region_sets.json
"""
import json, re, collections, warnings, numpy as np
warnings.simplefilter("ignore")
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform

E = json.load(open("pgb/epistasis.json"))
assert E.get("ctrl_content"), "run pg_epistasis.py --ctrl_content first"
Z = np.load("pgb/chrom.npz"); OFF = Z["offsets"]; FAM = Z["fam"]; SPOT = Z["spot"]; RGP = Z["rgp"]
NG = len(OFF) - 1; GL = np.diff(OFF); GID = np.repeat(np.arange(NG), GL); LOC = np.arange(len(FAM)) - OFF[GID]
CJ = json.load(open("pgb/chrom_genomes.json")); FN = CJ["families"]; FIDX = {n: i for i, n in enumerate(FN)}
NAME = np.array([x or "" for g in CJ["gene_names"] for x in g], dtype=object)
FI = json.load(open("pgb/fam_info.json")); PRODUCT = FI["product"]

hits = E["hits"]
fams = sorted({FIDX[h[k]] for h in hits for k in ("a_family", "b_family")})
order = np.argsort(FAM, kind="stable"); sf = FAM[order]
info = {}
for f in fams:
    a, b = np.searchsorted(sf, f), np.searchsorted(sf, f, "right"); idx = order[a:b]
    sp = collections.Counter(SPOT[idx].tolist()).most_common(1)[0][0]
    nm = collections.Counter(n for n in NAME[idx] if n)
    info[f] = dict(spot=int(sp), pos=float(np.median(LOC[idx])), genomes=np.unique(GID[idx]), rgp=float(RGP[idx].mean()),
                   name=nm.most_common(1)[0][0] if nm else "", product=PRODUCT[f])

# --- regions: spots, and families outside spots grouped by position
reg_of = {}; members = collections.defaultdict(list)
for f in fams:
    if info[f]["spot"] >= 0: reg_of[f] = ("s", info[f]["spot"])
loose = sorted((f for f in fams if info[f]["spot"] < 0), key=lambda f: info[f]["pos"]); k = -1
for i, f in enumerate(loose):
    if i == 0 or info[f]["pos"] - info[loose[i - 1]]["pos"] > 10: k += 1
    reg_of[f] = ("p", k)
for f, r in reg_of.items(): members[r].append(f)
split = collections.defaultdict(list)                  # a spot's alternative elements, apart
for r, fs in members.items():
    if len(fs) == 1: split[r + (0,)] = fs; continue
    P = np.zeros((len(fs), NG), bool)
    for i, f in enumerate(fs): P[i, info[f]["genomes"]] = True
    inter = P.astype(np.float32) @ P.T.astype(np.float32); cnt = P.sum(1)
    J = 1 - inter / np.maximum(cnt[:, None] + cnt[None] - inter, 1)
    lab = fcluster(linkage(squareform(np.clip((J + J.T) / 2, 0, 1), checks=False), "average"), 0.5, "distance")
    for f, c in zip(fs, lab): split[r + (int(c),)].append(f)
members = split; reg_of = {f: r for r, fs in members.items() for f in fs}
keys = sorted(members, key=lambda r: np.median([info[f]["pos"] for f in members[r]]))
RID = {r: i for i, r in enumerate(keys)}

GENERIC = re.compile(r"^(hypothetical protein|.*domain-containing protein|DUF\d+.*|.*family protein|putative protein|protein)$", re.I)
MOBILE = re.compile(r"transpos|integrase|recombinase|phage|prophage|insertion|IS\d|terminase|capsid|portal|tail|excisionase|resolvase", re.I)
def specific(p): return bool(p) and not GENERIC.match(p.strip())

# presence: genomes in cgMLST order, lineage clusters as in pg_epistasis.py (cut 0.4)
D2 = np.load("pgb/cgmlst_dist.npy").astype(np.float32); acc = json.load(open("pgb/cgmlst_loci.json"))["genomes"]
ix = {a: i for i, a in enumerate(acc)}; sel = np.array([ix[g["acc"]] for g in CJ["genomes"]]); DM = D2[np.ix_(sel, sel)]
Lk = linkage(squareform(np.maximum(DM, DM.T), checks=False), "average"); leaf = leaves_list(Lk)
cl = fcluster(Lk, E["cut"], "distance"); rank = np.empty(NG, int); rank[leaf] = np.arange(NG)

regions = []
for r in keys:
    fs = members[r]; P = np.zeros((len(fs), NG), bool)
    for i, f in enumerate(fs): P[i, info[f]["genomes"]] = True
    share = P.mean(0)
    names = collections.Counter(info[f]["name"] for f in fs if info[f]["name"])
    prods = collections.Counter(info[f]["product"] for f in fs if specific(info[f]["product"]))
    pos = [info[f]["pos"] for f in fs]
    lab = ", ".join(n for n, _ in names.most_common(3)) or (prods.most_common(1)[0][0] if prods else "hypothetical proteins")
    regions.append(dict(id=RID[r], spot=r[1] if r[0] == "s" else None, element=r[2], start=round(min(pos)), end=round(max(pos)), center=round(float(np.median(pos))),
                        n=len(fs), label=lab, genes=[n for n, _ in names.most_common(8)], products=[p for p, _ in prods.most_common(4)],
                        mobile=round(sum(bool(MOBILE.search(info[f]["product"] or "")) for f in fs) / len(fs), 2),
                        freq=round(float((share >= 0.5).mean()), 3), rgp=round(float(np.mean([info[f]["rgp"] for f in fs])), 2),
                        presence="".join(str(min(9, int(v * 9.999))) for v in share[leaf]),
                        families=[FN[f] for f in fs]))

# --- links between regions
L = collections.defaultdict(list)
for h in hits:
    a, b = RID[reg_of[FIDX[h["a_family"]]]], RID[reg_of[FIDX[h["b_family"]]]]
    if a != b: L[(min(a, b), max(a, b))].append(h)
prodset = {r["id"]: {info[f]["product"] for f in members[keys[r["id"]]] if specific(info[f]["product"])} for r in regions}
links = []
for (a, b), hs in L.items():
    pos = sum(h["z"] > 0 for h in hs); best = max(hs, key=lambda h: abs(h["z"]))
    shared = sorted(prodset[a] & prodset[b])
    links.append(dict(a=a, b=b, sign=1 if pos * 2 >= len(hs) else -1, pairs=len(hs), pos=pos, neg=len(hs) - pos,
                      z=best["z"], zA=best["zA"], zB=best["zB"], best=[best["a"], best["b"]], distance=best["distance"],
                      shared=shared[:3]))

# --- sets: communities of the co-occurrence links
import networkx as nx
from networkx.algorithms.community import louvain_communities
G = nx.Graph(); G.add_nodes_from(r["id"] for r in regions)
for l in links:
    if l["sign"] > 0: G.add_edge(l["a"], l["b"], w=l["pos"])
comm = louvain_communities(G, weight="w", seed=0)
sets = []
for rs in comm:
    rs = list(rs)
    if len(rs) < 2: continue
    inner = [l for l in links if l["a"] in rs and l["b"] in rs]
    if not any(l["sign"] > 0 for l in inner): continue
    sets.append(dict(regions=sorted(rs, key=lambda i: regions[i]["center"]), links=len([l for l in inner if l["sign"] > 0]),
                     avoid=len([l for l in inner if l["sign"] < 0]), pairs=sum(l["pos"] for l in inner),
                     zmax=round(max(abs(l["z"]) for l in inner if l["sign"] > 0), 2),
                     shared=round(sum(bool(l["shared"]) for l in inner if l["sign"] > 0) / max(1, len([l for l in inner if l["sign"] > 0])), 2),
                     families=sum(regions[i]["n"] for i in rs)))
sets.sort(key=lambda s: (-s["pairs"], -s["zmax"]))
for i, s in enumerate(sets):
    s["id"] = i
    for r in s["regions"]: regions[r]["set"] = i
for r in regions: r.setdefault("set", None)

bounds = [int(i) for i in np.flatnonzero(np.diff(cl[leaf]) != 0) + 1]
out = dict(meta=dict(method="CMH within lineage x accessory-content strata, 99.9th pct of a within-stratum permutation null, replicated in two disjoint lineage halves",
                     cut=E["cut"], mindist=E["mindist"], strata=E["clusters"], genomes_tested=E["genomes"], families_tested=E["families"],
                     null_999=round(E["null_999"], 2), pairs_far=E["far"], expected_by_chance=round(E["far"] * 0.001, 1), pairs=len(hits),
                     regions=len(regions), links=len(links), sets=len(sets), genomes=NG, chrom_len=int(np.median(GL))),
           genomes=[dict(acc=CJ["genomes"][g]["acc"], strain=CJ["genomes"][g].get("strain", ""), cl=int(cl[g])) for g in leaf],
           bounds=bounds, regions=regions, links=links, sets=sets)
json.dump(out, open("pgb/region_sets.json", "w"), separators=(",", ":"))
print(f"{len(hits)} family pairs -> {len(regions)} regions ({sum(r['spot'] is not None for r in regions)} spots), {len(links)} links "
      f"({sum(l['sign'] > 0 for l in links)} co-occurrence, {sum(l['sign'] < 0 for l in links)} avoidance), {len(sets)} sets of >= 2 regions")
for s in sets[:12]:
    print(f"  set {s['id']}: {len(s['regions'])} regions, {s['links']} links / {s['pairs']} pairs, |Z| max {s['zmax']}, shared-product links {s['shared']:.0%} | "
          + " ; ".join(f"{regions[i]['label'][:22]} @{regions[i]['center']}" for i in s["regions"][:5]))
import os; print(f"pgb/region_sets.json {os.path.getsize('pgb/region_sets.json') / 1e3:.0f} kB")
