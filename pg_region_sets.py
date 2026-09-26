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
    python3 pg_region_sets.py --in pgb/epistasis_close.json --out pgb/region_sets_close.json

  close range: when the input comes from `pg_epistasis.py --maxdist` (pairs 50-500 genes apart), each link
  also gets `dmin` (the smallest distance among its family pairs, median positions) and `gap` (median over
  the genomes carrying both regions of the smallest number of genes between a member gene of one and of the
  other), and `tract` = min(dmin, gap) < 100 genes: possibly ONE transfer tract (or one element straddling
  two spots) rather than two regions that depend on each other; `same_spot` (median over its family pairs of
  the share of co-carriers where a copy of each sits in one panRGP spot) and `one_site` = same_spot >= 0.5:
  one element, or parts of one, that sits at different spots in different genomes. meta gains maxdist, dist,
  bands, bands_carriers, tract_links, one_site_links.
  Also in close mode: `expected_by_chance` is the permutation null of the selected pairs (sum of the bands'
  `null_beyond`; the 0.1 % global tail, which the uninformative pairs make up, is kept as
  `expected_by_chance_global_tail`) and `replicated_by_chance` the null of the replication filter; per link
  `site` = the consensus backbone flanks (left, right: the nearest single-copy >= 95 % family on each side of the
  element's block in its carriers) of each region, and for an avoidance link `site_competition` = the two share a
  flank on the same side: two elements competing for ONE insertion site (they exclude each other physically,
  not through a coupling); with an input made with --sep, `sep` (median over the family pairs of the share of
  co-carriers with >= K backbone genes between them), `alone` (min over the pairs of the genomes carrying one
  family without the other) and `geometry` (carriers / median: whether the pair's geometry was read in >= 10
  co-carriers). meta gains halves (lineage-disjoint replication or not) and sep.
  Without such an input the output is exactly as before.
"""
import json, re, argparse, collections, warnings, numpy as np
warnings.simplefilter("ignore")
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default="pgb/epistasis.json")
ap.add_argument("--out", default="pgb/region_sets.json")
A = ap.parse_args()
E = json.load(open(A.inp))
CLOSE = E.get("maxdist") is not None
TRACT = 100                                            # genes: below this, possibly one transfer tract
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
if CLOSE:                                              # consensus backbone flanks of a region's element (its insertion site)
    _uk, _cnt = np.unique(GID.astype(np.int64) * len(FN) + FAM, return_counts=True)
    BB = np.bincount((_uk % len(FN))[_cnt == 1], minlength=len(FN)) >= 0.95 * NG
    _site = {}
    def site_of(r_):
        """(left, right): the family of the nearest backbone gene on each side of the region's largest block (genes of
        its families <= 30 apart) in each carrier (>= half of its families), most common over carriers (>= 50 %
        of them, else None)"""
        if r_ in _site: return _site[r_]
        fs = members[keys[r_]]; byg = collections.defaultdict(list); nf_ = collections.Counter()
        for f in fs:
            i0, i1 = np.searchsorted(sf, f), np.searchsorted(sf, f, "right"); idx = order[i0:i1]
            for gg in np.unique(GID[idx]).tolist(): nf_[gg] += 1
            for gg, lc in zip(GID[idx].tolist(), LOC[idx].tolist()): byg[gg].append(lc)
        lc_, rc_ = collections.Counter(), collections.Counter(); n_ = 0
        for gg, c in nf_.items():
            if c / len(fs) < 0.5: continue
            ps = sorted(byg[gg]); cut = [0] + [i for i in range(1, len(ps)) if ps[i] - ps[i - 1] > 30] + [len(ps)]
            blk = max((ps[cut[k]:cut[k + 1]] for k in range(len(cut) - 1)), key=len)
            sl = FAM[OFF[gg]:OFF[gg + 1]]; Lg = len(sl); n_ += 1
            for side, p0, step in ((lc_, blk[0], -1), (rc_, blk[-1], 1)):
                p = p0
                for _ in range(200):
                    p = (p + step) % Lg
                    if BB[sl[p]]: side[int(sl[p])] += 1; break
        pick = lambda c: (FN[c.most_common(1)[0][0]] if c and c.most_common(1)[0][1] >= 0.5 * n_ else None)
        _site[r_] = (pick(lc_), pick(rc_))
        return _site[r_]
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
    if CLOSE:                                          # how far apart are the two elements in the genomes carrying both?
        loc = []
        for r_ in (a, b):
            fs = members[keys[r_]]; byg = collections.defaultdict(list); nf = collections.Counter()
            for f in fs:
                i0, i1 = np.searchsorted(sf, f), np.searchsorted(sf, f, "right"); idx = order[i0:i1]
                for gg in np.unique(GID[idx]).tolist(): nf[gg] += 1
                for gg, lc in zip(GID[idx].tolist(), LOC[idx].tolist()): byg[gg].append(lc)
            loc.append((byg, {gg for gg, c in nf.items() if c / len(fs) >= 0.5}))
        both = sorted(loc[0][1] & loc[1][1]); gaps = []
        for gg in both:
            dd = np.abs(np.array(loc[0][0][gg])[:, None] - np.array(loc[1][0][gg])[None]); dd = np.minimum(dd, GL[gg] - dd)
            gaps.append(int(dd.min()) - 1)
        dmin = min(h["distance"] for h in hs); gap = float(np.median(gaps)) if gaps else None
        links[-1].update(dmin=dmin, gap=gap, both=len(both), tract=bool(dmin < TRACT or (gap is not None and gap < TRACT)))
        if "same_spot" in hs[0]:                       # per family pair, from pg_epistasis.py --maxdist
            ss = float(np.median([h["same_spot"] for h in hs]))
            links[-1].update(same_spot=round(ss, 2), one_site=bool(ss >= 0.5))
        if "sep" in hs[0]:                             # from pg_epistasis.py --sep: two separate insertions?
            links[-1].update(sep=round(float(np.median([h["sep"] for h in hs])), 2),
                             alone=int(min(min(h["alone_a"], h["alone_b"]) for h in hs)),
                             geometry="carriers" if any(h["geometry"] == "carriers" for h in hs) else "median")
        fa, fb = site_of(a), site_of(b)
        links[-1]["site"] = [fa, fb]
        if links[-1]["sign"] < 0:                      # one insertion site for both: they exclude each other physically
            links[-1]["site_competition"] = bool((fa[0] is not None and fa[0] == fb[0]) or (fa[1] is not None and fa[1] == fb[1]))

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
    if CLOSE:
        co_ = [l for l in inner if l["sign"] > 0]
        sets[-1]["tract"] = round(sum(l["tract"] for l in co_) / max(1, len(co_)), 2)
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
if CLOSE:
    bnd = E.get("bands") or []
    out["meta"].update(method=out["meta"]["method"] + f"; pairs {E['mindist']}-{E['maxdist']} genes apart in different spots"
                       + ("" if E.get("halves") == "lineage" else " (replication halves: halves of the lineage x content STRATA, "
                          "which share lineages)"),
                       maxdist=E["maxdist"], expected_by_chance=round(sum(b["null_beyond"] for b in bnd), 1) if bnd else None,
                       expected_by_chance_global_tail=round(E["far"] * E.get("tail", 0.001), 1),
                       replicated_by_chance=round(sum(b["null_replicated"] for b in bnd), 2) if bnd else None,
                       halves=E.get("halves", "strata"), sep=E.get("sep"),
                       site_competition_links=sum(l.get("site_competition", False) for l in links),
                       separate_links=sum(1 for l in links if l.get("geometry") == "carriers"),
                       bands=E.get("bands"),
                       tract_links=sum(l["tract"] for l in links), tract_below=TRACT, dist=E.get("dist", "median"),
                       one_site_links=sum(l.get("one_site", False) for l in links), bands_carriers=E.get("bands_carriers"))
# sanity: a link joins two different elements, at two different spots (or a spot and a loose group elsewhere)
bad = [l for l in links if l["a"] == l["b"] or (regions[l["a"]]["spot"] is not None and regions[l["a"]]["spot"] == regions[l["b"]]["spot"])
       or (regions[l["a"]]["spot"] is None and regions[l["b"]]["spot"] is None)]
print(f"links joining the same element / the same spot / two loose groups: {len(bad)}")
json.dump(out, open(A.out, "w"), separators=(",", ":"))
print(f"{len(hits)} family pairs -> {len(regions)} regions ({sum(r['spot'] is not None for r in regions)} spots), {len(links)} links "
      f"({sum(l['sign'] > 0 for l in links)} co-occurrence, {sum(l['sign'] < 0 for l in links)} avoidance), {len(sets)} sets of >= 2 regions")
for s in sets[:12]:
    print(f"  set {s['id']}: {len(s['regions'])} regions, {s['links']} links / {s['pairs']} pairs, |Z| max {s['zmax']}, shared-product links {s['shared']:.0%} | "
          + " ; ".join(f"{regions[i]['label'][:22]} @{regions[i]['center']}" for i in s["regions"][:5]))
if CLOSE:
    tl = [l for l in links if l["tract"]]
    print(f"close range: {len(tl)} of {len(links)} links possibly one transfer tract (dmin or median gap in carriers < {TRACT} genes); "
          f"{sum(l.get('one_site', False) for l in links)} whose family pairs sit in ONE spot in most genomes carrying both; "
          f"{sum(bool(l['shared']) for l in links)} with a shared specific product; "
          f"{sum(l.get('site_competition', False) for l in links)} avoidance links whose two elements share an insertion site; "
          f"geometry read in co-carriers for {sum(1 for l in links if l.get('geometry') == 'carriers')}; "
          f"expected by chance {out['meta']['expected_by_chance']} beyond the null, {out['meta']['replicated_by_chance']} replicated")
import os; print(f"{A.out} {os.path.getsize(A.out) / 1e3:.0f} kB")
