"""Flags for every link of a sets file (pgb/region_sets.json format): what kind of object each "coupling" is,
read in the genomes, without the model. They go next to the model's verdict on each arc (pg_knockout.py --flags).

    python3 pg_link_flags.py                                                   # -> pgb/link_flags.json
    python3 pg_link_flags.py --sets pgb/region_sets_close.json --epi pgb/epistasis_close_sep.json \
                             --out pgb/link_flags_close.json

Per family pair behind a link (the hits of --epi that join its two regions), in the genomes carrying BOTH families
(all 540 complete chromosomes, every copy):
  carriers    how many genomes carry both
  near_t      share of them whose nearest copies are <= t genes apart (t = 10, 50, 100, 500; the short way round)
  same_spot   share of them where a copy of each sits in one panRGP spot, < 200 genes apart (as pg_epistasis.py)
  sep         share of them with >= --sep backbone genes (families single-copy in >= 95 % of the genomes) between
              every copy of one family and every copy of the other: two separate insertions, not one element
  dist        median nearest-copy distance over them
  alone_a/b   genomes carrying one family and not the other
Per link: the median of these over its family pairs, and
  one_site          same_spot >= 0.5: one element (or parts of one) that sits at different spots in different
                    genomes, not two coupled regions (pg_region_sets.py's definition)
  tract             carrier distance < 100 genes: possibly one transfer tract
  separate          sep >= 0.5 and each family found alone in >= 3 genomes (geometry read in >= 10 co-carriers)
  site_competition  avoidance link whose two elements share a consensus backbone flank on the same side: they compete
                    for one insertion site (pg_region_sets.py close mode computes the same)
  rep_lineage       the link's family pairs replicated in two LINEAGE-disjoint halves: --lin (the same test run with
                    pg_epistasis.py --halves lineage) lists the pairs that pass; pairs = how many of the link's pairs
                    do. The default replication of pg_epistasis.py --ctrl_content halves the lineage x content strata,
                    whose halves share lineages (6 of 8 lineage clusters, 172 of 185 genomes), so it does not show that
                    a link holds in independent lineages.
Reads the project's data; writes only --out.
"""
import json, argparse, collections, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--sets", default="pgb/region_sets.json")
ap.add_argument("--epi", default="pgb/epistasis.json", help="the pg_epistasis.py output the sets file was built from")
ap.add_argument("--lin", default=None, help="pg_epistasis.py --halves lineage output over the same pairs (default: --epi if it "
                                            "was made with --halves lineage, else pgb/epistasis_linhalves.json)")
ap.add_argument("--sep", type=int, default=10)
ap.add_argument("--out", default="pgb/link_flags.json")
A = ap.parse_args()
RS = json.load(open(A.sets)); E = json.load(open(A.epi))
LIN = A.lin or (A.epi if E.get("halves") == "lineage" else "pgb/epistasis_linhalves.json")
EL = json.load(open(LIN)); assert EL.get("halves") == "lineage", f"{LIN} was not made with --halves lineage"
Z = np.load("pgb/chrom.npz"); OFF = Z["offsets"]; FAM = Z["fam"]; SPOT = Z["spot"]
NG = len(OFF) - 1; GL = np.diff(OFF); GID = np.repeat(np.arange(NG), GL); LOC = np.arange(len(FAM)) - OFF[GID]
CJ = json.load(open("pgb/chrom_genomes.json")); FN = CJ["families"]; FIDX = {n: i for i, n in enumerate(FN)}
uk, cnt = np.unique(GID.astype(np.int64) * len(FN) + FAM, return_counts=True)
BB = np.bincount((uk % len(FN))[cnt == 1], minlength=len(FN)) >= 0.95 * NG
CB = [np.cumsum(BB[FAM[OFF[g]:OFF[g + 1]]]) for g in range(NG)]              # inclusive backbone count per genome
order = np.argsort(FAM, kind="stable"); sf = FAM[order]


def copies(f):
    i0, i1 = np.searchsorted(sf, f), np.searchsorted(sf, f, "right"); idx = order[i0:i1]
    d = collections.defaultdict(list)
    for g, p in zip(GID[idx].tolist(), LOC[idx].tolist()): d[g].append(p)
    return d


_cp = {}
def cp(f):
    if f not in _cp: _cp[f] = copies(f)
    return _cp[f]


def between_bb(g, p, q):
    """backbone genes strictly between positions p and q of genome g, the short way round"""
    C = CB[g]; L = int(GL[g]); lo, hi = min(p, q), max(p, q)
    b = lambda i: int(BB[FAM[OFF[g] + i]])
    lin = C[hi] - b(hi) - C[lo]; wrap = C[-1] - C[hi] + C[lo] - b(lo)
    return int(lin if hi - lo <= L - (hi - lo) else wrap)


def pair_geometry(fa, fb):
    A_, B_ = cp(fa), cp(fb); both = sorted(set(A_) & set(B_))
    near = {t: 0 for t in (10, 50, 100, 500)}; same = sep = 0; dists = []
    for g in both:
        L = int(GL[g]); pa = np.array(A_[g]); pb = np.array(B_[g])
        dd = np.abs(pa[:, None] - pb[None]); dd = np.minimum(dd, L - dd); m = int(dd.min()); dists.append(m)
        for t in near: near[t] += m <= t
        sa = SPOT[OFF[g] + pa]; sb = SPOT[OFF[g] + pb]
        same += bool(((sa[:, None] == sb[None]) & (sa[:, None] >= 0) & (dd < 200)).any())
        sep += min(between_bb(g, int(p), int(q)) for p in pa for q in pb) >= A.sep
    n = len(both)
    return dict(carriers=n, **{f"near{t}": near[t] / max(n, 1) for t in near}, same_spot=same / max(n, 1),
                sep=sep / max(n, 1), dist=float(np.median(dists)) if dists else None,
                alone_a=len(set(A_) - set(B_)), alone_b=len(set(B_) - set(A_)))


# consensus backbone flanks of a region (as pg_region_sets.py close mode)
def site_of(fs):
    byg = collections.defaultdict(list); nf_ = collections.Counter()
    for f in fs:
        for g, ps in cp(f).items(): nf_[g] += 1; byg[g] += ps
    lc, rc = collections.Counter(), collections.Counter(); n_ = 0
    for g, c in nf_.items():
        if c / len(fs) < 0.5: continue
        ps = sorted(byg[g]); cut = [0] + [i for i in range(1, len(ps)) if ps[i] - ps[i - 1] > 30] + [len(ps)]
        blk = max((ps[cut[k]:cut[k + 1]] for k in range(len(cut) - 1)), key=len)
        sl = FAM[OFF[g]:OFF[g + 1]]; Lg = len(sl); n_ += 1
        for side, p0, step in ((lc, blk[0], -1), (rc, blk[-1], 1)):
            p = p0
            for _ in range(200):
                p = (p + step) % Lg
                if BB[sl[p]]: side[int(sl[p])] += 1; break
    pick = lambda c: (FN[c.most_common(1)[0][0]] if c and c.most_common(1)[0][1] >= 0.5 * n_ else None)
    return pick(lc), pick(rc)


REG = RS["regions"]; FAMS = [set(r["families"]) for r in REG]
lin_hits = {(h["a_family"], h["b_family"], h["sign"]) for h in EL["hits"]}
lin_hits |= {(b, a, s) for a, b, s in lin_hits}
out = []
for li, l in enumerate(RS["links"]):
    a, b = l["a"], l["b"]
    hs = [h for h in E["hits"] if (h["a_family"] in FAMS[a] and h["b_family"] in FAMS[b]) or
          (h["a_family"] in FAMS[b] and h["b_family"] in FAMS[a])]
    assert len(hs) == l["pairs"], f"link {li}: {len(hs)} family pairs found in {A.epi}, the sets file says {l['pairs']}"
    geo = [pair_geometry(FIDX[h["a_family"]], FIDX[h["b_family"]]) for h in hs]
    med = lambda k: float(np.median([x[k] for x in geo if x[k] is not None])) if any(x[k] is not None for x in geo) else None
    nrep = sum((h["a_family"], h["b_family"], h["sign"]) in lin_hits for h in hs)
    rec = dict(li=li, a=a, b=b, sign=l["sign"], pairs=len(hs), carriers=med("carriers"),
               carrier_dist=med("dist"), near100=med("near100"), same_spot=med("same_spot"), sep=med("sep"),
               alone=int(min(min(x["alone_a"], x["alone_b"]) for x in geo)),
               rep_lineage=nrep > 0, rep_lineage_pairs=nrep)
    rec["one_site"] = bool(rec["same_spot"] is not None and rec["same_spot"] >= 0.5)
    rec["tract"] = bool(rec["carrier_dist"] is not None and rec["carriers"] >= 10 and rec["carrier_dist"] < 100)
    rec["separate"] = bool(rec["carriers"] is not None and rec["carriers"] >= 10 and rec["sep"] >= 0.5 and rec["alone"] >= 3)
    fa, fb = site_of(sorted(FIDX[f] for f in REG[a]["families"])), site_of(sorted(FIDX[f] for f in REG[b]["families"]))
    rec["site"] = [fa, fb]
    rec["site_competition"] = bool(l["sign"] < 0 and ((fa[0] is not None and fa[0] == fb[0]) or (fa[1] is not None and fa[1] == fb[1])))
    out.append(rec)
    if li % 100 == 0: print(f"link {li}/{len(RS['links'])}", flush=True)
co = [r for r in out if r["sign"] > 0]; av = [r for r in out if r["sign"] < 0]
summ = dict(sets=A.sets, epi=A.epi, lin=LIN, sep=A.sep, links=len(out),
            one_site=sum(r["one_site"] for r in out), one_site_co=sum(r["one_site"] for r in co),
            tract=sum(r["tract"] for r in out), separate=sum(r["separate"] for r in out),
            site_competition=sum(r["site_competition"] for r in out),
            rep_lineage=sum(r["rep_lineage"] for r in out), rep_lineage_co=sum(r["rep_lineage"] for r in co),
            rep_lineage_av=sum(r["rep_lineage"] for r in av),
            rep_lineage_clean=sum(r["rep_lineage"] and not r["one_site"] and not r["site_competition"] for r in out))
import hashlib, os
# sets_md5: build_page.py and pg_knockout.py can check that these flags describe the sets file they read; written through a
# temporary file, so that a reader never sees half of it
json.dump(dict(meta=summ, sets=A.sets, sets_md5=hashlib.md5(open(A.sets, "rb").read()).hexdigest(), links=out),
          open(A.out + ".tmp", "w"), separators=(",", ":"))
os.replace(A.out + ".tmp", A.out)
print(json.dumps(summ))
print(f"-> {A.out}")
