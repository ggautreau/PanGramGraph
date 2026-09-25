"""Any window of the complete chromosomes, as graph data for the PanGramGraph page, built
when the page asks (serve_live.py /region).

The page opens on the dnaA window, 80 genes from dnaA in all 2,002 genomes (pgb_graph.py).
Further along the chromosome only the 540 complete genomes are in one piece, and only they
have Bacformer's call at every gene (pg_chrom.py, pg_calls.py: each call has at least 400
proteins of upstream context). A window elsewhere is read in them.

  anchor     a backbone family: one copy in >= 95 % of the 540 chromosomes. The window
             starts at its gene and walks W genes in dnaA's direction, in every chromosome
             carrying it once. Near its anchor a family sits at nearly the same column in
             every genome, as near dnaA; columns are offsets from the anchor.
  graph      as pgb_graph.py: a family at the column where it most often sits, adjacencies
             weighted by genomes, the model's call after each family averaged over the
             genomes carrying it, the decoder P(family | model cluster) learned on the
             window's calls and cross-validated on genome halves, perplexity on the family
             actually there, and ghosts (expected after the best-carried family of a column,
             following it in no genome).
  proteins   the most common protein of each family, as "c<index>" into pgb/chrom_emb.f16,
             so a drawn path can be sent to the model.

    python3 pgb_region.py [family or gene or position]     # builds one window, prints a summary
"""
import os, re, json, time, collections, warnings, numpy as np
from scipy import sparse
warnings.simplefilter("ignore")
BACKBONE_MIN = 0.95
Z = np.load("pgb/chrom.npz"); OFF = Z["offsets"]; FAM = Z["fam"]; PID = Z["pid"]; RGP = Z["rgp"]
NG = len(OFF) - 1; NP = int(OFF[-1]); GL = np.diff(OFF); GID = np.repeat(np.arange(NG), GL); LOC = np.arange(NP) - OFF[GID]
CJ = json.load(open("pgb/chrom_genomes.json")); FN = CJ["families"]; NF = len(FN); FIDX = {n: i for i, n in enumerate(FN)}
CGEN = CJ["genomes"]
TF = np.memmap("pgb/calls_top15_f.i32", dtype=np.int32, mode="r", shape=(NP, 15))
TP = np.memmap("pgb/calls_top15_p.f16", dtype=np.float16, mode="r", shape=(NP, 15))
ENT = np.memmap("pgb/calls_ent.f16", dtype=np.float16, mode="r", shape=(NP,))

# gene names, coded
_names = collections.Counter(x for g in CJ["gene_names"] for x in g if x)
NAMES = [""] + sorted(_names); _ni = {n: i for i, n in enumerate(NAMES)}
NAME = np.array([_ni.get(x or "", 0) for g in CJ["gene_names"] for x in g], dtype=np.int32)

# per family: partition and one product, from the PanGBank file once, then cached
if not os.path.exists("pgb/fam_info.json"):
    import pandas as pd, tables
    h = tables.open_file("pgb/ecoli_11587.h5"); A = h.root.annotations
    info = h.root.geneFamiliesInfo.read(); PART = {"P": "persistent", "S": "shell", "C": "cloud"}
    part = [PART.get(x.decode(), x.decode()) for x in info["partition"]]
    gf = h.root.geneFamilies.read(); gfam = pd.Index(FN).get_indexer([x.decode() for x in gf["geneFam"]])
    one = pd.Series(gf["gene"]).groupby(gfam).first()
    gid_all = pd.Index(A.genes.read(field="ID")); ggd = A.genes.read(field="genedata_id")
    rows = gid_all.get_indexer(one.values); gd = ggd[rows]; o = np.argsort(gd)
    pr = A.genedata.read_coordinates(gd[o], field="product"); h.close()
    prod = [""] * NF
    for i, p in zip(o, pr): prod[int(one.index[i])] = p.decode()
    json.dump(dict(partition=part, product=prod), open("pgb/fam_info.json", "w"))
_fi = json.load(open("pgb/fam_info.json")); PARTITION = _fi["partition"]; PRODUCT = _fi["product"]

# the model's clusters, named from its exemplar bank
annot = json.load(open("fam_annot.json"))
def bflabel(f):
    a = annot.get(str(int(f)))
    if not a: return f"fam{int(f)}"
    if a["gene"]: return a["gene"][0][0]
    return a["product"][0][0] if a["product"] else f"fam{int(f)}"

# genomes as the page lists them (the 2,002 of the dnaA window), reference strains first in examples
_W = json.load(open("pgb/window.json"))
PAGE_IDX = {g["acc"]: i for i, g in enumerate(_W["genomes"])}; del _W
PREF = {acc: i for i, acc in enumerate(json.load(open("eco/manifest.json")))}
CPAGE = np.array([PAGE_IDX.get(g["acc"], -1) for g in CGEN])

# --- the backbone: one copy in >= 95 % of chromosomes, ordered by median position from dnaA
key = GID.astype(np.int64) * NF + FAM
uk, first, cnt = np.unique(key, return_index=True, return_counts=True)
single = cnt == 1
f_single = (uk % NF)[single]; pos_single = LOC[first[single]]
n_single = np.bincount(f_single, minlength=NF)
BB = np.where(n_single >= BACKBONE_MIN * NG)[0]
_med = {}
order = np.argsort(f_single, kind="stable"); fs = f_single[order]; ps = pos_single[order]
starts = np.searchsorted(fs, BB); ends = np.searchsorted(fs, BB, "right")
for f, a, b in zip(BB, starts, ends): _med[int(f)] = float(np.median(ps[a:b]))
BB = np.array(sorted(BB, key=lambda f: _med[int(f)])); BB_POS = np.array([_med[int(f)] for f in BB])
BB_RANK = {int(f): i for i, f in enumerate(BB)}

def fam_name_label(f):
    """the family's most common gene name in the chromosomes, else its product, else its PanGBank name"""
    idx = np.flatnonzero(FAM == f)
    c = collections.Counter(NAME[idx].tolist()); c.pop(0, None)
    if c: return NAMES[c.most_common(1)[0][0]], True
    return (PRODUCT[f] or FN[f]), False

def resolve(q):
    """a query (PanGBank family, gene name, or position from dnaA) -> the backbone anchor at or just before it"""
    q = (q or "").strip()
    if not q: return int(BB[0])
    if re.fullmatch(r"\d+", q):                                  # a position from dnaA
        i = int(np.clip(np.searchsorted(BB_POS, float(q), "right") - 1, 0, len(BB) - 1)); return int(BB[i])
    f = FIDX.get(q)
    if f is None:                                                # a gene name, most carried first
        k = _ni.get(q) or next((i for n, i in _ni.items() if n.lower() == q.lower()), None)
        if not k: raise KeyError(f"no gene or family called {q} in the complete chromosomes")
        f = collections.Counter(FAM[NAME == k].tolist()).most_common(1)[0][0]
    if f in BB_RANK: return int(f)
    idx = np.flatnonzero(FAM == f)
    if not len(idx): raise KeyError(f"{q} is in no complete chromosome")
    p = float(np.median(LOC[idx]))                              # a few genes before it, so it shows in the window
    i = int(np.clip(np.searchsorted(BB_POS, p - 8, "right") - 1, 0, len(BB) - 1)); return int(BB[i])

def neighbours(f, step):
    i = BB_RANK[f]; p = BB_POS[i]
    nx = int(np.clip(np.searchsorted(BB_POS, p + step, "right") - 1, i + 1, len(BB) - 1)) if i + 1 < len(BB) else None
    pv = int(np.clip(np.searchsorted(BB_POS, p - step, "left"), 0, i - 1)) if i > 0 else None
    lab = lambda j: None if j is None else dict(fam=FN[int(BB[j])], label=fam_name_label(int(BB[j]))[0], pos=round(float(BB_POS[j])))
    return lab(pv), lab(nx)

def build(anchor, W=80):
    t0 = time.time(); a = int(anchor)
    idx = np.flatnonzero(FAM == a); g = GID[idx]
    once = np.bincount(g, minlength=NG) == 1
    keep = once[g]; idx = idx[keep]; g = g[keep]                    # chromosomes carrying the anchor once
    G = len(g)
    if not G: raise KeyError("the anchor is in no chromosome once")
    K = np.arange(W)
    WI = OFF[g][:, None] + (LOC[idx][:, None] + K[None]) % GL[g][:, None]      # G x W global gene indices
    WF = FAM[WI]; WL = LOC[WI]
    # the calls predicting each gene of the window (the first gene of a chromosome has no context: none)
    has = WL != 0
    rows = WI[has]; fam_at_f = WF[has]; call_g = np.repeat(np.arange(G), W)[has.ravel()]; call_k = np.tile(K, G)[has.ravel()]
    top_f = np.asarray(TF[np.sort(rows)]); top_p = np.asarray(TP[np.sort(rows)]).astype(np.float32); ent = np.asarray(ENT[np.sort(rows)]).astype(np.float32)
    back = np.argsort(np.argsort(rows)); top_f = top_f[back]; top_p = top_p[back]; ent = ent[back]
    fams = np.unique(WF); FI = {int(f): i for i, f in enumerate(fams)}; NFr = len(fams)
    fam_at = np.array([FI[int(f)] for f in fam_at_f]); KC = top_f.shape[1]
    def decoder(mask):
        A_ = sparse.csr_matrix((top_p[mask].ravel(), (np.repeat(fam_at[mask], KC), top_f[mask].ravel())), shape=(NFr, 50001))
        return A_, np.asarray(A_.sum(0)).ravel()
    A_all, tot = decoder(np.ones(len(fam_at), bool)); Ac = A_all.tocsc()
    def reads(c, min_mass=1.0):
        if tot[c] < min_mass: return None
        col = Ac.getcol(c); return int(fams[col.indices[col.data.argmax()]])
    def dec_weights(f):
        r = A_all.getrow(FI[f]); out = [[int(c), round(float(v / tot[c]), 4)] for c, v in zip(r.indices, r.data) if v >= 0.5 and v / tot[c] >= 0.05]
        return sorted(out, key=lambda x: -x[1])[:12]
    hit = np.zeros(len(fam_at), bool); ptrue = np.zeros(len(fam_at)); half = call_g % 2
    for hh in (0, 1):                                                # read with the decoder of the other half
        A_, t_ = decoder(half != hh); Pn = sparse.diags(1 / np.maximum(t_, 1e-12)) @ A_.T
        m = np.where(half == hh)[0]
        for s in range(0, len(m), 4000):
            mm = m[s:s + 4000]
            Q = sparse.csr_matrix((top_p[mm].ravel(), (np.repeat(np.arange(len(mm)), KC), top_f[mm].ravel())), shape=(len(mm), 50001))
            S = (Q @ Pn).toarray(); hit[mm] = (S.argmax(1) == fam_at[mm]) & (S.max(1) > 0); ptrue[mm] = S[np.arange(len(mm)), fam_at[mm]]
    p_eff = np.where(ptrue > 0, ptrue, top_p[:, -1])
    ppl = lambda m: round(float(2 ** np.mean(-np.log2(np.maximum(p_eff[m], 1e-12)))), 3)
    per_locus = []
    for k in range(W):
        m = call_k == k
        if m.sum(): per_locus.append(dict(locus=k, n=int(m.sum()), entropy=round(float(ent[m].mean()), 3), ppl=ppl(m),
                                          unread=round(float((ptrue[m] == 0).mean()), 4), top1=None, top1_dec=round(float(hit[m].mean()), 3)))
    call_of = {(int(gg), int(kk)): i for i, (gg, kk) in enumerate(zip(call_g, call_k))}
    # --- families: carriers, columns, neighbours, proteins, names
    occ = collections.defaultdict(list); loci = collections.defaultdict(collections.Counter); rg = collections.defaultdict(list)
    pids = collections.defaultdict(collections.Counter); nms = collections.defaultdict(collections.Counter)
    nxt = collections.defaultdict(collections.Counter); edges = collections.Counter()
    WR = RGP[WI]; WP = PID[WI]; WN = NAME[WI]
    for gi in range(G):
        row = WF[gi]
        for k in range(W):
            f = int(row[k]); occ[f].append((gi, k)); loci[f][k] += 1; rg[f].append(bool(WR[gi, k])); pids[f][int(WP[gi, k])] += 1
            if WN[gi, k]: nms[f][int(WN[gi, k])] += 1
            if k + 1 < W: f2 = int(row[k + 1]); nxt[f][f2] += 1; edges[(f, f2)] += 1
    carriers = {f: {gi for gi, _ in oc} for f, oc in occ.items()}
    def label(f):
        if nms[f]: return NAMES[nms[f].most_common(1)[0][0]], True
        return (PRODUCT[f] or FN[f]), False
    gpage = CPAGE[g]
    nodes = []
    for f, oc in occ.items():
        acc = collections.defaultdict(float); n = 0; hd = 0
        for gi, k in oc:
            i = call_of.get((gi, k + 1))
            if i is None: continue
            n += 1; hd += int(hit[i])
            for ff, pp in zip(top_f[i], top_p[i]): acc[int(ff)] += float(pp)
        top = sorted(acc.items(), key=lambda x: -x[1])[:8]
        call = []
        for ff, pp in top:
            r = reads(ff); call.append([ff, round(pp / n, 5), int(r is not None and r in nxt[f]), f"p{r}" if r is not None else ""])
        dw = dec_weights(f); lab, named = label(f)
        ex = sorted(carriers[f], key=lambda gi: (PREF.get(CGEN[g[gi]]["acc"], 99), gi))[:10]
        nodes.append(dict(id=f"p{f}", fam=FN[f], label=lab, named=named, product=PRODUCT[f], partition=PARTITION[f],
                          n=len(carriers[f]), locus=loci[f].most_common(1)[0][0], rgp=round(float(np.mean(rg[f])), 3),
                          pid=f"c{pids[f].most_common(1)[0][0]}", bf=dw[0][0] if dw else -1, bfw=dw, bfs=[c for c, w in dw if w >= 0.5] or [c for c, _ in dw[:1]],
                          next=[[f"p{f2}", c] for f2, c in nxt[f].most_common(5)], ex=[int(gpage[gi]) for gi in ex if gpage[gi] >= 0],
                          calls=n, top1=None, top1_dec=round(hd / n, 3) if n else None, med_rank=None, call=call))
    E = [dict(s=f"p{a_}", t=f"p{b_}", n=c) for (a_, b_), c in edges.items()]
    bycol = collections.defaultdict(list)
    for nd in nodes: bycol[nd["locus"]].append(nd)
    ghosts = []
    for col, nds in bycol.items():
        src = max(nds, key=lambda x: x["n"])
        for ff, pp, r in [(ff, pp, r) for ff, pp, seen, r in src["call"] if not seen][:2]:
            ghosts.append(dict(src=src["id"], locus=col + 1, bf=ff, label=label(int(r[1:]))[0] if r else bflabel(ff), p=pp,
                               reads=r, elsewhere=int(bool(r))))
    step = max(20, W - 20); prev, nextw = neighbours(a, step); alab = label(a)[0]
    meta = dict(region=True, anchor=FN[a], anchor_label=alab, anchor_pos=round(float(BB_POS[BB_RANK[a]])) if a in BB_RANK else None,
                prev=prev, next=nextw, window=W, n_genomes=G, genomes_total=NG, calls=int(len(fam_at)),
                top1=None, top1_dec=round(float(hit.mean()), 3), ppl=ppl(np.ones(len(fam_at), bool)), unread=round(float((ptrue == 0).mean()), 4),
                families=len(nodes), edges=len(E), ms=round((time.time() - t0) * 1000))
    return dict(meta=meta, per_locus=per_locus, nodes=nodes, edges=E, ghosts=ghosts,
                bflabels={str(ff): bflabel(ff) for nd in nodes for ff, *_ in nd["call"]})

if __name__ == "__main__":
    import sys
    t = time.time(); a = resolve(sys.argv[1] if len(sys.argv) > 1 else "")
    print(f"backbone: {len(BB)} families (one copy in >= {BACKBONE_MIN:.0%} of {NG} chromosomes), loaded in {time.time() - t:.1f} s")
    R = build(a); m = R["meta"]
    print(f"window from {m['anchor_label']} ({m['anchor']}, ~gene {m['anchor_pos']} from dnaA): {m['n_genomes']} chromosomes, "
          f"{m['families']} families, {m['edges']} links, {m['calls']} calls, top-1 {m['top1_dec']:.1%}, perplexity {m['ppl']}, built in {m['ms']} ms")
    print("previous", m["prev"], "| next", m["next"])
    print("row 0:", [max((n for n in R["nodes"] if n["locus"] == k), key=lambda n: n["n"])["label"] for k in range(12)])
    print(f"json {len(json.dumps(R, separators=(',', ':'))) / 1e6:.2f} MB")
