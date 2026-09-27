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
             following it in no genome). A family's counts: n, the window's chromosomes where
             it falls inside the window; chrom, the complete chromosomes carrying it anywhere;
             pan, the pangenome's genomes carrying it (of meta.pan_genomes, 2,002: pgb/fam_info.json),
             which its partition is about. A persistent family can have a small n: in most
             chromosomes it lies outside the window.
  proteins   the most common protein of each family, as "c<index>" into pgb/chrom_emb.f16,
             so a drawn path can be sent to the model.
  locate     a gene or a family asked for (the page's search) opens the window that shows it in the
             most chromosomes: each carrier's copies are placed against every backbone anchor of that
             chromosome, the anchor at or before a copy within REACH genes that covers the most carriers
             is taken (its copy about LEAD genes in, when several do as well), and the other places the
             gene sits, in carriers this window misses, are listed (want_meta: the /region's meta.want).
  neighbours the windows of the page's Next and Previous: Next is the last anchor within `step` genes;
             Previous undoes Next (on the chain of Next from dnaA always, elsewhere when a window leads
             there by Next).

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

# per family: partition, one product and the genomes carrying it, from the PanGBank file once, then cached
HYPOTHETICAL = ("hypothetical protein", "")
def build_fam_info(out="pgb/fam_info.json"):
    """each family's partition (PPanGGOLiN's, over all the pangenome's genomes), product and genomes:
    product   the product most of its genes carry, "hypothetical protein" set aside (lacZ: 1,942 beta-galactosidase,
              15 hypothetical protein); a family with no other product keeps that one
    genomes   how many of the pangenome's genomes (n_genomes, 2,002) carry at least one of its genes: the number the
              partition is about (bcp: 1,997, persistent), not the genomes of one window, which only counts those
              where the family falls inside it
    NOTE: an earlier pgb/fam_info.json had the product of the family's first gene (lacZ "hypothetical protein") and no
    genomes. Its products feed the facts of the readings written ahead: after a rebuild that changes a product,
    regenerate them (pg_reading.py pregen; check with space/assemble.py --dry-run --check-facts). Adding genomes changes
    no fact."""
    import pandas as pd, tables
    h = tables.open_file("pgb/ecoli_11587.h5"); A = h.root.annotations
    info = h.root.geneFamiliesInfo.read(); PART = {"P": "persistent", "S": "shell", "C": "cloud"}
    assert [x.decode() for x in info["name"]] == FN, "the families of the PanGBank file are not those of pgb/chrom_genomes.json"
    part = [PART.get(x.decode(), x.decode()) for x in info["partition"]]
    gf = h.root.geneFamilies.read(); gfam = pd.Index(FN).get_indexer([x.decode() for x in gf["geneFam"]])
    genes = A.genes.read(); gi = pd.Index(genes["ID"]).get_indexer(gf["gene"])
    gd = genes["genedata_id"][gi]
    ctg = A.contigs.read(); gnm = A.genomes.read(field="name")
    genome = pd.Index(gnm).get_indexer(ctg["genome"][pd.Index(ctg["ID"]).get_indexer(genes["contig"][gi])])
    pr = A.genedata.read(field="product")[gd]; h.close()
    ok = (gfam >= 0) & (genome >= 0)
    ng = np.bincount(np.unique(gfam[ok].astype(np.int64) * len(gnm) + genome[ok]) // len(gnm), minlength=NF)
    df = pd.DataFrame(dict(f=gfam, p=pr)); df = df[df.f >= 0]
    n = df.groupby(["f", "p"]).size().reset_index(name="n"); n["p"] = [x.decode() for x in n.p]
    n["hyp"] = n.p.isin(HYPOTHETICAL)                           # most carried first, "hypothetical protein" last
    n = n.sort_values(["f", "hyp", "n", "p"], ascending=[True, True, False, True]).drop_duplicates("f")
    prod = [""] * NF
    for f, p in zip(n.f, n.p): prod[int(f)] = p
    json.dump(dict(partition=part, product=prod, n_genomes=int(len(gnm)), genomes=ng.astype(int).tolist()), open(out, "w"))
if not os.path.exists("pgb/fam_info.json"): build_fam_info()
_fi = json.load(open("pgb/fam_info.json")); PARTITION = _fi["partition"]; PRODUCT = _fi["product"]
# the genomes of the pangenome carrying each family, of PAN_TOTAL (None with a fam_info.json written before they were
# counted: the page then shows the partition alone)
PAN_N = _fi.get("genomes"); PAN_TOTAL = _fi.get("n_genomes")

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
N_CHROM = np.bincount(uk % NF, minlength=NF)     # the complete chromosomes carrying each family (a node's chrom)
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
# where each backbone anchor sits in each chromosome (-1: not there once): the window from anchor i in chromosome g
# is its genes ANC_LOC[i, g] .. ANC_LOC[i, g] + W - 1 (around the origin), as build() walks them
_rank = np.full(NF, -1, np.int64); _rank[BB] = np.arange(len(BB))
ANC_LOC = np.full((len(BB), NG), -1, np.int32)
_bs = _rank[f_single] >= 0
ANC_LOC[_rank[f_single[_bs]], GID[first[single]][_bs]] = pos_single[_bs]
del _bs
WINDOW = 80                      # genes per window (build's W)
REACH = 72                       # a gene is looked for at most this far after an anchor: a few columns follow it
LEAD = 12                        # when several anchors show a gene in as many chromosomes: the one that puts it about here
LAST_END = int(np.max(GL))       # the longest chromosome, in genes

def fam_name_label(f):
    """the family's most common gene name in the chromosomes, else its product, else its PanGBank name"""
    idx = np.flatnonzero(FAM == f)
    c = collections.Counter(NAME[idx].tolist()); c.pop(0, None)
    if c: return NAMES[c.most_common(1)[0][0]], True
    return (PRODUCT[f] or FN[f]), False

class NotFound(KeyError):
    """a query that names nothing here (an expected miss: the server answers 200 with found=false)"""


def _family(q):
    """a PanGBank family or a gene name (exact, else ignoring case; the family most carrying it) -> (family, name)"""
    f = FIDX.get(q)
    if f is not None: return f, None
    k = _ni.get(q) or next((i for n, i in _ni.items() if n.lower() == q.lower()), None)
    if not k: raise NotFound(f"no gene or family called {q} in the {NG} complete chromosomes")
    return collections.Counter(FAM[NAME == k].tolist()).most_common(1)[0][0], NAMES[k]


def _places(f, W=WINDOW, chunk=2048):
    """for family f: its carrier chromosomes, and for every backbone anchor and carrier the offset of the carrier's copy
    nearest after the anchor (the smallest (copy - anchor) mod length; 32767 when the chromosome lacks the anchor once)"""
    idx = np.flatnonzero(FAM == f)                           # sorted: by chromosome, then position
    g = GID[idx]; car, first_ = np.unique(g, return_index=True)
    off = np.full((len(BB), len(car)), 32767, np.int32)
    bounds = np.r_[first_, len(idx)]; s = 0
    while s < len(idx):                                      # chunks of whole chromosomes: bounded memory for many copies
        e = int(bounds[np.searchsorted(bounds, s + chunk, "right") - 1])
        if e <= s: e = int(bounds[np.searchsorted(bounds, s, "right")])
        gi, li = g[s:e], LOC[idx[s:e]]
        al = ANC_LOC[:, gi]
        d = np.where(al >= 0, (li[None, :] - al) % GL[gi][None, :], 32767)
        seg = np.flatnonzero(np.r_[True, gi[1:] != gi[:-1]])
        cols = np.searchsorted(car, gi[seg])
        off[:, cols] = np.minimum(off[:, cols], np.minimum.reduceat(d, seg, axis=1))
        s = e
    return car, off


def locate(q, W=WINDOW):
    """a query -> dict(anchor=<backbone family index>, want=<what was asked for, or None>).
    q: empty (the dnaA window), a position in genes from dnaA (the window whose anchor is at or just before it), a
    PanGBank family or a gene name: the anchor at or before a copy of it, within REACH genes, that shows it in the most
    carrier chromosomes (then the one putting it nearest LEAD genes in; when no copy is within REACH, within the window's
    W genes); a backbone family is its own window's anchor.
    want: q, fam, node (its node's id in the window), label, carriers (the complete chromosomes carrying it), and
    places: the anchors that show it in the carriers the chosen window misses, most carriers first (greedy, up to 3).
    Raises NotFound for a name or a position that is not there."""
    q = (q or "").strip()
    if not q: return dict(anchor=int(BB[0]), want=None)
    if re.fullmatch(r"\d+", q):                                  # a position from dnaA
        p = int(q)
        if p >= LAST_END: raise NotFound(f"gene {p:,} from dnaA is past the end of the chromosomes (the longest has {LAST_END:,} genes)")
        i = int(np.clip(np.searchsorted(BB_POS, float(p), "right") - 1, 0, len(BB) - 1)); return dict(anchor=int(BB[i]), want=None)
    f, nm = _family(q)
    car, off = _places(f, W)
    if not len(car): raise NotFound(f"{q} is in none of the {NG} complete chromosomes (only in draft genomes)")
    want = dict(q=q, fam=FN[f], node=f"p{f}", label=nm or fam_name_label(f)[0], carriers=int(len(car)))
    # anywhere: some window shows it in some carrier (a copy at most W - 1 genes after an anchor of that chromosome)
    want["anywhere"] = bool((off <= W - 1).any())
    reach = REACH if (off <= REACH).any() else W - 1              # past REACH everywhere: a window showing it near its end
    if f in BB_RANK: a = f                                        # a backbone family: the window it anchors
    elif not want["anywhere"]:                                    # no window shows it: the one before its median place
        p = float(np.median(LOC[FAM == f]))
        a = int(BB[int(np.clip(np.searchsorted(BB_POS, p - 8, "right") - 1, 0, len(BB) - 1))])
    else:
        good = (off <= reach).sum(1)
        top = np.flatnonzero(good == good.max())
        med = np.array([np.median(off[j][off[j] <= reach]) for j in top])
        a = int(BB[top[int(np.argmin(np.abs(med - LEAD)))]])      # ties: the lower anchor (np.argmin keeps the first)
    want["places"] = _other_places(a, car, off, W)
    return dict(anchor=int(a), want=want)


def _other_places(a, car, off, W=WINDOW, most=3):
    """the anchors that show the family in the carriers the window of anchor a misses, greedily, most carriers first"""
    left = off[BB_RANK[a]] > W - 1
    out = []
    while left.any() and len(out) < most:
        good = ((off <= REACH) & left[None, :]).sum(1)
        j = int(np.argmax(good)); n = int(good[j])
        if n < max(2, 0.02 * len(car)): break                    # a place in one chromosome, or in under 2 % of them: not listed
        hit = (off[j] <= REACH) & left
        out.append(dict(fam=FN[int(BB[j])], label=fam_name_label(int(BB[j]))[0], pos=round(float(BB_POS[j])), carriers=n,
                        column=int(np.median(off[j][hit]))))
        left &= ~(off[j] <= W - 1)
    return out


def want_meta(want, D, place_of=None):
    """the /region's meta.want for a window built (D, build's output) and what was asked for (locate's or want_at's
    want): found (its node is in the window), column (its node's), carriers_in_window (the chromosomes showing it
    here), carriers (the complete chromosomes carrying it), window_genomes, and other_anchors: the other places it sits
    in the chromosomes this window misses, each {fam, label, pos (the anchor's, from dnaA), carriers (of those
    chromosomes, shown there), column (its median column there)}, and anywhere (false when every copy sits more than
    W - 1 genes after every anchor of its chromosome: no window shows it)"""
    if not want: return None
    n = next((x for x in D["nodes"] if x["id"] == want["node"]), None)
    anc = D["meta"]["anchor"]
    return dict(q=want["q"], fam=want["fam"], node=want["node"], label=want["label"], found=n is not None,
                column=n["locus"] if n else None, carriers_in_window=n["n"] if n else 0, carriers=want["carriers"],
                window_genomes=D["meta"]["n_genomes"], other_anchors=[p for p in want["places"] if p["fam"] != anc],
                anywhere=want.get("anywhere"))


def want_at(q, anchor, W=WINDOW):
    """what was asked for, seen from a window chosen by its anchor (the page's button to another place of a gene): the
    places listed are those that show it in the carriers this window misses, as locate lists them for its own choice"""
    f, nm = _family(q)
    car, off = _places(f, W)
    if not len(car): raise NotFound(f"{q} is in none of the {NG} complete chromosomes (only in draft genomes)")
    return dict(q=q, fam=FN[f], node=f"p{f}", label=nm or fam_name_label(f)[0], carriers=int(len(car)),
                places=_other_places(int(anchor), car, off, W), anywhere=bool((off <= W - 1).any()))


def resolve(q):
    """a query (PanGBank family, gene name, or position from dnaA) -> the anchor of the window that shows it (locate)"""
    return locate(q)["anchor"]


_NAV = {}
def _nav(step):
    """Next of every anchor (the last one within `step` genes, at least the following one; -1 for the last) and the
    chain of Next from the dnaA window (the page's Next from dnaA goes to the anchor at or before gene `step`)"""
    if step not in _NAV:
        i = np.arange(len(BB))
        nx = np.clip(np.searchsorted(BB_POS, BB_POS + step, "right") - 1, i + 1, len(BB) - 1); nx[-1] = -1
        chain, a = {}, 0
        while a >= 0 and a not in chain: chain[a] = len(chain); a = int(nx[a])
        _NAV[step] = (nx, {a: k for a, k in chain.items()}, list(chain))
    return _NAV[step]


def neighbours(f, step):
    """the windows of the page's Previous and Next from anchor f. Next: the last anchor within `step` genes. Previous
    undoes Next: on the chain of Next from dnaA, the window before in it; elsewhere the nearest window whose Next is
    this one; if none (a window opened by a search), the first anchor at least `step` genes before"""
    i = BB_RANK[f]; p = BB_POS[i]; nx, at, chain = _nav(step)
    n = int(nx[i]) if nx[i] >= 0 else None
    if i == 0: pv = None
    elif i in at and at[i] > 0: pv = chain[at[i] - 1]
    else:
        c = np.flatnonzero(nx[:i] == i)
        pv = int(c[-1]) if len(c) else int(np.clip(np.searchsorted(BB_POS, p - step, "left"), 0, i - 1))
    lab = lambda j: None if j is None else dict(fam=FN[int(BB[j])], label=fam_name_label(int(BB[j]))[0], pos=round(float(BB_POS[j])))
    return lab(pv), lab(n)


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
    rgp_c = RGP[rows].astype(bool); part_c = np.array([PARTITION[int(f)] for f in fam_at_f])
    per_locus = []
    for k in range(W):
        m = call_k == k
        if m.sum(): per_locus.append(dict(locus=k, n=int(m.sum()), entropy=round(float(ent[m].mean()), 3), ppl=ppl(m),
                                          unread=round(float((ptrue[m] == 0).mean()), 4), top1=None, top1_dec=round(float(hit[m].mean()), 3),
                                          rgp=round(float(rgp_c[m].mean()), 3)))
    def table(m):                                                    # the calls of the window, by the gene actually there
        return dict(n=int(m.sum()), top1=None, top1_dec=round(float(hit[m].mean()), 3), med_rank=None,
                    entropy=round(float(ent[m].mean()), 3), ppl=ppl(m)) if m.sum() else None
    by_partition = {k: table(part_c == k) for k in ("persistent", "shell", "cloud")}
    by_rgp = {"outside an RGP": table(~rgp_c), "inside an RGP": table(rgp_c)}
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
                          pan=PAN_N[f] if PAN_N else None, chrom=int(N_CHROM[f]),
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
                prev=prev, next=nextw, window=W, n_genomes=G, genomes_total=NG, pan_genomes=PAN_TOTAL, calls=int(len(fam_at)),
                top1=None, top1_dec=round(float(hit.mean()), 3), ppl=ppl(np.ones(len(fam_at), bool)), unread=round(float((ptrue == 0).mean()), 4),
                families=len(nodes), edges=len(E), ms=round((time.time() - t0) * 1000))
    return dict(meta=meta, per_locus=per_locus, by_partition=by_partition, by_rgp=by_rgp, nodes=nodes, edges=E, ghosts=ghosts,
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
