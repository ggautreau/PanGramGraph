"""In-silico knockout of the "Coupled regions" links (pg_region_sets.py) with Bacformer: the batch.

    python3 pg_knockout.py --baselines        540 fp32 full-chromosome passes -> pgb/knock/base_top64_{f,p}.npy
                                              (int32 / float16, row OFF[g] + k = the call for gene k), base_ent.npy
                                              (bits), base_meta.json; the lineage-CV decoder is built from them
    python3 pg_knockout.py --pilot            24 links (8 co-occurrence and 6 avoidance of highest |z|, 2 sharing a
                                              product + 2 mostly mobile, 6 by seed 0), every testable genome
                                              -> pgb/knock/pilot/g*.npz, pilot_units.parquet, pilot_links.parquet,
                                                 pilot_posctrl.parquet, pilot_summary.json, pilot_report.md
    python3 pg_knockout.py --all              every link (same code; not run yet) -> pgb/knock/all/...
    python3 pg_knockout.py --decay            framing diagnostic on the positive-control units: effect of a 2-gene deletion
                                              or of a copy of D at d = 1..1,600 genes upstream -> pgb/knock/pilot_decay.parquet
    python3 pg_knockout.py --probe 35,20      content vs position for given links: U deletion vs sham deletion next to U
                                              vs in-place substitution of U -> pgb/knock/pilot_probe.parquet
    add --aggregate to rebuild the tables and the report from the per-genome parts without the GPU,
    --genomes 0,1,2 to restrict (testing), --tag NAME for another output folder.

    --sets PATH       another link set in the format of pgb/region_sets.json (e.g. pgb/region_sets_close.json);
                      needs --tag (outputs pgb/knock/<tag>/ and pgb/knock/<tag>_*); the units cache is keyed by the
                      file's md5, and plan.json records it: resuming a tag with another sets file is refused
    --min_apart N     units need >= N genes between U's last gene and D's readout (default 100, as before)
    --influence       influence readouts in every deletion pass (U, whole-element controls, check b): the decoded
                      logit P(true family) at every gene out to 1,500 genes downstream of the deleted element, as
                      25-gene bins (mean delta, mean |delta|, mean delta over panRGP genes), and at the entry gene of
                      every accessory element (pg_knock.Data.elements) out to 2,400 genes; stored per genome in
                      g*.npz (inf_*, el_*, pass_spot; float16 deltas, int16 element indices). On by default with
                      --all, off otherwise (--no-influence / --influence to change); see load_influence()
    --mem_frac F      per-process GPU memory cap (default 0.35: two runs share the 8 GB card)
    --max_apart N     units need <= N genes between U's last gene and D's readout (default: no bound, as before). For
                      the close-range sets use 600: a unit is one genome's U-D pair in dnaA reading order, and in the close
                      smoke run 21.6 % of them were > 600 genes apart (up to 3,844; pairs across the origin), outside the
                      model's range
    --ctrl_near       add, per unit whose matched null is thin, whole elements ending at about the unit's own distance
                      upstream of D (pg_knock.Data.near_controls; recorded as pass_near in g*.npz). Close range only: a
                      genome rarely holds them, so the gain is small (+0-2 % of units with >= 3 matched controls)
    --match_log F     a third matching tier (aggregation): distance within x1/F..xF on a log scale, size x1/3-3, when the
                      +-35 % / +-60 % windows give < 3 controls (default: off, as before). Close units 50-100 / 100-250 /
                      250-600 genes apart with >= 3 matched controls: 10 / 35 / 73 % without, 31 / 74 / 95 % with F = 3
    --flags PATH      pg_link_flags.py output for the sets file: joined into {tag}_arcs.json, the per-arc verdict for the
                      page (distance, power, and whether the "link" is one element at several spots, one transfer tract,
                      two elements competing for one site, or not replicated in lineage-disjoint halves)
    The unit rules (--min_apart, --max_apart, --ctrl_near, --match_log) are recorded in the tag's plan.json; a later call
    on the same tag (resume, --aggregate) takes them from there unless given, and refuses a different explicit value.

Per genome, every pass has the same padded length (n + 1), batch 1, fp32; readouts are decoded in float64.
Self-checks run in every genome and stop the run if they fail:
  (a) a pass deleting nothing gives exactly the baseline readouts (delta == 0.0)
  (b) a pass deleting 10 genes right after the last readout row changes no readout; and in EVERY pass, every
      readout row before the first deleted (or inserted) gene is bit-identical to the baseline (causality)
  (c) every readout row, after the deletion/insertion shifts positions, lands on the intended original gene
      (index identity and family identity asserted)
  (d) a genome is read with the decoder of the other lineage half: its lineage and its genes are not in it
  (e) the row convention, against the model; (f) the numerical floor (see the report)
  (g) with --influence: every gene read in the 25 genes before the first deleted gene is bit-identical to the
      baseline (every deletion pass), and the whole baseline read again with another head-chunk composition
      (in the check-a pass) is bit-identical too -- the per-gene decode does not depend on the other rows
"""
import os, sys, json, time, argparse, collections
import numpy as np
import pg_knock as K
from pg_knock import log

ap = argparse.ArgumentParser()
ap.add_argument("--baselines", action="store_true")
ap.add_argument("--pilot", action="store_true")
ap.add_argument("--all", action="store_true")
ap.add_argument("--aggregate", action="store_true")
ap.add_argument("--decay", action="store_true")
ap.add_argument("--probe", default=None)
ap.add_argument("--genomes", default=None)
ap.add_argument("--tag", default=None)
ap.add_argument("--sets", default=K.SETS_DEFAULT, help="link set (format of pgb/region_sets.json); another file needs --tag")
ap.add_argument("--min_apart", type=int, default=None, help="genes between U's last gene and D's readout (default 100)")
ap.add_argument("--max_apart", type=int, default=None, help="at most N genes between U's last gene and D's readout (default: none)")
ap.add_argument("--ctrl_near", action=argparse.BooleanOptionalAction, default=None,
                help="extra whole-element controls at about each unit's own distance upstream of D (default: off)")
ap.add_argument("--match_log", type=float, default=None, help="third matching tier: distance within x1/F..xF (default: off)")
ap.add_argument("--flags", default=None, help="pg_link_flags.py output, joined into {tag}_arcs.json")
ap.add_argument("--influence", action=argparse.BooleanOptionalAction, default=None,
                help="influence readouts in every deletion pass (default: on with --all, off otherwise)")
ap.add_argument("--mem_frac", type=float, default=K.MEM_FRAC, help="per-process GPU memory cap")
args = ap.parse_args()
if os.path.abspath(args.sets) != K.SETS_DEFAULT and not args.tag and not args.baselines:
    ap.error("--sets with another file than pgb/region_sets.json needs --tag (its own output folder)")
INF = bool(args.influence) if args.influence is not None else bool(args.all)
CTRL_NEAR = False
PLAN_KEYS = (("min_apart", 100), ("max_apart", None), ("ctrl_near", False), ("match_log", None))


def settle(tag):
    """The unit rules of a tag: given on the command line, else recorded in its plan.json (a resume or an --aggregate then
    rebuilds exactly the units of the run), else the defaults (MIN_APART 100, no maximum, no near controls, no log tier)."""
    global CTRL_NEAR
    pf = f"{K.KDIR}/{tag}/plan.json"; old = json.load(open(pf)) if os.path.exists(pf) else {}
    val = {}
    for k, dflt in PLAN_KEYS:
        cli = getattr(args, k)
        val[k] = cli if cli is not None else old.get(k, dflt)      # an explicit different value is refused in run()
    K.MIN_APART = int(val["min_apart"]); K.MAX_APART = val["max_apart"]; K.MATCH_LOG = val["match_log"]
    CTRL_NEAR = bool(val["ctrl_near"])
    if any(getattr(args, k) is None and k in old for k, _ in PLAN_KEYS):
        log(f"{tag}: unit rules from plan.json: " + ", ".join(f"{k}={val[k]}" for k, _ in PLAN_KEYS))
    return old

D = K.Data(sets=args.sets)
os.makedirs(K.KDIR, exist_ok=True)


# ============================================================================ 1. baselines
def run_baselines():
    R = K.Runner(frac=args.mem_frac)
    NP = D.NP
    ff, fp, fe = f"{K.KDIR}/base_top64_f.npy", f"{K.KDIR}/base_top64_p.npy", f"{K.KDIR}/base_ent.npy"
    mode = "r+" if os.path.exists(ff) else "w+"
    TF = np.lib.format.open_memmap(ff, mode=mode, dtype=np.int32, shape=(NP, K.TOPK))
    TP = np.lib.format.open_memmap(fp, mode=mode, dtype=np.float16, shape=(NP, K.TOPK))
    EN = np.lib.format.open_memmap(fe, mode=mode, dtype=np.float16, shape=(NP,))
    mf = f"{K.KDIR}/base_meta.json"
    meta = json.load(open(mf)) if os.path.exists(mf) and mode == "r+" else dict(done=[], seconds={})
    done = set(meta["done"]); t0 = time.time()
    for g in range(D.NG):
        if g in done: continue
        a, b = D.OFF[g], D.OFF[g + 1]; n = int(b - a)
        R.set_genome(D.EMB[D.PID[a:b]], n + 1)
        R.torch.cuda.synchronize(); t = time.time()
        h = R.hidden(np.arange(n)); tf, tp, en = R.full_calls(h, n)
        R.torch.cuda.synchronize(); meta["seconds"][str(g)] = round(time.time() - t, 3)
        TF[a:b] = tf; TP[a:b] = tp.astype(np.float16); EN[a:b] = en.astype(np.float16)
        meta["done"].append(g)
        if g % 20 == 0 or g == D.NG - 1:
            TF.flush(); TP.flush(); EN.flush(); json.dump(meta, open(mf, "w"))
            log(f"baseline {g+1}/{D.NG}  {meta['seconds'][str(g)]:.2f}s/pass  top-1 p median {np.median(tp[:,0]):.3f}")
    TF.flush(); TP.flush(); EN.flush()
    sec = np.array(list(meta["seconds"].values()))
    meta.update(total_seconds=round(time.time() - t0, 1), seconds_per_pass_median=float(np.median(sec)),
                precision="float32", topk=K.TOPK, rows="row OFF[g]+k = softmax call for gene k given genes 0..k-1 (k=0: from CLS)")
    json.dump(meta, open(mf, "w"))
    # the decoder: halves, and a sanity readout of the true family on genes of each half
    dec = K.Decoder(D)
    rep = dict(halves={str(h): dict(genomes=int((D.HALF == h).sum()), lineages=len(dec.lineages[h])) for h in (0, 1)})
    rng = np.random.default_rng(0); acc = {}
    for h in (0, 1):
        gs = np.flatnonzero(D.HALF == h)
        idx = np.concatenate([D.OFF[g] + rng.choice(np.arange(1, D.GL[g]), 40, replace=False) for g in rng.choice(gs, 30, replace=False)])
        hd = 1 - h; pt = []
        for i in idx:
            w = dec.row(int(D.FAM[i]), hd); pt.append(float((np.asarray(dec.TP[i], np.float64) * w[np.asarray(dec.TF[i])]).sum()))
        acc[str(h)] = dict(genes=len(idx), ptrue_top64_median=round(float(np.median(pt)), 4), ptrue_top64_mean=round(float(np.mean(pt)), 4))
    rep["decoder_ptrue_on_held_out_half"] = acc
    meta["decoder"] = rep; json.dump(meta, open(mf, "w"))
    log(f"baselines done: {len(meta['done'])} chromosomes, median {np.median(sec):.3f}s/pass; decoder check {acc}")


# ============================================================================ 2. link selection
def pilot_links():
    L = D.LINKS; REG = D.REG
    co = sorted([i for i, l in enumerate(L) if l["sign"] > 0], key=lambda i: -abs(L[i]["z"]))
    av = sorted([i for i, l in enumerate(L) if l["sign"] < 0], key=lambda i: -abs(L[i]["z"]))
    sel = collections.OrderedDict()
    for i in co[:8]: sel[i] = "top co-occurrence z"
    for i in av[:6]: sel[i] = "top avoidance |z|"
    shared = sorted([i for i, l in enumerate(L) if l["shared"] and i not in sel], key=lambda i: -abs(L[i]["z"]))
    for i in shared[:2]: sel[i] = "shared product: " + ", ".join(L[i]["shared"])
    mob = sorted([i for i, l in enumerate(L) if REG[l["a"]]["mobile"] >= 0.5 and REG[l["b"]]["mobile"] >= 0.5 and i not in sel],
                 key=lambda i: -abs(L[i]["z"]))
    for i in mob[:2]: sel[i] = "both regions mostly mobile"
    rest = [i for i in range(len(L)) if i not in sel]
    for i in sorted(np.random.default_rng(0).choice(rest, 6, replace=False).tolist()): sel[int(i)] = "random (seed 0)"
    return sel


# ============================================================================ 3. one genome
def genome_job(R, dec, g, units, target_sizes, posctrl, out):
    t_start = time.time()
    a, b = int(D.OFF[g]), int(D.OFF[g + 1]); n = b - a
    FAMg = D.FAM[a:b]
    # ---- readouts: primary (dedup), off-targets, secondary (D's genes, co only)
    rkey = {}; prim = []

    def add(row, kind, f1, f2):
        k = (int(row), int(kind), int(f1), int(f2))
        if k not in rkey: rkey[k] = len(prim); prim.append(k)
        return rkey[k]
    for u in units:
        u["j"] = add(u["row"], u["sign"], u["f1"], u["f2"])
    # off-targets: other coupled elements whose readout row lies within +-35 % of D's distance from U
    need = set()
    for u in units:
        lo = u["u_end"] + int((1 - K.OFF_TOL) * u["dist"]); hi = u["u_end"] + int((1 + K.OFF_TOL) * u["dist"])
        need |= set(range(max(0, lo), min(n, hi + 1)))
    offs = [o for o in D.offtargets(g, need) if o["row"] not in {u["row"] for u in units}]
    for o in offs: o["j"] = add(o["row"], o["kind"], o["f1"], o["f2"])
    skey = {}; sec = []
    for u in units:
        u["js"] = []
        for p in u["d_sec"]:
            if p not in skey: skey[p] = len(sec); sec.append((p, int(FAMg[p])))
            u["js"].append(skey[p])
    for pc in posctrl:
        pc["j"] = add(pc["row"], 1, pc["f1"], -1)
    P_row = np.array([k[0] for k in prim], np.int64); P_kind = np.array([k[1] for k in prim])
    P_f1 = np.array([k[2] for k in prim]); P_f2 = np.array([k[3] for k in prim])
    S_row = np.array([s[0] for s in sec], np.int64); S_fam = np.array([s[1] for s in sec], np.int64)
    # (c) the readout rows are the intended genes in the unmodified chromosome
    assert np.all(FAMg[P_row[P_kind == 1]] == P_f1[P_kind == 1]), "co readout row is not D's entry family"
    assert np.all(FAMg[P_row[P_kind == -1]] == P_f2[P_kind == -1]), "avoidance readout row is not the resident"
    # ---- decoder rows for this genome (d)
    hh = dec.half_for(g)
    fams = sorted(set(P_f1.tolist()) | set(P_f2[P_f2 >= 0].tolist()) | set(S_fam.tolist()))
    fi = {f: i for i, f in enumerate(fams)}
    W = np.stack([dec.row(f, hh) for f in fams]) if fams else np.zeros((1, K.NCL))
    seen = W.sum(1) > 0
    R.set_families(W)
    i1 = np.array([fi[f] for f in P_f1]); i2 = np.array([fi[f] if f >= 0 else -1 for f in P_f2])
    iS = np.array([fi[f] for f in S_fam], np.int64)
    P_seen = seen[i1] & np.where(P_f2 >= 0, seen[np.maximum(i2, 0)], True)
    S_seen = seen[iS] if len(iS) else np.zeros(0, bool)
    ALLROWS = np.unique(np.concatenate([P_row, S_row]))
    P_pos = np.searchsorted(ALLROWS, P_row); S_pos = np.searchsorted(ALLROWS, S_row)
    # ---- influence readouts: the true family of every gene, through a sparse copy of the same decoder rows
    LASTH = {}
    if INF:
        EL_ = D.elements(g); EL_entry = EL_["entry"].astype(np.int64)
        fams_t = np.unique(FAMg); TFI = np.searchsorted(fams_t, FAMg)
        tab_ = dec.sparse_table(fams_t, hh); R.set_sparse(*tab_)
        SEEN_T = tab_[3][TFI] > 0; RGPg = D.RGP[a:b]
        BASE_LG = np.full(n, np.nan)

    def evaluate(src, extra=None, rowshift=0):
        """src: new position -> original gene (>=0) or inserted protein; returns (prim values, lp1, lp2/lc1, sec).
        rowshift != 0 reads the neighbouring row instead (the row-convention check below)."""
        src = np.asarray(src)
        orig_pos = np.full(n, -1, np.int64)
        o = np.flatnonzero(src >= 0) if extra is None else np.flatnonzero(src < n)
        # map original gene -> its new position (for inserted copies of own genes, the copy is not the original:
        # copies are passed as src values >= n below)
        orig_pos[src[o]] = o
        alive = orig_pos[ALLROWS] >= 0
        rows_new = orig_pos[ALLROWS[alive]] + rowshift
        pa_ = orig_pos[P_row] >= 0; npr = orig_pos[P_row[pa_]]
        want = np.where(P_kind[pa_] == 1, P_f1[pa_], P_f2[pa_])
        if rowshift == 0:
            # (c) after the shift, each readout row is the intended original gene, carrying the intended family.
            # This only checks the bookkeeping of the deletion; that row k is the call FOR gene k (and not for k+1)
            # is checked against the model itself in row_convention_check() below -- the assert here cannot fail by
            # construction (mechanics review finding 3).
            assert np.all(src[orig_pos[ALLROWS[alive]]] == ALLROWS[alive]), "readout row does not land on its gene after the shift"
            assert np.all(FAMg[src[npr]] == want), "primary readout row carries the wrong family after the shift"
        if len(S_row):
            sa_ = orig_pos[S_row] >= 0
            if rowshift == 0:
                assert np.all(FAMg[src[orig_pos[S_row[sa_]]]] == S_fam[sa_]), "secondary readout row: wrong family after the shift"
        if rowshift == 0: CHK["identity_rows"] += int(pa_.sum()) + (int(sa_.sum()) if len(S_row) else 0)
        srcE = np.where(src < n, src, -1 - (src - n)) if extra is not None else src
        h = R.hidden(srcE, extra)
        if INF: LASTH["h"] = h                        # read again by the influence readouts of this pass
        S = np.full((len(ALLROWS), W.shape[0]), np.nan); C = S.copy()
        if alive.any():
            s_, c_ = R.decode(h, rows_new); S[alive] = s_; C[alive] = c_
        v, lp1, l2 = K.readout_values(P_kind, S[P_pos], C[P_pos], i1, i2)
        sv = np.log(np.maximum(S[S_pos, iS], K.FLOOR)) if len(S_pos) else np.zeros(0)
        dead_p = ~alive[P_pos]; dead_s = ~alive[S_pos] if len(S_pos) else np.zeros(0, bool)
        v[dead_p] = np.nan; lp1[dead_p] = np.nan; l2[dead_p] = np.nan; sv[dead_s] = np.nan
        return v, lp1, l2, sv

    CHK = collections.Counter()

    def timed(fn, *a_):
        R.torch.cuda.synchronize(); t = time.time(); r = fn(*a_); R.torch.cuda.synchronize(); return r, time.time() - t

    R.set_genome(D.EMB[D.PID[a:b]], n + 1)
    passes = []                      # (kind, id, start, end, dels)
    passes.append(("base", -1, -1, -1, []))
    passes.append(("check_a", -1, -1, -1, []))
    kmax = int(ALLROWS.max()) if len(ALLROWS) else n - 1
    if kmax + 1 < n: passes.append(("check_b", -1, kmax + 1, min(n, kmax + 11) - 1, list(range(kmax + 1, min(n, kmax + 11)))))
    ukey = {}
    for u in units:
        if u["up"] not in ukey:
            ukey[u["up"]] = len(passes); passes.append(("U", u["up"], u["u_start"], u["u_end"], u["u_dels"]))
        u["pass"] = ukey[u["up"]]
    # no readout-row exclusion: an off-target readout IS the entry gene of a present element, so excluding
    # controls that touch a readout row would remove most coupled elements from the pool. A control that deletes a
    # readout gene simply leaves that readout undefined (NaN) and is dropped from that readout's null.
    pool = D.control_pool(g, sizes=[u["u_size"] for u in units])
    near_keys = set()
    if CTRL_NEAR:                                     # extra whole elements at about each thin unit's own distance
        near_ = D.near_controls(g, units, pool)
        near_keys = {(e["start"], e["end"], e["arm"], e["ri"]) for e in near_}
        pool = sorted(pool + near_, key=lambda c: c["start"])
    for e in pool: passes.append((f"ctrl_{e['arm']}", e["ri"], e["start"], e["end"], e["dels"]))
    NPASS = len(passes)
    # the spot id of each deleted element (region's spot for U / ctrl_cpl, the run's spot for ctrl_nat; -1 otherwise)
    rspot = lambda ri: (D.REG[ri]["spot"] if D.REG[ri]["spot"] is not None else -1)
    pass_spot = np.array([rspot(p[1]) if p[0] in ("U", "ctrl_cpl") else -1 for p in passes[:NPASS - len(pool)]]
                         + [int(e["spot"]) if e["arm"] == "nat" else rspot(e["ri"]) for e in pool], np.int32)
    V = np.full((NPASS, len(prim)), np.nan); LP1 = V.copy(); L2 = V.copy(); SV = np.full((NPASS, len(sec)), np.nan)
    secs = np.zeros(NPASS); secs_inf = np.zeros(NPASS); INFR = [None] * NPASS
    for pi, (kind, pid_, s0, e0, dels) in enumerate(passes):
        keep = np.setdiff1d(np.arange(n), np.asarray(dels, np.int64), assume_unique=True)
        (v, lp1, l2, sv), secs[pi] = timed(evaluate, keep)
        V[pi], LP1[pi], L2[pi], SV[pi] = v, lp1, l2, sv
        if INF:
            # ---- influence readouts (b), from the same hidden states; check (g)
            R.sync(); t_ = time.time(); h_ = LASTH.pop("h")
            if pi == 0:                                   # the baseline logit P(true family) of every gene
                s_, c_ = R.decode_true(h_, np.arange(n), TFI); BASE_LG[:] = K.logit_true(s_, c_)
            elif kind == "check_a":                       # every gene again, another head-chunk composition
                rr_ = np.roll(np.arange(n), -7)
                s_, c_ = R.decode_true(h_, rr_, TFI[rr_]); d_ = K.logit_true(s_, c_) - BASE_LG[rr_]
                assert np.all(d_ == 0.0), f"genome {g}: check (g): baseline re-read with another chunk composition " \
                                          f"moved (max {np.max(np.abs(d_))})"
                CHK["inf_check_a_rows"] += n
                CHK["inf_check_a_max_abs_delta"] = max(CHK["inf_check_a_max_abs_delta"], float(np.max(np.abs(d_))))
            else:
                r_ = K.influence_readout(R, h_, keep, n, dels, e0, BASE_LG, TFI, SEEN_T, RGPg, EL_entry)
                INFR[pi] = r_
                CHK["inf_passes"] += 1; CHK["inf_rows"] += r_["n_rows"]; CHK["inf_upstream_rows"] += r_["n_up"]
                CHK["inf_upstream_max_abs_delta"] = max(CHK["inf_upstream_max_abs_delta"], r_["up_max"])   # asserted == 0
            del h_; R.sync(); secs_inf[pi] = time.time() - t_
        if pi == 0: continue
        # (a) / (b) and causality in every pass: rows at or before the first deleted gene are unchanged
        d0 = min(dels) if len(dels) else n            # rows k < d0 see genes 0..k-1, all before the first deleted gene
        up_p = P_row < d0; up_s = S_row < d0
        ok = np.array_equal(V[pi, up_p], V[0, up_p], equal_nan=True) and np.array_equal(SV[pi, up_s], SV[0, up_s], equal_nan=True) \
            and np.array_equal(LP1[pi, up_p], LP1[0, up_p], equal_nan=True)
        assert ok, f"genome {g} pass {pi} ({kind}): readouts before the first deleted gene changed " \
                   f"(max {np.nanmax(np.abs(V[pi, up_p] - V[0, up_p])) if up_p.any() else 0})"
        CHK["causal_rows_exact"] += int(up_p.sum() + up_s.sum()); CHK["passes"] += 1
        if kind in ("check_a", "check_b"):
            assert up_p.all() and up_s.all()
            CHK[kind] += 1; CHK[kind + "_max_abs_delta"] = max(CHK[kind + "_max_abs_delta"], float(np.nanmax(np.abs(V[pi] - V[0])) if len(prim) else 0.0))
    # ---- (e) the row convention, checked against the model: row k must be the call FOR gene k. Decoded
    # log P(own family) at the readout row must beat the same family read one row later (an off-by-one would
    # reverse this). Asserted on the mean over this genome's readout rows.
    # the family that IS at the row: f1 for a co readout, the resident f2 for an avoidance readout
    lp_here = np.where(P_kind == 1, LP1[0], L2[0])
    (v1, lp1_n, l2_n, _s), _t = timed(evaluate, np.arange(n), None, 1)
    lp_next = np.where(P_kind == 1, lp1_n, l2_n)
    m_ = np.isfinite(lp_here) & np.isfinite(lp_next)
    if m_.sum() >= 3:
        gain = float(np.mean(lp_here[m_] - lp_next[m_]))
        assert gain > 0.5, f"genome {g}: row convention: mean log P(own family) at row k only beats row k+1 by {gain:.3f}"
        CHK["row_conv_gain"] = gain; CHK["row_conv_rows"] = int(m_.sum())
    # ---- (f) numerical floor: the same baseline at a padded length 64 longer. Two correct fp32 runs differ by up to
    # ~1e-5 (mechanics review finding 4), so a delta below this is not resolved.
    R.set_genome(D.EMB[D.PID[a:b]], n + 1 + K.HEAD_CHUNK)
    (vf, lpf, _l, _s), _t = timed(evaluate, np.arange(n))
    mf = np.isfinite(vf) & np.isfinite(V[0])
    floor_g = float(np.nanmax(np.abs(vf[mf] - V[0][mf]))) if mf.any() else np.nan
    CHK["noise_floor_max"] = floor_g
    R.set_genome(D.EMB[D.PID[a:b]], n + 1)
    # ---- positive control: insertions ~1,600 genes upstream of D (own baseline at the longer padded length)
    PC = []
    if posctrl:
        # what is inserted, smallest first: D's member genes only (what the pilot did: median 2 proteins, 37 % a
        # single protein -- mechanics review finding 6), D's whole element span, and that span with 30 genes of the
        # context that precedes D's entry in this genome. Each at ~1,600 genes upstream and at the link's OWN U-D
        # distance, against a foreign element of the same protein count at the same point (inference finding 5).
        ins_max = max(max(len(v) for v in pc["ins"].values()) for pc in posctrl)
        R.set_genome(D.EMB[D.PID[a:b]], n + 1 + ins_max)
        base2, t_b2 = timed(evaluate, np.arange(n))
        for pc in posctrl:
            j = pc["j"]; rec = dict(li=pc["li"], g=g, row=pc["row"], f1=pc["f1"], base=float(base2[0][j]),
                                    n_mem=len(pc["ins"]["mem"]), n_span=len(pc["ins"]["span"]), n_ctx=len(pc["ins"]["span30"]))
            for name, q in (("far", pc["q_far"]), ("own", pc["q_own"])):
                if q is None: continue
                rec[f"q_{name}"] = int(q); rec[f"dist_{name}"] = int(pc["row"] - q)
                for what, genes in pc["ins"].items():
                    src = np.concatenate([np.arange(q), n + np.arange(len(genes)), np.arange(q, n)])
                    extra = np.asarray(D.EMB[D.PID[a + np.asarray(genes)]], np.float32)     # copies of D's own proteins
                    (v, *_), _t = timed(evaluate, src, extra)
                    up_p = P_row < q                                      # causality: rows before the insertion unchanged
                    assert np.array_equal(v[up_p], base2[0][up_p], equal_nan=True), f"genome {g}: insertion changed an upstream readout"
                    CHK["causal_rows_exact"] += int(up_p.sum()); CHK["insertion_passes"] += 1
                    rec[f"{what}_{name}"] = float(v[j] - base2[0][j])
                ce = pc["ctrl_emb"][:len(pc["ins"]["span"])]              # foreign element, same protein count as the span
                src = np.concatenate([np.arange(q), n + np.arange(len(ce)), np.arange(q, n)])
                (v, *_), _t = timed(evaluate, src, ce)
                rec[f"ctrl_{name}"] = float(v[j] - base2[0][j]); rec[f"n_ctrl_{name}"] = len(ce)
            PC.append(rec)
    # ---- save
    dl = [np.asarray(p[4], np.int32) for p in passes]
    inf = {}
    if INF:
        # per pass: 60 bins x (mean delta, mean |delta|, mean delta over panRGP genes) as float16, genes per bin as
        # uint8; the element entries as (int16 index into el_*, float16 delta), ragged with offsets inf_e_off
        NB = K.INF_WIN // K.INF_BIN
        IM = np.full((NPASS, NB), np.nan, np.float16); IA = IM.copy(); IR = IM.copy()
        IN = np.zeros((NPASS, NB), np.uint8); INR = IN.copy()
        eo = [0]; ei_ = []; ed_ = []
        for pi, r_ in enumerate(INFR):
            if r_ is not None:
                IM[pi], IA[pi], IR[pi] = r_["mean"], r_["abs"], r_["rgp"]; IN[pi], INR[pi] = r_["n"], r_["n_rgp"]
                ei_.append(r_["ent"]); ed_.append(r_["ent_delta"])
            eo.append(eo[-1] + (len(r_["ent"]) if r_ is not None else 0))
        assert len(EL_entry) < 32767
        inf = dict(inf_mean=IM, inf_abs=IA, inf_rgp=IR, inf_n=IN, inf_n_rgp=INR, inf_e_off=np.array(eo, np.int32),
                   inf_e_el=(np.concatenate(ei_) if ei_ else np.zeros(0)).astype(np.int16),
                   inf_e_delta=(np.concatenate(ed_) if ed_ else np.zeros(0)).astype(np.float16),
                   inf_base=BASE_LG.astype(np.float32), inf_seen=SEEN_T,
                   inf_param=np.array([K.INF_WIN, K.INF_BIN, K.INF_ENT, K.INF_UP], np.int32),
                   el_kind=EL_["kind"], el_id=EL_["id"], el_start=EL_["start"], el_end=EL_["end"], el_entry=EL_["entry"],
                   el_size=EL_["size"], el_fam=EL_["fam"], pass_spot=pass_spot, secs_inf=secs_inf)
    if CTRL_NEAR:
        inf["pass_near"] = np.array([False] * (NPASS - len(pool)) +
                                    [(e["start"], e["end"], e["arm"], e["ri"]) in near_keys for e in pool], bool)
    np.savez_compressed(f"{out}/g{g}.npz", **inf,
                        P_row=P_row, P_kind=P_kind, P_f1=P_f1, P_f2=P_f2, P_seen=P_seen, S_row=S_row, S_fam=S_fam, S_seen=S_seen,
                        pass_kind=np.array([p[0] for p in passes]), pass_id=np.array([p[1] for p in passes]),
                        pass_start=np.array([p[2] for p in passes]), pass_end=np.array([p[3] for p in passes]),
                        pass_size=np.array([len(p[4]) for p in passes]), pass_dels=np.concatenate(dl) if dl else np.zeros(0, np.int32),
                        pass_off=np.concatenate([[0], np.cumsum([len(x) for x in dl])]),
                        off_j=np.array([o["j"] for o in offs], np.int64), off_ri=np.array([o["ri"] for o in offs], np.int64),
                        V=V, LP1=LP1, L2=L2, SV=SV, secs=secs, decoder_half=hh, panel_candidates=len(pool),
                        noise_floor=floor_g, wall=time.time() - t_start)
    CHK["decoder_half"] = hh; CHK["genome_half"] = int(D.HALF[g]); CHK["lineage"] = D.LIN[g]
    CHK["decoder_saw_lineage"] = D.LIN[g] in dec.lineages[hh]
    json.dump(dict(units=[{k: u[k] for k in ("li", "j", "js", "pass")} for u in units], posctrl=PC, checks=CHK),
              open(f"{out}/g{g}.json", "w"))
    return NPASS, float(secs.mean()), len(PC), float(secs_inf.mean())


# ============================================================================ 4. positive-control specs
def posctrl_specs(units, pilot_sel, rng):
    """6 pilot co-occurrence links (top z first, >= 10 genomes with D >= 1,700 genes from dnaA), 10 genomes each
    (distinct lineages first): copy D's proteins between two backbone genes ~1,600 genes upstream of D's entry (far)
    and at the link's own U-D distance (own); three insert sizes (member genes, whole span, span + 30 genes of
    preceding context); control: as many proteins of a natural accessory element of another genome"""
    by_link = collections.defaultdict(list)
    for u in units:
        # room for an insertion well upstream of D, and a real U-D distance to reproduce. The first pilot required
        # row >= 1700, which excluded every short-range link -- including the only one that came out significant.
        if u["sign"] > 0 and u["row"] >= 350 and u["dist"] >= 200: by_link[u["li"]].append(u)
    links = [li for li in pilot_sel if D.LINKS[li]["sign"] > 0 and len(by_link[li]) >= 10][:6]
    specs = collections.defaultdict(list)
    for li in links:
        us = by_link[li]; order = rng.permutation(len(us)); seen = set(); pick = []
        for k in order:
            if D.LIN[us[k]["g"]] not in seen: pick.append(us[k]); seen.add(D.LIN[us[k]["g"]])
            if len(pick) == 10: break
        for k in order:
            if len(pick) == 10: break
            if us[k] not in pick: pick.append(us[k])
        for u in pick:
            g = u["g"]; a = D.OFF[g]; fam = D.FAM[a:D.OFF[g + 1]]; bb = D.BB[fam]
            okq = np.flatnonzero(bb[:-1] & bb[1:]) + 1              # insert before q: genes q-1 and q are backbone
            def near(target, lo):
                if target < lo: return None                    # no room this far upstream of D
                c = okq[(okq >= lo) & (okq < u["row"] - 50)]
                return int(c[np.argmin(np.abs(c - target))]) if len(c) else None
            q_far = near(u["row"] - 1600, 1); q_own = near(u["row"] - u["dist"], u["u_end"] + 1)
            ed = D.EL[(u["dn"], g)]
            ins = dict(mem=D.members(u["dn"], g), span=list(range(ed["start"], ed["end"] + 1)),
                       span30=list(range(max(0, ed["start"] - 30), ed["end"] + 1)))
            # control insertion: a whole natural accessory element of another genome, >= as many proteins
            ctrl = None
            for _ in range(60):
                g2 = int(rng.integers(D.NG))
                if g2 == g: continue
                pl = [e for e in D.control_pool(g2) if e["arm"] == "nat" and e["size"] >= len(ins["span"])]
                if pl:
                    e = pl[int(rng.integers(len(pl)))]; ctrl = [D.OFF[g2] + p for p in e["dels"]]; break
            if (q_far is None and q_own is None) or ctrl is None: continue
            specs[g].append(dict(li=li, row=u["row"], f1=u["f1"], ins=ins, q_far=q_far, q_own=q_own,
                                 ctrl_emb=np.asarray(D.EMB[D.PID[np.asarray(ctrl)]], np.float32)))
    return specs


# ============================================================================ 5. run a set of links
def run(tag, links, pilot):
    import pandas as pd
    out = f"{K.KDIR}/{tag}"; os.makedirs(out, exist_ok=True)
    # one tag = one sets file and one set of readouts: resuming with another would mix per-genome parts
    pf = f"{out}/plan.json"
    new_tag = not os.path.exists(pf); old = {}
    if not new_tag:
        old = json.load(open(pf))
        parts = any(f.endswith(".npz") for f in os.listdir(out))
        if old.get("sets_md5", D.SETS_MD5 if D.SETS == K.SETS_DEFAULT else None) != D.SETS_MD5:
            sys.exit(f"{out} holds parts made with another sets file ({old.get('sets', 'pgb/region_sets.json')}); use another --tag")
        if bool(old.get("influence", False)) != INF and parts:
            sys.exit(f"{out} holds parts made with influence={old.get('influence', False)}; use another --tag")
        if int(old.get("min_apart", 100)) != K.MIN_APART and parts:
            sys.exit(f"{out} holds parts made with --min_apart {old.get('min_apart', 100)}; use another --tag")
        if old.get("max_apart") != K.MAX_APART and parts:
            sys.exit(f"{out} holds parts made with --max_apart {old.get('max_apart')}; use another --tag")
        if bool(old.get("ctrl_near", False)) != CTRL_NEAR and parts:
            sys.exit(f"{out} holds parts made with ctrl_near={old.get('ctrl_near', False)}; use another --tag")
        if CTRL_NEAR and old.get("match_log") != K.MATCH_LOG and parts:     # the near controls depend on the log tier
            sys.exit(f"{out} holds parts made with --ctrl_near and --match_log {old.get('match_log')}; use another --tag")
    units, skip, skipped = D.units(links)
    allu, _, _ = D.units(range(len(D.LINKS)))
    dsz = np.array([u["u_size"] for u in allu])
    target = np.quantile(dsz, (np.arange(K.PANEL_MAX) + 0.5) / K.PANEL_MAX, method="nearest").astype(int).tolist()
    plan = dict(links=list(links), n_units=len(units), skip=dict(skip), panel_target_sizes=target,
                u_size_median=float(np.median(dsz)), u_size_p90=float(np.percentile(dsz, 90)))
    # every new tag records its sets file (md5) and unit rules, so that a resume after the sets file changed is refused;
    # the pilot's existing plan.json (made before these keys existed) stays as it was
    if new_tag or "sets_md5" in old or D.SETS != K.SETS_DEFAULT or INF or K.MIN_APART != 100 or K.MAX_APART is not None \
            or CTRL_NEAR or K.MATCH_LOG is not None:
        plan.update(sets=os.path.relpath(D.SETS, K.PROJ), sets_md5=D.SETS_MD5, min_apart=K.MIN_APART, influence=INF,
                    influence_params=dict(window=K.INF_WIN, bin=K.INF_BIN, entries_to=K.INF_ENT, upstream_check=K.INF_UP)
                    if INF else None)
        if K.MAX_APART is not None or "max_apart" in old: plan["max_apart"] = K.MAX_APART
        if CTRL_NEAR or "ctrl_near" in old: plan["ctrl_near"] = CTRL_NEAR
        if K.MATCH_LOG is not None or "match_log" in old: plan["match_log"] = K.MATCH_LOG
    json.dump(plan, open(pf, "w"), indent=1)
    # the design asks WHY each (link, genome) pair was skipped, not only how many (mechanics review finding 7)
    pd.DataFrame(skipped).to_parquet(f"{out}/skipped.parquet", index=False)
    by_g = collections.defaultdict(list)
    for u in units: by_g[u["g"]].append(u)
    pcs = posctrl_specs(units, links, np.random.default_rng(7)) if pilot else {}
    gl = sorted(set(by_g) | set(pcs))
    if args.genomes: gl = [g for g in gl if g in {int(x) for x in args.genomes.split(",")}]
    log(f"{tag}: {len(links)} links, {len(units)} units in {len(by_g)} genomes; skipped {dict(skip)}; "
        f"positive-control genomes {len(pcs)}; panel targets median {np.median(target)} p90 {np.percentile(target, 90)}")
    R = K.Runner(frac=args.mem_frac); dec = K.Decoder(D)
    t0 = time.time(); done = 0; npass = 0
    for i, g in enumerate(gl):
        if os.path.exists(f"{out}/g{g}.npz") and os.path.exists(f"{out}/g{g}.json"): continue
        n_, sp, npc, spi = genome_job(R, dec, g, by_g.get(g, []), target, pcs.get(g, []), out)
        done += 1; npass += n_
        if done % 10 == 1 or i == len(gl) - 1 or args.genomes:
            el = time.time() - t0
            log(f"genome {g} ({i+1}/{len(gl)}): {n_} passes, {sp:.3f}s/pass" + (f" + {spi:.3f}s influence" if INF else "") +
                f"; {npass} passes in {el:.0f}s ({el/max(npass,1):.3f}s/pass incl. overhead)")
    log(f"{tag}: GPU part done")



def write_anchor_check(out, links):
    """The anchor rule for an absent element's entry, validated on the carriers of each D region, plus what the model
    says at D's real entry in those carriers (the scale the absent-element readout should be compared with)."""
    import pandas as pd
    # the anchor rule for an absent element's entry, validated on the carriers of each D region
    av = sorted({D.LINKS[li]["a"] for li in links if D.LINKS[li]["sign"] < 0} |
                {D.LINKS[li]["b"] for li in links if D.LINKS[li]["sign"] < 0})
    ar = []
    dec0 = K.Decoder(D)
    for ri in av:
        nn, sh, rule, both = D.anchor_check(ri)
        # what the model says at D's REAL entry in its carriers (from the stored fp32 baseline top-64 calls): the
        # scale the absent-element readout should be compared with
        pc = []
        for g in range(D.NG):
            if not D.TEST[ri, g]: continue
            i = int(D.OFF[g] + D.EL[(ri, g)]["start"])
            if int(D.FAM[i]) != (D.ENTRY[ri]["family"] or -1): continue
            w = dec0.row(int(D.FAM[i]), 1 - int(D.HALF[g]))
            pc.append(float((np.asarray(dec0.TP[i], np.float64) * w[np.asarray(dec0.TF[i])]).sum()))
        ar.append(dict(ri=ri, label=D.REG[ri]["label"][:40], carriers=nn, rule=rule or "", carrier_share_correct=sh,
                       pred_share_correct=both.get("pred", np.nan), succ_share_correct=both.get("succ", np.nan),
                       carrier_P_entry_median=float(np.median(pc)) if pc else np.nan, carrier_n=len(pc)))
    pd.DataFrame(ar).to_parquet(f"{out}/anchor_check.parquet", index=False)
    log("anchor rule on carriers: " + "; ".join(f"{r['ri']}({r['rule']}) {r['carrier_share_correct']:.2f}" for r in ar))


# ============================================================================ 6. aggregation
def aggregate(tag, links, pilot, sel=None):
    import pandas as pd
    from scipy import stats
    out = f"{K.KDIR}/{tag}"
    write_anchor_check(out, links)
    units, skip, skipped = D.units(links)
    pd.DataFrame(skipped).to_parquet(f"{out}/skipped.parquet", index=False)   # kept in step with the current rules
    rel = D.release_dates()
    by_g = collections.defaultdict(list)
    for u in units: by_g[u["g"]].append(u)
    rows = []; pc_rows = []; pass_secs = []; pan_sizes = []; pan_n = []; ncand = []
    inf_secs = []; inf_kb = []; inf_ent = []
    floors = []; checks = {}; off_rows = []
    arms = collections.Counter()                         # what the control pool actually holds (report; not in the summary)
    NDRAW = 2000
    for g in sorted(by_g):
        f = f"{out}/g{g}.npz"
        if not os.path.exists(f): continue
        Z = np.load(f, allow_pickle=False); J = json.load(open(f"{out}/g{g}.json"))
        pass_secs.append(Z["secs"]); ncand.append(int(Z["panel_candidates"]))
        if "secs_inf" in Z:                              # influence readouts: time of the deletion passes' readouts
            dp_ = (Z["pass_kind"] != "base") & (Z["pass_kind"] != "check_a")
            inf_secs.append(Z["secs_inf"][dp_]); inf_kb.append(os.path.getsize(f) / 1024); inf_ent.append(int(len(Z["inf_e_el"])))
        pk = Z["pass_kind"]; ps = Z["pass_size"]; pst = Z["pass_start"]; pen = Z["pass_end"]; pid_ = Z["pass_id"]
        V, LP1, L2, SV = Z["V"], Z["LP1"], Z["L2"], Z["SV"]
        base = V[0]
        pan = np.flatnonzero((pk == "ctrl_cpl") | (pk == "ctrl_nat"))
        pan_sizes += ps[pan].tolist(); pan_n.append(len(pan))
        arms["ctrl_cpl"] += int((pk == "ctrl_cpl").sum()); arms["ctrl_nat"] += int((pk == "ctrl_nat").sum())
        arms["genomes"] += 1; arms["genomes_without_nat"] += int(not (pk == "ctrl_nat").any())
        if "pass_near" in Z: arms["near"] += int(Z["pass_near"].sum())
        fl = float(Z["noise_floor"]) if "noise_floor" in Z else np.nan; floors.append(fl)
        P_row, P_kind, P_seen = Z["P_row"], Z["P_kind"], Z["P_seen"]
        dV = V - base[None]
        off_j = Z["off_j"] if "off_j" in Z else np.zeros(0, np.int64)
        off_ri = Z["off_ri"] if "off_ri" in Z else np.zeros(0, np.int64)
        # the null for one deletion at one readout: whole accessory elements of this genome, matched on size and
        # distance, excluding overlaps with U and D and regions linked to either
        c_end = pen[pan]; c_size = ps[pan]; c_ri = pid_[pan]

        def null_ok(u_start, u_end, d_start, d_end, banned):
            o = ~((pst[pan] <= u_end) & (pen[pan] >= u_start)) & ~((pst[pan] <= d_end) & (pen[pan] >= d_start))
            if banned: o &= ~np.isin(c_ri, list(banned))
            return o

        def stat(j, delta, size, dist, row, ok, e_sign, flo):
            """robust z, rank percentile and the leave-one-out pseudo-U z values of the matched controls"""
            idx, wid = K.match_controls(c_end, c_size, ok & np.isfinite(dV[pan, j]), size, dist, row)
            cv = dV[pan[idx], j]
            z, med, mad = K.robust_z(delta, cv, flo)
            rp = K.rank_pct(delta, cv, e_sign)
            ps_ = {}
            for e in idx:                                     # each control in turn as a pseudo-U, keyed by the control
                ok2 = ok.copy(); ok2[e] = False
                ok2 &= ~((pst[pan] <= pen[pan][e]) & (pen[pan] >= pst[pan][e]))
                i2, _w = K.match_controls(c_end, c_size, ok2 & np.isfinite(dV[pan, j]), c_size[e], row - c_end[e], row)
                if len(i2) < 3: continue
                z2, _m, _d = K.robust_z(dV[pan[e], j], dV[pan[i2], j], flo)
                r2 = K.rank_pct(dV[pan[e], j], dV[pan[i2], j], e_sign)
                ps_[int(e)] = (float(z2 * e_sign), float(r2 - 0.5))
            return dict(z=z, n_ctrl=len(idx), widened=wid, ctrl_med=med, ctrl_mad=mad, rank_pct=rp,
                        ctrl=cv, pseudo=ps_)
        for u, ju in zip(by_g[g], J["units"]):
            assert ju["li"] == u["li"]
            j = ju["j"]; pi = ju["pass"]; e_sign = -1 if u["sign"] > 0 else 1
            delta = float(dV[pi, j])
            banned = {u["up"], u["dn"]} | D.links_of(u["up"]) | D.links_of(u["dn"])
            ok = null_ok(u["u_start"], u["u_end"], u["d_start"], u["d_end"], banned)
            flo = max(fl if np.isfinite(fl) else 0.0, 1e-6)
            st = stat(j, delta, u["u_size"], u["dist"], u["row"], ok, e_sign, flo)
            ctrl = st["ctrl"]
            # ---- off-targets: the same U pass read at other elements at a comparable distance. A U whose deletion
            # moves every element downstream earns "knows" on any link it belongs to (inference review blocker).
            lo = u["u_end"] + (1 - K.OFF_TOL) * u["dist"]; hi = u["u_end"] + (1 + K.OFF_TOL) * u["dist"]
            zo = []; zo_ps = []; do = []; do_ps = {}
            cand = [(jj, ri) for jj, ri in zip(off_j, off_ri)
                    if lo <= P_row[jj] <= hi and P_seen[jj] and P_kind[jj] == u["sign"] and ri not in banned
                    and np.isfinite(dV[pi, jj])]
            cand.sort(key=lambda t: abs(P_row[t[0]] - u["row"]))
            for jj, ri in cand[:12]:
                ed = D.EL[(ri, g)] if D.TEST[ri, g] else None
                ds, de = (int(ed["start"]), int(ed["end"])) if ed is not None else (int(P_row[jj]), int(P_row[jj]))
                ok2 = null_ok(u["u_start"], u["u_end"], ds, de, {u["up"], ri} | D.links_of(ri))
                s2 = stat(jj, float(dV[pi, jj]), u["u_size"], int(P_row[jj] - u["u_end"]), int(P_row[jj]), ok2, e_sign, flo)
                if not np.isfinite(s2["z"]): continue
                zo.append(s2["z"] * e_sign); zo_ps.append(s2["pseudo"])
                # the same comparison in the raw readout units: a z is scale-free, so a readout whose own null happens
                # to be tight can look special without having moved much. Both are reported, and "knows" needs both.
                do.append(float(dV[pi, jj]) * e_sign)
                for e in st["pseudo"]: do_ps.setdefault(int(e), []).append(float(dV[pan[int(e)], jj]) * e_sign)
                off_rows.append(dict(li=u["li"], g=g, lineage=D.LIN[g], ri=int(ri), row=int(P_row[jj]),
                                     dist=int(P_row[jj] - u["u_end"]), delta=float(dV[pi, jj]), z_exp=s2["z"] * e_sign,
                                     phi_U=D.phi(u["up"], int(ri))))
            if u["sign"] > 0 and ju["js"]:
                sd_ = SV[pi, ju["js"]] - SV[0, ju["js"]]; sec_mean = float(np.nanmean(sd_)); sec_list = sd_.round(5).tolist()
            else:
                sec_mean = np.nan; sec_list = []
            # pseudo-U draws for the calibration: the same statistics with one control in U's place. Keyed by the
            # control, so the pseudo-U's z at D and at the off-targets come from the SAME deletion.
            pz = []; prk = []; poff = []; poff_raw = []
            for e, (z2, r2) in st["pseudo"].items():
                pz.append(round(z2, 4)); prk.append(round(r2, 4))
                oz = [zo_ps[k][e][0] for k in range(len(zo_ps)) if e in zo_ps[k]]
                poff.append(round(float(np.mean(oz)), 4) if oz else None)
                dr = do_ps.get(int(e))
                poff_raw.append([round(float(dV[pan[int(e)], j]) * e_sign, 9), round(float(np.mean(dr)), 9)] if dr else None)
            rr = dict(li=u["li"], g=g, acc=D.ACC[g], lineage=D.LIN[g], half=int(D.HALF[g]), rel_date=rel[g],
                      sign=u["sign"], up=u["up"], dn=u["dn"], u_size=u["u_size"], u_span=u["u_span"], dist=u["dist"],
                      row=u["row"], row_how=u["row_how"], f1=D.FN[u["f1"]], f2=(D.FN[u["f2"]] if u["f2"] >= 0 else ""),
                      u_fam=u["u_fam"], u_left=u["u_left"], f1_up=u["f1_up"],
                      seen=bool(P_seen[j]), base=float(base[j]), ko=float(V[pi, j]), delta=delta,
                      d_lp1=float(LP1[pi, j] - LP1[0, j]), d_l2=float(L2[pi, j] - L2[0, j]),
                      base_p1=float(np.exp(LP1[0, j])), noise_floor=fl, resolved=bool(abs(delta) > flo),
                      n_ctrl=st["n_ctrl"], widened=bool(st["widened"]), ctrl=json.dumps([round(float(c), 7) for c in ctrl]),
                      ctrl_med=st["ctrl_med"], ctrl_mad=st["ctrl_mad"], z=st["z"], rank_pct=st["rank_pct"],
                      ctrl_mean=float(np.mean(ctrl)) if len(ctrl) else np.nan,
                      n_off=len(zo), z_off=float(np.mean(zo)) if zo else np.nan,
                      d_off=float(np.mean(do)) if do else np.nan,
                      pseudo_z=json.dumps(pz), pseudo_rank=json.dumps(prk), pseudo_off=json.dumps(poff),
                      pseudo_off_raw=json.dumps(poff_raw),
                      sec_mean=sec_mean, sec=json.dumps(sec_list))
            rows.append(rr)
        for r in J["posctrl"]: pc_rows.append(r)
        for k_, v_ in J["checks"].items():
            if k_ in ("row_conv_gain", "noise_floor_max"): checks.setdefault(k_ + "_list", []).append(v_)
            elif k_.endswith("max_abs_delta"): checks[k_] = max(checks.get(k_, 0.0), v_)
            elif isinstance(v_, bool): checks[k_] = checks.get(k_, 0) + int(v_)
            elif isinstance(v_, (int, float)) and k_ not in ("decoder_half", "genome_half"): checks[k_] = checks.get(k_, 0) + v_
        checks["genomes"] = checks.get("genomes", 0) + 1
        checks["decoder_other_half"] = checks.get("decoder_other_half", 0) + int(J["checks"]["decoder_half"] != J["checks"]["genome_half"])
    for k_ in ("row_conv_gain", "noise_floor_max"):
        v_ = checks.pop(k_ + "_list", None)
        if v_: checks[k_ + "_min"] = float(np.min(v_)); checks[k_ + "_median"] = float(np.median(v_)); checks[k_ + "_max"] = float(np.max(v_))
    U = pd.DataFrame(rows)
    U["exp_sign"] = np.where(U.sign > 0, -1, 1)
    U["z_exp"] = U.z * U.exp_sign; U["delta_exp"] = U.delta * U.exp_sign
    U["spec"] = U.z_exp - U.z_off                     # does D move more than the other elements in the same pass?
    U["spec_raw"] = U.delta_exp - U.d_off             # the same comparison in the raw readout units
    U["recent"] = U.rel_date >= "2024-01-01"
    lin_old = set(U.loc[~U.recent, "lineage"]); U["new_lin"] = U.recent & ~U.lineage.isin(lin_old)
    U["full_ko"] = U.u_left == 0                      # the deletion removed every copy of U's families
    U.to_parquet(f"{K.KDIR}/{tag}_units.parquet", index=False)
    OT = pd.DataFrame(off_rows)
    if len(OT): OT.to_parquet(f"{K.KDIR}/{tag}_offtarget.parquet", index=False)
    PCd = pd.DataFrame(pc_rows)
    if len(PCd): PCd.to_parquet(f"{K.KDIR}/{tag}_posctrl.parquet", index=False)

    def wil(x, alt):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        return float(stats.wilcoxon(x, alternative=alt).pvalue) if len(x) >= 5 and np.any(x != 0) else np.nan

    def link_stats(ut, col):
        """lineage means of `col` -> one-sided p each way"""
        lm = ut.groupby("lineage")[col].mean().dropna()
        return lm, wil(lm.values, "greater"), wil(lm.values, "less")
    rng = np.random.default_rng(3)
    # ---- per link
    lk = []
    for li in links:
        l = D.LINKS[li]; u = U[U.li == li]
        ut = u[u.seen & np.isfinite(u.z_exp) & u.full_ko]          # units with a readout, a null, and a complete knockout
        lm, p_exp, p_opp = link_stats(ut, "z_exp")
        lh = ut.groupby("lineage").half.first()
        # the same test with the partial knockouts put back, so both readings are on the record
        uwp = u[u.seen & np.isfinite(u.z_exp)]
        lmw, p_exp_wp, _o = link_stats(uwp, "z_exp")
        us = ut[np.isfinite(ut.spec)]; lms, p_spec, _ = link_stats(us, "spec")
        usr = ut[np.isfinite(ut.spec_raw)]; lmsr, p_spec_raw, p_spec_raw_opp = link_stats(usr, "spec_raw")
        rec = dict(li=li, sign=l["sign"], link_z=l["z"], pairs=l["pairs"], why=(sel or {}).get(li, ""),
                   a=D.REG[l["a"]]["label"][:40], b=D.REG[l["b"]]["label"][:40],
                   n_units=len(u), n_unseen=int((~u.seen).sum()), n_partial=int((~u.full_ko).sum()),
                   n_z=len(ut), n_lin=len(lm), n_widened=int(ut.widened.sum()),
                   median_delta=float(u[u.seen].delta.median()) if u.seen.any() else np.nan,
                   median_z=float(ut.z.median()) if len(ut) else np.nan,
                   median_z_exp_lin=float(lm.median()) if len(lm) else np.nan,
                   median_z_exp_lin_withpartial=float(lmw.median()) if len(lmw) else np.nan, p_exp_withpartial=p_exp_wp,
                   share_resolved=float(u[u.seen].resolved.mean()) if u.seen.any() else np.nan,
                   share_beyond95=float((ut.rank_pct >= 0.95).mean()) if len(ut) else np.nan,
                   share_beyond95_null=K.rank_share_null(ut.n_ctrl.values) if len(ut) else np.nan,
                   median_base_p1=float(u.base_p1.median()) if len(u) else np.nan,
                   share_f1_upstream=float((u.f1_up > 1).mean()) if len(u) else np.nan,
                   share_resident_floored=float(((np.log(u.base_p1.clip(lower=1e-300)) - u.base) <= np.log(K.FLOOR) + 1e-6).mean())
                   if (len(u) and l["sign"] < 0) else 0.0,
                   n_off_units=len(us), median_spec_lin=float(lms.median()) if len(lms) else np.nan,
                   median_z_off_lin=float(us.groupby("lineage").z_off.mean().median()) if len(us) else np.nan,
                   p_exp=p_exp, p_opp=p_opp, p_spec=p_spec, p_spec_raw=p_spec_raw, p_spec_raw_opp=p_spec_raw_opp,
                   median_spec_raw_lin=float(lmsr.median()) if len(lmsr) else np.nan,
                   median_d_off_lin=float(usr.groupby("lineage").d_off.mean().median()) if len(usr) else np.nan)
        for h in (0, 1):
            m = lh == h; rec[f"half{h}_n_lin"] = int(m.sum()); rec[f"half{h}_median"] = float(lm[m].median()) if m.any() else np.nan
        for nm, col in (("recent", "recent"), ("newlin", "new_lin")):
            ur = ut[ut[col]]; lmr = ur.groupby("lineage").z_exp.mean()
            rec[f"{nm}_n_units"] = len(ur); rec[f"{nm}_n_lin"] = len(lmr)
            rec[f"{nm}_median"] = float(lmr.median()) if len(lmr) else np.nan
        if len(lm) >= 5:
            npos = int((lm > 0).sum()); nnz = int((lm != 0).sum())
            rec["p_sign"] = float(stats.binomtest(npos, nnz, 0.5, alternative="greater").pvalue) if nnz else np.nan
            rec["mde_z"] = K.mde_wilcoxon(float(lm.std(ddof=1)), len(lm), alpha=0.05 / max(1, len(links)))
            # ---- empirical calibration: the same test with a control deletion in U's place. The tails of the
            # matched null are skewed, so the nominal p is not the real one (inference review finding 2).
            uu = ut[ut.pseudo_z != "[]"]
            PZ = [json.loads(s) for s in uu.pseudo_z]; PO = [json.loads(s) for s in uu.pseudo_off]
            PR = [json.loads(s) for s in uu.pseudo_off_raw]
            LI = uu.lineage.values
            # the observed p on exactly the units that have pseudo-U draws, so that the null is the same test
            lmu, p_obs, p_obs_o = link_stats(uu, "z_exp")
            _lms, p_obs_s, _x = link_stats(uu[np.isfinite(uu.spec)], "spec")
            _lmr, p_obs_r, p_obs_ro = link_stats(uu[np.isfinite(uu.spec_raw)], "spec_raw")
            # how global is the deletion? the mean shift over the OTHER readouts of the same pass, for U and for its
            # matched controls. If U's global shift is much larger, its excess at D can be entirely that.
            gU = []; gC = []
            for s_, ru in zip(uu.d_off.values, PR):
                v = [x[1] for x in ru if x is not None]
                if v and np.isfinite(s_): gU.append(s_); gC.append(float(np.mean(v)))
            rec["global_shift_U"] = float(np.median(gU)) if gU else np.nan
            rec["global_shift_ctrl"] = float(np.median(gC)) if gC else np.nan
            if gU:
                gl = pd.DataFrame(dict(lin=uu.lineage.values[:len(gU)], d=np.array(gU) - np.array(gC))).groupby("lin").d.mean()
                rec["p_global"] = wil(gl.values, "greater")     # does U shift the OTHER elements more than its controls do?
            sc = [float(np.mean([x[0] - x[1] for x in ru if x is not None])) for ru in PR if any(x is not None for x in ru)]
            rec["global_shift_ctrl_specraw"] = float(np.median(sc)) if sc else np.nan
            rec["n_lin_cal"] = len(lmu); rec["p_exp_cal_ref"] = p_obs; rec["p_spec_cal_ref"] = p_obs_s
            rec["p_spec_raw_cal_ref"] = p_obs_r
            # vectorised over draws: one pseudo-U per unit per draw, lineage means by matrix product, then the same
            # one-sided signed-rank test on every draw at once. NDRAW must be large enough that the calibrated p can
            # fall below the BH threshold (its floor is 1 / (NDRAW + 1)).
            nu = len(PZ); kmax = max(len(z) for z in PZ)
            Zm = np.full((nu, kmax), np.nan); Sm = np.full((nu, kmax), np.nan); Rm = np.full((nu, kmax), np.nan)
            for i_, (z, o, r) in enumerate(zip(PZ, PO, PR)):
                Zm[i_, :len(z)] = z
                Sm[i_, :len(z)] = [(zz - oo) if oo is not None else np.nan for zz, oo in zip(z, o)]
                Rm[i_, :len(z)] = [(rr[0] - rr[1]) if rr is not None else np.nan for rr in r]
            nk = np.array([len(z) for z in PZ])
            codes, luniq = pd.factorize(LI)
            ind = np.zeros((nu, len(luniq))); ind[np.arange(nu), codes] = 1.0
            picks = (rng.random((NDRAW, nu)) * nk).astype(int)
            vz = Zm[np.arange(nu)[None, :], picks]                     # (NDRAW, nu)
            vs = Sm[np.arange(nu)[None, :], picks]; vr = Rm[np.arange(nu)[None, :], picks]
            mz = (vz @ ind) / ind.sum(0)[None, :]                      # lineage means, (NDRAW, n_lin)
            oks = np.isfinite(vs); ns = oks @ ind
            with np.errstate(invalid="ignore", divide="ignore"):
                ms = np.where(ns > 0, (np.where(oks, vs, 0.0) @ ind) / np.maximum(ns, 1), np.nan)
                okr = np.isfinite(vr); nr = okr @ ind
                mr = np.where(nr > 0, (np.where(okr, vr, 0.0) @ ind) / np.maximum(nr, 1), np.nan)

            def wil_rows(M, alt):
                """one-sided signed-rank p per row, NaN where the row has fewer than 5 usable lineages"""
                out = np.full(len(M), np.nan)
                good = np.isfinite(M).sum(1) >= 5
                if good.any():
                    with np.errstate(all="ignore"):
                        out[good] = stats.wilcoxon(M[good], alternative=alt, axis=1, nan_policy="omit").pvalue
                return out
            pe = wil_rows(mz, "greater"); po = wil_rows(mz, "less"); psp = wil_rows(ms, "greater")
            psr = wil_rows(mr, "greater"); pso = wil_rows(mr, "less")
            rec["p_cal_exp"] = float((1 + np.sum(pe[np.isfinite(pe)] <= p_obs)) / (1 + np.isfinite(pe).sum())) if np.isfinite(p_obs) else np.nan
            rec["p_cal_opp"] = float((1 + np.sum(po[np.isfinite(po)] <= p_obs_o)) / (1 + np.isfinite(po).sum())) if np.isfinite(p_obs_o) else np.nan
            rec["p_cal_spec"] = float((1 + np.sum(psp[np.isfinite(psp)] <= p_obs_s)) / (1 + np.isfinite(psp).sum())) if np.isfinite(p_obs_s) else np.nan
            rec["p_cal_spec_raw"] = float((1 + np.sum(psr[np.isfinite(psr)] <= p_obs_r)) / (1 + np.isfinite(psr).sum())) if np.isfinite(p_obs_r) else np.nan
            rec["p_cal_spec_raw_opp"] = float((1 + np.sum(pso[np.isfinite(pso)] <= p_obs_ro)) / (1 + np.isfinite(pso).sum())) if np.isfinite(p_obs_ro) else np.nan
            rec["cal_fp_exp"] = float(np.mean(pe[np.isfinite(pe)] < 0.05)); rec["cal_fp_opp"] = float(np.mean(po[np.isfinite(po)] < 0.05))
            rec["cal_draws"] = int(np.isfinite(pe).sum())
        else:
            for k_ in ("p_sign", "mde_z", "p_cal_exp", "p_cal_opp", "p_cal_spec", "p_cal_spec_raw", "cal_fp_exp", "cal_fp_opp",
                       "p_global", "p_cal_spec_raw_opp"): rec[k_] = np.nan
            rec["cal_draws"] = 0
            rec["nt_why"] = ("no testable genome" if not len(u) else
                             "no readout (D's entry family unseen by the other half's decoder)" if not u.seen.any() else
                             "every unit is a partial knockout (U's families remain elsewhere)" if not u.full_ko.any() else
                             "no unit with >= 3 matched controls" if not np.isfinite(u.z_exp).any() else
                             f"{len(lm)} lineages < 5")
        lk.append(rec)
    Lk = pd.DataFrame(lk)
    for c in ("p_exp", "p_opp", "p_cal_exp", "p_cal_opp", "p_cal_spec", "p_cal_spec_raw", "p_cal_spec_raw_opp"):
        Lk["q" + c[1:]] = K.bh(Lk[c].values)

    def verdict(r):
        if not np.isfinite(r.p_exp): return "nt"
        if not (r.share_resolved > 0.5): return "nt"          # deltas below the fp32 floor: nothing was measured
        own = (r.q_cal_exp < 0.05 and r.p_sign < 0.05 and r.half0_median > 0 and r.half1_median > 0)
        if own and r.q_cal_spec < 0.05 and r.q_cal_spec_raw < 0.05: return "knows"
        # the deletion moves D more than other deletions move D, but not more than it moves the other elements it
        # hits: a genome-wide effect of removing U, which would earn "knows" on every link U belongs to
        if own: return "not specific"
        opp = (r.q_cal_opp < 0.05 and r.half0_median < 0 and r.half1_median < 0)
        if opp and r.q_cal_spec_raw_opp < 0.05: return "opposite"
        if opp: return "not specific (opposite)"
        return "none"
    Lk["verdict"] = Lk.apply(verdict, axis=1)
    Lk["nt_why"] = Lk.apply(lambda r: (r.get("nt_why") if isinstance(r.get("nt_why"), str) else
                                       ("deltas below the fp32 numerical floor" if r.verdict == "nt" else "")), axis=1)
    Lk.to_parquet(f"{K.KDIR}/{tag}_links.parquet", index=False)
    # ---- global over links
    tst = Lk[np.isfinite(Lk.p_exp) & (Lk.verdict != "nt")]
    glob = {}
    if len(tst) >= 3:
        eff = tst.median_z_exp_lin.values
        # links share regions (the T6SS region is D in 15, 20 and 162), so also test one link per region pair
        seen_r = set(); keep = []
        for i, r in tst.reset_index().iterrows():
            key = tuple(sorted((int(D.LINKS[int(r.li)]["a"]), int(D.LINKS[int(r.li)]["b"]))))
            rg = {int(D.LINKS[int(r.li)]["a"]), int(D.LINKS[int(r.li)]["b"])}
            if rg & seen_r: continue
            seen_r |= rg; keep.append(i)
        eff_i = eff[keep]
        glob = dict(n_links=len(tst), median_effect=float(np.median(eff)),
                    wilcoxon_p_exp=wil(eff, "greater"), sign_pos=int((eff > 0).sum()),
                    sign_p=float(stats.binomtest(int((eff > 0).sum()), len(eff), 0.5, alternative="greater").pvalue),
                    n_links_region_disjoint=len(eff_i), median_effect_disjoint=float(np.median(eff_i)),
                    wilcoxon_p_exp_disjoint=wil(eff_i, "greater"),
                    spearman_vs_link_absz=[float(x) for x in stats.spearmanr(eff, np.abs(tst.link_z.values))],
                    spearman_vs_pairs=[float(x) for x in stats.spearmanr(eff, tst.pairs.values)])
        if len(OT) and OT.phi_U.notna().sum() >= 20:
            m_ = OT.groupby("ri").agg(phi=("phi_U", "first"), z=("z_exp", "median"), n=("z_exp", "size")).query("n >= 5")
            if len(m_) >= 6:
                glob["offtarget_vs_phi_spearman"] = [float(x) for x in stats.spearmanr(m_.phi, m_.z)]
    secs = np.concatenate(pass_secs) if pass_secs else np.zeros(1)
    summ = dict(tag=tag, links=len(links), units=len(U), genomes=int(U.g.nunique()) if len(U) else 0, skip=dict(skip),
                passes=int(len(secs)), seconds_per_pass_median=float(np.median(secs)), seconds_per_pass_mean=float(secs.mean()),
                gpu_seconds=float(secs.sum()),
                **(dict(influence=dict(seconds_per_pass_median=float(np.median(np.concatenate(inf_secs))),
                                       seconds_total=float(np.concatenate(inf_secs).sum()),
                                       genome_file_kb_median=float(np.median(inf_kb)), entries=int(np.sum(inf_ent)),
                                       entries_per_genome_median=float(np.median(inf_ent))))
                   if inf_secs else {}),               # no key without influence readouts: the pilot's summary is unchanged
                noise_floor=dict(median=float(np.nanmedian(floors)), p90=float(np.nanpercentile(floors, 90)),
                                 share_units_resolved=float(U[U.seen].resolved.mean()) if U.seen.any() else np.nan),
                panel=dict(per_genome_median=float(np.median(pan_n)) if pan_n else 0, candidates_median=float(np.median(ncand)) if ncand else 0,
                           size_median=float(np.median(pan_sizes)) if pan_sizes else 0, size_p90=float(np.percentile(pan_sizes, 90)) if pan_sizes else 0),
                matched_ctrl=dict(median=float(U.n_ctrl.median()), share_ge3=float((U.n_ctrl >= 3).mean()),
                                  share_widened=float(U.widened.mean()), share_0=float((U.n_ctrl == 0).mean())),
                partial_ko=dict(share_units=float((~U.full_ko).mean()), links_all_partial=int((Lk.n_partial == Lk.n_units).sum())),
                offtarget=dict(units_with_off=int(U.n_off.gt(0).sum()), median_n_off=float(U.n_off.median())),
                calibration=dict(median_fp_exp=float(np.nanmedian(Lk.cal_fp_exp)), median_fp_opp=float(np.nanmedian(Lk.cal_fp_opp)),
                                 draws=int(NDRAW)),
                verdicts=Lk.verdict.value_counts().to_dict(), global_test=glob, checks=checks)
    json.dump(summ, open(f"{K.KDIR}/{tag}_summary.json", "w"), indent=1, default=float)
    summ["_arcs"] = write_arcs(tag, U, Lk, dict(arms))   # a separate file: the summary above is unchanged
    return U, Lk, PCd, summ


MODEL_REACH = ("the model's content-specific use of the context decays with distance: a 2-gene deletion moves the logit "
               "of the gene read by 6.3 at 1 gene, 0.044 at 10, 0.014 at 100 and 0.006 at 1,000 genes (pilot --decay)")


def write_arcs(tag, U, Lk, arms):
    """{tag}_arcs.json: one record per link for the page's arcs, the model's verdict with what it can and cannot mean:
    the U-D distances actually tested (units with a readout, a matched null and a complete knockout), the share within
    500 genes (the model's range), the smallest effect the test detects (mde_z, z units), the calibrated q values, and
    with --flags what the link is in the genomes (pg_link_flags.py): one_site (one element at different spots), tract
    (< 100 genes apart in co-carriers), site_competition (avoidance of two elements for one insertion site),
    rep_lineage (holds in lineage-disjoint halves). `caveats` spells them out."""
    flags = {}
    if args.flags:
        FJ = json.load(open(args.flags))
        for r in FJ["links"]:
            l = D.LINKS[r["li"]]
            assert (r["a"], r["b"], r["sign"]) == (l["a"], l["b"], l["sign"]), f"{args.flags} does not describe {D.SETS} (link {r['li']})"
            flags[r["li"]] = r
        assert len(flags) == len(D.LINKS) or tag == "pilot", f"{args.flags}: {len(flags)} links, the sets file has {len(D.LINKS)}"
    ok = U.seen & np.isfinite(U.z_exp) & U.full_ko if len(U) else None
    arcs = []
    for _, r in Lk.iterrows():
        li = int(r.li); u = U[U.li == li] if len(U) else U; ut = u[ok[u.index]] if len(u) else u
        rec = dict(li=li, a=int(D.LINKS[li]["a"]), b=int(D.LINKS[li]["b"]), sign=int(r.sign), link_z=float(r.link_z),
                   verdict=r.verdict, nt_why=r.nt_why if isinstance(r.nt_why, str) else "",
                   n_units=int(r.n_units), n_tested=int(len(ut)), n_lin=int(r.n_lin),
                   dist_median=float(ut.dist.median()) if len(ut) else (float(u.dist.median()) if len(u) else None),
                   dist_q10=float(ut.dist.quantile(0.1)) if len(ut) else None, dist_q90=float(ut.dist.quantile(0.9)) if len(ut) else None,
                   share_within_500=float((ut.dist <= 500).mean()) if len(ut) else None,
                   units_within_500=int((ut.dist <= 500).sum()) if len(ut) else 0,
                   effect_z=float(r.median_z_exp_lin) if np.isfinite(r.median_z_exp_lin) else None,
                   mde_z=float(r.mde_z) if np.isfinite(r.get("mde_z", np.nan)) else None,
                   q_cal_exp=float(r.q_cal_exp) if np.isfinite(r.get("q_cal_exp", np.nan)) else None,
                   q_cal_spec=float(r.q_cal_spec) if np.isfinite(r.get("q_cal_spec", np.nan)) else None,
                   q_cal_spec_raw=float(r.q_cal_spec_raw) if np.isfinite(r.get("q_cal_spec_raw", np.nan)) else None)
        cav = []
        if rec["verdict"] in ("none", "nt") and rec["dist_median"] is not None and rec["dist_median"] > 500:
            cav.append(f"tested at a median {rec['dist_median']:.0f} genes, beyond the model's range: 'none' here means the "
                       f"test could not see it, not that the model does not know the link")
        f_ = flags.get(li)
        if f_:
            rec.update({k: f_[k] for k in ("one_site", "tract", "separate", "site_competition", "rep_lineage", "rep_lineage_pairs",
                                           "carrier_dist", "same_spot", "sep", "carriers")})
            if f_["one_site"]: cav.append(f"one element, not two regions: in the genomes carrying both, a copy of each sits in one "
                                          f"spot (median nearest distance {f_['carrier_dist']:.0f} genes)")
            elif f_["tract"]: cav.append(f"possibly one transfer tract: {f_['carrier_dist']:.0f} genes apart in the genomes carrying both")
            if f_["site_competition"]: cav.append("the two elements share an insertion site: they may exclude each other physically")
            if not f_["rep_lineage"]: cav.append("not replicated in two lineage-disjoint halves (the default replication halves "
                                                 "shared lineages)")
        rec["caveats"] = cav
        arcs.append(rec)
    vc = collections.Counter(a["verdict"] for a in arcs)
    cross = {}
    if flags:
        for k in ("one_site", "tract", "site_competition", "rep_lineage"):
            cross[k] = dict(collections.Counter(f"{a['verdict']}|{a.get(k)}" for a in arcs))
    uu = U[ok] if len(U) else U
    band = {}
    if len(U):
        for lo_, hi_ in ((0, 100), (100, 250), (250, 500), (500, 1000), (1000, 10 ** 9)):
            m = U.seen & U.full_ko & (U.dist > lo_) & (U.dist <= hi_)
            if m.any(): band[f"{lo_}-{hi_ if hi_ < 10 ** 9 else 'max'}"] = dict(units=int(m.sum()), share_ge3=float((U[m].n_ctrl >= 3).mean()))
    meta = dict(tag=tag, sets=os.path.relpath(D.SETS, K.PROJ), sets_md5=D.SETS_MD5, flags=args.flags, links=len(arcs),
                verdicts=dict(vc), verdict_by_flag=cross, arms=arms, matched_ctrl_by_distance=band,
                units_tested=int(len(uu)), units_tested_within_500=int((uu.dist <= 500).sum()) if len(uu) else 0,
                rules=dict(min_apart=K.MIN_APART, max_apart=K.MAX_APART, ctrl_near=CTRL_NEAR, match_log=K.MATCH_LOG),
                model_reach=MODEL_REACH)
    json.dump(dict(meta=meta, arcs=arcs), open(f"{K.KDIR}/{tag}_arcs.json", "w"), indent=0, default=float)
    log(f"arcs -> {K.KDIR}/{tag}_arcs.json: {dict(vc)}")
    return meta


# ============================================================================ 7. report
def write_report(tag, U, Lk, PCd, summ, sel):
    import pandas as pd
    out = f"{K.KDIR}/{tag}"
    bm = json.load(open(f"{K.KDIR}/base_meta.json"))
    plan = json.load(open(f"{out}/plan.json"))
    ck = summ["checks"]
    f = lambda x, d=2: ("" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}")
    L = []
    L.append(f"# In-silico knockout of the coupled regions: {tag}\n")
    L.append(f"Model `{K.MODEL}` (causal), fp32 weights and activations, one pass per whole chromosome (batch 1, "
             f"padded to n + 1 positions per genome), readouts decoded in float64. Generated by `pg_knockout.py --{'pilot' if sel else 'all'}` "
             f"on {time.strftime('%Y-%m-%d %H:%M')}.\n")
    L.append("## What was measured\n")
    L.append("- **co-occurrence** links: genomes carrying both elements at their spots; the upstream element U (dnaA reading order, "
             "per genome) is deleted (member genes only, never a backbone family); readout = logit P(D entry family) at D's entry gene. "
             "Expected: the knockout LOWERS it.")
    L.append("- **avoidance** links: genomes carrying U and cleanly lacking D (no D family anywhere, none at D's spot); readout = "
             "log P(D entry family) - log P(resident family) at the row where D's entry WOULD sit, placed from the carriers' "
             "context (1 + the gene of the family that precedes D's entry in >= 60 % of carriers, nearest copy within 40 genes; "
             "else the gene of the family that follows D's last gene). Expected: the knockout RAISES it.")
    L.append("- P(family | context) = sum over the full 50,001-way softmax of P(cluster | context) x P(family | cluster); the decoder "
             "P(family | cluster) comes from the fp32 baseline top-64 calls of the other lineage half.")
    L.append("- delta = readout(knockout) - readout(baseline). z = (delta - median of matched controls) / (1.4826 x MAD), floored at the "
             "genome's numerical floor; z_exp = z x expected sign (> 0 = the expected direction). Matched controls: WHOLE accessory "
             "elements of the same genome (every coupled element present at its spot, plus every whole panRGP run at a spot with no "
             "coupled region), upstream of D, size x0.5-x2 of U's, distance to D within +-35 % of U's, not overlapping U or D and not "
             "belonging to a region linked to U or D; windows widened to x1/3-x3 and +-60 % when fewer than 3 match.")
    am = (summ.get("_arcs") or {}).get("arms") or {}
    if am:
        nc = am.get("ctrl_cpl", 0) + am.get("ctrl_nat", 0)
        L.append(f"- what the control pool actually holds: {am.get('ctrl_cpl', 0)} coupled-element and {am.get('ctrl_nat', 0)} natural-run "
                 f"control passes ({am.get('ctrl_nat', 0) / max(nc, 1):.1%} natural; {am.get('genomes_without_nat', 0)} of "
                 f"{am.get('genomes', 0)} genomes have none): the cap of {K.POOL_MAX} keeps the controls that match most U sizes, "
                 f"and natural runs are longer than the 1-2-gene coupled elements, so the null is in effect \"delete another "
                 f"coupled element\" (not linked to U or D), with a small natural arm"
                 + (f"; {am['near']} extra controls near D (--ctrl_near)" if am.get("near") else "") + ".")
        if K.MATCH_LOG is not None:
            L.append(f"- a third matching tier (--match_log {K.MATCH_LOG:g}): distance within x1/{K.MATCH_LOG:g}..x{K.MATCH_LOG:g} on a "
                     f"log scale, size x1/3-3, when the +-35 % / +-60 % windows give fewer than 3 controls")
    L.append("- **specificity**: the same U pass is also read at up to 12 other elements of the same kind at a comparable distance "
             "(off-targets). spec = z_exp(D) - mean z_exp(off-targets). A deletion that moves everything downstream is not knowledge "
             "of the link.")
    L.append("- **calibration**: every per-link p is calibrated empirically by putting one of the unit's own matched controls in U's "
             "place (200 draws), because the matched null has skewed tails.\n")
    L.append("## Runtime\n")
    L.append(f"- baselines: {len(bm['done'])} chromosomes, median {f(bm['seconds_per_pass_median'],3)} s per full pass incl. the "
             f"head on every row and top-64 (total {f(bm.get('total_seconds', 0),0)} s)")
    L.append(f"- knockouts: {summ['passes']} passes in {summ['genomes']} genomes, median {f(summ['seconds_per_pass_median'],3)} s per pass "
             f"(mean {f(summ['seconds_per_pass_mean'],3)}; encoder + decoded readouts), {f(summ['gpu_seconds']/60,1)} GPU-minutes\n")
    if summ.get("influence"):
        L.append(f"- influence readouts (every deletion pass, genes out to {K.INF_WIN} and element entries out to {K.INF_ENT} "
                 f"genes downstream): median {f(summ['influence']['seconds_per_pass_median'],3)} s per pass on top of the above, "
                 f"{f(summ['influence']['seconds_total']/60,1)} GPU-minutes; {summ['influence']['entries']} entry readouts; "
                 f"per-genome file median {f(summ['influence']['genome_file_kb_median'],0)} kB\n")
    L.append("## Self-checks (asserted in every genome; the run stops on a failure)\n")
    L.append(f"- (a) delete nothing: {ck.get('check_a',0)} passes, max |delta| = {ck.get('check_a_max_abs_delta', float('nan'))} (exactly 0 required)")
    L.append(f"- (b) delete 10 genes right after the last readout row: {ck.get('check_b',0)} passes, max |delta| = "
             f"{ck.get('check_b_max_abs_delta', float('nan'))}; and in every one of the {ck.get('passes',0)} deletion passes plus "
             f"{ck.get('insertion_passes',0)} insertion passes, all readouts before the first modified gene were bit-identical "
             f"to the baseline ({ck.get('causal_rows_exact',0)} readouts checked)")
    L.append(f"- (c) readout rows land on the intended gene after the shift: {ck.get('identity_rows',0)} (row, pass) identities and "
             f"families asserted")
    L.append(f"- (e) the row convention, against the model: decoded log P(own family) at the readout row beats the same family read one "
             f"row later by {f(ck.get('row_conv_gain_median'),2)} on average per genome (worst genome {f(ck.get('row_conv_gain_min'),2)}, "
             f"> 0.5 required). An off-by-one in which output row is read would reverse this; the pilot's check (c) could not fail.")
    L.append(f"- (f) numerical floor: the same baseline at a padded length 64 longer moves a readout by at most "
             f"{ck.get('noise_floor_max_median', float('nan')):.2e} per genome (worst {ck.get('noise_floor_max_max', float('nan')):.2e}). "
             f"Deltas below their genome's floor are flagged unresolved: "
             f"{summ['noise_floor']['share_units_resolved']:.0%} of units are above it.")
    if ck.get("inf_passes"):
        L.append(f"- (g) influence readouts: in every one of the {ck.get('inf_passes',0)} deletion passes, the {ck.get('inf_upstream_rows',0)} "
                 f"genes read in the {K.INF_UP} genes before the first deleted gene were bit-identical to the baseline (max |delta| = "
                 f"{ck.get('inf_upstream_max_abs_delta', float('nan'))}); the whole baseline re-read with another head-chunk "
                 f"composition in {ck.get('inf_check_a_rows',0)} genes: max |delta| = {ck.get('inf_check_a_max_abs_delta', float('nan'))} "
                 f"(exactly 0 required); {ck.get('inf_rows',0)} gene readouts in all")
    L.append(f"- (d) decoder of the other lineage half: {ck.get('decoder_other_half',0)}/{ck.get('genomes',0)} genomes; decoder saw "
             f"the genome's lineage in {ck.get('decoder_saw_lineage',0)} genomes (0 required; its genes asserted absent too)")
    dq = bm.get("decoder", {})
    if dq:
        hv = dq["halves"]; pt = dq["decoder_ptrue_on_held_out_half"]
        L.append(f"- lineage halves: {hv['0']['genomes']} genomes / {hv['0']['lineages']} lineages vs {hv['1']['genomes']} / "
                 f"{hv['1']['lineages']}; decoded P(true family) on held-out genes, median {pt['0']['ptrue_top64_median']} / "
                 f"{pt['1']['ptrue_top64_median']} (top-64 truncation)\n")
    L.append("## Units and controls\n")
    L.append(f"- {summ['units']} units (link x genome) in {summ['genomes']} genomes; skipped (link, genome) pairs: " +
             "; ".join(f"{k}: {v}" for k, v in summ["skip"].items()))
    L.append(f"- units whose readout family the other half's decoder never saw (no readout, excluded): {int((~U.seen).sum())}")
    pk_ = summ["partial_ko"]
    L.append(f"- PARTIAL knockouts: in {pk_['share_units']:.1%} of units at least one of U's families survives the deletion elsewhere in "
             f"the chromosome (multi-copy elements). These units are excluded from the per-link test and counted separately; "
             f"{pk_['links_all_partial']} links consist only of such units.")
    pn = summ["panel"]; mc = summ["matched_ctrl"]
    L.append(f"- control pool: median {f(pn['per_genome_median'],0)} whole accessory elements per genome, size median "
             f"{f(pn['size_median'],0)}, 90th pct {f(pn['size_p90'],0)} member genes vs U deletions median "
             f"{f(plan['u_size_median'],0)}, 90th pct {f(plan['u_size_p90'],0)}")
    L.append(f"- matched controls per unit: median {f(mc['median'],0)}, >= 3 (z computable) in {mc['share_ge3']:.0%}, none in "
             f"{mc['share_0']:.0%}; windows widened for {mc['share_widened']:.0%} of units")
    mb = (summ.get("_arcs") or {}).get("matched_ctrl_by_distance") or {}
    if mb:
        L.append("- units (readout seen, complete knockout) with >= 3 matched controls, by U-D distance: " +
                 "; ".join(f"{k} genes {v['share_ge3']:.0%} of {v['units']}" for k, v in mb.items()))
    ot_ = summ["offtarget"]
    L.append(f"- off-targets: {ot_['units_with_off']} units have at least one, median {f(ot_['median_n_off'],0)} per unit")
    cal = summ["calibration"]
    L.append(f"- calibration: under pseudo-U draws the nominal test rejects at p < 0.05 in {cal['median_fp_exp']:.1%} (expected "
             f"direction) / {cal['median_fp_opp']:.1%} (opposite) of draws at the median link (nominal 5 %)\n")
    # distributions of raw deltas
    for s, nm in ((1, "co-occurrence"), (-1, "avoidance")):
        u = U[(U.sign == s) & U.seen]
        if not len(u): continue
        L.append(f"- {nm}: {len(u)} units, delta quartiles {np.round(np.percentile(u.delta, [25, 50, 75]), 4).tolist()}, "
                 f"|delta| median {f(float(np.median(np.abs(u.delta))),4)}; matched-null MAD median {f(float(u.ctrl_mad.median()),4)}; "
                 f"baseline P(D entry) median {f(float(u.base_p1.median()),4)}")
    L.append("")
    L.append("## Per link\n")
    L.append("verdict: **knows** = calibrated BH q < 0.05 (one-sided Wilcoxon signed-rank on lineage-mean z_exp) AND the effect survives "
             "the specificity test (D moves more than the off-targets of the same pass, calibrated BH q < 0.05) AND the sign test p < 0.05 "
             "AND the same (expected) sign in both lineage halves, where the specificity test must hold BOTH in z units and in the "
             "raw readout units; **not specific** = the deletion moves D more than other deletions of the same kind move D, "
             "but not more than the same deletion moves the other elements it hits (a genome-wide effect of removing U, "
             "which would earn \"knows\" on every link U belongs to); **opposite** = calibrated BH q < 0.05 the other way "
             "with the same sign in both halves AND the same specificity requirement in that direction; **not specific (opposite)** "
             "= the same, without specificity; **none** (with the smallest effect in z units the test detects at 80 % power, alpha Bonferroni-adjusted over the tested links); "
             "**nt** = not testable (< 5 lineages, no readout, only partial knockouts, or every delta below the numerical floor).\n")
    L.append("| link | type | selected as | regions (a / b labels) | link z | units | partial | lineages | median delta | resolved | median z | "
             "beyond 95th pct (null) | q_cal | spec z: median / q_cal | spec raw: median / q_cal | half 0 / half 1 | 2024+ new lin | verdict | MDE z |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in Lk.iterrows():
        L.append(f"| {r.li} | {'co' if r.sign > 0 else 'av'} | {r.why} | {r.a} / {r.b} | {f(r.link_z)} | {r.n_units} | {r.n_partial} | {r.n_lin} | "
                 f"{f(r.median_delta,4)} | {f(r.share_resolved)} | {f(r.median_z)} | "
                 f"{f(r.share_beyond95)} ({f(r.share_beyond95_null)}) | {f(r.q_cal_exp,3)} | "
                 f"{f(r.median_spec_lin)} / {f(r.q_cal_spec,3)} | {f(r.median_spec_raw_lin,5)} / {f(r.q_cal_spec_raw,3)} | "
                 f"{f(r.half0_median)} ({r.half0_n_lin}) / {f(r.half1_median)} ({r.half1_n_lin}) | {f(r.newlin_median)} ({r.newlin_n_lin}) | "
                 f"**{r.verdict}**{(' (' + r.nt_why + ')') if r.verdict == 'nt' and isinstance(r.get('nt_why'), str) and r.nt_why else ''} | "
                 f"{f(r.mde_z)} |")
    L.append("\nColumns: partial = units where a copy of U's families survives the deletion (excluded from the test); median delta over units "
             "(raw readout units: logit for co, log-odds for av); resolved = share of units whose |delta| exceeds their genome's fp32 floor; "
             "median z over units; share of units whose delta lies beyond the 95th percentile of their own matched controls in the expected "
             "direction (expected share under the null in brackets); q_cal = BH over the empirically calibrated p; spec = median lineage-mean "
             "(D minus the off-targets of the same pass), in z units and in raw readout units; half columns: median lineage-mean z_exp (lineages); 2024+ new lin: median "
             "lineage-mean z_exp over genomes released in 2024 or later whose LINEAGE has no older genome (the only ones the model may not "
             "have memorised through a relative).\n")
    dd_ = Lk[np.isfinite(Lk.p_exp)]
    if len(dd_):
        L.append("**How global is each deletion?** Mean shift (expected direction) over the OTHER readouts of the same "
                 "pass, for the U deletion and for its own matched control deletions. A U whose deletion shifts every "
                 "element downstream can beat the null at D without knowing anything about D.\n")
        L.append("| link | U: mean shift at the other readouts | matched controls | ratio | p (U > controls) | U: D minus the others | controls: D minus the others |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in dd_.iterrows():
            rt = (r.global_shift_U / r.global_shift_ctrl) if (np.isfinite(r.get("global_shift_ctrl", np.nan)) and r.global_shift_ctrl) else np.nan
            L.append(f"| {int(r.li)} | {f(r.get('global_shift_U'),5)} | {f(r.get('global_shift_ctrl'),5)} | {f(rt,1)} | "
                     f"{f(r.get('p_global'),4)} | {f(r.median_spec_raw_lin,5)} | {f(r.get('global_shift_ctrl_specraw'),5)} |")
        L.append("")
        L.append("Also on the record, per link: the same test with the partial knockouts put back (median lineage-mean z_exp / p): " +
                 "; ".join(f"{int(r.li)}: {f(r.median_z_exp_lin_withpartial)} / {f(r.p_exp_withpartial,3)}" for _, r in dd_.iterrows()
                           if r.n_partial > 0) + ".")
        L.append("Units whose readout family also occurs upstream of the readout row (so the readout is not specific to D): " +
                 "; ".join(f"{int(r.li)}: {r.share_f1_upstream:.0%}" for _, r in dd_.iterrows() if r.share_f1_upstream > 0) + ".")
        fl_ = [f"{int(r.li)}: {r.share_resident_floored:.1%}" for _, r in dd_.iterrows() if r.share_resident_floored > 0]
        if fl_: L.append("Avoidance units whose resident probability sits at the 1e-12 floor: " + "; ".join(fl_) + ".")
        L.append("")
    g = summ.get("global_test") or {}
    if g:
        L.append("## Global test over links\n")
        L.append(f"- {g['n_links']} tested links, median link effect (lineage-mean z_exp) {f(g['median_effect'])}; Wilcoxon over links "
                 f"(expected direction) p = {f(g['wilcoxon_p_exp'],3)}; {g['sign_pos']}/{g['n_links']} links in the expected direction "
                 f"(sign test p = {f(g['sign_p'],3)})")
        L.append(f"- links share regions, so also on a region-disjoint subset: {g['n_links_region_disjoint']} links, median effect "
                 f"{f(g['median_effect_disjoint'])}, p = {f(g['wilcoxon_p_exp_disjoint'],3)}")
        L.append(f"- does the effect track the within-lineage coupling? Spearman vs |link z| rho = {f(g['spearman_vs_link_absz'][0])} "
                 f"(p {f(g['spearman_vs_link_absz'][1],3)}), vs link pairs rho = {f(g['spearman_vs_pairs'][0])} (p {f(g['spearman_vs_pairs'][1],3)})")
        if "offtarget_vs_phi_spearman" in g:
            L.append(f"- do the off-target effects track each region's plain across-genome correlation with U (a strain marker would)? "
                     f"Spearman rho = {f(g['offtarget_vs_phi_spearman'][0])} (p {f(g['offtarget_vs_phi_spearman'][1],3)})")
        L.append("")
    acf = f"{out}/anchor_check.parquet"
    if os.path.exists(acf):
        ac = pd.read_parquet(acf)
        if len(ac):
            L.append("## Where an absent element's entry would sit (fix of the pilot's blocker)\n")
            L.append("The pilot read an absent D at the start of its panRGP spot block. In carriers D's element almost never starts "
                     "there (nanS: 12 genes in, in 100 % of carriers; T6SS: 1 %), so the readout sat in the far tail of the softmax "
                     "(P(D entry) 1e-6 to 1e-3) and its delta was uncorrelated with the delta at the right row (r = 0.06-0.26 over "
                     "12 genomes per link, GPU check). The row is now placed from the carriers' flanking families, and the rule is "
                     "validated on the carriers themselves: applied to a carrier it must return D's real entry gene.\n")
            L.append("| D region | label | carriers | rule | share of carriers where the rule returns D's real entry | "
                     "P(D entry) at D's real entry in carriers |")
            L.append("|---|---|---|---|---|---|")
            for _, r in ac.iterrows():
                L.append(f"| {r.ri} | {r.label} | {r.carriers} | {r.rule or 'none'} | {f(r.carrier_share_correct)} | "
                         f"{f(r.carrier_P_entry_median,4)} |")
            L.append("")
    if PCd is not None and len(PCd):
        L.append("## Positive control: can the model use context that far upstream?\n")
        L.append("A copy of D inserted between two backbone genes ~1,600 genes upstream of D's entry (far) and at the link's OWN U-D "
                 "distance (own); three sizes: D's member genes only (what the pilot inserted: often a single protein), D's whole "
                 "element span, and that span preceded by the 30 genes that precede D's entry here. Control = a whole natural "
                 "accessory element of another genome with as many proteins, at the same point. Change of logit P(D entry family) at "
                 "D's entry, same padded length for baseline and insertions.\n")
        L.append("| link | genomes | proteins mem / span / span+30 | mem far | span far | span+30 far | ctrl far | span own | ctrl own | span far > ctrl far |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for li, d in PCd.groupby("li"):
            gt = (d.span_far.abs() > d.ctrl_far.abs()).sum() if "span_far" in d else 0
            L.append(f"| {li} | {len(d)} | {int(d.n_mem.median())} / {int(d.n_span.median())} / {int(d.n_ctx.median())} | "
                     f"{f(d.mem_far.abs().median(),3)} | {f(d.span_far.abs().median(),3)} | {f(d.span30_far.abs().median(),3)} | "
                     f"{f(d.ctrl_far.abs().median(),3)} | {f(d.span_own.abs().median(),3) if 'span_own' in d else ''} | "
                     f"{f(d.ctrl_own.abs().median(),3) if 'ctrl_own' in d else ''} | {gt}/{len(d)} |")
        uu = U[U.li.isin(PCd.li.unique()) & U.seen]
        L.append(f"\nValues are median |change| of the readout. All links: far distance median {f(PCd.dist_far.median(),0)} genes, "
                 f"own-distance median {f(PCd.dist_own.median(),0) if 'dist_own' in PCd else ''}; a copy of D vs a foreign element at "
                 f"the same point, far: {f(PCd.span_far.abs().median(),3)} vs {f(PCd.ctrl_far.abs().median(),3)}"
                 + (f", at the link's own distance: {f(PCd.span_own.abs().median(),3)} vs {f(PCd.ctrl_own.abs().median(),3)}"
                    if "span_own" in PCd else "") +
                 f". For comparison the knockout |delta| of these links' units: median "
                 f"{f(float(uu.delta.abs().median()) if len(uu) else np.nan,4)}.")
        L.append("The insert being D rather than a foreign element of the same size is what a working assay must detect: if the two "
                 "move the readout equally, the model is reacting to the presence of an insertion, not to its content.\n")
    dfile = f"{K.KDIR}/{tag}_decay.parquet"
    if os.path.exists(dfile):
        dd = pd.read_parquet(dfile)
        L.append("## Sensitivity vs distance (framing diagnostic, `pg_knockout.py --decay`)\n")
        L.append(f"Same {len(dd)} positive-control units: delete 2 consecutive genes (any family) ending d genes before D's entry, or insert "
                 "a copy of D's member proteins ending d genes before it; |change| of logit P(D entry family) at D's entry.\n")
        L.append("| d (genes) | " + " | ".join(str(d) for d in DECAY_D) + " |")
        L.append("|---|" + "---|" * len(DECAY_D))
        for kind, nm in (("del", "2-gene deletion: median abs"), ("ins", "copy of D inserted: median abs"), ("ins", "copy of D: median signed")):
            fn = (lambda c: c.abs().median()) if "abs" in nm else (lambda c: c.median())
            L.append(f"| {nm} | " + " | ".join(f(float(fn(dd[f'{kind}_{d}'])), 4) if f"{kind}_{d}" in dd else "" for d in DECAY_D) + " |")
        L.append("")
    pfile = f"{K.KDIR}/{tag}_probe.parquet"
    if os.path.exists(pfile):
        from scipy import stats as _st
        pp = pd.read_parquet(pfile)
        L.append("## Content or position? (`pg_knockout.py --probe LINKS`)\n")
        L.append("For each unit of the probed links: the U deletion; a SHAM deletion of as many accessory genes as close as possible "
                 "to U (outside U and every coupled element); a SUBSTITUTION of U's proteins, in place, by a natural accessory element "
                 "of another genome (positions unchanged). Values: change of the readout, expected direction positive "
                 "(x -1 for co-occurrence). Tests: one-sided Wilcoxon on lineage means.\n")
        L.append("| link | units | lineages | U deletion: median | sham deletion: median | substitution: median | U - sham > 0: p | U vs subst: p (two-sided) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for li, d in pp.groupby("li"):
            e = np.where(d.sign > 0, -1, 1)
            x = pd.DataFrame(dict(lin=d.lineage, U=d.d_U * e, sham=d.get("d_sham", np.nan) * e, sub=d.get("d_subst", np.nan) * e))
            lm = x.groupby("lin").mean()
            m1 = (lm.U - lm.sham).dropna(); m2 = (lm.U - lm["sub"]).dropna()
            p1 = _st.wilcoxon(m1, alternative="greater").pvalue if len(m1) >= 5 else np.nan
            p2 = _st.wilcoxon(m2).pvalue if len(m2) >= 5 else np.nan
            L.append(f"| {li} | {len(d)} | {len(lm)} | {f(float(x.U.median()),5)} | {f(float(x.sham.median()),5)} | {f(float(x['sub'].median()),5)} | "
                     f"{f(p1,4)} | {f(p2,4)} |")
        L.append("")
    ar = summ.get("_arcs") or {}
    if ar and ar.get("verdict_by_flag"):
        L.append("## Per arc: what a verdict can mean (`{tag}_arcs.json`)\n".replace("{tag}", tag))
        L.append(f"- {ar['units_tested']} tested units, {ar['units_tested_within_500']} of them within 500 genes; {MODEL_REACH}. "
                 "A 'none' on an arc tested far away means the test could not see an effect, not that the model does not know it.")
        for k, nm in (("one_site", "one element at different spots (not two regions)"), ("tract", "possibly one transfer tract"),
                      ("site_competition", "two elements competing for one insertion site"),
                      ("rep_lineage", "replicated in lineage-disjoint halves")):
            c = ar["verdict_by_flag"].get(k, {})
            L.append(f"- {nm}: " + "; ".join(f"{v} {kk.split('|')[0]} ({'yes' if kk.split('|')[1] == 'True' else 'no'})"
                                             for kk, v in sorted(c.items())))
        L.append("")
    L.append("## Files\n")
    L.append(f"- `pgb/knock/base_top64_f.npy`, `base_top64_p.npy`, `base_ent.npy`, `base_meta.json`: fp32 full-chromosome baseline calls")
    L.append(f"- `pgb/knock/{tag}/g*.npz|json`: per genome, every pass (kind, deleted genes) x every readout (raw values), self-check counters"
             + ("; with the influence readouts (`inf_*`: 25-gene bins per deletion pass, element entries; `el_*`: the genome's "
                "accessory elements; `pass_spot`), read with `pg_knock.load_influence()`" if summ.get("influence") else ""))
    L.append(f"- `pgb/knock/{tag}_units.parquet` (one row per unit: delta, its matched control deltas, robust z, rank percentile, the "
             f"off-target mean, the pseudo-U draws, the numerical floor, the partial-knockout and non-specific-readout "
             f"flags, secondary per-gene deltas), `{tag}_links.parquet`, `{tag}_offtarget.parquet`, `{tag}_posctrl.parquet`, "
             f"`{tag}_summary.json`, `{tag}_arcs.json` (per arc: verdict, tested distances, power, link flags, caveats), "
             f"`{tag}_decay.parquet`, `{tag}_probe.parquet`")
    L.append(f"- `pgb/knock/{tag}/plan.json`, `skipped.parquet` (why each (link, genome) pair was skipped), "
             f"`anchor_check.parquet` (the insertion-point rule validated on each D region's carriers), `notes.md`")
    L.append(f"- `pgb/knock/v1/`: the first pilot, kept for comparison\n")
    nf = f"{out}/notes.md"
    if os.path.exists(nf): L.append(open(nf).read())
    txt = "\n".join(L) + "\n"
    open(f"{out}/{tag}_report.md", "w").write(txt)
    if tag == "pilot": open(f"{K.KDIR}/pilot_report.md", "w").write(txt)
    log(f"report -> {out}/{tag}_report.md")


# ============================================================================ 8. sensitivity vs distance (framing diagnostic)
DECAY_D = (1, 3, 10, 30, 100, 300, 1000, 1600)


def run_decay(tag):
    """For the positive-control units (6 co-occurrence links x 10 genomes): delete 2 consecutive genes ending d genes
    before D's entry, or insert a copy of D's whole element span d genes before it, d in DECAY_D; read logit P(D entry)
    at D's entry (same padded length, float64 decoding, identity and causality asserted). Shows how fast the model's
    dependence on upstream content decays, and that the readout does register large changes when they happen."""
    import pandas as pd
    sel = pilot_links(); links = list(sel)
    units, _, _ = D.units(links)
    pcs = posctrl_specs(units, links, np.random.default_rng(7))
    R = K.Runner(frac=args.mem_frac); dec = K.Decoder(D); out = []
    for g in sorted(pcs):
        a, b = int(D.OFF[g]), int(D.OFF[g + 1]); n = b - a; FAMg = D.FAM[a:b]; hh = dec.half_for(g)
        fams = sorted({pc["f1"] for pc in pcs[g]}); fi = {f: i for i, f in enumerate(fams)}
        R.set_families(np.stack([dec.row(f, hh) for f in fams]))
        R.set_genome(D.EMB[D.PID[a:b]], n + 1 + max(len(pc["ins"]["span"]) for pc in pcs[g]))

        def val(src, extra, row):
            src = np.asarray(src); o = np.flatnonzero(src < n); pos = np.full(n, -1); pos[src[o]] = o
            k = pos[row]; assert k >= 0 and src[k] == row and FAMg[src[k]] == FAMg[row]
            srcE = np.where(src < n, src, -1 - (src - n))
            S, C = R.decode(R.hidden(srcE, extra), [k])
            return float(np.log(max(S[0, fi[FAMg[row]]], K.FLOOR)) - np.log(max(C[0, fi[FAMg[row]]], K.FLOOR)))
        for pc in pcs[g]:
            row = pc["row"]; base = val(np.arange(n), None, row)
            cp_ = pc["ins"]["span"]                             # D's whole element span, not only its member genes
            rec = dict(li=pc["li"], g=g, row=row, base=base, n_copy=len(cp_))
            cp = np.asarray(D.EMB[D.PID[a + np.asarray(cp_)]], np.float32)
            for d in DECAY_D:
                if row - d - 1 < 0: continue
                keep = np.setdiff1d(np.arange(n), [row - d - 1, row - d])
                rec[f"del_{d}"] = val(keep, None, row) - base
                q = row - d + 1                                      # the copy ends d genes before D's entry
                src = np.concatenate([np.arange(q), n + np.arange(len(cp)), np.arange(q, n)])
                rec[f"ins_{d}"] = val(src, cp, row) - base
            out.append(rec)
    df = pd.DataFrame(out); df.to_parquet(f"{K.KDIR}/{tag}_decay.parquet", index=False)
    log(f"decay: {len(df)} units -> {K.KDIR}/{tag}_decay.parquet")
    for kind in ("del", "ins"):
        log(kind, {d: round(float(df[f"{kind}_{d}"].abs().median()), 4) for d in DECAY_D if f"{kind}_{d}" in df})


# ============================================================================ 9. content vs position probe for significant links
def run_probe(tag, links):
    """For every unit of `links`: (1) the U deletion again, (2) a SHAM deletion of as many accessory (non-backbone)
    genes as close as possible to U (within 60 genes, outside U and every coupled element), (3) a SUBSTITUTION of
    U's member proteins, in place, by as many proteins of a natural accessory element (non-coupled spot) of another
    genome: positions unchanged, content changed. If (1) ~ (2) the effect is positional; if (1) ~ (3) and both
    differ from (2), the model reads U's content."""
    import pandas as pd
    units, _, _ = D.units(links)
    by_g = collections.defaultdict(list)
    for u in units: by_g[u["g"]].append(u)
    R = K.Runner(frac=args.mem_frac); dec = K.Decoder(D); out = []; rng = np.random.default_rng(11)
    for g in sorted(by_g):
        a, b = int(D.OFF[g]), int(D.OFF[g + 1]); n = b - a; FAMg = D.FAM[a:b]; hh = dec.half_for(g)
        fams = sorted({u["f1"] for u in by_g[g]} | {u["f2"] for u in by_g[g] if u["f2"] >= 0})
        fi = {f: i for i, f in enumerate(fams)}
        R.set_families(np.stack([dec.row(f, hh) for f in fams])); R.set_genome(D.EMB[D.PID[a:b]], n + 1)
        avoid = set()
        for ri in range(D.NR):
            if D.HAS[ri, g]:
                e = D.EL[(ri, g)]
                if e["end"] >= e["start"]: avoid |= set(range(e["start"], e["end"] + 1))

        def val(src, extra, u):
            src = np.asarray(src); o = np.flatnonzero(src < n); pos = np.full(n, -1); pos[src[o]] = o
            k = pos[u["row"]]; assert k >= 0 and src[k] == u["row"]
            assert FAMg[src[k]] == (u["f1"] if u["sign"] > 0 else u["f2"])
            srcE = np.where(src < n, src, -1 - (src - n))
            S, C = R.decode(R.hidden(srcE, extra), [k])
            v, *_ = K.readout_values(np.array([u["sign"]]), S, C, np.array([fi[u["f1"]]]), np.array([fi[u["f2"]] if u["f2"] >= 0 else -1]))
            return float(v[0])
        basev = {}
        for u in by_g[g]:
            key = (u["row"], u["sign"], u["f1"], u["f2"])
            if key not in basev: basev[key] = val(np.arange(n), None, u)
            base = basev[key]; t = u["u_size"]
            rec = dict(li=u["li"], g=g, lineage=D.LIN[g], half=int(D.HALF[g]), sign=u["sign"], u_start=u["u_start"], u_size=t,
                       dist=u["dist"], base=base)
            rec["d_U"] = val(np.setdiff1d(np.arange(n), u["u_dels"]), None, u) - base
            cand = [p for p in range(max(0, u["u_start"] - 60), min(n, u["u_end"] + 61))
                    if not (u["u_start"] <= p <= u["u_end"]) and p not in avoid and not D.BB[FAMg[p]] and p < u["row"] - 50]
            cand = sorted(cand, key=lambda p: min(abs(p - u["u_start"]), abs(p - u["u_end"])))[:t]
            if len(cand) == t:
                rec["d_sham"] = val(np.setdiff1d(np.arange(n), cand), None, u) - base
                rec["sham_offset"] = float(np.mean(cand) - np.mean(u["u_dels"]))
            for _ in range(50):
                g2 = int(rng.integers(D.NG))
                if g2 == g: continue
                pan = [e for e in D.control_pool(g2) if e["size"] >= t]
                if pan:
                    e2 = pan[int(rng.integers(len(pan)))]
                    extra = np.asarray(D.EMB[D.PID[D.OFF[g2] + np.asarray(e2["dels"][:t])]], np.float32)
                    src = np.arange(n); src[np.asarray(u["u_dels"])] = n + np.arange(t)
                    rec["d_subst"] = val(src, extra, u) - base
                    break
            out.append(rec)
    df = pd.DataFrame(out); df.to_parquet(f"{K.KDIR}/{tag}_probe.parquet", index=False)
    log(f"probe: {len(df)} units of links {links} -> {K.KDIR}/{tag}_probe.parquet")
    for li, d in df.groupby("li"):
        log(li, {c: round(float(d[c].median()), 5) for c in ("d_U", "d_sham", "d_subst") if c in d})


def main():
    if args.baselines:
        run_baselines(); return
    if args.decay:
        run_decay(args.tag or "pilot"); return
    if args.probe:
        run_probe(args.tag or "pilot", [int(x) for x in args.probe.split(",")]); return
    if args.pilot:
        tag = args.tag or "pilot"; sel = pilot_links(); links = list(sel)
    elif args.all:
        tag = args.tag or "all"; sel = None; links = list(range(len(D.LINKS)))
    else:
        ap.print_help(); return
    settle(tag)
    if not args.aggregate: run(tag, links, pilot=args.pilot)
    U, Lk, PCd, summ = aggregate(tag, links, args.pilot, sel)
    write_report(tag, U, Lk, PCd, summ, sel)


if __name__ == "__main__":
    main()
