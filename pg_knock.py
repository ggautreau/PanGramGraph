"""In-silico knockout of the "Coupled regions" links with Bacformer (causal): the library.

pg_region_sets.py found 533 links between 290 accessory elements from presence/absence within lineages,
without the model. Here each link is put to the model: in every complete chromosome where the link can be
tested, the upstream element (dnaA reading order, decided per genome) is deleted and the model's belief in
the downstream element is read, in ONE causal fp32 pass over the whole chromosome (<= 5,584 proteins).

  units      (link, genome) where the link is testable; carriers are defined AT the element's spot (the
             validated construction of the scouts' ko_units3.py, ported here unchanged: panRGP spot block,
             consensus backbone flanks for loose regions, per-genome orientation, entry gene, insertion
             point of an absent element)
               co-occurrence  genome carries both        expected: deleting U LOWERS logit P(D entry)
               avoidance      carries U, not D (cleanly:  expected: deleting U RAISES
                              no D family anywhere, none   log P(D entry family) - log P(resident)
                              at D's spot)                 at D's insertion point
             skipped (reason recorded): U or D overlapping, < 100 genes apart (MIN_APART; > MAX_APART when set),
             wrapping the origin, resident = D's entry family, no entry family, no position.
  deletion   U's member genes (families of the region, never a backbone family) inside its element span
  readout    decoded family probability P(f | ctx) = sum_c P(c | ctx) P(f | c) over the FULL softmax at the
             readout row, the decoder P(f | c) rebuilt from fp32 full-chromosome baseline passes (top 64 per
             gene), cross-validated by lineage: a genome is always read with the decoder of the other half
  controls   per genome a panel of <= 40 natural accessory elements (runs of panRGP genes of one spot, at
             spots holding none of the 290 coupled regions), sizes matched to the U deletions, spread along
             the chromosome; matched null per unit + a per-genome regression
  numerics   every pass of a genome has the same padded length and the head is applied on fixed-size row
             chunks, so identical prefixes give bit-identical readouts: deleting nothing gives delta == 0
             exactly and deleting genes after a readout row leaves it exactly unchanged (asserted in every
             pass, not only in the dedicated check passes)

  link sets  Data(sets=PATH) reads any file in the format of pgb/region_sets.json (default); the units cache
             is keyed by the md5 of the whole file (for the default file, < 1 MiB, the key is unchanged)
  close range  (pgb/region_sets_close.json: pairs 50-500 genes apart in the genomes carrying both, two separate
             insertions, replicated in lineage-disjoint halves) MAX_APART bounds a unit's U-D distance (600 for these
             sets; a unit is one genome's pair in dnaA reading order and can be thousands of genes apart, across the
             origin); near_controls() adds whole-element controls at about a unit's own distance upstream of D, and
             MATCH_LOG = F a third matching tier (distance within x1/F..xF): at 50-100 genes a genome rarely holds 3
             whole elements in the +-35 % / +-60 % windows. Defaults leave the long-range behaviour unchanged.
  influence  what the model itself knows, independent of any link: after a deletion, the decoded logit
             P(true family) (same lineage-CV decoder, fp32 pass, float64 decoding, read through a sparse
             copy of the decoder rows: Decoder.srow, Runner.decode_true) at every gene downstream of the
             deleted element. Data.elements(g) lists the accessory elements of a genome (panRGP runs at
             every spot, by spot id; coupled regions present at their spot, by region id) and their entry
             genes. influence_readout() turns one pass into 25-gene bins of the mean delta (and mean
             |delta|, and the mean over panRGP genes) out to 1,500 genes, plus the delta at every element
             entry out to 2,400 genes (1,500 x 1.6, so that the matched null of an entry near 1,500 genes,
             distance +-35 % widened to +-60 %, is on file too). The 25 genes before the first deleted gene
             are read as well and must be exactly unchanged (causality, check (g)).
             load_influence(g*.npz) reads them back as tables; influence_matrix(g*.npz) is a reference
             aggregation of the element-to-element matrix against the units' matched whole-element null (same
             entry, size x0.5-2, distance +-35 %). influence(g, element) does it all on demand for one genome
             (baseline + deletion + 8 size-matched whole-element controls), served by serve_live.py (/influence,
             /elements: the page's own server, locally and in the Space) and by serve_influence.py (a process of its
             own: its own port and GPU cap).
  compact decoder  influence() reads only the sparse decoder rows (Decoder.srow) of the families of the genome, with
             the decoder of the other lineage half. export_decoder() writes exactly those rows, both halves, to
             pgb/knock/decoder_halves.npz (cluster ids uint16, P(f | c) float64: the same rows bit for bit, 1/12 of
             base_top64_{f,p}.npy), and Decoder uses that file when base_top64_{f,p}.npy are absent (the Space).
                 python3 -c "import pg_knock as K; K.export_decoder()"

Reads the project's data read-only; writes only under pgb/knock/ (through pg_knockout.py, the units cache and
export_decoder()).
"""
import os, json, time, zlib, pickle, hashlib, collections
import numpy as np

PROJ = os.path.dirname(os.path.abspath(__file__))        # the project (also the Space's app directory)
KDIR = f"{PROJ}/pgb/knock"
MODEL = "macwiatrak/bacformer-causal-complete-genomes"
NCL = 50001                     # softmax width of gm_head (50,000 clusters + END)
TOPK = 64
CLS, PROT = 2, 4
SLACK, NEAR, GAP, CARRY = 5, 200, 30, 0.5      # as ko_units3.py
MIN_APART = 100                 # genes between U's last gene and D's readout gene
MAX_APART = None                # at most this many (None: no bound, as before); 600 for the close-range sets
FLOOR = 1e-12                   # probability floor of every log
HEAD_CHUNK = 64                 # rows per head call (fixed shape -> bit-identical rows across passes)
TRUE_CHUNK = HEAD_CHUNK         # rows per head call of decode_true (256 was no faster and moved the logits by 2.5e-5)
PANEL_MAX = 40
ANCHOR_WIN, ANCHOR_SHARE = 40, 0.6      # where an absent element's entry would sit: carriers' flanking families
POOL_MAX = 72                   # control deletions (whole accessory elements) per genome
MATCH = (0.5, 2.0, 0.35)        # matched null: size x0.5-2 of U's, distance to the readout within +-35 %
WIDEN = (1 / 3.0, 3.0, 0.6)     # adaptive widening when fewer than 3 controls match
MATCH_LOG = None                # a third tier when still < 3: distance within x1/F..xF (log scale), size x1/3-3 (None: off,
                                # as before). At 50-100 genes a genome rarely holds 3 whole elements within +-60 % of the
                                # unit's distance (10 % of the close units); x2 reaches 21 %, x3 31 % (pg_knockout --match_log)
OFF_TOL = 0.35                  # off-target readouts: within +-35 % of D's distance from U
SETS_DEFAULT = f"{PROJ}/pgb/region_sets.json"
DECODER_NPZ = f"{KDIR}/decoder_halves.npz"   # the compact decoder (export_decoder), used when base_top64 is absent
INF_WIN, INF_BIN = 1500, 25     # influence profile: genes downstream of the deleted element, bin width
INF_ENT = int(INF_WIN * (1 + WIDEN[2]))   # element entries read out to 2,400 genes: the widened matched null
INF_UP = 25                     # genes read before the first deleted gene: must be exactly unchanged (check g)
MEM_FRAC = 0.35                 # per-process GPU memory cap (two runs share the 8 GB card)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================ data and units
def units_cache_file(sets=None):
    """the units cache that Data(sets) reads or writes (the same key as Data.__init__): space/assemble.py ships it so that
    the Space's first /elements does not rebuild it (~5-10 s)"""
    raw = open(os.path.abspath(sets) if sets else SETS_DEFAULT, "rb").read()
    NP = int(np.load(f"{PROJ}/pgb/chrom.npz")["offsets"][-1])
    return f"{KDIR}/units_cache_{hashlib.md5(raw + str((NP, SLACK, NEAR, GAP, CARRY)).encode()).hexdigest()[:12]}.pkl"


class Data:
    """chromosomes, regions, links, and the per (region, genome) element / insertion point of ko_units3.py"""

    def __init__(self, cache=True, sets=None):
        t0 = time.time()
        self.SETS = os.path.abspath(sets) if sets else SETS_DEFAULT
        raw = open(self.SETS, "rb").read()
        self.SETS_MD5 = hashlib.md5(raw).hexdigest()
        self.RS = RS = json.loads(raw)
        Z = np.load(f"{PROJ}/pgb/chrom.npz")
        self.OFF = OFF = Z["offsets"]; self.FAM = FAM = Z["fam"]; self.SPOT = Z["spot"]; self.RGP = Z["rgp"]; self.PID = Z["pid"]
        self.NG = NG = len(OFF) - 1; self.GL = np.diff(OFF); self.NP = int(OFF[-1])
        CJ = json.load(open(f"{PROJ}/pgb/chrom_genomes.json"))
        self.FN = FN = CJ["families"]; self.NF = len(FN); self.FIDX = {n: i for i, n in enumerate(FN)}
        self.ACC = [g["acc"] for g in CJ["genomes"]]
        self.gene_names = CJ.get("gene_names")
        self.REG = RS["regions"]; self.LINKS = RS["links"]; self.NR = len(self.REG)
        self.FAMS = [[self.FIDX[n] for n in r["families"]] for r in self.REG]
        rsg = {g["acc"]: g for g in RS["genomes"]}
        self.CL = np.array([rsg[a]["cl"] for a in self.ACC])
        self.STRAIN = [rsg[a].get("strain", "") for a in self.ACC]
        self.first = np.zeros(self.NP, bool); self.first[OFF[:-1]] = True
        self.GID = np.repeat(np.arange(NG), self.GL)
        self.LOC = np.arange(self.NP) - OFF[self.GID]
        NPROT = os.path.getsize(f"{PROJ}/pgb/chrom_emb.f16") // (480 * 2)
        self.EMB = np.memmap(f"{PROJ}/pgb/chrom_emb.f16", dtype=np.float16, mode="r", shape=(NPROT, 480))
        self._lineages()
        # keyed by the WHOLE sets file (the key used to hash its first MiB only: identical for files < 1 MiB,
        # as the default one is, so its existing cache stays valid)
        key = hashlib.md5(raw + str((self.NP, SLACK, NEAR, GAP, CARRY)).encode()).hexdigest()[:12]
        cf = f"{KDIR}/units_cache_{key}.pkl"
        loaded = False
        if cache and os.path.exists(cf):
            try: self.__dict__.update(pickle.load(open(cf, "rb"))); loaded = True
            except Exception as e: log(f"units cache {os.path.basename(cf)} unreadable ({type(e).__name__}: {e}): rebuilt")
        if not loaded:
            self._locate()
            if cache:
                keep = ("BB", "PRES", "HAS", "TEST", "EL", "INS", "ENTRY", "FLANK", "how")
                try:
                    os.makedirs(KDIR, exist_ok=True)
                    with open(cf + ".tmp", "wb") as fh: pickle.dump({k: self.__dict__[k] for k in keep}, fh)
                    os.replace(cf + ".tmp", cf)
                except OSError as e: log(f"units cache {os.path.basename(cf)} not written ({e}): kept in memory")
        self.COUPLED_SPOTS = {r["spot"] for r in self.REG if r["spot"] is not None}
        self.cache_file = cf
        log(f"data ready in {time.time()-t0:.0f}s: {NG} chromosomes, {self.NR} regions, {len(self.LINKS)} links"
            + ("" if self.SETS == SETS_DEFAULT else f" (sets {os.path.relpath(self.SETS, PROJ)}, md5 {self.SETS_MD5[:12]})"))

    # ---------------------------------------------------------------- lineage clusters and their halves
    def _lineages(self):
        """lineage = region_sets.json genomes[i]['cl'] (cut 0.4); cl <= 0 counts as its own lineage (singletons
        already are). Halves: a fixed hash (crc32) of the lineage id, so a lineage is always on one side."""
        lin = []
        for g in range(self.NG):
            c = int(self.CL[g]); lin.append(f"cl{c}" if c > 0 else f"acc:{self.ACC[g]}")
        self.LIN = np.array(lin, dtype=object)
        self.HALF = np.array([zlib.crc32(l.encode()) % 2 for l in lin])
        self.LIN_HALF = {l: int(h) for l, h in zip(lin, self.HALF)}

    def release_dates(self):
        """ncbi_seq_rel_date from the PanGBank metadata in the pangenome file (pgb/genome_meta.json lacks it)"""
        import tables, warnings
        warnings.simplefilter("ignore")
        with tables.open_file(f"{PROJ}/pgb/ecoli_11587.h5") as h:
            t = h.root.metadata.genomes.pangbank_wf.read()
        d = {i.decode(): r.decode() for i, r in zip(t["ID"], t["ncbi_seq_rel_date"])}
        return [d.get(a, "") for a in self.ACC]

    # ---------------------------------------------------------------- ko_units3.py, ported
    def _circ_blocks(self, ps, L, gap=GAP):
        ps = sorted(ps)
        if len(ps) == 1: return [(ps[0], ps[0], 1)]
        d = [(ps[(i + 1) % len(ps)] - ps[i]) % L for i in range(len(ps))]
        cuts = [i for i, x in enumerate(d) if x > gap]
        if not cuts: return [(ps[0], ps[-1], len(ps))]
        out = []
        for k in range(len(cuts)):
            i = (cuts[k] + 1) % len(ps); j = cuts[(k + 1) % len(cuts)]; idx = []
            while True:
                idx.append(ps[i])
                if i == j: break
                i = (i + 1) % len(ps)
            out.append((idx[0], idx[-1], len(idx)))
        return out

    def _locate(self):
        OFF, FAM, SPOT, NG, GL, NF, REG, NR, FAMS = self.OFF, self.FAM, self.SPOT, self.NG, self.GL, self.NF, self.REG, self.NR, self.FAMS
        fampos, spotpos = [], []
        for g in range(NG):
            a = collections.defaultdict(list); b = collections.defaultdict(list)
            for p, (f, s) in enumerate(zip(FAM[OFF[g]:OFF[g + 1]].tolist(), SPOT[OFF[g]:OFF[g + 1]].tolist())):
                a[f].append(p)
                if s >= 0: b[s].append(p)
            fampos.append(a); spotpos.append(b)
        uk, cnt = np.unique(self.GID.astype(np.int64) * NF + FAM, return_counts=True)
        self.BB = BB = np.bincount((uk % NF)[cnt == 1], minlength=NF) >= 0.95 * NG
        share = np.zeros((NR, NG))
        for ri, fs in enumerate(FAMS):
            for g in range(NG): share[ri, g] = sum(1 for f in fs if fampos[g].get(f)) / len(fs)
        self.PRES = PRES = share >= CARRY
        leaf = self._leaf()
        bad = sum(str(min(9, int(share[ri, g] * 9.999))) != REG[ri]["presence"][leaf[g]]
                  for ri in range(NR) for g in range(NG))
        assert bad == 0, f"presence strings of region_sets.json not reproduced ({bad} digits differ)"

        def bb_flanks(g, s, e):
            L = int(GL[g]); sl = FAM[OFF[g]:OFF[g + 1]]; out = []
            for step, p0 in ((-1, s), (+1, e)):
                p = p0
                for _ in range(500):
                    p = (p + step) % L; f = int(sl[p])
                    if BB[f] and len(fampos[g].get(f, ())) == 1: out.append(f); break
                else: out.append(None)
            return out
        FLANK = []
        for ri, r in enumerate(REG):
            lc = collections.Counter(); rc = collections.Counter(); n = 0
            for g in range(NG):
                if not PRES[ri, g]: continue
                ps = [p for f in FAMS[ri] for p in fampos[g].get(f, ())]
                if not ps: continue
                ref = (min(spotpos[g][r["spot"]]) if (r["spot"] is not None and spotpos[g].get(r["spot"])) else r["center"])
                bl = sorted(self._circ_blocks(ps, int(GL[g])), key=lambda b: (abs((b[0] + b[1]) / 2 - ref), -b[2]))[0]
                lf, rf = bb_flanks(g, bl[0], bl[1]); n += 1
                if lf is not None: lc[lf] += 1
                if rf is not None: rc[rf] += 1
            FLANK.append(dict(left=int(lc.most_common(1)[0][0]) if lc else None, right=int(rc.most_common(1)[0][0]) if rc else None,
                              lfrac=(lc.most_common(1)[0][1] / n) if lc and n else 0., n=n))
        EL = {}; INS = np.full((NR, NG), -1); how = collections.Counter()
        for ri, r in enumerate(REG):
            s = r["spot"]
            for g in range(NG):
                blk = spotpos[g].get(s) if s is not None else None
                member = [p for f in FAMS[ri] for p in fampos[g].get(f, ())]
                if blk:
                    lo, hi = min(blk), max(blk); INS[ri, g] = lo; how["spot block"] += 1
                    inside = sorted(p for p in member if lo - SLACK <= p <= hi + SLACK)
                    if inside:
                        cl = sorted(self._circ_blocks(inside, int(GL[g])), key=lambda b: (-b[2], abs(b[0] - lo)))[0]
                        EL[(ri, g)] = dict(start=cl[0], end=cl[1], members=cl[2], span=(cl[1] - cl[0]) % int(GL[g]) + 1,
                                           where="spot", spot_lo=lo, spot_hi=hi)
                    continue
                lf = FLANK[ri]["left"]; lp = fampos[g].get(lf, ()) if lf is not None else ()
                rf = FLANK[ri]["right"]; rp = fampos[g].get(rf, ()) if rf is not None else ()
                pt = lp[0] + 1 if len(lp) == 1 else (rp[0] if len(rp) == 1 else None)
                if pt is None: how["nowhere"] += 1; continue
                INS[ri, g] = pt; how["flanks"] += 1
                if member:
                    bl = sorted(self._circ_blocks(member, int(GL[g])), key=lambda b: (abs((b[0] + b[1]) / 2 - pt), -b[2]))[0]
                    if abs(bl[0] - pt) <= NEAR:
                        EL[(ri, g)] = dict(start=bl[0], end=bl[1], members=bl[2], span=(bl[1] - bl[0]) % int(GL[g]) + 1,
                                           where="flanks", spot_lo=None, spot_hi=None)
        HAS = np.zeros((NR, NG), bool)
        for (ri, g) in EL: HAS[ri, g] = True
        TEST = PRES & HAS
        ENTRY = []
        for ri in range(NR):
            c = collections.Counter(); n = 0
            for g in range(NG):
                if TEST[ri, g]: c[int(FAM[OFF[g] + EL[(ri, g)]["start"]])] += 1; n += 1
            ENTRY.append(dict(family=int(c.most_common(1)[0][0]) if c else None, frac=(c.most_common(1)[0][1] / n) if n else 0., n=n))
        self.EL, self.INS, self.HAS, self.TEST, self.ENTRY, self.FLANK, self.how = EL, INS, HAS, TEST, ENTRY, FLANK, dict(how)

    def _leaf(self):
        m = {g["acc"]: i for i, g in enumerate(self.RS["genomes"])}
        return [m[a] for a in self.ACC]

    # ---------------------------------------------------------------- where an ABSENT element's entry would sit
    def anchors(self, ri):
        """The families that flank region ri in its carriers: (pred family, share, succ family, share).
        pred = the family of the gene just before the element's entry, succ = just after its last gene."""
        if not hasattr(self, "_anch"): self._anch = {}
        if ri not in self._anch:
            pc, sc, n = collections.Counter(), collections.Counter(), 0
            for g in range(self.NG):
                if not self.TEST[ri, g]: continue
                e = self.EL[(ri, g)]; a = self.OFF[g]; L = int(self.GL[g]); n += 1
                pc[int(self.FAM[a + (e["start"] - 1) % L])] += 1
                sc[int(self.FAM[a + (e["end"] + 1) % L])] += 1
            self._anch[ri] = ((int(pc.most_common(1)[0][0]), pc.most_common(1)[0][1] / n) if n else (None, 0.0),
                              (int(sc.most_common(1)[0][0]), sc.most_common(1)[0][1] / n) if n else (None, 0.0))
        return self._anch[ri]

    def ins_row(self, ri, g):
        """Where region ri's ENTRY gene would sit in genome g if the element were inserted at its spot, read from the
        CARRIERS' context. INS (the spot-block start, or the gene after the left backbone flank) is NOT where the
        element starts in carriers -- for nanS it is 12 genes off, for the T6SS the preceding gene is itself absent --
        so reading P(D entry) there lands in the far tail (1e-6) and measures nothing (both reviews' blocker).
          1. row = 1 + the copy nearest INS (within ANCHOR_WIN) of the family preceding the entry in >= 60 % of carriers
          2. else row = the copy nearest INS of the family following the last gene in >= 60 % of carriers
             (the element would be inserted before that gene, so its entry would take that gene's index)
          3. else no insertion anchor -> the unit is skipped
        Returns (row, how) or (None, why)."""
        p0 = int(self.INS[ri, g])
        if p0 < 0: return None, "no insertion point"
        rules = self.anchor_rules(ri)
        if not rules: return None, "no insertion anchor (no flanking family of D that is consensus in its carriers " \
                                   "and reproduces D's entry when the rule is applied to a carrier)"
        a, L = self.OFF[g], int(self.GL[g])
        sl = self.FAM[a:a + L]
        for fam, off, how in rules:
            c = [p for p in range(max(0, p0 - ANCHOR_WIN), min(L, p0 + ANCHOR_WIN + 1)) if int(sl[p]) == fam]
            if not c: continue
            r = min(c, key=lambda p: (abs(p - p0), p)) + off
            if 0 <= r < L: return int(r), how
        return None, "no insertion anchor (D's flanking family is not within 40 genes of the spot here)"

    def anchor_rules(self, ri):
        """The flanking families that can place region ri's entry, in preference order: the family preceding the entry
        in >= 60 % of carriers, then the family following the last gene, keeping only those whose rule reproduces D's
        real entry in >= 50 % of the carriers. Frequency alone is not enough -- for 13 % of the regions on the absent
        side of an avoidance link the most frequent neighbour does not mark where the element starts, and reading there
        would repeat the error this fix is about. Each entry is (family, offset, name)."""
        if not hasattr(self, "_arule"): self._arule = {}
        if ri not in self._arule:
            (fp, sp), (fs, ss) = self.anchors(ri)
            out = []
            for fam, share, off, how in ((fp, sp, 1, "pred"), (fs, ss, 0, "succ")):
                if fam is None or share < ANCHOR_SHARE: continue
                if self._anchor_validate(ri, fam, off) >= 0.5: out.append((int(fam), off, how))
            self._arule[ri] = out
        return self._arule[ri]

    def _anchor_validate(self, ri, fam, off):
        """share of ri's carriers in which the rule returns D's real entry gene (off = 1) / the gene after its last
        gene (off = 0), which is where the entry would land if the element were inserted before that gene"""
        ok = n = 0
        for g in range(self.NG):
            if not self.TEST[ri, g]: continue
            e = self.EL[(ri, g)]; a = self.OFF[g]; L = int(self.GL[g]); sl = self.FAM[a:a + L]
            p0 = int(self.INS[ri, g])
            if p0 < 0: continue
            want = e["start"] if off == 1 else (e["end"] + 1) % L
            c = [p for p in range(max(0, p0 - ANCHOR_WIN), min(L, p0 + ANCHOR_WIN + 1)) if int(sl[p]) == fam]
            n += 1
            if c and min(c, key=lambda p: (abs(p - p0), p)) + off == want: ok += 1
        return (ok / n) if n else 0.0

    def anchor_check(self, ri):
        """Validation of the rule actually used for region ri: (n carriers, share of them where it returns D's real
        entry, rule name). Both candidate rules are also reported, to show what was rejected."""
        rules = self.anchor_rules(ri)
        n = int(self.TEST[ri].sum())
        (fp, sp), (fs, ss) = self.anchors(ri)
        both = {}
        for fam, share, off, how in ((fp, sp, 1, "pred"), (fs, ss, 0, "succ")):
            both[how] = (self._anchor_validate(ri, fam, off) if (fam is not None and share >= ANCHOR_SHARE) else np.nan)
        if not rules: return n, np.nan, None, both
        return n, self._anchor_validate(ri, rules[0][0], rules[0][1]), "+".join(r[2] for r in rules), both

    def pos_of(self, ri, g):
        if self.TEST[ri, g]: return int(self.EL[(ri, g)]["start"])
        return int(self.INS[ri, g]) if self.INS[ri, g] >= 0 else None

    def members(self, ri, g):
        """U's member genes inside its element span (0-based positions in genome g), never a backbone family"""
        e = self.EL[(ri, g)]; a = self.OFF[g]
        fs = set(self.FAMS[ri])
        return [p for p in range(e["start"], e["end"] + 1) if int(self.FAM[a + p]) in fs and not self.BB[self.FAM[a + p]]]

    # ---------------------------------------------------------------- testable units
    def units(self, links):
        """one unit = one link in one genome: delete U (upstream in this genome), read D.
        Returns (units, skip counter, skipped list)."""
        OFF, FAM = self.OFF, self.FAM
        units, skipped = [], []
        skip = collections.Counter()
        for li in links:
            l = self.LINKS[li]; a, b, sg = l["a"], l["b"], l["sign"]
            for g in range(self.NG):
                ca, cb = bool(self.TEST[a, g]), bool(self.TEST[b, g])
                if sg > 0 and not (ca and cb): continue                       # not a co-occurrence genome: not a unit
                if sg < 0 and ca == cb: continue                              # both or neither: not an avoidance genome
                pa, pb = self.pos_of(a, g), self.pos_of(b, g)
                why = None
                if pa is None or pb is None: why = "no position for an element"
                else:
                    up, dn = (a, b) if pa < pb else (b, a)
                    if sg < 0 and not self.TEST[up, g]: why = "avoidance: the absent element is upstream (would need an insertion)"
                    elif sg < 0 and (self.PRES[dn, g] or self.HAS[dn, g]):
                        why = "avoidance: D not cleanly absent (its families elsewhere or a minority at its spot)"
                if why is None:
                    eu = self.EL[(up, g)]
                    if eu["end"] < eu["start"]: why = "U wraps the origin"
                if why is None and sg > 0:
                    ed = self.EL[(dn, g)]
                    if ed["end"] < ed["start"]: why = "D wraps the origin"
                how = ""
                if why is None:
                    ue = eu["end"]
                    if sg > 0:
                        rd = int(self.EL[(dn, g)]["start"]); f1 = int(FAM[OFF[g] + rd]); f2 = -1
                    else:
                        f1 = self.ENTRY[dn]["family"]
                        rd, how = self.ins_row(dn, g)
                        if f1 is None: why = "avoidance: D has no entry family (no carrier)"
                        elif rd is None: why = "avoidance: " + how
                        else:
                            f2 = int(FAM[OFF[g] + rd])
                            if f1 == f2: why = "avoidance: the resident gene is D's entry family"
                if why is None:
                    if rd <= ue: why = "U and D overlap (D's readout gene not after U's last gene)"
                    elif rd - ue - 1 < MIN_APART: why = f"U and D < {MIN_APART} genes apart"
                    elif MAX_APART is not None and rd - ue - 1 > MAX_APART: why = f"U and D > {MAX_APART} genes apart"
                if why is None:
                    dels = self.members(up, g)
                    if not dels: why = "U has no member gene outside the backbone"
                if why is not None:
                    skip[why] += 1; skipped.append(dict(li=li, g=g, sign=sg, why=why)); continue
                sec = []
                if sg > 0:                                                     # D's member genes (secondary readout)
                    sec = self.members(dn, g)
                # how complete is the knockout? U's families left elsewhere in the chromosome (multi-copy elements:
                # for some links the "knockout" leaves every family in place -- mechanics review finding 2)
                sl = FAM[OFF[g]:OFF[g + 1]]; left = np.delete(sl, np.asarray(dels, np.int64))
                fs = set(self.FAMS[up]); pres_f = {f for f in fs if (sl == f).any()}
                u_left = sum(1 for f in pres_f if (left == f).any())
                units.append(dict(li=li, g=g, sign=sg, up=up, dn=dn, u_start=int(eu["start"]), u_end=int(ue),
                                  u_dels=dels, u_size=len(dels), u_span=int(eu["span"]), row=rd, f1=int(f1), f2=int(f2),
                                  dist=int(rd - ue), d_sec=sec, row_how=how,
                                  u_fam=len(pres_f), u_left=int(u_left),
                                  f1_up=int((sl[:rd] == f1).sum()),      # is the readout family specific to D?
                                  d_start=int(self.EL[(dn, g)]["start"]) if sg > 0 else rd,
                                  d_end=int(self.EL[(dn, g)]["end"]) if sg > 0 else rd))
        return units, skip, skipped

    # ---------------------------------------------------------------- the exchangeable null and the off-targets
    def links_of(self, ri):
        """regions linked to ri in region_sets.json (either sign)"""
        if not hasattr(self, "_lof"):
            self._lof = collections.defaultdict(set)
            for l in self.LINKS: self._lof[l["a"]].add(l["b"]); self._lof[l["b"]].add(l["a"])
        return self._lof[ri]

    def control_pool(self, g, want=None, sizes=(), uncapped=False):
        """Whole accessory elements of genome g, each deleted by its member genes: the null for a U deletion.
        The pilot's null was a contiguous SUB-RUN of a natural element, which is not the same kind of object as a
        whole element (whole-vs-part shifts the readout as much as the effects being tested, inference review
        finding 3), and it matched only 4 controls per unit (31 % of units unusable, finding 6). Whole COUPLED
        elements are the right null: same construction as U, same size distribution (77 % are 1-2 genes), ~93 per
        genome. Whole natural elements at spots holding no coupled region are kept as a second arm.
          arm 'cpl': a coupled element present at its spot here; its region id is kept so that, per unit, controls
                     linked to U or D (or U/D themselves) are excluded
          arm 'nat': a maximal run of panRGP genes of one spot at a spot with no coupled region, whole
        `want`: rows that must not be deleted (readout genes). Capped at POOL_MAX, keeping a spread of sizes and
        positions (uncapped=True: every candidate, for near_controls).
        The 'nat' arm is small in practice: the cap keeps the controls that match most U sizes, and the natural runs
        are longer than the 1-2-gene coupled elements, so ~4 % of the kept controls are natural (pilot: 1,714 of
        38,654 control passes; 0 in some genomes). The null is in effect "delete another coupled element"."""
        a, b = int(self.OFF[g]), int(self.OFF[g + 1]); n = b - a
        fam = self.FAM[a:b]; rgp = self.RGP[a:b]; sp = self.SPOT[a:b]
        want = set() if want is None else set(int(x) for x in want)
        pool = []
        for ri in range(self.NR):
            if not (self.HAS[ri, g] and self.PRES[ri, g]): continue
            e = self.EL[(ri, g)]
            if e["end"] < e["start"]: continue                      # wraps the origin
            dels = self.members(ri, g)
            if dels and not (set(dels) & want):
                pool.append(dict(arm="cpl", ri=ri, start=int(e["start"]), end=int(e["end"]), dels=dels,
                                 size=len(dels), spot=(e["spot_lo"] if e["where"] == "spot" else -1)))
        i = 0
        while i < n:
            if rgp[i] and sp[i] >= 0:
                j = i
                while j + 1 < n and rgp[j + 1] and sp[j + 1] == sp[i]: j += 1
                if int(sp[i]) not in self.COUPLED_SPOTS:
                    dels = [p for p in range(i, j + 1) if not self.BB[fam[p]]]
                    if dels and not (set(dels) & want):
                        pool.append(dict(arm="nat", ri=-1, start=i, end=j, dels=dels, size=len(dels), spot=int(sp[i])))
                i = j + 1
            else: i += 1
        if not uncapped and len(pool) > POOL_MAX:
            # keep the controls that can actually serve as a matched null here: score each by the number of this
            # genome's U sizes it matches (x0.5-2), then by a spread of sizes and positions
            sz = np.asarray(sizes, float)
            def score(c):
                n_ = int(np.sum((c["size"] >= MATCH[0] * sz) & (c["size"] <= MATCH[1] * sz))) if len(sz) else 0
                return (-n_, c["size"], c["start"])
            pool.sort(key=score)
            head = [c for c in pool if score(c)[0] < 0][:POOL_MAX]
            if len(head) < POOL_MAX:
                rest = [c for c in pool if score(c)[0] == 0]
                idx = np.unique(np.linspace(0, len(rest) - 1, POOL_MAX - len(head)).round().astype(int)) if rest else []
                head += [rest[k] for k in idx]
            pool = head
        return sorted(pool, key=lambda c: c["start"])

    def near_controls(self, g, units, pool, want=5):
        """Extra whole-element controls for units whose matched null is thin (close range: with <= 72 controls spread
        over ~5,000 genes, 0 % of the units 50-100 genes apart and 37 % of those 100-250 apart had >= 3 matched
        controls in the close smoke run). For each unit of genome g, every candidate whole element (control_pool,
        uncapped: coupled elements present at their spot, natural panRGP runs at spots without a coupled region) that
        is usable in its null (no overlap with U or D, not U, D or a region linked to either, ending before D's
        readout gene) and matches it under the widened rule (size x1/3-3, distance to the readout within +-60 %, or
        within x1/F..xF with MATCH_LOG = F) is added, nearest in log distance and log size first, while it raises the unit's matched count as
        match_controls() will compute it in the aggregation, until `want`. Returns the extra controls (not already in
        `pool`), sorted by start."""
        allc = self.control_pool(g, uncapped=True)
        key = lambda c: (c["start"], c["end"], c["arm"], c["ri"])
        have = {key(c) for c in pool}; extra = {}
        for u in units:
            banned = {u["up"], u["dn"]} | self.links_of(u["up"]) | self.links_of(u["dn"])

            def usable(c):
                return (c["end"] < u["row"] and not (c["start"] <= u["u_end"] and c["end"] >= u["u_start"])
                        and not (c["start"] <= u["d_end"] and c["end"] >= u["d_start"])
                        and not (c["ri"] >= 0 and c["ri"] in banned))

            def nmatch(cs):
                if not cs: return 0
                idx, _w = match_controls(np.array([c["end"] for c in cs]), np.array([c["size"] for c in cs]),
                                         np.ones(len(cs), bool), u["u_size"], u["dist"], u["row"])
                return len(idx)
            cur = [c for c in pool if usable(c)] + [c for c in extra.values() if usable(c)]
            n0 = nmatch(cur)
            if n0 >= want: continue
            cand = [c for c in allc if key(c) not in have and key(c) not in extra and usable(c)
                    and WIDEN[0] * u["u_size"] <= c["size"] <= WIDEN[1] * u["u_size"]
                    and (abs((u["row"] - c["end"]) - u["dist"]) <= WIDEN[2] * u["dist"] or
                         (MATCH_LOG is not None and abs(np.log(max(u["row"] - c["end"], 1) / u["dist"])) <= np.log(MATCH_LOG)))]
            cand.sort(key=lambda c: (abs(np.log((u["row"] - c["end"]) / u["dist"])), abs(np.log(c["size"] / u["u_size"])), c["start"]))
            for c in cand:
                n1 = nmatch(cur + [c])
                if n1 > n0: cur.append(c); extra[key(c)] = c; n0 = n1
                if n0 >= want: break
        return sorted(extra.values(), key=lambda c: c["start"])

    def offtargets(self, g, rows_needed):
        """Readouts of OTHER coupled elements in genome g, to test whether a U deletion moves D specifically or moves
        every element downstream (inference review blocker): a present element is read at its entry gene (co kind),
        an absent element with an insertion anchor at that row (avoidance kind, resident = the gene there).
        `rows_needed`: only positions in this set are built (the union of the units' distance windows)."""
        a = int(self.OFF[g]); out = []
        for ri in range(self.NR):
            if self.TEST[ri, g]:
                e = self.EL[(ri, g)]
                if e["end"] < e["start"]: continue
                r = int(e["start"])
                if r in rows_needed: out.append(dict(ri=ri, kind=1, row=r, f1=int(self.FAM[a + r]), f2=-1))
            elif not self.PRES[ri, g] and not self.HAS[ri, g]:
                f1 = self.ENTRY[ri]["family"]
                if f1 is None: continue
                r, _how = self.ins_row(ri, g)
                if r is None or r not in rows_needed: continue
                f2 = int(self.FAM[a + r])
                if f2 != f1: out.append(dict(ri=ri, kind=-1, row=int(r), f1=int(f1), f2=f2))
        return out

    def phi(self, ri, rj):
        """across-genome (largely phylogenetic) correlation of two regions' presence: a region that merely marks a
        strain background will move together with U without any within-lineage coupling"""
        x = self.PRES[ri].astype(float); y = self.PRES[rj].astype(float)
        if x.std() == 0 or y.std() == 0: return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    # ---------------------------------------------------------------- the accessory elements of a genome (influence)
    def elements(self, g):
        """Every accessory element present in genome g, with its entry gene (the first gene in dnaA reading order):
          kind 0  a maximal run of panRGP genes sharing one spot id (every spot, coupled or not), id = spot id;
                  size = its non-backbone genes (what a deletion of the whole run removes, as control_pool's 'nat')
          kind 1  a coupled region present at its spot (PRES and HAS, the carriers of the units), id = region id,
                  element span EL, size = its member genes; elements wrapping the origin are left out
        panRGP genes outside every spot are not listed (no id to follow them across genomes). A coupled element
        usually lies inside a spot run, so both can share an entry gene. Sorted by entry; cached per genome."""
        if not hasattr(self, "_elts"): self._elts = {}
        if g in self._elts: return self._elts[g]
        a, b = int(self.OFF[g]), int(self.OFF[g + 1]); n = b - a
        fam = self.FAM[a:b]; rgp = self.RGP[a:b]; sp = self.SPOT[a:b]
        out = []
        i = 0
        while i < n:
            if rgp[i] and sp[i] >= 0:
                j = i
                while j + 1 < n and rgp[j + 1] and sp[j + 1] == sp[i]: j += 1
                out.append((0, int(sp[i]), i, j, i, sum(1 for p in range(i, j + 1) if not self.BB[fam[p]])))
                i = j + 1
            else: i += 1
        for ri in range(self.NR):
            if not (self.HAS[ri, g] and self.PRES[ri, g]): continue
            e = self.EL[(ri, g)]
            if e["end"] < e["start"]: continue
            out.append((1, ri, int(e["start"]), int(e["end"]), int(e["start"]), len(self.members(ri, g))))
        out.sort(key=lambda t: (t[4], t[0], t[1]))
        A = np.array(out, np.int64).reshape(-1, 6)
        E = dict(kind=A[:, 0].astype(np.int8), id=A[:, 1].astype(np.int32), start=A[:, 2].astype(np.int32),
                 end=A[:, 3].astype(np.int32), entry=A[:, 4].astype(np.int32), size=A[:, 5].astype(np.int32))
        E["fam"] = fam[E["entry"]].astype(np.int32) if len(A) else np.zeros(0, np.int32)
        self._elts[g] = E
        return E

    def element_label(self, kind, eid):
        return (self.REG[int(eid)]["label"] if int(kind) == 1 else f"spot {int(eid)}")

    # ---------------------------------------------------------------- control panel
    def panel(self, g, target_sizes, avoid, rng):
        """<= PANEL_MAX control deletions in genome g, one per target size.
        Natural accessory elements = maximal runs of panRGP genes sharing one spot id, at spots holding none of
        the 290 coupled regions, not touching any coupled element here (`avoid` = positions). They are ~24 per
        genome with >= 3-4 member genes each, while half the U deletions are 1-2 genes, so a control deletion is
        a contiguous sub-run of t member genes (t = target size) of such an element -- itself a contiguous run of
        panRGP genes of one spot, as a U element is a sub-run of its own spot's RGP. An element hosts several
        non-overlapping sub-runs only when every element has been used (fewest uses first); among those,
        farthest-point choice spreads the panel along the chromosome."""
        a, b = self.OFF[g], self.OFF[g + 1]
        rgp = self.RGP[a:b]; sp = self.SPOT[a:b]; fam = self.FAM[a:b]
        cand = []
        i, n = 0, b - a
        while i < n:
            if rgp[i] and sp[i] >= 0:
                j = i
                while j + 1 < n and rgp[j + 1] and sp[j + 1] == sp[i]: j += 1
                if int(sp[i]) not in self.COUPLED_SPOTS:
                    pos = [p for p in range(i, j + 1) if not self.BB[fam[p]]]
                    if pos and not (set(range(i, j + 1)) & avoid):
                        cand.append(dict(start=i, end=j, pos=pos, spot=int(sp[i]), free=[True] * len(pos), uses=0))
                i = j + 1
            else: i += 1
        chosen = []
        order = sorted(range(len(target_sizes)), key=lambda k: -target_sizes[k])     # large first: rarer
        for k in order:
            t = int(target_sizes[k]); opts = []
            for ci, c in enumerate(cand):                                             # free windows of t member genes
                w = [s0 for s0 in range(len(c["pos"]) - t + 1) if all(c["free"][s0:s0 + t])]
                if w: opts.append((ci, w))
            if not opts: continue
            mu = min(cand[ci]["uses"] for ci, _ in opts); opts = [(ci, w) for ci, w in opts if cand[ci]["uses"] == mu]
            if chosen:
                mids = np.array([(e["start"] + e["end"]) / 2 for e in chosen])
                dmin = [np.abs(mids - (cand[ci]["start"] + cand[ci]["end"]) / 2).min() for ci, _ in opts]
                best = max(dmin); opts = [o for o, d in zip(opts, dmin) if d >= best - 1e-9]
            ci, w = opts[int(rng.integers(len(opts)))]
            s0 = w[int(rng.integers(len(w)))]; c = cand[ci]
            for q in range(s0, s0 + t): c["free"][q] = False
            c["uses"] += 1
            dels = c["pos"][s0:s0 + t]
            chosen.append(dict(start=dels[0], end=dels[-1], dels=dels, size=t, spot=c["spot"],
                               elem_start=c["start"], elem_end=c["end"], elem_size=len(c["pos"])))
        return sorted(chosen, key=lambda e: e["start"]), len(cand)


# ============================================================================ decoder P(family | cluster)
class Decoder:
    """A[f, c] = sum over genes of family f (not the first gene of a genome) of P(c | context), top 64 of the
    fp32 full-chromosome baseline calls, one per lineage half; P(f | c) = A[f, c] / sum_f A[f, c].
    Genome g (half h) is read with the decoder of half 1 - h, built from genomes of the other lineages only.
    Two sources: the fp32 baseline calls base_top64_{f,p}.npy (~0.9 GB; every method), or, when they are absent (or
    compact=True), the export of export_decoder() (DECODER_NPZ, or `path`): the sparse rows srow() of every family of
    the genomes each half reads, bit for bit (float64; a float32 export moves the logit of a gene predicted at
    P > 0.99 by up to 2e-4, the deltas by < 1e-6). row() is then rebuilt from srow(); TF, TP and colsum do not exist,
    and a family outside those genomes raises KeyError."""

    def __init__(self, data, compact=None, path=None):
        self.d = data
        NP = data.NP
        ff, fp = f"{KDIR}/base_top64_f.npy", f"{KDIR}/base_top64_p.npy"
        if compact is None: compact = not (os.path.exists(ff) and os.path.exists(fp))
        self.compact = bool(compact)
        half_gene = data.HALF[data.GID]
        self.mask = {h: (half_gene == h) & ~data.first for h in (0, 1)}
        self.lineages = {h: set(data.LIN[data.HALF == h].tolist()) for h in (0, 1)}
        self._rows = {}
        if self.compact:
            self._load_compact(path or DECODER_NPZ); return
        self.TF = np.load(ff, mmap_mode="r")
        self.TP = np.load(fp, mmap_mode="r")
        assert self.TF.shape == (NP, TOPK)
        self.colsum = {}
        for h in (0, 1):
            m = np.flatnonzero(self.mask[h]); cs = np.zeros(NCL)
            for s in range(0, len(m), 200000):
                mm = m[s:s + 200000]
                cs += np.bincount(np.asarray(self.TF[mm]).ravel(), weights=np.asarray(self.TP[mm], np.float64).ravel(), minlength=NCL)
            self.colsum[h] = cs
        # genes of each family, per half (sorted global indices)
        order = np.argsort(data.FAM, kind="stable"); fs = data.FAM[order]
        self._ord = order; self._fs = fs

    def _load_compact(self, path):
        """the arrays of export_decoder(), checked against this Data (same genes, same lineage halves)"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"no decoder: neither pgb/knock/base_top64_{{f,p}}.npy nor {os.path.relpath(path, PROJ)}")
        with np.load(path) as Z:
            meta = json.loads(str(Z["meta"]))
            assert meta["np"] == self.d.NP and meta["ncl"] == NCL, f"{path}: built for other data ({meta})"
            assert np.array_equal(Z["half"], self.d.HALF), f"{path}: its lineage halves are not this Data's"
            self.sp = {h: tuple(np.asarray(Z[f"{k}{h}"]) for k in ("fam", "ptr", "col", "val", "cover")) for h in (0, 1)}
        self.meta = meta

    def half_for(self, g):
        """the decoder that reads genome g: the other half. Self-check (d): it never saw g's lineage."""
        h = 1 - int(self.d.HALF[g])
        assert self.d.LIN[g] not in self.lineages[h], f"genome {g}: decoder {h} saw its lineage {self.d.LIN[g]}"
        assert not self.mask[h][self.d.OFF[g]:self.d.OFF[g + 1]].any(), f"genome {g}: its own genes are in decoder {h}"
        return h

    def row(self, f, h):
        """P(f | c) for every cluster c (float64, length NCL), decoder half h"""
        k = (int(f), h)
        if k not in self._rows and self.compact:
            c, w = self.srow(f, h); a = np.zeros(NCL); a[c] = w; self._rows[k] = a
        if k not in self._rows:
            lo, hi = np.searchsorted(self._fs, [f, f + 1]); idx = np.sort(self._ord[lo:hi])
            idx = idx[self.mask[h][idx]]
            a = np.zeros(NCL)
            if len(idx):
                a = np.bincount(np.asarray(self.TF[idx]).ravel(), weights=np.asarray(self.TP[idx], np.float64).ravel(), minlength=NCL)
            with np.errstate(invalid="ignore", divide="ignore"):
                w = np.where(self.colsum[h] > 0, a / np.maximum(self.colsum[h], 1e-300), 0.0)
            self._rows[k] = np.clip(w, 0.0, 1.0)
        return self._rows[k]

    def srow(self, f, h):
        """row(f, h) as (clusters, values): the same float64 values, only the non-zero ones (median ~380 of 50,001
        clusters per family, at most ~12,700), so that the true family of EVERY gene of a genome can be decoded
        (~4,500 families per genome: dense rows would take 1.8 GB). Cached (~5 kB per family and half); the dense
        row() cache is not touched. Compact decoder: the exported row."""
        if self.compact:
            fam, ptr, col, val, cover = self.sp[h]; f = int(f)
            i = int(np.searchsorted(fam, f))
            if i < len(fam) and fam[i] == f:
                return col[ptr[i]:ptr[i + 1]].astype(np.int64), val[ptr[i]:ptr[i + 1]].astype(np.float64)
            j = int(np.searchsorted(cover, f))
            if j < len(cover) and cover[j] == f: return np.zeros(0, np.int64), np.zeros(0)
            raise KeyError(f"family {f}: not in the compact decoder's half {h} (only the families of the genomes it reads)")
        if not hasattr(self, "_srows"): self._srows = {}
        k = (int(f), h)
        if k not in self._srows:
            lo, hi = np.searchsorted(self._fs, [f, f + 1]); idx = np.sort(self._ord[lo:hi])
            idx = idx[self.mask[h][idx]]
            if len(idx):
                a = np.bincount(np.asarray(self.TF[idx]).ravel(), weights=np.asarray(self.TP[idx], np.float64).ravel(), minlength=NCL)
                c = np.flatnonzero(a > 0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    w = np.where(self.colsum[h][c] > 0, a[c] / np.maximum(self.colsum[h][c], 1e-300), 0.0)
                w = np.clip(w, 0.0, 1.0); keep = w > 0
                self._srows[k] = (c[keep].astype(np.int64), w[keep])
            else:
                self._srows[k] = (np.zeros(0, np.int64), np.zeros(0))
        return self._srows[k]

    def sparse_table(self, fams, h):
        """CSR of the decoder rows of `fams` (half h): cols, vals, ptr (len(fams)+1), nnz"""
        rs = [self.srow(f, h) for f in fams]
        nnz = np.array([len(c) for c, _ in rs], np.int64)
        ptr = np.concatenate([[0], np.cumsum(nnz)])
        cols = np.concatenate([c for c, _ in rs]) if len(rs) else np.zeros(0, np.int64)
        vals = np.concatenate([v for _, v in rs]) if len(rs) else np.zeros(0)
        return cols, vals, ptr, nnz


def export_decoder(path=None, dtype=np.float64, data=None, compress=True):
    """The compact decoder for influence() where base_top64_{f,p}.npy (~0.9 GB) cannot go (the Space): for each half h,
    the sparse rows srow(f, h) of every family f of the genomes read with h (half_for: the genomes of half 1 - h), the
    only rows influence() reads. Per half: cover{h} (those families, int32), fam{h} (the ones with a non-empty row),
    ptr{h} (int64 CSR offsets), col{h} (cluster ids, uint16: NCL = 50,001 < 65,536), val{h} (P(f | c), `dtype`:
    float64 keeps srow() bit for bit, ~75 MB; float32 ~45 MB); `half` (the lineage half of each genome, checked on
    load) and `meta`. ~90 s. Returns (path, bytes)."""
    assert NCL <= 65536
    path = path or DECODER_NPZ
    D = data if data is not None else Data(); dec = Decoder(D, compact=False)
    out = dict(half=D.HALF.astype(np.int8)); info = {}
    for h in (0, 1):
        gs = np.flatnonzero(D.HALF == 1 - h)
        cover = np.unique(np.concatenate([D.FAM[D.OFF[g]:D.OFF[g + 1]] for g in gs]))
        fams, ptr, cols, vals = [], [0], [], []
        for f in cover.tolist():
            c, w = dec.srow(f, h)
            if not len(c): continue
            fams.append(f); cols.append(c.astype(np.uint16)); vals.append(w.astype(dtype)); ptr.append(ptr[-1] + len(c))
        out[f"cover{h}"] = cover.astype(np.int32); out[f"fam{h}"] = np.array(fams, np.int32)
        out[f"ptr{h}"] = np.array(ptr, np.int64); out[f"col{h}"] = np.concatenate(cols); out[f"val{h}"] = np.concatenate(vals)
        info[h] = dict(genomes_read=int(len(gs)), families=int(len(cover)), rows=len(fams), nnz=int(ptr[-1]))
        dec._srows = {}
    out["meta"] = np.array(json.dumps(dict(np=int(D.NP), ncl=NCL, topk=TOPK, dtype=np.dtype(dtype).name, halves=info,
                                           sets_md5=D.SETS_MD5, built=time.strftime("%Y-%m-%d %H:%M"),
                                           source="pgb/knock/base_top64_{f,p}.npy (fp32 baseline calls, top 64)",
                                           source_files={os.path.basename(f): dict(size=os.path.getsize(f), mtime=os.path.getmtime(f))
                                                         for f in (f"{KDIR}/base_top64_f.npy", f"{KDIR}/base_top64_p.npy")})))
    tmp = path + ".tmp.npz"
    (np.savez_compressed if compress else np.savez)(tmp, **out)
    os.replace(tmp, path)
    log(f"decoder -> {path}: {os.path.getsize(path) / 1e6:.1f} MB, {info}")
    return path, os.path.getsize(path)


# ============================================================================ the model
class Runner:
    """Bacformer causal, fp32, one chromosome per pass (batch 1), padded to a fixed length per genome.
    Readouts go through the decoder in float64 on the GPU:
       S[r, f] = sum_c p[r, c] W[f, c]          = P(f | ctx)
       C[r, f] = sum_c p[r, c] (1 - W[f, c])    = 1 - P(f | ctx), computed directly (no cancellation)"""

    def __init__(self, dev="cuda:0", frac=MEM_FRAC, threads=None):
        """frac: per-process GPU memory cap (0.35 so that two runs share the 8 GB card; None leaves the process's
        cap alone, e.g. inside serve_live). dev='cpu' runs on the CPU with `threads` threads."""
        import torch
        from transformers import AutoModelForCausalLM
        self.torch = torch
        if str(dev).startswith("cuda") and frac is not None:
            torch.cuda.set_per_process_memory_fraction(frac, torch.device(dev).index or 0)
        if not str(dev).startswith("cuda") and threads: torch.set_num_threads(int(threads))
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        self.dev = dev
        self.cm = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True).float().eval().to(dev)
        self.E = None

    def sync(self):
        if str(self.dev).startswith("cuda"): self.torch.cuda.synchronize()

    def set_genome(self, emb_rows, L):
        """emb_rows: (n, 480) float array of the genome's proteins; L: padded sequence length for all passes"""
        t = self.torch
        self.E = t.tensor(np.asarray(emb_rows, np.float32), device=self.dev)
        self.L = int(L)
        self.stm = t.full((1, self.L), PROT, device=self.dev, dtype=t.long); self.stm[0, 0] = CLS
        self.tt = t.zeros((1, self.L), device=self.dev, dtype=t.long)

    def set_families(self, W):
        """W: (F, NCL) float64 decoder rows used for this genome (fixed for all its passes)"""
        t = self.torch
        self.W = t.tensor(W, device=self.dev, dtype=t.float64)
        self.Wc = 1.0 - self.W

    def hidden(self, src, extra=None):
        """src: int array, new position -> row of self.E (>= 0) or -(1 + k) for extra[k] (inserted proteins)"""
        t = self.torch
        m = len(src); assert m + 1 <= self.L
        x = t.zeros((1, self.L, 480), device=self.dev)
        s = t.as_tensor(np.asarray(src), device=self.dev)
        if extra is None:
            x[0, 1:1 + m] = self.E[s]
        else:
            X = t.cat([self.E, t.tensor(np.asarray(extra, np.float32), device=self.dev)])
            s = t.where(s >= 0, s, self.E.shape[0] + (-s - 1))
            x[0, 1:1 + m] = X[s]
        with t.no_grad():
            return self.cm.bacformer(protein_embeddings=x, special_tokens_mask=self.stm, token_type_ids=self.tt,
                                     attention_mask=None, return_attn_weights=False, return_dict=True,
                                     is_causal=True).last_hidden_state[0]

    def decode(self, h, rows):
        """rows (new positions) -> (S, C) float64 numpy arrays (len(rows), F); fixed-size head chunks"""
        t = self.torch
        S, C = [], []
        rows = np.asarray(rows, np.int64)
        with t.no_grad():
            for s in range(0, len(rows), HEAD_CHUNK):
                r = rows[s:s + HEAD_CHUNK]; k = len(r)
                rr = np.zeros(HEAD_CHUNK, np.int64); rr[:k] = r
                p = t.softmax(self.cm.gm_head(h[t.as_tensor(rr, device=self.dev)]), -1).double()
                S.append((p @ self.W.T)[:k].cpu().numpy()); C.append((p @ self.Wc.T)[:k].cpu().numpy())
        if not S:
            F = self.W.shape[0]; return np.zeros((0, F)), np.zeros((0, F))
        return np.concatenate(S), np.concatenate(C)

    def set_sparse(self, cols, vals, ptr, nnz):
        """the decoder rows of every family of the genome (Decoder.sparse_table), for decode_true. One fixed width per
        genome (its largest row), so that every head chunk has the same shape in every pass."""
        t = self.torch
        self.sp_cols = t.as_tensor(np.concatenate([cols, [0]]), dtype=t.long, device=self.dev)       # + a zero slot
        self.sp_vals = t.as_tensor(np.concatenate([vals, [0.0]]), dtype=t.float64, device=self.dev)
        self.sp_ptr = t.as_tensor(np.asarray(ptr[:-1], np.int64), device=self.dev)
        self.sp_nnz = t.as_tensor(np.asarray(nnz, np.int64), device=self.dev)
        self.sp_zero = len(cols)
        w = max(1, int(np.max(nnz)) if len(nnz) else 1)
        self.sp_ar = t.arange(-(-w // 64) * 64, device=self.dev)[None, :]      # a multiple of 64: aligned rows

    def decode_true(self, h, rows, fidx):
        """rows (new positions), fidx (index into the set_sparse table: each row's own family) -> (S, C) float64 numpy
        arrays (len(rows),): S = P(f | ctx) = sum_c p[r, c] W[f, c] over the full softmax (the same float64 sum as
        decode, over the non-zero W only; agrees with decode to ~2e-7 in logit, the fp32 rounding of the softmax),
        C = sum_c p[r, c] - S = sum_c p[r, c] (1 - W[f, c]). Fixed-size head chunks (TRUE_CHUNK rows)
        and a fixed gather width: a row's value does not depend on the other rows of its chunk nor on its slot in it
        (asserted per genome in check (g) by reading every row of the baseline again in other slots). CUDA's row
        reductions (softmax, sum) take another summation path when a row does not start on an aligned address, and a
        50,001-wide row starts misaligned in half the slots, so the softmax runs on rows padded to 50,048 with -inf
        (probability exactly 0) and the gather width is a multiple of 64: every row starts 256-byte aligned."""
        t = self.torch
        rows = np.asarray(rows, np.int64); fidx = np.asarray(fidx, np.int64)
        S, C = [], []
        NPAD = -(-NCL // 64) * 64; CH = TRUE_CHUNK
        with t.no_grad():
            zp = t.full((CH, NPAD), float("-inf"), device=self.dev)
            for s in range(0, len(rows), CH):
                r = rows[s:s + CH]; k = len(r)
                rr = np.zeros(CH, np.int64); rr[:k] = r
                ff = np.zeros(CH, np.int64); ff[:k] = fidx[s:s + CH]
                zp[:, :NCL] = self.cm.gm_head(h[t.as_tensor(rr, device=self.dev)])
                p = t.softmax(zp, -1).double()
                ft = t.as_tensor(ff, device=self.dev)
                ok = self.sp_ar < self.sp_nnz[ft][:, None]
                ix = t.where(ok, self.sp_ptr[ft][:, None] + self.sp_ar, self.sp_zero)
                sv = (p.gather(1, self.sp_cols[ix]) * self.sp_vals[ix]).sum(1)
                S.append(sv[:k].cpu().numpy()); C.append((p.sum(1) - sv)[:k].cpu().numpy())
        if not S: return np.zeros(0), np.zeros(0)
        return np.concatenate(S), np.concatenate(C)

    def full_calls(self, h, n, chunk=512):
        """top-64 clusters, probabilities and entropy (bits) at rows 0..n-1 (row k = the call for gene k)"""
        t = self.torch
        TF, TPp, EN = [], [], []
        with t.no_grad():
            for s in range(0, n, chunk):
                r = t.arange(s, min(n, s + chunk), device=self.dev)
                p = t.softmax(self.cm.gm_head(h[r]), -1)
                tp, tf = p.topk(TOPK, 1)
                TF.append(tf.int().cpu().numpy()); TPp.append(tp.cpu().numpy())
                EN.append((-(p * t.log2(p.clamp(min=1e-30))).sum(1)).cpu().numpy())
        return np.concatenate(TF), np.concatenate(TPp), np.concatenate(EN)


# ============================================================================ readouts
def readout_values(kind, S, C, i1, i2):
    """primary value per readout, from decoded S = P(f|ctx), C = 1 - P(f|ctx) (arrays over readouts)
         co: logit P(f1) = log P(f1) - log(1 - P(f1))
         av: log P(f1) - log P(f2)            (f1 = D's entry family, f2 = the resident gene's family)
       each probability floored at FLOOR"""
    lp1 = np.log(np.maximum(S[np.arange(len(i1)), i1], FLOOR))
    lc1 = np.log(np.maximum(C[np.arange(len(i1)), i1], FLOOR))
    lp2 = np.log(np.maximum(S[np.arange(len(i2)), np.maximum(i2, 0)], FLOOR))
    v = np.where(kind == 1, lp1 - lc1, lp1 - lp2)
    return v, lp1, np.where(kind == 1, lc1, lp2)


# ============================================================================ influence readouts
def logit_true(S, C):
    """logit P(true family) from decode_true, each probability floored at FLOOR (as readout_values' co kind)"""
    return np.log(np.maximum(S, FLOOR)) - np.log(np.maximum(C, FLOOR))


def inf_bins(dd, dl, ok, rgp, win=INF_WIN, width=INF_BIN):
    """dd: distance of each read gene from the deleted element's last gene (1 = the next gene), dl: its delta, ok: usable
    (the decoder saw its family), rgp: panRGP gene. -> per bin of `width` genes out to `win`: mean delta, mean |delta|,
    mean delta over panRGP genes, genes (all, panRGP). NaN where a bin is empty."""
    nb = win // width
    m = ok & (dd >= 1) & (dd <= win)
    b = (dd[m] - 1) // width; x = dl[m]; rg = rgp[m]
    n = np.bincount(b, minlength=nb); nr = np.bincount(b[rg], minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.bincount(b, x, nb) / n
        mabs = np.bincount(b, np.abs(x), nb) / n
        mr = np.bincount(b[rg], x[rg], nb) / nr
    return mean, mabs, mr, n, nr


def influence_readout(R, h, keep, n, dels, pend, base_lg, tfi, seen, rgp, entry, lim=None,
                      up=INF_UP, win=INF_WIN, ent_max=INF_ENT):
    """The influence readouts of one deletion pass, from its hidden states h.
      keep     original genes in the pass, in order (new position = index in keep); n genes in the chromosome
      dels     the deleted genes; pend = the deleted element's last gene: distances count from it, as a unit's
               distance counts from U's end
      base_lg  baseline logit P(true family) per original gene; tfi each gene's row in the sparse decoder table;
               seen: the decoder saw its family; rgp: panRGP gene; entry: entry genes of Data.elements(g)
      lim      genes >= lim are not in the pass (a truncated sequence, influence()); default n
    Reads every gene from `up` genes before the first deleted gene to `win` genes after pend, and the entries of
    elements out to `ent_max` genes. Check (g), asserted here: every gene before the first deleted gene is
    bit-identical to the baseline. Returns the bins (inf_bins), the entries (index into the element table, delta)
    and the check counters."""
    lim = n if lim is None else lim
    orig_pos = np.full(n, -1, np.int64); orig_pos[keep] = np.arange(len(keep))
    d0 = int(np.min(dels)); pend = int(pend)
    lo = max(0, d0 - up); hi = min(lim - 1, pend + win)
    rows = np.arange(lo, hi + 1); rows = rows[orig_pos[rows] >= 0]
    ent_ok = (entry > pend) & (entry - pend <= ent_max) & (entry < lim)
    ent_ok[ent_ok] &= orig_pos[entry[ent_ok]] >= 0                # an entry deleted by this pass is not read
    far = np.unique(entry[ent_ok & (entry > hi)])
    rows = np.concatenate([rows, far])
    S, C = R.decode_true(h, orig_pos[rows], tfi[rows])
    dl = logit_true(S, C) - base_lg[rows]
    u = rows < d0
    if u.any():
        assert np.all(dl[u] == 0.0), f"check (g): a gene before the deletion changed (max |delta| {np.max(np.abs(dl[u]))})"
    mean, mabs, mr, nn, nr = inf_bins(rows - pend, dl, seen[rows], rgp[rows], win)
    val = np.full(n, np.nan); val[rows] = dl
    ei = np.flatnonzero(ent_ok)
    return dict(mean=mean, abs=mabs, rgp=mr, n=nn, n_rgp=nr, ent=ei, ent_delta=val[entry[ei]],
                n_rows=len(rows), n_up=int(u.sum()), up_max=float(np.max(np.abs(dl[u]))) if u.any() else 0.0)


def load_influence(path, checks=False):
    """The influence readouts of one genome file written by pg_knockout.py --influence (g*.npz) as two tables:
      passes    one row per deletion pass: pass index, kind (U / ctrl_cpl / ctrl_nat / check_b), deleted region id
                (-1 for a natural run), spot id, start, end, size (deleted genes), and the 60-bin profiles as arrays
                (mean, abs, rgp, n, n_rgp; bin k = genes 25k+1 .. 25k+25 after `end`)
      entries   one row per (pass, element entry read): the affected element (kind 0 = panRGP run by spot id,
                1 = coupled region by region id, its entry gene), distance = entry - pass end, delta (float16 ->
                float64), the baseline logit at that entry and whether the decoder saw its family
    The element-to-element matrix is `entries` with the deleted element (pass columns) joined in; the matched null of
    one (pass, entry) is the control passes of the same genome with size x0.5-2 of the pass's and distance to the
    same entry within +-35 % (match_controls, as for the units).
    checks=False leaves out the check_b passes (10 arbitrary genes deleted after the last readout row, possibly
    backbone genes: a self-check, not an element); checks=True keeps them."""
    import pandas as pd
    Z = np.load(path, allow_pickle=False)
    assert "inf_e_off" in Z, f"{path}: no influence readouts (run with --influence)"
    pk = Z["pass_kind"]; off = Z["inf_e_off"]
    dp = np.flatnonzero((pk != "base") & (pk != "check_a") & (checks | (pk != "check_b")))
    g = int(os.path.basename(path)[1:].split(".")[0])
    P = pd.DataFrame(dict(g=g, pass_i=dp, kind=pk[dp], del_id=Z["pass_id"][dp], del_spot=Z["pass_spot"][dp],
                          start=Z["pass_start"][dp], end=Z["pass_end"][dp], size=Z["pass_size"][dp]))
    for k in ("mean", "abs", "rgp"): P[k] = list(Z[f"inf_{k}"][dp].astype(np.float64))
    for k in ("n", "n_rgp"): P[k] = list(Z[f"inf_{k}"][dp].astype(np.int64))
    cnt = np.diff(off); pi = np.repeat(np.arange(len(cnt)), cnt); el = Z["inf_e_el"].astype(np.int64)
    if not checks:
        k_ = pk[pi] != "check_b"; pi = pi[k_]; el = el[k_]; de = Z["inf_e_delta"][k_]
    else:
        de = Z["inf_e_delta"]
    ent = Z["el_entry"][el].astype(np.int64)
    Ent = pd.DataFrame(dict(g=g, pass_i=pi, kind=pk[pi], del_id=Z["pass_id"][pi], del_spot=Z["pass_spot"][pi],
                            del_size=Z["pass_size"][pi], del_end=Z["pass_end"][pi],
                            el=el, el_kind=Z["el_kind"][el], el_id=Z["el_id"][el], el_entry=ent,
                            dist=ent - Z["pass_end"][pi], delta=de.astype(np.float64),
                            base_logit=Z["inf_base"][ent].astype(np.float64), seen=Z["inf_seen"][ent]))
    return P, Ent


def influence_matrix(path, kinds=("U",), win=INF_WIN, floor=1e-5):
    """Reference aggregation of one genome file: the element-to-element influence matrix of the passes of `kinds`
    against the whole-element null already used for the units. One row per (deleted element, affected element entry
    within `win` genes): delta, and the matched null = the control passes (ctrl_cpl / ctrl_nat) of the same genome that
    read the SAME entry, do not overlap the deleted element, and match it in size (x0.5-2) and distance to the entry
    (+-35 %), widened to x1/3-3 / +-60 % when fewer than 3 match (match_controls); robust z floored at `floor` (the
    fp32 floor, ~1e-5). A control pass put in `kinds` is scored against the OTHER controls (a pseudo-U)."""
    import pandas as pd
    P, E = load_influence(path)
    E = E[E.seen.values]
    ctl = E[E.kind.str.startswith("ctrl_")]
    pst = dict(zip(P.pass_i, P.start)); pen = dict(zip(P.pass_i, P.end)); psz = dict(zip(P.pass_i, P["size"]))
    by_el = {e: (d.pass_i.values, d.del_end.values, d.del_size.values, d.delta.values,
                 np.array([pst[i] for i in d.pass_i.values])) for e, d in ctl.groupby("el")}
    rows = []
    for r in E[E.kind.isin(kinds) & (E.dist >= 1) & (E.dist <= win)].itertuples(index=False):
        n_c, z = 0, np.nan; med = mad = np.nan
        if r.el in by_el:
            ci, ce, cs, cd, cst = by_el[r.el]
            ok = (ci != r.pass_i) & ~((cst <= pen[r.pass_i]) & (ce >= pst[r.pass_i]))
            idx, _w = match_controls(ce, cs, ok, psz[r.pass_i], r.dist, r.el_entry)
            n_c = len(idx)
            if n_c >= 3: z, med, mad = robust_z(r.delta, cd[idx], floor)
        rows.append(dict(g=r.g, pass_i=r.pass_i, kind=r.kind, del_id=r.del_id, del_spot=r.del_spot, del_size=r.del_size,
                         el_kind=r.el_kind, el_id=r.el_id, el_entry=r.el_entry, dist=r.dist, delta=r.delta,
                         n_ctrl=n_c, ctrl_med=med, ctrl_mad=mad, z=z))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- on demand, for serve_live
_CTX = {}


def context(dev=None, threads=2, frac=MEM_FRAC, sets=None, data=None, compact=None):
    """(Data, Decoder, Runner) for influence(), built once per process. dev None: cuda:0 when available, else the CPU
    with `threads` threads. Loads an fp32 copy of the model (the assay's precision; serve_live's copy is bf16). `frac`
    caps the GPU memory of the WHOLE calling process (torch.cuda.set_per_process_memory_fraction): serve_influence.py
    sets it (a process of its own), serve_live.py passes None (its process is not capped here). `data`: a Data already
    loaded (serve_live's /elements), else Data(sets=sets). `compact`: Decoder's (None: the compact decoder only when
    base_top64 is absent)."""
    if "ctx" not in _CTX:
        import torch
        dev = dev or ("cuda:0" if torch.cuda.is_available() else "cpu")
        D = data if data is not None else Data(sets=sets); dec = Decoder(D, compact=compact)
        R = Runner(dev=dev, frac=frac, threads=threads)
        _CTX["ctx"] = (D, dec, R); _CTX["base"] = collections.OrderedDict()
    return _CTX["ctx"]


def resolve_element(D, g, element):
    """element: a region id (int) or ('region', ri): the coupled element of region ri in genome g (must be present at
    its spot); ('spot', s): the panRGP run at spot s (the one with most non-backbone genes if the spot holds several);
    ('genes', [positions]): those genes. The deleted genes never include a backbone family."""
    if isinstance(element, (int, np.integer)): element = ("region", int(element))
    kind, x = element[0], element[1]
    a = int(D.OFF[g]); fam = D.FAM[a:int(D.OFF[g + 1])]
    if kind == "region":
        ri = int(x)
        assert D.HAS[ri, g] and D.PRES[ri, g], f"region {ri} is not present at its spot in genome {g}"
        e = D.EL[(ri, g)]; assert e["end"] >= e["start"], f"region {ri} wraps the origin in genome {g}"
        dels = D.members(ri, g)
        return dict(kind=1, id=ri, start=int(e["start"]), end=int(e["end"]), dels=dels, size=len(dels),
                    label=D.REG[ri]["label"], spot=D.REG[ri]["spot"] if D.REG[ri]["spot"] is not None else -1)
    if kind == "spot":
        E = D.elements(g); m = np.flatnonzero((E["kind"] == 0) & (E["id"] == int(x)) & (E["size"] > 0))
        assert len(m), f"no panRGP run at spot {x} in genome {g}"
        k = m[np.argmax(E["size"][m])]; s, t = int(E["start"][k]), int(E["end"][k])
        dels = [p for p in range(s, t + 1) if not D.BB[fam[p]]]
        return dict(kind=0, id=int(x), start=s, end=t, dels=dels, size=len(dels), label=f"spot {int(x)}", spot=int(x))
    if kind == "genes":
        dels = sorted(int(p) for p in x if not D.BB[fam[int(p)]])
        assert dels, "('genes', ...): every gene given is of a backbone family (never deleted)"
        return dict(kind=-1, id=-1, start=dels[0], end=dels[-1], dels=dels, size=len(dels), label="genes", spot=-1)
    raise ValueError(f"element {element!r}")


def influence_controls(D, g, el, n_ctrl=8):
    """n_ctrl whole accessory elements of genome g (control_pool: the null of the knockouts) not overlapping the
    element, size x0.5-2 of its (widened to x1/3-3, then any size, when fewer match), nearest to it first, so that
    their distances to the element's neighbours are close to the element's own"""
    pool = D.control_pool(g, sizes=[el["size"]])
    ok = [c for c in pool if not (c["start"] <= el["end"] and c["end"] >= el["start"])]
    for lo_, hi_ in ((MATCH[0], MATCH[1]), (WIDEN[0], WIDEN[1]), (0.0, np.inf)):
        cand = [c for c in ok if lo_ * el["size"] <= c["size"] <= hi_ * el["size"]]
        if len(cand) >= n_ctrl: break
    cand.sort(key=lambda c: (abs(c["end"] - el["end"]), c["start"]))
    return cand[:n_ctrl], (lo_, hi_)


def _nan2none(x):
    if isinstance(x, dict): return {k: _nan2none(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_nan2none(v) for v in x]
    if isinstance(x, np.ndarray): return _nan2none(x.tolist())
    if isinstance(x, (float, np.floating)): return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer): return int(x)
    return x


def _spec_scores(r, entry, end, win=INF_WIN, floor=1e-5):
    """Specificity of one pass at each element entry it read (1..win genes after the deleted element's end `end`):
    the entry's delta against the SAME pass's deltas at the other entries at a comparable distance (+-35 %, widened to
    +-60 % when < 3; entries sharing the gene excluded) -- the off-target logic of the verdicts. A deletion that moves
    everything downstream moves every entry alike and scores ~0 everywhere. Returns {element index: (dist, excess,
    z, n_ref)} (excess = delta - median of the others; z = excess / (1.4826 MAD), floored)."""
    ei = np.asarray(r["ent"], np.int64); dv = np.asarray(r["ent_delta"], float)
    gene = entry[ei]; d = gene - end
    m = np.isfinite(dv) & (d >= 1) & (d <= win)
    ug, first = np.unique(gene[m], return_index=True)             # one value per entry gene
    gi = ei[m][first]; gd = d[m][first]; gv = dv[m][first]
    out = {}
    for k in range(len(ug)):
        oth = np.array([], float)
        for tol in (OFF_TOL, WIDEN[2]):
            sel = (np.arange(len(ug)) != k) & (np.abs(gd - gd[k]) <= tol * gd[k])
            oth = gv[sel]
            if len(oth) >= 3: break
        if len(oth) < 3: continue
        med = float(np.median(oth)); mad = 1.4826 * float(np.median(np.abs(oth - med)))
        out_v = (int(gd[k]), float(gv[k] - med), float((gv[k] - med) / max(mad, floor)), int(len(oth)))
        for i in ei[m][gene[m] == ug[k]]: out[int(i)] = out_v
    return out


def influence(g, element, n_ctrl=8, truncate=None, ctx=None, z_floor=1e-5, progress=None):
    """The radius of influence of one accessory element of genome g, as the model sees it: baseline, deletion of the
    element (its member genes, never a backbone family), and n_ctrl size-matched whole-element control deletions of
    the same genome (influence_controls), one fp32 pass each at one padded length, read as in --all
    (influence_readout; check (g) asserted in every pass).
      truncate  run the passes only as far as the last gene read (the element's or a control's end + 2,400 genes);
                exact for a causal model up to the fp32 floor (~1e-5, pilot check f). Default: on the CPU only.
    Returns a JSON-ready dict:
      element, controls         what was deleted
      bins                      [lo, hi] genes after the element's last gene, 25 per bin, out to 1,500
      profile                   the element's mean delta, mean |delta|, mean delta over panRGP genes, genes per bin
      ctrl_mean, ctrl_abs       the same per control (n_ctrl x bins)
      z_abs, z_mean             robust z of the element's bin against its controls' same bin (median, 1.4826 MAD)
      radius                    upper edge (genes) of the last bin with z_abs >= 3; radius_contig: of the first run of
                                such bins from the element on; n_bins_z3: how many of the 60 bins (with 8 controls and a
                                robust z, a few bins reach 3 by chance: read the profile against ctrl_abs, not the radius)
      ctrl_abs_median           the controls' median |delta| profile (the reference curve: every deletion moves the
                                model's expectations downstream, decaying with distance)
      ctrl_abs_q                per bin, min / median / max of the controls' mean |delta|: the band to draw the profile in
      null                      the same summaries with each control in turn as the element, against the other controls
                                (n_bins_z3, radius, radius_contig, broad): what 8 controls give by chance. radius_p /
                                n_bins_z3_p = share of these pseudo-elements with at least the element's value (+1 smoothed)
      broad                     median over bins of the element's mean |delta| / the controls' median: how much more (or
                                less) than a typical deletion this one moves the model's expectations downstream
      neighbours                every element (Data.elements) whose entry lies 1-1,500 genes downstream: its delta
                                (logit P(its entry family)), and
                                  spec_excess, z_spec   SPECIFICITY: the neighbour's delta minus the median delta of the
                                                        element's own deletion at the other entries at a comparable distance
                                                        (+-35 %, widened to +-60 %; _spec_scores): does the deletion move THIS
                                                        neighbour more than it moves everything else at that distance?
                                  p_spec, q_spec        calibrated: z_spec against the same score computed for every entry of
                                                        every control deletion at a comparable distance (the controls scored
                                                        as if deleted elements); two-sided, +1 smoothed; q = BH over the
                                                        neighbours' distinct entry genes (an element and the panRGP run of
                                                        its spot often share one: one test, listed twice)
                                  z_same, z_dist        NOT specificity: the neighbour's delta against the controls' deltas at
                                                        the same entry (z_same) or at any entry at that distance (z_dist). An
                                                        element whose deletion moves everything downstream more than a typical
                                                        deletion flags every neighbour with z_same (yidK/yidJ, genome 1: 35 of
                                                        39 neighbours |z_same| > 3, the linked one included, while z_dist = -0.3)
                                and whether it is linked to the element in the sets file. Floors: 1e-5 (the fp32 floor).
                                Close to the element (< 100 genes) entries are few, so many scores are missing there.
      seconds                   per pass and in total
    progress: called as progress(done, total, what) before each pass (what: the pass about to run) and at the end,
    for a server's job status (serve_live.py); total = the passes of this call (baseline included when not cached)."""
    D, dec, R = ctx or context()
    tick = progress or (lambda *a: None)
    t0 = time.time()
    a, b = int(D.OFF[g]), int(D.OFF[g + 1]); n = b - a
    FAMg = D.FAM[a:b]; rgp = D.RGP[a:b]
    E = D.elements(g); entry = E["entry"].astype(np.int64)
    el = resolve_element(D, g, element)
    ctrls, widen = influence_controls(D, g, el, n_ctrl)
    dev_cuda = str(R.dev).startswith("cuda")
    if truncate is None: truncate = not dev_cuda
    lim = n if not truncate else min(n, max([el["end"]] + [c["end"] for c in ctrls]) + INF_ENT + 1)
    hh = dec.half_for(g)
    key = (g, lim, hh)
    if key in _CTX.get("base", {}):
        fams, tfi, seen, table, base_lg, t_base = _CTX["base"][key]
    else:
        fams = np.unique(FAMg); tfi = np.searchsorted(fams, FAMg)
        table = dec.sparse_table(fams, hh); seen = table[3][tfi] > 0
        base_lg = None
    total = 1 + len(ctrls) + (base_lg is None)
    R.set_sparse(*table)
    R.set_genome(D.EMB[D.PID[a:a + lim]], lim + 1)
    if base_lg is None:
        tick(0, total, "baseline pass (the genome as it is)")
        R.sync(); t = time.time()
        h = R.hidden(np.arange(lim)); S, C = R.decode_true(h, np.arange(lim), tfi[:lim])
        base_lg = np.full(n, np.nan); base_lg[:lim] = logit_true(S, C)
        R.sync(); t_base = time.time() - t
        _CTX.setdefault("base", collections.OrderedDict())[key] = (fams, tfi, seen, table, base_lg, t_base)
        while len(_CTX["base"]) > 4: _CTX["base"].popitem(last=False)
    res = []; secs = []
    for k_, p in enumerate([el] + ctrls):
        tick(total - 1 - len(ctrls) + k_, total, "deletion of the element" if k_ == 0 else f"control deletion {k_} of {len(ctrls)}")
        R.sync(); t = time.time()
        keep = np.setdiff1d(np.arange(lim), np.asarray(p["dels"], np.int64))
        h = R.hidden(keep)
        r = influence_readout(R, h, keep, n, p["dels"], p["end"], base_lg, tfi, seen, rgp, entry, lim=lim)
        R.sync(); secs.append(time.time() - t); res.append(r)
    nb = INF_WIN // INF_BIN
    r0 = res[0]; cr = res[1:]
    cm = np.array([r["mean"] for r in cr]).reshape(-1, nb); ca = np.array([r["abs"] for r in cr]).reshape(-1, nb)
    z_abs = np.full(nb, np.nan); z_mean = np.full(nb, np.nan)
    for k in range(nb):
        x = ca[:, k][np.isfinite(ca[:, k])]
        if np.isfinite(r0["abs"][k]) and len(x) >= 3: z_abs[k] = robust_z(r0["abs"][k], x, z_floor)[0]
        x = cm[:, k][np.isfinite(cm[:, k])]
        if np.isfinite(r0["mean"][k]) and len(x) >= 3: z_mean[k] = robust_z(r0["mean"][k], x, z_floor)[0]
    def radii(z):
        sig_ = np.flatnonzero(z >= 3); k_ = 0
        while k_ < nb and np.isfinite(z[k_]) and z[k_] >= 3: k_ += 1
        return (int((sig_[-1] + 1) * INF_BIN) if len(sig_) else 0), k_ * INF_BIN, int(len(sig_))
    radius, radius_contig, n_z3 = radii(z_abs)
    sig = np.arange(n_z3)
    # the same with each control as the element, against the others: what the summaries give by chance
    cmed = np.nanmedian(ca, 0) if len(ca) else np.full(nb, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        broad = float(np.nanmedian(np.asarray(r0["abs"], float) / cmed))
    null = dict(radius=[], radius_contig=[], n_bins_z3=[], broad=[])
    for c in range(len(ca)):
        oth = np.delete(np.arange(len(ca)), c); zc = np.full(nb, np.nan)
        for k_ in range(nb):
            x = ca[oth, k_][np.isfinite(ca[oth, k_])]
            if np.isfinite(ca[c, k_]) and len(x) >= 3: zc[k_] = robust_z(ca[c, k_], x, z_floor)[0]
        rr = radii(zc)
        null["radius"].append(rr[0]); null["radius_contig"].append(rr[1]); null["n_bins_z3"].append(rr[2])
        with np.errstate(invalid="ignore", divide="ignore"):
            null["broad"].append(float(np.nanmedian(ca[c] / np.nanmedian(ca[oth], 0))))
    pshare = lambda v, xs: float((1 + sum(x >= v for x in xs)) / (1 + len(xs))) if xs else np.nan
    # specificity at each neighbour, and its null: every control deletion scored the same way
    spec0 = _spec_scores(r0, entry, el["end"])
    spec_null = [v for c, r in zip(ctrls, cr) for v in _spec_scores(r, entry, c["end"]).values()]
    sn_d = np.array([v[0] for v in spec_null], float); sn_z = np.array([v[2] for v in spec_null], float)
    # the neighbours: the element's entries and each control's, by element index
    ctl_ent = [dict(zip(r["ent"].tolist(), r["ent_delta"].tolist())) for r in cr]
    linked = D.links_of(el["id"]) if el["kind"] == 1 else set()
    nbrs = []
    for i, dv in zip(r0["ent"].tolist(), r0["ent_delta"].tolist()):
        d = int(entry[i] - el["end"])
        if d > INF_WIN: continue
        same = []
        for tol in (OFF_TOL, WIDEN[2]):
            same = [ce[i] for c, ce in zip(ctrls, ctl_ent)
                    if i in ce and np.isfinite(ce[i]) and abs((entry[i] - c["end"]) - d) <= tol * d]
            if len(same) >= 3: break
        pooled = [v for c, ce in zip(ctrls, ctl_ent) for j, v in ce.items()
                  if np.isfinite(v) and abs((entry[j] - c["end"]) - d) <= OFF_TOL * d]
        zs = robust_z(dv, same, z_floor)[0] if len(same) >= 3 else np.nan
        zd = robust_z(dv, pooled, z_floor)[0] if len(pooled) >= 3 else np.nan
        sp_ = spec0.get(int(i)); ex_ = zsp = psp = np.nan; nref = nnull = 0
        if sp_ is not None:
            ex_, zsp, nref = sp_[1], sp_[2], sp_[3]
            for tol in (OFF_TOL, WIDEN[2], np.inf):
                mm = np.abs(sn_d - d) <= tol * d
                if mm.sum() >= 20 or tol == np.inf: break
            nnull = int(mm.sum())
            if nnull: psp = float((1 + np.sum(np.abs(sn_z[mm]) >= abs(zsp))) / (1 + nnull))
        kd, idd = int(E["kind"][i]), int(E["id"][i])
        nbrs.append(dict(kind=("region" if kd == 1 else "spot"), id=idd, label=D.element_label(kd, idd)[:60],
                         entry=int(entry[i]), start=int(E["start"][i]), end=int(E["end"][i]), dist=d,
                         family=D.FN[int(E["fam"][i])], seen=bool(seen[int(entry[i])]),
                         base_p=float(1.0 / (1.0 + np.exp(-base_lg[int(entry[i])]))), delta=dv,
                         spec_excess=ex_, z_spec=zsp, n_spec_ref=nref, p_spec=psp, n_spec_null=nnull,
                         z_same=zs, n_same=len(same), z_dist=zd, n_dist=len(pooled),
                         linked=bool(kd == 1 and idd in linked)))
    # BH over the distinct entry genes: an element and its spot's panRGP run often share one (the same readout, the same
    # p), and counting it twice would inflate the number of tests
    pe = {}
    for x in nbrs: pe.setdefault(x["entry"], x["p_spec"])
    qe = dict(zip(pe, bh(list(pe.values()))))
    for x in nbrs: x["q_spec"] = qe[x["entry"]]
    tick(total, total, "done")
    out = dict(g=int(g), acc=D.ACC[g], n_genes=n, decoder_half=hh, device=str(R.dev), truncated_at=int(lim),
               element={k_: el[k_] for k_ in ("kind", "id", "label", "start", "end", "size", "spot")},
               controls=[dict(arm=c["arm"], ri=int(c["ri"]), spot=int(c["spot"]), start=int(c["start"]), end=int(c["end"]),
                              size=int(c["size"])) for c in ctrls], control_size_window=list(widen),
               bins=[[k_ * INF_BIN + 1, (k_ + 1) * INF_BIN] for k_ in range(nb)],
               profile=dict(mean=r0["mean"], abs=r0["abs"], rgp=r0["rgp"], n=r0["n"], n_rgp=r0["n_rgp"]),
               ctrl_mean=cm, ctrl_abs=ca, z_abs=z_abs, z_mean=z_mean, radius=radius, radius_contig=radius_contig,
               n_bins_z3=int(len(sig)), ctrl_abs_median=cmed,
               ctrl_abs_q=(np.stack([np.nanmin(ca, 0), cmed, np.nanmax(ca, 0)]) if len(ca) else np.full((3, nb), np.nan)),
               broad=broad, null=null, radius_p=pshare(radius, null["radius"]), n_bins_z3_p=pshare(int(len(sig)), null["n_bins_z3"]),
               broad_p=pshare(broad, null["broad"]),
               spec_null=dict(n=len(spec_null), abs_z_q=[float(x) for x in np.nanpercentile(np.abs(sn_z), [50, 90, 99])]
                              if len(sn_z) else None),
               neighbours=nbrs,
               checks=dict(upstream_rows_exact=int(sum(r["n_up"] for r in res)), rows_read=int(sum(r["n_rows"] for r in res))),
               seconds=dict(baseline=t_base, passes=secs, total=time.time() - t0))
    return _nan2none(out)


# ============================================================================ statistics
def bh(p):
    p = np.asarray(p, float); q = np.full(len(p), np.nan); ok = np.isfinite(p)
    if ok.sum():
        pp = p[ok]; o = np.argsort(pp); r = pp[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]; qq = np.empty_like(pp); qq[o] = np.minimum(r, 1); q[ok] = qq
    return q


def rank_share_null(n_ctrl, thr=0.95):
    """expected share of units with rank percentile >= thr among n controls, under exchangeability"""
    n = np.asarray(n_ctrl, float)
    return float(np.mean([(k - np.ceil(thr * k) + 1) / (k + 1) for k in n])) if len(n) else np.nan


def match_controls(pool_end, pool_size, ok, size, dist, row, widen=True):
    """indices of the controls matched to a deletion of `size` genes lying `dist` genes before `row`.
    `ok` already excludes overlaps with U and D and controls linked to either. Returns (idx, widened).
    With MATCH_LOG = F, a third tier when the widened one still has < 3: distance within x1/F..xF of the unit's."""
    d = row - pool_end
    for (lo, hi, tol), w in ((MATCH, False), (WIDEN, True)):
        m = ok & (pool_end < row) & (pool_size >= lo * size) & (pool_size <= hi * size) & (np.abs(d - dist) <= tol * dist)
        idx = np.flatnonzero(m)
        if len(idx) >= 3 or not widen: return idx, w
    if MATCH_LOG is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            m = ok & (pool_end < row) & (pool_size >= WIDEN[0] * size) & (pool_size <= WIDEN[1] * size) & \
                (np.abs(np.log(np.maximum(d, 1) / dist)) <= np.log(MATCH_LOG))
        idx = np.flatnonzero(m)
    return idx, True


def robust_z(delta, ctrl, floor):
    """(delta - median) / (1.4826 x MAD) of the control deltas, floored. The pilot used mean/sd of 3-5 skewed
    control deltas, which is what made the tails miscalibrated (inference review finding 2)."""
    if len(ctrl) < 3: return np.nan, np.nan, np.nan
    c = np.asarray(ctrl, float); med = float(np.median(c))
    mad = 1.4826 * float(np.median(np.abs(c - med)))
    return (delta - med) / max(mad, floor), med, mad


def rank_pct(delta, ctrl, e_sign):
    """percentile of delta among its controls in the expected direction (0.5 = a tie)"""
    c = np.asarray(ctrl, float) * e_sign; x = delta * e_sign
    return float((np.sum(c < x) + 0.5 * np.sum(c == x)) / len(c)) if len(c) else np.nan


def mde_wilcoxon(sd, n, alpha=0.05, power=0.8):
    """smallest mean shift (same units as sd) a one-sided signed-rank test on n lineage means detects with
    the given power (normal approximation, Wilcoxon ARE 3/pi)"""
    from scipy.stats import norm
    if n < 2 or not np.isfinite(sd): return np.nan
    return float((norm.ppf(1 - alpha) + norm.ppf(power)) * sd / np.sqrt(n) * np.sqrt(np.pi / 3))
