"""The reading of a window written by an LLM, for the Findings tab of the PanGramGraph page (serve_live.py /reading).

The page writes a rule-based reading of the window shown (autoReading in page_template.html). Here an LLM writes it
in prose, in the style of the hand-written reading of the dnaA window, and every number and gene name of its text is
checked against the window's own data before the page shows it.

    the browser sends only the window's id: dnaA, or the anchor family of a region window (pgb_region.build; a backbone
    family). The server builds the facts itself from its own data (facts(D)): the columns, their stretches (spine,
    fork, variable, the same rules as the page's autoReading), the stretch most and least in regions of plasticity,
    the genes and their products, the model's accuracy and hesitation, all as display-ready numbers: 2,000-2,900
    tokens, after ~1,000 of instructions and style example. No text of the client reaches the LLM.
    tidy(text) puts the model's slips of form right without a new call: "92-99%" -> "92 to 99%", "RGP 5%" -> "5% in a
    region of plasticity", "share 90%" -> "in 90% of chromosomes", a row of more than 6 genes -> `first` to `last`.
    check(text, F): every number (and gene name, number word, identifier with a digit) must be one the facts display
    (a number rounded further than the facts, 0.47 bits -> 0.5, is allowed, not 0.47 -> 0 or 99.8% -> 100%), and one
    of the facts lines of the genes or columns its sentence names (or of a line about the whole window); a gene next to
    a column must sit at that column; the inside/outside comparison must keep its direction; "most hesitant", "fewest
    named", "most/least plastic" only where the facts say so; no "lack", "absent", unstated superlatives, "least
    hesitation". On a failure the LLM is asked once again with the offending items named, then the page keeps its
    rule-based reading (status "fallback"). The check verifies numbers and names, not the wording or the biology.
    cache: per window, in memory and in <READING_CACHE>/<window>.json, also read from pgb/readings/ (the readings the
    Space carries), keyed by a hash of the facts, the prompt and the model: new data or a new prompt make a new
    reading. Limits on new generations: per client (X-Forwarded-For only behind a trusted proxy), per client per day,
    per day for the whole server (count and dollars), one generation per window at a time, two at once (others wait
    their turn in a short queue), only for requests from the page's own site (Sec-Fetch-Site / Origin).

Model: deepseek-ai/DeepSeek-V4.1-Flash through Hugging Face Inference Providers (router.huggingface.co, OpenAI
compatible), no reasoning (chat_template_kwargs thinking=false), at most 800 tokens out (median 540). Token: HF_TOKEN (the
Space's secret), else the token of `huggingface-cli login`; used only in the request's header (never followed
through a redirect), never logged, never written, never sent to the page; the router's error text is not passed on.

    READING=off                disables new readings (cached ones are still served; /health says so)
    READING_MODEL, READING_PROVIDERS=deepinfra,novita   the model, and the providers tried in that order
    READING_PER_IP=6 READING_PER_IP_S=600 READING_PER_IP_DAY=30   new readings per client per 600 s, and per day
    READING_DAILY_CAP=150      new readings per UTC day, the whole server;  READING_DAILY_USD=0.30  and dollars per day
                               (every call counted, a call without usage or cut by a timeout at its worst case)
    READING_TIMEOUT=22         seconds per LLM call (then the next provider);  READING_QUEUE=6  generations waiting
    READING_CACHE=pgb/readings where new readings and the day's counter are written (e.g. /data/readings on a Space
                               with persistent storage); pgb/readings/ is read too
    TRUST_PROXY=1              read the client from X-Forwarded-For (always on a Space: SPACE_ID)

    python3 pg_reading.py facts dnaA | <window.json>          the facts of a window (a region: pgb_region.build output)
    python3 pg_reading.py check dnaA | <window.json> <text.md>   the checker on a text, against that window's facts
    python3 pg_reading.py pregen [--chain] [--dry] [dnaA] [anchor ...]   write readings ahead (--chain: the windows of
        the page's Next and Previous buttons from the dnaA window on), into READING_CACHE, without the server
"""
import os, re, json, time, html, hashlib, threading, collections, ipaddress, urllib.request, urllib.error, urllib.parse, datetime
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN, InvalidOperation


def _env(k, d, f=float):
    try: return f(os.environ.get(k, d))
    except ValueError: return f(d)


MODEL = os.environ.get("READING_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
PROVIDERS = [p.strip() for p in os.environ.get("READING_PROVIDERS", "deepinfra,novita").split(",") if p.strip()]
PRICES = {"deepinfra": (0.2, 0.6), "novita": (0.3, 1.2), "baseten": (0.3, 1.2)}   # $ per million tokens in, out (router, 2026-09)
URL = "https://router.huggingface.co/v1/chat/completions"
MAX_TOKENS = 800                 # median 540 out; 700 cut 2 of 130 texts
TEMPERATURE = 0.3
TIMEOUT = _env("READING_TIMEOUT", "22")
BUDGET = 70.0                    # seconds for a whole generation (two attempts, a provider fallback)
PER_IP = _env("READING_PER_IP", "6", int)
PER_IP_S = _env("READING_PER_IP_S", "600")
PER_IP_DAY = _env("READING_PER_IP_DAY", "30", int)
DAILY_CAP = _env("READING_DAILY_CAP", "150", int)
DAILY_USD = _env("READING_DAILY_USD", "0.30")
BUNDLED = "pgb/readings"         # the readings shipped with the code (the Space serves them without a call)
CACHE_DIR = os.environ.get("READING_CACHE") or BUNDLED
FAIL_TTL = 6 * 3600              # a window whose text failed the checker twice is tried again after this long
AUTH_RETRY = 600                 # a token refused by the router: tried again after this long
MAX_PARALLEL = 2                 # LLM generations at once
QUEUE_MAX = _env("READING_QUEUE", "6", int)     # new generations waiting for one of those slots
SLOT_WAIT = 30.0                 # seconds a new generation waits for a slot
WAITERS = 4                      # requests waiting for the generation of the same window
PROMPT_VERSION = "r7"
DNAA = "dnaA"                    # the id of the page's home window (pgb/graph_pgb.json, all 2,002 genomes)
SPACE = bool(os.environ.get("SPACE_ID"))
TRUST_PROXY = SPACE or os.environ.get("TRUST_PROXY") == "1"


class ReadingError(Exception):
    """an answer other than a reading: HTTP status, a machine-readable code (the page words it), retry_s"""
    status, code = 500, "error"
    def __init__(self, msg, retry_s=None, code=None):
        super().__init__(msg); self.retry_s = retry_s; self.code = code or type(self).code
    def body(self):
        b = dict(error=str(self), code=self.code)
        if self.retry_s: b["retry_s"] = int(self.retry_s)
        if isinstance(self, Unavailable): b["available"] = False
        return b
class Unavailable(ReadingError): status, code = 503, "unavailable"   # off, no token, token refused, no credit, the day's cap
class Limited(ReadingError): status, code = 429, "limit"             # this client's new readings
class Busy(ReadingError): status, code = 503, "busy"                 # other readings are being written: retry_s
class Upstream(ReadingError): status, code = 502, "upstream"         # the LLM could not be reached
class Refused(ReadingError): status, code = 403, "cross_site"        # a new reading asked for by another site's page


# ---------------------------------------------------------------- numbers as the facts display them
def D_(x): return Decimal(str(x))
def fmt_pct(v):
    """a share (0-1) as a percent: whole numbers, one decimal when that would show 0% or 100% for a share that is not"""
    x = D_(v) * 100
    q = x.quantize(Decimal("1"), ROUND_HALF_UP)
    if (q == 100 and x < 100) or (q == 0 and x > 0): q = x.quantize(Decimal("0.1"), ROUND_HALF_UP)
    if q == 100 and x < 100: q = Decimal("99.9")
    if q == 0 and x > 0: q = Decimal("0.1")
    return f"{q}%"
def fmt_bits(v): return f"{D_(v).quantize(Decimal('0.01'), ROUND_HALF_UP)}"
def fmt_fam(v): return f"{D_(v).quantize(Decimal('0.1'), ROUND_HALF_UP)}"
def fmt_int(v): return f"{int(v):,}"


class Facts:
    """the facts text of a window, every number it displays by kind (pct, bits, fam, int), and for the checker: each
    line's columns and stretch, the columns of each gene, the columns stated most hesitant and fewest named, the
    stretches most and least in regions of plasticity, the inside/outside comparison"""
    def __init__(self):
        self.lines = []; self.info = []; self.nums = collections.defaultdict(set); self.genes = set(); self.frags = set()
        self.at = collections.defaultdict(set); self.atp = collections.defaultdict(set); self.hes_cols = set(); self.low_cols = set(); self.plastic = {}
        self.comp = None; self.unit = "genomes"; self.label = ""
    def reg(self, s, kind):
        v = Decimal(s.rstrip("%").replace(",", "")); self.nums[kind].add(v); return s
    def pct(self, v): return self.reg(fmt_pct(v), "pct")
    def bits(self, v): return self.reg(fmt_bits(v), "bits")
    def fam(self, v): return self.reg(fmt_fam(v), "fam")
    def int(self, v): return self.reg(fmt_int(v), "int")
    def name(self, n, short=40, k=None):
        """a family as the facts name it: its gene name, else its product in quotes (k: its column, for the checker)"""
        if n.get("named"):
            self.genes.add(n["label"])
            if k is not None: self.at[n["label"]].add(k)
            return n["label"]
        lab = n.get("product") or n.get("label") or "unnamed family"
        if re.fullmatch(r"[A-Z0-9]+_RS\d+", lab): lab = "unnamed family"
        if k is not None: self.atp[lab.lower()].add(k); self.atp[self.text(lab, short).lower()].add(k)
        return '"' + self.text(lab, short) + '"'
    def text(self, s, short=48):
        """a product: shortened; its words followed by a number ("family 4") are names, not numbers, in a text"""
        s = s if len(s) <= short else s[:short - 1].rstrip() + "…"
        for m in re.finditer(r"([A-Za-z][\w'-]*)\s+(\d+)\b(?!\.\d)", s): self.frags.add(f"{m.group(1)} {m.group(2)}".lower())
        return s
    def add(self, s, cols=(), head=None, glob=False):
        """a line; cols: the columns it is about; head: the index of its stretch's line; glob: about the whole window"""
        self.lines.append(s); self.info.append(dict(cols=set(cols), head=head, glob=glob)); return len(self.lines) - 1
    def txt(self): return "\n".join(self.lines)


def columns(D):
    """the window's columns as the page's autoReading sees them: the most carried family (at its most common
    column), its share of the genomes, the share of the second one; spine >= 90 %, fork: a second one >= 10 %"""
    W, NG = D["meta"]["window"], D["meta"]["n_genomes"]
    pl = {r["locus"]: r for r in D["per_locus"]}; by = collections.defaultdict(list)
    for n in D["nodes"]: by[n["locus"]].append(n)
    cols = []
    for k in range(W):
        here = sorted(by.get(k, []), key=lambda n: -n["n"])
        if not here: continue
        top = here[0]; tot = sum(n["n"] for n in here) or 1
        s1 = top["n"] / NG; s2 = here[1]["n"] / NG if len(here) > 1 else 0
        r = pl.get(k)
        rgp = r["rgp"] if r and r.get("rgp") is not None else sum(n["n"] * n["rgp"] for n in here) / tot
        cols.append(dict(k=k, top=top, here=here, s1=s1, kind="spine" if s1 >= 0.9 else "fork" if s2 >= 0.1 else "variable",
                         alts=[n for n in here[1:] if n["n"] / NG >= 0.1], rgp=rgp, r=r))
    runs = []
    for c in cols:
        if runs and runs[-1]["kind"] == c["kind"] and c["k"] == runs[-1]["b"] + 1: runs[-1]["b"] = c["k"]; runs[-1]["cols"].append(c)
        else: runs.append(dict(kind=c["kind"], a=c["k"], b=c["k"], cols=[c]))
    return cols, runs


PART = {"persistent": "P", "shell": "S", "cloud": "C"}
GENERIC = set("""protein family domain domain-containing containing putative probable transcriptional regulator subunit hypothetical
uncharacterized dna-binding system component type like bifunctional membrane inner outer protein-containing lipoprotein
periplasmic cytoplasmic small large chain""".split())
def words(p):
    """what a product says: its words that are neither generic nor a protein name"""
    return {w.lower() for w in re.findall(r"[A-Za-z][\w-]{3,}", p or "") if w.lower() not in GENERIC and not re.fullmatch(r"[A-Z][a-z]{2}[A-Z]\w*", w)}
def informative(p):
    """a product that says more than its gene name: a long word, or one with capitals or digits"""
    if not p or p.lower() == "hypothetical protein": return False
    ws = [w.lower() for w in re.findall(r"[A-Za-z][\w/-]{3,}", p)]
    return any((w not in GENERIC and not re.fullmatch(r"[a-z]{3}[a-z0-9]*", w)) or len(w) > 7 for w in ws if w not in GENERIC)


def facts(D, wid):
    """the facts of one window (graph data: pgb/graph_pgb.json for dnaA, pgb_region.build for the others)"""
    F = Facts(); m = D["meta"]; W = m["window"]; NG = m["n_genomes"]; home = wid == DNAA
    cols, runs = columns(D); unit = "genomes" if home else "chromosomes"; F.unit = unit
    if home:
        full = m.get("full_windows")
        F.add(f"WINDOW: the dnaA window: {F.int(W)} columns from dnaA (column {F.int(0)}) in its direction, in all {F.int(NG)} genomes "
              f"of the PanGBank E. coli pangenome; {F.int(m['calls'])} calls of the model, each on a genome's real proteins."
              + (f" {F.int(full)} of the {F.int(NG)} genomes have all {F.int(W)} columns on one contig; in the others the window "
                 f"stops earlier, at a contig end." if full else ""), glob=True)
    else:
        a = m["anchor_label"]
        if re.fullmatch(r"[a-z]{3}[A-Z]?\w*", a or ""): F.genes.add(a); F.at[a].add(0)
        pos = m.get("anchor_pos")
        where = f"; about genes {F.int(pos)} to {F.int(pos + W - 1)} from dnaA" if pos is not None else ""
        F.add(f"WINDOW: the {a} window: {F.int(W)} columns, column {F.int(0)} is the anchor {a}, then the next {F.int(W - 1)} genes "
              f"in dnaA's direction{where}. Read in {F.int(NG)} of the {F.int(m['genomes_total'])} complete chromosomes (those "
              f"carrying {a} once); {F.int(m['calls'])} calls of the model, each on a chromosome's real proteins.", glob=True)
    one = unit[:-1]
    F.add(f"TERMS: share: {unit} carrying the family in the window. named: calls where the model names the gene at that column "
          f"from the genes before it. bits: the model's entropy there, its hesitation (the more bits, the more hesitant). "
          f"perplexity (whole window only): how surprised the model is by the real gene. RGP N%: N% of the calls there are on a "
          f"gene inside a region of genomic plasticity (panRGP). P/S/C: persistent, shell, cloud (PPanGGOLiN). Spine column: "
          f"one family in {F.pct(0.9)} of {unit} or more; fork: a second family in {F.pct(0.1)} or more; variable: neither. A "
          f"persistent family is in nearly every {one}: in a fork or variable column its lower share means it lies beyond the "
          f"window's end in the other {unit} (genes inserted upstream push it out{', or a contig ends' if home else ''}); never "
          f"call it rare or say they lack it.", glob=True)
    pl = D["per_locus"]; n_all = sum(r["n"] for r in pl) or 1
    ent = sum(r["entropy"] * r["n"] for r in pl) / n_all
    part, rg = D.get("by_partition") or {}, D.get("by_rgp") or {}
    t = lambda r: f"{F.int(r['n'])} calls, named {F.pct(r['top1_dec'])}, {F.bits(r['entropy'])} bits" if r else "no calls"
    nk = collections.Counter(c["kind"] for c in cols)
    F.add(f"WHOLE WINDOW: the model names {F.pct(m['top1_dec'])} of genes, {F.bits(ent)} bits on average (perplexity "
          f"{F.fam(m['ppl'])} families). By partition of the gene: persistent {t(part.get('persistent'))}; shell "
          f"{t(part.get('shell'))}; cloud {t(part.get('cloud'))}. Outside an RGP {t(rg.get('outside an RGP'))}; inside an "
          f"RGP {t(rg.get('inside an RGP'))}. Columns: {F.int(nk['spine'])} spine, {F.int(nk['fork'])} fork, "
          f"{F.int(nk['variable'])} variable.", glob=True)
    RI, RO = rg.get("inside an RGP"), rg.get("outside an RGP")
    if RI and RO and RI["n"] and RO["n"]:
        e_i, e_o, a_i, a_o = RI["entropy"], RO["entropy"], RI["top1_dec"], RO["top1_dec"]
        h = "hesitates more" if e_i > 1.2 * e_o + 0.02 else "hesitates less" if e_i < 0.8 * e_o - 0.02 else "hesitates about as much"
        n_ = "names fewer genes" if a_i < a_o - 0.03 else "names more genes" if a_i > a_o + 0.03 else "names about as many genes"
        F.comp = dict(h=h, n=n_, ei=F.bits(e_i), eo=F.bits(e_o), ai=F.pct(a_i), ao=F.pct(a_o))
        F.add(f"COMPARISON (computed, over calls): inside an RGP the model {h} than outside ({F.comp['ei']} against {F.comp['eo']} bits) "
              f"and {n_} ({F.comp['ai']} against {F.comp['ao']}); {F.int(RI['n'])} of the {F.int(RI['n'] + RO['n'])} calls are inside an RGP.", glob=True)
    by_k = {c["k"]: c for c in cols}
    def mean(rs, key):
        n = sum(r["n"] for r in rs); return sum(r[key] * r["n"] for r in rs) / n if n else None
    hot = [r for r in pl if r["locus"] in by_k and by_k[r["locus"]]["rgp"] >= 0.5]
    cold = [r for r in pl if r["locus"] in by_k and by_k[r["locus"]]["rgp"] < 0.5]
    top = [r for r in sorted(pl, key=lambda r: -r["entropy"])[:3] if r["locus"] in by_k]
    if pl: F.low_cols.add(min(pl, key=lambda r: r["top1_dec"])["locus"])       # the window's fewest named: its value is in the facts
    F.hes_cols |= {r["locus"] for r in top}
    hs = "; ".join(f"column {F.int(r['locus'])} ({F.name(by_k[r['locus']]['top'], k=r['locus'])}) {F.bits(r['entropy'])} bits, "
                   f"RGP {F.pct(by_k[r['locus']]['rgp'])}" for r in top)
    F.add(f"HESITATION: the most hesitant columns of the window: {hs}. "
          + (f"The {F.int(len(hot))} columns where most calls are in an RGP: named {F.pct(mean(hot, 'top1_dec'))}, {F.bits(mean(hot, 'entropy'))} "
             f"bits on average; the other {F.int(len(cold))} columns: named {F.pct(mean(cold, 'top1_dec'))}, {F.bits(mean(cold, 'entropy'))} bits."
             if hot and cold else "No column has most of its calls in an RGP." if not hot else "Every column has most of its calls in an RGP."),
          glob=True)

    def span(r): return f"column {F.int(r['a'])}" if r["a"] == r["b"] else f"columns {F.int(r['a'])}-{F.int(r['b'])}"
    def spann(r): return span(r) + (f" ({F.int(r['b'] - r['a'] + 1)} columns)" if r["b"] > r["a"] else "")
    def rng(vals, f):
        lo, hi = f(min(vals)), f(max(vals)); return lo if lo == hi else f"{lo.rstrip('%')}-{hi}"
    def model(cs, listed=True):
        """the model on a stretch: named (range, mean), where it names fewest, where it hesitates most (or that it hardly does)"""
        w = [c for c in cs if c["r"]]
        if not w: return "no calls"
        n = sum(c["r"]["n"] for c in w); acc = sum(c["r"]["top1_dec"] * c["r"]["n"] for c in w) / n
        pk = max(w, key=lambda c: c["r"]["entropy"]); lo = min(w, key=lambda c: c["r"]["top1_dec"])
        span_ = rng([c['r']['top1_dec'] for c in w], F.pct)
        s = f"named {span_}" + (f" (mean {F.pct(acc)})" if "-" in span_ else "")
        who = (lambda c: f" ({F.name(c['top'], k=c['k'])})") if listed else (lambda c: "")
        if len(w) == 1: return s + f"; {F.bits(pk['r']['entropy'])} bits"
        hb = pk["r"]["entropy"]; calm = hb < 0.2
        if "-" in span_:
            F.low_cols.add(lo["k"])
            if lo is pk and not calm:
                F.hes_cols.add(pk["k"])
                return s + f", fewest named and most hesitant at column {F.int(lo['k'])}{who(lo)}, named {F.pct(lo['r']['top1_dec'])}, {F.bits(hb)} bits"
            s += f", fewest named at column {F.int(lo['k'])}{who(lo)}, named {F.pct(lo['r']['top1_dec'])}"
        F.hes_cols.add(pk["k"])
        if calm: return s + f"; no column of this stretch hesitates more than {F.bits(hb)} bits (column {F.int(pk['k'])}{who(pk)})"
        return s + f"; most hesitant at column {F.int(pk['k'])}{who(pk)}, {F.bits(hb)} bits"
    def parts(cs):
        c = collections.Counter(PART.get(x["top"]["partition"], "?") for x in cs)
        return ("all " + next(iter(c))) if len(c) == 1 else ", ".join(f"{F.int(v)} {k}" for k, v in c.most_common())
    group_prod = {}                   # one product for each run of named genes sharing a prefix (an operon, often): its most specific
    grp = []
    for c in cols + [None]:
        n = c["top"] if c else None
        pre = n["label"][:3] if n and n.get("named") and not n["label"].startswith("y") and len(n["label"]) >= 4 else None
        if grp and pre and pre == grp[-1][1]["label"][:3] and c["k"] == grp[-1][0] + 1: grp.append((c["k"], n)); continue
        if len(grp) >= 2:                                                   # the product whose words the group shares most
            ws = [words(x.get("product", "")) for _, x in grp]
            sc = [sum(1 + sum(w in o for o in ws) - 1 for w in wi) for wi in ws]
            j = max(range(len(grp)), key=lambda j: (sc[j], -len(grp[j][1].get("product", ""))))
            if sc[j]: group_prod[id(grp[j][1])] = grp[j][1]["product"]
        grp = [(c["k"], n)] if pre else []
    def prod(n):
        """a y-gene's product (its name says little), and one product per group of genes sharing a prefix; an unnamed
        family is named by its product already"""
        p = (n.get("product") or "").strip()
        if not n.get("named") or not p or p.lower() == "hypothetical protein": return ""
        if n["label"].startswith("y") or id(n) in group_prod: return f' "{F.text(p, 36)}"'
        return ""
    def fam_s(n, k, main=None):
        same = " (another family)" if main is not None and n.get("named") and main.get("named") and n["label"] == main["label"] else ""
        return f"{F.name(n, k=k)}{same} {PART.get(n['partition'], '?')} {F.pct(n['n'] / NG)}"

    E = {(e["s"], e["t"]): e["n"] for e in D.get("edges", [])}
    def branches(cs):
        """second families of consecutive fork columns that follow one another in most of their genomes: a parallel path"""
        out, cur = [], []
        for c in cs:
            if cur and c["alts"]:
                p = cur[-1][1]; nx = max(c["alts"], key=lambda n: E.get((p["id"], n["id"]), 0))
                if c["k"] == cur[-1][0] + 1 and E.get((p["id"], nx["id"]), 0) >= 0.5 * min(p["n"], nx["n"]): cur.append((c["k"], nx)); continue
            if len(cur) >= 3: out.append(cur)
            cur = [(c["k"], c["alts"][0])] if c["alts"] else []
        if len(cur) >= 3: out.append(cur)
        return out
    # sections: each spine run, and each run of fork and variable columns between two spines (one header for them all)
    secs = []
    for r in runs:
        if r["kind"] != "spine" and secs and secs[-1]["kind"] == "open" and r["a"] == secs[-1]["b"] + 1:
            secs[-1]["b"] = r["b"]; secs[-1]["runs"].append(r)
        else: secs.append(dict(kind="spine" if r["kind"] == "spine" else "open", a=r["a"], b=r["b"], runs=[r]))
    for sec in secs:
        sec["cols"] = [c for r in sec["runs"] for c in r["cols"]]
        kinds = sorted({r["kind"] for r in sec["runs"]})
        sec["what"] = "spine" if sec["kind"] == "spine" else {("fork",): "fork", ("variable",): "variable"}.get(tuple(kinds), "fork and variable columns")
        sec["rgp_mean"] = sum(c["rgp"] for c in sec["cols"]) / len(sec["cols"])
    if len(secs) >= 2:                  # the stretches most and least in regions of plasticity: the only ones to call so
        hi = max(secs, key=lambda s: s["rgp_mean"]); lo = min(secs, key=lambda s: s["rgp_mean"])
        if hi is not lo:
            F.plastic = dict(most=(hi["a"], hi["b"]), least=(lo["a"], lo["b"]))
            F.add(f"PLASTICITY: the stretch most in regions of plasticity: {span(hi)} ({hi['what']}), RGP {rng([c['rgp'] for c in hi['cols']], F.pct)}; "
                  f"the least: {span(lo)} ({lo['what']}), RGP {rng([c['rgp'] for c in lo['cols']], F.pct)}.", glob=True)
    F.add(f"STRETCHES, in window order (each spine; between spines, the fork and variable columns; the family listed is the "
          f"most carried at its column; share of {unit}; RGP: range over the columns):", glob=True)
    for i, sec in enumerate(secs, 1):
        cs = sec["cols"]; rgp = rng([c["rgp"] for c in cs], F.pct); ks = [c["k"] for c in cs]
        if sec["kind"] == "spine":
            genes = ", ".join(F.name(c["top"], k=c["k"]) + prod(c["top"]) for c in cs)
            F.add(f"{i}. {spann(sec)} spine: share {rng([c['s1'] for c in cs], F.pct)}, {parts(cs)}, RGP {rgp}. Genes: {genes}. {model(cs)}.", cols=ks)
            continue
        h = F.add(f"{i}. {spann(sec)} {sec['what']}: RGP {rgp}. {model(cs, listed=False)}.", cols=ks)
        inb = set()
        for r in sec["runs"]:
            for bb in (branches(r["cols"]) if r["kind"] == "fork" else []):
                ns = [n for _, n in bb]; inb |= {id(n) for n in ns}
                F.add(f"   a second branch of {F.int(len(ns))} families through columns {F.int(bb[0][0])}-{F.int(bb[-1][0])}, a parallel path "
                      f"that some {unit} carry instead of the most carried families there, each following the previous one in most of its "
                      f"{unit}, share {rng([n['n'] / NG for n in ns], F.pct)}, {parts([dict(top=n) for n in ns])}; the branch's families: "
                      + ", ".join(F.name(n, k=k) for k, n in bb), cols=[k for k, _ in bb], head=h)
        mark = (lambda r: f" {r['kind']}") if sec["what"] == "fork and variable columns" else (lambda r: "")
        for r in sec["runs"]:
            if r["kind"] == "variable" and len(r["cols"]) > 4:                   # a long variable run: one line
                F.add(f"   {span(r)}{mark(r)}, share {rng([c['s1'] for c in r['cols']], F.pct)}, {parts(r['cols'])}: "
                      + ", ".join(f"{F.name(c['top'], k=c['k'])}{prod(c['top'])} {F.pct(c['s1'])}" for c in r["cols"]),
                      cols=[c["k"] for c in r["cols"]], head=h)
                continue
            for c in r["cols"]:
                al = [n for n in c["alts"][:3] if id(n) not in inb]
                alts = "; also " + ", ".join(fam_s(n, c["k"], c["top"]) for n in al) if al else ""
                F.add(f"   {F.int(c['k'])}{mark(r)}: {fam_s(c['top'], c['k'])}{prod(c['top'])}{alts}", cols=[c["k"]], head=h)
    # PRODUCTS: the informative products of named genes in groups sharing a prefix, and of the second families, not shown yet
    txt = F.txt(); cand = []; tops = [c["top"] for c in cols]
    for k, n in enumerate(tops):
        if not n.get("named") or n["label"].startswith("y"): continue
        if [x for x in tops[max(0, k - 1): k + 2] if x is not n and x.get("named") and x["label"][:3] == n["label"][:3]]: cand.append(n)
    cand += [a for c in cols for a in c["alts"] if a.get("named")]
    seen, items = set(), []
    for n in cand:
        lab, p = n["label"], (n.get("product") or "").strip()
        if lab in seen or not informative(p) or f'{lab} "' in txt: continue
        seen.add(lab); items.append(f"{lab} {F.text(p, 40)}")
    if items: F.add("PRODUCTS of named genes (for [[ ]]): " + "; ".join(items[:40]), glob=True)
    last = cols[-1] if cols else None
    if last: F.add(f"END: the window ends at column {F.int(last['k'])}, {F.name(last['top'], k=last['k'])}, in {F.pct(last['s1'])} of {unit}.",
                   cols=[last["k"]], glob=True)
    return F


# ---------------------------------------------------------------- the prompt
STYLE = """**The spine starts at the origin.** Columns 0 to 5, `dnaA`, `dnaN`, `recF`, `gyrB`, `yidB` and `yidA`, sit in essentially every genome, persistent, and hardly ever in a region of plasticity: 0 to 0.5% of calls. The model names `dnaN`, `recF` and `gyrB` in 99.7 to 100% of genomes, `yidB` in 93% and `yidA` in 99%.

**The first fork is at `yidX`**, column 6. Then the `dgo` operon, [[galactonate catabolism]], variable columns 7 to 11: about 85% of genomes carry it, and panRGP puts it in a region of plasticity in 90% of them. The model is not lost there: it names each gene in 87 to 99% of genomes, with hardly any hesitation. A variable region can keep a predictable internal order.

**Then a long persistent stretch**, `cbrA` to `yicS` through the `ilv` and `uhp` operons, where the model names the next gene in 92% of genomes and seldom hesitates. Around columns 42 to 52 the graph opens: 50% of the calls there fall in a region of plasticity, and the model's hesitation peaks, up to 1.4 bits. Some genomes carry [[type III secretion genes]] there, others [[sugar transport genes]].

**The window ends in the `waa` locus**, [[which builds the lipopolysaccharide core]]: `waaA` in most genomes, then a spread of glycosyltransferases."""

SYSTEM = f"""You write the reading of one window of an E. coli pangenome graph for a scientific web page. The page shows Bacformer, a genomic language model, reading chromosomes gene by gene: at each gene it calls the next one. A window is 80 columns of genes; the facts you get describe its stretches, its genes and how well the model names them.

Rules:
1. Use only the facts. Every number you write must appear in the facts, written as there (you may round bits to 1 decimal), next to the genes or columns it belongs to: a range belongs to the stretch the facts give it for, never to a part of that stretch or to another one. Write numbers in digits, never in words: no "two", "four-gene", "half", "nine in ten", "a few hundred", "dozens". Write a range as "92 to 99%".
2. Gene names only from the facts, each in backticks: `dnaA`. An operon may be named by the shared prefix of its genes in the facts: `dgo`. Name an unnamed family briefly by its product, without backticks. At most 6 genes in a row; for a longer run write `first` to `last`.
3. Biology only inside double square brackets, [[galactonate catabolism]]: 2 to 6 words on what a group of genes does together, built from the words of its products or gene names in the facts (the PRODUCTS line gives more of them). Never copy a product into brackets, never bracket a single gene's product, and never name a substance, pathway or function the products and names do not state. Outside brackets no biology at all (no "nitrate", "energy", "respiration", "ribosomal", "envelope" in bold sentences) and no causes.
4. Senses: bits measure hesitation, so the column with the most bits is the most hesitant, never "the lowest hesitation"; "named" is the share of calls where the model names the gene. RGP N% means N% of the calls there are inside a region of plasticity: write "in a region of plasticity in N% of calls", never "outside any region of plasticity in N%". Superlatives only as the facts state them (PLASTICITY, HESITATION, "most hesitant", "fewest named"), no others. A persistent family in a fork or variable column is carried by nearly every chromosome but sits beyond the window in some: never call it rare or say the others lack it.
5. Spine, fork and variable exactly as the facts label each column; a stretch mixing them is "fork and variable columns". Never write "RGP", "share" or the letters P, S, C: write "in a region of plasticity", "carried by", "persistent", "shell", "cloud".
6. 4 or 5 short paragraphs of 2 or 3 sentences, about 200 words in all (never more than 250: leave stretches out rather than list them all), separated by a blank line, each opening with a short bold sentence in **double asterisks**, in window order; open on what the window starts with, not on "The window opens". At most 3 numbers in a sentence: choose the telling ones.
7. Cover the spines, the forks and what their branches carry (a branch carries only the families of its own line, not the ones listed at the same columns), the variable stretches, and where the model names genes readily and where it hesitates. State the COMPARISON as it is given (inside against outside), never its opposite.
8. Plain, exact English, like the example. No headings, lists, other markup, or words about yourself or the facts.

Style example, the hand-written reading of the dnaA window (its numbers and its biology belong to that window, not to yours):

{STYLE}"""


def user_msg(F, wid):
    extra = ("\n\nThis is the dnaA window, the one of the style example: write your own reading from these facts; any number "
             "that is not in the facts is rejected.") if wid == DNAA else ""
    return f"Facts:\n{F.txt()}{extra}\n\nWrite the reading."


# ---------------------------------------------------------------- slips of form, put right without a new call
PARTN = {"P": "persistent", "S": "shell", "C": "cloud"}
def tidy(text, F):
    """the model's text with its slips of form put right: no headings or list markers, ranges as "A to B", "RGP" and
    "share" in words, the letters P/S/C spelled out, a row of more than 6 genes as `first` to `last`"""
    t = re.sub(r"(?m)^\s*#+\s*", "", text.strip())
    t = re.sub(r"(?<=\d)[\u202f\u00a0\u2009](?=\d{3}\b)", ",", t)
    t = re.sub(r"(?m)^\s*(?:[-•]|\d+\.)\s+", "", t)
    t = re.sub(r"(?m)^\s*\*\s+(?!\*)", "", t)
    segs = re.split(r"(`[^`\n]*`)", t)
    num = r"\d[\d,]*(?:\.\d+)?"
    for i, s in enumerate(segs):
        if s.startswith("`") and s.endswith("`") and len(s) > 1: continue
        s = re.sub(rf"(?<![\w.,\-–])({num})\s?[-–]\s?({num})(?![\w\-–]|\.\d)", r"\1 to \2", s)
        s = re.sub(r"\b(inside|outside|within|in) (any|an) RGPs?\b", lambda m: f"{m.group(1)} {'any' if m.group(2) == 'any' else 'a'} region of plasticity", s)
        s = re.sub(r"\b(inside|outside|within|in) RGPs\b", r"\1 regions of plasticity", s)
        s = re.sub(rf"\b(?:with |at )?RGP (?:of |at |is |was )?({num}(?: to {num})?%)", r"\1 in a region of plasticity", s)
        s = re.sub(r"\ban RGP\b", "a region of plasticity", s)
        s = re.sub(r"\bthe RGPs\b", "the regions of plasticity", s)
        s = re.sub(r"\bthe RGP\b", "the region of plasticity", s)
        s = re.sub(r"\b(in|within) RGPs?\b", r"\1 regions of plasticity", s)
        s = re.sub(r"(?<![\w-])RGPs?\b", lambda m: "the share in regions of plasticity", s)
        s = re.sub(rf"\b(?:at |with |by |of )?(?:a )?share (?:of )?({num}(?: to {num})?%)", rf"in \1 of {F.unit}", s)
        s = re.sub(rf"({num}(?: to {num})?%) shares?\b", rf"\1 of {F.unit}", s)
        s = re.sub(r"\ball ([PSC])\b(?=[,.;:)]|\s+and\b|\s+families|\s*$)", lambda m: "all " + PARTN[m.group(1)], s)
        s = re.sub(r"(?<![\w.])(\d+) ([PSC])\b(?=[,.;:)]|\s+and\b)", lambda m: f"{m.group(1)} {PARTN[m.group(2)]}", s)
        segs[i] = s
    t = "".join(segs)
    t = re.sub(r"`([^`\n]*?)([.,;:!?]?\*\*)`", r"`\1`\2", t)          # "`mutS.**`": the bold's end inside the gene's backticks
    t = re.sub(r"(^|[.;:!?]\s+|\*\*|\n)the share in regions", lambda m: m.group(1) + "The share in regions", t)
    G = r"`[^`\n]+`"
    def rows(m):
        items = re.findall(G, m.group(0))
        return f"{items[0]} to {items[-1]}" if len(items) > 6 else m.group(0)
    t = re.sub(rf"{G}(?:,\s+{G}){{5,}}(?:,?\s+and\s+{G})?", rows, t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


# ---------------------------------------------------------------- the checker
NUMW = dict(two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10, eleven=11, twelve=12, thirteen=13, fourteen=14,
            fifteen=15, sixteen=16, seventeen=17, eighteen=18, nineteen=19, twenty=20, thirty=30, forty=40, fifty=50, sixty=60,
            seventy=70, eighty=80, ninety=90)
NUMW_RE = re.compile(r"\b(" + "|".join(NUMW) + r")\b", re.I)
VAGUE_RE = re.compile(r"\b(hundreds?|thousands?|dozens?|half|halves|quarters?|twice|thrice|tenths?|(?:a|one|two)\s+thirds?|"
                      r"(?:nine|eight|seven|six|five|four|three|two|one) (?:in|out of) (?:ten|five|four|three|two))\b", re.I)
NUM_RE = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\w]|\.\d|,\d{3})")
META_RE = re.compile(r"\b(?:COMPARISON|HESITATION|STRETCHES|WHOLE WINDOW|TERMS|PLASTICITY|PRODUCTS)\b|\b(?i:the facts|as stated)\b")
UNIT_RE = re.compile(r"\s*(%|percent\b|per cent\b|-?bits?\b|famil(?:y|ies)\b|(?:complete\s+)?(?:genomes?|chromosomes?)\b)", re.I)
LOWER_RE = re.compile(r"\s*(?:to|-|–)\s*(\d[\d,]*(?:\.\d+)?)\s*(%|percent\b|per cent\b|bits?\b)", re.I)
IDENT_RE = re.compile(r"[\w'\-/.]*\d[\w'\-/.]*")
GENE_RE = re.compile(r"(?<![\w`])([a-z]{3}[A-Z][A-Za-z0-9_]*)(?![\w])")
PROT_RE = re.compile(r"(?<![\w`])([A-Z][a-z]{2}[A-Z][0-9]?)(?![\w])")
OPERON_RE = re.compile(r"(?<![\w`])([a-z]{3}[A-Z]?)\s+(?:operon|genes|locus|loci|cluster)\b")
WORDS3 = set("""the all its few new two six ten one any are old big odd set top end our key own net raw lac far low how who
why but not nor yet per via has had was can may use may six ten""".split())
ALLOWED_WORDS = {"panRGP", "PanGBank", "PPanGGOLiN", "Bacformer"}
NEG_RE = re.compile(r"\b(lacks?|lacking|absent|missing|do(?:es)? not carry|carry none)\b", re.I)
LEAST_RE = re.compile(r"hesitat\w*\s+(?:is\s+)?(?:the\s+)?(?:least|lowest)|least hesitant|lowest hesitation|hesitation is lowest|lowest in hesitation", re.I)
SUP_RE = re.compile(r"\b(?:(?:most|least) (?:variable|certain|uncertain|sure|unsure|confident|predictable|unpredictable|stable|"
                    r"conserved|reliable|difficult|hard|easy|surprising|open)|surest|hardest|easiest)\b", re.I)
BIO_RE = re.compile(r"\b(envelope|respirat\w*|ribosom\w*|nitrate|nitrite|energy|metaboli\w*|transport\w*|synthes\w*|biosynth\w*|catabol\w*|"
                    r"flagell\w*|fimbri\w*|pil(?:us|i)|secret\w*|lipopolysacch\w*|lipid\w*|sugars?|iron|sulfur|stress|toxins?|prophage|phage|"
                    r"motility|chemotaxis|division|replication|translation|transcription\w*|repair|membranes?|peptidoglycan|capsul\w*|"
                    r"biofilm|virulence|pathogen\w*|vitamins?|cofactors?|hydrogenase|chaperones?|efflux|resistance|antibiotics?|"
                    r"housekeeping|essential|enzym\w*|amino acids?|nucleotides?|carbon|nitrogen|phosphate|osmotic|adhesion|defen[cs]e)\b", re.I)
PLASTIC_RE = re.compile(r"\b(most|least) (plastic|in regions? of plasticity)\b", re.I)
HES_RE = re.compile(r"most hesitant|hesitates? (?:the )?most|hesitation peaks|peak hesitation|hesitation is highest|highest hesitation", re.I)
LOW_RE = re.compile(r"\bfewest\b|\blowest at\b|least often named|names? (?:the )?fewest|named (?:the )?least", re.I)


def roundings(v, q0=0):
    """a fact number and the same number rounded to fewer decimals (at least q0)"""
    out = {v}; e = -v.as_tuple().exponent
    for q in range(q0, max(e, 0)):
        for mode in (ROUND_HALF_UP, ROUND_HALF_EVEN): out.add(v.quantize(Decimal(1).scaleb(-q), rounding=mode))
    return out


def _allowed(nums):
    """the numbers a text may write for the facts' numbers: rounded further, but bits to 1 decimal at least (the text's
    whole-number bits are refused apart), and never to 0% or 100% unless the facts show it"""
    a = {k: set().union(*(roundings(v, 1 if k == "bits" else 0) for v in vs)) if vs else set() for k, vs in nums.items()}
    if "pct" in a: a["pct"] = {x for x in a["pct"] if x not in (0, 100) or x in nums["pct"]}
    return a


_LNUM = re.compile(r"(?<![\w.,])(\d+(?:\.\d+)?)(?:\s*(%)|\s*(bits?)\b)")
_LRANGE = re.compile(r"(?<![\w.,])(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)(%)")
def _line_index(F):
    """per facts line: its words, and its percent and bits numbers (a range's lower end is a percent)"""
    if getattr(F, "_idx", None) is not None: return F._idx
    idx = []
    for ln, inf in zip(F.lines, F.info):
        nums = collections.defaultdict(set)
        for mm in _LRANGE.finditer(ln): nums["pct"].add(Decimal(mm.group(1)))
        for mm in _LNUM.finditer(ln): nums["pct" if mm.group(2) else "bits"].add(Decimal(mm.group(1)))
        idx.append(dict(inf, words=set(re.findall(r"[\w'-]+", ln)), nums=nums))
    F._idx = idx; return idx


def sentences(text):
    """paragraphs, each a list of sentences (bold marks dropped)"""
    out = []
    for p in re.split(r"\n\s*\n", text.replace("**", "").strip()):
        out.append([s for s in re.split(r"(?<=[.;])\s+(?=[A-Z`\[\"])", " ".join(p.split())) if s])
    return out


def cols_of(s, W=80):
    """the columns a sentence names: "column 5", "columns 3 to 9", "columns 3, 4 and 7", "fork columns, 51 to 79",
    "the last 29 columns", "the first 6 columns" """
    cs = set()
    for m in re.finditer(r"\bcolumns?,\s+(\d+)\s+to\s+(\d+)(?![.,]\d|\s*%)", s, re.I): cs |= set(range(int(m.group(1)), int(m.group(2)) + 1))
    for m in re.finditer(r"\b(first|last) (\d+) columns\b", s, re.I):
        n = min(int(m.group(2)), W); cs |= set(range(0, n)) if m.group(1).lower() == "first" else set(range(W - n, W))
    for m in re.finditer(r"\bcolumns? ((?:\d+(?![.,]\d)(?:\s*(?:,|and|to|-|–)\s*(?=\d))?)+)", s, re.I):
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        seps = re.findall(r"\d+\s*(to|-|–|,|and)\s*(?=\d)", m.group(1))
        cs |= set(nums)
        for (a, b), sep in zip(zip(nums, nums[1:]), seps):
            if sep in ("to", "-", "–") and b >= a: cs |= set(range(a, b + 1))
    return cs


def genes_of(s, F):
    g = {x.strip() for x in re.findall(r"`([^`\n]+)`", s)}
    for x in F.genes:
        if len(x) > 2 and re.search(rf"(?<![\w`]){re.escape(x)}(?![\w])", s): g.add(x)
    return g


def _gene_lines(g, F, idx):
    pats = {g} if g in F.genes else {x for x in F.genes if len(g) == 3 and x.startswith(g)} or {g}
    return {i for i, L in enumerate(idx) if L["words"] & pats}


def _scope(s, F, idx):
    """the facts lines a sentence points to: those of its genes and columns, the lines between `a` and `b` of a run,
    and each one's stretch line"""
    genes = genes_of(s, F); cols = cols_of(s); hit = set(); low = s.lower()
    for p, ks in F.atp.items():                                   # an unnamed family named by its product
        if len(p) >= 6 and p.rstrip("…") in low: cols = cols | ks
    for g in genes: hit |= {i for i in _gene_lines(g, F, idx) if not idx[i]["glob"]}
    for a, b in re.findall(r"`([^`]+)`\s+(?:to|through)\s+`([^`]+)`", s):
        ia = [i for i in _gene_lines(a, F, idx) if not idx[i]["glob"]]; ib = [i for i in _gene_lines(b, F, idx) if not idx[i]["glob"]]
        if ia and ib: hit |= set(range(min(ia), max(ib) + 1))
    for i, L in enumerate(idx):
        if L["cols"] & cols and not L["glob"]: hit.add(i)
    heads = {idx[i]["head"] for i in hit if idx[i]["head"] is not None}
    gcols = set().union(*(F.at.get(g, set()) for g in genes)) if genes else set()
    return hit | heads, cols | gcols


def check(text, F):
    """-> dict(ok, numbers, names, words, bad=[...], hints=[...]): every number, gene name, number word and identifier with a
    digit of text, each number against its sentence's genes and columns, and the claims the facts make (see the docstring)"""
    ftxt = F.txt(); flow = ftxt.lower(); idx = _line_index(F)
    allowed = _allowed(F.nums)
    anyv = set().union(*allowed.values()) if allowed else set()
    bad = []; nnum = nname = 0
    body = text
    for fr in sorted(F.frags, key=len, reverse=True):                      # "family 4" of a product: a name, not a number
        body = re.sub(re.escape(fr), lambda mm: re.sub(r"\d", "#", mm.group(0)), body, flags=re.I)
    interp_spans = [mm.span() for mm in re.finditer(r"\[\[.*?\]\]", body, flags=re.S)]
    # identifiers with a digit and letters (IS3, 16S, DUF1234, 2-dehydro-3-deoxy...) must be in the facts; inside [[ ]] a
    # chemical name with a locant (2-oxoglutarate) is allowed
    def ident(mm):
        tok = mm.group(0).strip("'-/.")
        if not re.search(r"[A-Za-z]", tok): return mm.group(0)
        u = re.fullmatch(r"(\d[\d,]*(?:\.\d+)?)-(bits?|fold|columns?|genes?|famil(?:y|ies)|percent)", tok)
        if u: return u.group(1) + " " + u.group(2)                          # 0.4-bit: a number and its unit
        inside = any(a <= mm.start() < b for a, b in interp_spans)
        chem = re.fullmatch(r"\d+(?:,\d+)*-[A-Za-z]{4,}[\w,-]*", tok) and not re.match(r"\d+(?:,\d+)*-(?:genes?|columns?|famil|fold|bits?|stretch)", tok)
        if tok.lower() not in flow and not (inside and chem): bad.append(dict(kind="identifier", text=tok, why="not in the facts"))
        return re.sub(r"\d", "#", mm.group(0))
    body = IDENT_RE.sub(ident, body)
    for mm in NUM_RE.finditer(body):
        raw = mm.group(0); nnum += 1
        try: v = Decimal(raw.replace(",", ""))
        except InvalidOperation: continue
        u = UNIT_RE.match(body, mm.end()); unit = (u.group(1).lower() if u else "")
        if not unit:                                                         # a range's lower end: the unit of its upper end
            lo = LOWER_RE.match(body, mm.end())
            if lo: unit = lo.group(2).lower()
        kind = "pct" if unit in ("%", "percent", "per cent") else "bits" if "bit" in unit else "fam" if unit.startswith("famil") and "." in raw else None
        ok = v in (allowed.get(kind, set()) if kind else anyv)
        if ok and kind == "bits" and "." not in raw and v not in F.nums["bits"]:
            bad.append(dict(kind="format", text=f"{raw} bits", why="bits rounded to a whole number: write them as the facts do")); continue
        if not ok:
            ctx = body[max(0, mm.start() - 30): mm.end() + 12].replace("\n", " ")
            bad.append(dict(kind="number", text=raw + ("%" if kind == "pct" else f" {unit}" if unit else ""),
                            why=f"not a {dict(pct='percent', bits='bits value', fam='families value').get(kind, 'number')} of the facts", context=ctx.strip()))
    plain = re.sub(r"\[\[.*?\]\]", " ", text, flags=re.S)
    for mm in re.finditer(r"(?<![\w.-])\.\d+\b|(?<![\w,-])\d+,\d{1,2}\b(?!,?\d)", plain):
        bad.append(dict(kind="format", text=mm.group(0), why="write numbers as the facts do (0.97, not .97 or 0,97)"))
    for mm in META_RE.finditer(text):
        bad.append(dict(kind="wording", text=mm.group(0), why="the reading speaks of the window, not of the facts"))
    for mm in NUMW_RE.finditer(text):
        nnum += 1
        if Decimal(NUMW[mm.group(1).lower()]) not in anyv: bad.append(dict(kind="number word", text=mm.group(0), why="not a number of the facts"))
    for mm in VAGUE_RE.finditer(text):
        bad.append(dict(kind="number word", text=mm.group(0), why="write numbers in digits, from the facts"))
    genes = F.genes
    def known(g):
        return g in genes or (g.isalpha() and g.islower() and len(g) == 3 and any(x.startswith(g) for x in genes))
    for mm in re.finditer(r"`([^`\n]+)`", text):
        g = mm.group(1).strip(); nname += 1
        if not known(g) and not re.search(rf"(?<!\w){re.escape(g.lower())}(?!\w)", flow) and \
           not any(k.endswith("…") and g.lower().startswith(k[:-1]) for k in F.atp):   # a product of the facts in backticks: a slip of form
            bad.append(dict(kind="gene", text=g, why="not a gene of the facts"))
    nog = re.sub(r"`[^`\n]*`", " ", text)
    for mm in GENE_RE.finditer(nog):
        g = mm.group(1); nname += 1
        if g not in genes and g not in ALLOWED_WORDS and g not in ftxt: bad.append(dict(kind="gene", text=g, why="not a gene of the facts"))
    for mm in PROT_RE.finditer(nog):
        g = mm.group(1)
        if g[0].lower() + g[1:] not in genes and g not in ftxt and g not in ALLOWED_WORDS:
            nname += 1; bad.append(dict(kind="gene", text=g, why="not a gene or product of the facts"))
    for mm in OPERON_RE.finditer(re.sub(r"\[\[.*?\]\]", " ", nog)):
        g = mm.group(1)
        if not known(g) and g.lower() not in WORDS3 and not re.search(rf"(?<![\w`]){g}(?![\w])", ftxt):
            bad.append(dict(kind="gene", text=g, why="not a gene or operon of the facts"))
    bad += _claims(text, F, idx)
    seen = set(); uniq = []
    for b in bad:
        k = (b["kind"], b["text"])
        if k not in seen: seen.add(k); uniq.append(b)
    return dict(ok=not uniq, numbers=nnum, names=nname, words=len(re.findall(r"[A-Za-z0-9][\w'.,%-]*", text.replace("**", ""))),
                bad=uniq, hints=interp_hints(text, F))


def _claims(text, F, idx):
    """numbers against their sentence's lines, genes against their columns, the comparison's direction, and the claims
    the facts do not make"""
    bad = []; allF = _allowed(F.nums); flow = F.txt().lower()
    glob = collections.defaultdict(set)
    for L in idx:
        if L["glob"]:
            for k, v in L["nums"].items(): glob[k] |= v
    for para in sentences(text):
        prev = set()
        for j, s in enumerate(para):
            hit, cols = _scope(s, F, idx)
            use = (hit | prev) if (hit or prev) else set(range(len(idx)))
            pool = collections.defaultdict(set)
            for i in use:
                for k, v in idx[i]["nums"].items(): pool[k] |= v
            for k in glob: pool[k] |= glob[k]
            al = _allowed(pool)
            s_ = re.sub(r"\[\[.*?\]\]", " ", s)
            for m in re.finditer(r"(?<![\w.,])(\d+(?:\.\d+)?)(?=\s*(%|percent\b|per cent\b|bits?\b))", s_):
                v = Decimal(m.group(1)); k = "bits" if m.group(2).startswith("bit") else "pct"
                if v not in al.get(k, set()) and v in allF.get(k, set()):
                    bad.append(dict(kind="scope", text=m.group(0) + ("%" if k == "pct" else " bits"),
                                    why="a number of another stretch or column than the ones its sentence names", context=s[:160]))
            for m in re.finditer(r"(?<![\w.,])(\d+(?:\.\d+)?)(?=\s*(?:to|-|–)\s*\d+(?:\.\d+)?\s*%)", s_):
                v = Decimal(m.group(1))
                if v not in al.get("pct", set()) and v in allF.get("pct", set()):
                    bad.append(dict(kind="scope", text=m.group(0) + "% (a range's lower end)",
                                    why="a number of another stretch or column than the ones its sentence names", context=s[:160]))
            for rx, where, what in ((HES_RE, F.hes_cols, "most hesitant"), (LOW_RE, F.low_cols, "fewest named")):
                for m in rx.finditer(s):
                    if what == "most hesitant" and re.search(r"\bno column\b|hardly|barely|seldom", s, re.I): continue
                    cc = _claim_cols(s, m, F) or cols
                    if cc and not (cc & where):
                        bad.append(dict(kind="claim", text=m.group(0), why=f"the facts do not state that column as the {what}", context=s[:160]))
            for m in PLASTIC_RE.finditer(s):
                w = m.group(1).lower(); ab = F.plastic.get("most" if w == "most" else "least")
                if not ab: bad.append(dict(kind="claim", text=m.group(0), why="the facts name no stretch so", context=s[:160])); continue
                cc = cols or (_scope(para[j + 1], F, idx)[1] if j + 1 < len(para) else set())    # a bold opener: the next sentence
                if cc and not (cc & set(range(ab[0], ab[1] + 1))):
                    bad.append(dict(kind="claim", text=m.group(0), why=f"the facts say so of columns {ab[0]} to {ab[1]} only", context=s[:160]))
            prev = hit
    bad += _colgene(text, F)
    bad += _direction(text, F)
    for m in NEG_RE.finditer(re.sub(r"\[\[.*?\]\]", " ", text)):
        bad.append(dict(kind="claim", text=m.group(0), why="the facts never say chromosomes lack a gene"))
    for m in re.finditer(r"outside (?:any|a) regions? of plasticity,? in (?:only )?\d[\d.]*(?: to \d[\d.]*)?%", text, re.I):
        bad.append(dict(kind="claim", text=m.group(0), why="the facts give the share of calls inside a region of plasticity"))
    for m in SUP_RE.finditer(re.sub(r"\[\[.*?\]\]", " ", text)): bad.append(dict(kind="claim", text=m.group(0), why="a superlative the facts do not state"))
    for m in re.finditer(r"\*\*(.+?)\*\*", text, re.S):                 # the bold sentences: structure only
        b = next((x for x in BIO_RE.finditer(re.sub(r"\[\[.*?\]\]|`[^`]*`|\"[^\"]*\"", " ", m.group(1)))
                  if x.group(0).lower() not in flow), None)
        if b: bad.append(dict(kind="claim", text=b.group(0), why="biology outside [[ ]], in a bold sentence", context=m.group(1)[:120]))
    for m in LEAST_RE.finditer(text): bad.append(dict(kind="claim", text=m.group(0), why="the facts give the most hesitant columns, not the least"))
    for m in re.finditer(r"\b\d+\s+(?:in|out of|of every)\s+\d+\b(?!\s*%|\.\d|\s+to\s+\d)", re.sub(r"\[\[.*?\]\]", " ", text)):
        bad.append(dict(kind="claim", text=m.group(0), why="a fraction the facts do not give"))
    return bad


def _claim_cols(s, m, F):
    """the columns a claim ("hesitates most at ...", "... the fewest named") is about: what follows it up to the next
    punctuation, else nothing (the caller takes the sentence's)"""
    obj = re.split(r"[;.]", s[m.end(): m.end() + 60])[0]
    cc = cols_of(obj)
    for g in re.findall(r"`([^`\n]+)`|\"([^\"\n]+)\"", obj):
        g = (g[0] or g[1]).strip()
        cc |= F.at.get(g, set()) | F.atp.get(g.lower(), set())
    for g in F.genes:
        if re.search(rf"(?<![\w`]){re.escape(g)}(?![\w])", obj): cc |= F.at.get(g, set())
    return cc


def _colgene(text, F):
    """a gene named next to a column number must sit at that column in the facts (most carried, second family or branch)"""
    bad = []
    C = r"columns? (\d+)(?:\s*(?:to|-|–)\s*(\d+))?"
    G = r"`([^`]+)`(?:\s+to\s+`([^`]+)`)?"
    pats = [(rf"\b{C},?\s+(?:\(|the\s+)?{G}", "cg"), (rf"{G},?\s+(?:\(|in |at ){C}", "gc")]
    for para in sentences(text):
        for s in para:
            for pat, order in pats:
                for m in re.finditer(pat, s, re.I):
                    if order == "cg":
                        a, b, g1, g2 = m.group(1), m.group(2), m.group(3), m.group(4)
                        if re.match(r"\s*,?\s*(?:at|in|through|across|over|column|columns)\s+(?:columns?\s+)?\d", s[m.end():]) or \
                           re.search(r"\b(?:at|in|through|across|over|from|to|between|into|on|by)\s+$", s[max(0, m.start() - 20): m.start()]): continue
                    else:
                        g1, g2, a, b = m.group(1), m.group(2), m.group(3), m.group(4)
                        if re.match(r"\s*(?:,|and)\s*\d", s[m.end():]): continue          # a list of columns: one for each gene
                        if re.search(r"columns?\s+\d+(?:\s*(?:to|-|–)\s*\d+)?\s*,?\s*$", s[max(0, m.start() - 20): m.start()], re.I): continue
                    a = int(a); b = int(b or a); single = (m.group(2) if order == "cg" else m.group(4)) is None
                    for g, end in ((g1, a), (g2, b)):
                        if not g or g not in F.at or (g is g2 and single): continue
                        want = {end} if (g2 and not single) else set(range(a, b + 1))
                        if not (want & F.at[g]):
                            bad.append(dict(kind="column", text=m.group(0), why=f"`{g}` is at column {', '.join(map(str, sorted(F.at[g])))} in the facts", context=s[:160]))
    return bad


def _direction(text, F):
    """inside/outside a region of plasticity: the direction words and which number goes with which side, as COMPARISON"""
    c = F.comp
    if not c: return []
    rset = lambda *xs: set().union(*(roundings(Decimal(x.rstrip("%"))) for x in xs))
    IN, OUT = rset(c["ei"], c["ai"]), rset(c["eo"], c["ao"])
    bad = []
    for para in sentences(text):
        for s in para:
            low = s.lower()
            if not re.search(r"inside|region of plasticity|plastic", low) or not re.search(r"outside|the other|against|than", low): continue
            if re.search(r"hesitat\w*\s+less|less hesita|hesitates? (?:more|less) outside", low) and c["h"] == "hesitates more":
                bad.append(dict(kind="direction", text="hesitation", why="inside a region of plasticity the model hesitates more", context=s[:160]))
            if re.search(r"names? more genes|names more\b", low) and c["n"] == "names fewer genes":
                bad.append(dict(kind="direction", text="naming", why="inside a region of plasticity the model names fewer genes", context=s[:160]))
            opens = re.match(r"\s*(?:and\s+)?(inside|in|outside) (?:a|any) regions? of plasticity\b", low)   # "A against B": A is this side
            if opens and not re.search(r"\bthere\b|\bthan inside\b", low):
                x, y = (OUT, IN) if opens.group(1) == "outside" else (IN, OUT)
                for a, b in re.findall(r"(\d+(?:\.\d+)?)%?\s*(?:bits\s+)?against\s+(\d+(?:\.\d+)?)", s):
                    A, B = Decimal(a), Decimal(b)
                    if A in y and B in x and not (A in x and B in y):
                        bad.append(dict(kind="direction", text=f"{a} against {b}", why="the two sides are swapped", context=s[:160]))
            fi = low.find("inside") if "inside" in low else low.find("in a region of plasticity")
            fi = fi if fi >= 0 else 10 ** 6
            in_first = fi < low.find("outside") if "outside" in low else True
            if "against" not in low and "outside" in low and "inside" in low and in_first:
                seg_in = s[low.find("inside"): low.find("outside")]; seg_out = s[low.find("outside"):]
                for mm in re.finditer(r"(?<![\w.,])(\d+(?:\.\d+)?)(?=\s*(?:%|bits?\b))", seg_in):
                    if Decimal(mm.group(1)) in OUT and Decimal(mm.group(1)) not in IN:
                        bad.append(dict(kind="direction", text=mm.group(0), why="the value outside a region of plasticity", context=s[:160]))
                for mm in re.finditer(r"(?<![\w.,])(\d+(?:\.\d+)?)(?=\s*(?:%|bits?\b))", seg_out):
                    if Decimal(mm.group(1)) in IN and Decimal(mm.group(1)) not in OUT:
                        bad.append(dict(kind="direction", text=mm.group(0), why="the value inside a region of plasticity", context=s[:160]))
    if c["h"] == "hesitates more" and re.search(r"(?:does not|doesn't|do not|never) follow|ignores? the regions", text, re.I):
        bad.append(dict(kind="direction", text="does not follow", why="inside a region of plasticity the model hesitates more"))
    if c["h"] != "hesitates more" and re.search(r"hesitation follows|follows (?:the regions of )?plasticity", text, re.I):
        bad.append(dict(kind="direction", text="follows plasticity", why=f"the model {c['h']} inside a region of plasticity"))
    return bad


STOP = set("""the a an and or of in on to for from by with as at into its their that which this these those genes gene
pathway pathways operon operons island islands system systems cluster locus loci branch path parallel stretch
metabolism catabolism biosynthesis synthesis assembly uptake transport transporter transporters utilization utilisation
degradation production maturation formation regulation response repair processing export import building builds build
carried instead main one core other different differ between types type another second first sugar sugars family
protein proteins breakdown handling use work working genes' related""".split())
GENERIC_OK = set("""containing like domain chain core antigen reduction oxidation respiration respiratory translation transcription
decay repair recombination envelope membrane cell division replication ribosome ribosomal secretion efflux multidrug fimbriae
fimbrial flagellar flagella motility chemotaxis lipid lipids fatty acid acids amino nucleotide energy stress iron sulfur
phosphate nitrogen carbon osmotic heat shock toxin antitoxin phage prophage mobile element elements insertion restriction
modification quorum sensing biofilm capsule polysaccharide""".split())
def interp_hints(text, F):
    """words of [[...]] found in no product or gene name of the facts (a hint for a retry, not a failure: a substrate
    can be named right from gene names alone)"""
    V = set(re.findall(r"[a-z]{3,}", F.txt().lower())); P = {w[:7] for w in V if len(w) >= 7}
    out = []
    for m in re.finditer(r"\[\[(.+?)\]\]", text):
        for w in re.findall(r"[a-z]{4,}", m.group(1).lower()):
            if w in STOP or w in GENERIC_OK or w in V or (len(w) >= 7 and w[:7] in P): continue
            out.append(dict(text=w, where=m.group(0)))
    return out


def retry_msg(chk, truncated=False):
    B = collections.defaultdict(list)
    for b in chk["bad"]: B[b["kind"]].append(b)
    say = []
    if B["number"]: say.append("these numbers are not in the facts: " + ", ".join(b["text"] for b in B["number"][:12]))
    if B["scope"]: say.append("these numbers belong to other genes or columns than the ones their sentence names: "
                              + "; ".join(f"{b['text']} in \"{b['context'][:90]}\"" for b in B["scope"][:5]))
    if B["column"]: say.append("these genes are not at the column given: " + "; ".join(f"{b['text']} ({b['why']})" for b in B["column"][:5]))
    if B["direction"]: say.append("the comparison is inverted: " + "; ".join(sorted({b["why"] for b in B["direction"]})))
    if B["claim"]: say.append("claims the facts do not make: " + "; ".join(f"\"{b['text']}\" ({b['why']})" for b in B["claim"][:6]))
    if B["number word"]: say.append("these number words are not allowed (use digits from the facts, or no number): " + ", ".join(b["text"] for b in B["number word"][:8]))
    if B["format"]: say.append("write numbers as the facts do: " + ", ".join(b["text"] for b in B["format"][:6]))
    if B["gene"]: say.append("these gene names are not in the facts: " + ", ".join(b["text"] for b in B["gene"][:10]))
    if B["identifier"]: say.append("these names are not in the facts: " + ", ".join(b["text"] for b in B["identifier"][:8]))
    if B["wording"]: say.append("do not speak of the facts or their sections (" + ", ".join(b["text"] for b in B["wording"][:4]) + "): speak of the window")
    if chk.get("hints"): say.append("inside [[ ]] use only words of the products and gene names: " + ", ".join(sorted({h["text"] for h in chk["hints"]}))[:200] + " are not there")
    if truncated: say.append("the text was cut at its length limit: write at most 220 words")
    return ("Your reading was rejected: " + "; ".join(say) + ". Rewrite the whole reading, following the rules: every number "
            "exactly as written in the facts and next to the genes or columns it belongs to, genes only from the facts.")


def reason(chk, truncated):
    """a short account of a failed check, for the page (its details in `bad`)"""
    b = chk["bad"]
    if truncated and not b: return "the model's text was cut at its length limit"
    kinds = collections.Counter(x["kind"] for x in b)
    what = dict(number="a number not in this window's data", scope="a number given to the wrong genes or columns",
                column="a gene put at the wrong column", direction="the comparison inside and outside regions of plasticity inverted",
                claim="a claim this window's data do not make", gene="a gene not in this window", identifier="a name not in this window",
                wording="words about the facts rather than the window", format="a number badly written")
    what["number word"] = "a number in words"
    return "the model's text still had " + ", ".join(what.get(k, k) for k, _ in kinds.most_common(3)) + " after a second attempt"


# ---------------------------------------------------------------- the reading as HTML (the page's classes)
INTERP_TITLE = "interpretation from the gene names and products, not measured here"
def to_html(text, F):
    text = re.sub(r"^\s*#+\s*", "", text.strip(), flags=re.M)                # no headings
    genes = sorted(F.genes, key=len, reverse=True)
    gre = re.compile(r"(?<![\w`])(" + "|".join(map(re.escape, genes)) + r")(?![\w])") if genes else None
    known = lambda g: g in F.genes or (g.isalpha() and g.islower() and len(g) == 3 and any(x.startswith(g) for x in F.genes))
    out = []
    for p in re.split(r"\n\s*\n", text):
        p = " ".join(p.split())
        if not p: continue
        p = re.sub(r"^[-*]\s+(?!\*)", "", p)                                  # no list markers
        segs = re.split(r"(`[^`]*`)", p)                                     # bare gene names of the facts get their style
        if gre: segs = [s if s.startswith("`") else gre.sub(r"`\1`", s) for s in segs]
        h = html.escape("".join(segs), quote=True)
        h = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", h)
        h = re.sub(r"(?<!\*)\*(?!\*)([^*]+?)(?<!\*)\*(?!\*)", r"\1", h)         # single-asterisk italics: plain
        h = re.sub(r"`([^`]+)`", lambda mm: f'<span class="gene">{mm.group(1)}</span>' if known(html.unescape(mm.group(1))) else mm.group(1), h)
        h = re.sub(r"\[\[(.+?)\]\]", rf'<span class="interp" title="{INTERP_TITLE}">\1</span>', h)
        h = h.replace("**", "").replace("`", "").replace("[[", "").replace("]]", "")
        if not balanced(h): h = html.escape(re.sub(r"\*\*|\[\[|\]\]|`", "", p), quote=True)
        out.append(f'<p class="note">{h}</p>')
    return "".join(out)


def balanced(h):
    st = []
    for mm in re.finditer(r"<(/?)(b|span)\b[^>]*>", h):
        if not mm.group(1): st.append(mm.group(2))
        elif not st or st.pop() != mm.group(2): return False
    return not st


# ---------------------------------------------------------------- the LLM
_TOKEN = [None, False]
def token():
    """HF_TOKEN, else the local login; read once, kept in memory only"""
    if not _TOKEN[1]:
        t = os.environ.get("HF_TOKEN") or None
        if not t:
            try:
                from huggingface_hub import get_token
                t = get_token()
            except Exception: t = None
        _TOKEN[0] = t; _TOKEN[1] = True
    return _TOKEN[0]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """a redirect is an error: the Authorization header is never sent to another address"""
    def redirect_request(self, *a, **k): return None
OPENER = urllib.request.build_opener(_NoRedirect)


def _bill(cost):
    with LOCK: _roll(); USAGE["cost"] += cost; USAGE["calls"] += 1


def llm(messages, deadline):
    """one completion: the providers in order, the next on a timeout, a redirect, a 429 or a 5xx.
    -> dict(text, finish, usage, provider, ms). Every call is counted in the day's dollars: a call cut by a timeout or
    without usage at its worst case (the prompt and MAX_TOKENS out). The router's error text is never passed on."""
    tok = token()
    if not tok: raise Unavailable("no Hugging Face token on this server (HF_TOKEN)", code="no_token")
    errs = []
    est_in = len(json.dumps(messages)) / 3.0
    for prov in PROVIDERS:
        left = deadline - time.time()
        if left < 5: errs.append(f"{prov}: no time left"); break
        pin, pout = PRICES.get(prov, (0.3, 1.2)); worst = (est_in * pin + MAX_TOKENS * pout) / 1e6
        body = dict(model=f"{MODEL}:{prov}", messages=messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                    chat_template_kwargs=dict(thinking=False), stream=False)
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
        t0 = time.time()
        try:
            with OPENER.open(req, timeout=min(TIMEOUT, left)) as r:
                out = json.load(r); used = r.headers.get("x-inference-provider") or prov
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                AUTH.update(t=time.time(), code="refused", why=f"the token was refused by the Inference Providers router (HTTP {e.code})")
                raise Unavailable(AUTH["why"], code="refused")
            if e.code == 402:
                AUTH.update(t=time.time(), code="credit", why="no Inference Providers credit left on this token's account (HTTP 402)")
                raise Unavailable(AUTH["why"], code="credit")
            errs.append(f"{prov}: HTTP {e.code}"); continue
        except Exception as e:                                               # timeout, connection: maybe billed
            _bill(worst); errs.append(f"{prov}: {type(e).__name__}"); continue
        try:
            ch = out["choices"][0]; text = (ch["message"].get("content") or "").strip()
        except Exception:
            _bill(worst); errs.append(f"{prov}: malformed answer"); continue
        u = out.get("usage") or {}
        pin, pout = PRICES.get(used, (pin, pout))
        cost = u.get("estimated_cost")
        if cost is None and u.get("prompt_tokens") is not None:
            cost = (u.get("prompt_tokens", 0) * pin + (u.get("completion_tokens") or MAX_TOKENS) * pout) / 1e6
        if cost is None: cost = worst
        _bill(float(cost))
        if not text: errs.append(f"{prov}: empty answer"); continue
        usage = dict(prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                     cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                     reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"), cost_usd=round(float(cost), 6))
        return dict(text=text, finish=ch.get("finish_reason"), usage=usage, provider=used, ms=round((time.time() - t0) * 1000))
    raise Upstream("the reading model could not be reached (" + "; ".join(errs) + ")", 30)


def generate(F, wid):
    """-> the cache entry: status llm (a checked text) or fallback (it failed the checker twice)"""
    deadline = time.time() + BUDGET
    msgs = [dict(role="system", content=SYSTEM), dict(role="user", content=user_msg(F, wid))]
    attempts = []
    for i in range(2):
        try: r = llm(msgs, deadline)
        except Upstream:
            if not attempts: raise
            break                                                            # the retry could not be made: fallback
        text = tidy(r["text"], F)
        chk = check(text, F); cut = r["finish"] == "length"
        attempts.append(dict(r, raw=r["text"], text=text, check=chk, truncated=cut))
        if chk["ok"] and not cut: break
        if i == 0:
            if deadline - time.time() < 12: break
            msgs += [dict(role="assistant", content=r["text"]), dict(role="user", content=retry_msg(chk, cut))]
    last = attempts[-1]; ok = last["check"]["ok"] and not last["truncated"]
    usage = dict(prompt_tokens=sum(a["usage"]["prompt_tokens"] or 0 for a in attempts),
                 completion_tokens=sum(a["usage"]["completion_tokens"] or 0 for a in attempts),
                 cost_usd=round(sum(a["usage"]["cost_usd"] for a in attempts), 6), ms=sum(a["ms"] for a in attempts))
    c = last["check"]
    e = dict(status="llm" if ok else "fallback", model=MODEL, provider=last["provider"], created=time.time(), usage=usage,
             attempts=attempts, checked=dict(ok=ok, numbers=c["numbers"], names=c["names"], words=c["words"], attempts=len(attempts)))
    if ok: e.update(text=last["text"], html=to_html(last["text"], F))
    else:
        e["reason"] = reason(c, last["truncated"])
        e["checked"]["bad"] = [{k: v for k, v in b.items() if k != "context"} for b in c["bad"][:12]]
    return e


# ---------------------------------------------------------------- cache, limits, the route
LOCK = threading.Lock()
MEM = collections.OrderedDict()          # window -> entry (with its facts hash)
FACTS = collections.OrderedDict()        # window -> (Facts, hash)
INFLIGHT = {}                            # window -> dict(ev, waiters, err): the generation under way
SLOTS = threading.BoundedSemaphore(MAX_PARALLEL)
QUEUED = [0]                             # new generations waiting for a slot
HITS = {}                                # client -> deque of its new generations' times (PER_IP_S)
DAYHITS = {}                             # client -> its new generations today
PRUNED = [0.0]
AUTH = dict(t=0.0, why=None, code=None)
USAGE = dict(day=None, n=0, cost=0.0, calls=0)
_HOME = [None]


def _safe(wid): return re.sub(r"[^A-Za-z0-9_.-]", "_", wid)[:80]
def _paths(wid):
    ds = [CACHE_DIR] + ([BUNDLED] if os.path.abspath(BUNDLED) != os.path.abspath(CACHE_DIR) else [])
    return [os.path.join(d, _safe(wid) + ".json") for d in ds]
def _today(): return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
def _to_midnight():
    n = datetime.datetime.now(datetime.timezone.utc)
    return int((n.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1) - n).total_seconds()) + 1
def _roll():
    """a new UTC day: the day's counters start again (under LOCK)"""
    if USAGE["day"] != _today(): USAGE.update(day=_today(), n=0, cost=0.0, calls=0); DAYHITS.clear()


def _usage_load():
    try:
        u = json.load(open(os.path.join(CACHE_DIR, "_usage.json")))
        if u.get("day") == _today(): USAGE.update(day=u["day"], n=int(u["n"]), cost=float(u.get("cost", 0)), calls=int(u.get("calls", 0)))
    except Exception: pass
    _roll()


def _write(path, obj):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True); tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w") as f: json.dump(obj, f, indent=1)
        os.replace(tmp, path)
    except OSError: pass                                                    # a read-only disk: memory only


def home_data():
    if _HOME[0] is None: _HOME[0] = json.load(open("pgb/graph_pgb.json"))
    return _HOME[0]


def window_id(a):
    w = (a.get("window") or a.get("anchor") or "").strip()
    if not w: raise ValueError("give window=dnaA or anchor=<the window's anchor family>")
    if w.lower() == DNAA.lower(): return DNAA
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", w): raise ValueError("unknown window")
    return w


def facts_of(wid, region):
    with LOCK:
        if wid in FACTS: FACTS.move_to_end(wid); return FACTS[wid]
    D = home_data() if wid == DNAA else region(wid)
    F = facts(D, wid); h = hashlib.sha1((PROMPT_VERSION + MODEL + SYSTEM + user_msg(F, wid)).encode()).hexdigest()[:12]
    F.label = DNAA if wid == DNAA else D["meta"].get("anchor_label") or wid
    with LOCK:
        FACTS[wid] = (F, h)
        while len(FACTS) > 64: FACTS.popitem(last=False)
    return F, h


def cached(wid, h):
    with LOCK:
        e = MEM.get(wid)
        if e is not None: MEM.move_to_end(wid)
    if e is None or e.get("facts_hash") != h:
        e = None
        for p in _paths(wid):
            try: x = json.load(open(p))
            except Exception: continue
            if x.get("facts_hash") == h: e = x; break
    if e is None: return None
    if e["status"] == "fallback" and time.time() - e["created"] > FAIL_TTL: return None
    with LOCK:
        MEM[wid] = e
        while len(MEM) > 256: MEM.popitem(last=False)
    return e


def availability():
    """-> None when new readings can be written here, else (code, why)"""
    if os.environ.get("READING", "").lower() in ("off", "0", "no", "false") or PER_IP <= 0 or PER_IP_DAY <= 0:
        return "off", "new readings are turned off on this server (READING=off)"
    if not token(): return "no_token", "no Hugging Face token on this server (set the HF_TOKEN secret)"
    if AUTH["why"] and time.time() - AUTH["t"] < AUTH_RETRY: return AUTH["code"], AUTH["why"]
    _roll()
    if USAGE["n"] >= DAILY_CAP: return "cap", f"the {DAILY_CAP} new readings of the day are written (until 00:00 UTC)"
    if USAGE["cost"] >= DAILY_USD: return "spend", f"the day's ${DAILY_USD:g} for new readings is spent (until 00:00 UTC)"
    return None


def _ip(x):
    """an X-Forwarded-For entry as an address: brackets and a port stripped; None if it is not one"""
    x = x.strip().strip('"')
    m = re.fullmatch(r"\[([0-9A-Fa-f:.]+)\](?::\d+)?", x)
    if m: x = m.group(1)
    elif re.fullmatch(r"[\d.]+:\d+", x): x = x.rsplit(":", 1)[0]
    try: return ipaddress.ip_address(x)
    except ValueError: return None


def client_ip(peer, xff):
    """the client, for its limits. Behind a trusted proxy (a Hugging Face Space, SPACE_ID set, or TRUST_PROXY=1): the
    rightmost public address of X-Forwarded-For (all its lines joined), the one the proxy appended; the walk stops at an
    entry that is not an address, which is then the key (a client can only prepend). Else the peer. IPv6 by /64."""
    key = None
    if TRUST_PROXY and xff:
        for x in reversed([s for s in xff.split(",") if s.strip()]):
            ip = _ip(x)
            if ip is None: key = "?" + x.strip()[:64]; break
            if ip.is_private or ip.is_loopback or ip.is_link_local: continue
            key = ip; break
    if key is None: key = _ip(peer or "") or (peer or "?")
    if isinstance(key, ipaddress.IPv6Address) and not key.ipv4_mapped: return str(ipaddress.ip_network(f"{key}/64", strict=False))
    if isinstance(key, ipaddress.IPv6Address): return str(key.ipv4_mapped)
    return str(key)


def site_ok(sfs, origin, host):
    """a new reading only for the page's own site: Sec-Fetch-Site same-origin (or none: typed in), an Origin of this
    host; locally (not a Space) also a page on localhost or opened as a file (Origin null). A request with no browser
    metadata (a script) passes: it has the same limits. Cached readings (peek, or written already) go to anyone."""
    sfs = (sfs or "").strip().lower(); origin = (origin or "").strip()
    if sfs in ("same-origin", "none"): return True
    if origin:
        if origin == "null": return not SPACE
        o = urllib.parse.urlparse(origin).netloc.lower()
        if host and o == host.strip().lower(): return True
        return not SPACE and bool(re.fullmatch(r"(localhost|127\.0\.0\.1|\[::1\])(:\d+)?", o))
    return sfs not in ("cross-site", "same-site")


def _prune(now):
    """forget the clients with no new reading in the last PER_IP_S (under LOCK)"""
    if now - PRUNED[0] < 60: return
    PRUNED[0] = now
    for k in [k for k, q in HITS.items() if not q or now - q[-1] > PER_IP_S]: del HITS[k]


def _admit(ip, commit):
    """may this client have a new generation now? commit: count it, for the client and the day"""
    now = time.time()
    with LOCK:
        why = availability()
        if why: raise Unavailable(why[1], _to_midnight() if why[0] in ("cap", "spend") else None, why[0])
        _prune(now)
        q = HITS.get(ip)
        while q and now - q[0] > PER_IP_S: q.popleft()
        if q and len(q) >= PER_IP:
            raise Limited(f"{PER_IP} new readings per {PER_IP_S / 60:.0f} minutes from one visitor", int(PER_IP_S - (now - q[0])) + 1, "limit")
        d = DAYHITS.get(ip, 0)
        if d >= PER_IP_DAY: raise Limited(f"{PER_IP_DAY} new readings a day from one visitor", _to_midnight(), "limit_day")
        if commit:
            HITS.setdefault(ip, collections.deque()).append(now); DAYHITS[ip] = d + 1; USAGE["n"] += 1


def _refund(ip):
    """a generation that could not be made: the client may ask again (the day's count and dollars stay)"""
    with LOCK:
        q = HITS.get(ip)
        if q: q.pop()
        if DAYHITS.get(ip): DAYHITS[ip] -= 1


def _slot():
    """one of the MAX_PARALLEL generation slots: a free one at once, else wait (at most QUEUE_MAX wait, SLOT_WAIT each)"""
    if SLOTS.acquire(blocking=False): return
    with LOCK:
        if QUEUED[0] >= QUEUE_MAX: raise Busy("other readings are being written", 10)
        QUEUED[0] += 1
    try: ok = SLOTS.acquire(timeout=SLOT_WAIT)
    finally:
        with LOCK: QUEUED[0] -= 1
    if not ok: raise Busy("other readings are being written", 5)


def public(e, wid, F, cached_, with_facts=False):
    out = dict(window=wid, label=F.label, status=e["status"], model=e["model"], provider=e.get("provider"), cached=cached_,
               created=datetime.datetime.fromtimestamp(e["created"], datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               checked=e["checked"], usage=dict(e["usage"]))
    if e["status"] == "llm": out.update(html=e["html"], text=e["text"])
    else: out["reason"] = e.get("reason", "")
    if with_facts: out["facts"] = F.txt()
    return out


def answer(a, peer="", xff="", region=None, site=True):
    """-> (HTTP code, JSON) for GET /reading?window=dnaA | anchor=<family> [&peek=1] [&facts=1]; raises a ReadingError
    (its status and body()) or ValueError / KeyError (400). site: the request comes from the page's own site"""
    t0 = time.time(); wid = window_id(a)
    try: F, h = facts_of(wid, region)
    except KeyError as e: raise ValueError(e.args[0] if e.args else "unknown window")
    wf = a.get("facts") == "1"
    done = lambda e, c: (200, dict(public(e, wid, F, c, wf), ms=round((time.time() - t0) * 1000)))
    e = cached(wid, h)
    if e: return done(e, True)
    if a.get("peek") == "1":
        with LOCK: why = availability()
        out = dict(window=wid, label=F.label, status="none", model=MODEL, available=why is None)
        if why: out.update(code=why[0], reason=why[1])
        if wf: out["facts"] = F.txt()
        return 200, out
    if not site: raise Refused("new readings are written only for the page of this server")
    ip = client_ip(peer, xff)
    with LOCK: u0 = (USAGE["n"], USAGE["calls"])
    for _ in range(3):                                                      # one generation per window at a time
        with LOCK:
            job = INFLIGHT.get(wid); mine = job is None
            if mine: job = INFLIGHT[wid] = dict(ev=threading.Event(), waiters=0, err=None)
            elif job["waiters"] >= WAITERS: raise Busy("this window's reading is being written", 5)
            else: job["waiters"] += 1
        if mine: break
        try: job["ev"].wait(BUDGET + SLOT_WAIT + 5)
        finally:
            with LOCK: job["waiters"] -= 1
        e = cached(wid, h)
        if e: return done(e, True)
        err = job["err"]
        if isinstance(err, (Unavailable, Upstream)): raise type(err)(str(err), err.retry_s, err.code)
        # the first request was refused for itself (its limit, its site) or had to wait: this one tries on its own
    else: raise Busy("this window's reading is being written", 5)
    try:
        _admit(ip, False)
        _slot()
        try:
            _admit(ip, True)
            try: e = generate(F, wid)
            except (Upstream, Unavailable): _refund(ip); raise
        finally: SLOTS.release()
        e.update(window=wid, label=F.label, facts_hash=h, prompt_version=PROMPT_VERSION, facts=F.txt())
        with LOCK:
            MEM[wid] = e
            while len(MEM) > 256: MEM.popitem(last=False)
        _write(_paths(wid)[0], e)
        return done(e, False)
    except ReadingError as x:
        job["err"] = x; raise
    finally:
        with LOCK:
            INFLIGHT.pop(wid, None); u = dict(USAGE)
        job["ev"].set()
        if (u["n"], u["calls"]) != u0: _write(os.path.join(CACHE_DIR, "_usage.json"), u)


def health():
    with LOCK:
        why = availability(); n = len(MEM); clients = len(HITS); u = dict(USAGE)
    nf = set()
    for d in {CACHE_DIR, BUNDLED}:
        try: nf |= {f for f in os.listdir(d) if f.endswith(".json") and not f.startswith("_")}
        except OSError: pass
    out = dict(available=why is None, model=MODEL, providers=PROVIDERS, max_tokens=MAX_TOKENS, cached=max(n, len(nf)),
               today=dict(new=u["n"], cap=DAILY_CAP, calls=u["calls"], cost_usd=round(u["cost"], 4), usd_cap=DAILY_USD),
               per_client=dict(new=PER_IP, per_s=int(PER_IP_S), per_day=PER_IP_DAY, clients=clients, proxy=TRUST_PROXY))
    if why: out.update(reason=why[1], code=why[0])
    return out


_usage_load()


def pregen(args):
    """write the readings of windows ahead, without the server (python3 pg_reading.py pregen ...)"""
    import concurrent.futures as cf
    import pgb_region as RG
    dry = "--dry" in args; ws = [a for a in args if not a.startswith("--")]
    if "--chain" in args:                     # the page's Next from the dnaA window, then Next again; Previous from each
        a = RG.resolve("60"); seen = []
        while a is not None and a not in seen:
            seen.append(a); _, nx = RG.neighbours(a, 60); a = RG.FIDX[nx["fam"]] if nx else None
        prev = [RG.FIDX[p["fam"]] for p in (RG.neighbours(a, 60)[0] for a in seen) if p and p["pos"] != 0]
        ws += [RG.FN[a] for a in seen] + [RG.FN[a] for a in prev if a not in seen]
    ws = list(dict.fromkeys(ws))
    region = lambda w: RG.build(RG.FIDX[w])
    todo = []
    for w in ws:
        wid = DNAA if w.lower() == DNAA.lower() else w
        if wid != DNAA and RG.FIDX.get(wid) not in RG.BB_RANK: print(f"{w}: not a window anchor", flush=True); continue
        F, h = facts_of(wid, region)
        if cached(wid, h): continue
        todo.append((wid, F, h))
    print(f"{len(ws)} windows, {len(todo)} to write", flush=True)
    if dry: return
    tot = dict(cost=0.0, n=0, fb=0)
    def one(x):
        wid, F, h = x; t = time.time()
        try: e = generate(F, wid)
        except ReadingError as err: return wid, f"error {err.code}: {err}", None
        e.update(window=wid, label=F.label, facts_hash=h, prompt_version=PROMPT_VERSION, facts=F.txt())
        _write(_paths(wid)[0], e)
        return wid, f"{e['status']} {e['checked']['attempts']} attempt(s) ${e['usage']['cost_usd']:.5f} {time.time() - t:.1f}s", e
    with cf.ThreadPoolExecutor(MAX_PARALLEL) as ex:
        for wid, msg, e in ex.map(one, todo):
            if e: tot["cost"] += e["usage"]["cost_usd"]; tot["n"] += 1; tot["fb"] += e["status"] == "fallback"
            print(wid, msg, flush=True)
    print(f"written {tot['n']} ({tot['fb']} fallback), ${tot['cost']:.4f}; calls counted ${USAGE['cost']:.4f}", flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "facts":
        w = sys.argv[2]; D = home_data() if w == DNAA else json.load(open(w))
        F = facts(D, DNAA if w == DNAA else D["meta"]["anchor"]); t = F.txt()
        print(t); print(f"\n[{len(t)} characters, ~{len(t) // 4} tokens; {sum(map(len, F.nums.values()))} numbers, {len(F.genes)} genes]", file=sys.stderr)
    elif len(sys.argv) >= 4 and sys.argv[1] == "check":
        w = sys.argv[2]; D = home_data() if w == DNAA else json.load(open(w))
        F = facts(D, DNAA if w == DNAA else D["meta"]["anchor"])
        print(json.dumps(check(tidy(open(sys.argv[3]).read(), F), F), indent=1))
    elif len(sys.argv) >= 2 and sys.argv[1] == "pregen": pregen(sys.argv[2:])
    else: print(__doc__)
