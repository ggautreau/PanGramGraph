"""The reading of a window written by an LLM, for the Findings tab of the PanGramGraph page (serve_live.py /reading).

The page writes a rule-based reading of the window shown (autoReading in page_template.html). Here an LLM writes it
in prose, by a skill document (reading_skill.md: what the numbers mean, a procedure, the hand-written reading of the
dnaA window as its reference), in the same five sections for every window, and every sentence names the facts lines it
is written from: each one is checked against those lines before the page shows it, and dropped if it does not hold.

    the browser sends only the window's id: dnaA, or the anchor family of a region window (pgb_region.build; a backbone
    family). The server builds the facts itself from its own data (facts(D)): the columns, their stretches (spine,
    fork, variable, the same rules as the page's autoReading), the stretch most and least in regions of plasticity,
    the genes and their products, the model's accuracy and hesitation, all as display-ready numbers: 2,000-2,900
    tokens, after a system message of ~2,400 (the skill, ~1,800, included). No text of the client reaches the LLM.
    skill: reading_skill.md beside this file (READING_SKILL for another path), read once and put in the system message:
    what the pangenome words mean (persistent, shell, cloud, regions of plasticity, spots, spine, fork, branch), what
    the model's three numbers mean and do not mean (named, bits, perplexity), what each section says, the procedure to
    follow in order (the gene groups and the one or two interpretations included), and two reference readings in the
    output form: the hand-written one of the dnaA window and one of a window nearly all spine. Its sha is part of the
    cache key, so a changed skill makes new readings; without the file the prompt keeps the hand-written reading alone.
    form: the same five paragraphs in the same order for every window, each opening with its tag and a short bold
    sentence: @start @variable @model @end @whole (SECTIONS), 150 to 200 words asked (WORDS_TARGET), at most 220
    served (WORDS_MAX), temperature 0.
    citations: every facts line carries its id ([W], [G], [C], [M], [S3], [S3.c12], [S3.b40], [Q]) and every sentence
    ends with the ids of the lines it is written from, in braces. The check is per sentence, against those lines only
    (stricter than a check over the whole facts): every number must be one they display, in the right role (a share, a
    share in a region of plasticity and a share named are not exchangeable), and bound to what it is said of: a gene's
    share to that gene, a column's to that column, a range to the lowest and highest of the columns named (or a range a
    line shows), "from A to B" to the first and last column, "most hesitant" / "fewest named" to the column and value
    the cited line states, [G]'s and [C]'s numbers to their own places. Every gene name, identifier and column must stand
    in them and sit where the sentence puts it; the kind of a column (spine, fork, variable) and the partition of a family
    as the facts give them; a branch spoken of with its own families and share. Every word inside [[ ]] must come from
    the products and gene names of the lines cited and the label must fit most of the families at the columns it names
    (in one operon, two of them); outside [[ ]] no product word. [M] and [C] in their direction; no "readily", no
    unstated trend ("fewer and fewer"), no "then" pointing back, no "lack", no unstated superlative. A sentence whose
    facts are not in its citations is dropped, a sentence repeating an earlier one is left out, and the model is asked
    once again for the whole reading with the offending sentences named (two dropped, a paragraph without its opening
    sentence, or no interpretation where [Q] offers one); if too little is left (a missing @start or @whole, two missing
    sections, under 110 words) the page keeps its rule-based reading (status "fallback").
    [Q], the groups: the server finds where an interpretation may stand (genes sharing a prefix, a word shared along
    a run of products, a branch whose families agree) and the words their products share; the model words the label.
    tidy(text) puts the model's slips of form right without a new call: "92-99%" -> "92 to 99%", "RGP 5%" -> "5% in a
    region of plasticity", "share 90%" -> "in 90% of chromosomes", a row of more than 6 genes -> `first` to `last`.
    length: over 220 words, whole sentences are dropped from the least telling sections, then a whole section; never a
    part of a sentence. The check verifies the numbers, the names and what the words inside [[ ]] rest on, not the prose.
    cache: per window, in memory and in <READING_CACHE>/<window>.json, also read from pgb/readings/ (the readings the
    Space carries), keyed by a hash of the facts, the prompt and the model: new data or a new prompt make a new
    reading. Limits on new generations: per client (X-Forwarded-For only behind a trusted proxy), per client per day,
    per day for the whole server (count and dollars), one generation per window at a time, two at once (others wait
    their turn in a short queue), only for requests from the page's own site (Sec-Fetch-Site / Origin).

Model: deepseek-ai/DeepSeek-V4.1-Flash through Hugging Face Inference Providers (router.huggingface.co, OpenAI
compatible), no reasoning (chat_template_kwargs thinking=false), at most 800 tokens out (median 410 per attempt over
the 112 readings of prompt r9, 460 at most). Token: HF_TOKEN (the
Space's secret), else the token of `huggingface-cli login`; used only in the request's header (never followed
through a redirect), never logged, never written, never sent to the page; the router's error text is not passed on.

    READING=off                disables new readings (cached ones are still served; /health says so)
    READING_MODEL, READING_PROVIDERS=deepinfra,novita   the model, and the providers tried in that order
    READING_SKILL=reading_skill.md  the skill document; READING_TEMPERATURE=0; READING_WORDS=220 the reading's length
    READING_PER_IP=6 READING_PER_IP_S=600 READING_PER_IP_DAY=30   new readings per client per 600 s, and per day
    READING_DAILY_CAP=150      new readings per UTC day, the whole server;  READING_DAILY_USD=0.30  and dollars per day
                               (every call counted, a call without usage or cut by a timeout at its worst case)
    READING_TIMEOUT=22         seconds per LLM call (then the next provider);  READING_QUEUE=6  generations waiting
    READING_CACHE=pgb/readings where new readings and the day's counter are written (e.g. /data/readings on a Space
                               with persistent storage); pgb/readings/ is read too
    TRUST_PROXY=1              read the client from X-Forwarded-For (always on a Space: SPACE_ID)

    python3 pg_reading.py facts dnaA | <window.json>          the facts of a window (a region: pgb_region.build output)
    python3 pg_reading.py check dnaA | <window.json> <text.md>   the checker on a text with its tags and citations,
        sentence by sentence, against that window's facts, and the reading it leaves
    python3 pg_reading.py skill                               the skill document read, its sha and its size
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
TEMPERATURE = _env("READING_TEMPERATURE", "0")   # 0: the same facts and the same skill give the same reading
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
PROMPT_VERSION = "r9"            # r9: the skill v2, [M] by kind of column, [Q] the groups, numbers bound to their genes and columns
WORDS_MAX = _env("READING_WORDS", "220", int)    # the length served: whole sentences are dropped to fit under it
WORDS_TARGET = _env("READING_TARGET", "200", int)  # the length asked of the model: below the limit, so nothing has to be cut
WORDS_MIN_ASK = 150             # the shortest reading asked for
WORDS_MIN = 110                  # what must be left after the drops for the reading to be served
SECTIONS = ("start", "variable", "model", "end", "whole")        # the sections, in this order, of every reading
KEEP = ("start", "whole")        # the sections a reading cannot lose
DROP_ORDER = ("variable", "model", "start", "end", "whole")      # whose last sentence goes first when the text is too long
CITES_MAX = 8                    # facts lines one sentence may cite (a stretch and the columns it walks)
SKILL_PATH = os.environ.get("READING_SKILL") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "reading_skill.md")
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
    line's id and columns and stretch, the columns of each gene, the columns stated most hesitant and fewest named, the
    stretches most and least in regions of plasticity, the inside/outside comparison. Each line is shown with its id
    ([W], [S3], [S3.c12]): a sentence of the reading cites the ids of the lines it is written from."""
    def __init__(self):
        self.ids = []
        self.lines = []; self.info = []; self.nums = collections.defaultdict(set); self.genes = set(); self.frags = set()
        self.at = collections.defaultdict(set); self.atp = collections.defaultdict(set); self.hes_cols = set(); self.low_cols = set(); self.plastic = {}
        self.comp = None; self.unit = "genomes"; self.label = ""
        self.pend = []; self.pshow = set(); self.pwords = set(); self.pnames = set(); self.kind = None; self.hes_spine = None
        self.gshare = collections.defaultdict(set); self.pshare = collections.defaultdict(set); self.gtop = None; self.gbits = None
    def reg(self, s, kind):
        v = Decimal(s.rstrip("%").replace(",", "")); self.nums[kind].add(v); return s
    def pct(self, v): return self.reg(fmt_pct(v), "pct")
    def bits(self, v): return self.reg(fmt_bits(v), "bits")
    def fam(self, v): return self.reg(fmt_fam(v), "fam")
    def int(self, v): return self.reg(fmt_int(v), "int")
    def fmly(self, n, k=None, alt=False):
        """a family of the line being built: its gene name and the words of its whole product, truncated or not shown
        or not — the scope of an interpretation is judged over the families of a line, each with what it really is
        (alt: a second family at that column, not what the stretch is made of)"""
        self.pend.append(dict(label=n.get("label") or "", col=k, words=pwords(n), alt=alt, part=PART.get(n.get("partition"))))
        self.pwords |= pwords(n, gene=False)
        return n
    def shown(self, s):
        """a product string the facts show: its words in full, so a product cut short ("…methy…") still carries them"""
        self.pshow |= {w for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", s or "")}
        return s
    def name(self, n, short=40, k=None, alt=False):
        """a family as the facts name it: its gene name, else its product in quotes (k: its column, for the checker)"""
        self.fmly(n, k, alt)
        if n.get("named"):
            self.genes.add(n["label"])
            if k is not None: self.at[n["label"]].add(k)
            return n["label"]
        lab = n.get("product") or n.get("label") or "unnamed family"
        if re.fullmatch(r"[A-Z0-9]+_RS\d+", lab): lab = "unnamed family"
        self.pnames |= {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", lab)}   # a family the facts name by its product:

        if k is not None: self.atp[lab.lower()].add(k); self.atp[self.text(lab, short).lower()].add(k)
        return '"' + self.text(lab, short) + '"'
    def text(self, s, short=48):
        """a product: shortened; its words followed by a number ("family 4") are names, not numbers, in a text"""
        self.shown(s)                                                   # its words in full: the cut is only in the display
        s = s if len(s) <= short else s[:short - 1].rstrip() + "…"
        for m in re.finditer(r"([A-Za-z][\w'-]*)\s+(\d+)\b(?!\.\d)", s): self.frags.add(f"{m.group(1)} {m.group(2)}".lower())
        return s
    def add(self, s, cols=(), head=None, glob=False, id=None):
        """a line; id: the id the reading cites it by; cols: the columns it is about; head: the index of its stretch's
        line; glob: about the whole window"""
        self.ids.append(id or f"F{len(self.lines)}")
        fams, show = self.pend, self.pshow; self.pend = []; self.pshow = set()
        self.lines.append(s); self.info.append(dict(cols=set(cols), head=head, glob=glob, fams=fams, show=show))
        return len(self.lines) - 1
    def txt(self): return "\n".join(f"[{i}] {s}" for i, s in zip(self.ids, self.lines))
    def index(self): return {i: k for k, i in enumerate(self.ids)}


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


def pwords(n, gene=True):
    """the words of a family: its whole product and its gene name, split on hyphens, slashes and apostrophes too
    ("pseudouridine-5'-phosphate glycosidase" gives phosphate and glycosidase), lowercase, the generic ones dropped"""
    out = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z]{2,}", n.get("product") or "")}
    out |= {w.lower() for w in re.findall(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b", n.get("product") or "") if re.search(r"[A-Za-z]", w)}
    if gene and n.get("named") and n.get("label"): out.add(n["label"].lower())
    return out - GENERIC


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


QSTOP = GENERIC | set("""binding atp dna abc duf cassette domain-containing hypothetical family protein subunit class enzyme
terminal substrate complex acid dependent dehydro deoxy phospho beta alpha gamma delta atp-binding substrate-binding
nad-binding fad-binding dna-binding site""".split())
def groups_line(F, cols, secs, BR):
    """[Q]: the gene groups whose products agree, found by the server, where a [[ ]] may stand: a run of columns whose
    most carried genes share a prefix (an operon: leu, cai, fli; waa and rfa are one), a run of columns whose products
    share a word though their names differ (flagellar: fli, flh, mot), a branch whose families share one. Each with the
    words most of its families' products carry (in an operon, a third of them and two at least): the words a label
    may use and still fit the families at those columns."""
    sid = {c["k"]: f"S{i}" for i, sec in enumerate(secs, 1) for c in sec["cols"]}
    def agree(ns, frac):
        cnt = collections.Counter(w for n in ns for w in sorted(pwords(n, gene=False) - QSTOP) if len(w) >= 4)
        return [w for w, v in sorted(cnt.items(), key=lambda x: (-x[1], x[0])) if v >= 2 and v >= frac * len(ns)][:4]   # ties: a-z
    lab = lambda n: n["label"] if n.get("named") else None
    out = []
    run = []                                                          # 1. operons: consecutive genes sharing a prefix
    for c in cols + [None]:
        n = c["top"] if c else None
        g = lab(n) if n else None
        pre = g[:3].lower() if g and re.fullmatch(r"[a-z]{3}[A-Z]\w*", g) and not g.startswith("y") else None
        pre = {"rfa": "waa"}.get(pre, pre)
        if run and pre and pre == run[-1][2] and c["k"] == run[-1][0] + 1: run.append((c["k"], n, pre)); continue
        if len(run) >= 3:
            ws = agree([x[1] for x in run], 1 / 3)
            if ws: out.append(dict(a=run[0][0], b=run[-1][0], first=lab(run[0][1]), last=lab(run[-1][1]), words=ws,
                                   ids=sorted({sid[x[0]] for x in run}, key=lambda x: int(x[1:]))))
        run = [(c["k"], n, pre)] if pre else []
    taken = set().union(set(), *(set(range(g["a"], g["b"] + 1)) for g in out))
    cnt = collections.Counter(w for c in cols for w in sorted(pwords(c["top"], gene=False) - QSTOP) if len(w) >= 5)
    for w, v in sorted(cnt.items(), key=lambda x: (-x[1], x[0]))[:8]:  # 2. a word shared along a run of columns (ties: a-z,
                                                                      # never the set's order, which changes with the process)
        if v < 3: break
        hits = [c["k"] for c in cols if w in pwords(c["top"], gene=False)]
        runs_ = [[hits[0]]]
        for k in hits[1:]:
            if k - runs_[-1][-1] <= 2: runs_[-1].append(k)
            else: runs_.append([k])
        for r in runs_:
            a, b = r[0], r[-1]
            if len(r) < 3 or len(r) < 0.6 * (b - a + 1) or len(set(range(a, b + 1)) & taken) * 2 > b - a + 1: continue
            ns = [c["top"] for c in cols if a <= c["k"] <= b]
            ws = [w] + [x for x in agree(ns, 0.5) if x != w][:3]
            first, last = next(c["top"] for c in cols if c["k"] == a), next(c["top"] for c in cols if c["k"] == b)
            out.append(dict(a=a, b=b, first=lab(first), last=lab(last), words=ws, ids=sorted({sid[k] for k in range(a, b + 1) if k in sid},
                                                                                         key=lambda x: int(x[1:]))))
            taken |= set(range(a, b + 1))
    for bid, bb in BR:                                                # 3. a branch whose families agree
        ws = agree([n for _, n in bb], 0.5)
        if ws: out.append(dict(a=bb[0][0], b=bb[-1][0], branch=bid, words=ws, ids=[bid]))
    out = [g for g in out if not any(h is not g and h["a"] <= g["a"] and g["b"] <= h["b"] and not g.get("branch")
                                     and set(h["words"]) & set(g["words"]) for h in out)]   # a group inside a larger one: one
    if not out: return
    out = sorted(sorted(out, key=lambda g: -(g["b"] - g["a"]))[:5], key=lambda g: g["a"])
    def say(g):
        where = f"columns {F.int(g['a'])}-{F.int(g['b'])}"
        if g.get("branch"): return f"the branch [{g['branch']}] through {where}: {', '.join(g['words'])}"
        who = f" ({g['first']} to {g['last']})" if g.get("first") and g.get("last") else ""
        return f"{where}{who} [{', '.join(g['ids'])}]: {', '.join(g['words'])}"
    F.add("GROUPS whose products agree, the places for a [[ ]] (name that group's columns exactly, cite its line and [Q], make "
          "one phrase from its words): " + "; ".join(say(g) for g in out) + ".", glob=True, id="Q")


def facts(D, wid):
    """the facts of one window (graph data: pgb/graph_pgb.json for dnaA, pgb_region.build for the others)"""
    F = Facts(); m = D["meta"]; W = m["window"]; NG = m["n_genomes"]; home = wid == DNAA
    cols, runs = columns(D); unit = "genomes" if home else "chromosomes"; F.unit = unit
    F.kinds = {c["k"]: c["kind"] for c in cols}
    for c in cols:                                                        # each gene's share at the columns it is most carried at
        if c["top"].get("named"): F.gshare[c["top"]["label"]].add(Decimal(fmt_pct(c["s1"]).rstrip("%")))
    if home:
        full = m.get("full_windows")
        F.add(f"WINDOW: the dnaA window: {F.int(W)} columns from dnaA (column {F.int(0)}) in its direction, in all {F.int(NG)} genomes "
              f"of the PanGBank E. coli pangenome; {F.int(m['calls'])} calls of the model, each on a genome's real proteins."
              + (f" {F.int(full)} of the {F.int(NG)} genomes have all {F.int(W)} columns on one contig; in the others the window "
                 f"stops earlier, at a contig end." if full else ""), glob=True, id="W")
    else:
        a = m["anchor_label"]
        if re.fullmatch(r"[a-z]{3}[A-Z]?\w*", a or ""): F.genes.add(a); F.at[a].add(0)
        pos = m.get("anchor_pos")
        where = f"; about genes {F.int(pos)} to {F.int(pos + W - 1)} from dnaA" if pos is not None else ""
        F.add(f"WINDOW: the {a} window: {F.int(W)} columns, column {F.int(0)} is the anchor {a}, then the next {F.int(W - 1)} genes "
              f"in dnaA's direction{where}. Read in {F.int(NG)} of the {F.int(m['genomes_total'])} complete chromosomes (those "
              f"carrying {a} once); {F.int(m['calls'])} calls of the model, each on a chromosome's real proteins.", glob=True, id="W")
    one = unit[:-1]
    F.add(f"TERMS: share: {unit} carrying the family in the window. named: calls where the model names the gene at that column "
          f"from the genes before it. bits: the model's entropy there, its hesitation (the more bits, the more hesitant). "
          f"perplexity (whole window only): how surprised the model is by the real gene. RGP N%: N% of the calls there are on a "
          f"gene inside a region of genomic plasticity (panRGP). P/S/C: persistent, shell, cloud (PPanGGOLiN). Spine column: "
          f"one family in {F.pct(0.9)} of {unit} or more; fork: a second family in {F.pct(0.1)} or more; variable: neither. A "
          f"persistent family is in nearly every {one}: in a fork or variable column its lower share means it lies beyond the "
          f"window's end in the other {unit} (genes inserted upstream push it out{', or a contig ends' if home else ''}); never "
          f"call it rare or say they lack it.", glob=True, id="T")
    pl = D["per_locus"]; n_all = sum(r["n"] for r in pl) or 1
    ent = sum(r["entropy"] * r["n"] for r in pl) / n_all
    part, rg = D.get("by_partition") or {}, D.get("by_rgp") or {}
    t = lambda r: f"{F.int(r['n'])} calls, named {F.pct(r['top1_dec'])}, {F.bits(r['entropy'])} bits" if r else "no calls"
    nk = collections.Counter(c["kind"] for c in cols)
    F.gtop = Decimal(fmt_pct(m["top1_dec"]).rstrip("%")); F.gbits = Decimal(fmt_bits(ent))
    F.add(f"WHOLE WINDOW: the model names {F.pct(m['top1_dec'])} of genes, {F.bits(ent)} bits on average (perplexity "
          f"{F.fam(m['ppl'])} families). By partition of the gene: persistent {t(part.get('persistent'))}; shell "
          f"{t(part.get('shell'))}; cloud {t(part.get('cloud'))}. Outside an RGP {t(rg.get('outside an RGP'))}; inside an "
          f"RGP {t(rg.get('inside an RGP'))}. Columns: {F.int(nk['spine'])} spine, {F.int(nk['fork'])} fork, "
          f"{F.int(nk['variable'])} variable.", glob=True, id="G")
    RI, RO = rg.get("inside an RGP"), rg.get("outside an RGP")
    if RI and RO and RI["n"] and RO["n"]:
        e_i, e_o, a_i, a_o = RI["entropy"], RO["entropy"], RI["top1_dec"], RO["top1_dec"]
        h = "hesitates more" if e_i > 1.2 * e_o + 0.02 else "hesitates less" if e_i < 0.8 * e_o - 0.02 else "hesitates about as much"
        n_ = "names fewer genes" if a_i < a_o - 0.03 else "names more genes" if a_i > a_o + 0.03 else "names about as many genes"
        F.comp = dict(h=h, n=n_, ei=F.bits(e_i), eo=F.bits(e_o), ai=F.pct(a_i), ao=F.pct(a_o))
        F.add(f"COMPARISON (computed, over calls): inside an RGP the model {h} than outside ({F.comp['ei']} against {F.comp['eo']} bits) "
              f"and {n_} ({F.comp['ai']} against {F.comp['ao']}); {F.int(RI['n'])} of the {F.int(RI['n'] + RO['n'])} calls are inside an RGP.", glob=True, id="C")
    by_k = {c["k"]: c for c in cols}
    def mean(rs, key):
        n = sum(r["n"] for r in rs); return sum(r[key] * r["n"] for r in rs) / n if n else None
    hot = [r for r in pl if r["locus"] in by_k and by_k[r["locus"]]["rgp"] >= 0.5]
    cold = [r for r in pl if r["locus"] in by_k and by_k[r["locus"]]["rgp"] < 0.5]
    top = [r for r in sorted(pl, key=lambda r: -r["entropy"])[:3] if r["locus"] in by_k]
    if pl: F.low_cols.add(min(pl, key=lambda r: r["top1_dec"])["locus"])       # the window's fewest named: its value is in the facts
    F.hes_cols |= {r["locus"] for r in top}
    hs = "; ".join(f"column {F.int(r['locus'])} ({F.name(by_k[r['locus']]['top'], k=r['locus'])}, a {by_k[r['locus']]['kind']} column) "
                   f"{F.bits(r['entropy'])} bits, RGP {F.pct(by_k[r['locus']]['rgp'])}" for r in top)
    def where(rs):
        """the columns of a short list, as the facts write columns ("columns 9, 36 and 40"; a run as "12-17")"""
        ks = sorted(r["locus"] for r in rs); runs_ = []
        for k in ks:
            if runs_ and k == runs_[-1][1] + 1: runs_[-1][1] = k
            else: runs_.append([k, k])
        if len(runs_) > 6: return ""
        it = [F.int(a) if a == b else f"{F.int(a)}-{F.int(b)}" for a, b in runs_]
        return " (column " + it[0] + ")" if len(it) == 1 and "-" not in it[0] else \
               " (columns " + (", ".join(it[:-1]) + " and " + it[-1] if len(it) > 1 else it[0]) + ")"
    F.add(f"HESITATION: the most hesitant columns of the window: {hs}. "
          + (f"The {F.int(len(hot))} column{'' if len(hot) == 1 else 's'}{where(hot)} where most calls are in an RGP: named "
             f"{F.pct(mean(hot, 'top1_dec'))}, {F.bits(mean(hot, 'entropy'))} bits on average; the other {F.int(len(cold))} columns: "
             f"named {F.pct(mean(cold, 'top1_dec'))}, {F.bits(mean(cold, 'entropy'))} bits."
             if hot and cold else "No column has most of its calls in an RGP." if not hot else "Every column has most of its calls in an RGP."),
          glob=True, id="H")
    sp = [r for r in pl if r["locus"] in by_k and by_k[r["locus"]]["kind"] == "spine"]
    ot = [r for r in pl if r["locus"] in by_k and by_k[r["locus"]]["kind"] != "spine"]
    if top: F.hes_spine = by_k[top[0]["locus"]]["kind"] == "spine"
    if sp and ot:                                # what the @model paragraph is about: the two kinds of column compared
        a_s, e_s, a_o, e_o = mean(sp, "top1_dec"), mean(sp, "entropy"), mean(ot, "top1_dec"), mean(ot, "entropy")
        F.kind = dict(named_spine=a_s, named_other=a_o, bits_spine=e_s, bits_other=e_o)
        nm = ("more genes on the spine columns than on" if a_s > a_o + 0.02 else "fewer genes on the spine columns than on"
              if a_s < a_o - 0.02 else "about as many genes on the spine columns as on")
        hz = ("hesitates less there" if e_s < e_o - 0.02 else "hesitates more there" if e_s > e_o + 0.02 else "hesitates about as much there")
        pl_ = lambda n, w: f"{F.int(n)} {w} column" + ("" if n == 1 else "s")
        kt = by_k[top[0]["locus"]]["kind"] if top else None
        F.add(f"BY KIND OF COLUMN (computed, over the columns with calls): the {pl_(len(sp), 'spine')}: named {F.pct(a_s)}, "
              f"{F.bits(e_s)} bits on average; the {pl_(len(ot), 'fork and variable')}: named {F.pct(a_o)}, {F.bits(e_o)} bits. "
              f"The model names {nm} the fork and variable ones, and {hz}."
              + (f" Its most hesitant column, {F.int(top[0]['locus'])}, is a {kt} column"
                 + (": the model can hesitate most inside a spine." if kt == "spine" else ".") if top else ""), glob=True, id="M")

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
    def fam_s(n, k, main=None, alt=False):
        same = " (another family)" if main is not None and n.get("named") and main.get("named") and n["label"] == main["label"] else ""
        sh = F.pct(n['n'] / NG); nm = F.name(n, k=k, alt=alt)
        if n.get("named"): F.gshare[n["label"]].add(Decimal(sh.rstrip("%")))
        else: F.pshare[nm.strip('"').rstrip("…").lower()].add(Decimal(sh.rstrip("%")))
        return f"{nm}{same} {PART.get(n['partition'], '?')} {sh}"
    def top_s(c):
        sh = F.pct(c["s1"])
        if c["top"].get("named"): F.gshare[c["top"]["label"]].add(Decimal(sh.rstrip("%")))
        return f"{F.name(c['top'], k=c['k'])}{prod(c['top'])} {sh}"

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
    secs = []; BR = []
    for r in runs:
        if r["kind"] != "spine" and secs and secs[-1]["kind"] == "open" and r["a"] == secs[-1]["b"] + 1:
            secs[-1]["b"] = r["b"]; secs[-1]["runs"].append(r)
        else: secs.append(dict(kind="spine" if r["kind"] == "spine" else "open", a=r["a"], b=r["b"], runs=[r]))
    for sec in secs:
        sec["cols"] = [c for r in sec["runs"] for c in r["cols"]]
        kinds = sorted({r["kind"] for r in sec["runs"]})
        sec["what"] = "spine" if sec["kind"] == "spine" else {("fork",): "fork", ("variable",): "variable"}.get(tuple(kinds), "fork and variable columns")
        sec["rgp_mean"] = sum(c["rgp"] for c in sec["cols"]) / len(sec["cols"])
    # the stretches most and least in regions of plasticity: the only ones to call so. Among three stretches or more, and
    # never one that covers most of the window ("the most plastic stretch" of 73 columns out of 80 says nothing)
    cand = [s for s in secs if len(s["cols"]) * 2 <= len(cols)] if len(secs) >= 3 else []
    if len(cand) >= 2:
        hi = max(cand, key=lambda s: s["rgp_mean"]); lo = min(cand, key=lambda s: s["rgp_mean"])
        if hi is not lo and hi["rgp_mean"] > lo["rgp_mean"] + 0.1:
            allp = min(c["rgp"] for c in lo["cols"]) >= 0.5           # nothing in the window is really outside: "least" says nothing
            F.plastic = dict(most=(hi["a"], hi["b"]), least=None if allp else (lo["a"], lo["b"]))
            F.add(f"PLASTICITY (the stretches below, by their mean over columns): the stretch most in regions of plasticity: {span(hi)} "
                  f"({hi['what']}), RGP {rng([c['rgp'] for c in hi['cols']], F.pct)}; "
                  + (f"every stretch of this window lies largely inside regions of plasticity (the lowest, {span(lo)}, RGP "
                     f"{rng([c['rgp'] for c in lo['cols']], F.pct)}): never call a stretch of it the least plastic."
                     if allp else f"the least: {span(lo)} ({lo['what']}), RGP {rng([c['rgp'] for c in lo['cols']], F.pct)}."), glob=True,
                  cols=set(range(hi["a"], hi["b"] + 1)) | set(range(lo["a"], lo["b"] + 1)), id="P")
    F.add(f"STRETCHES, in window order (each spine; between spines, the fork and variable columns; the family listed is the "
          f"most carried at its column; share of {unit}; RGP: range over the columns):", glob=True, id="L")
    for i, sec in enumerate(secs, 1):
        cs = sec["cols"]; rgp = rng([c["rgp"] for c in cs], F.pct); ks = [c["k"] for c in cs]
        if sec["kind"] == "spine":
            genes = ", ".join(F.name(c["top"], k=c["k"]) + prod(c["top"]) for c in cs)
            F.add(f"{i}. {spann(sec)} spine: share {rng([c['s1'] for c in cs], F.pct)}, {parts(cs)}, RGP {rgp}. Genes: {genes}. {model(cs)}.",
                  cols=ks, id=f"S{i}")
            continue
        h = F.add(f"{i}. {spann(sec)} {sec['what']}: share {rng([c['s1'] for c in cs], F.pct)}, {parts(cs)}, RGP {rgp}. "
                  f"{model(cs, listed=False)}.", cols=ks, id=f"S{i}")
        inb = set()
        for r in sec["runs"]:
            for bb in (branches(r["cols"]) if r["kind"] == "fork" else []):
                ns = [n for _, n in bb]; inb |= {id(n) for n in ns}; BR.append((f"S{i}.b{bb[0][0]}", bb))
                F.add(f"   a second branch of {F.int(len(ns))} families through columns {F.int(bb[0][0])}-{F.int(bb[-1][0])}, a parallel path "
                      f"that some {unit} carry instead of the most carried families there, each following the previous one in most of its "
                      f"{unit}, share {rng([n['n'] / NG for n in ns], F.pct)}, {parts([dict(top=n) for n in ns])}; the branch's families: "
                      + ", ".join(F.name(n, k=k, alt=True) for k, n in bb), cols=[k for k, _ in bb], head=h,
                      id=f"S{i}.b{bb[0][0]}")
        mark = (lambda r: f" {r['kind']}") if sec["what"] == "fork and variable columns" else (lambda r: "")
        for r in sec["runs"]:
            if r["kind"] == "variable" and len(r["cols"]) > 4:                   # a long variable run: one line
                F.add(f"   {span(r)}{mark(r)}, share {rng([c['s1'] for c in r['cols']], F.pct)}, {parts(r['cols'])}: "
                      + ", ".join(top_s(c) for c in r["cols"]),
                      cols=[c["k"] for c in r["cols"]], head=h, id=f"S{i}.c{r['cols'][0]['k']}-{r['cols'][-1]['k']}")
                continue
            for c in r["cols"]:
                al = [n for n in c["alts"][:3] if id(n) not in inb]
                alts = "; also " + ", ".join(fam_s(n, c["k"], c["top"], alt=True) for n in al) if al else ""
                F.add(f"   {F.int(c['k'])}{mark(r)}: {fam_s(c['top'], c['k'])}{prod(c['top'])}{alts}", cols=[c["k"]], head=h, id=f"S{i}.c{c['k']}")
    groups_line(F, cols, secs, BR)
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
    if items: F.add("PRODUCTS of named genes (for [[ ]]): " + "; ".join(items[:40]), glob=True, id="X")
    last = cols[-1] if cols else None
    if last: F.add(f"END: the window ends at column {F.int(last['k'])}, {F.name(last['top'], k=last['k'])}, in {F.pct(last['s1'])} of {unit}.",
                   cols=[last["k"]], glob=True, id="E")
    return F


# ---------------------------------------------------------------- the prompt
STYLE = """**The spine starts at the origin.** Columns 0 to 5, `dnaA`, `dnaN`, `recF`, `gyrB`, `yidB` and `yidA`, sit in essentially every genome, persistent, and hardly ever in a region of plasticity: 0 to 0.5% of calls. The model names `dnaN`, `recF` and `gyrB` in 99.7 to 100% of genomes, `yidB` in 93% and `yidA` in 99%.

**The first fork is at `yidX`**, column 6. Then the `dgo` operon, [[galactonate catabolism]], variable columns 7 to 11: about 85% of genomes carry it, and panRGP puts it in a region of plasticity in 90% of them. The model is not lost there: it names each gene in 87 to 99% of genomes, with hardly any hesitation. A variable region can keep a predictable internal order.

**Then a long persistent stretch**, `cbrA` to `yicS` through the `ilv` and `uhp` operons, where the model names the next gene in 92% of genomes and seldom hesitates. Around columns 42 to 52 the graph opens: 50% of the calls there fall in a region of plasticity, and the model's hesitation peaks, up to 1.4 bits. Some genomes carry [[type III secretion genes]] there, others [[sugar transport genes]].

**The window ends in the `waa` locus**, [[which builds the lipopolysaccharide core]]: `waaA` in most genomes, then a spread of glycosyltransferases."""

TAGS = dict(start="the first stretch", variable="the middle: fork and variable stretches, branches, gene groups",
            model="[M], then the most hesitant column of [H]", end="the last stretch and [E]", whole="[G], then [C]")
CONTRACT = f"""You write the reading of one window of an E. coli pangenome graph for a scientific web page, from facts the server computes for it: Bacformer, a genomic language model, reads chromosomes gene by gene and at each gene calls the next one; a window is 80 columns of genes. Follow the skill below.

FORM. Exactly {len(SECTIONS)} paragraphs in this order, one blank line apart, each opening with its tag and a short bold sentence in **double asterisks**: """ + "; ".join(f"@{t} ({w})" for t, w in TAGS.items()) + f""". Two sentences each (@variable may take three): {WORDS_MIN_ASK} to {WORDS_TARGET} words in all, at most 3 numbers a sentence. Beyond {WORDS_MAX} words whole sentences are cut.

CITATIONS. Each facts line begins with its id: [W], [G], [M], [S3], [S3.c12], [S4.b45]. Each sentence ends with the ids of the lines it is written from, in braces, before its full stop: "The model names 93% of them {{S1}}." Every number, gene name, product word and column of a sentence must stand in those lines, or the sentence is dropped. A stretch line [S4] gives its ranges and lets you name the genes, columns and shares of its sub-lines: for columns 15 to 20 of it, their lowest and highest share exactly, or "from A to B%", the first and the last. A branch: its own line and clause. Two or three ids a sentence, {CITES_MAX} at most.

RULES. 1. Numbers as the facts write them, in digits (bits may be rounded to 1 decimal); a range as "92 to 99%"; a share, a share in regions of plasticity and a share named are three different numbers. 2. Gene names in backticks; an operon by its prefix (`dgo`); more than 6 in a row as `first` to `last`; an unnamed family by its product, without backticks. 3. Biology only inside [[ ]]: 2 to 6 words from the products of the lines cited, true of most families at the columns named; outside the brackets no product word, function, pathway or cause, not even in bold. 4. Superlatives only as [P], [H] and the stretch lines state them. Write "in a region of plasticity in N% of calls", never "RGP", "share" or the letters P, S, C; spine, fork and variable as the facts label them. Never call a persistent family rare, never say chromosomes lack a gene. 5. [C] and [M] in their direction; never "readily", never "where the graph opens" unless [M] and [H] say so. 6. Plain, exact English; no headings, lists or other markup; nothing about yourself or the facts; no sentence of the reference readings, their windows are not yours."""

_SKILL = [None, None, None]
def skill():
    """the skill document (reading_skill.md beside this file, READING_SKILL for another path) and its sha, read once:
    the domain notes, the procedure and the reference reading the model works by. The sha is part of the cache key, so
    a changed skill makes new readings. Without the file the prompt keeps the reference reading alone."""
    if _SKILL[2] is None:
        try: t = open(SKILL_PATH, encoding="utf-8").read().strip()
        except OSError: t = ""
        _SKILL[2] = bool(t)
        if not t: t = ("# Reading a window of the pangenome graph — the skill document is not on this server\n\nThe reference "
                       "reading, the hand-written one of the dnaA window (its numbers and its biology belong to that window, "
                       f"not to yours):\n\n{STYLE}")
        _SKILL[0] = t; _SKILL[1] = hashlib.sha1(t.encode()).hexdigest()[:12]
    return _SKILL[0], _SKILL[1]


def system_msg():
    t, sha = skill()
    return f"{CONTRACT}\n\n{t}", sha


def user_msg(F, wid):
    extra = ("\n\nThis is the dnaA window, the one of the first reference reading: write your own reading from these facts; any "
             "number that is not in the facts is rejected.") if wid == DNAA else ""
    return (f"Facts, one line each with its id:\n{F.txt()}{extra}\n\nWrite the reading: "
            + " ".join(f"@{t}" for t in SECTIONS) + ", each sentence ending with the ids of the facts lines it is written from.")


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
META_RE = re.compile(r"\b(?:COMPARISON|HESITATION|STRETCHES|WHOLE WINDOW|TERMS|PLASTICITY|PRODUCTS|GROUPS)\b|"
                     r"\b(?i:the facts|as stated|products agree|whose products|the group of \[)")
UNIT_RE = re.compile(r"\s*(%|percent\b|per cent\b|-?bits?\b|famil(?:y|ies)\b|(?:complete\s+)?(?:genomes?|chromosomes?)\b)", re.I)
LOWER_RE = re.compile(r"\s*(?:to|-|–)\s*(\d[\d,]*(?:\.\d+)?)\s*(%|percent\b|per cent\b|bits?\b)", re.I)
IDENT_RE = re.compile(r"[\w'\-/.]*\d[\w'\-/.]*")
GENE_RE = re.compile(r"(?<![\w`])([a-z]{3}[A-Z][A-Za-z0-9_]*)(?![\w])")
PROT_RE = re.compile(r"(?<![\w`])([A-Z][a-z]{2}[A-Z][0-9]?)(?![\w])")
OPERON_RE = re.compile(r"(?<![\w`])([a-z]{3}[A-Z]?)\s+(?:operon|genes|locus|loci|cluster)\b")
WORDS3 = set("""the all its few new two six ten one any are old big odd set top end our key own net raw lac far low how who
why but not nor yet per via has had was can may use may six ten and for hold holds carry carries same both that this those
these more less than into onto over such very each many most some only then when where which while with from run runs""".split())
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
ONLY_RE = re.compile(r"\bonly (?:in |about |around )?(\d+(?:\.\d+)?)\s?%", re.I)   # "only 89%": "only" before a majority share
def _only_majority(text):
    """"only NN%" with NN >= 50: a majority called small (the checker drops it)"""
    return [dict(kind="claim", text=m.group(0), why="\u201conly\u201d before a majority share")
            for m in ONLY_RE.finditer(text) if float(m.group(1)) >= 50]
HES_RE = re.compile(r"most hesitant|hesitates? (?:the )?most|hesitation peaks|peak hesitation|hesitation is highest|highest hesitation", re.I)
LOW_RE = re.compile(r"\bfewest\b|\blowest at\b|least often named|names? (?:the )?fewest|named (?:the )?least", re.I)


READY_RE = re.compile(r"\breadily\b|\bwith ease\b|\beasily\b|\bwithout (?:trouble|difficulty|hesitation)\b|\bhas no trouble\b", re.I)
OPENS_RE = re.compile(r"where the graph opens|where it opens|where the graph is open|where chromosomes differ|where the order varies|"
                      r"\b(?:in|on|at|across) the (?:fork|variable|plastic|open)\w*|\b(?:in|on|at|across) the fork and variable\b", re.I)
PROSE = set("""region regions plasticity model column columns share shares named naming names hesitates hesitation hesitant bits
perplexity family families gene genes call calls chromosome chromosomes genome genomes persistent shell cloud spine spines fork
forks variable branch branches path paths parallel stretch stretches window windows anchor direction carried carries second first
most least other others unnamed average mean readily window's columns' opens""".split())
ALIAS = dict(waa=("rfa",), rfa=("waa",))
OPERON_RE2 = re.compile(r"`([a-z]{3})[A-Z]?[A-Za-z0-9]*`\s+(?:operon|genes|locus|loci|cluster|region)\b")


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
    for m in re.finditer(r"\bcolumn,\s+(\d+)\b(?![.,]\d|\s*(?:%|percent|bits?\b)|\s+to\b)", s, re.I): cs.add(int(m.group(1)))
    for m in re.finditer(r"\b(first|last) (\d+) columns\b", s, re.I):
        n = min(int(m.group(2)), W); cs |= set(range(0, n)) if m.group(1).lower() == "first" else set(range(W - n, W))
    for m in re.finditer(r"\bcolumns? ((?:\d+(?![.,]\d)(?:\s*(?:,|and|to|-|–)\s*(?=\d))?)+)", s, re.I):
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        seps = re.findall(r"\d+\s*(to|-|–|,|and)\s*(?=\d)", m.group(1))
        val = {int(x.group(0)) for x in re.finditer(r"\d+", m.group(1))       # "fewest at column 6, 72%", "columns 61 to 79, 4 to 77%":
               if re.match(r"\s*(?:(?:to|-|–)\s*\d[\d,]*(?:\.\d+)?\s*)?(?:%|percent|per cent|bits?\b|famil)",   # a value, not a column
                           s[m.start(1) + x.end():])}
        cs |= set(nums) - val
        for (a, b), sep in zip(zip(nums, nums[1:]), seps):
            if sep in ("to", "-", "–") and b >= a and a not in val and b not in val: cs |= set(range(a, b + 1))
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
    """the whole-text check (r7), kept for comparison with the per-sentence one (pg_reading.py check prints both, and it
    reads a text that carries no citations) -> dict(ok, numbers, names, words, bad=[...], hints=[...]): every number, gene
    name, number word and identifier with a digit of text, each number against the facts lines of its sentence's genes and
    columns, and the claims the facts make. Nothing ties a number to the line the sentence is written from: that is what
    check_cited does."""
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
    bad += _only_majority(re.sub(r"\[\[.*?\]\]", " ", text))
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
    C = r"columns?,? (\d+)(?:\s*(?:to|-|–)\s*(\d+))?"
    G = r"`([^`]+)`(?:\s+to\s+`([^`]+)`)?"
    pats = [(rf"\b{C},?\s+(?:\(|the\s+)?{G}", "cg"), (rf"{G},?\s+(?:\(|in |at ){C}", "gc")]
    for para in sentences(text.replace("**", "")):
        for s in para:
            for pat, order in pats:
                for m in re.finditer(pat, s, re.I):
                    if order == "cg":
                        a, b, g1, g2 = m.group(1), m.group(2), m.group(3), m.group(4)
                        if re.match(r"\s*,?\s*(?:at|in|through|across|over|column|columns)\s+(?:columns?\s+)?\d", s[m.end():]) or \
                           re.search(r"\b(?:between|into|from)\s+$", s[max(0, m.start() - 20): m.start()]): continue
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


# ---------------------------------------------------------------- the sections and the citations (one per sentence)
ROLE_KIND = dict(share="pct", rgp="pct", named="pct", definition="pct", bits="bits", ppl="fam", count="int")
ROLE_SAY = dict(share="a share of {unit}", rgp="a share in a region of plasticity", named="a share of genes named",
                bits="a hesitation in bits", ppl="a perplexity", count="a count", definition="a number of the TERMS line")
_FNUM = re.compile(r"(?<![\w.,])(\d[\d,]*(?:\.\d+)?)(?:\s*(?:to|-|–)\s*(\d[\d,]*(?:\.\d+)?))?\s*(%|bits?\b|famil(?:y|ies)\b)?", re.I)
_FROLE = ((re.compile(r"\bRGP\b"), "rgp"), (re.compile(r"\bnam(?:e|es|ed|ing)\b|\bmean\b", re.I), "named"),
          (re.compile(r"\bshare\b", re.I), "share"))
_SROLE = ((re.compile(r"regions?\s+of\s+plasticity|\bplastic\w*", re.I), "rgp"),
          (re.compile(r"\bnam(?:e|es|ed|ing)\b", re.I), "named"),
          (re.compile(r"\bcarr(?:y|ies|ied|ying)\b|\bshare\b", re.I), "share"),
          (re.compile(r"\bperplexity\b", re.I), "ppl"))
_FOLLOW = ((re.compile(r"\s*(?:of\s+(?:the\s+)?\w+\s+)?(?:are\s+|fall\s+|lie\s+|sit\s+|is\s+)?(?:in|inside|within)\s+"
                       r"(?:a\s+|any\s+)?regions?\s+of\s+plasticity", re.I), "rgp"),
           (re.compile(r"\s*of\s+(?:the\s+|all\s+)?(?:genomes|chromosomes|them)\b", re.I), "share"),
           (re.compile(r"\s*of\s+(?:the\s+)?genes\b", re.I), "named"))
_JOIN = re.compile(r"[\s(),]*(?:against|to|and|or|versus|vs\.?)?[\s(),]*")    # between two numbers of one pair
TAG_RE = re.compile(r"(?:(?<=\n)|(?<=\s))[\[(]?@(" + "|".join(SECTIONS) + r")\b[\])]?[:.]?[ \t]*")   # a tag anywhere after a space
CITE_RE = re.compile(r"\{([^{}\n]{0,160}?)\}[ \t]*([.;:!?]*)")
CID_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:\.[A-Za-z]?\d+(?:-\d+)?)?")


def _stem(w): return w[:7] if len(w) >= 7 else w


_SUF = sorted(("ases", "ase", "ysis", "oses", "osis", "ations", "ation", "isation", "ization", "ising", "izing", "ing",
               "ers", "er", "ors", "or", "ions", "ion", "ive", "ic", "al", "ed", "es", "s"), key=len, reverse=True)
def _norm(w):
    """a product word cut back to its stem: hydrolysis and glycosylhydrolase both give hydrol, secretion and secreted
    both secret, so a word of an interpretation is recognised in the product it comes from"""
    w = w.lower().strip("'-")
    for suf in _SUF:
        if w.endswith(suf) and len(w) - len(suf) >= 5: return w[:len(w) - len(suf)]
    return w


def _same(a, b):
    """two words of the same thing: equal, or one's stem inside the other (never two words that merely start alike,
    so transport does not match transposase)"""
    na, nb = _norm(a), _norm(b)
    return na == nb or (len(na) >= 6 and na in nb) or (len(nb) >= 6 and nb in na)


def _covers(w, pool): return any(_same(w, x) for x in pool)


def _role_left(s, rules):
    """the role the last keyword before a number gives it (None: no keyword)"""
    best, role = -1, None
    for rx, r in rules:
        for m in rx.finditer(s):
            if m.start() > best: best, role = m.start(), r
    return role


def _cite_index(F):
    """per facts line, by its id: the words it shows, its numbers by role (share, rgp, named, bits, ppl, count), its
    columns, and whether it speaks of the whole window. The role of a number is the last keyword before it (RGP, share,
    named, mean), else its unit, else a share: what the facts lines are built to say."""
    if getattr(F, "_cidx", None): return F._cidx
    out = {}
    for lid, ln, inf in zip(F.ids, F.lines, F.info):
        roles = collections.defaultdict(set); prev = 0; terms = ln.startswith("TERMS"); got = []
        for m in _FNUM.finditer(ln):
            unit = (m.group(3) or "").lower(); left = ln[prev:m.start()]; prev = m.end()
            soft = False
            if unit.startswith("bit"): role = "bits"
            elif unit.startswith("famil"): role = "ppl" if re.search(r"perplexity", left, re.I) else "count"
            elif unit == "%":
                role = "definition" if terms else _role_left(left, _FROLE)
                soft = role is None; role = role or "share"
            else: role = "count"; soft = True
            got.append([role, [m.group(1), m.group(2)], _JOIN.fullmatch(left) is not None, "." in (m.group(1) or ""), soft])
        for j, g in enumerate(got):                                       # "78% against 94%", "0.83 against 0.21 bits": one pair, one role
            if g[4] and g[2] and j and got[j - 1][0] != "count": g[0] = got[j - 1][0]
            elif g[4] and g[3] and j + 1 < len(got) and got[j + 1][2] and got[j + 1][0] != "count": g[0] = got[j + 1][0]
        spans = collections.defaultdict(list); colline = bool(re.fullmatch(r"S\d+\.c\d+", lid))
        for role, gs, _j, _d, _s in got:
            for g in gs:
                if not g: continue
                try: roles[role].add(Decimal(g.replace(",", "")))
                except InvalidOperation: pass
            if role in ("share", "rgp", "named") and not (colline and spans["share"]):
                try: spans[role].append(tuple(sorted(Decimal(g.replace(",", "")) for g in (gs[0], gs[1] or gs[0]))))
                except InvalidOperation: pass
        colvals = {}                                                      # the share at each column the line lists, in order
        mr = re.fullmatch(r"S\d+\.c(\d+)-(\d+)", lid)
        if colline and spans.get("share"): colvals[int(lid.split(".c")[1])] = spans["share"][0][0]
        elif mr and ":" in ln:
            vs = [Decimal(x) for x in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)%", ln.split(":", 1)[1])]
            a_, b_ = int(mr.group(1)), int(mr.group(2))
            if len(vs) == b_ - a_ + 1: colvals = {a_ + j: v for j, v in enumerate(vs)}
        elif lid == "E" and inf["cols"] and spans.get("share"): colvals[min(inf["cols"])] = spans["share"][0][0]
        hesv = [(int(a), Decimal(b)) for a, b in re.findall(r"most hesitant at column (\d+)(?: \([^)]*\))?,(?: named [\d.]+%,)? (\d+\.\d+) bits", ln)]
        hesv += [(int(b), Decimal(a)) for a, b in re.findall(r"hesitates more than (\d+\.\d+) bits \(column (\d+)", ln)]
        if lid == "H": hesv = [(int(a), Decimal(b)) for a, b in re.findall(r"column (\d+) \([^)]*\) (\d+\.\d+) bits", ln)[:1]]
        lowv = [(int(a), Decimal(b)) for a, b in re.findall(r"fewest named(?: and most hesitant)? at column (\d+)(?: \([^)]*\))?, named (\d+(?:\.\d+)?)%", ln)]
        out[lid] = dict(id=lid, words={w.lower() for w in re.findall(r"[\w'-]+", ln)}, low=ln.lower(), roles=roles, spans=spans, colvals=colvals,
                        hesv=hesv, lowv=lowv,
                        cols=set(inf["cols"]) | cols_of(ln), glob=inf["glob"], head=inf.get("head"), subs=[],
                        fams=inf.get("fams") or [], show={w.lower() for w in (inf.get("show") or ())})
    for lid in list(out):                       # a stretch's own sub-lines: [S4] -> [S4.c34], [S4.c53-79], [S4.b45]
        p = lid.split(".")[0]
        if p != lid and p in out: out[p]["subs"].append(lid)
    F._cidx = out; return out


def parse(text):
    """the model's answer as its sections: [dict(tag, sents=[dict(text, punct, cites, bold)])], each sentence what ends
    at a citation in braces -> (sections, dict(tail, untagged, dup, order)): what stands outside a section or after the
    last citation of one is a fragment the reading cannot use"""
    secs = []; notes = dict(tail=[], untagged="", dup=[], tags=[])
    parts = TAG_RE.split("\n" + text.strip())
    if parts[0].strip(): notes["untagged"] = " ".join(parts[0].split())[:200]
    for tag, body in zip(parts[1::2], parts[2::2]):
        body = " ".join(body.split())
        notes["tags"].append(tag)
        if any(s["tag"] == tag for s in secs): notes["dup"].append(tag); continue
        sents = []; pos = 0
        for m in CITE_RE.finditer(body):
            t = body[pos: m.start()].strip(); pos = m.end()
            ids = CID_RE.findall(m.group(1))
            if not t: continue
            p = m.group(2) or ""
            if t[-1] in ".;:!?": p = ""
            elif not p: p = "."
            sents.append(dict(text=t, punct=p, cites=ids, bold=bool(re.search(r"\*\*.+?\*\*", t))))
        rest = body[pos:].strip()
        if len(rest) > 3: notes["tail"].append(rest[:160])
        if sents: secs.append(dict(tag=tag, sents=sents))
    notes["order"] = [s["tag"] for s in secs]
    return secs, notes


def _sent_bad(s, ids, F, X, extra=()):
    """what a sentence says that the facts lines it cites do not carry -> (bad, numbers, names): its numbers by role,
    its identifiers, its gene names, its columns, the words inside [[ ]], and the claims the facts make elsewhere.
    A stretch line carries its own sub-lines for gene names, products and columns, never for a number: a number belongs
    to the line that states it, and a stretch's own line states its ranges. extra: the other
    sentences' ids, for the sentence that opens a paragraph — an opener may summarise what its paragraph states."""
    bad = []; lines = []
    for i in ids:
        got = _resolve_id(i, X)
        if not got: bad.append(dict(kind="citation", text=i, why="no facts line has this id", context=s[:120]))
        lines += [L for L in got if L not in lines]
    if len(ids) > CITES_MAX:                                              # a sentence citing everything is a sentence citing nothing
        bad.append(dict(kind="citation", text=", ".join(ids), why=f"a sentence cites at most {CITES_MAX} lines", context=s[:120]))
    if not lines:
        return bad + [dict(kind="citation", text=s[:60], why="the sentence cites no facts line", context=s[:120])], 0, 0
    inh = [L for i in dict.fromkeys(extra) for L in _resolve_id(i, X) if L not in lines]
    main = lines + inh
    subs = [X[j] for L in main for j in L["subs"] if X[j] not in main]
    roles = collections.defaultdict(set); wlow = set(); cols = set(); low = []
    for L in main:
        for k, v in L["roles"].items(): roles[k] |= v
        hd = X.get(L["id"].split(".")[0])                                 # a stretch of one column: its line and its column's
        if hd is not None and hd is not L and hd["cols"] == L["cols"]:     # line speak of the same column
            for k, v in hd["roles"].items(): roles[k] |= v
    vis = set()
    for L in main + subs:
        wlow |= L["words"] | L["show"]; low.append(L["low"])
        if L["id"] not in ("Q", "X"): cols |= L["cols"]
        vis |= set().union(set(), *(f["words"] for f in L["fams"]))
    byhead = collections.defaultdict(set)                                 # two column lines of one stretch cited: the columns
    for L in main:                                                        # between them are that stretch's too
        if "." in L["id"] and L["cols"]: byhead[L["id"].split(".")[0]] |= L["cols"]
    for h, cs in byhead.items():
        if h in X and len(cs) >= 1 and sum(1 for L in main if L["id"].startswith(h + ".")) >= 2:
            cols |= {c for c in X[h]["cols"] if min(cs) <= c <= max(cs)}
    heads = [X[h] for h in {L["id"].split(".")[0] for L in main} if h in X]
    vis |= wlow
    low = " ".join(low)
    cited = ", ".join(L["id"] for L in lines)
    pool = collections.defaultdict(set)
    for r, vs in roles.items(): pool[ROLE_KIND[r]] |= vs
    br = [L for L in lines if re.search(r"\.b\d+$", L["id"])]             # a branch spoken of: its own families, its own shares,
    bm = re.search(r"\bbranch(?:es)?\b|\bparallel path\b", s, re.I)       # in the clause that speaks of it (from the clause
    bmode = bool(br) and bool(bm) and len(lines) > len(br)                # before "branch" to the end of the sentence)
    bstart = max(s.rfind(",", 0, bm.start()), s.rfind(";", 0, bm.start()), s.rfind(":", 0, bm.start())) + 1 if bmode else 0
    if bmode and "[[" in s[bstart: bm.start()] or bmode and s[:bstart].count("[[") != s[:bstart].count("]]"): bstart = 0
    inb = lambda i: bmode and i >= bstart
    if bmode:
        bwords = set().union(*(L["words"] | L["show"] for L in br))
        broles = collections.defaultdict(set)
        for L in br:
            for k, v in L["roles"].items(): broles[k] |= v
    nnum = nname = 0
    body = s
    for fr in sorted(F.frags, key=len, reverse=True):                     # "family 4" of a product: a name, not a number
        body = re.sub(re.escape(fr), lambda mm: re.sub(r"\d", "#", mm.group(0)), body, flags=re.I)
    spans = [mm.span() for mm in re.finditer(r"\[\[.*?\]\]", body, flags=re.S)]
    inside = lambda i: any(a <= i < b for a, b in spans)
    def ident(mm):
        tok = mm.group(0).strip("'-/.")
        if not re.search(r"[A-Za-z]", tok): return mm.group(0)
        u = re.fullmatch(r"(\d[\d,]*(?:\.\d+)?)-(bits?|fold|columns?|genes?|famil(?:y|ies)|percent)", tok)
        if u: return u.group(1) + " " + u.group(2)                        # 0.4-bit: a number and its unit
        chem = re.fullmatch(r"\d+(?:,\d+)*-[A-Za-z]{4,}[\w,-]*", tok) and not re.match(r"\d+(?:,\d+)*-(?:genes?|columns?|famil|fold|bits?|stretch)", tok)
        if tok.lower() not in low and not (inside(mm.start()) and chem):
            bad.append(dict(kind="identifier", text=tok, why=f"not in the lines cited ({cited})", context=s[:120]))
        return re.sub(r"\d", "#", mm.group(0))
    body = IDENT_RE.sub(ident, body)
    rsp = [(m.start(1), m.end(1), m.start(2), m.end(2)) for m in                       # the two ends of a range
           re.finditer(r"(?<![\w.,])(\d[\d,]*(?:\.\d+)?)\s*(?:to|-|–)\s*(\d[\d,]*(?:\.\d+)?)(?=\s*(?:%|bits?\b|famil))", body)]
    def inrange(i): return any(a <= i < b or c <= i < d for a, b, c, d in rsp)
    prole = None; froms = []; skip_at = set(); hi_at = None
    scols_all = cols_of(s)
    cvals = {}
    for L in main + subs:
        for k2, x in L.get("colvals", {}).items(): cvals.setdefault(k2, x)
    colset = cols_of(s) & cols                                            # a column the lines cited speak of ("columns 69 to 79" of
    for mm in NUM_RE.finditer(body):                                      # [S4.c53-79]): a column, checked as one below
        if not UNIT_RE.match(body, mm.end()) and not LOWER_RE.match(body, mm.end()) and "." not in mm.group(0) \
           and mm.group(0).isdigit() and int(mm.group(0)) in colset and \
           re.search(r"\bcolumns?\b[\w\s,–-]{0,40}$", body[max(0, mm.start() - 48): mm.start()], re.I):
            continue
        if mm.start() in skip_at: continue                                # the upper end of a range verified with its lower end
        if inside(mm.start()):
            if not re.match(r"\d+(?:,\d+)*-[A-Za-z]{4,}", body[mm.start():]):
                bad.append(dict(kind="interpretation", text=mm.group(0), why="a number inside [[ ]]", context=s[:120]))
            continue
        raw = mm.group(0); nnum += 1
        try: v = Decimal(raw.replace(",", ""))
        except InvalidOperation: continue
        u = UNIT_RE.match(body, mm.end()); unit = (u.group(1).lower() if u else ""); aend = u.end() if u else mm.end()
        hi = None
        if not unit:
            lo = LOWER_RE.match(body, mm.end())
            if lo: unit = lo.group(2).lower(); aend = lo.end(); hi = lo.group(1); hi_at = lo.start(1)   # a range's lower end: its unit
        kind = "pct" if unit in ("%", "percent", "per cent") else "bits" if "bit" in unit else "fam" if unit.startswith("famil") and "." in raw else None
        role = "bits" if kind == "bits" else "ppl" if kind == "fam" else None
        if kind == "pct":
            after = body[aend:][:44]
            fol = next((r for rx, r in _FOLLOW if rx.match(after)), None)     # "N% in a region of plasticity" decides; "of chromosomes"
            left = re.split(r"[.!?]\s|\*\*", body[:mm.start()])[-1][-90:]     # only over a keyword of another clause, or a distant one
            role = fol if fol == "rgp" else (_role_left(left[-22:], _SROLE) or fol or _role_left(left, _SROLE) or prole)
        prole = role or prole
        if inb(mm.start()) and role == "share": vs = broles.get("share", set())   # a branch's share is the branch's, not its columns'
        else: vs = roles.get(role, set()) if (role and ROLE_KIND.get(role) == kind) else (pool.get(kind, set()) if kind else set().union(*pool.values()) if pool else set())
        ok = v in _allowed({kind or "int": vs})[kind or "int"]
        rrole = role or ("share" if kind == "pct" else None)               # a bare percent: a share, as the facts write it
        if kind == "pct" and rrole == "share" and not inb(mm.start()) and not re.search(r"\bfrom\s+$", body[max(0, mm.start() - 8): mm.start()], re.I):
            ccs = _clause_cols(s, mm.start(), F)                          # the shares at the columns named, from the column lines of
            if ccs and all(k2 in cvals for k2 in ccs):                    # the lines cited: "columns 15 to 20 ... 73 to 79%" is exactly
                vals = [cvals[k2] for k2 in sorted(ccs)]                  # their lowest and highest, "column 14 ... 68%" its value
                al_ = lambda x: _allowed({"pct": {x}})["pct"]
                if hi is not None:
                    try: vh_ = Decimal(hi.replace(",", ""))
                    except InvalidOperation: vh_ = None
                    if vh_ is not None and {min(v, vh_), max(v, vh_)} and min(v, vh_) in al_(min(vals)) and max(v, vh_) in al_(max(vals)):
                        skip_at.add(hi_at); continue
                elif len(ccs) == 1 and v in al_(vals[0]): continue
        if ok and hi is not None and kind == "pct" and rrole in ("share", "rgp", "named"):
            try: vh = Decimal(hi.replace(",", ""))
            except InvalidOperation: vh = None
            src = br if inb(mm.start()) and rrole == "share" else main
            if vh is not None and re.search(r"\bfrom\s+$", body[max(0, mm.start() - 8): mm.start()], re.I):
                ends = _ends(_clause_cols(s, mm.start(), F), main + subs)          # "from 90 to 48%": a direction along the columns
                if ends and v in _allowed({"pct": {ends[0]}})["pct"] and vh in _allowed({"pct": {ends[1]}})["pct"]:
                    froms.append((v, vh)); continue
                bad.append(dict(kind="number", text=f"from {raw} to {hi}%", why=f"\"from A to B\" is the share at the first and at the last "
                                f"column named, and the lines cited ({cited}) do not show both: cite those columns' lines, or give "
                                f"the range as a line shows it", context=s[:120]))
                continue
            if vh is not None and not _range_ok(v, vh, rrole, src, scols_all, X):
                bad.append(dict(kind="number", text=f"{raw} to {hi}%", why=f"a range the lines cited ({cited}) do not give: the range a line "
                                f"shows, or the lowest and highest of the lines cited, or the values at the first and last column named",
                                context=s[:120]))
                continue
        if ok and kind == "pct" and rrole == "share" and hi is None and not inb(mm.start()):
            pre_ = body[max(0, mm.start() - 44): mm.start()]              # "`xylE`, carried by 69%": that gene's share
            gm = list(re.finditer(r"`([^`\n]+)`", pre_))
            g_ = gm[-1].group(1) if gm and not re.search(r"\d", pre_[gm[-1].end():]) else None
            if g_ and F.gshare.get(g_):
                if v not in set().union(*(_allowed({"pct": {x}})["pct"] for x in F.gshare[g_])):
                    bad.append(dict(kind="number", text=f"{raw}%", why=f"`{g_}` is carried by {', '.join(str(x) + '%' for x in sorted(F.gshare[g_]))} "
                                    f"in the facts", context=s[:120])); continue
            pre_s = s[max(0, mm.start() - 44): mm.start()]
            pk = pre_s[max((x.end() for x in re.finditer(r"%|\bbits?\b|`", pre_s)), default=0):].lower()
            pf = [x for p_, x in F.pshare.items() if len(p_) >= 6 and p_[:14] in pk]
            if not g_ and pf:                                             # "an IS1-like element transposase in 29%": that family's
                if v not in set().union(*(_allowed({"pct": {y}})["pct"] for x in pf for y in x)):
                    bad.append(dict(kind="number", text=f"{raw}%", why="not the share of the family it follows in the facts", context=s[:120])); continue
            elif not g_:
                ccs = _clause_cols(s, mm.start(), F)                      # "column 10 ... carried by 64%": that column's share
                if len(ccs) == 1 and min(ccs) in cvals and not re.search(r"\bgenomes\b|\bchromosomes\b\s*$", pre_[-3:]) \
                   and v not in _allowed({"pct": {cvals[min(ccs)]}})["pct"] and re.search(r"carri|share|\bin\s*$|\bat\s*$", pre_[-24:]):
                    bad.append(dict(kind="number", text=f"{raw}%", why=f"column {min(ccs)} is carried by {cvals[min(ccs)]}% in the lines cited "
                                    f"({cited})", context=s[:120])); continue
        if ok and kind == "pct" and role == "named" and F.gtop is not None and any(L["id"] == "G" for L in lines) \
           and re.search(r"names?\s+$", body[max(0, mm.start() - 12): mm.start()]) and re.match(r"%\s+of\s+(?:the\s+)?genes\b", body[mm.end():]) \
           and not re.search(r"persistent|shell|cloud|inside|outside|spine|fork|variable", re.split(r"[.;]", s[:mm.start()])[-1], re.I):
            if v not in _allowed({"pct": {F.gtop}})["pct"]:               # "across the window it names 89% of genes": [G]'s own
                bad.append(dict(kind="number", text=f"{raw}%", why=f"across the window the model names {F.gtop}% of genes [G]", context=s[:120])); continue
        if ok and kind == "bits" and F.gbits is not None and any(L["id"] == "G" for L in lines) and not any(L["id"] in ("H", "M") for L in lines) \
           and re.match(r"\s*bits?\s+on\s+average", body[mm.end():]) and v not in _allowed({"bits": {F.gbits}})["bits"]:
            bad.append(dict(kind="number", text=f"{raw} bits", why=f"across the window it hesitates {F.gbits} bits on average [G]", context=s[:120])); continue
        if ok and kind == "bits" and "." not in raw and v not in roles.get("bits", set()):
            bad.append(dict(kind="format", text=f"{raw} bits", why="bits rounded to a whole number: write them as the facts do", context=s[:120])); continue
        if not ok:
            say = ROLE_SAY.get(role, "a number").format(unit=F.unit) if role else "a number"
            bad.append(dict(kind="number", text=raw + ("%" if kind == "pct" else f" {unit}" if unit else ""),
                            why=f"not {say} of the lines cited ({cited})", context=s[:120]))
    for mm in re.finditer(r"(?<![\w.-])\.\d+\b|(?<![\w,-])\d+,\d{1,2}\b(?!,?\d)", re.sub(r"\[\[.*?\]\]", " ", s)):
        bad.append(dict(kind="format", text=mm.group(0), why="write numbers as the facts do (0.97, not .97 or 0,97)"))
    for rx, down in ((TREND_DOWN, True), (TREND_UP, False)):              # "fewer and fewer": shown by a "from A to B" that falls
        mt = rx.search(re.sub(r"\[\[.*?\]\]", " ", s))
        if mt and not any((a_ > b_) if down else (a_ < b_) for a_, b_ in froms):
            bad.append(dict(kind="claim", text=mt.group(0), why="a trend along the columns needs its values: \"from A to B%\" at the first "
                            "and last column named, from the lines cited", context=s[:120]))
    allv = set().union(*pool.values()) if pool else set()
    for mm in NUMW_RE.finditer(s):
        nnum += 1
        if Decimal(NUMW[mm.group(1).lower()]) not in allv:
            bad.append(dict(kind="number word", text=mm.group(0), why="write numbers in digits, from the lines cited"))
    for mm in VAGUE_RE.finditer(s): bad.append(dict(kind="number word", text=mm.group(0), why="write numbers in digits, from the lines cited"))
    for mm in META_RE.finditer(s): bad.append(dict(kind="wording", text=mm.group(0), why="the reading speaks of the window, not of the facts"))
    whyg = (f"a branch is spoken of with its own families only ({', '.join(L['id'] for L in br)})" if bmode
            else f"not in the lines cited ({cited})")
    def known(g, at=0):
        gl = g.lower(); b = inb(at)
        pool_ = bwords if b else wlow
        if gl in pool_: return True
        if len(gl) == 3 and any(w.startswith(gl) for w in pool_ if w in {x.lower() for x in F.genes}): return True
        if b: return False
        return gl in low or any(k.endswith("…") and gl.startswith(k[:-1]) for k in F.atp)
    whyn = f"not in the lines cited ({cited})"
    for mm in re.finditer(r"`([^`\n]+)`", s):
        g = mm.group(1).strip(); nname += 1
        if not known(g, mm.start()): bad.append(dict(kind="gene", text=g, why=whyg if inb(mm.start()) else whyn, context=s[:120]))
    nog = re.sub(r"`[^`\n]*`", " ", s)
    for mm in GENE_RE.finditer(nog):
        g = mm.group(1); nname += 1
        if not known(g, mm.start()) and g not in ALLOWED_WORDS: bad.append(dict(kind="gene", text=g, why=whyg if inb(mm.start()) else whyn, context=s[:120]))
    for mm in PROT_RE.finditer(nog):
        g = mm.group(1)
        if not known(g, mm.start()) and not known(g[0].lower() + g[1:], mm.start()) and g not in ALLOWED_WORDS:
            nname += 1; bad.append(dict(kind="gene", text=g, why=whyg if inb(mm.start()) else whyn, context=s[:120]))
    for mm in OPERON_RE.finditer(re.sub(r"\[\[.*?\]\]", " ", nog)):
        g = mm.group(1)
        if not known(g, mm.start()) and g.lower() not in WORDS3:
            bad.append(dict(kind="gene", text=g, why=f"not an operon of the lines cited ({cited})", context=s[:120]))
    for mm in OPERON_RE2.finditer(s):                                     # "the `waa` locus" over a stretch: most of its families
        pre = mm.group(1).lower()                                         # must carry that name, or the label covers what it is not
        oc = _clause_cols(s[:mm.start()] + s[mm.end():], mm.start(), None) # the columns named nearest, not the operon's own genes
        fs = [f for L in main + subs if not L["glob"] for f in L["fams"]
              if not f["alt"] and (not oc or f["col"] is None or f["col"] in oc)]
        pres = {pre} | set(ALIAS.get(pre, ()))                            # rfa is the old name of waa
        nin = sum(f["label"].lower().startswith(tuple(pres)) for f in fs)
        if len(fs) >= 4 and nin * 2 < len(fs):
            bad.append(dict(kind="claim", text=mm.group(0), why=f"only {nin} of the {len(fs)} families there carry that name: name "
                            f"the columns it is true of", context=s[:120]))
    miss = sorted(cols_of(s) - cols)
    if miss: bad.append(dict(kind="column", text="column " + ", ".join(map(str, miss[:6])),
                             why=f"the lines cited ({cited}) do not speak of that column", context=s[:120]))
    scols = cols_of(s)
    for mm in re.finditer(r"\[\[(.+?)\]\]", s):
        nw = len(re.findall(r"[A-Za-z0-9][\w'-]*", mm.group(1)))
        if nw < 2 or nw > 6 or re.search(r"[,;]", mm.group(1)):
            bad.append(dict(kind="interpretation", text=mm.group(1)[:40], why="an interpretation is one phrase of 2 to 6 words, not a word "
                            "or a list", context=mm.group(0)[:80]))
        cw = [w for w in re.findall(r"[A-Za-z][\w'-]{3,}|\b[A-Z][A-Za-z0-9]{1,2}\b", mm.group(1))    # GMP, DNA, Fe: words too
              if w.lower() not in STOP and w.lower() not in ("the", "and", "for", "its")]
        for w in cw:                                                      # every word of the label: in the lines cited
            if _covers(w.lower(), vis): continue
            bad.append(dict(kind="interpretation", text=w, why=f"not a word of the products or gene names of the lines cited ({cited})",
                            context=mm.group(0)[:80]))
        if not cw:                                                        # "[[sugar transport]]": nothing the products must say
            bad.append(dict(kind="interpretation", text=mm.group(1)[:40], why="it names nothing the products of the lines cited say: build "
                            "it from their words", context=mm.group(0)[:80])); continue
        fams, seen_f = [], set(); scols = _clause_cols(s, mm.start(), F) # the families the label is about: those of the
        b_ = (inb(mm.start()) or bstart == 0) if bmode else bool(br) and len(br) == len(lines)   # stretch, column or branch
        nb_ = [x for x in main + subs if not x["glob"] and not (bmode and x in br)]    # lines cited, narrowed to the columns named
        for L in (br if b_ else nb_):
            for f in L["fams"]:
                k = (f["label"], f["col"])
                if k in seen_f or (f["alt"] and not b_) or (scols and f["col"] is not None and f["col"] not in scols and not b_): continue
                seen_f.add(k); fams.append(f)
        if not fams:
            for g in genes_of(s, F):
                for L in X.values():
                    for f in L["fams"]:
                        if f["label"] == g and (f["label"], f["col"]) not in seen_f:
                            seen_f.add((f["label"], f["col"])); fams.append(f)
        if not fams:
            bad.append(dict(kind="interpretation", text=mm.group(1)[:40], why=f"the lines cited ({cited}) name no family this could be about",
                            context=mm.group(0)[:80])); continue
        hit = [f for f in fams if any(_covers(w.lower(), f["words"]) for w in cw)]
        pre = {f["label"][:3].lower() for f in fams if re.fullmatch(r"[a-z]{3}[A-Z]\w*", f["label"] or "")}
        operon = len(pre) == 1 and all(re.fullmatch(r"[a-z]{3}[A-Z]\w*", f["label"] or "") for f in fams)
        fits = len(hit) >= 2 and len(hit) * 3 >= len(fams) if operon else len(hit) * 2 >= len(fams)
        if len(fams) >= 3 and not fits:                                   # a label must fit most of the families it covers; in one
            bad.append(dict(kind="interpretation", text=mm.group(1)[:40],
                            why=f"this says what {len(hit)} of the {len(fams)} families of the lines cited ({cited}) do: name the columns "
                                f"it is true of, or leave the stretch unnamed", context=mm.group(0)[:80]))
        else:
            parts_ = [[w for w in re.findall(r"[A-Za-z][\w'-]{3,}", x) if w.lower() not in STOP and w.lower() not in GENERIC
                       and w.lower()[:-1] not in GENERIC]
                      for x in re.split(r"\s+and\s+|,\s*|\s*/\s*", mm.group(1))]
            for cp in [x for x in parts_ if len(x) >= 2] if len(fams) >= 3 else []:
                if not any(sum(_covers(w.lower(), f["words"]) for w in cp) >= 2 for f in hit):
                    bad.append(dict(kind="interpretation", text=" ".join(cp)[:40],
                                    why="its words come one from each family: no family of the lines cited does all of it",
                                    context=mm.group(0)[:80])); break
    for rx, what, key, unit_ in ((HES_RE, "most hesitant", "hesv", r"(\d+(?:\.\d+)?)\s*bits?\b"), (LOW_RE, "fewest named", "lowv", r"(\d+(?:\.\d+)?)\s*%")):
        for mm in rx.finditer(s):
            if what == "most hesitant" and re.search(r"\bno column\b|hardly|barely|seldom", s, re.I): continue
            st_ = [t for L in main for t in L[key]]                        # what the lines cited state, column and value
            if not st_:
                bad.append(dict(kind="claim", text=mm.group(0), why=f"the lines cited ({cited}) do not state a column as the {what}", context=s[:120])); continue
            cc = _claim_cols(s, mm, F) or cols_of(s)
            if cc and not (cc & {k2 for k2, _ in st_}):
                bad.append(dict(kind="claim", text=mm.group(0), why=f"the lines cited ({cited}) state column {', '.join(str(k2) for k2, _ in st_[:3])} "
                                f"as the {what}", context=s[:120])); continue
            seg = re.split(r"[;]|\.\s", s[mm.end():])[0]
            vm = re.search(unit_, seg)
            if not vm:
                bef = list(re.finditer(unit_, s[:mm.start()])); vm = bef[-1] if bef else None
            if vm:                                                        # its value: the one stated with that column
                vv = Decimal(vm.group(1)); okv = [x for k2, x in st_ if (not cc or k2 in cc)]
                if okv and not any(vv in _allowed({"bits" if key == "hesv" else "pct": {x}})["bits" if key == "hesv" else "pct"] for x in okv):
                    bad.append(dict(kind="number", text=vm.group(0), why=f"the {what} there is {okv[0]}{' bits' if key == 'hesv' else '%'} in the lines "
                                    f"cited ({cited})", context=s[:120]))
    for mm in PLASTIC_RE.finditer(s):
        ab = F.plastic.get("most" if mm.group(1).lower() == "most" else "least")
        if not ab or not any(L["id"] == "P" for L in lines):
            bad.append(dict(kind="claim", text=mm.group(0), why="only the PLASTICITY line [P] names a stretch so", context=s[:120])); continue
        cc = cols_of(s)
        if cc and not (cc & set(range(ab[0], ab[1] + 1))):
            bad.append(dict(kind="claim", text=mm.group(0), why=f"the facts say so of columns {ab[0]} to {ab[1]} only", context=s[:120]))
    plain = re.sub(r"\[\[.*?\]\]", " ", s)
    K, ready, opens = F.kind, READY_RE.search(plain), OPENS_RE.search(plain)
    hesit = re.search(r"\bhesitat", plain, re.I)
    mcited = any(L["id"] == "M" for L in main)
    if mcited and K: bad += _m_direction(plain, K, s)                     # the two kinds of column compared, as [M] states it
    nbh = [X[f"S{int(h['id'][1:]) + d}"] for h in heads if re.fullmatch(r"S\d+", h["id"]) for d in (-1, 1)
           if f"S{int(h['id'][1:]) + d}" in X]                          # "breaks the spine": the stretches on either side
    for kw in ("spine", "fork", "variable", "branch"):                    # a kind of column the sentence speaks of: in its lines
        if re.search(rf"\b{kw}(?:e?s)?\b", plain, re.I) and not any(kw in L["low"] for L in main + subs + heads
                                                                        + (nbh if kw != "branch" else [])):
            bad.append(dict(kind="claim", text=kw, why=f"no line cited ({cited}) has a {kw} here", context=s[:120]))
    for mm in re.finditer(r"(?<!\d )\b(?:all (persistent|shell|cloud)|(persistent|shell|cloud) famil\w+)\b", plain, re.I):
        pw = PART[(mm.group(1) or mm.group(2)).lower()]                   # "all persistent", "persistent families": every family
        cc = _clause_cols(s, mm.start(), F)                               # of the columns named (a branch: of the branch)
        src_ = br if inb(mm.start()) or (br and len(br) == len(lines)) else [L for L in main + subs if not L["glob"] and L not in br]
        fam_ = [f for L in src_ for f in L["fams"] if f.get("part") and (L in br or not f["alt"])
                and (L in br or not cc or f["col"] is None or f["col"] in cc)]
        other = [f for f in fam_ if f["part"] != pw]
        if fam_ and other:
            bad.append(dict(kind="claim", text=mm.group(0), why=f"{len(other)} of the {len(fam_)} families of the lines cited there are "
                            f"not {mm.group(1) or mm.group(2)}", context=s[:120]))
    for mm in re.finditer(r"(?<!\d )(?<!all )\b(persistent|shell|cloud)\b(?! famil)(?! and)", plain, re.I):  # "`selB`, persistent"
        cc = _clause_cols(s, mm.start(), F)                               # of one column: that column's family
        if len(cc) != 1 or inb(mm.start()): continue
        fam_ = [f for L in main + subs if not L["glob"] and L not in br for f in L["fams"] if not f["alt"] and f.get("part") and f["col"] in cc]
        if fam_ and fam_[0]["part"] != PART[mm.group(1).lower()]:
            bad.append(dict(kind="claim", text=mm.group(0), why=f"the family at column {min(cc)} is {PARTN[fam_[0]['part']]}", context=s[:120]))
    kinds = getattr(F, "kinds", {})                                       # "variable columns 63 to 65", "column 2 is a spine"
    for mm in list(re.finditer(r"\b(spine|fork|variable) columns? (\d+)(?:\s*(?:to|-)\s*(\d+))?(?!\s*%)", plain, re.I)) + \
              list(re.finditer(r"\bcolumns? (\d+)(?:\s*(?:to|-)\s*(\d+))?,? (?:is|are) (?:a |an |all )?(spine|fork|variable)\b(?! and)", plain, re.I)):
        g = mm.groups()
        kw, a_, b_ = (g[0], g[1], g[2]) if not g[0].isdigit() else (g[2], g[0], g[1])
        ks = range(int(a_), int(b_ or a_) + 1)
        wrong = [k for k in ks if k in kinds and kinds[k] != kw.lower()]
        if wrong: bad.append(dict(kind="claim", text=mm.group(0), why=f"column {wrong[0]} is a {kinds[wrong[0]]} column", context=s[:120]))
    for mm in re.finditer(r"\b(\d+) (persistent|shell|cloud) famil\w+", plain, re.I):   # "a branch of 9 shell families": 9 of them
        pw = PART[mm.group(2).lower()]; cc = _clause_cols(s, mm.start(), F)
        src_ = br if inb(mm.start()) or (br and len(br) == len(lines)) else [L for L in main + subs if not L["glob"] and L not in br]
        fam_ = [f for L in src_ for f in L["fams"] if f.get("part") and (L in br or not f["alt"])
                and (L in br or not cc or f["col"] is None or f["col"] in cc)]
        if fam_ and sum(f["part"] == pw for f in fam_) < int(mm.group(1)):
            bad.append(dict(kind="claim", text=mm.group(0), why=f"{sum(f['part'] == pw for f in fam_)} of the {len(fam_)} families of the "
                            f"lines cited there are {mm.group(2).lower()}", context=s[:120]))
    if ready or (hesit and opens and not mcited):                         # "reads the spines readily, hesitates where it opens"
        if not any(L["id"] in ("M", "H", "C", "G") for L in main):
            bad.append(dict(kind="claim", text=(ready or opens).group(0), why="where the model reads well and where it hesitates is "
                            "stated by [M] (by kind of column), [H], [C] or [G]: cite the line", context=s[:120]))
        if ready and re.search(r"\bspines?\b", plain, re.I) and K and K["named_spine"] < K["named_other"] - 0.005:
            bad.append(dict(kind="claim", text=ready.group(0), why="the model names fewer genes on the spine columns of this window "
                            "than on the fork and variable ones [M]", context=s[:120]))
        if ready and re.search(r"\b(?:fork|variable)\b", plain, re.I) and not re.search(r"\bspines?\b", plain, re.I) \
           and K and K["named_other"] < K["named_spine"] - 0.005:
            bad.append(dict(kind="claim", text=ready.group(0), why="the model names fewer genes on the fork and variable columns of this "
                            "window than on the spine ones [M]", context=s[:120]))
        if hesit and opens and not _claim_cols(s, opens, F) and not scols:
            if K and K["bits_other"] <= K["bits_spine"] + 0.02:
                bad.append(dict(kind="claim", text=opens.group(0), why="the fork and variable columns of this window hesitate no more "
                                "than the spine columns [M]", context=s[:120]))
            elif F.hes_spine:
                bad.append(dict(kind="claim", text=opens.group(0), why="the most hesitant column of this window is a spine column [M], "
                                "[H]: say where it hesitates most, or name the columns", context=s[:120]))
    for mm in re.finditer(r"[A-Za-z][A-Za-z']{4,}", re.sub(r"\[\[.*?\]\]|`[^`]*`|\"[^\"]*\"", " ", s)):
        w = mm.group(0).lower()                                           # biology belongs inside [[ ]]: a product's word outside it
        if w in PROSE or w in STOP or w in F.pnames or w not in F.pwords: continue
        bad.append(dict(kind="interpretation", text=mm.group(0), why="a word of the products outside [[ ]]: what genes do is written "
                        "inside the brackets", context=s[:120]))
    for mm in NEG_RE.finditer(plain): bad.append(dict(kind="claim", text=mm.group(0), why="the facts never say chromosomes lack a gene"))
    for mm in SUP_RE.finditer(plain): bad.append(dict(kind="claim", text=mm.group(0), why="a superlative the facts do not state"))
    bad += _only_majority(plain)
    for mm in LEAST_RE.finditer(s): bad.append(dict(kind="claim", text=mm.group(0), why="the facts give the most hesitant columns, not the least"))
    for mm in re.finditer(r"outside (?:any|a) regions? of plasticity,? in (?:only )?\d[\d.]*(?: to \d[\d.]*)?%", s, re.I):
        bad.append(dict(kind="claim", text=mm.group(0), why="the facts give the share of calls inside a region of plasticity"))
    for mm in re.finditer(r"\b\d+\s+(?:in|out of|of every)\s+\d+\b(?!\s*%|\.\d|\s+to\s+\d)", plain):
        bad.append(dict(kind="claim", text=mm.group(0), why="a fraction the facts do not give"))
    for mm in re.finditer(r"\*\*(.+?)\*\*", s, re.S):                     # the bold opening: structure, no biology
        b = next((x for x in BIO_RE.finditer(re.sub(r"\[\[.*?\]\]|`[^`]*`|\"[^\"]*\"", " ", mm.group(1))) if x.group(0).lower() not in low), None)
        if b: bad.append(dict(kind="claim", text=b.group(0), why="biology outside [[ ]], in a bold sentence", context=mm.group(1)[:120]))
    if F.comp and any(L["id"] == "C" for L in lines) and not re.search(r"\bspines?\b", plain, re.I):
        c = F.comp; low_ = plain.lower()                                  # "0.83 against 0.21 bits", "78% against 94%": [C]'s pairs,
        inside_first = low_.find("inside") < low_.find("outside") if "outside" in low_ else True   # the side named first first
        for mm in re.finditer(r"(\d+(?:\.\d+)?)(%|\s*bits)?\s+against\s+(\d+(?:\.\d+)?)\s*(%|bits?)", plain):
            pc = mm.group(4).startswith("%") or (mm.group(2) or "").strip() == "%"
            want = (c["ai"], c["ao"]) if pc else (c["ei"], c["eo"])
            if not inside_first: want = want[::-1]
            A, B = Decimal(mm.group(1)), Decimal(mm.group(3))
            k_ = "pct" if pc else "bits"
            if A not in _allowed({k_: {Decimal(want[0].rstrip("%"))}})[k_] or B not in _allowed({k_: {Decimal(want[1].rstrip("%"))}})[k_]:
                bad.append(dict(kind="number", text=mm.group(0), why=f"[C] gives {want[0]} against {want[1]}{'' if pc else ' bits'}", context=s[:120]))
    if F.kind and any(L["id"] == "M" for L in lines) and re.search(r"\bspines?\b", plain, re.I):
        K = F.kind; low_ = plain.lower()                                  # [M]'s pairs, the kind named first first
        ks = [(low_.find(w), w) for w in ("spine", "fork", "variable", "other") if w in low_]
        sp_first = min(ks)[1] == "spine" if ks else True
        for mm in re.finditer(r"(\d+(?:\.\d+)?)(%|\s*bits)?\s+against\s+(\d+(?:\.\d+)?)\s*(%|bits?)", plain):
            pc = mm.group(4).startswith("%") or (mm.group(2) or "").strip() == "%"
            want = (fmt_pct(K["named_spine"]), fmt_pct(K["named_other"])) if pc else (fmt_bits(K["bits_spine"]), fmt_bits(K["bits_other"]))
            if not sp_first: want = want[::-1]
            k_ = "pct" if pc else "bits"
            if Decimal(mm.group(1)) not in _allowed({k_: {Decimal(want[0].rstrip("%"))}})[k_] or \
               Decimal(mm.group(3)) not in _allowed({k_: {Decimal(want[1].rstrip("%"))}})[k_]:
                bad.append(dict(kind="number", text=mm.group(0), why=f"[M] gives {want[0]} against {want[1]}{'' if pc else ' bits'}, the kind "
                                f"named first first", context=s[:120]))
    bad += _direction(s, F) + _colgene(s, F)
    return bad, nnum, nname


_MENTION = re.compile(r"\b(?:variable |fork |spine )?columns?,?\s+\d+(?:\s*(?:to|-|–|,|and)\s*\d+)*(?![.,]?\d|\s*%)|"
                      r"`[^`\n]+`(?:\s+(?:to|through)\s+`[^`\n]+`)?", re.I)
def _clause_cols(s, i, F=None):
    """the columns a label or an operon at position i of a sentence is about: in its clause ("columns 12 to 14 ... [[x]],
    and columns 29 to 32 ... [[y]]": each label its own), the mention nearest to it, columns ("columns 30 to 43") or genes
    ("`thiB` to `thiQ`", "the `cai` genes"); the whole sentence's columns when the clause names none"""
    m = re.sub(r"\[\[.*?\]\]", lambda x: "#" * len(x.group(0)), s)
    cuts = [0] + [x.end() for x in re.finditer(r",\s+and\s+|;\s+|,\s+while\s+|,\s+then\s+", m)] + [len(s) + 1]
    a, b = next(((a, b) for a, b in zip(cuts, cuts[1:]) if a <= i < b), (0, len(s)))
    best = None
    for mm in _MENTION.finditer(m, a, min(b, len(m))):
        t = s[mm.start(): mm.end()]
        if t.startswith("`"):
            if F is None: continue
            gs = re.findall(r"`([^`\n]+)`", t)
            ks = [F.at.get(g) or set().union(set(), *(F.at[x] for x in F.genes if len(g) == 3 and x.startswith(g))) for g in gs]
            if not all(ks): continue
            c = set(range(min(ks[0]), max(ks[-1]) + 1)) if len(ks) == 2 and min(ks[0]) <= max(ks[-1]) else set().union(*ks)
            ex = cols_of(s[a:b])
            if ex and c & ex: c = c & ex
        else:
            c = cols_of(t)
        if not c: continue
        d = mm.start() - i if mm.start() >= i else i - mm.end()
        if best is None or d < best[0]: best = (d, c)
    return best[1] if best else cols_of(s[a:b]) or cols_of(s)


TREND_DOWN = re.compile(r"\bfewer and fewer\b|\b(?:falling|decreasing|declining|dwindling|dropping|thinning)\b", re.I)
TREND_UP = re.compile(r"\bmore and more\b|\b(?:rising|increasing|growing)\b", re.I)
def _ends(cs, lines):
    """the shares the lines show at the first and last of the columns cs (None when one of them is not shown)"""
    if not cs: return None
    cv = {}
    for L in lines:
        for k, v in L.get("colvals", {}).items(): cv.setdefault(k, v)
    a, b = min(cs), max(cs)
    return (cv[a], cv[b]) if a in cv and b in cv and a != b else None


def _range_ok(a, b, role, lines, scols, X):
    """a range of a sentence ("84 to 88%"): one a line cited shows; or the lowest and highest the lines cited show, in
    that role (several column lines); or, for a share, the values at the first and last column the sentence names"""
    al = lambda x: _allowed({"pct": {x}})["pct"]
    lo_, hi_ = min(a, b), max(a, b)
    sp = [t for L in lines for t in L.get("spans", {}).get(role, [])]
    if not sp: return True                                                # nothing to hold it against: the value check decides
    if any(lo_ in al(x) and hi_ in al(y) for x, y in sp): return True
    covered = set().union(set(), *(L["cols"] for L in lines if L.get("spans", {}).get(role)))
    if lo_ in al(min(x for x, _ in sp)) and hi_ in al(max(y for _, y in sp)) and (not scols or scols <= covered): return True
    if role == "share" and scols:
        ends = []
        for k in (min(scols), max(scols)):
            L = next((L for L in lines if re.fullmatch(r"S\d+\.c\d+", L["id"]) and L["cols"] == {k}), None)
            if L is None or not L["spans"].get("share"): return False
            ends.append(L["spans"]["share"][0][0])
        if {x for e in ends for x in al(e)} >= {a, b} and a in al(ends[0]) and b in al(ends[1]): return True
    return False


def _resolve_id(i, X):
    """the lines an id stands for: itself; a range the model made up ("S2.c77-79") or one column of a range line
    ("S2.c20" for [S2.c20-24]): the column lines of that stretch within it"""
    if i in X: return [X[i]]
    m = re.fullmatch(r"(S\d+)\.c(\d+)(?:-(\d+))?", i)
    if not m or m.group(1) not in X: return []
    a = int(m.group(2)); b = int(m.group(3) or a)
    out = [X[j] for j in X[m.group(1)]["subs"] if re.match(r"S\d+\.c", j) and X[j]["cols"] and
           (min(X[j]["cols"]) <= b and max(X[j]["cols"]) >= a)]
    return out if out and set().union(*(L["cols"] for L in out)) >= set(range(a, b + 1)) & X[m.group(1)]["cols"] else []


_KIND_W = r"(spines?|spine columns?|fork|variable|others?|other ones)"
def _m_direction(p, K, s):
    """"names more genes on the spine columns", "hesitates less there": in the direction [M] gives, for the kind of
    column the clause is about (said, or the first one the sentence names)"""
    bad = []; low = p.lower()
    kinds = [(low.find(w), w) for w in ("spine", "fork", "variable") if w in low]
    first = min(kinds)[1] if kinds else None
    side = lambda w: "spine" if w and w.startswith("spine") else "other" if w else None
    for m in re.finditer(r"\bnames?\s+(?:about\s+|nearly\s+|roughly\s+)?(more|fewer|as many)\s+(?:genes\s+)?(?:(?:on|in|at|over)\s+the\s+"
                         + _KIND_W + r"|there)?", low):
        sd = side(m.group(2)) or side(first)
        if not sd: continue
        d = K["named_spine"] - K["named_other"]; d = d if sd == "spine" else -d
        want = "more" if d > 0.02 else "fewer" if d < -0.02 else "as many"
        if m.group(1) != want:
            bad.append(dict(kind="claim", text=m.group(0), why=f"[M]: the model names {'about as many' if want == 'as many' else want} genes "
                            f"on the {'spine' if sd == 'spine' else 'fork and variable'} columns", context=s[:120]))
    for m in re.finditer(r"\bhesitat\w*\s+(?:about\s+|nearly\s+)?(more|less|as much)\b(?:\s+(?:(?:on|in|at|over)\s+the\s+" + _KIND_W
                         + r"))?", low):
        sd = side(m.group(2)) or ("other" if OPENS_RE.search(low[m.end():]) else side(first))
        if not sd: continue
        d = K["bits_spine"] - K["bits_other"]; d = d if sd == "spine" else -d
        want = "more" if d > 0.02 else "less" if d < -0.02 else "as much"
        if m.group(1) != want:
            bad.append(dict(kind="claim", text=m.group(0), why=f"[M]: the model hesitates {'about as much' if want == 'as much' else want} "
                            f"on the {'spine' if sd == 'spine' else 'fork and variable'} columns", context=s[:120]))
    return bad


def check_cited(text, F):
    """the model's answer against the facts, sentence by sentence, each against the lines it cites (stricter than the
    whole-text check: nothing may come from another line) -> dict(ok, sections, notes, bad, numbers, names, words,
    sentences, cited, structure)"""
    X = _cite_index(F)
    secs, notes = parse(text)
    nnum = nname = 0; bad = []; ns = 0
    for sec in secs:
        rest = [i for st in sec["sents"][1:] for i in st["cites"]]        # an opener may summarise its own paragraph
        for i, st in enumerate(sec["sents"]):
            st["inherit"] = [x for x in dict.fromkeys(rest) if x not in st["cites"]] if i == 0 else []
            b, n1, n2 = _sent_bad(st["text"], st["cites"], F, X, st["inherit"])
            st["bad"] = b; st["ok"] = not b; st["opens"] = i == 0
            nnum += n1; nname += n2; ns += 1; bad += b
    for sec in secs:                                                      # "the spine after it", "then": further along the window
        prev = None
        for i, st in enumerate(sec["sents"]):
            cs = cols_of(st["text"]) or set().union(set(), *[L["cols"] for c in st["cites"] for L in _resolve_id(c, X) if not L["glob"]])
            if prev and cs and i and re.search(r"\b(?:after (?:it|them|this|that)|then|next|further on|beyond (?:it|them))\b",
                                               re.sub(r"\[\[.*?\]\]", " ", st["text"]), re.I) and min(cs) < min(prev):
                b = dict(kind="claim", text="after it", why=f"columns {min(cs)} to {max(cs)} come before the columns of the sentence "
                         f"before (from {min(prev)}): the window is read in its order", context=st["text"][:120])
                st["bad"].append(b); st["ok"] = False; bad.append(b)
            prev = cs or prev
    order = [s["tag"] for s in secs]
    want = [t for t in SECTIONS if t in order]
    struct = []
    if order != want: struct.append(f"the paragraphs are {' '.join('@' + t for t in order) or 'not tagged'}: write "
                                    + " ".join("@" + t for t in SECTIONS) + " in that order")
    for t in SECTIONS:
        if t not in order: struct.append(f"the @{t} paragraph is missing")
    if notes["dup"]: struct.append("these paragraphs are written twice: " + ", ".join("@" + t for t in notes["dup"]))
    if notes["untagged"]: struct.append(f'this stands before the first paragraph, with no tag: "{notes["untagged"][:80]}"')
    for t in notes["tail"]: struct.append(f'this sentence cites no facts line: "{t[:80]}"')
    return dict(ok=not bad and not struct, sections=secs, notes=notes, bad=bad, numbers=nnum, names=nname,
                words=_words(render(secs)), sentences=ns, cited=sum(len(st["cites"]) for sec in secs for st in sec["sents"]),
                structure=struct, _F=F, _X=X)


def _lead(t):
    """a paragraph whose opening sentence was dropped: its first clause takes the bold, so every paragraph still opens
    on one (the bold is the page's form, not a claim)"""
    if "**" in t: return t
    m = re.match(r"([^,;:*]{10,70})(?=[,;:]\s)", t)
    if m and 3 <= len(m.group(1).split()) <= 10 and m.group(1).count("`") % 2 == 0 and "[[" not in m.group(1):
        return f"**{m.group(1)}**" + t[m.end():]
    return t


def render(secs):
    """the sections as the reading the page shows: no tags, no citations, one paragraph each"""
    return "\n\n".join(_lead(" ".join(st["text"] + st["punct"] for st in sec["sents"])) for sec in secs if sec["sents"])


def _words(t): return len(re.findall(r"[A-Za-z0-9][\w'.,%-]*", t.replace("**", "")))


def _carries(t):
    """what a sentence tells the reader: its numbers with their unit, its gene names, the columns it names — what is
    lost if it goes, and what another sentence already says if it stays"""
    out = ({" ".join(m.group(0).split()) for m in re.finditer(r"\d[\d,]*(?:\.\d+)?\s*(?:%|bits?|famil\w+)", t)}
           | {g.strip() for g in re.findall(r"`([^`\n]+)`", t)} | {f"c{c}" for c in cols_of(t)})
    for m in re.finditer(r"\[\[(.+?)\]\]", t): out.add("[[" + m.group(1).lower() + "]]")
    return out


def resolve(chk, limit=WORDS_MAX):
    """the reading to serve: the sentences whose facts hold, then sentences dropped until the text fits the word limit
    — the ones that add least (no id of their own, nothing of a sub-line or a branch, no interpretation) first, and never
    an opening sentence whose paragraph would be left with its promise alone. An opener that leant on its paragraph's
    other sentences is checked again against what is left of them: the citation holds for the text served, not only for
    the text written -> (text or None, dict(dropped, dropped_len, unled, sections, words, why))"""
    F, X = chk.get("_F"), chk.get("_X")
    secs = []; nbad = 0
    for sec in chk["sections"]:
        keep = [st for st in sec["sents"] if st["ok"]]
        nbad += len(sec["sents"]) - len(keep)
        if keep: secs.append(dict(sec, sents=keep, n0=len(sec["sents"])))  # nothing of the section holds: the section goes
    rep = dict(dropped=nbad, dropped_len=0, unled=0, sections=[], words=0, repeats=0)
    seen_c = set()                                                        # a sentence whose numbers, genes and columns were
    for sec in secs:                                                      # all said before: a repeat, left out
        keep = []
        for st in sec["sents"]:
            c = _carries(st["text"])
            if c and c <= seen_c and len(sec["sents"]) > 1 and not (st is sec["sents"][0] and sec["tag"] in KEEP):
                rep["repeats"] += 1; continue
            seen_c |= c; keep.append(st)
        sec["sents"] = keep
    secs = [x for x in secs if x["sents"]]

    def prune():
        """after every drop: an opener no longer carried by its own paragraph, and a paragraph left with nothing but the
        promise its opener made"""
        nonlocal secs
        for sec in secs:
            st = sec["sents"][0] if sec["sents"] else None
            if not (st and st["opens"] and st.get("inherit") and F is not None and X is not None): continue
            left = [i for s2 in sec["sents"][1:] for i in s2["cites"]]
            if all(i in left for i in st["inherit"]): continue
            b, _n1, _n2 = _sent_bad(st["text"], st["cites"], F, X, [i for i in st["inherit"] if i in left])
            if b: sec["sents"] = sec["sents"][1:]; rep["dropped"] += 1
        secs = [s for s in secs if s["sents"]
                and not (len(s["sents"]) == 1 and s["sents"][0]["opens"] and s["n0"] > 1
                         and not re.search(r"\d", s["sents"][0]["text"]) and s["tag"] not in KEEP)]
    prune()
    tags = [s["tag"] for s in secs]
    rep.update(sections=tags, words=_words(render(secs)))
    if any(t not in tags for t in KEEP) or len(secs) < len(SECTIONS) - 1:
        rep["why"] = "too little of the reading is supported by the facts it cites"; return None, rep
    for _ in range(40):                                                   # the word limit: whole sentences at a time
        if _words(render(secs)) <= limit: break
        all_ = [(s, j, st) for s in secs for j, st in enumerate(s["sents"])]
        carries = {(id(s), j): _carries(st["text"]) for s, j, st in all_}
        cand = []
        for s, j, st in all_:
            if len(s["sents"]) == 1 and s["tag"] in KEEP: continue         # @start and @whole stay
            others = set().union(set(), *(v for k, v in carries.items() if k != (id(s), j)))
            uniq = len(carries[(id(s), j)] - others)                       # the numbers, genes and columns it alone carries
            sub = any("." in i for i in st["cites"]); interp = "[[" in st["text"]
            if j != len(s["sents"]) - 1 and (uniq or sub or interp): continue   # the tail of a paragraph, or a sentence
            if st["opens"] and len(s["sents"]) > 1: continue                     # that repeats what another one carries
            cand.append((uniq + 2 * sub + 2 * interp, DROP_ORDER.index(s["tag"]) if s["tag"] in DROP_ORDER else 9,
                         -_words(st["text"]), id(s), s, j))
        if not cand: break
        *_k, s, j = min(cand, key=lambda c: c[:4])
        del s["sents"][j]; rep["dropped_len"] += 1
        prune()
    text = render(secs)
    rep.update(words=_words(text), sections=[s["tag"] for s in secs], unled=sum(not s["sents"][0]["opens"] for s in secs),
               interp=len(re.findall(r"\[\[", text)),
               kept=[dict(tag=s["tag"], text=st["text"], cites=st["cites"], inherit=st.get("inherit") or [])
                     for s in secs for st in s["sents"]])
    cov = set()                                                           # the columns the reading served speaks of
    for s in secs:
        for st in s["sents"]:
            cov |= cols_of(st["text"]) | set().union(set(), *[X[i]["cols"] for i in st["cites"] if X and i in X and not X[i]["glob"]])
    rep["covered"] = len(cov)
    if any(t not in rep["sections"] for t in KEEP) or len(secs) < len(SECTIONS) - 1:
        rep["why"] = "too little of the reading is supported by the facts it cites"; return None, rep
    if rep["words"] < WORDS_MIN: rep["why"] = f"only {rep['words']} words of the reading are supported by the facts"; return None, rep
    if rep["words"] > limit: rep["why"] = f"{rep['words']} words after the sections were trimmed"; return None, rep
    return text, rep


def retry_cited(chk, rep, truncated=False):
    """what to tell the model about its rejected reading: the sentences whose facts are not in their citations, what
    the structure asks, and the length"""
    say = list(chk["structure"][:4])
    n = 0
    for sec in chk["sections"]:
        for st in sec["sents"]:
            if st["ok"] or n >= 6: continue
            n += 1
            why = "; ".join(dict.fromkeys(f"{b['text']}: {b['why']}" for b in st["bad"][:3]))
            say.append(f'in @{sec["tag"]}, "{st["text"][:90]}" {{{", ".join(st["cites"]) or "no id"}}} — {why}')
    if truncated: say.append(f"the text was cut at its length limit: write at most {WORDS_MAX} words")
    elif chk["words"] > WORDS_TARGET: say.append(f"the reading is {chk['words']} words: write {WORDS_MIN_ASK} to {WORDS_TARGET}, "
                                                "so that nothing has to be cut")
    if rep and rep.get("unled"): say.append("a paragraph is left without its opening sentence: write an opener that stands on the "
                                            "lines its own paragraph cites")
    if rep and rep.get("interp", 1) == 0 and chk.get("_F") is not None and "Q" in chk["_F"].ids:
        say.append("the reading names no biology: put one [[interpretation]] of 2 to 6 words on a group of [Q], from the words [Q] "
                   "gives it, naming that group's columns and citing its stretch line and [Q]")
    if rep and rep.get("why"): say.append(rep["why"])
    return ("Your reading was rejected: " + "; ".join(say) + ". Rewrite the whole reading, following the form, the citations "
            "and the rules: every number, gene name and column of a sentence standing in the facts lines that sentence cites, "
            "nothing from any other line.")


def reason(chk, truncated, rep=None):
    """a short account of a failed check, for the page (its details in `bad`)"""
    b = chk["bad"]
    if truncated and not b: return "the model's text was cut at its length limit"
    kinds = collections.Counter(x["kind"] for x in b)
    what = dict(number="a number the facts lines its sentence cites do not carry", scope="a number given to the wrong genes or columns",
                column="a gene or a column its sentence does not cite", direction="the comparison inside and outside regions of plasticity inverted",
                claim="a claim this window's data do not make", gene="a gene not in the lines its sentence cites",
                identifier="a name not in this window", interpretation="an interpretation the products and gene names do not carry",
                citation="a sentence written from facts it does not cite", wording="words about the facts rather than the window",
                format="a number badly written")
    what["number word"] = "a number in words"
    if not kinds and chk.get("structure"): return "the model's text did not keep the reading's sections after a second attempt"
    if not kinds and rep and rep.get("why"): return rep["why"]
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


def _attempt(r, F):
    """one answer of the model: tidied, parsed into its sections, every sentence checked against the lines it cites,
    then the reading that would be served (its unsupported sentences dropped, trimmed to the word limit)"""
    text = tidy(r["text"], F)
    chk = check_cited(text, F)
    out, rep = resolve(chk)
    cut = r["finish"] == "length"
    drops = [dict(tag=sec["tag"], text=st["text"][:200], cites=st["cites"], opens=st["opens"],
                  bad=[{k: v for k, v in b.items() if k != "context"} for b in st["bad"][:4]])
             for sec in chk["sections"] for st in sec["sents"] if not st["ok"]]
    return dict(r, raw=r["text"], text=out or text, annotated=text,
                check={k: v for k, v in chk.items() if k not in ("sections", "_F", "_X")},
                report=rep, dropped=drops, served=out, truncated=cut, _chk=chk,
                ok=bool(out) and not cut and not chk["structure"] and rep["dropped"] <= 1     # one sentence dropped, five
                and len(rep["sections"]) == len(SECTIONS) and not rep["unled"]                # sections left, every paragraph
                and (rep.get("interp", 0) >= 1 or "Q" not in F.ids))                              # led, one interpretation: no 2nd call


def generate(F, wid):
    """-> the cache entry: status llm (the sections that hold against their citations) or fallback (too little of the
    text was supported, twice)"""
    deadline = time.time() + BUDGET
    sysmsg, sha = system_msg()
    msgs = [dict(role="system", content=sysmsg), dict(role="user", content=user_msg(F, wid))]
    attempts = []
    for i in range(2):
        try: r = llm(msgs, deadline)
        except Upstream:
            if not attempts: raise
            break                                                            # the retry could not be made: fallback
        a = _attempt(r, F); attempts.append(a)
        if a["ok"]: break                                                    # only its length was over: the trimming fits it
        if i == 0:
            if deadline - time.time() < 12: break
            msgs += [dict(role="assistant", content=r["text"]),
                     dict(role="user", content=retry_cited(a["_chk"], a["report"], a["truncated"]))]
    clean = [a for a in attempts if a["ok"]]
    served = [a for a in attempts if a["served"] and not a["truncated"]]
    best = (clean[0] if clean else min(served, key=lambda a: (a["report"]["dropped"] + a["report"]["unled"],
            -a["report"].get("interp", 0), -a["report"].get("covered", 0))) if served else attempts[-1])
    ok = bool(best["served"]) and not best["truncated"]
    usage = dict(prompt_tokens=sum(a["usage"]["prompt_tokens"] or 0 for a in attempts),
                 completion_tokens=sum(a["usage"]["completion_tokens"] or 0 for a in attempts),
                 cost_usd=round(sum(a["usage"]["cost_usd"] for a in attempts), 6), ms=sum(a["ms"] for a in attempts))
    c = best["check"]; rep = best["report"]
    e = dict(status="llm" if ok else "fallback", model=MODEL, provider=best["provider"], created=time.time(), usage=usage,
             attempts=[{k: v for k, v in a.items() if k != "_chk"} for a in attempts],
             checked=dict(ok=ok, numbers=c["numbers"], names=c["names"], words=rep["words"] if ok else c["words"],
                          attempts=len(attempts), sentences=c["sentences"], cited=c["cited"], skill=sha,
                          dropped=rep["dropped"], trimmed=rep["dropped_len"], repeats=rep.get("repeats", 0), unled=rep["unled"],
                          sections=rep["sections"],
                          interp=rep.get("interp", 0), covered=rep.get("covered", 0)))
    if ok: e.update(text=best["served"], html=to_html(best["served"], F))
    else:
        e["reason"] = reason(c, best["truncated"], rep)
        e["checked"]["bad"] = [{k: v for k, v in b.items() if k != "context"} for b in c["bad"][:12]]
        e["checked"]["structure"] = c["structure"][:4]
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
    F = facts(D, wid); sysmsg, sha = system_msg()                         # the skill's sha is part of the key: a new skill, a new reading
    h = hashlib.sha1((PROMPT_VERSION + MODEL + sha + sysmsg + user_msg(F, wid)).encode()).hexdigest()[:12]
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
        e.update(window=wid, label=F.label, facts_hash=h, prompt_version=PROMPT_VERSION, skill_sha=skill()[1], facts=F.txt())
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
               prompt=dict(version=PROMPT_VERSION, skill=skill()[1], skill_file=_SKILL[2], temperature=TEMPERATURE, words=WORDS_MAX),
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
        e.update(window=wid, label=F.label, facts_hash=h, prompt_version=PROMPT_VERSION, skill_sha=skill()[1], facts=F.txt())
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
        chk = check_cited(tidy(open(sys.argv[3]).read(), F), F); out, rep = resolve(chk)
        for sec in chk["sections"]:
            for st in sec["sents"]:
                print(f"{'ok  ' if st['ok'] else 'DROP'} @{sec['tag']:8s} {{{', '.join(st['cites'])}}} {st['text'][:110]}")
                for b in st["bad"]: print(f"       - {b['kind']}: {b['text']} — {b['why']}")
        print(json.dumps(dict({k: v for k, v in chk.items() if k not in ("sections", "bad", "notes")}, report=rep), indent=1))
        old = check(render(chk["sections"]), F)                           # the whole-text check of r7, for comparison
        print(f"the whole-text check: {'ok' if old['ok'] else 'not ok'}, {len(old['bad'])} item(s) "
              + ", ".join(sorted({b['kind'] for b in old['bad']})))
        print("\n" + (out or "(nothing servable)"))
    elif len(sys.argv) >= 2 and sys.argv[1] == "skill":
        t, sha = skill(); print(f"[{SKILL_PATH} {'read' if _SKILL[2] else 'MISSING'}, sha {sha}, {len(t)} characters, ~{len(t) // 4} tokens]")
        sysmsg, _ = system_msg(); print(f"[system message {len(sysmsg)} characters, ~{len(sysmsg) // 4} tokens]")
    elif len(sys.argv) >= 2 and sys.argv[1] == "pregen": pregen(sys.argv[2:])
    else: print(__doc__)
