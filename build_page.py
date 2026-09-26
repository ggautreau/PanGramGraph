"""Assemble the page, standalone/<name in lower case>.html: page_template.html + the 2,002-genome graph
data (pgb/graph_pgb.json) + the coupled regions + the pangenome's numbers, the genomes' metadata and the chromosome map
(pgb/pangenome_info.json, genome_meta.json, chrom_map.json, pgb_page_meta.py) + the base style, taken from the
11-strain page, pgb/origin-fork-11strains.html.

The coupled regions come in two layers, each injected compactly (fields the page does not read are dropped, presence strings run-length coded):
  long   pgb/region_sets.json (pg_region_sets.py, > 500 genes apart), flags pgb/link_flags.json (pg_link_flags.py)
  close  pgb/region_sets_close.json (50-500 genes apart), flags pgb/link_flags_close.json
with, per layer, the model's verdict per link from the in-silico knockout (pg_knockout.py --all):
pgb/knock/all_arcs.json (long) and pgb/knock/close_arcs.json (close) when they exist and describe the same sets file
(md5); otherwise no verdicts and the page says they are being computed. --dev-arcs takes the smoke runs instead
(pgb/knock/smoke_all2_arcs.json, smoke_close2_arcs.json: 3 genomes, for developing the page only; the page names the
run it shows). The genomes carrying each element at its spot (the influence picker) come from pgb/region_carriers.json
(pg_region_carriers.py).

Earlier names (OLD) and standalone/index.html become redirects to it that keep the #hash,
so old links and http://localhost:8765/ open the page.

    python3 build_page.py [--dev-arcs] [--arcs long=PATH] [--arcs close=PATH]
"""
import json, os, re, sys, hashlib, math
NAME = "PanGramGraph"                 # the page's name; its file is the name in lower case
SLUG = NAME.lower()
OLD = ["origin-fork"]                 # earlier names
DEV = "--dev-arcs" in sys.argv
# --arcs long=PATH (or close=PATH): another run's arcs file for a layer (its _links.parquet and _summary.json beside it),
# to check the page on other verdicts; a tag starting with "smoke" is labelled on the page as not a result
ARCS = dict(a.split("=", 1) for i, a in enumerate(sys.argv) if i and sys.argv[i - 1] == "--arcs")
T=open("page_template.html").read(); old=open("pgb/origin-fork-11strains.html").read()
style=old[old.index("<style>"):old.index("</style>")+len("</style>")]
data=json.dumps(json.load(open("pgb/graph_pgb.json")),separators=(",",":"))
small=lambda f: json.dumps(json.load(open(f)),separators=(",",":"),ensure_ascii=False)   # pgb_page_meta.py


def num(x, sig=3):
    """a number with `sig` significant digits (None stays None): the page shows 2-3 digits"""
    if x is None or (isinstance(x, float) and not math.isfinite(x)): return None
    if isinstance(x, bool) or isinstance(x, int): return x
    if x == 0: return 0
    d = max(0, sig - 1 - int(math.floor(math.log10(abs(x)))))
    v = round(x, d); return int(v) if float(v).is_integer() else v


def rle(p):
    """a presence string (digits 0-9, one per genome) with runs of 3-54 equal digits written as the digit and a letter,
    A-Z for 3-28 and a-z for 29-54 (the page expands them back): 234 kB -> 91 kB for the two layers"""
    o = []; i = 0
    while i < len(p):
        j = i
        while j < len(p) and p[j] == p[i] and j - i < 54: j += 1
        n = j - i
        o.append(p[i] * n if n < 3 else p[i] + (chr(65 + n - 3) if n < 29 else chr(97 + n - 29)))
        i = j
    return "".join(o)


def lean(d):
    """drop the empty fields of a verdict record (the page reads a missing field as none)"""
    return {k: v for k, v in d.items() if v not in (None, "", [], [None, None, None]) and not (k == "wu" and v == 0)}


LAYERS = [("long", "pgb/region_sets.json", "pgb/link_flags.json", "all", "smoke_all2"),
          ("close", "pgb/region_sets_close.json", "pgb/link_flags_close.json", "close", "smoke_close2")]
CARR = json.load(open("pgb/region_carriers.json")) if os.path.exists("pgb/region_carriers.json") else {}
VCODE = {"knows": "k", "not specific": "s", "not specific (opposite)": "so", "opposite": "o", "none": "n", "nt": "t"}
RANGE_CAVEAT = "tested at a median "          # pg_knockout's caveat for 'none' and 'nt' beyond 500 genes: said for 'none' only


def fin(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def failed(r):
    """what the verdict of pg_knockout.py (verdict()) found missing, from the same numbers (a NaN fails every test):
    own direction  q_exp (calibrated q < 0.05), sign (sign test p < 0.05), halves (median > 0 in both lineage halves);
    specificity    spec_z (q_cal_spec), spec_raw (q_cal_spec_raw); the opposite way: q_opp, halves_opp, spec_raw_opp.
    -> (verdict recomputed, codes of the conditions that failed on the way to it)"""
    lt = lambda k: fin(r.get(k)) and r[k] < 0.05
    h0, h1 = r.get("half0_median"), r.get("half1_median")
    own_f = [c for c, ok in (("q_exp", lt("q_cal_exp")), ("sign", lt("p_sign")), ("halves", fin(h0) and fin(h1) and h0 > 0 and h1 > 0)) if not ok]
    spec_f = [c for c, ok in (("spec_z", lt("q_cal_spec")), ("spec_raw", lt("q_cal_spec_raw"))) if not ok]
    if not own_f: return ("knows", []) if not spec_f else ("not specific", spec_f)
    opp_f = [c for c, ok in (("q_opp", lt("q_cal_opp")), ("halves_opp", fin(h0) and fin(h1) and h0 < 0 and h1 < 0)) if not ok]
    if not opp_f: return ("opposite", []) if lt("q_cal_spec_raw_opp") else ("not specific (opposite)", ["spec_raw_opp"])
    return "none", own_f + opp_f


def model_of(path, md5, tag, links, regs):
    """the verdicts of one knockout run, keyed by link index, or None (absent, unreadable, made for another sets file, or
    not covering exactly its links: a missing link must not pass for 'not testable'). The q values each verdict used
    (the opposite direction's too) come from the run's {tag}_links.parquet."""
    if not os.path.exists(path): return None
    bad = lambda why: print(f"  !! {path}: {why}: NOT USED (the page says the verdicts are being computed)") or None
    try: J = json.load(open(path)); M = J["meta"]; A = J["arcs"]
    except (ValueError, KeyError) as e: return bad(f"unreadable ({type(e).__name__}: {e}); still being written?")
    if M.get("sets_md5") != md5:
        return bad(f"made for another version of {M.get('sets')} (md5 {str(M.get('sets_md5', ''))[:12]} vs {md5[:12]})")
    if M.get("links") not in (None, len(links)): return bad(f"meta says {M.get('links')} links, the sets file has {len(links)}")
    lis = [a.get("li") for a in A]
    if sorted(lis) != list(range(len(links))):
        return bad(f"{len(set(lis))} distinct links of the {len(links)} (missing {len(set(range(len(links))) - set(lis))}, duplicated {len(lis) - len(set(lis))})")
    wrong = [a["li"] for a in A if (a.get("a"), a.get("b"), a.get("sign")) != (links[a["li"]]["a"], links[a["li"]]["b"], links[a["li"]]["sign"])]
    if wrong: return bad(f"{len(wrong)} links do not match the sets file's ends and sign (first: {wrong[:5]})")
    unk = sorted({a.get("verdict") for a in A} - set(VCODE))
    if unk: return bad(f"unknown verdict(s) {unk}")
    pq = path.replace("_arcs.json", "_links.parquet"); P = {}
    if os.path.exists(pq):
        import pandas as pd
        T = pd.read_parquet(pq)
        cols = ["q_cal_exp", "q_cal_opp", "q_cal_spec", "q_cal_spec_raw", "q_cal_spec_raw_opp", "p_sign", "half0_median", "half1_median",
                "half0_n_lin", "half1_n_lin", "verdict"]
        for r in T[["li", "a", "b", "sign"] + cols].to_dict("records"):
            li = int(r["li"])
            l = links[li] if 0 <= li < len(links) else None      # the parquet names the ends by label (40 characters)
            if l and (r["a"], r["b"], int(r["sign"])) == (regs[l["a"]]["label"][:40], regs[l["b"]]["label"][:40], l["sign"]):
                P[li] = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in r.items()}
        if len(P) != len(links): return bad(f"{pq} covers {len(P)} of the {len(links)} links")
    else: print(f"  {pq} missing: the q values of the opposite direction and the failed conditions are not shown")
    sf = path.replace("_arcs.json", "_summary.json"); S = json.load(open(sf)) if os.path.exists(sf) else {}
    cav = []; cix = {}; arcs = {}; mism = []
    for a in A:
        v = VCODE[a["verdict"]]; ci = []
        for c in a.get("caveats", []):
            if v == "t" and c.startswith(RANGE_CAVEAT): continue
            c = re.sub(r"(?<![\w.])(\d{4,})(?![\w.])", lambda m: f"{int(m.group(1)):,}", c).replace("'none'", "“none”")
            if c not in cix: cix[c] = len(cav); cav.append(c)
            ci.append(cix[c])
        rec = dict(v=v, why=a.get("nt_why") or "", nu=a["n_units"], nt=a["n_tested"], nl=a["n_lin"],
                   d=[num(a.get("dist_q10")), num(a.get("dist_median")), num(a.get("dist_q90"))],
                   w=num(a.get("share_within_500"), 2), wu=a.get("units_within_500", 0),
                   e=num(a.get("effect_z"), 2), mde=num(a.get("mde_z"), 2),
                   q=[num(a.get("q_cal_exp"), 2), num(a.get("q_cal_spec"), 2), num(a.get("q_cal_spec_raw"), 2)], c=ci)
        p = P.get(a["li"])
        if p:
            if v != "t":
                vv, fl = failed(p)
                if vv != a["verdict"]: mism.append(a["li"])
                rec.update(qo=[num(p["q_cal_opp"], 2), num(p["q_cal_spec_raw_opp"], 2)], ps=num(p["p_sign"], 2),
                           h=[num(p["half0_median"], 2), num(p["half1_median"], 2)], hn=[p["half0_n_lin"], p["half1_n_lin"]], f=fl)
        arcs[a["li"]] = lean(rec)
    if mism: print(f"  !! {path}: {len(mism)} verdicts differ from their conditions recomputed from {pq} (first: {mism[:5]})")
    meta = dict(tag=M["tag"], dev=M["tag"].startswith("smoke"), genomes=S.get("genomes") or (M.get("arms") or {}).get("genomes"),
                units=S.get("units"), units_tested=M.get("units_tested"), units_within_500=M.get("units_tested_within_500"),
                verdicts=M.get("verdicts"), rules=M.get("rules"), reach=M.get("model_reach"), caveats=cav, qopp=bool(P))
    return meta, arcs


def rkey(fams, taken):
    """a region's address on the page (#rs=e<key>): a hash of its families, stable when the sets are regenerated"""
    h = hashlib.md5("\n".join(sorted(fams)).encode()).hexdigest()
    for n in range(6, 33):
        if h[:n] not in taken: taken.add(h[:n]); return h[:n]


def layer(key, sets, flags, tag, smoke):
    raw = open(sets, "rb").read(); md5 = hashlib.md5(raw).hexdigest(); RS = json.loads(raw)
    FJ = json.load(open(flags)) if os.path.exists(flags) else {}
    if FJ.get("sets_md5") not in (None, md5): sys.exit(f"{flags} was made for another version of {sets} (md5): rerun pg_link_flags.py")
    FL = {r["li"]: r for r in FJ.get("links", [])}
    for li, l in enumerate(RS["links"]):
        if li in FL: assert (FL[li]["a"], FL[li]["b"], FL[li]["sign"]) == (l["a"], l["b"], l["sign"]), f"{flags} does not describe {sets}"
    real = ARCS.get(key, f"pgb/knock/{tag}_arcs.json")
    mo = (model_of(real, md5, tag, RS["links"], RS["regions"])
          or (model_of(f"pgb/knock/{smoke}_arcs.json", md5, smoke, RS["links"], RS["regions"]) if DEV else None))
    taken = set()
    regions = [dict(id=r["id"], k=rkey(r["families"], taken), spot=r["spot"], start=r["start"], end=r["end"], center=r["center"], n=r["n"],
                    label=r["label"], genes=r["genes"], products=r["products"], mobile=r["mobile"], freq=r["freq"],
                    presence=rle(r["presence"]), set=r["set"], fams=r["families"] if key == "close" else None) for r in RS["regions"]]
    if key == "close":                     # the same element in the long-range set (same families): the server's region id
        L = json.load(open("pgb/region_sets.json"))["regions"]; byf = {frozenset(r["families"]): r["id"] for r in L}
        for r in regions: r["twin"] = byf.get(frozenset(r.pop("fams")))
    else:
        for r in regions: r.pop("fams")
    links = []
    for li, l in enumerate(RS["links"]):
        f = FL.get(li, {})
        x = dict(a=l["a"], b=l["b"], sign=l["sign"], pairs=l["pairs"], z=l["z"], distance=l["distance"], shared=l["shared"])
        if l.get("geometry") == "median": x["geo"] = "m"      # read on median positions: fewer than 10 genomes carry both
        if f:
            x.update(rep=f["rep_lineage"], repP=f["rep_lineage_pairs"], one=f["one_site"], tract=f["tract"], sep=f["separate"],
                     comp=f["site_competition"], cdist=num(f.get("carrier_dist")), same=num(f.get("same_spot"), 2),
                     carriers=num(f.get("carriers")))
        if mo and li in mo[1]: x["m"] = mo[1][li]
        links.append(x)
    M = RS["meta"]
    meta = {k: M[k] for k in ("mindist", "maxdist", "pairs", "expected_by_chance", "replicated_by_chance", "regions", "links",
                              "sets", "genomes", "chrom_len", "halves") if k in M}
    meta["rep_lineage"] = sum(1 for x in links if x.get("rep")); meta["one_site"] = sum(1 for x in links if x.get("one"))
    meta["flags"] = bool(FL); meta["md5"] = md5
    C = CARR.get(key)
    carriers = C["carriers"] if C and C["md5"] == md5 else None
    if C and not carriers: print(f"  pgb/region_carriers.json: made for another version of {sets}: rerun pg_region_carriers.py")
    print(f"  {key}: {len(regions)} elements, {len(links)} links, {meta['rep_lineage']} replicated in lineage-disjoint halves, "
          f"{meta['one_site']} one element at several spots; verdicts: "
          + (f"{mo[0]['tag']} ({mo[0]['genomes']} genomes) {mo[0]['verdicts']}" if mo else "being computed")
          + ("" if carriers else "; no carriers file"))
    return dict(meta=meta, regions=regions, links=links, sets=RS["sets"], model=mo[0] if mo else None, carriers=carriers), RS


LY = {}; base = None
for key, *rest in LAYERS:
    if not os.path.exists(rest[0]): continue
    LY[key], RS = layer(key, *rest)
    if base is None: base = RS
    else: assert [g["acc"] for g in RS["genomes"]] == [g["acc"] for g in base["genomes"]] and RS["bounds"] == base["bounds"]
rsets = (json.dumps(dict(genomes=[[g["acc"], g.get("strain", ""), g["cl"]] for g in base["genomes"]], bounds=base["bounds"],
                         layers=LY), separators=(",", ":"), ensure_ascii=False) if LY else "null")
page=(T.replace("__STYLE__",style).replace("__DATA__",data).replace("__RSETS__",rsets).replace("__NAME__",NAME).replace("__SLUG__",SLUG)
       .replace("__PAN__",small("pgb/pangenome_info.json")).replace("__GMETA__",small("pgb/genome_meta.json")).replace("__CMAP__",small("pgb/chrom_map.json"))
       .replace("__PGBID__",json.dumps(json.load(open("pgb/pangbank_genome_ids.json"))["ids"] if os.path.exists("pgb/pangbank_genome_ids.json") else {},separators=(",",":"))))
assert not [x for x in ("__STYLE__","__DATA__","__RSETS__","__NAME__","__SLUG__","__PAN__","__GMETA__","__CMAP__","__PGBID__") if x in page]
open(f"standalone/{SLUG}.html","w").write(page)
go=(f'<!doctype html><meta charset="utf-8"><title>{NAME}</title><meta http-equiv="refresh" content="0;url={SLUG}.html">'
    f'<script>location.replace("{SLUG}.html"+location.hash)</script><p><a href="{SLUG}.html">{NAME}</a></p>')
for o in OLD+["index"]: open(f"standalone/{o}.html","w").write(go)
print(f"standalone/{SLUG}.html {len(page)/1e6:.2f} MB (coupled regions {len(rsets)/1e3:.0f} kB); "
      f"redirects from {', '.join(o + '.html' for o in OLD + ['index'])}")
