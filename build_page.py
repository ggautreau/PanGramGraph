"""Assemble the page, standalone/<name in lower case>.html: page_template.html + the 2,002-genome graph
data (pgb/graph_pgb.json) + the coupled regions (pgb/region_sets.json, pg_region_sets.py) + the base
style, taken from the 11-strain page, pgb/origin-fork-11strains.html.

Earlier names (OLD) and standalone/index.html become redirects to it that keep the #hash,
so old links and http://localhost:8765/ open the page.

    python3 build_page.py
"""
import json
NAME = "PanGramGraph"                 # the page's name; its file is the name in lower case
SLUG = NAME.lower()
OLD = ["origin-fork"]                 # earlier names
T=open("page_template.html").read(); old=open("pgb/origin-fork-11strains.html").read()
style=old[old.index("<style>"):old.index("</style>")+len("</style>")]
data=json.dumps(json.load(open("pgb/graph_pgb.json")),separators=(",",":"))
import os
sets=json.dumps(json.load(open("pgb/region_sets.json")),separators=(",",":")) if os.path.exists("pgb/region_sets.json") else "null"
page=T.replace("__STYLE__",style).replace("__DATA__",data).replace("__SETS__",sets).replace("__NAME__",NAME).replace("__SLUG__",SLUG)
assert not [x for x in ("__STYLE__","__DATA__","__SETS__","__NAME__","__SLUG__") if x in page]
open(f"standalone/{SLUG}.html","w").write(page)
go=(f'<!doctype html><meta charset="utf-8"><title>{NAME}</title><meta http-equiv="refresh" content="0;url={SLUG}.html">'
    f'<script>location.replace("{SLUG}.html"+location.hash)</script><p><a href="{SLUG}.html">{NAME}</a></p>')
for o in OLD+["index"]: open(f"standalone/{o}.html","w").write(go)
print(f"standalone/{SLUG}.html {len(page)/1e6:.2f} MB; redirects from {', '.join(o + '.html' for o in OLD + ['index'])}")
