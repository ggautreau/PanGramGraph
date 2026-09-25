"""What the page says about the pangenome, its genomes and the chromosome, beyond the graph itself.

  pgb/pangenome_info.json  the pangenome as PanGBank built it (PPanGGOLiN's own counts, from the
                           file's info table): genomes, genes, families by partition, edges,
                           regions of plasticity, spots, modules
  pgb/genome_meta.json     per genome, what one searches a strain by: strain, organism when not
                           plain E. coli, serotype, sequence type, host, isolation source, country,
                           year, assembly level, contigs, size
  pgb/chrom_map.json       the complete chromosomes as a map from dnaA: their length in genes, the
                           backbone anchors, every gene name the local server can go to, and the
                           share of genes in a region of plasticity along the way

    python3 pgb_page_meta.py            # needs pgb/ecoli_11587.h5 and the chromosome data of pgb_region.py
"""
import json, re, collections, warnings
import numpy as np, tables
warnings.simplefilter("ignore")
H5 = "pgb/ecoli_11587.h5"
dec = lambda x: x.decode() if isinstance(x, bytes) else x


def pangenome_info():
    h = tables.open_file(H5); a = h.root.info._v_attrs; s = h.root.status._v_attrs
    v = lambda k: (a[k].item() if hasattr(a[k], "item") else a[k])
    out = dict(pangenome=11587, ppanggolin=str(s["version"]), genomes=v("numberOfGenomes"), genes=v("numberOfGenes"),
               families=v("numberOfClusters"), persistent=v("numberOfPersistent"), shell=v("numberOfShell"), cloud=v("numberOfCloud"),
               edges=v("numberOfEdges"), rgp=v("numberOfRGP"), spots=v("numberOfSpots"), modules=v("numberOfModules"),
               families_in_modules=v("numberOfFamiliesInModules"))
    h.close()
    assert out["persistent"] + out["shell"] + out["cloud"] == out["families"]
    return out


NA = {"", "na", "n/a", "none", "missing", "unknown", "not known", "not collected", "not applicable", "not provided",
      "not available", "-", "null", "unspecified", "other", "restricted access"}
HOST = {"homo sapiens": "human", "human": "human", "bos taurus": "cattle", "cattle": "cattle", "cow": "cattle", "bovine": "cattle",
        "sus scrofa": "pig", "sus scrofa domesticus": "pig", "swine": "pig", "porcine": "pig", "pig": "pig",
        "gallus gallus": "chicken", "gallus gallus domesticus": "chicken", "chicken": "chicken", "broiler": "chicken",
        "ovis aries": "sheep", "sheep": "sheep", "capra hircus": "goat", "goat": "goat", "canis lupus familiaris": "dog",
        "canis familiaris": "dog", "dog": "dog", "felis catus": "cat", "cat": "cat", "equus caballus": "horse", "horse": "horse",
        "mus musculus": "mouse", "mouse": "mouse", "anas platyrhynchos": "duck", "duck": "duck", "meleagris gallopavo": "turkey",
        "turkey": "turkey", "oryctolagus cuniculus": "rabbit", "rabbit": "rabbit"}
SERO_O = re.compile(r"\b(O(?:\d+[a-z]*|NT|R|rough)(?:/O?\d+[a-z]*)?(?::(?:H\d+|H-|HNM|NM|H\?))?)(?![\w-])")
SERO = re.compile(SERO_O.pattern + r"|\b(H\d+)\b")


def clean(x):
    x = re.sub(r"\s+", " ", str(dec(x) if x is not None else "")).strip().strip(";,")
    return "" if x.lower() in NA or x.lower().startswith(("not available", "not applicable", "not collected", "missing")) else x


def genome_meta():
    h = tables.open_file(H5)
    af = h.root.metadata.genomes.annotation_file.read(); wf = h.root.metadata.genomes.pangbank_wf.read(); h.close()
    assert (af["ID"] == wf["ID"]).all()
    LEVEL = {"Complete Genome": 0, "Chromosome": 1, "Scaffold": 2, "Contig": 3}
    out = {}
    for r, w in zip(af, wf):
        acc = dec(r["ID"]); org = clean(w["ncbi_organism_name"]) or clean(r["organism"])
        strain = clean(r["strain"]) or clean(w["ncbi_strain_identifiers"]) or clean(r["isolate"]) or clean(w["ncbi_isolate"])
        sero = next((m.group(1) or m.group(2) for m in (SERO.search(clean(r[c])) for c in ("serotype", "serovar")) if m), "")
        if not sero:                                   # "Escherichia coli O157:H7 str. Sakai": an O antigen only
            m = SERO_O.search(org); sero = m.group(1) if m else ""
        st = (re.search(r"\bST ?(\d+)\b", clean(r["genotype"])) or [None, ""])[1]
        host = clean(r["host"]); host = HOST.get(host.lower(), host)
        src = clean(r["isolation_source"]) or clean(w["ncbi_isolation_source"])
        ctry = (clean(r["geo_loc_name"]) or clean(w["ncbi_country"])).split(":")[0].strip()
        year = (re.search(r"\b(19\d\d|20\d\d)\b", clean(r["collection_date"])) or [None, ""])[1]
        out[acc] = [strain, "" if org == "Escherichia coli" else org, sero, "ST" + st if st else "", host, src, ctry,
                    int(year) if year else 0, LEVEL.get(clean(w["ncbi_assembly_level"]), 3), int(w["contig_count"]),
                    round(int(w["genome_size"]) / 1e6, 2), 1 if clean(r["type_material"]) else 0]
    return out


def chrom_map():
    import pgb_region as R
    L = int(np.median(R.GL))
    anchors = [[R.fam_name_label(int(f))[0], int(round(p))] for f, p in zip(R.BB, R.BB_POS)]
    # every gene name, read as its most common family, as the server's resolve() does
    m = R.NAME > 0; key = R.NAME[m].astype(np.int64) * R.NF + R.FAM[m]
    uk, cnt = np.unique(key, return_counts=True); nk, fk = uk // R.NF, uk % R.NF
    o = np.lexsort((-cnt, nk)); first = o[np.r_[True, nk[o][1:] != nk[o][:-1]]]
    best = dict(zip(nk[first].tolist(), fk[first].tolist()))
    of = np.argsort(R.FAM, kind="stable"); sf = R.FAM[of]
    genes = []
    for k, f in best.items():
        a, b = np.searchsorted(sf, f), np.searchsorted(sf, f, "right"); ix = of[a:b]
        genes.append([R.NAMES[k], int(round(float(np.median(R.LOC[ix])))), int(len(np.unique(R.GID[ix])))])
    genes.sort(key=lambda g: g[0].lower())
    # the share of genes in a region of plasticity, by bins of 20 genes from dnaA
    B = 20; b = R.LOC // B; n = np.bincount(b); r = np.bincount(b, weights=R.RGP.astype(float))
    keep = n >= R.NG // 2
    rgp = [round(float(x), 3) for x in (r / np.maximum(n, 1))[: int(np.flatnonzero(keep).max()) + 1]]
    return dict(chromosomes=int(R.NG), length=L, bin=B, rgp=rgp, anchors=anchors, genes=genes)


if __name__ == "__main__":
    P = pangenome_info(); json.dump(P, open("pgb/pangenome_info.json", "w"), indent=1); print(P)
    G = genome_meta(); json.dump(G, open("pgb/genome_meta.json", "w"), separators=(",", ":"), ensure_ascii=False)
    lv = collections.Counter(v[8] for v in G.values()); fill = lambda i: sum(1 for v in G.values() if v[i])
    print(f"genome_meta: {len(G)} genomes, levels {dict(sorted(lv.items()))}; strain {fill(0)}, organism {fill(1)}, serotype {fill(2)}, "
          f"ST {fill(3)}, host {fill(4)}, source {fill(5)}, country {fill(6)}, year {fill(7)}")
    C = chrom_map(); json.dump(C, open("pgb/chrom_map.json", "w"), separators=(",", ":"), ensure_ascii=False)
    print(f"chrom_map: {C['chromosomes']} chromosomes of {C['length']} genes (median), {len(C['anchors'])} anchors, "
          f"{len(C['genes'])} gene names, {len(C['rgp'])} bins of {C['bin']}")
