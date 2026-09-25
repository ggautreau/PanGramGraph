"""One PanGBank pangenome file -> what the PanGBank-wide PanGramGraph run keeps.

Every genome, every contig, every gene (CDS), in genome order:
  contigs   longest first (in a complete genome, the chromosome); a circular contig carrying the
            dnaA family starts at it and reads in dnaA's direction of transcription, a linear one
            only turns to that direction; every other contig keeps its own order
  genes     PPanGGOLiN family, distinct protein (translated with the gene's own genetic code,
            first residue M, as in the E. coli run), whether panRGP puts it in a region of
            plasticity and in which spot, gene name, strand relative to the reading direction
Output in out/<id>/: genes.npz, meta.json, proteins.txt (removed once embedded by gpu.py),
extract.ok. With --delete the PanGBank file is removed once the output is written.

    python extract.py raw/11587.h5 [--out out] [--delete]
    python extract.py --loop [--delete]        # a worker: takes downloaded files until none is left
"""
import os, json, time, glob, collections, argparse, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd, tables
from Bio.Seq import Seq
warnings.simplefilter("ignore")
ROOT = os.environ.get("PGG_ROOT", ".")


def translate(args):
    """distinct DNA sequences -> proteins, with each one's genetic code"""
    dna, codes = args
    out = []
    for d, c in zip(dna, codes):
        p = str(Seq(d.decode()).translate(table=int(c) or 11)).rstrip("*")
        out.append(("M" + p[1:]) if p else "M")
    return out


def extract(path, outdir, workers=8):
    t0 = time.time(); pgid = os.path.basename(path).split(".")[0]
    h = tables.open_file(path); A = h.root.annotations
    # genes and their data
    g_id = A.genes.read(field="ID"); g_ctg = A.genes.read(field="contig").astype(np.int64)
    g_gd = A.genes.read(field="genedata_id"); gd = A.genedata
    gd_id = gd.read(field="genedata_id"); o = np.argsort(gd_id); ix = o[np.searchsorted(gd_id[o], g_gd)]
    pos = gd.read(field="position")[ix]; strand = gd.read(field="strand")[ix]
    gname = gd.read(field="name")[ix]; gcode = gd.read(field="genetic_code")[ix]
    n = len(g_id)
    # families
    info = h.root.geneFamiliesInfo.read(); fam_names = [x.decode() for x in info["name"]]
    fam_part = [x.decode()[:1].upper() for x in info["partition"]]
    gf = h.root.geneFamilies.read()
    gfam = pd.Index(fam_names).get_indexer([x.decode() for x in gf["geneFam"]])
    g_fam = gfam[pd.Index(gf["gene"]).get_indexer(g_id)].astype(np.int32)
    # regions of plasticity and spots (a small pangenome may have none)
    in_rgp = np.zeros(n, bool); spot = np.full(n, -1, np.int32)
    if "RGP" in h.root and h.root.RGP.nrows:
        rg = h.root.RGP.read()
        spot_of = dict(zip(h.root.spots.read()["RGP"], h.root.spots.read()["spot"])) if "spots" in h.root and h.root.spots.nrows else {}
        s = pd.Series(rg["RGP"], index=rg["gene"]); s = s[~s.index.duplicated()]
        gi = s.index.get_indexer(g_id); in_rgp = gi >= 0
        spot[in_rgp] = [spot_of.get(x, -1) for x in s.values[gi[in_rgp]]]
    # contigs and genomes
    ct = A.contigs.read(); c_row = {int(i): k for k, i in enumerate(ct["ID"])}
    c_gen = [x.decode() for x in ct["genome"]]; c_len = ct["length"].astype(np.int64); c_circ = ct["is_circular"].astype(bool)
    genomes = [x.decode() for x in A.genomes.read(field="name")]
    meta_g = {}
    try:
        mw = h.root.metadata.genomes.pangbank_wf.read(); cols = mw.dtype.names
        for r in mw:
            g = r["ID"].decode()
            meta_g[g] = {k: (r[c].decode() if isinstance(r[c], bytes) else r[c].item()) for k, c in
                         (("level", "ncbi_assembly_level"), ("strain", "ncbi_strain_identifiers"), ("org", "ncbi_organism_name"))
                         if c in cols}
    except Exception:
        pass
    # the dnaA family: the family most genes named dnaA belong to
    dn = g_fam[gname == b"dnaA"]; FA = int(np.bincount(dn[dn >= 0]).argmax()) if (dn >= 0).any() else -1
    # genes of each contig in position order
    order = np.lexsort((pos, g_ctg)); sc = g_ctg[order]
    starts = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]]); ends = np.r_[starts[1:], len(order)]
    by_ctg = {int(sc[a]): order[a:b] for a, b in zip(starts, ends)}
    ctg_of_gen = collections.defaultdict(list)
    for cid in by_ctg: ctg_of_gen[c_gen[c_row[cid]]].append(cid)
    idx, rel, goff, cstart, cgen, ccirc, clen, gmeta = [], [], [0], [0], [], [], [], []
    for gi_, g in enumerate(genomes):
        cids = sorted(ctg_of_gen.get(g, []), key=lambda c: -c_len[c_row[c]]); oriented = False; ng = 0
        for c in cids:
            L = by_ctg[c]; d = np.where(strand[L] == b"+", 1, -1).astype(np.int8)
            a = np.flatnonzero(g_fam[L] == FA) if FA >= 0 else []
            if len(a) and not oriented:
                k = int(a[0]); fw = d[k] == 1; m = len(L)
                # a complete genome's chromosome is circular even when the file does not say so
                circ = c_circ[c_row[c]] or (meta_g.get(g, {}).get("level") == "Complete Genome" and c == cids[0])
                if circ: sel = (k + (1 if fw else -1) * np.arange(m)) % m
                else: sel = np.arange(m) if fw else np.arange(m)[::-1]
                L = L[sel]; d = d[sel] * (1 if fw else -1); oriented = True
            idx.append(L); rel.append(d); ng += len(L)
            cstart.append(cstart[-1] + len(L)); cgen.append(gi_); ccirc.append(bool(c_circ[c_row[c]])); clen.append(int(c_len[c_row[c]]))
        goff.append(goff[-1] + ng)
        gmeta.append(dict(acc=g, n_contigs=len(cids), n_genes=ng, oriented=oriented, **meta_g.get(g, {})))
    idx = np.concatenate(idx) if idx else np.zeros(0, np.int64); rel = np.concatenate(rel) if rel else np.zeros(0, np.int8)
    # distinct proteins, from the distinct DNA sequences, translated in parallel
    gs = A.geneSequences.read(); seqid = gs["seqid"][pd.Index(gs["gene"]).get_indexer(g_id[idx])]
    useq, inv = np.unique(seqid, return_inverse=True)
    code_of = np.zeros(len(useq), np.int64); code_of[inv] = gcode[idx]
    sq = A.sequences; s_id = sq.read(field="seqid"); so = np.argsort(s_id); rows = so[np.searchsorted(s_id[so], useq)]
    CH = 20000; chunks = []
    for s in range(0, len(useq), CH):
        r = rows[s:s + CH]; ro = np.argsort(r); dna = np.empty(len(r), object)
        dna[ro] = sq.read_coordinates(r[ro], field="dna"); chunks.append((list(dna), code_of[s:s + CH]))
    with Pool(workers) as pool: prots = [p for ch in pool.map(translate, chunks) for p in ch]
    pid_of = {}; plist = []; seq_pid = np.empty(len(useq), np.int32)
    for i, p in enumerate(prots):
        j = pid_of.get(p)
        if j is None: j = pid_of[p] = len(plist); plist.append(p)
        seq_pid[i] = j
    # a product for each family, from one of its genes
    one = pd.Series(gf["gene"]).groupby(gfam).first(); grow = pd.Index(g_id).get_indexer(one.values)
    prod = [""] * len(fam_names); ok = grow >= 0
    if ok.any():
        r = ix[grow[ok]]; ro = np.argsort(r); pr = np.empty(len(r), object); pr[ro] = gd.read_coordinates(r[ro], field="product")
        for f, p in zip(one.index[ok], pr): prod[int(f)] = p.decode()
    names, name_idx = np.unique(gname[idx], return_inverse=True)
    h.close()
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(os.path.join(outdir, "genes.npz"), goff=np.array(goff, np.int64), cstart=np.array(cstart, np.int64),
                        cgenome=np.array(cgen, np.int32), ccirc=np.array(ccirc, bool), clen=np.array(clen, np.int64),
                        fam=g_fam[idx], pid=seq_pid[inv], rgp=in_rgp[idx], spot=spot[idx], name=name_idx.astype(np.int32), strand=rel)
    with open(os.path.join(outdir, "proteins.txt"), "w") as f: f.write("\n".join(plist) + "\n")
    meta = dict(id=int(pgid) if pgid.isdigit() else pgid, dnaA_family=fam_names[FA] if FA >= 0 else None,
                n_genomes=len(genomes), n_genes=int(len(idx)), n_contigs=len(cgen), n_proteins=len(plist), n_families=len(fam_names),
                genomes=gmeta, families=fam_names, partition="".join(fam_part), product=prod,
                gene_names=[x.decode() for x in names], seconds=round(time.time() - t0, 1))
    json.dump(meta, open(os.path.join(outdir, "meta.json"), "w"), separators=(",", ":"))
    open(os.path.join(outdir, "extract.ok"), "w").write(str(time.time()))
    return meta


def loop(delete):
    """take downloaded files (raw/<id>.ok) one at a time until the downloader is finished and nothing is left"""
    raw = os.path.join(ROOT, "raw"); out = os.path.join(ROOT, "out"); me = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    while True:
        todo = sorted(glob.glob(os.path.join(raw, "*.ok")), key=os.path.getmtime)
        if not todo:
            if os.path.exists(os.path.join(ROOT, "state", "download.finished")): return
            time.sleep(60); continue
        ok = todo[0]; claim = ok[:-3] + f".x{me}"
        try: os.rename(ok, claim)                       # atomic: one worker per file
        except OSError: continue
        h5 = ok[:-3] + ".h5"; pg = os.path.basename(h5)[:-3]
        try:
            m = extract(h5, os.path.join(out, pg))
            print(f"{pg}: {m['n_genomes']} genomes, {m['n_genes']} genes, {m['n_proteins']} proteins, {m['seconds']} s", flush=True)
            if delete: os.remove(h5)
            os.remove(claim)
        except Exception as e:
            print(f"{pg}: FAILED {type(e).__name__}: {e}", flush=True)
            os.rename(claim, ok[:-3] + ".failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("path", nargs="?"); ap.add_argument("--out", default=None)
    ap.add_argument("--loop", action="store_true"); ap.add_argument("--delete", action="store_true")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args()
    if a.loop: loop(a.delete)
    else:
        pg = os.path.basename(a.path).split(".")[0]
        m = extract(a.path, a.out or os.path.join(ROOT, "out", pg), a.workers)
        print(json.dumps({k: m[k] for k in ("id", "dnaA_family", "n_genomes", "n_genes", "n_contigs", "n_proteins", "n_families", "seconds")}))
        if a.delete: os.remove(a.path)
