# PanGramGraph (origin-walk)

Can a genomic language model tell you where a bacterial chromosome is constrained?
Every *E. coli* genome of PanGBank pangenome 11587 (2,002 genomes) is walked gene by
gene from `dnaA`, the pangenome graph of that window is drawn from PanGBank's own
families and partitions, and Bacformer's prediction at every position (130,837 calls)
is laid on top. The answer, in one line: **it knows where the chromosome is
constrained, and its hesitation rises where the pangenome is plastic.**

**Online:** https://huggingface.co/spaces/ggautreau/PanGramGraph (the page and its model's
server, on a CPU). Locally:

```bash
python3 serve_live.py        # then open http://localhost:8765/
```

The page, **PanGramGraph** (first called Origin Fork; `origin-fork.html` now leads to it),
has five tabs: **Graph** (the pangenome graph with the model's calls, drawn paths, windows
anywhere on the chromosome), **Attention** (the model's attention maps while it reads a
path: its 96 heads, the full matrix, the start token left out on demand), **Coupled
regions**, **Findings** and **About**, with a help behind every "?". It also opens directly
(`standalone/pangramgraph.html`): the dnaA window, the coupled regions and the findings
need nothing else; the live features (a genome's call in the family cards, drawn paths,
windows beyond dnaA, attention) need the server. The first pass on eleven strains is
kept in `pgb/origin-fork-11strains.html`.

---

## The discovery: allelic contingency

**[DISCOVERY.md](DISCOVERY.md)** — at the variable loci of the pangenome, which family
comes next is partly written in the *alleles* of the upstream genes: information a
pangenome graph cannot hold, since it collapses every variant of a family into one node.
540 complete chromosomes, 1,652 forks, 619,112 sites. Same model, only the input alleles
differ: 0.080 bits/site, 91 % of forks, p = 6e-184; 0.070 under a clade split. Causal:
swapping 10 upstream proteins for a donor's, at an identical family path, moves the
prediction to the donor's branch 76 % of the time, while a same-branch swap does not
(7.9 %). Concentrated in regions of plasticity (+0.150 vs +0.053 bits) and on LPS, IS,
defence and prophage loci. **Confirmed with no model at all**: among pairs of genomes
matched on an identical 5-family upstream path and binned by cgMLST distance, those
sharing the fork family's allele take the same branch more often in every distance bin
(Mantel-Haenszel odds ratio 2.11 over 20,897 strata; +3.5 points, 24 sigma above a
within-stratum permutation). Chain: `pg_chrom.py`, `pg_embed.py`, `pg_calls.py`,
`pg_allelic.py`, `pg_swap.py`, `pgb_cgmlst.py`. References in `REFERENCES.md`.

## Results on all 2,002 PanGBank genomes

130,837 calls, each joined to the gene actually there: its PPanGGOLiN partition and
whether panRGP places it in a region of genomic plasticity (RGP).

| gene being predicted | calls | top-1 | median rank | entropy |
|---|---|---|---|---|
| persistent | 82,603 | 24.4 % | 2,222 | 0.16 bits |
| shell | 46,278 | 13.9 % | 828 | 0.73 bits |
| cloud | 1,956 | 5.0 % | 1,344 | 3.33 bits |
| outside an RGP | 89,567 | 22.6 % | 2,139 | 0.21 bits |
| inside an RGP | 41,270 | 15.7 % | 757 | **0.83 bits** |

**Confidence tracks plasticity after all.** Controls: at 91 % of the 65 columns where
both occur, calls inside an RGP are more hesitant (median +0.84 bits); in 84 % of 1,942
genomes likewise; entropy alone separates RGP from non-RGP calls with AUC 0.755, and
flags errors (0.45 bits wrong vs 0.23 right). **This reverses the one-chromosome
reading below**: on Sakai's 79 loci entropy was 0.44 bits inside RGPs vs 0.55 outside,
too small a sample. Not uniform: the `dgo` operon (columns 6-11) is in an RGP in 90 %
of genomes, yet the model is calm there and names `dgoA` in 97 %.

| columns | calls | entropy | top-1 |
|---|---|---|---|
| 1-3 (`dnaN`, `recF`, `gyrB`) | 5,994 | 0.13 bits | 99.9 % |
| 1-10 | 19,705 | 0.16 bits | 57.9 % |
| 11-40 | 54,895 | 0.26 bits | 10.8 % |
| 41-79 | 56,237 | 0.64 bits | 16.6 % |

The window: 2,540 PPanGGOLiN families (575 persistent, 1,125 shell, 840 cloud). All
2,002 genomes anchor (1,994 on the `dnaA` family, 8 on `dnaN` where `dnaA` is not a
gene); 1,359 reach 80 genes, drafts stop at their contig end. Around columns 42-52 about
fifty genomes carry the LEE pathogenicity island (`ler`, `esc`, `ces`, `tir`, `eae`),
where the model hesitates most (up to 1.4 bits). Only two of the first eleven strains
are among PanGBank's 2,002 (Sakai, Nissle 1917): the others were dereplicated away or
moved species.

### Does upstream context decide the branch at a fork? (`fork_rules.py`)

48 forks (a family followed by >= 2 families, each in >= 20 genomes). Exact branch
probabilities from the full softmax, against the branch frequencies (what the graph
knows) and a lineage lookup (the branch taken by genomes with the same 11 proteins up
to the fork, leave-one-out). The model beats frequency by > 0.1 bit/genome at 2 forks,
and beats the lineage lookup at 1: **ysdE (column 28), tisB (1,555) vs ivbL (179)**:
0.30 bits vs 0.48 frequency and 0.36 lookup; on 1,097 genomes whose upstream haplotype
no other genome shares, 93 % right, +0.20 bit. Elsewhere it mostly follows the dominant
branch, often over-confidently (gltS: 0.84 bits vs 0.71). "Unique haplotype" is a weak
lineage control (one residue differs): a phylogeny-aware baseline is still owed.

## First pass on eleven strains (superseded, kept for the record)


### The anchor is the dominant variable

| loci from `dnaA` | mean entropy | model top-1 |
|---|---|---|
| 1 – 5 | 0.22 bits | **78.2 %** |
| 1 – 10 | 0.19 bits | 52.7 % |
| 40 – 79 | 0.65 bits | 16.2 % |

Per-step accuracy over the first twelve loci: **100, 100, 100, 0, 91, 9, 0, 0, 64, 64,
9, 18 %**. The model names `dnaN`, `recF` and `gyrB` in **all eleven strains** without
error. Not a decay — a cliff, landing exactly where the strains stop agreeing.

The same model on badly anchored input (mixed species, starting at each record's first
gene) scored 23.6 % over loci 0–9. Anchoring and orienting is not a detail.

### The graph

256 families over 80 loci: **23 persistent** (all 11 strains), 123 shell, 110 cloud;
333 observed adjacencies and 1,343 adjacencies the model expects that **no strain
carries**. Persistent families are biologically right: the replication cassette, the
`ilv` operon, the `uhp` operon, `ibpA`. First fork is the `dgo` operon (galactonate
catabolism), absent from O157:H7 Sakai. Further out the walk reaches `waa`, the LPS
core locus, among the most variable in the species.

### Against PanGBank — the founding prediction splits in two

79 loci of O157:H7 Sakai joined **on genomic coordinate** (100 % matched) to PanGBank
pangenome **11587** (`GTDB_refseq` v2.0.0, GTDB R232, 2,002 genomes).

| PPanGGOLiN partition | loci | top-1 | median rank | entropy |
|---|---|---|---|---|
| persistent | 28 | **25.0 %** | 2,200 | 0.33 bits |
| shell | 46 | 6.5 % | 1,239 | 0.50 bits |
| cloud | 5 | 0.0 % | 1,096 | 1.25 bits |

| region | loci | top-1 | entropy |
|---|---|---|---|
| outside an RGP | 35 | **22.9 %** | 0.48 bits |
| inside an RGP | 44 | **4.5 %** | **0.49 bits** |

**Accuracy holds the prediction.** Five times more often right outside a region of
plasticity than inside; monotone from persistent to shell to cloud. An independently
built pangenome over two thousand genomes agrees about where the chromosome is
constrained.

**Confidence fails it.** 0.48 bits outside an RGP versus 0.49 inside — no difference.
The model is exactly as sure of itself where it is almost always wrong. The distribution
does not flatten when it errs, it **moves**: at locus 17 in APEC O1 it puts 0.999 on
`ybjL` with 0.02 bits of doubt while the real gene sits at rank 1,031. For trust work:
**accuracy carries the signal, confidence does not.**

Caveat: 79 loci, one strain, five of them cloud. Direction consistent, estimates loose.

### Are the predicted edges real? Tested against 2,002 genomes

The 1,343 adjacencies the model expects but no strain of the 11 carries, looked up
in the full PanGBank 11587 graph (`edges_vs_pgb.py`). Both graphs are put in the
Bacformer family vocabulary: each PanGBank representative protein goes to its nearest
real exemplar at cosine >= 0.95 (56,134 of 59,165 families mapped). 678 predicted
edges are testable; 655 target a family outside the 9,814-family exemplar bank.

| edges | n | found in PanGBank | null, same source | null, same target | enrichment (same target) |
|---|---|---|---|---|---|
| observed in the 11 (positive control) | 330 | 60.0 % | 1.2 % | 7.6 % | 7.9x |
| **predicted, never carried by the 11** | 678 | **13.9 %** | 1.1 % | 5.2 % | **2.7x** |
| — target seen elsewhere in the 11 | 149 | 34.2 % | | 11.8 % | 2.9x (95 % CI 2.2–3.7) |
| — **target never seen in the 11** | 529 | **8.1 %** | | 3.4 % | **2.4x (95 % CI 1.8–3.2)** |

**The model generalises synteny beyond what it was shown.** Predicted adjacencies
exist in the 2,002-genome graph 2.7 times more often than the same target placed
next to another source, and the effect holds for families the 11 strains never carry
at all. Model probability ranks them: 16.8 % found above the median p, 10.9 % below.

Read against the positive control: the mapping recovers only 60 % of edges that are
certainly real, so 13.9 % is about a quarter of the attainable ceiling. The
target-matched null is the one to quote; the same-source null (12.9x) flatters,
because predicted targets are hubs of the projected graph.

### Two identity surprises

GTDB R232 moved **K-12 MG1655 out of *E. coli*** into its own species cluster
(pangenome 11651, `s__G047199095_sp047199095`). **ST131 EC958 is in no PanGBank
pangenome at all.** The most studied bacterial genome in biology is not in its own
species' pangenome.

---

## Traps that cost real time

- **The masked head is a bad family assigner.** Across these 11 strains it gives
  orthologues the same family in **0 %** of the 41 genes tested (mean majority share
  0.48), because it predicts from genomic context as well as sequence. **Nearest
  exemplar in embedding space** — what the 50,000 clusters actually are — agrees across
  all eleven for **90 %** of genes (0.985) and puts `dnaA`, `dnaN`, `recF`, `gyrB` each
  in one family. Any measurement built on the masked head is confounded.
- **Assemblies are not consistently oriented.** MG1655 and W3110 run opposite ways, so
  walking up the coordinate axis from `dnaA` gives `dnaN` in one and `rpmH` in the
  other. Score the canonical cassette in both directions and pick.
- **Circular reverse walk.** `reversed(cds[:i] + cds[i+1:])` starts from the end of the
  genome, not the preceding gene. Write `cds[:i][::-1] + cds[i+1:][::-1]`.
- **`dnaA` is annotated a pseudogene** in 2 of 11 RefSeq assemblies (CFT073, Nissle
  1917): no DnaA protein at all. Anchor on the *gene* feature's coordinate, not on the
  protein, and shift those lanes by one.
- **Product search for `dnaA` catches `seqA`** if you match "replicat" + "initiat"
  (SeqA is *replication initiation regulator*). Require "chromosomal replication init".
- **The free-running loop cannot be closed with prototypes.** Feeding back a family's
  mean embedding changes the model's top-1 **77 %** of the time (agreement 0.232; top-1
  0.220 on real embeddings vs 0.089 on prototypes). Everything here is teacher-forced:
  the prefix is always real protein embeddings.
- **Nucleus (`top_p`) branching degenerates.** Median nucleus is **1** at top_p 0.60,
  0.80, 0.90 and 0.95, and 2 at 0.99, while the mean runs 92 → 1,857. Spike or plateau,
  no usable middle. Use top-k.
- **PanGBank: take the CGView map, not the HDF5.** `/pangenomes/{pid}/{gid}/cgview_map`
  is **2.9 MB** and carries partition, RGP and modules per gene. The `.h5` is **1.33 GB**.
  Rate limit is **one request per 30 s** across every route.
- **Bacformer attention is O(L²) in proteins.** 4,000 proteins exhaust an 8 GB GPU; 900
  is comfortable. `protein_seqs_to_bacformer_inputs` reloads ESM-2 on every call — load
  it once yourself.
- **Two label dictionaries** will disagree on the same family (`yidA` here, `ybjQ`
  there). The page prefers the *E. coli* annotation.

## The generative model exists, but not publicly

Figure 5 of the Bacformer preprint (`10.1101/2025.07.20.665723`) already generates whole
genomes from a **property token plus seed families**, sampled by **temperature** (top-p
and "nucleus" appear nowhere in the paper), conditioned on oxygen requirement and
optimal growth temperature. `BacformerForCausalProteinFamilyModeling` is in the code
with `generate()` and `property_ids`. **No published checkpoint carries its weights** —
verified by listing safetensors keys; neither causal checkpoint has
`protein_family_embeddings`. Worth a GitHub issue.

What is *not* in the paper: any graph structure, any comparison to a real pangenome
graph, any entropy-versus-plasticity test. That gap is where this project lives.

---

## Pipeline

All 2,002 PanGBank genomes (the current page):

```bash
# PanGBank HDF5 of pangenome 11587, 1.33 GB, md5 ae6cb5d3... (GET /pangenomes/11587/file)
python3 pgb_window.py     # dnaA windows, families, partitions, RGPs, proteins  -> pgb/window*.json   (~3 min)
python3 pgb_model.py      # ESM-2 once per distinct protein, one causal pass per genome
                          #   -> pgb/window_emb.npy, pgb/model_calls.npz                        (~3 min, GPU)
python3 pgb_graph.py      # graph + model calls aggregated  -> pgb/graph_pgb.json, summary tables  (~3 s)
python3 pg_epistasis.py --ctrl_content   # distant families that go together within lineages -> pgb/epistasis.json
python3 pg_region_sets.py # ... grouped into elements and sets                   -> pgb/region_sets.json
python3 build_page.py     # page_template.html + data -> standalone/pangramgraph.html (NAME in the script)
python3 serve_live.py     # per-genome calls and search, live, for the page's family cards
```

One causal pass per genome equals one pass per prefix: checked on two genomes, top-1
identical at 14 of 14 positions, |dp| <= 0.006 (bfloat16 noise).

The first pass on eleven strains:

```bash
# 1. family bank: real protein embeddings per family, from 37 annotated genomes  (~13 min, GPU)
MAXP=1200 stdbuf -oL python3 build_exemplars.py > exemplars.log 2>&1
#    -> fam_exemplars.npz, fam_annot.json, genomes.json        [already built, 28 MB]

# 2. the 11 E. coli, anchored on dnaA and oriented                               (~3 min)
python3 fetch_eco.py          # NCBI datasets API -> eco/*.gbff  (~140 MB, not kept in git)
python3 parse_eco.py          # -> eco_parsed.json

# 3. the graph + the model's call at every position                              (~4 min, GPU)
python3 eco_graph.py --loci 80 --topk 4     # -> eco_graph.json
python3 dists.py                            # -> dists.json   (11 x ~80 forward passes, top-15)

# 4. against PanGBank                        (respects 1 req / 30 s)
./pgb_query.sh && ./pgb2.sh                 # -> pgb/*.json  incl. the CGView map
python3 compare_pgb.py                      # -> the Sakai tables
python3 edges_vs_pgb.py                     # predicted edges vs the 2,002-genome graph
```

### Controls, run these before believing anything

```bash
python3 diag.py       # masked head vs nearest exemplar as family assigners
python3 validate.py   # prototype loop soundness + nucleus coverage
python3 sweep.py      # nucleus size vs top_p
```

### One genome, live, from the page

Click a family: the card computes Bacformer's call for the next gene in one genome
(Sakai by default when it carries the family; any of the 2,002 can be picked), and a
search box gives the exact rank and softmax probability of any of the 50,000 families
there. Needs `serve_live.py` (loads `pgb/window_emb.npy`, one forward pass per request,
cached). Tail ranks are soft: two runs of the same bfloat16 model moved a gene's rank
by about 1 % and its probability by up to 3.7 %.

### Draw a path, read the next family

Switch the graph to **Draw a path** and click boxes in order: Bacformer's call after
exactly that sequence of proteins comes back live (`/path`), the expected families are
marked on the graph with their probability, and `+` extends the path with one. Each box
brings its family's most common protein in the 2,002 windows (`pid` in the page data);
**Draw from this path**, in a pinned card, starts from a genome's own proteins
(`/prefix`). Paths no genome carries are allowed, which is the point: dnaA -> recF,
skipping dnaN, still gives gyrB at 95.6 % (dnaN 2 %); dnaA -> gyrB gives 4.7 bits of
doubt. A box is linked to every model cluster holding >= 10 % of its proteins (67 of
289 common families are split). At each step every box of the graph glows by the probability that the next gene belongs
to its family (log scale, 1 in 100 million to 1), the 12 likeliest with a label, and a
hovered box states its own value. A box's probability is the sum over its model clusters
of p(cluster) x P(family | cluster), from the window proteins, so a cluster shared by two
families is not counted twice (`bfw` in the page data; `/path` returns p for the 1,834
clusters present in the windows). A prediction with no box is marked *in no window*: after
Sakai's path to cbrA the model puts 99.3 % on a cluster absent from all 2,002 windows,
while yidR follows in 1,888 genomes.

### One position, full distribution over all 50,000 families

```bash
python3 probs_at.py --strain "Sakai" --locus 3            # 0.997 on gyrB, 0.04 bits
python3 probs_at.py --strain "Sakai" --locus 13 --csv d.csv   # real gene at rank 5
```

## Hugging Face Space

The page and its model's server run together in a Docker Space, on a CPU: Bacformer has 27
million parameters and the ESM-2 embeddings are precomputed, so two cores answer a drawn
path in about 0.1 s and a genome's call with 800 proteins of context in about 1 to 2 s.

```bash
python3 build_page.py && python3 space/assemble.py      # -> space_build/: Dockerfile, code, page, data (~680 MB)
docker build -t pangramgraph-space space_build && docker run -p 7861:7860 pangramgraph-space   # try it here
huggingface-cli upload ggautreau/PanGramGraph space_build . --repo-type space              # publish
```

`space/` holds the Space's own files: the `Dockerfile` (the model is baked into the image),
the `README.md` with the Space's settings (`sdk: docker`, `app_port: 7860`), `requirements.txt`
(CPU PyTorch, transformers 4.53) and `start.py`. With `space/assemble.py --no-data` the data
stay out of the Space and `start.py` fetches them at start from the dataset repository named by
the variable `PGG_DATA_REPO`, with the secret `HF_TOKEN` if it is private. `serve_live.py`
takes `--host`/`--port` (or `HOST`/`PORT`), runs in float32 on a CPU, lets at most
`MAX_QUEUE` (6) requests wait for the model and answers 503 beyond; the page finds the server
that served it, else one on the visitor's own machine.

## Every bacterial species of PanGBank (`scale/`)

The same reading for all 2,029 bacterial pangenomes of PanGBank's GTDB_refseq (latest
release; the 15 archaeal ones left out): 219,049 genomes and 807 M genes, every genome and
every gene, run on the LaBIA cluster (Slurm, RTX A6000).

```bash
bash scale/setup_env.sh                   # micromamba env in the project space (PyTorch 2.6 / CUDA 12.4, transformers 4.53)
sbatch scale/slurm/smoke.sbatch <id>      # one species end to end, with the 8-bit check
sbatch scale/slurm/download.sbatch        # the only process that talks to PanGBank: one request per 31 s
sbatch scale/slurm/extract.sbatch         # file -> compact tables; the PanGBank file is then deleted
sbatch scale/slurm/gpu.sbatch             # ESM-2 embeddings + Bacformer's call at every gene (one or two of these)
python scale/status.py                    # where it stands, and the disk it will take
```

- `extract.py`: genomes in order, contigs longest first, a circular (or complete-genome)
  chromosome started at dnaA in its direction; families, regions of plasticity and spots,
  distinct proteins translated with each gene's own genetic code. Checked on *E. coli*:
  539 of the 540 complete chromosomes identical to `pg_chrom.py` (the last has no dnaA).
- `gpu.py`: ESM-2 t12 35M mean-pooled embeddings (its attention run through PyTorch's fused
  kernel: 2 to 2.5x faster, same embeddings to cosine 0.99993), then Bacformer's call at every
  gene, the genome read as its own preprocessing reads it (CLS, contigs, separators, END,
  token type = contig; windows of 6,000). Calls are computed from full-precision embeddings;
  the embeddings are kept in 8 bits with one scale per dimension (a scale per protein changed
  the top-1 in 2 to 16 % of calls; per dimension, 0.2 to 2 %).
- Kept per species (`out/<id>/`): `genes.npz`, `meta.json`, `emb.u8` + `emb_range.npy`,
  `calls_f.u16`, `calls_p.f16`, `calls_ent.f16`; about 77 GB for all. PanGBank files, protein
  sequences and float16 embeddings are deleted as soon as they are used.

## Files

| | |
|---|---|
| `standalone/pangramgraph.html` | **the page**: 2,002 genomes, built by `build_page.py` from `page_template.html`; `index.html` and `origin-fork.html` lead to it |
| `pgb_region.py` | any other window of the 540 complete chromosomes, built when the page asks (through `serve_live.py`) |
| `pg_region_sets.py`, `pgb/region_sets.json` | the coupled regions: elements, links and sets, from `pg_epistasis.py --ctrl_content` |
| `serve_live.py` | local server: serves the page, computes per-genome calls and searches |
| `pgb/ecoli_11587.h5` | PanGBank pangenome 11587, 1.33 GB (not in git) |
| not in git either | what the pipeline regenerates from it: `pgb/window*.json`, `pgb/window_emb.npy`, `pgb/model_calls.npz`, `pgb/chrom*`, `pgb/calls_*`, `pgb/*.npy`, `pgb/longrange.json`, `pgb/contingency.json`, `fam_exemplars.npz`, `fam_proto.npz` (see `.gitignore`). The page, `standalone/pangramgraph.html`, holds its own data and opens without them; the live server needs them |
| `pgb/window.json`, `pgb/window_prot.json` | the 2,002 dnaA windows and their 19,198 distinct proteins |
| `pgb/window_emb.npy`, `pgb/model_calls.npz` | ESM-2 embeddings; the 130,837 calls (top 15, rank and probability of the real gene) |
| `pgb/graph_pgb.json` | the page's data |
| `pgb/origin-fork-11strains.html`, `pgb/serve_live-11strains.py` | the first pass on eleven strains |
| `probs_search.py` | precomputed search for the 11-strain page (superseded by `serve_live.py`) |
| `fam_exemplars.npz` | 9,814 families, 26,371 real protein embeddings |
| `eco_parsed.json`, `eco_graph.json`, `dists.json` | the eleven-strain walk |
| `pgb/cgview_sakai.json` | PanGBank per-gene partitions, RGPs, modules for Sakai |

## Open

- Other species: the whole chain is PanGBank-generic; a species with a different origin
  architecture would test whether the cassette result is E. coli-specific.
- Beyond 80 genes: PanGBank windows cover whole contigs; the model saw 6,000-protein
  genomes.
- Calibration per column: which RGPs does the model find predictable (the `dgo` case)?

## Credit

Genomes, families, partitions and plasticity regions from **PanGBank**
(pangenome 11587, GTDB_refseq v2.0.0, GTDB R232), built with **PPanGGOLiN** — Gautreau
*et al.* 2020, PLOS Comput Biol 16(3):e1007732 — with plasticity regions from
**panRGP** — Bazin *et al.* 2020, Bioinformatics 36(Suppl_2):i651. PanGBank data are
**CC BY-SA 4.0**, as is the GTDB taxonomy it adapts; share-alike propagates to
redistributed derivatives. Chromosomes from NCBI RefSeq. Model: Bacformer, Wiatrak
*et al.*, bioRxiv 2025.07.20.665723.

---

Created using Claude Opus 5.5, prompted by G. Gautreau
