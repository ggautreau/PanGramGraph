# Allelic contingency at the variable loci of the *E. coli* pangenome

**Claim.** At the variable loci of the *Escherichia coli* pangenome, which gene family
comes next is partly written in the **alleles** — the particular sequence variants — of
the genes immediately upstream. This information is not reducible to the genome's
phylogeny, and a pangenome graph cannot represent it, because the graph collapses every
variant of a family into a single node. A genomic language model reading real protein
sequences recovers it; the same model reading only the order of families does not.

Data: PanGBank pangenome **11587** (`GTDB_refseq` v2.0.0, GTDB R232), its **540 complete
chromosomes**, 2,413,361 genes, rotated to the *dnaA* family and read in its direction of
transcription. Model: **Bacformer** (causal, complete genomes) on **ESM-2** protein
embeddings. Everything below is computed in `origin-walk/`; references in
`REFERENCES.md`, all checked on Crossref.

---

## 1. The measurement

A **fork** is a gene family followed, across the 540 chromosomes, by at least two
different families, each after ≥ 10 % of that family's occurrences and ≥ 20 of them.
1,652 such forks, **619,112 sites** (one site = one genome at one fork).

For each site we ask: which branch does this genome take? Five predictors, all
leave-one-genome-out, all with smoothing fitted on one half of the genomes and scored on
the other, in bits per site (lower is better):

| predictor | what it knows | bits/site | 
|---|---|---|
| frequency | how often each branch is taken | 0.714 |
| **phylogeny** | branches of the 15 nearest genomes (cgMLST over 3,098 core loci) | 0.622 |
| **graph** | longest shared upstream family path (back-off over 20/10/5/3/2 families) | **0.208** |
| model, family order only | Bacformer, every protein replaced by its family's most common one | 0.248 |
| **model, real proteins** | Bacformer on the genome's own proteins | **0.168** |

Two things follow.

**Phylogeny predicts poorly.** The nearest relatives are barely better than the overall
branch frequencies (0.622 vs 0.714), and far worse than the local family path (0.208).
The graph beats phylogeny at 81 % of forks (Wilcoxon p = 4·10⁻²⁰⁸). What comes next is a
property of the local genomic neighbourhood, not of the lineage.

**Alleles carry information the graph cannot hold.** Same model, same pretraining, same
family order: only the input alleles differ. Real proteins beat modal proteins by
**0.080 bits/site**, at **91 % of forks** (Wilcoxon p = 6·10⁻¹⁸⁴). Restricted to family
order the model does *not* beat the graph (0.248 vs 0.208); with real alleles it does
(0.168).

## 2. The causal test

Reading a modal-protein genome is reading a chimera, and a language model may be
disturbed by an unusual input rather than by a loss of information. The objection is
settled by a swap that keeps everything real.

At a fork, take a genome on branch *A*. Replace its **10 upstream proteins** by those of
a donor genome whose upstream **family path is identical** over those 10 genes — the
pangenome graph sees exactly the same path before and after. Two donors: one that also
took *A*, one that took *B*. 3,346 swaps over 60 forks, 120 genes of context:

| | P(own branch) | P(other branch) | top-1 = other branch |
|---|---|---|---|
| no swap | 0.850 | 0.123 | 7.4 % |
| **same-branch** swap | 0.846 | 0.125 | 7.9 % |
| **cross-branch** swap | 0.248 | **0.715** | **76.0 %** |

Cross minus same: **+0.590** in P(other branch), Wilcoxon p below float precision. A
same-branch swap — equally chimeric, equally real — changes nothing. A cross-branch swap
moves the prediction to the donor's branch three times out of four. **The alleles carry
the branch, and they carry it causally.**

## 3. Where it sits

The allelic advantage is **+0.150 bits/site inside a region of plasticity (panRGP)
against +0.053 outside**, and it concentrates on mobile and surface functions (bootstrap
95 % CI over forks):

| class | allelic gap (bits/site) | forks |
|---|---|---|
| LPS, O-antigen, capsule | **+0.192** [+0.033, +0.443] | 17 |
| IS, transposase | **+0.137** [+0.094, +0.188] | 188 |
| toxin-antitoxin, defence | **+0.123** [+0.081, +0.177] | 122 |
| prophage | +0.100 [+0.068, +0.139] | 294 |
| secretion, adhesion, motility | +0.076 [+0.032, +0.131] | 101 |
| other | +0.063 [+0.040, +0.093] | 654 |
| transport | +0.050 [+0.028, +0.080] | 276 |

A model-free confirmation, no Bacformer at all: predict the branch from the 15 sites
whose protein of the **fork family itself** is most similar (ESM-2 cosine). At 40 of
1,652 forks this alone beats both phylogeny and the graph in both halves of the genomes;
those forks are enriched inside regions of plasticity (0.64 vs 0.45 of sites).

### The test without any model

The prediction of section 7 below, run on the pangenome alone — no Bacformer, no ESM-2,
no embedding, no threshold. Take every pair of genomes at a fork whose **upstream family
path over 5 genes is identical** (the pangenome graph sees exactly the same thing for
both), and ask whether they carry the **same allele** of the fork family, meaning the
identical protein sequence. Then ask whether they go on to the same family. Strata are
(fork × exact family path × cgMLST distance bin), so phylogeny is matched too:
**20,897 strata, 9,340 of them informative, 200,000 pairs sampled from 8.0 million.**

| cgMLST distance bin | pairs | P(same branch), same allele | different allele | gap |
|---|---|---|---|---|
| closest | 38,333 | 96.6 % | 91.3 % | **+5.3** |
| | 33,190 | 95.5 % | 91.6 % | +3.9 |
| | 35,892 | 94.6 % | 91.2 % | +3.4 |
| | 32,140 | 92.2 % | 87.7 % | +4.5 |
| | 32,369 | 92.0 % | 87.0 % | +5.0 |
| most distant | 28,076 | 89.9 % | 85.1 % | **+4.7** |

The gap is there in **every** distance bin, from the closest relatives to the most
distant. Mantel-Haenszel odds ratio over strata: **2.11**. Stratum-weighted difference:
**+3.5 points**. Permuting the allele label inside each stratum gives a null of
+0.00004 ± 0.00143, so the observed +0.0348 sits **24 standard deviations** above it
(p = 0.002, the floor of 500 permutations).

It is stronger inside regions of plasticity (+7.5 vs +3.2 points) and follows the same
functional order as the language-model measurement:

| class | same allele | different allele | gap | pairs |
|---|---|---|---|---|
| LPS, O-antigen, capsule | 96.8 % | 77.7 % | **+19.1** | 2,358 |
| IS, transposase | 85.6 % | 74.7 % | **+10.9** | 24,768 |
| secretion, adhesion, motility | 97.4 % | 87.9 % | +9.5 | 12,398 |
| toxin-antitoxin, defence | 91.3 % | 82.6 % | +8.7 | 16,923 |
| other | 97.1 % | 92.5 % | +4.6 | 83,264 |
| transport | 95.1 % | 92.2 % | +3.0 | 38,687 |
| prophage | 92.6 % | 90.3 % | +2.3 | 21,602 |

**This settles the memorisation objection for the central claim.** The language model is
what found the effect and what measures its size; the effect itself is a property of the
pangenome, visible with nothing but sequence identity, matched paths and a permutation.
(Prophage forks rank low here and high in the model measurement: they carry many alleles,
so exact-identity pairs are rarer and the binary test is blunter than bits.)

### The clearest case: *waaZ* in the LPS core locus

125 genomes, every site inside a region of plasticity. After *waaZ* the chromosome goes
either to **waaU** (47 genomes) or to another **glycosyltransferase** (78 genomes).

| predictor | bits/site |
|---|---|
| frequency | 0.972 |
| phylogeny | 0.575 |
| graph (family paths) | 0.364 |
| **allele of *waaZ* alone, model-free** | **0.349** |
| model, family order only | 2.992 |
| **model, real proteins** | **0.060** |

The 125 genomes carry **36 distinct *waaZ* alleles, and the two branches share none of
them**. The allele of *waaZ* partitions the two futures exactly. Stripped of alleles the
model collapses to 2.992 bits, worse than guessing from frequencies; given them it reaches
0.060.

## 4. Why this is the expected biology

The *waa* locus builds the lipopolysaccharide core, and *E. coli* carries a small number
of discrete core types — R1, R2, R3, R4, K-12 — that differ by which glycosyltransferases
are present and in what order (Heinrichs *et al.* 1998, *Mol Microbiol* 30:221;
Amor *et al.* 2000, *Infect Immun* 68:1116). The locus is inherited as a recombining
cassette: the neighbouring region around *rfb*–*gnd* is one of the best documented
recombination hotspots of the species (Nelson & Selander 1994, *PNAS* 91:10227;
Tarr *et al.* 2000, *J Bacteriol* 182:6183; Milkman *et al.* 2003, *Genetics* 163:475).
A cassette inherited whole is exactly a structure in which **the allele of a flanking gene
tags which cassette is present, and therefore which gene comes next**. That is what the
numbers say, measured without any prior knowledge of the locus.

The same logic explains the other enriched classes. IS elements have family-specific
target preferences and transpose by copy-in routes that fix their neighbourhood
(Siguier *et al.* 2014, *FEMS Microbiol Rev* 38:865; Harmer & Hall 2020, *mSphere* 5), so
which IS variant sits there constrains what it sits next to. Prophages and their
satellites carry hotspots of antiviral systems whose content depends on the element
(Rousset *et al.* 2022, *Cell Host Microbe* 30:740), and defective prophages are
domesticated as units (Bobay *et al.* 2014, *PNAS* 111:12127). In each case the unit of
inheritance is larger than the gene, so identity at one position predicts content at the
next — and identity is allelic.

This extends, one level down, the contingency that Beavan, Domingo-Sananes & McInerney
established for gene **content**: in the *E. coli* pangenome the presence or absence of
many genes is predictable from the presence or absence of others (2023, *PNAS*
121:e2304934120; PanForest, 2026, *Bioinformatics* 42). Their result is about which genes
cohabit. Ours is about what happens **when gene content is held fixed**: with an identical
family path upstream, the sequence variants still decide what follows. Contingency in a
pangenome is not only a matter of which genes are present, but of which versions of them.

That the effect is local rather than phylogenetic also fits what is known of recombining
bacterial populations, where linkage is broken over short scales and haplotype structure
is maintained around selected loci (Vos & Didelot 2008, *ISME J* 3:199; Arnold *et al.*
2019, *Mol Biol Evol* 37:417; Croucher *et al.* 2011, *Science* 331:430).

## 5. What is not claimed

- **Memorisation cannot be excluded for the model measurements.** Bacformer was pretrained
  on ~1.3 M genomes, very likely including these assemblies. The real-vs-modal comparison
  holds model and pretraining fixed and varies only the input, and the swap test is causal,
  but neither rules out that the model recognises a particular strain and recalls its
  genome. This is why the model-free test above matters: it reaches the same conclusion
  with no model at all, so the *claim* does not rest on the model, only its quantification
  in bits does.
- **The decoder is lineage-sensitive.** Bacformer predicts among its own 50,000 clusters;
  mapping them to PPanGGOLiN families requires a decoder learned from the data. Trained
  on one set of clades and applied to others it degrades, and the model then no longer
  beats the graph overall (0.269 vs 0.208) — **but the allelic gap survives: 0.070
  bits/site under a clade split, against 0.080 under a random split.** The allelic result
  does not depend on the split; the "model beats graph" headline does.
- **The phylogeny proxy is crude**: allele-sharing distance over 3,098 single-copy
  persistent loci, not a tree with branch lengths.
- **One species, complete genomes only.** Draft assemblies were excluded so that forks are
  not contig ends. Whether the effect size holds in species with less recombination is open.
- Effect sizes are small per site (0.08 bits) but systematic; they matter where they
  concentrate, not on average.

## 6. Reproducing it

```bash
python3 pg_chrom.py          # 540 complete chromosomes, anchored on dnaA          (~25 min)
python3 pg_embed.py          # ESM-2 on 397,521 distinct proteins                  (~45 min, GPU)
python3 pg_calls.py          # Bacformer, windows of 800 proteins every 400        (~3 min, GPU)
python3 pg_calls.py --modal  # the same with each family's most common protein     (~3 min, GPU)
python3 pg_allelic.py        # the table of section 1                              (~25 min, GPU)
python3 pg_allelic.py --clade  # the clade-split control
python3 pg_swap.py           # the causal swap of section 2                        (~10 min, GPU)
python3 pg_noLM.py           # the model-free test of section 3                    (~8 min, CPU)
python3 pool_loss.py         # what average pooling costs, section 7               (~5 min, GPU)
python3 pgb_cgmlst.py        # the phylogeny proxy (run once, before pg_allelic)
```

Outputs: `pgb/allelic.json`, `pgb/allelic_clade.json`, `pgb/swap.json`,
`pgb/contingency.json`, `pgb/cgmlst_dist.npy`, `pgb/noLM.json`, `pgb/pool_loss.json`.

## 7. The cost of average pooling (`pool_loss.py`)

Bacformer's input is one 480-dimensional vector per protein: the **mean over residues**
of ESM-2 t12 35M. Measured on the 397,521 distinct proteins here:

- **Pooling encodes the fraction of residues changed, not which ones or how many.**
  Spearman(fraction of residues differing, cosine distance) = **0.825**; against the raw
  *number* of differing residues, only 0.419. Two proteins at 95–100 % identity are
  5 times closer in pooled space when they are longer than 500 aa than when they are
  shorter than 250 aa — the 1/L dilution of an average, seen directly.
- **Allelic variation lives at ~2.8 % of the scale of family differences**: one
  substitution moves the vector by 1.6 % of its norm, two different families by 56 %.
  It is not a precision problem — that displacement is 10.6 × the bfloat16 noise floor —
  but any representation scored by overall similarity treats alleles as interchangeable.
- **On the task, pooling costs half.** Predicting the branch from the fork family's
  allele: frequency alone 0.714 bits/site; neighbours chosen by pooled cosine 0.652;
  neighbours chosen by **exact** allele identity 0.589. Pooled ESM-2 recovers **50 %** of
  what exact identity carries.
- 3.5 % of proteins have, as their nearest neighbour in pooled space, a protein of a
  *different* PPanGGOLiN family.

**So the effect reported above is a lower bound.** Bacformer extracted 0.080 bits/site of
allelic information from a representation that has already discarded about half of it.

A bigger protein encoder does not fix this: the dilution is set by the averaging, not by
the encoder. ESM-C (300M/600M/6B, widths 960/1152/2560) is in any case not a drop-in —
Bacformer's contract is 480 dimensions — and would require retraining it. For the allele
question the best representation is not a larger pLM but the cheapest possible one:
**the identity of the sequence**, which is what `pg_noLM.py` uses and what gives the
strongest result in this report.

## 8. What it suggests next

- **Pangenome graphs could carry allele-level annotation at plastic loci.** The
  information a graph loses is concentrated, not diffuse: RGPs, IS, prophages, LPS. A
  graph that split the nodes of those families by allele would recover most of it, and
  `pg_allelic.py` measures, family by family, whether that split is worth making.
- The falsifiable prediction that motivated section 3's model-free test is now confirmed
  here; the same test (`pg_noLM.py`) runs on any PPanGGOLiN pangenome, with no model.
- Other species of PanGBank, especially ones with low recombination rates, where the
  cassette logic should weaken.
