---
title: PanGramGraph
emoji: 🧬
colorFrom: purple
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: E. coli pangenome graph read by a genomic LM
---

# PanGramGraph

The pangenome graph of *Escherichia coli* read with the grammar of gene order that a genomic
language model learned. Every genome of PanGBank's pangenome 11587 (2,002 genomes) is walked
gene by gene from `dnaA`; the graph shows what the genomes carry, and Bacformer, a causal
genomic language model, what it expects next. Read in its own vocabulary it names the next
gene 88.8 % of the time, and it hesitates where the pangenome is plastic.

The page has five tabs:

- **Graph**: the pangenome graph with the model's calls, from `dnaA` or anywhere along the
  chromosome. Hover a family for its card, click to pin it and have the model's exact call
  computed in any of the 2,002 genomes, with links to the genomes' pages on PanGBank. Draw
  paths through the graph to ask the model about any order of genes, including orders no
  genome carries. Move along the chromosome with the map of the 540 complete chromosomes,
  Previous and Next, or a search for any gene name.
- **Attention**: the model's 96 heads and its attention matrix while it reads a path.
- **Coupled regions**: accessory regions whose presence goes together within lineages, at
  long range (533 links) and close range (113 links), with the model's verdict on each link
  from in-silico knockouts. The model knows none of the long-range couplings, and the one
  close-range link it knows is explained by genomic background. Click an element to have it
  deleted live and see how far the model's expectations move downstream.
- **Findings**: what the model gets right and where it doubts, for the window shown, with a
  reading of the window written by DeepSeek-V4.1-Flash from facts the server computes, every
  sentence checked against the facts it cites.
- **About**: the data, the model, their credit, and how each part was computed.

This Space runs the page and its model's server on a CPU (2 vCPU). Bacformer has 50.85
million parameters and the ESM-2 protein embeddings are precomputed: a drawn path is
answered in about 0.1 s, a genome's call in 1 to 2 s, a window elsewhere on the chromosome in
1 to 3 s, and an element deleted and the model read again, against 8 control deletions, in
under a minute. A reading written before comes back at once, a new one in 10 to 30 s, within
daily limits. The Space sleeps after 48 hours without visits and takes a minute or two to
wake up; several visitors at once wait their turn.

**Code:** https://github.com/ggautreau/PanGramGraph

**Data and model.** Genomes, gene families, partitions and regions of plasticity from
PanGBank (pangenome 11587, GTDB_refseq v2.0.0, GTDB R232), built with PPanGGOLiN
(Gautreau *et al.* 2020, PLOS Computational Biology 16(3):e1007732) and panRGP (Bazin *et
al.* 2020, Bioinformatics 36(Suppl_2):i651); PanGBank data are CC BY-SA 4.0. Model:
Bacformer (Wiatrak *et al.*, bioRxiv 2025.07.20.665723), `macwiatrak/bacformer-causal-complete-genomes`,
Apache-2.0, on ESM-2 embeddings (Lin *et al.* 2023, Science 379:1123). Readings of the
Findings tab by DeepSeek-V4.1-Flash through Hugging Face Inference Providers.
