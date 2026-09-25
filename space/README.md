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
genomic language model, what it expects next. Explore the graph, draw paths through it to ask
the model about any order of genes, move along the chromosome, look at the model's attention,
and see which distant regions go together.

This Space runs the page and its model's server on a CPU (Bacformer has 27 million
parameters; the ESM-2 protein embeddings are precomputed). It sleeps after 48 hours without
visits and takes a minute or two to wake up; several visitors at once wait their turn.

**Data and model.** Genomes, gene families, partitions and regions of plasticity from
PanGBank (pangenome 11587, GTDB_refseq v2.0.0, GTDB R232), built with PPanGGOLiN
(Gautreau *et al.* 2020, PLOS Computational Biology 16(3):e1007732) and panRGP (Bazin *et
al.* 2020, Bioinformatics 36(Suppl_2):i651); PanGBank data are CC BY-SA 4.0. Model:
Bacformer (Wiatrak *et al.*, bioRxiv 2025.07.20.665723), `macwiatrak/bacformer-causal-complete-genomes`,
Apache-2.0, on ESM-2 embeddings (Lin *et al.* 2023, Science 379:1123).
