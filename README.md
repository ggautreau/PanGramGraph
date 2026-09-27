<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/mark-dark.svg">
  <img src="assets/mark.svg" width="72" alt="PanGramGraph logo: a fork of the gene graph in a speech bubble">
</picture>

# PanGramGraph

Can a genomic language model tell you where a bacterial chromosome is constrained?
Every *E. coli* genome of PanGBank pangenome 11587 (2,002 genomes) is walked gene by
gene from `dnaA`, the pangenome graph of that window is drawn from PanGBank's own
families and partitions, and Bacformer's prediction at every position (130,837 calls)
is laid on top. The answer, in one line: **it knows where the chromosome is
constrained, and its hesitation rises where the pangenome is plastic.** Read in its own
vocabulary it names the next gene 88.8 % of the time, about as often as the graph's most
frequent successor (90.0 %) and more often than the graph after a fork (79.6 % against
72.0 %), with 0.83 bits of doubt inside regions of plasticity against 0.21 outside.
Asked by in-silico knockout whether it knows any *pairwise* coupling between distant
accessory regions, it does not: of 298 testable long-range links none survives, and the
single close-range one that does is a genomic-background marker. What deleting an element
does move is the model's expectation of all the accessory content downstream, broadly
([Coupled regions](#coupled-regions-and-what-the-model-knows-of-them)).

**Online:** https://huggingface.co/spaces/ggautreau/PanGramGraph, the page and its model's
server on a CPU ([Hugging Face Space](#hugging-face-space)).

**Locally**, from a clone of this repository (Python 3.11 with the packages of
`space/requirements.txt`; a CUDA build of PyTorch to use a GPU):

```bash
git clone https://github.com/ggautreau/PanGramGraph && cd PanGramGraph
# the server's data (~750 MB) are not in git: rebuild them (Pipeline, below) or take the Space's copy
huggingface-cli download ggautreau/PanGramGraph --repo-type space --include "pgb/*" --exclude "pgb/readings/*" --local-dir .
python3 serve_live.py        # then open http://localhost:8765/ (it listens on 127.0.0.1 only)
```

The page alone, `standalone/pangramgraph.html`, also opens as a file, with no server: the
dnaA window, the coupled regions with the model's verdicts, and the findings of the dnaA
window are in it. The live features (a genome's call in the family cards, drawn paths,
windows elsewhere on the chromosome, attention, what the model does without an element,
and a reading written by a language model) need the server. The first pass on eleven
strains is kept in `pgb/origin-fork-11strains.html`.

---

## The page

**PanGramGraph** (first called Origin Fork; `origin-fork.html` and `index.html` lead to it)
has five tabs, **Graph**, **Attention**, **Coupled regions**, **Findings** and **About**,
and a help behind every "?" (the key `?` opens it too).

### For a first visit

The page is written so that a microbiologist who has never met a pangenome graph or a
language model understands within a minute what it does and why it matters, and an expert
keeps every number. Six parts, all in the page's own style (IBM Plex, its colour tokens,
light and dark, phone), none of which ever covers the app:

- **The lede** says it for a microbiologist: the gene order of *E. coli* across the 2,002
  genomes, where strains part ways (islands, prophages, operons some strains lack), and an
  AI that guesses the next gene, with where it is right, where it doubts and what it has not
  learned.
- **The page in three pictures**, three inline drawings with one sentence each: a genome is a
  sentence of genes (dnaA, dnaN, recF … numbered from 0); 2,002 genomes overlaid make a graph
  that forks where strains differ (the real *dgo* fork after *yidX*, 1,231 genomes against
  160); a language model reads the sentence and suggests the next gene like a phone keyboard,
  sure near dnaA, hesitant in an island. Then one line on why it matters to a
  microbiologist. It is open on a first visit and folded to its bar afterwards: the page
  keeps `pgg-intro` in `localStorage` (`seen` after the first visit, then `open` or
  `closed` as the visitor leaves it; without storage it simply opens). On a phone the three
  drawings are swiped one after the other; on a tablet each sits beside its sentence.
- **Start with a question**: four buttons that set the page up for one example and answer
  it in one line, under the question and in a banner under the tab bar where the page shows
  it. *Where do strains part ways near the origin?* opens the dnaA window, pins *yidX*'s
  card and computes the model's call in O157:H7 Sakai, which lacks *dgo* (it expects
  *cbrA* from Sakai's first 7 proteins alone: their alleles mark the lineage). *Does the
  model know the lac operon?* opens the lacZ window (the server) and pins lacZ's card:
  *lacY* follows in 487 of the 527 complete chromosomes with one lacZ, and the model names
  the right next gene in 96 % of them; in 21 the stretch lies the other way round and
  *lacI* comes next, which the model names too; lacZ alone leaves it torn between *lacI* and
  *lacY*. *What does it make of a gene order no genome has?* draws
  dnaA → dnaN → recF → gyrB → dgoR, an order no genome carries, and reads the model's call:
  *dgoK* at 96 %, as inside the operon: the gene just read counts for more than the place
  (which shows it follows neighbours, not that it knows the operon as a unit). The card a
  question pins stands beside the boxes it is about (`popAvoid`), not over them; on a phone a
  question pins no card (its sheet would hide the answer and the graph): the answer says to
  tap the box. *Which accessory genes travel together, and
  does it know?* opens Coupled regions on the CRISPR–Cas I-E ↔ type VI secretion link
  (verdict *none*; none of the 298 testable long-range links is known). The numbers of the
  answers are read from the data or the server's answer when it runs, not written in.
- **What we learned**, in plain words, positive first, each with a link to its evidence:
  the model knows conserved gene order (95 % on persistent genes, 99.97 % right after dnaA);
  its doubt marks where genomes vary (4 times more inside regions of plasticity); it
  recognises the kind of strain from the proteins themselves (Sakai at the *dgo* fork, from 7
  conserved proteins: it reads their sequences, whose alleles mark the lineage, which a graph
  of families cannot see; 80 % right after a fork against 72 % for the most common next
  gene); it does not know which distant accessory elements go together (the knockout).
- **A glossary on the page's own words.** A term underlined with dots gives a short plain
  definition on hover, on focus (Tab) or on a tap; a click or <kbd>Enter</kbd> keeps it open,
  <kbd>Esc</kbd> or a tap elsewhere closes it. About sixty terms (persistent, shell, cloud,
  region of plasticity, spot, fork, perplexity, entropy, bits, top-1, decoder, knockout,
  lineage, cgMLST, PPanGGOLiN, PanGBank, Bacformer, clusters, exemplar bank, never here,
  calibration, permutation null, logit, AUC, start token…) are found in the text by
  `glossify()` (`GLOSS`, `GLOSS_PAT` in the page), the first occurrence per card, note or help
  section, including what the server sends back (cards, readings, a link's detail). The same
  definitions make the help's **Glossary**.
- **A guided tour** of five steps over the page itself (the graph, a box and its card, Draw a
  path, the chromosome map and the window bar, Findings), a ring round each and a card with
  Next, Back and Skip (<kbd>→</kbd>, <kbd>←</kbd>, <kbd>Esc</kbd>; step 3's *Show me an
  example* draws the example path and rings it, and the tour goes on); launched from the
  introduction, from the questions' box and from the help. On a phone its card sits at the
  bottom (at the top while a family's card is open).

The help opens on a **Quick start** (six lines, and the tour), and ends on the Glossary. The
tabs that need it start with a plain summary (*In short*): Attention (what a map shows, and
that it is a view for the curious) and Coupled regions (the couplings are real; the model
does not know them as pairs).

### The header

The logo is a fork of the gene graph in a speech bubble: a persistent family (orange)
followed either by a shell family (green) or by the family the model expects (purple).
Beside it, a link to this repository. Under the lede, the introduction, the questions and
what we learned (above). Then the pangenome as PanGBank built it, read
from its file by `pgb_page_meta.py`: 2,002 genomes, 9.33 M genes, 59,165 gene families
(3,188 persistent, 7,524 shell, 48,453 cloud), 153,366 edges, 165,128 regions of plasticity
in 1,864 spots and 2,114 modules, with a link to pangenome 11587 on PanGBank and
PPanGGOLiN's version (2.3.0). Then Bacformer on the dnaA window (the first 80 genes of the
2,002 genomes, 130,837 calls): 88.8 % of next genes guessed right at the first try, read in
its own vocabulary, and 0.83 / 0.21 bits of doubt inside / outside regions of plasticity,
four times more where genomes vary.

### Graph

- **The graph.** Left to right, genes from `dnaA` (column 0), or from the window's first
  gene elsewhere. A box is a PanGBank family at the column where it most often sits,
  coloured by its PPanGGOLiN partition (orange persistent, green shell, blue cloud); boxes
  stacked in a column are a fork, the most carried on top. Lines are adjacencies, wider
  when more genomes carry them (log scale). A dashed box, up to two per column, is a family
  the model expects right after the column's best-carried family and that follows it in no
  genome, drawn faint when the model gave it little probability (125 of the 150 in the dnaA
  window are under 0.1 %; its card says *a long shot* or *a real expectation*). Bars on top
  give the model's perplexity at each column; the whole column above the boxes answers the
  pointer (or a tap) with its perplexity, entropy and top-1. A slider hides the families
  carried by few genomes; when a family is shown that it hides, it goes down and says so
  beside it, with a button back to the visitor's setting. Zoom, **Overview**, **Fill height**
  and **Full screen** size it; below 50 % a line says the names are too small to read. The
  graph is as tall as its tallest column: where the columns in view are short, a line in the
  blank under them says which column is taller and where (`gapUpd`). The legend says that a
  box sits at its family's most common column, so a line can run back leftwards (the
  methyltransferase after *stxA2*).
- **Family cards.** Hover a box: a card gives the family across the genomes, what follows
  it and the model's call after it, averaged over every genome that carries it. It opens
  only when the pointer moves onto a box (not when the page scrolls under a still pointer),
  and while it previews it lets the pointer through to the boxes under it; a pinned card
  waits out of sight while its box is scrolled away. The card's header names its partition
  beside the colour of its top edge. The model's doubt at the box's column comes first as a
  verdict (`doubtSay`: *sure, and right*; *sure, yet wrong in some genomes*; *hesitant, yet
  right in N %*; *unsure*), then as perplexity (its surprise at the gene actually there) and
  entropy (how spread its own guess is); the perplexity bars' tip says the same. Where the
  partition surprises, the card says why: a shell family in 90 % of the genomes or more, or a
  persistent one in fewer than 95 %, is PPanGGOLiN's call, which weighs the graph neighbours
  too (in this pangenome persistent spans 92–100 %, shell 1.5–95 %: *lacY*, 95 %, is shell,
  *lacZ*, 97 %, persistent); a persistent family mostly inside a region of plasticity lies
  among variable genes, which panRGP counts as one stretch (*lacZ*). The product is said to
  be that of the family's representative, since a family can group variants (intimin
  types). A key under each table of calls explains the tags it shows (*N clusters*, *other
  cluster* for a cluster named like a family above it but not read as it, *never here*,
  *not read here*, →); a live call in one genome says when its first choice differs from the
  average over the genomes (the model follows the strain). Click to
  pin it: the server computes the model's exact call in one genome, with a search box for
  the exact rank and softmax probability of any of its 50,000 families at that position.
  Under **In PanGBank**, up to four genomes carrying the family link to their own page on
  PanGBank (`/pangenome/11587/genome/<id>`, the ids fetched by `pgb_pangbank_ids.py`), then
  the whole pangenome; PanGBank has no page per family, so **copy the family name** puts it
  on the clipboard. The live call links the chosen genome's page (*&lt;strain&gt; in
  PanGBank*) and offers **Draw from this path**. On a phone a tap pins, and the card opens
  as a sheet at the bottom of the screen. A card counts the family twice: **in this window**,
  the genomes (beyond dnaA, the complete chromosomes) where it falls inside the window, at its
  column; **in the pangenome**, the genomes of the 2,002 carrying it anywhere, which its
  PPanGGOLiN partition is about (`pgb/fam_info.json`, `pgb_region.build_fam_info`). *bcp* is
  persistent, in 1,997 of 2,002 genomes, yet inside the nudK window in 26 of 537 chromosomes:
  the others carry it just before the window, and **Where they carry it** opens the window
  that shows it. Windows read from dnaA in its direction of transcription, the reverse of the
  K-12 map's numbering (most chromosomes read *hyfA*, *bcp*, *gcvR*). The families before and
  after it (**Previous** and **Next gene**) and every family named on the card or under a drawn
  path are links: a family of the window shows its box (the slider lowered if it hides it, and
  put back at the next family shown at it or in another window; its card pinned); a gene the
  model names that no family of the window reads as opens the window where it sits. At a
  window's first or last column, **Show it in the previous window** (or next) opens the window
  beside with the family's card there: *bcp* starts its own window, and the one before shows
  *hyfA* before it. The browser's Back returns to a window as it was left, its card pinned or
  its path drawn.
- **The genome picker.** Before you type, it lists genomes that carry the family (tagged
  *carries it*), then, under a line, well-known strains, which may not. Type any mix of strain, serotype, sequence type, host, isolation
  source, country, year or accession; every word must match (*human urine*, *O157*,
  *cattle 2012*). The metadata are PanGBank's (`pgb_page_meta.py`): a serotype is given for
  about 1 genome in 10 and a sequence type for about 1 in 40. Beyond the dnaA window the
  drafts are greyed out, since only the complete chromosomes are read there. K-12 MG1655 is
  not among the 2,002 ([Two identity surprises](#two-identity-surprises)).
- **Drawing a path.** Switch to **Draw a path** and click boxes in order, or press on one and
  slide across the next ones, like glide typing; on a phone, tap them in turn. The model's
  call after exactly that sequence of proteins comes back live (`/path`), every box glows by
  the probability that its family comes next (the 12 likeliest labelled), and `+ add`
  extends the path with one. Drag a box of the path onto another to change that step, or
  off the graph to remove it; **Undo** reverts the last change. **Take a genome's path up
  to the box** uses a genome's own proteins, and in a window beyond dnaA the model then
  also reads the 400 proteins before the window. **Attention** opens the path in the
  Attention tab. After one or two proteins a note says that the model has little context
  (lacZ alone: not lacY first, but lacI, its neighbour on the other side). A box's label and
  its row of the table give the same figure: the family's clusters, each weighed by how
  often it stands for that family (the decoder; `pathRows`).
- **Along the chromosome.** A map of the 540 complete chromosomes from dnaA, in genes,
  shaded by the share of genes in a region of plasticity, sits above the graph. Its frame
  is the window shown: drag it, swipe it sideways (one finger, or two on a trackpad), or
  click anywhere, and the window opens where the frame is left. With the map selected,
  <kbd>→</kbd>/<kbd>↑</kbd> and <kbd>←</kbd>/<kbd>↓</kbd> move one window, <kbd>PgUp</kbd>
  and <kbd>PgDn</kbd> 500 genes, <kbd>Home</kbd> goes back to dnaA and <kbd>End</kbd> to the
  end. **Previous** and **Next** (keys <kbd>P</kbd>, <kbd>N</kbd>) move 60 genes, so a window
  shares 20 columns with the next. The search box suggests every gene name of the complete
  chromosomes with its place, and also takes a PanGBank family or a position from dnaA
  (*1500*); under the list, that names are as annotated (synonyms such as *stx2A* and
  *stxA2* apart) and a gene's place is its median over the chromosomes that carry it. A
  window beyond dnaA starts at a backbone family (present once in at least 95 %
  of the chromosomes), walks 80 genes in dnaA's direction and is read in the 540 complete
  chromosomes; the server builds it in 1 to 3 s (`pgb_region.py`). A link to a place:
  `pangramgraph.html#at=lacZ`.
- **Greyed loading screens.** While the server computes, only the part being computed is
  greyed, under a card that says what is being done: building a window (the graph greyed,
  the seconds counted), or deleting an element and reading the model (the influence panel
  greyed, with the stage, a bar of the passes done, the time left and the place in the
  queue; the first request after a start also loads the model's decoder). The rest of the
  page stays usable, and an influence computation keeps running while you look elsewhere.

### Attention

The model's attention while it reads the drawn path (`/attn`): Bacformer's 96 heads
(12 layers of 8) as a map, coloured by how far back each looks on average or by its share
on the previous protein, the start token or the protein itself; click one to show its
matrix (rows: the path's proteins; columns: what each attends to), and the last protein's
row in every layer and head, where the model looks when it calls the next family. For a
genome's own path beyond dnaA, the 400 proteins before it form one summed column. **Leave
the start token out** (on by default) hides the attention sink and renormalises each row.
**Take the first ten genes of this window** fills the Graph tab's path without switching the
graph to Draw a path. Attention shows where the model looks, not why it calls what it calls.

### Coupled regions

The tab opens on *In short*: accessory elements far apart travel together in the genomes
beyond shared ancestry, and the model does not know these pairs (counted from the runs);
the corrections to earlier versions of the tab are folded under its notes. Two layers,
long range (533 links between 290 elements) and close range (113 links between
143 elements), drawn as arcs over the chromosome (co-occurrence above, avoidance dashed
below). An arc is coloured by the model's verdict on the link (or by its set), and its line
says what the link is in the genomes (dotted: one element at several spots). Filters: only
links replicated in lineage-disjoint halves (on by default), links behind at least N family
pairs, avoidance; a click on a legend entry hides its arcs. Three lists, sets, links and
elements, are also the way in on a phone and from the keyboard. A set's detail gives its
elements, its strongest links and their presence across the genomes; an element's, its
links, and **graph** opens the Graph tab at its window.

- **A link's detail**: the verdict of the in-silico knockout with every number it was
  decided on (genomes tested, lineages, the median effect overall and in each lineage half,
  the share tested within the model's range, the smallest effect the test detects at 80 %
  power on a robust scale, 1.4826 × MAD of the lineage means, with alpha Bonferroni-adjusted
  over the tested links of the layer, and the q values, Benjamini–Hochberg over the tested
  links), its flags (one element at several spots, one transfer tract, separate insertions,
  one insertion site), and the presence of its two elements across the genomes in cgMLST
  order, lineage clusters marked. The one *knows* link, `espX1` → `chpB`, is explained by
  genomic background ([below](#coupled-regions-and-what-the-model-knows-of-them)).
- **An element, on click: what the model does without it.** Pick a genome carrying it at its
  spot (`pg_region_carriers.py`) and **Delete it and read the model**: the server deletes it,
  then 8 whole accessory elements of about its size in turn (`/influence`,
  `pg_knock.influence`), and draws the mean change of the model's expectation per 25 genes
  out to 1,500 genes downstream against the controls' range. *specific*: some elements
  downstream move differently from the others at their distance (q < 0.05); *broad*: none
  does, while the deletion moves the model more than a typical one; *like a control
  deletion*: none does, and it moves the model no more than a random element of its size.
  Each linked element is then placed: read, with its change and whether it goes the way the
  genomes go, or before the element, beyond 1,500 genes, not at its spot, or absent; the
  elements downstream are listed by distance or by q. A few seconds on a GPU, under a minute
  on the Space's CPU.
- The address keeps the layer and the selection (a set, a link or an element), so it can be
  shared: `pangramgraph.html#tab=regions&rl=close&rs=…`.

### Findings

The tab follows the window shown. It opens on the window's **genes of note**, sorted by
their names and products (CRISPR–Cas, toxin–antitoxin systems, toxins, secretion systems,
adhesins, iron uptake, antibiotic resistance, surface polysaccharides, phage, mobile DNA),
each a link to its box: the biology the numbers and the written readings do not say.

- **Reading the window**, in up to three versions: in the dnaA window a hand-written reading
  (the default); in every window an automatic one, written by rule in the page from the
  window's own numbers (its spine, fork and variable stretches); and one written by
  **DeepSeek-V4.1-Flash** from facts the server computes, every sentence checked against the
  facts it cites ([below](#the-reading-of-a-window-written-by-a-language-model)). A reading
  written before comes back at once; the automatic one shows meanwhile, and stays when the
  model's text is late, fails its check or cannot be written. In the dnaA window the model's
  reading is offered on request.
- **Where the model is sure, and where it hesitates**: the calls by the partition of the gene
  actually there and by region of plasticity: top-1 in the model's own vocabulary (and, in
  the dnaA window, on request, via the exemplar bank, which underestimates it) and entropy; in the dnaA
  window, the three robustness checks and the comparison with the graph.
- **How often the model named the next gene**, by range of columns.

### About

The data, the model and their credit, and how each part was computed (the scripts and
their outputs).

### Mouse and keys

In the Graph tab: drag the background (or with the wheel pressed) to move, Shift + wheel to
scroll along (the wheel alone in full screen), Ctrl + wheel to zoom; <kbd>+</kbd>
<kbd>−</kbd> zoom, <kbd>0</kbd> back to 100 %, <kbd>F</kbd> overview, <kbd>H</kbd> fill
height, <kbd>N</kbd> <kbd>P</kbd> next and previous window. <kbd>Esc</kbd> closes a card,
cancels a drag or a stroke, closes a list or leaves full screen; <kbd>?</kbd> opens the help.
In the tab bar <kbd>←</kbd> <kbd>→</kbd> move between tabs; in the lists of Coupled regions
<kbd>↑</kbd> <kbd>↓</kbd> move and <kbd>Enter</kbd> picks. In the tour <kbd>→</kbd> goes on,
<kbd>←</kbd> back and <kbd>Esc</kbd> ends it. A sideways swipe scrolls what is under it and
never turns the page back to the previous tab (`overscroll-behavior-x: none`); the Back
button still does.

---

## The discovery: allelic contingency

**[DISCOVERY.md](DISCOVERY.md)**: at the variable loci of the pangenome, which family
comes next is partly written in the *alleles* of the upstream genes, information a
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

## Coupled regions, and what the model knows of them

Accessory families whose presence goes together (or apart) **within** lineages and within
tertiles of accessory content, beyond a permutation null, in the 281 complete genomes that
belong to a lineage cluster of at least 4 (of 540). Long range: 2,290 family pairs more than
500 genes apart in different insertion spots, replicated in two halves of the lineage ×
content strata (halves that share lineages: 267 of the 533 links also replicate in
lineage-disjoint halves, `pg_link_flags.py`), grouped into 290 elements and 533 links
(`pg_epistasis.py --ctrl_content`, `pg_region_sets.py`). Close range: 362 pairs at 50-500
genes in the genomes carrying both (about 20 expected by chance), separate insertions
only, replicated in lineage-disjoint halves (`--halves lineage`), giving 143 elements and
113 links. These tests read presence in the genomes; the model is not in them.

**Does Bacformer know these couplings?** In-silico knockout on the 540 complete
chromosomes (`pg_knock.py`, `pg_knockout.py`): in each genome carrying both (or the
upstream one only, for avoidance), the upstream element is deleted from the chromosome and
the model's decoded probability of the downstream one is read, in fp32, one causal pass
over the whole chromosome, against deletions of *whole* accessory elements matched on size
and distance, with off-target elements at comparable distance scored the same way,
lineage-mean aggregation in two lineage-disjoint halves, and an empirical calibration by
pseudo-knockout draws. *knows*: the downstream element moves the way the genomes go, beyond
the control deletions (calibrated q < 0.05), in both halves, and more than the off-targets
the same deletion moves, in z and in raw units. q values are Benjamini–Hochberg over the
tested links; the smallest detectable effect is taken at 80 % power on a robust scale
(1.4826 × MAD of the lineage means), alpha Bonferroni-adjusted over the tested links.
105,276 passes, 5.0 h of GPU; every pass is checked bit for bit (deleting nothing changes
nothing, and nothing before a deletion moves). Reports: `pgb/knock/all/all_report.md`,
`pgb/knock/close/close_report.md`.

| | tested links | knows | not specific | opposite | none | not testable |
|---|---|---|---|---|---|---|
| long range, > 500 genes | 298 | **0** | 20 (+21 opposite) | 1 | 256 | 235 |
| close range, 50-500 genes | 47 | **1** | 3 (+11 opposite) | 1 | 31 | 66 |

*Not specific*: the element moves the way the genomes go (or the opposite way), but no more
than everything else downstream. *Not testable*: fewer than 5 lineages, no readout, only
partial deletions, too few matched control deletions, or no calibration.

The first run gave 7 long-range "knows"; all seven fell when the raw-specificity
calibration was fixed. Its pseudo-knockout values were not centred within each unit, so its
null spanned 0.59 instead of 1 and rejected 0.1 % of null draws instead of 5 %, turning
non-effects into the very gate that was meant to catch them. Centred, and gated on
max(nominal, calibrated), the long-range result is empty. An independent re-derivation
matches every verdict of both runs.

The one surviving link, `espX1` to `chpB` at 184 genes, is **not** a functional coupling: the
two sit about 200 kb apart with no shared family or protein, 75-85 % of their association is
absorbed by the rest of the accessory genome, both track the core phylogeny, and the effect
concentrates in the 30 genomes where `espX1` is the only remaining clue of its insertion
site; remove it where the rest of its operon stays, and the model barely moves. What
Bacformer reads is *which kind of genome* it is looking at, then what such a genome
usually carries. A real statistical skill, not knowledge of the pair.

What the knockouts do show is that deleting an element shifts the model's expectation of
**all** accessory content downstream, over a few hundred genes: broad, position-dependent,
not pairwise. The page turns that into an on-demand measurement: pick an element and a
genome, and the server deletes it live and draws how far the model's expectations move
(`serve_live.py /influence`: a few seconds on a GPU, under a minute on the Space's CPU,
23 to 47 s measured).

## Results on all 2,002 PanGBank genomes

130,837 calls, each joined to the gene actually there: its PPanGGOLiN partition and
whether panRGP places it in a region of genomic plasticity (RGP). Top-1 is read two ways.
The model predicts its own 50,000 clusters: *own vocabulary* reads a cluster as the
families actually present when the model predicts it, P(family | cluster), learned on one
half of the genomes and applied to the other (`pgb_graph.py`). *Via the exemplar bank*
maps each real gene to its nearest cluster among the 9,814 of a 37-genome bank, which
misses many of the model's clusters and so scores it wrong when it is right: an
underestimate (at `yidB` 0.2 % by the bank, 92.6 % in its own vocabulary). The median
rank is on the bank's basis.

| gene being predicted | calls | top-1, own vocabulary | top-1 via the exemplar bank (underestimates) | median rank (bank) | entropy |
|---|---|---|---|---|---|
| persistent | 82,603 | **95.1 %** | 24.4 % | 2,222 | 0.16 bits |
| shell | 46,278 | **80.2 %** | 13.9 % | 828 | 0.73 bits |
| cloud | 1,956 | **28.6 %** | 5.0 % | 1,344 | 3.33 bits |
| outside an RGP | 89,567 | **93.6 %** | 22.6 % | 2,139 | 0.21 bits |
| inside an RGP | 41,270 | **78.4 %** | 15.7 % | 757 | **0.83 bits** |
| all | 130,837 | **88.8 %** | 20.4 % | 1,604 | 0.41 bits |

**Confidence tracks plasticity after all.** Controls: at 91 % of the 65 columns where
both occur, calls inside an RGP are more hesitant (median +0.84 bits); in 84 % of 1,942
genomes likewise; entropy alone separates RGP from non-RGP calls with AUC 0.755, and
flags errors (0.45 bits wrong vs 0.23 right). **This reverses the one-chromosome
reading below**: on Sakai's 79 loci entropy was 0.44 bits inside RGPs vs 0.55 outside,
too small a sample. Not uniform: the `dgo` operon (columns 7-11, after `yidX` at column 6)
is carried by 84 to 88 % of genomes and lies in an RGP in almost all of them, yet the model
is calm there and names the genes at those columns in 87 to 99 % of calls.

**Against the graph.** Predicting the next family from the previous one alone, by its most
frequent successor learned on the other half of the genomes (the halves of the decoder),
gives 90.0 %; the model gives 88.8 %. Where the graph is certain the model adds nothing.
After a fork (the 35,640 calls whose previous family has at least two successors, each in
at least 5 % of its adjacencies), the model is right 79.6 % of the time against 72.0 % for
the graph: it reads upstream context that single adjacencies do not hold (`pg_graph_baseline.py`
computes these figures, and two other definitions of a fork, in a few seconds).

| columns | calls | entropy | top-1, own vocabulary | top-1 via the exemplar bank |
|---|---|---|---|---|
| 1-3 (`dnaN`, `recF`, `gyrB`) | 5,994 | 0.13 bits | 99.97 % | 99.9 % |
| 1-10 | 19,705 | 0.16 bits | 94.7 % | 57.9 % |
| 11-40 | 54,895 | 0.26 bits | 92.0 % | 10.8 % |
| 41-79 | 56,237 | 0.64 bits | 83.7 % | 16.6 % |

The window: 2,540 PPanGGOLiN families (575 persistent, 1,125 shell, 840 cloud). All
2,002 genomes anchor (1,994 on the `dnaA` family, 8 on `dnaN` where `dnaA` is not a
gene); 2,000 reach column 3 and 1,359 column 79, since drafts stop at their contig end.
From column 42 about 45 genomes carry the LEE pathogenicity island (`ler`, `esc`, `ces`,
`tir`, `eae`). The model hesitates most at column 45 (`yicJ`, 1.38 bits), a fork where 11 %
of genomes take a branch of 8 shell families. Only two of the first eleven strains
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

## The reading of a window, written by a language model

The Findings tab offers, for every window, a reading written by **DeepSeek-V4.1-Flash**
(through Hugging Face Inference Providers) and checked sentence by sentence (`pg_reading.py`,
served at `/reading`).

- **Facts, not the page's text.** The page sends only the window's id. The server builds the
  facts from its own data, 2,000 to 2,900 tokens: the window's stretches (spine, fork,
  variable, by the page's own rules), each column's families, genes and products,
  partitions, shares in regions of plasticity, the model's naming and hesitation, and the
  groups where the products agree, each line with an id (`[S3]`, `[S3.c12]`, `[M]`, `[Q]`...).
- **A skill.** The system message (about 2,400 tokens) carries `reading_skill.md`
  (version 2): what the pangenome words and the model's three numbers mean and do not mean,
  the five sections of every
  reading (@start, @variable, @model, @end, @whole), a six-step procedure, and two reference
  readings (the hand-written one of the dnaA window, and one of a window nearly all spine).
  Prompt version r9, temperature 0, 150 to 200 words asked, at most 220 served.
- **Every sentence checked.** Each sentence names the facts lines it is written from, and is
  checked against those lines only: every number must be one they show, in its role (a
  share, a share in regions of plasticity, a share named, a hesitation) and bound to the gene
  or column it is said of; every gene, column, kind of column and partition must stand in
  them; the words of an interpretation (`[[ ]]`, shaded on the page) must come from the
  products and gene names of the lines cited and fit most of the families they label, and no
  product word may stand outside one; no superlative or trend the lines do not state. A
  sentence that fails is dropped and the model is asked once more, with the faults named; if
  too little is left (under 110 words, or no @start or @whole), the page keeps its automatic
  reading.
- **Costs.** The 112 readings written ahead (the dnaA window, the 106 windows that Next and
  Previous reach from it, and 5 more; `pg_reading.py pregen --chain dnaA`) all passed. 67 of
  them needed the second attempt, and 59 lost one to three sentences to the check. Both
  attempts counted, a reading took 10,800 tokens in and 790 out, about $0.0005 and 11 s in
  the median (28 s at most); all 112 cost $0.065. Readings are cached per window, keyed on
  the facts, the prompt, the skill and the model, so they come back at once.
- **Limits.** The check verifies the numbers, the names, the columns and what the
  interpretations rest on, not the prose: a sentence can still be clumsy or slanted in its
  wording, and an interpretation is read from gene names and products, not measured in the
  window. New readings are capped: 6 per visitor in 10 minutes and 30 a day, 150 and $0.30
  a day for the whole server, two written at once (six waiting), and only for the page's own
  site. The LLM never receives text from the browser.

## First pass on eleven strains (superseded, kept for the record)


### The anchor is the dominant variable

| loci from `dnaA` | mean entropy | model top-1 |
|---|---|---|
| 1 – 5 | 0.22 bits | **78.2 %** |
| 1 – 10 | 0.19 bits | 52.7 % |
| 40 – 79 | 0.65 bits | 16.2 % |

Per-step accuracy over the first twelve loci: **100, 100, 100, 0, 91, 9, 0, 0, 64, 64,
9, 18 %**. The model names `dnaN`, `recF` and `gyrB` in **all eleven strains** without
error. Not a decay: a cliff, landing exactly where the strains stop agreeing.

The same model on badly anchored input (mixed species, starting at each record's first
gene) scored 23.6 % over loci 0–9. Anchoring and orienting is not a detail.

### The graph

256 families over 80 loci: **23 persistent** (all 11 strains), 123 shell, 110 cloud;
333 observed adjacencies and 1,343 adjacencies the model expects that **no strain
carries**. Persistent families are biologically right: the replication cassette, the
`ilv` operon, the `uhp` operon, `ibpA`. First fork is the `dgo` operon (galactonate
catabolism), absent from O157:H7 Sakai. Further out the walk reaches `waa`, the LPS
core locus, among the most variable in the species.

### Against PanGBank: the founding prediction splits in two

79 loci of O157:H7 Sakai joined **on genomic coordinate** (100 % matched) to PanGBank
pangenome **11587** (`GTDB_refseq` v2.0.0, GTDB R232, 2,002 genomes). Top-1 here is via
the exemplar bank.

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

**Confidence fails it.** 0.48 bits outside an RGP versus 0.49 inside: no difference.
The model is exactly as sure of itself where it is almost always wrong. The distribution
does not flatten when it errs, it **moves**: at locus 17 in APEC O1 it puts 0.999 on
`ybjL` with 0.02 bits of doubt while the real gene sits at rank 1,031. For trust work:
**accuracy carries the signal, confidence does not.** (On all 2,002 genomes this reverses:
see above.)

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
  exemplar in embedding space**, what the 50,000 clusters actually are, agrees across
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
  is comfortable. `protein_seqs_to_bacformer_inputs` reloads ESM-2 on every call: load
  it once yourself.
- **Two label dictionaries** will disagree on the same family (`yidA` here, `ybjQ`
  there). The page prefers the *E. coli* annotation.
- **Score a model in its own vocabulary.** An exemplar bank that misses the model's
  clusters scores it wrong where it is right (top-1 20.4 % by the bank, 88.8 % read
  through a cross-validated decoder).

## The generative model exists, but not publicly

Figure 5 of the Bacformer preprint (`10.1101/2025.07.20.665723`) already generates whole
genomes from a **property token plus seed families**, sampled by **temperature** (top-p
and "nucleus" appear nowhere in the paper), conditioned on oxygen requirement and
optimal growth temperature. `BacformerForCausalProteinFamilyModeling` is in the code
with `generate()` and `property_ids`. **No published checkpoint carries its weights**:
verified by listing safetensors keys; neither causal checkpoint has
`protein_family_embeddings`. Worth a GitHub issue.

What is *not* in the paper: any graph structure, any comparison to a real pangenome
graph, any entropy-versus-plasticity test. That gap is where this project lives.

---

## Pipeline

All 2,002 PanGBank genomes (the current page):

```bash
# PanGBank HDF5 of pangenome 11587, 1.33 GB, md5 ae6cb5d3... (GET /pangenomes/11587/file) -> pgb/ecoli_11587.h5
python3 pgb_window.py     # dnaA windows, families, partitions, RGPs, proteins  -> pgb/window*.json   (~3 min)
python3 pgb_model.py      # ESM-2 once per distinct protein, one causal pass per genome
                          #   -> pgb/window_emb.npy, pgb/model_calls.npz                        (~3 min, GPU)
python3 pgb_graph.py      # graph + model calls aggregated, the decoder  -> pgb/graph_pgb.json, summary tables  (~3 s)
python3 pg_graph_baseline.py  # the model against the graph's most frequent successor (Results)   (~5 s)
python3 pg_chrom.py && python3 pg_embed.py && python3 pg_calls.py   # the 540 complete chromosomes, their proteins
                          #   embedded, the model's call at every gene -> pgb/chrom*, pgb/calls_*       (GPU)
python3 pgb_page_meta.py  # the header's numbers, what each genome is searched by, the chromosome map
                          #   -> pgb/pangenome_info.json, pgb/genome_meta.json, pgb/chrom_map.json
python3 pgb_pangbank_ids.py   # PanGBank's genome ids, for the links from the family cards -> pgb/pangbank_genome_ids.json
                          #   (one request per 31 s: ~11 min)
# coupled regions, long range (> 500 genes) and close range (50-500 genes)
python3 pg_epistasis.py --ctrl_content   # distant families that go together within lineages -> pgb/epistasis.json
python3 pg_region_sets.py # ... grouped into elements and sets                   -> pgb/region_sets.json
python3 pg_epistasis.py --ctrl_content --mindist 50 --maxdist 500 --dist carriers --halves lineage --sep 10 \
        --out pgb/epistasis_close_sep.json
python3 pg_region_sets.py --in pgb/epistasis_close_sep.json --out pgb/region_sets_close.json
python3 pg_link_flags.py  # per link: one element at two spots, tract, replication   -> pgb/link_flags.json
python3 pg_link_flags.py --sets pgb/region_sets_close.json --epi pgb/epistasis_close_sep.json --out pgb/link_flags_close.json
python3 pg_region_carriers.py             # the genomes carrying each element at its spot -> pgb/region_carriers.json
# the in-silico knockout
python3 pg_knockout.py --baselines        # 540 fp32 whole-chromosome passes          (~3 min, GPU)
python3 pg_knockout.py --all --tag all --flags pgb/link_flags.json                  # (~4 h, GPU)
python3 pg_knockout.py --all --sets pgb/region_sets_close.json --tag close --min_apart 50 \
        --max_apart 600 --ctrl_near --match_log 3 --flags pgb/link_flags_close.json # (~2 h, GPU)
python3 -c "import pg_knock as K; K.export_decoder()"   # the compact decoder the Space reads -> pgb/knock/decoder_halves.npz
# the page, its readings, the server
python3 pg_reading.py pregen --chain dnaA   # the readings of the windows Next and Previous reach -> pgb/readings/
                                            #   (needs HF_TOKEN or `huggingface-cli login`)
python3 build_page.py     # page_template.html + data -> standalone/pangramgraph.html (NAME in the script)
python3 serve_live.py     # the page and the model, live: http://localhost:8765/
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
there. Needs the server (`serve_live.py` loads `pgb/window_emb.npy`, one forward pass per
request, cached). Tail ranks are soft: two runs of the same bfloat16 model moved a gene's
rank by about 1 % and its probability by up to 3.7 %.

### Draw a path, read the next family

Each box brings its family's most common protein in the 2,002 windows (`pid` in the page
data); **Draw from this path**, in a pinned card, starts from a genome's own proteins
(`/prefix`). Paths no genome carries are allowed, which is the point: dnaA -> recF,
skipping dnaN, still gives gyrB at 95.6 % (dnaN 2 %); dnaA -> gyrB gives 4.7 bits of
doubt. A box is linked to every model cluster holding >= 10 % of its proteins (67 of
289 common families are split). A box's probability is the sum over its model clusters
of p(cluster) x P(family | cluster), from the window proteins, so a cluster shared by two
families is not counted twice (`bfw` in the page data; `/path` also returns p for the
model clusters read on the boxes of the windows the server has built). A prediction with
no box is marked *in no window*: after Sakai's path to cbrA the model puts 99.3 % on a
cluster absent from all 2,002 windows, while yidR follows in 1,888 genomes.

### One position, full distribution over all 50,000 families

```bash
python3 probs_at.py --strain "Sakai" --locus 3            # 0.997 on gyrB, 0.04 bits
python3 probs_at.py --strain "Sakai" --locus 13 --csv d.csv   # real gene at rank 5
```

## Hugging Face Space

https://huggingface.co/spaces/ggautreau/PanGramGraph runs the page and its model's server
together in a Docker Space on free CPU hardware (2 vCPU, 16 GB). Bacformer has 50.85 million
parameters (its output head over 50,000 families included) and the ESM-2 embeddings are
precomputed, so two cores answer a drawn path in about 0.1 s, a genome's call with 800
proteins of context in about 1 to 2 s, a window elsewhere on the chromosome in 1 to 3 s, and
what the model does without an element in under a minute (23 to 47 s measured). A reading
written before comes back at once; a new one takes 10 to 30 s. The Space sleeps after 48
hours without visits and takes a minute or two to wake up; visitors wait their turn for the
model (at most 6 requests queue; beyond, the server answers busy).

```bash
python3 build_page.py && python3 space/assemble.py      # -> space_build/: Dockerfile, code, page, readings, data (~760 MB)
docker build -t pangramgraph-space space_build && docker run -p 7861:7860 -e HF_TOKEN pangramgraph-space   # try it here
huggingface-cli upload ggautreau/PanGramGraph space_build . --repo-type space              # publish
```

`space/` holds the Space's own files: the `Dockerfile` (the model is baked into the image),
the `README.md` with the Space's card and settings (`sdk: docker`, `app_port: 7860`),
`requirements.txt` (CPU PyTorch, transformers 4.53) and `start.py`. The Space's settings:

- **`HF_TOKEN`, a secret**: a Hugging Face token allowed to "Make calls to Inference
  Providers". The server uses it to have DeepSeek-V4.1-Flash write new Findings readings,
  billed to the token's account and capped (by default 150 readings and $0.30 a day). It goes
  only in the request's header to `router.huggingface.co`; it is never logged, written or sent
  to the page. Without it the readings shipped in `pgb/readings/` are still served, and
  `/health` says that new ones are off.
- **`PGG_DATA_REPO`, a variable**: with `space/assemble.py --no-data` the data stay out of the
  Space and `start.py` fetches them at start from that dataset repository (with `HF_TOKEN` if
  it is private).
- **`READING_CACHE=/data/readings`**, with the Space's persistent storage: keeps the readings
  written online and the day's counters across restarts. Without it both are lost when the
  Space restarts (the shipped readings stay). The other limits are set by `READING_*`
  variables (`pg_reading.py`); `READING=off` stops new readings.

`serve_live.py` takes `--host`/`--port` (or `HOST`/`PORT`), runs in float32 on a CPU, lets at
most `MAX_QUEUE` (6) requests wait for the model and answers 503 beyond; the page talks to the
server that served it, and, opened as a file, looks for one on the visitor's own machine
(127.0.0.1:8765).

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
| `assets/` | the logo (`mark.svg`, `mark-dark.svg`) and the icons |
| `serve_live.py` | the page's server: serves the page, computes per-genome calls, drawn paths, searches, windows, attention, `/elements` and `/influence`, and `/reading` |
| `pgb_region.py` | any other window of the 540 complete chromosomes, built when the page asks (through `serve_live.py`) |
| `pgb_page_meta.py` | the pangenome's own numbers, what each genome is searched by, the chromosome map |
| `pgb_pangbank_ids.py`, `pgb/pangbank_genome_ids.json` | PanGBank's id of each of the 2,002 genomes, for the links from the family cards |
| `pg_region_sets.py`, `pgb/region_sets.json` | the coupled regions: elements, links and sets, from `pg_epistasis.py --ctrl_content`; `pgb/region_sets_close.json` the 50-500 gene set |
| `pg_link_flags.py`, `pgb/link_flags*.json` | per link: one element at several spots, transfer tract, replication in lineage-disjoint halves |
| `pg_region_carriers.py`, `pgb/region_carriers.json` | the complete genomes carrying each coupled element at its spot, the influence panel's genome picker |
| `pg_knock.py`, `pg_knockout.py` | the in-silico knockout: units, whole-element null, off-target specificity, lineage halves, calibration, and one element's influence on demand; `pgb/knock/{all,close}_arcs.json` are the verdicts the page shows, `pgb/knock/all/all_report.md` and `pgb/knock/close/close_report.md` the two runs, `pgb/knock/pilot_report.md` the 24-link pilot |
| `serve_influence.py` | the same `/elements` and `/influence` in a process of its own (its own port and GPU cap) |
| `pg_reading.py`, `reading_skill.md`, `pgb/readings/*.json` | the reading of a window written by a language model: facts, skill, prompt, per-sentence checker, cache, limits; served at `/reading`; the 112 readings written ahead |
| `space/` | the Hugging Face Space: `Dockerfile`, card (`README.md`), `requirements.txt`, `start.py`, and `assemble.py`, which puts it together in `space_build/` |
| `pgb/ecoli_11587.h5` | PanGBank pangenome 11587, 1.33 GB (not in git) |
| not in git either | what the pipeline regenerates from it: `pgb/window*.json`, `pgb/window_emb.npy`, `pgb/model_calls.npz`, `pgb/chrom*`, `pgb/calls_*`, `pgb/*.npy`, `pgb/longrange.json`, `pgb/contingency.json`, the knockout's per-genome passes and baselines under `pgb/knock/`, `fam_exemplars.npz`, `fam_proto.npz` (see `.gitignore`). The page, `standalone/pangramgraph.html`, holds its own data and opens without them; the server needs them |
| `pgb/window.json`, `pgb/window_prot.json` | the 2,002 dnaA windows and their 19,198 distinct proteins |
| `pgb/window_emb.npy`, `pgb/model_calls.npz` | ESM-2 embeddings; the 130,837 calls (top 15, rank and probability of the real gene) |
| `pgb/graph_pgb.json` | the page's data |
| `pg_graph_baseline.py` | the model against the graph: the most frequent successor of the previous family, overall and after a fork (Results, "Against the graph") |
| `DISCOVERY.md`, `REFERENCES.md` | the allelic-contingency result, and its references |
| `scale/` | the same reading for every bacterial species of PanGBank |
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
- Couplings the model could reach: the close-range test (50-500 genes, where its use of
  the context is strongest) found one *knows* in 47 testable links, from genomic background.
  Still open is whether the broad influence of a deletion has element-specific structure
  across genomes; the page's *specific* verdict asks it one genome at a time.
- A competitor model on the same test: PanBART (Horsfield *et al.*, 2026) reads one gene
  family per token over a whole chromosome and is trained on 394,000 *E. coli* genomes, so
  it would say whether the null result is Bacformer's or the question's. Its published
  weights lack the family representatives needed to tokenise new chromosomes; the masked
  Bacformer and a plain co-occurrence baseline cost nothing and isolate the objective and
  the population structure.

## Credit

Genomes, families, partitions and plasticity regions from **PanGBank**
(pangenome 11587, GTDB_refseq v2.0.0, GTDB R232), built with **PPanGGOLiN**: Gautreau
*et al.* 2020, PLOS Comput Biol 16(3):e1007732, with plasticity regions from
**panRGP**: Bazin *et al.* 2020, Bioinformatics 36(Suppl_2):i651. PanGBank data are
**CC BY-SA 4.0**, as is the GTDB taxonomy it adapts; share-alike propagates to
redistributed derivatives. Chromosomes from NCBI RefSeq. Model: Bacformer, Wiatrak
*et al.*, bioRxiv 2025.07.20.665723 (`macwiatrak/bacformer-causal-complete-genomes`,
Apache-2.0), on ESM-2 protein embeddings, Lin *et al.* 2023, Science 379:1123. The
Findings readings are written by DeepSeek-V4.1-Flash (DeepSeek) through Hugging Face
Inference Providers.

---

Created using Claude Opus 5.5, prompted by G. Gautreau
