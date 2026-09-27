# Reading a window of the pangenome graph — skill, version 2

You read 80 columns of an *E. coli* pangenome graph, and how Bacformer, a genomic language
model, calls the next gene along them. Work only from the facts lines, each with its id in
brackets; every sentence names the ids it used.

## The words of the pangenome

- **persistent, shell, cloud**: PPanGGOLiN's partitions of the species, a family in nearly
  every genome, in many, in few. The share in the facts is another thing: the chromosomes
  carrying the family *in this window*. A persistent family with a low share lies beyond the
  window's end in the others, pushed out by genes inserted upstream: not rare, not lacking.
- **region of plasticity**: panRGP's stretches of a chromosome unlike its conserved backbone,
  where genes come and go. `RGP N%`: N% of the calls there are on a gene inside one, a
  property of each chromosome's gene. The facts name no spot, island or prophage: never call
  a stretch one.
- **spine, fork, variable**: one family in 90% of the unit or more; a second family in 10% or
  more; neither. **branch**: the second families of consecutive fork columns, a parallel path
  carried *instead of* the most carried families; its line lists its own families, and the
  genes listed at those columns are the other path.

## The model's three numbers

- **named** (top-1): the share of calls where the model's first choice is the family there.
  Not a probability that the biology is right, not conservation: high where chromosomes agree
  on the next gene, low where the order varies.
- **bits** (entropy): its hesitation. Not error: the model can be sure and wrong (in the dnaA
  window it names `ibpB` in 27% of calls while that spine stays under 0.19 bits). The most
  hesitant column is not the least conserved, and it can be a spine column.
- **perplexity**: the whole window only, in families, never for one column.

Every number is a mean over the calls of the chromosomes that reach that column.

## The five sections, always in this order

- **@start**: the first stretch: columns, genes `first` to `last`, how many carry it, its
  partition, how well the model names it and where it names fewest.
- **@variable**: the middle: the fork and variable stretches, their branches (each in its own
  clause, from its own line), and the gene groups of the spines between them. The reading's
  biology usually lives here.
- **@model**: opens on [M], in its direction, with its numbers; then the most hesitant column
  from [H], with its kind. [M] differs from window to window: say what it says of this one.
- **@end**: the last stretch and the last column [E].
- **@whole**: [G], then the comparison [C] in its direction, inside against outside.

## The procedure, every window

1. **Place the window**: [W] and [T], the unit (genomes or chromosomes).
2. **Walk the stretches** [S1] … [Sn]: kind, columns, share range, naming range, fewest
   named, most hesitant, and the gene groups they carry.
3. **Choose the biology** from [Q], the groups the server found where the products agree:
   one or two, the largest or the most telling. From the words [Q] gives and the products,
   make one phrase of 2 to 6 words (`[[galactonate catabolism]]`, never the list of words),
   name that group's columns next to it, and cite its line with [Q]. A branch is labelled
   from its own families. No [Q]: no `[[ ]]`.
4. **Say the extremes the facts state, and no other**: the most hesitant column of [H] (in
   @model), the most plastic stretch of [P] when [P] is there (where that stretch is read),
   the fewest named of a stretch line.
5. **Choose each sentence's ids**: a stretch line and a sub-line, or a whole-window line. A
   range is one a line shows, or, for the columns you name, exactly their lowest and highest
   share. An opener is a claim too: true of the lines its paragraph cites.
6. **Write, then check**: 150 to 200 words, two sentences a section (@variable may take
   three), each with its verb, at most three numbers a sentence. "Then" and "after it" lead
   further along the window, never back; "from A to B%" is the share at the first and at the
   last column named. Every number, gene, column and bracketed word stands in the lines its
   sentence cites.

## Two reference readings

Read them for style, depth, length and citation: short bold openers true of their paragraph,
plain exact English, few numbers well chosen. Their numbers, ids and biology belong to their
windows, not to yours; do not reuse their sentences.

The dnaA window, the hand-written reading of the page in this form (195 words):

@start **The spine starts at the origin.** Columns 0 to 5, `dnaA` to `yidA`, are carried by 92 to 99.9% of genomes, all persistent, in a region of plasticity in 0 to 0.5% of calls {S1}. The model names 93 to 100% of them, fewest at column 4, `yidB`, 93% {S1}.

@variable **The first fork is at `yidX`**, column 6, carried by 70% {S2.c6}. Then the `dgo` genes, variable columns 7 to 11, [[galactonate catabolism]], carried by 84 to 88%: the most plastic stretch of the window {S2.c7-11, P, Q}. Through columns 45 to 52 a branch of 8 shell families runs in 11%, [[fructose transport and aldolases]] {S4.b45}.

@model **The model names as many genes on the spine columns as on the others, 90 against 89%, and hesitates less there**, 0.14 against 0.46 bits {M}. It hesitates most at column 45, `yicJ`, a fork column, 1.38 bits {H}.

@end **The window ends in the `waa` genes**, columns 69 to 79, [[lipopolysaccharide core]] {S4.c53-79, Q}. The last column is `rfaF`, in 12% of genomes {E}.

@whole **Across the window it names 89% of genes**, 0.41 bits on average, perplexity 1.6 families {G}. Inside a region of plasticity it hesitates more than outside, 0.83 against 0.21 bits, and names fewer genes, 78% against 94% {C}.

The qmcA window, nearly all spine, where the model hesitates more on the spine (188 words):

@start **The window opens on a spine.** Columns 0 to 5, `qmcA` to `copA`, are carried by 94 to 100% of chromosomes, all persistent {S1}. The model names 89 to 99.8% of them, fewest at column 3, `ybaT`, 89% {S1}.

@variable **A single variable column breaks the first spine**: column 6, `ybaQ`, shell, carried by 80%, the most plastic stretch of the window {S2.c6, P}. Further along the long spine, columns 56 to 60, `cyoA` to `cyoE`, are [[cytochrome o ubiquinol oxidase]] {S5, Q}.

@model **Here the model hesitates more on the spine than off it**, 0.22 against 0.13 bits, while naming 97% of genes on both {M}. Its most hesitant column, 10, `fsr`, is a spine column, at 0.59 bits {H}.

@end **The window ends on persistent families.** Columns 69 to 79, `yajO` to `secF`, are carried from 90 to 48% of chromosomes {S6, E}. They are not rare: in the other chromosomes they lie beyond the window's end {T}.

@whole **Across the window it names 97% of genes**, 0.20 bits on average, perplexity 1.2 families {G}. Only 2,177 of the 42,960 calls fall in a region of plasticity, where it hesitates more, 1.54 against 0.13 bits, and names fewer genes, 69% against 98% {C}.
