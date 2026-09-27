"""The model against the graph (README, Results, "Against the graph"): on the 130,837 calls of the dnaA
window, predict the next family from the previous one alone, by its most frequent successor, and
compare with the model's call read in its own vocabulary.

  model   the model's call decoded to a family: P(family | model cluster) learned on the other half of
          the genomes (genome index parity, the halves of the decoder), the family with the highest
          summed probability over the call's top 15 clusters; top-1 = that family is the real one
  graph   the most frequent successor of the previous gene's family, counted over the adjacencies of the
          other half of the genomes (a call whose previous family has no successor there, or the first
          gene of a window, counts as wrong)
  fork    the calls whose previous family has at least two successors, each in at least 5 % of its
          adjacencies (all genomes): where the graph must choose

    python3 pg_graph_baseline.py        # a few seconds; reads pgb/window.json and pgb/model_calls.npz

Prints: model 88.8 %, graph 90.0 % on all calls; after a fork (35,640 calls) model 79.6 %, graph 72.0 %,
and two other fork definitions for comparison.
"""
import os, json, collections, numpy as np
from scipy import sparse
os.chdir(os.path.dirname(os.path.abspath(__file__)))

W = json.load(open("pgb/window.json")); C = dict(np.load("pgb/model_calls.npz").items())
GEN = W["genomes"]; FAMS = {int(k): v for k, v in W["families"].items()}
FI = {f: i for i, f in enumerate(sorted(FAMS))}; NF = len(FI); NCL = 50001
pos = [int(l) - GEN[int(g)]["start"] for g, l in zip(C["genome"], C["locus"])]
fam_at = np.array([FI[GEN[int(g)]["genes"][k][1]] for g, k in zip(C["genome"], pos)])
prev_at = np.array([FI[GEN[int(g)]["genes"][k - 1][1]] if k >= 1 else -1 for g, k in zip(C["genome"], pos)])
half = C["genome"] % 2
K = C["top_f"].shape[1]

# the model, decoded in its own vocabulary on the other half
hit = np.zeros(len(fam_at), bool)
for h in (0, 1):
    tr = half != h
    A = sparse.csr_matrix((C["top_p"][tr].ravel(), (np.repeat(fam_at[tr], K), C["top_f"][tr].ravel())), shape=(NF, NCL))
    tot = np.asarray(A.sum(0)).ravel(); Pn = sparse.diags(1 / np.maximum(tot, 1e-12)) @ A.T
    m = np.flatnonzero(half == h)
    Q = sparse.csr_matrix((C["top_p"][m].ravel(), (np.repeat(np.arange(len(m)), K), C["top_f"][m].ravel())), shape=(len(m), NCL))
    S = (Q @ Pn).toarray(); hit[m] = (S.argmax(1) == fam_at[m]) & (S.max(1) > 0)

# the graph: the most frequent successor of the previous family, counted on the other half
succ = {0: collections.defaultdict(collections.Counter), 1: collections.defaultdict(collections.Counter)}
for gi, g in enumerate(GEN):
    L = g["genes"]
    for k in range(len(L) - 1): succ[gi % 2][FI[L[k][1]]][FI[L[k + 1][1]]] += 1
allsucc = collections.defaultdict(collections.Counter)
for h in (0, 1):
    for f, c in succ[h].items(): allsucc[f].update(c)
gh = np.zeros(len(fam_at), bool)
for i in range(len(fam_at)):
    p = prev_at[i]; tr = succ[1 - half[i]].get(p)
    if p >= 0 and tr: gh[i] = tr.most_common(1)[0][0] == fam_at[i]

print(f"all calls ({len(fam_at):,}): model {hit.mean():.1%}, graph {gh.mean():.1%}")


def forks(share=None, count=None):
    """the calls whose previous family has >= 2 successors, each in >= share of its adjacencies (or >= count of them)"""
    F = np.zeros(len(fam_at), bool)
    for i in range(len(fam_at)):
        c = allsucc.get(prev_at[i]) if prev_at[i] >= 0 else None
        if not c: continue
        tot = sum(c.values())
        F[i] = sum(1 for v in c.values() if (v / tot >= share if share is not None else v >= count)) >= 2
    return F


for lab, F in (("each successor in >= 5 % of adjacencies (the README's)", forks(share=0.05)),
               ("each in >= 10 %", forks(share=0.10)), ("each in >= 20 adjacencies", forks(count=20))):
    print(f"after a fork, {lab}: {F.sum():,} calls, model {hit[F].mean():.1%}, graph {gh[F].mean():.1%}")
