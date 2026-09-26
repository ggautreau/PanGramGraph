"""Which complete chromosomes carry each coupled element AT ITS SPOT, for the page's genome picker ("What the model does
without it", Coupled regions tab): the genomes in which pg_knock.influence(g, ('region', ri)) can delete it, i.e.
pg_knock.Data's TEST = PRES & HAS (>= half of the region's coupled families present, and an element located at its
spot or between its consensus flanks). No model, no GPU; reads the units cache of pg_knock.Data (built on first use).

    python3 pg_region_carriers.py            # -> pgb/region_carriers.json, read by build_page.py

Per sets file (region_sets.json, region_sets_close.json): its md5, and per region a bitset over the 540 genomes in the
sets file's own genome order (cgMLST order, as the presence strings), base64, bit g = byte g >> 3, bit g & 7.
"""
import json, base64, hashlib
import numpy as np
import pg_knock as K

OUT = "pgb/region_carriers.json"
res = {}
for key, path in (("long", "pgb/region_sets.json"), ("close", "pgb/region_sets_close.json")):
    D = K.Data(sets=path)
    RS = json.load(open(path)); order = [g["acc"] for g in RS["genomes"]]
    ix = {a: i for i, a in enumerate(D.ACC)}; perm = np.array([ix[a] for a in order])
    T = (D.PRES & D.HAS)[:, perm]                               # regions x genomes, in the sets file's genome order
    bits = [base64.b64encode(np.packbits(t, bitorder="little").tobytes()).decode() for t in T]
    res[key] = dict(sets=path, md5=hashlib.md5(open(path, "rb").read()).hexdigest(), genomes=len(order),
                    carriers=bits, n=[int(x) for x in T.sum(1)])
    print(f"{path}: {T.shape[0]} regions, carriers at the spot median {int(np.median(T.sum(1)))} of {T.shape[1]} genomes, "
          f"none for {int((T.sum(1) == 0).sum())}")
json.dump(res, open(OUT, "w"), separators=(",", ":"))
print(f"-> {OUT}")
