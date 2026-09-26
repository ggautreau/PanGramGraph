"""What the model itself knows about one accessory element, on demand: its "radius of influence" (pg_knock.influence),
for the Coupled regions tab. serve_live.py now answers the same /elements and /influence itself (same origin as the
page, locally and in the Space, with a job queue on a CPU); this small server remains for a separate process: its own
port, its own fp32 copy of the model and its own GPU memory cap, so that it can run while serve_live.py and two
knockout runs share the 8 GB card.

    python3 serve_influence.py                    # http://127.0.0.1:8767, GPU cap 0.2 of the card
    python3 serve_influence.py --device cpu       # no GPU: ~1 min per element (sequence truncated after the last read)

    GET /health
    GET /elements?g=<genome index 0-539> | acc=<assembly>
        the accessory elements of that complete chromosome (pg_knock.Data.elements): panRGP runs by spot, coupled
        regions present at their spot; entry gene, span, size, label
    GET /influence?g=<index> | acc=<assembly>  &region=<region id> | &spot=<spot id> | &genes=<p,q,...>  [&n_ctrl=8]
        baseline, deletion of the element and n_ctrl size-matched whole-element control deletions of the same genome,
        one fp32 pass each (~1.5 s per element on the GPU once the genome is loaded, ~8 s for a genome's first call);
        the profile of |change| per 25-gene bin out to 1,500 genes against the controls' band, the radius and what the
        controls give by chance, and every neighbour's change with its calibrated specificity (see pg_knock.influence)

One request at a time reaches the model (influence() is not thread-safe); at most --max_queue wait, beyond that the
page is told to try again (503). Results are cached in memory (last 64). CORS: the page served by serve_live.py
(localhost:8765), a page opened as a local file (Origin null), and --origins.
"""
import os, json, time, argparse, threading, urllib.parse, collections
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

P = argparse.ArgumentParser()
P.add_argument("--host", default="127.0.0.1")
P.add_argument("--port", type=int, default=8767)
P.add_argument("--device", default=None, help="cuda:0 (default when available) or cpu")
P.add_argument("--threads", type=int, default=2, help="CPU threads (--device cpu)")
P.add_argument("--mem_frac", type=float, default=0.2, help="GPU memory cap of this process (fraction of the card)")
P.add_argument("--sets", default=None, help="link set (default pgb/region_sets.json): which elements are 'linked'")
P.add_argument("--max_queue", type=int, default=4)
P.add_argument("--origins", default="", help="extra allowed origins, comma-separated")
A = P.parse_args()
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import pg_knock as K

t0 = time.time()
CTX = K.context(dev=A.device, threads=A.threads, frac=A.mem_frac, sets=A.sets)
D = CTX[0]
ACC = {a: i for i, a in enumerate(D.ACC)}
ORIGINS = {"null", "http://localhost:8765", "http://127.0.0.1:8765", f"http://localhost:{A.port}", f"http://127.0.0.1:{A.port}"} | \
          {o.strip() for o in A.origins.split(",") if o.strip()}
LOCK = threading.Lock(); QL = threading.Lock(); QN = [0]
CACHE = collections.OrderedDict()
K.log(f"serve_influence ready in {time.time() - t0:.0f}s on {CTX[2].dev}: {D.NG} chromosomes, {D.NR} regions, {len(D.LINKS)} links")


class Busy(Exception): pass


def genome(a):
    if "acc" in a:
        if a["acc"] not in ACC: raise ValueError(f"unknown assembly {a['acc']} (the 540 complete chromosomes)")
        return ACC[a["acc"]]
    g = int(a.get("g", -1))
    if not 0 <= g < D.NG: raise ValueError(f"g must be a genome index 0-{D.NG - 1}")
    return g


def elements(a):
    g = genome(a); E = D.elements(g)
    return dict(g=g, acc=D.ACC[g], strain=D.STRAIN[g], n_genes=int(D.GL[g]),
                elements=[dict(kind="region" if int(k) == 1 else "spot", id=int(i), label=D.element_label(k, i)[:60],
                               start=int(s), end=int(e), entry=int(en), size=int(z))
                          for k, i, s, e, en, z in zip(E["kind"], E["id"], E["start"], E["end"], E["entry"], E["size"])])


def influence(a):
    g = genome(a); n_ctrl = max(3, min(16, int(a.get("n_ctrl", 8))))
    if "region" in a: el = ("region", int(a["region"]))
    elif "spot" in a: el = ("spot", int(a["spot"]))
    elif "genes" in a: el = ("genes", tuple(sorted(int(x) for x in a["genes"].split(",") if x.strip())))
    else: raise ValueError("give region=, spot= or genes=")
    key = (g, el, n_ctrl)
    if key in CACHE:
        CACHE.move_to_end(key); return CACHE[key]
    with QL:
        if QN[0] >= A.max_queue: raise Busy("the model is busy, try again in a moment")
        QN[0] += 1
    try:
        with LOCK:
            try:
                res = K.influence(g, el, n_ctrl=n_ctrl, ctx=CTX)
            except AssertionError as e:
                raise ValueError(str(e))
    finally:
        with QL: QN[0] -= 1
    CACHE[key] = res
    while len(CACHE) > 64: CACHE.popitem(last=False)
    return res


class H(BaseHTTPRequestHandler):
    def end_headers(self):
        o = self.headers.get("Origin") or ""
        if o in ORIGINS:
            self.send_header("Access-Control-Allow-Origin", o)
            self.send_header("Access-Control-Allow-Private-Network", "true")
        super().end_headers()

    def do_OPTIONS(self): self.send_response(204); self.end_headers()

    def js(self, obj, code=200):
        b = json.dumps(obj).encode(); self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path); a = dict(urllib.parse.parse_qsl(u.query))
        if u.path == "/health":
            return self.js(dict(ok=True, device=str(CTX[2].dev), genomes=D.NG, regions=D.NR, links=len(D.LINKS),
                                sets=os.path.relpath(D.SETS, K.PROJ), waiting=QN[0], cached=len(CACHE)))
        route = {"/elements": elements, "/influence": influence}.get(u.path)
        if route is None: return self.js(dict(error="not found"), 404)
        try: self.js(route(a))
        except Busy as e: self.js(dict(error=str(e)), 503)
        except (KeyError, ValueError) as e: self.js(dict(error=str(e).strip("'\"")), 400)

    def log_message(self, format, *args): pass


print(f"influence server on http://{A.host}:{A.port}/ (GET /health, /elements, /influence)", flush=True)
ThreadingHTTPServer.daemon_threads = True
ThreadingHTTPServer((A.host, A.port), H).serve_forever()
