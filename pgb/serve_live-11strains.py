"""The model behind the Origin Fork page, live: exact rank and softmax probability of
any of the 50,000 families at any position, computed when the page asks.

    python3 serve_live.py            # then open http://localhost:8765/

Serves standalone/ (the page and probs/) and answers the page's searches. The page
also finds it when opened as a local file; without it, the page falls back to the
precomputed probs/ files, whose ranks are approximate beyond the top 64. Listens on
127.0.0.1 only. Each position is one forward pass on the real prefix, as in
dists.py, and is cached, so a search while typing costs nothing after the first.

    GET /health                                  model, device, strains
    GET /call?li=<strain>&locus=<L>&q=<text>     search at one position
"""
import json, re, time, argparse, threading, urllib.parse, collections, numpy as np, torch
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

P=argparse.ArgumentParser(); P.add_argument("--port",type=int,default=8765); A=P.parse_args()
dev="cuda:0" if torch.cuda.is_available() else "cpu"; CLS,PROT=2,4; D=80
MODEL="macwiatrak/bacformer-causal-complete-genomes"
tok=AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm=AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(torch.float16).eval().to(dev)
cm=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True).to(torch.bfloat16).eval().to(dev)
ex=np.load("fam_exemplars.npz"); ids,emb,offs=ex["ids"],ex["emb"],ex["offsets"]
E=torch.tensor(emb,device=dev); E=E/E.norm(dim=1,keepdim=True).clamp(min=1e-9)
FAM=torch.tensor(np.concatenate([[f]*(offs[i+1]-offs[i]) for i,f in enumerate(ids)]),device=dev)

@torch.no_grad()
def embed(s,bs=24):
    o=[]
    for i in range(0,len(s),bs):
        t=tok([x[:1022] for x in s[i:i+bs]],return_tensors="pt",padding=True,truncation=True,max_length=1024).to(dev)
        h=esm(**t).last_hidden_state;m=t["attention_mask"].unsqueeze(-1).to(h.dtype);m[:,0]=0
        for j,L in enumerate(t["attention_mask"].sum(1)):m[j,L-1]=0
        o.append(((h*m).sum(1)/m.sum(1).clamp(min=1)).float().cpu().numpy());del h,t
    return np.concatenate(o,0)
@torch.no_grad()
def nxt(pe):
    n=pe.shape[0]+1;x=np.zeros((1,n,480),dtype=np.float32);x[0,1:]=pe
    return torch.softmax(cm(protein_embeddings=torch.tensor(x,device=dev).to(torch.bfloat16),
        special_tokens_mask=torch.tensor([[CLS]+[PROT]*pe.shape[0]],device=dev),
        token_type_ids=torch.zeros((1,n),dtype=torch.long,device=dev),
        return_dict=True).logits[0,-1].float(),-1).cpu().numpy()
@torch.no_grad()
def assign(pe):
    q=torch.tensor(pe,device=dev);q=q/q.norm(dim=1,keepdim=True).clamp(min=1e-9)
    return FAM[(q@E.T).argmax(1)].cpu().numpy()

# --- the 11 chromosomes, embedded once; names as the page shows them -----------
G=json.load(open("eco_parsed.json")); PE=[]; FAMS=[]; OFF=[]
for g in G:
    off=1 if g.get("dnaA_pseudo") else 0; pe=embed(g["proteins"][:D-off])
    PE.append(pe); FAMS.append(assign(pe)); OFF.append(off)
STRAINS=[g["desc"].split(" (")[0] for g in G]
annot=json.load(open("fam_annot.json")); node={n["family"]:n for n in json.load(open("eco_graph.json"))["nodes"]}
NAMES={}; PROD={}
for f in set(map(int,annot))|set(node):
    a=annot.get(str(f),{}); nm=[]
    if f in node and not node[f]["label"].startswith("fam") and " " not in node[f]["label"]:
        nm.append(node[f]["label"])        # a gene symbol, not a product used as a label
    nm+=[x for x,_ in a.get("gene",[]) if x not in nm]
    NAMES[f]=nm; PROD[f]=(node[f]["product"] if f in node and node[f]["product"] else
                          (a["product"][0][0] if a.get("product") else ""))
LNAME={f:[x.lower() for x in v] for f,v in NAMES.items()}; LPROD={f:v.lower() for f,v in PROD.items()}
print(f"ready on {dev}: {len(G)} strains, {len(NAMES)} named families",flush=True)

LOCK=threading.Lock(); CACHE=collections.OrderedDict()
def dist(li,locus):
    """Softmax over all outputs at `locus` of strain `li`, plus exact 1-based ranks."""
    k=locus-OFF[li]
    if not 1<=k<len(FAMS[li]): raise KeyError(f"no locus {locus} in strain {li}")
    with LOCK:
        if (li,locus) not in CACHE:
            p=nxt(PE[li][:k]); rk=np.empty(len(p),dtype=np.int64); rk[np.argsort(-p,kind="stable")]=np.arange(1,len(p)+1)
            CACHE[(li,locus)]=(p,rk)
            if len(CACHE)>300: CACHE.popitem(last=False)
        CACHE.move_to_end((li,locus)); return CACHE[(li,locus)]
def search(q,rk):
    q=q.strip().lower(); m=re.fullmatch(r"(?:#|fam)?\s*(\d+)",q)
    if m: f=int(m.group(1)); return [(0,f)] if f<50000 else []
    out=[]
    for f,names in LNAME.items():
        sc=min([0 if a==q else 1 if a.startswith(q) else 2 if q in a else 9 for a in names] or [9])
        if sc==9 and len(q)>=3 and q in LPROD[f]: sc=3
        if sc<9: out.append((sc,f))
    return sorted(out,key=lambda x:(min(x[0],2),rk[x[1]]))     # exact, prefix, then partial by rank

class H(SimpleHTTPRequestHandler):
    extensions_map={**SimpleHTTPRequestHandler.extensions_map,".html":"text/html; charset=utf-8",
                    ".js":"text/javascript; charset=utf-8"}
    def __init__(self,*a,**k): super().__init__(*a,directory="standalone",**k)
    def end_headers(self):
        o=self.headers.get("Origin") or ""
        if o in ("null","http://localhost:%d"%A.port,"http://127.0.0.1:%d"%A.port):
            self.send_header("Access-Control-Allow-Origin",o)
            self.send_header("Access-Control-Allow-Private-Network","true")
        super().end_headers()
    def do_OPTIONS(self): self.send_response(204); self.end_headers()
    def js(self,obj,code=200):
        b=json.dumps(obj).encode(); self.send_response(code)
        self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(b)))
        self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u=urllib.parse.urlparse(self.path); a=dict(urllib.parse.parse_qsl(u.query))
        if u.path=="/health": return self.js(dict(ok=True,model=MODEL,device=dev,strains=STRAINS))
        if u.path!="/call": return super().do_GET()
        try:
            li,locus=int(a["li"]),int(a["locus"]); t0=time.time(); p,rk=dist(li,locus)
            hits=search(a.get("q",""),rk); real=int(FAMS[li][locus-OFF[li]])
            row=lambda sc,f: dict(fam=f,name=NAMES.get(f,[""])[0] if NAMES.get(f) else "",
                                  product=PROD.get(f,""),rank=int(rk[f]),p=float(p[f]),score=sc)
            nz=p[p>1e-12]
            self.js(dict(li=li,locus=locus,strain=STRAINS[li],n=len(hits),matches=[row(*h) for h in hits[:8]],
                         real=row(0,real),entropy=float(-(nz*np.log2(nz)).sum()),
                         ms=round((time.time()-t0)*1000,1)))
        except (KeyError,ValueError) as e: self.js(dict(error=str(e)),400)
    def log_message(self,format,*args): pass

print(f"open http://localhost:{A.port}/origin-fork.html",flush=True)
ThreadingHTTPServer(("127.0.0.1",A.port),H).serve_forever()
