"""The model behind the PanGramGraph page, live, on any of the 2,002 PanGBank genomes:
Bacformer's call after a family in one genome, and the exact rank and softmax
probability of any of its 50,000 families there, computed when the page asks.

    python3 serve_live.py            # then open http://localhost:8765/

Serves standalone/ and answers the page; the page also finds it when opened as a
local file. Listens on 127.0.0.1 only. Proteins are the dnaA windows extracted by
pgb_window.py, embedded once by pgb_model.py (pgb/window_emb.npy). Each call is one
forward pass on the genome's real prefix, cached.

    GET /health
    GET /gcall?acc=<assembly>&pfam=<PanGBank family>&q=<text>
        the call for the gene after `pfam` in genome `acc`; with q, the families
        matching q at that position, with rank and probability
    GET /path?pids=<i,j,k...>&q=<text>
        the call after any sequence of window proteins, a path drawn on the page
    GET /prefix?acc=<assembly>&pfam=<PanGBank family>
        genome `acc`'s own path from dnaA to `pfam`, to start a drawn path from
    GET /region?q=<gene, PanGBank family or position from dnaA>   or   /region?anchor=<family>
        any other window of the chromosome, in the 540 complete genomes (pgb_region.py);
        /gcall and /prefix then take &anchor=<family> and read that window, and /path takes
        their proteins as c<index> (pgb/chrom_emb.f16)
    GET /attn?pids=<...>&ctx=<...>&layer=<0-11 | mean>
        the model's attention while it reads a path: for the layer, each head's matrix and
        their mean (rows: the path's proteins; columns: the start token, the context before
        the path summed, the path's proteins), and for every layer and head the last
        protein's row, where the model looks when it calls the next family. Per mille.
"""
import json, re, time, argparse, threading, urllib.parse, collections, numpy as np, torch
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from transformers import AutoModelForCausalLM

P=argparse.ArgumentParser(); P.add_argument("--port",type=int,default=8765); A=P.parse_args()
dev="cuda:0" if torch.cuda.is_available() else "cpu"; CLS,PROT=2,4
MODEL="macwiatrak/bacformer-causal-complete-genomes"
cm=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True).to(torch.bfloat16).eval().to(dev)
import sys, math
def _attn_weights(query,key,value,attn_mask=None,dropout_p=0.0,is_causal=False,scale=None):
    """the model's own attention with its weights, fixed: its causal mask was built on the CPU (fails on
    a GPU) and its weights in bfloat16; here on the tensors' device, in float32, no dropout (inference)"""
    L,S=query.size(-2),key.size(-2); sf=1/math.sqrt(query.size(-1)) if scale is None else scale
    w=(query.float()@key.float().transpose(-2,-1))*sf
    if is_causal: w=w.masked_fill(~torch.ones(L,S,dtype=torch.bool,device=query.device).tril(),float("-inf"))
    if attn_mask is not None: w=w.masked_fill(~attn_mask,float("-inf")) if attn_mask.dtype==torch.bool else w+attn_mask.float()
    w=torch.softmax(w,-1); return (w@value.float()).to(query.dtype),w
sys.modules[type(cm).__module__].scaled_dot_product_attention_w_attn_weights=_attn_weights
W=json.load(open("pgb/window.json")); EMB=np.load("pgb/window_emb.npy").astype(np.float32)
BF=np.load("pgb/model_calls.npz")["bf_family"]
GEN={g["acc"]:g for g in W["genomes"]}; PFAM={v[0]:int(k) for k,v in W["families"].items()}
annot=json.load(open("fam_annot.json"))
NAMES={int(f):[x for x,_ in a.get("gene",[])] for f,a in annot.items()}
PROD={int(f):(a["product"][0][0] if a.get("product") else "") for f,a in annot.items()}
# the gene names PanGBank gives the proteins of each model cluster, so every name shown
# in the graph can be searched (yidP is a GntR regulator with no name in the model's bank)
ALIAS=collections.defaultdict(collections.Counter)
for g in W["genomes"]:
    for x in g["genes"]:
        if x[4]: ALIAS[int(BF[x[3]])][x[4]]+=1
for f,c in ALIAS.items():
    nm=NAMES.setdefault(f,[]); PROD.setdefault(f,"")
    nm+=[a for a,_ in c.most_common(3) if a not in nm]
LNAME={f:[x.lower() for x in v] for f,v in NAMES.items()}; LPROD={f:v.lower() for f,v in PROD.items()}
def name(f): return NAMES[f][0] if NAMES.get(f) else ""
GRAPH=json.load(open("pgb/graph_pgb.json"))["nodes"]
# the clusters the graph's boxes read (decoder of pgb_graph.py), and exemplar clusters as before
CL=np.unique(np.concatenate([BF,np.array([c for n in GRAPH for c,_ in n["bfw"]],dtype=BF.dtype)]))
for n in GRAPH:                      # a cluster the model uses for a family is searchable by its name
    for c,w in n["bfw"]:
        if w>=0.5 and n.get("named") and n["label"] not in NAMES.setdefault(int(c),[]):
            NAMES[int(c)].insert(0,n["label"]); PROD.setdefault(int(c),n.get("product",""))
LNAME={f:[x.lower() for x in v] for f,v in NAMES.items()}; LPROD={f:v.lower() for f,v in PROD.items()}
# the rest of the chromosome: windows of the complete genomes, built on demand
import pgb_region as RG
NCE=sum(1 for _ in open("pgb/chrom_prot.txt")); CEMB=np.memmap("pgb/chrom_emb.f16",dtype=np.float16,mode="r",shape=(NCE,480))
CIDX={g["acc"]:i for i,g in enumerate(RG.CGEN)}
CLSET=set(CL.tolist()); REG=collections.OrderedDict(); REG_LOCK=threading.Lock()
def register(D):
    """a window's clusters are read on its boxes, and its names become searchable"""
    global CL,LNAME,LPROD
    for n in D["nodes"]:
        for c,w in n["bfw"]:
            CLSET.add(int(c))
            if w>=0.5 and n.get("named") and n["label"] not in NAMES.setdefault(int(c),[]):
                NAMES[int(c)].insert(0,n["label"]); PROD.setdefault(int(c),n.get("product",""))
    CL=np.array(sorted(CLSET)); LNAME={f:[x.lower() for x in v] for f,v in NAMES.items()}; LPROD={f:v.lower() for f,v in PROD.items()}
def region_of(anc):
    with REG_LOCK:
        if anc in REG: REG.move_to_end(anc); return REG[anc]
    D=RG.build(anc)
    with REG_LOCK:
        register(D); REG[anc]=D
        if len(REG)>16: REG.popitem(last=False)
    return D
def region(a):
    anc=RG.FIDX.get(a.get("anchor","")) if a.get("anchor") else None
    return region_of(anc if anc is not None else RG.resolve(a.get("q","")))
print(f"ready on {dev}: {len(GEN)} genomes, {len(EMB)} proteins, {len(NAMES)} named model families; "
      f"{RG.NG} complete chromosomes, {len(RG.BB)} backbone anchors, {NCE} of their proteins",flush=True)

@torch.no_grad()
def nxt(pe):
    n=pe.shape[0]+1;x=np.zeros((1,n,480),dtype=np.float32);x[0,1:]=pe
    return torch.softmax(cm(protein_embeddings=torch.tensor(x,device=dev).to(torch.bfloat16),
        special_tokens_mask=torch.tensor([[CLS]+[PROT]*pe.shape[0]],device=dev),
        token_type_ids=torch.zeros((1,n),dtype=torch.long,device=dev),
        return_dict=True).logits[0,-1].float(),-1).cpu().numpy()
LOCK=threading.Lock(); CACHE=collections.OrderedDict()
def emb(pids):
    """window proteins by index, chromosome proteins as c<index>"""
    return np.stack([CEMB[int(x[1:])].astype(np.float32) if isinstance(x,str) else EMB[x] for x in pids])
def dist(pids):
    """Call after this sequence of proteins, with exact 1-based ranks. Cached."""
    key=tuple(pids)
    with LOCK:
        if key not in CACHE:
            p=nxt(emb(pids))
            rk=np.empty(len(p),dtype=np.int64); rk[np.argsort(-p,kind="stable")]=np.arange(1,len(p)+1)
            CACHE[key]=(p,rk)
            if len(CACHE)>400: CACHE.popitem(last=False)
        CACHE.move_to_end(key); return CACHE[key]
def row(p,rk,f,sc=0):
    return dict(fam=int(f),name=name(int(f)),product=PROD.get(int(f),""),rank=int(rk[f]),p=float(p[f]),score=sc)
def summary(p,rk,a,ntop=8):
    nz=p[p>1e-12]; out=dict(top=[row(p,rk,f) for f in np.argsort(-p)[:ntop]],entropy=float(-(nz*np.log2(nz)).sum()))
    if a.get("q","").strip():
        hits=search(a["q"],rk); out.update(n_match=len(hits),matches=[row(p,rk,f,sc) for sc,f in hits[:8]])
    return out
lab=lambda x: x[4] or x[5] or "?"
def locate(a):
    g=GEN.get(a.get("acc","")); f=PFAM.get(a.get("pfam",""))
    if g is None: raise KeyError("unknown genome")
    if f is None: raise KeyError("unknown family")
    ks=[k for k,x in enumerate(g["genes"]) if x[1]==f]
    if not ks: raise KeyError("this family is not in this genome's dnaA window")
    return g,ks[0]
def search(q,rk):
    q=q.strip().lower(); m=re.fullmatch(r"(?:#|fam)?\s*(\d+)",q)
    if m: f=int(m.group(1)); return [(0,f)] if f<50000 else []
    out=[]
    for f,names in LNAME.items():
        sc=min([0 if a==q else 1 if a.startswith(q) else 2 if q in a else 9 for a in names] or [9])
        if sc==9 and len(q)>=3 and q in LPROD[f]: sc=3
        if sc<9: out.append((sc,f))
    return sorted(out,key=lambda x:(min(x[0],2),rk[x[1]]))     # exact, prefix, then partial by rank

def in_window(a):
    """a complete genome's chromosome, the window from the anchor, and the family's column in it"""
    anc=RG.FIDX.get(a.get("anchor","")); f=RG.FIDX.get(a.get("pfam","")); gi=CIDX.get(a.get("acc",""))
    if anc is None or f is None: raise KeyError("unknown family")
    if gi is None: raise KeyError("beyond the dnaA window only the 540 complete genomes are read: pick one of them")
    s,e=int(RG.OFF[gi]),int(RG.OFF[gi+1]); L=e-s; ia=np.flatnonzero(RG.FAM[s:e]==anc)
    if len(ia)!=1: raise KeyError("the window's anchor is not once in this chromosome")
    W=region_of(anc)["meta"]["window"]; seq=[(int(ia[0])+k)%L for k in range(W)]
    ks=[k for k,q in enumerate(seq) if RG.FAM[s+q]==f]
    if not ks: raise KeyError("this family is not in this genome's window")
    return gi,s,L,seq,ks[0],anc
def glab(i):
    return RG.NAMES[int(RG.NAME[i])] or RG.PRODUCT[int(RG.FAM[i])] or RG.FN[int(RG.FAM[i])]
def gcall_region(a):
    gi,s,L,seq,k,anc=in_window(a); j=seq[k]
    if j+1>=L: raise KeyError("the chromosome ends after this gene")
    ctx=list(range(max(0,j-799),j+1)); t0=time.time()
    p,rk=dist(tuple(f"c{int(RG.PID[s+q])}" for q in ctx))
    f2=int(RG.FAM[s+j+1]); node=next((n for n in region_of(anc)["nodes"] if n["id"]==f"p{f2}"),None)
    ws=[(c,w) for c,w in (node["bfw"] if node else [])]
    if ws: c=max(ws,key=lambda x:p[x[0]]*x[1])[0]; real=dict(row(p,rk,c))
    else: real=dict(fam=-1,name="",product=RG.PRODUCT[f2],rank=None,p=None,score=0)
    real.update(gene=RG.NAMES[int(RG.NAME[s+j+1])],gene_product=RG.PRODUCT[f2])
    g=RG.CGEN[gi]
    out=dict(acc=g["acc"],strain=g.get("strain",""),org=g.get("org",""),locus=k+1,ctx=len(ctx),ctx_first=glab(s+ctx[0]),ctx_last=glab(s+j),
             real=real,**summary(p,rk,a))
    out["ms"]=round((time.time()-t0)*1000,1); return out
def gcall(a):
    if a.get("anchor"): return gcall_region(a)
    g,k=locate(a); k+=1; L=g["genes"]
    if k>=len(L): raise KeyError("this genome's window ends right after this family (contig end)")
    t0=time.time(); p,rk=dist([x[3] for x in L[:k]]); real=int(BF[L[k][3]])
    out=dict(acc=g["acc"],strain=g.get("strain",""),org=g.get("org",""),locus=k+g["start"],
             ctx=k,ctx_first=lab(L[0]),ctx_last=lab(L[k-1]),
             real=dict(row(p,rk,real),gene=L[k][4],gene_product=L[k][5]),**summary(p,rk,a))
    out["ms"]=round((time.time()-t0)*1000,1); return out
def pathcall(a):
    pids,ctx=path_args(a)
    t0=time.time(); p,rk=dist(ctx+pids)
    out=dict(n=len(pids),ctx=len(ctx),**summary(p,rk,a,ntop=30))
    keep=CL[p[CL]>=1e-9]; out["clusters"]=[[int(c),float(f"{p[c]:.4g}")] for c in keep]   # for the boxes of the graph
    out["ms"]=round((time.time()-t0)*1000,1); return out
def parse(txt):
    out=[]
    for x in (x.strip() for x in txt.split(",")):
        if re.fullmatch(r"\d+",x) and int(x)<len(EMB): out.append(int(x))
        elif re.fullmatch(r"c\d+",x) and int(x[1:])<NCE: out.append(x)
        elif x: raise KeyError("unknown protein in the path")
    return out
def path_args(a):
    pids=parse(a.get("pids","")); ctx=parse(a.get("ctx",""))     # ctx: a genome's proteins before the path, read too
    if not pids: raise KeyError("empty path")
    if len(pids)>400 or len(ctx)>400: raise KeyError("path too long (400 proteins at most)")
    return pids,ctx
ACACHE=collections.OrderedDict()
@torch.no_grad()
def attention(pids,nctx):
    """(layers, heads, path queries, [start token, context summed, path keys]) for one read of ctx + path"""
    n=len(pids)+1; x=np.zeros((1,n,480),dtype=np.float32); x[0,1:]=emb(pids)
    out=cm(protein_embeddings=torch.tensor(x,device=dev).to(torch.bfloat16),
           special_tokens_mask=torch.tensor([[CLS]+[PROT]*len(pids)],device=dev),
           token_type_ids=torch.zeros((1,n),dtype=torch.long,device=dev),return_attn_weights=True,return_dict=True)
    A=torch.stack([w[0] for w in out.attentions]).float()                 # layers x heads x n x n
    q=A[:,:,1+nctx:,:]
    return torch.cat([q[...,:1],q[...,1:1+nctx].sum(-1,keepdim=True),q[...,1+nctx:]],-1).cpu().numpy()
def attncall(a):
    pids,ctx=path_args(a); key=(tuple(ctx),tuple(pids)); t0=time.time()
    with LOCK:
        if key not in ACACHE:
            ACACHE[key]=attention(ctx+pids,len(ctx))
            if len(ACACHE)>12: ACACHE.popitem(last=False)
        ACACHE.move_to_end(key); R=ACACHE[key]
    L,H=R.shape[:2]; lay=a.get("layer","mean")
    M=R.mean(0) if lay=="mean" else R[int(lay)]                           # heads x n x (n+2)
    M=np.concatenate([M,M.mean(0,keepdims=True)])                          # and the mean of the heads
    pm=lambda X: np.rint(X*1000).astype(int).ravel().tolist()
    return dict(layers=L,heads=H,n=len(pids),ctx=len(ctx),layer=lay,m=pm(M),last=pm(R[:,:,-1,:]),ms=round((time.time()-t0)*1000,1))
def prefix(a):
    if a.get("anchor"):                        # a complete genome's own path from the window's anchor
        gi,s,L,seq,k,anc=in_window(a); g=RG.CGEN[gi]
        return dict(acc=g["acc"],strain=g.get("strain",""),
                    items=[dict(pid=f"c{int(RG.PID[s+q])}",pfam=RG.FN[int(RG.FAM[s+q])],gene=glab(s+q)) for q in seq[:k+1]],
                    ctx=[f"c{int(RG.PID[s+q])}" for q in range(max(0,seq[0]-400),seq[0])])   # what precedes the anchor
    g,k=locate(a)
    return dict(acc=g["acc"],strain=g.get("strain",""),
                items=[dict(pid=x[3],pfam=W["families"][str(x[1])][0],gene=lab(x)) for x in g["genes"][:k+1]])

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
        if u.path=="/health": return self.js(dict(ok=True,model=MODEL,device=dev,genomes=len(GEN)))
        route={"/gcall":gcall,"/path":pathcall,"/prefix":prefix,"/region":region,"/attn":attncall}.get(u.path)
        if route is None: return super().do_GET()
        try: self.js(route(a))
        except (KeyError,ValueError) as e: self.js(dict(error=str(e).strip("'\"")),400)
    def log_message(self,format,*args): pass

print(f"open http://localhost:{A.port}/",flush=True)
ThreadingHTTPServer(("127.0.0.1",A.port),H).serve_forever()
