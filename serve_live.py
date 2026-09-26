"""The model behind the PanGramGraph page, live, on any of the 2,002 PanGBank genomes:
Bacformer's call after a family in one genome, and the exact rank and softmax
probability of any of its 50,000 families there, computed when the page asks.

    python3 serve_live.py            # then open http://localhost:8765/

Serves standalone/ and answers the page; the page also finds it when opened as a
local file. Listens on 127.0.0.1 unless --host says otherwise (0.0.0.0 in the
Hugging Face Space, space/). Proteins are the dnaA windows extracted by
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

What the model itself knows about one accessory element (the Coupled regions tab): pg_knock.influence,
the in-silico knockout of pg_knockout.py on demand, with the request and response of serve_influence.py.
pg_knock is imported at the first of these requests, never at start: without its files (pg_knock.py,
pgb/region_sets.json, and the decoder pgb/knock/decoder_halves.npz or base_top64_{f,p}.npy) the server
starts and serves the rest as before, and /health says why influence is not available.
    sets=long (default) | close: the link sets the elements and their links come from, the page's two layers
        (pgb/region_sets.json, pgb/region_sets_close.json, which is optional); /health lists the sets present
        with the md5 of each file, and every answer names its set and md5 (the page checks it shows the same)
    GET /elements?acc=<assembly> | g=<index 0-539 of the complete chromosomes> [&sets=]
        the accessory elements of that chromosome (panRGP runs by spot, coupled regions present at
        their spot): kind, id, label, start, end, entry gene, size
    GET /influence?acc= | g=  &region=<region id> | &spot=<spot id> | &genes=<p,q,...>  [&sets=] [&n_ctrl=8] [&wait=<s>]
        baseline, deletion of the element and n_ctrl size-matched whole-element control deletions, one
        fp32 pass each (its own fp32 copy of the model): ~3 s on a GPU (~8 s for the first request), ~2-4 min
        on 2 CPU threads (a Hugging Face Space). One computation at a time, in a queue; results cached.
          200  the result (serve_influence.py's JSON) with sets, sets_md5 and links: the elements linked to
               the deleted one and where each is in this genome (read among the neighbours, before it, beyond
               the 1,500 genes read, not located at its spot, absent)
          202  a job: {job, state: queued | running, stage, loading, done, total, position, elapsed_s, eta_s};
               poll GET /influence?job=<id>[&wait=<s>] until 200 (or an error). eta_s: from the passes done
               (none before the first one), for a queued job from the jobs before it
        wait: hold the request until the result is ready or for that many seconds (at most 60; default
        60 on a GPU, 0 on a CPU, where a call is a background job to poll)
          400  bad request (unknown genome or set, element not present in it...), checked before any queue
          404  unknown or expired job;  410  a queued job dropped because nobody asked for it for 30 s
          500  the computation failed (also a failed numerical check)
          503  influence not available here (why in `error`; a failed load is retried after a minute), or too
               many requests waiting (`retry_s`)
    GPU_MEM_FRAC=<0-1> caps the GPU memory of the whole process (a second instance next to a running one);
    INFLUENCE_DECODER=compact | full forces the decoder (default: the compact one when it is there, identical);
    INFLUENCE_JOBS=1 makes every influence call a job on a GPU too; INFLUENCE_QUEUE caps the jobs waiting (2 on a
    CPU, 4 on a GPU). On a CPU the influence passes run on 2 threads (the whole process then does): their
    exactness check (genes before the deletion unchanged, bit for bit) holds at 1-2 threads, not at 4 or more.
"""
import os, json, re, time, argparse, threading, urllib.parse, collections, numpy as np, torch
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from transformers import AutoModelForCausalLM

P=argparse.ArgumentParser()
P.add_argument("--host",default=os.environ.get("HOST","127.0.0.1"),help="0.0.0.0 inside a container (a Space)")
P.add_argument("--port",type=int,default=int(os.environ.get("PORT",8765))); A=P.parse_args()
dev="cuda:0" if torch.cuda.is_available() else "cpu"; CLS,PROT=2,4
DT=torch.bfloat16 if dev!="cpu" else torch.float32         # bfloat16 on a GPU; float32 on a CPU, where it is faster
if dev=="cpu": torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS","2")))
elif os.environ.get("GPU_MEM_FRAC"): torch.cuda.set_per_process_memory_fraction(float(os.environ["GPU_MEM_FRAC"]),0)
MODEL="macwiatrak/bacformer-causal-complete-genomes"
cm=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True).to(DT).eval().to(dev)
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
NCE=os.path.getsize("pgb/chrom_emb.f16")//(480*2); CEMB=np.memmap("pgb/chrom_emb.f16",dtype=np.float16,mode="r",shape=(NCE,480))
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
BUILD_LOCK=threading.Lock()                                # one window built at a time: memory stays bounded
def region_of(anc):
    with REG_LOCK:
        if anc in REG: REG.move_to_end(anc); return REG[anc]
    with BUILD_LOCK:
        with REG_LOCK:
            if anc in REG: return REG[anc]
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
    return torch.softmax(cm(protein_embeddings=torch.tensor(x,device=dev).to(DT),
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
def summary(p,rk,a,ntop=16):
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
    out=cm(protein_embeddings=torch.tensor(x,device=dev).to(DT),
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
    # each head over all rows: its share on the start token, on the context before the path, on the previous
    # protein and on itself, and how far back it looks among the path's proteins (mean distance, in proteins)
    n=len(pids); P=R[:,:,:,2:]; ii=np.arange(n)
    dist=(P*np.clip(ii[:,None]-ii[None,:],0,None)).sum(-1)/np.maximum(P.sum(-1),1e-9)
    prev=P[:,:,ii[1:],ii[1:]-1].mean(-1) if n>1 else np.zeros((L,H))
    r3=lambda X: np.round(np.asarray(X,dtype=np.float64),3).tolist()
    stats=dict(start=r3(R[:,:,:,0].mean(-1)),before=r3(R[:,:,:,1].mean(-1)),prev=r3(prev),self=r3(P[:,:,ii,ii].mean(-1)),dist=r3(dist.mean(-1)))
    return dict(layers=L,heads=H,n=n,ctx=len(ctx),layer=lay,m=pm(M),last=pm(R[:,:,-1,:]),stats=stats,ms=round((time.time()-t0)*1000,1))
def prefix(a):
    if a.get("anchor"):                        # a complete genome's own path from the window's anchor
        gi,s,L,seq,k,anc=in_window(a); g=RG.CGEN[gi]
        return dict(acc=g["acc"],strain=g.get("strain",""),
                    items=[dict(pid=f"c{int(RG.PID[s+q])}",pfam=RG.FN[int(RG.FAM[s+q])],gene=glab(s+q)) for q in seq[:k+1]],
                    ctx=[f"c{int(RG.PID[s+q])}" for q in range(max(0,seq[0]-400),seq[0])])   # what precedes the anchor
    g,k=locate(a)
    return dict(acc=g["acc"],strain=g.get("strain",""),
                items=[dict(pid=x[3],pfam=W["families"][str(x[1])][0],gene=lab(x)) for x in g["genes"][:k+1]])

MAXQ=int(os.environ.get("MAX_QUEUE","6")); QN=[0]; QL=threading.Lock()
class Busy(Exception): pass
def queued(fn,a,admitted=False):
    """at most MAXQ requests wait for the model; beyond that the page is asked to try again
    (admitted: an influence job, refused or not when it was queued, only counted while it runs)"""
    with QL:
        if QN[0]>=MAXQ and not admitted: raise Busy("the model is busy, try again in a moment")
        QN[0]+=1
    try: return fn(a)
    finally:
        with QL: QN[0]-=1

# ---------------------------------------------------------------- influence of one accessory element (Coupled regions)
# pg_knock (its data, its lineage-CV decoder and its own fp32 copy of the model) is loaded at the first request, per
# link set (long: pgb/region_sets.json, close: pgb/region_sets_close.json; one model and one decoder serve both).
# Each computation is a job run by one worker thread, first in first out, counted by queued() while it runs;
# a request waits for it (GPU, ~3 s) or gets the job to poll (CPU, ~2-4 min on 2 threads).
import queue, secrets, hashlib
class Unavailable(Exception): pass
INF_NEED=["pg_knock.py","pgb/region_sets.json","pgb/chrom.npz","pgb/chrom_genomes.json","pgb/chrom_emb.f16"]
INF_SETS={"long":"pgb/region_sets.json","close":"pgb/region_sets_close.json"}   # the page's two layers; close is optional
INF_DECODERS={"full":["pgb/knock/base_top64_f.npy","pgb/knock/base_top64_p.npy"],"compact":["pgb/knock/decoder_halves.npz"]}
INF_FORCE=os.environ.get("INFLUENCE_DECODER","").strip().lower() or None       # compact | full | None (auto)
INF_JOB=dev=="cpu" or os.environ.get("INFLUENCE_JOBS")=="1"
INF_MAXQ=int(os.environ.get("INFLUENCE_QUEUE","2" if dev=="cpu" else "4"))     # jobs waiting at most (a CPU job is minutes)
INF_KEEP=600                     # seconds a finished job stays to be fetched
INF_IDLE=30.0                    # a queued job that nobody has asked about for this long is dropped (the page polls every ~4 s)
INF_RETRY=60.0                   # a failed load is tried again by a request after this long
INF_THREADS=2                    # CPU threads of the influence passes: its exactness check (g) holds at 1-2 threads, not at 4+
INF=dict(K=None,D={},ACC=None,ctx={},error=None,error_t=0.0,load_s=None,last_s=None)
INF_LOAD=threading.Lock(); JL=threading.Lock(); JOBS=collections.OrderedDict(); INF_CACHE=collections.OrderedDict()
INF_Q=queue.Queue(); INF_W=[]; _MD5={}
def inf_decoder():
    """which decoder influence reads here, or None: the compact one (the same rows bit for bit, loaded in 1 s instead of
    3, and a genome's rows in 50 ms instead of ~9 s) unless the full one is newer (rebuilt since the export)"""
    have={k:all(os.path.exists(f) for f in v) for k,v in INF_DECODERS.items()}
    if INF_FORCE in have: return INF_FORCE if have[INF_FORCE] else None
    if have["compact"] and have["full"]:
        return "full" if max(map(os.path.getmtime,INF_DECODERS["full"]))>os.path.getmtime(INF_DECODERS["compact"][0]) else "compact"
    return "full" if have["full"] else "compact" if have["compact"] else None
def inf_missing():
    miss=[f for f in INF_NEED if not os.path.exists(f)]
    if inf_decoder() is None: miss.append("the decoder pgb/knock/decoder_halves.npz" if INF_FORCE!="full" else "pgb/knock/base_top64_{f,p}.npy")
    return miss
def inf_sets():
    """{set: md5 of its file} for the link sets present here: the page checks that it shows the same elements"""
    out={}
    for k,p in INF_SETS.items():
        try: st=os.stat(p)
        except OSError: continue
        c=_MD5.get(p)
        if not c or c[0]!=(st.st_mtime,st.st_size): c=_MD5[p]=((st.st_mtime,st.st_size),hashlib.md5(open(p,"rb").read()).hexdigest())
        out[k]=c[1]
    return out
def inf_failed():
    """the load error, if any; after INF_RETRY seconds the next request tries again"""
    if INF["error"] and time.time()-INF["error_t"]>INF_RETRY: INF["error"]=None
    return INF["error"]
def inf_fail(msg):
    INF.update(error=msg,error_t=time.time()); raise Unavailable(msg)
def inf_setname(a):
    s=(a.get("sets") or "long").strip().lower()
    if s not in INF_SETS: raise ValueError("sets: long (the default) or close")
    if s not in inf_sets(): raise ValueError(f"this server has no {s}-range link sets ({INF_SETS[s]} is missing): it reads the "+" and ".join(inf_sets()))
    return s
def inf_data(sets="long"):
    """pg_knock and the Data of one link set (chromosomes, accessory elements, controls): enough for /elements"""
    if sets not in INF["D"]:
        with INF_LOAD:
            if sets not in INF["D"]:
                miss=inf_missing()
                if miss: raise Unavailable("influence is not available on this server: missing "+", ".join(miss))
                if inf_failed(): raise Unavailable(INF["error"])
                try:
                    t0=time.time(); import pg_knock as K
                    path=None if sets=="long" else INF_SETS[sets]
                    try: D=K.Data(sets=path)
                    except Exception as e:          # units cache unreadable or not writable: rebuilt in memory (~5-10 s)
                        print(f"influence: the units cache of the {sets}-range sets is not usable ({type(e).__name__}: {e}); rebuilding it in memory",flush=True)
                        D=K.Data(cache=False,sets=path)
                    INF["K"]=K; INF["D"][sets]=D
                    if INF["ACC"] is None: INF["ACC"]={a:i for i,a in enumerate(D.ACC)}
                    INF["load_s"]=(INF["load_s"] or 0)+time.time()-t0
                except Exception as e: inf_fail(f"influence could not load its data: {type(e).__name__}: {e}")
    return INF["K"],INF["D"][sets]
def inf_ctx(sets="long"):
    """(Data, Decoder, Runner) of pg_knock for one link set: the decoder and the fp32 model too, for /influence. One
    model and one decoder serve both sets when their lineage halves agree (the same 540 genomes and clusters)"""
    K,D=inf_data(sets)
    if sets not in INF["ctx"]:
        with INF_LOAD:
            if inf_failed(): raise Unavailable(INF["error"])
            if sets not in INF["ctx"]:
                try:
                    t0=time.time(); comp={"compact":True,"full":False}.get(inf_decoder())
                    if not INF["ctx"]:
                        INF["ctx"][sets]=K.context(dev=dev,threads=min(INF_THREADS,torch.get_num_threads()),frac=None,data=D,compact=comp)
                    else:
                        D0,dec,R=next(iter(INF["ctx"].values()))
                        if not (D0.NP==D.NP and np.array_equal(D0.HALF,D.HALF) and list(D0.LIN)==list(D.LIN)): dec=K.Decoder(D,compact=comp)
                        INF["ctx"][sets]=(D,dec,R)
                    INF["load_s"]=(INF["load_s"] or 0)+time.time()-t0
                except Exception as e: inf_fail(f"influence could not load the model or its decoder: {type(e).__name__}: {e}")
    return INF["ctx"][sets]
def inf_genome(D,a):
    if a.get("acc"):
        if a["acc"] not in INF["ACC"]: raise ValueError(f"unknown assembly {a['acc']} (influence reads the {D.NG} complete chromosomes)")
        return INF["ACC"][a["acc"]]
    try: g=int(a.get("g",-1))
    except ValueError: g=-1
    if not 0<=g<D.NG: raise ValueError(f"give acc=<assembly> or g=<genome index 0-{D.NG-1}>")
    return g
def elements(a):
    sets=inf_setname(a); K,D=inf_data(sets); g=inf_genome(D,a); E=D.elements(g)
    return dict(g=g,acc=D.ACC[g],strain=D.STRAIN[g],n_genes=int(D.GL[g]),sets=sets,sets_md5=D.SETS_MD5,
                elements=[dict(kind="region" if int(k)==1 else "spot",id=int(i),label=D.element_label(k,i)[:60],
                               start=int(s),end=int(e),entry=int(en),size=int(z))
                          for k,i,s,e,en,z in zip(E["kind"],E["id"],E["start"],E["end"],E["entry"],E["size"])])
def inf_request(a):
    """-> (cache key, genome, element, n_ctrl), the element checked against the genome (400 before any queue)"""
    sets=inf_setname(a); K,D=inf_data(sets); g=inf_genome(D,a)
    try:
        n_ctrl=max(3,min(16,int(a.get("n_ctrl",8))))
        if "region" in a: el=("region",int(a["region"]))
        elif "spot" in a: el=("spot",int(a["spot"]))
        elif "genes" in a:
            el=("genes",tuple(sorted({int(x) for x in a["genes"].split(",") if x.strip()})))
            if not el[1] or not all(0<=p<int(D.GL[g]) for p in el[1]): raise ValueError(f"genes: positions 0-{int(D.GL[g])-1} of this chromosome")
            if len(el[1])>2000: raise ValueError("genes: 2,000 at most")
        else: raise ValueError("give region=, spot= or genes=")
        if el[0]=="region" and not 0<=el[1]<D.NR: raise ValueError(f"region: an id 0-{D.NR-1} of the {sets}-range sets")
        K.resolve_element(D,g,el)
    except AssertionError as e: raise ValueError(str(e))
    return (sets,g,el,n_ctrl),g,el,n_ctrl
def inf_left(x,now):
    """seconds left of a running job, from its passes so far (None before the first pass is done)"""
    if x.get("loading") or not (x["done"] and x["total"] and x.get("t_pass0")): return None
    tp=(x["t_tick"]-x["t_pass0"])/x["done"]; return max(0.5,tp*(x["total"]-x["done"])-(now-x["t_tick"]))
def inf_status(j):
    """what a job is doing, for the page's progress"""
    now=time.time()
    with JL: js=list(JOBS.values())
    run=[x for x in js if x["state"]=="running"]
    ahead=sum(1 for x in js if x["state"]=="queued" and x["t"]<j["t"])
    per=INF["last_s"] or (150.0 if dev=="cpu" else 4.0)                  # a whole call: the last one measured here
    if j["state"]=="running": pos=0; eta=inf_left(j,now)
    else:
        pos=ahead+len(run)
        lr=[inf_left(x,now) for x in run]; eta=sum(per if v is None else v for v in lr)+per*(ahead+1)
    stage=j["stage"] if j["state"]!="queued" else ("starting" if pos==0 else f"waiting for {pos} computation{'s' if pos>1 else ''} before it")
    return dict(job=j["id"],state=j["state"],stage=stage,loading=bool(j.get("loading")),done=j["done"],total=j["total"],position=pos,
                elapsed_s=round(now-j["t"],1),eta_s=None if eta is None else round(eta,1),device=dev,poll=f"/influence?job={j['id']}")
def inf_worker():
    while True:
        j=INF_Q.get()
        with JL: idle=j["watch"]==0 and time.time()-j["t_seen"]>INF_IDLE
        if idle:                                                          # nobody is waiting for it any more
            j.update(state="error",error=f"dropped: nobody asked for it for {INF_IDLE:.0f} s; ask again",code=410,t_end=time.time())
            j["event"].set(); continue
        try: queued(inf_run,j,admitted=True)
        except Exception as e:
            j.update(state="error",error=f"the computation failed: {type(e).__name__}: {e}",code=500)
            if dev!="cpu": torch.cuda.empty_cache()
        finally:
            j["t_end"]=time.time(); j["event"].set()
def inf_links(D,g,res):
    """the elements linked to the deleted one in the set read, and where each is in this genome: among the neighbours
    read (where="read"), before it (the model reads from dnaA on: the deletion cannot change it), beyond the 1,500
    genes read, listed but not read, carried but not located at its spot, or absent"""
    el=res["element"]
    if el["kind"]!=1: return []
    ri=int(el["id"]); nb={x["id"] for x in res["neighbours"] if x["kind"]=="region"}
    E=D.elements(g); at={int(i):(int(s),int(e)) for k,i,s,e in zip(E["kind"],E["id"],E["start"],E["end"]) if int(k)==1}
    out=[]
    for l in D.LINKS:
        if ri not in (l["a"],l["b"]): continue
        rj=int(l["b"] if l["a"]==ri else l["a"]); r=D.REG[rj]
        x=dict(id=rj,label=r["label"][:60],sign=int(l["sign"]),spot=-1 if r["spot"] is None else int(r["spot"]))
        if rj in at:
            s,e=at[rj]; d=s-int(el["end"])
            x.update(start=s,end=e,dist=d,where="read" if rj in nb else "before" if d<=0 else "beyond" if d>1500 else "unread")
        else: x["where"]="unlocated" if D.PRES[rj,g] else "absent"
        out.append(x)
    return out
def inf_run(j):
    sets,g,el,n_ctrl=j["key"]; first=sets not in INF["ctx"]
    j.update(state="running",loading=first,t_run=time.time(),
             stage=("loading the model and its decoder (first request)" if not INF["ctx"] else "loading the close-range elements") if first else "choosing the controls")
    try: ctx=inf_ctx(sets)
    except Unavailable as e: j.update(state="error",error=str(e),code=503); return
    j.update(stage="choosing the controls",loading=False)
    K=INF["K"]
    def tick(done,total,what): j.update(done=done,total=total,stage=what,t_pass0=j.get("t_pass0") or time.time(),t_tick=time.time())
    try: res=K.influence(g,el,n_ctrl=n_ctrl,ctx=ctx,progress=tick)
    except AssertionError as e:                                    # an internal check (the input was checked before queueing)
        j.update(state="error",error=f"a numerical check of the computation failed ({e})",code=500); return
    res=dict(res,sets=sets,sets_md5=ctx[0].SETS_MD5,links=inf_links(ctx[0],g,res))
    INF["last_s"]=time.time()-(j["t_pass0"] or j["t_run"])
    with JL:
        INF_CACHE[j["key"]]=res
        while len(INF_CACHE)>64: INF_CACHE.popitem(last=False)
    j.update(state="done",stage="done",result=res,t_end=time.time())
def inf_submit(key):
    """the job computing `key`: the one already queued or running, else a new one (503 when too many wait)"""
    with JL:
        now=time.time()
        old=[k for k,x in JOBS.items() if x["state"] in ("done","error") and now-(x["t_end"] or now)>INF_KEEP]
        old+=[k for k,x in JOBS.items() if x["state"] in ("done","error")][:max(0,len(JOBS)-len(old)-200)]
        for k in set(old): del JOBS[k]
        for x in JOBS.values():
            if x["key"]==key and x["state"] in ("queued","running"): x["t_seen"]=now; return x
        waiting=sum(1 for x in JOBS.values() if x["state"] in ("queued","running"))
        if waiting>=INF_MAXQ: raise Busy(f"{waiting} influence computation{'s are' if waiting>1 else ' is'} already waiting, try again "
                                         +("in a few minutes" if dev=="cpu" else "in a moment"))
        if QN[0]>=MAXQ: raise Busy("the model is busy, try again in a moment")
        j=dict(id=secrets.token_hex(6),key=key,state="queued",stage="",done=0,total=None,
               t=now,t_seen=now,watch=0,t_pass0=None,t_end=None,event=threading.Event())
        JOBS[j["id"]]=j
        if not INF_W:
            INF_W.append(threading.Thread(target=inf_worker,daemon=True)); INF_W[0].start()
    INF_Q.put(j); return j
def inf_answer(j,wait):
    with JL: j["watch"]+=1; j["t_seen"]=time.time()
    try:
        if wait>0: j["event"].wait(min(wait,60.0))
    finally:
        with JL: j["watch"]-=1; j["t_seen"]=time.time()
    if j["state"]=="done": return 200,j["result"]
    if j["state"]=="error": return j.get("code",500),dict(error=j["error"],job=j["id"],state="error")
    return 202,inf_status(j)
def influence(a):
    """-> (HTTP code, JSON) for /influence: a result (200), a job to poll (202) or an error"""
    wait=a.get("wait")
    try: wait=float(wait) if wait not in (None,"") else None
    except ValueError: raise ValueError("wait: seconds")
    if a.get("job"):
        with JL: j=JOBS.get(a["job"])
        if j is None: return 404,dict(error="unknown or expired job: ask again with the genome and the element")
        return inf_answer(j,wait or 0)
    if inf_failed(): raise Unavailable(INF["error"])
    key,g,el,n_ctrl=inf_request(a)
    with JL:
        if key in INF_CACHE: INF_CACHE.move_to_end(key); return 200,INF_CACHE[key]
    return inf_answer(inf_submit(key),(0.0 if INF_JOB else 60.0) if wait is None else wait)
def inf_health():
    miss=inf_missing(); err=inf_failed(); ok=not miss and not err
    with JL: waiting=sum(1 for x in JOBS.values() if x["state"] in ("queued","running"))
    r1=lambda x: None if x is None else round(x,1)
    out=dict(available=ok,loaded=bool(INF["ctx"]),device=dev,mode="job" if INF_JOB else "wait",decoder=inf_decoder(),sets=inf_sets(),
             waiting=waiting,max_waiting=INF_MAXQ,cached=len(INF_CACHE),
             seconds=dict(load=r1(INF["load_s"]),last=r1(INF["last_s"]),     # measured here: loading; the last computation
                          expected=r1(INF["last_s"]) or (150 if dev=="cpu" else 4),   # per element until one is measured
                          expected_first_load=15 if dev=="cpu" else 8))
    if not ok: out["reason"]=err or "missing "+", ".join(miss)
    return out
def emsg(e):
    """the message of a KeyError without its quotes, of anything else as is"""
    return str(e.args[0]) if isinstance(e,KeyError) and e.args else str(e)

class H(SimpleHTTPRequestHandler):
    extensions_map={**SimpleHTTPRequestHandler.extensions_map,".html":"text/html; charset=utf-8",
                    ".js":"text/javascript; charset=utf-8"}
    def __init__(self,*a,**k): super().__init__(*a,directory="standalone",**k)
    def end_headers(self):
        o=self.headers.get("Origin") or ""
        if o=="null" or re.fullmatch(r"http://(localhost|127\.0\.0\.1)(:\d+)?",o):     # a local page, or one opened as a file
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
        if u.path=="/health": return self.js(dict(ok=True,model=MODEL,device=dev,genomes=len(GEN),influence=inf_health()))
        if u.path in ("/influence","/elements"):
            try:
                if u.path=="/elements": return self.js(elements(a))
                code,obj=influence(a); return self.js(obj,code)
            except Busy as e: return self.js(dict(error=str(e),retry_s=10 if dev=="cpu" else 3),503)
            except Unavailable as e: return self.js(dict(error=str(e),available=False),503)
            except (KeyError,ValueError) as e: return self.js(dict(error=emsg(e)),400)
            except Exception as e: return self.js(dict(error=f"{type(e).__name__}: {e}"),500)
        route={"/gcall":gcall,"/path":pathcall,"/prefix":prefix,"/region":region,"/attn":attncall}.get(u.path)
        if route is None: return super().do_GET()
        try: self.js(queued(route,a))
        except Busy as e: self.js(dict(error=str(e)),503)
        except (KeyError,ValueError) as e: self.js(dict(error=emsg(e)),400)
    def list_directory(self,path): self.send_error(404); return None     # files only, no listings
    def log_message(self,format,*args): pass

print(f"open http://localhost:{A.port}/",flush=True)
ThreadingHTTPServer.daemon_threads=True
ThreadingHTTPServer((A.host,A.port),H).serve_forever()
