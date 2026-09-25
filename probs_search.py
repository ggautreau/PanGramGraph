"""Precompute what the page needs to look up any family at any position.

dists.json keeps the top 15 of each distribution, so a family ranked 1,000th at a
position cannot be found from the page. The full distributions (867 positions x
50,001 outputs) are ~170 MB, too much for a page. This keeps, for every position,
the softmax probability of every family a reader can name or see on the page
(~10,000: families with a gene name or product in the 37 reference annotations,
plus every family the graph or the stored top-15 lists show), and, to give ranks,
the full sorted distribution sampled at 192 ranks (every rank to 64, then
log-spaced to 50,001).

Probabilities are stored as uint16 of -ln p over [0, LMAX]: 0.035 % relative
precision down to p = 1e-20. One lazily loaded script per strain, so the page only
pulls the strain being searched. The loop is the one in dists.py, and the real
gene's rank and probability are checked against dists.json at the end.

    python3 probs_search.py            # -> standalone/probs/index.js, s0.js .. s10.js
"""
import json, os, base64, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
dev="cuda:0"; CLS,PROT=2,4; D=80; LMAX=46.0517; OUT="standalone/probs"
tok=AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm=AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(torch.float16).eval().to(dev)
cm=AutoModelForCausalLM.from_pretrained("macwiatrak/bacformer-causal-complete-genomes",
     trust_remote_code=True).to(torch.bfloat16).eval().to(dev)
ex=np.load("fam_exemplars.npz"); ids,emb,offs=ex["ids"],ex["emb"],ex["offsets"]
E=torch.tensor(emb,device=dev); E=E/E.norm(dim=1,keepdim=True).clamp(min=1e-9)
FAM=torch.tensor(np.concatenate([[f]*(offs[i+1]-offs[i]) for i,f in enumerate(ids)]),device=dev)
annot=json.load(open("fam_annot.json")); dists=json.load(open("dists.json"))
graph=json.load(open("eco_graph.json"))

# --- the searchable families and how a reader names them -------------------
node={n["family"]:n for n in graph["nodes"]}
S=set(map(int,annot))|set(map(int,dists["labels"]))|set(node)
S|={int(e[k][1:]) for e in graph["edges"] for k in ("source","target")}
S=np.array(sorted(S)); col={int(f):i for i,f in enumerate(S)}
def names(f):
    a=annot.get(str(f),{}); out=[]
    if f in node and not node[f]["label"].startswith("fam"): out.append(node[f]["label"])
    out+=[g for g,_ in a.get("gene",[]) if g not in out]
    return out
def product(f):
    if f in node and node[f]["product"]: return node[f]["product"]
    a=annot.get(str(f),{})
    return a["product"][0][0] if a.get("product") else ""
RK=np.unique(np.concatenate([np.arange(1,65),np.round(np.logspace(np.log10(65),np.log10(50001),128))])).astype(int)

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
def quant(p):
    nl=-np.log(np.clip(p.astype(np.float64),1e-300,1.0))
    return np.round(np.clip(nl,0,LMAX)/LMAX*65535).astype("<u2")
b64=lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()

os.makedirs(OUT,exist_ok=True)
G=json.load(open("eco_parsed.json")); check=[]
for gi,g in enumerate(G):
    off=1 if g.get("dnaA_pseudo") else 0
    prots=g["proteins"][:D-off]; pe=embed(prots); fam=assign(pe)
    loci=[]; rows=[]; knots=[]
    for k in range(1,len(fam)):
        p=nxt(pe[:k]); srt=np.sort(p)[::-1]
        loci.append(k+off); rows.append(quant(p[S])); knots.append(quant(srt[RK-1]))
        ref=dists["strains"][gi]["steps"].get(str(k+off))
        t=int(fam[k])
        if ref and t in col:
            check.append((int((p>p[t]).sum()),ref["r"]["k"],float(p[t]),ref["r"]["p"]))
    js=(f"(window.OF_PROBS=window.OF_PROBS||{{}})[{gi}]={{loci:{json.dumps(loci)},"
        f"q:\"{b64(np.stack(rows))}\",k:\"{b64(np.stack(knots))}\"}};\n")
    open(f"{OUT}/s{gi}.js","w").write(js)
    print(f"[{gi+1}/{len(G)}] {g['desc'][:38]:40s} {len(loci)} loci  {len(js)/1e6:.1f} MB",flush=True)

idx=dict(lmax=LMAX,ranks=RK.tolist(),fam=S.tolist(),
         names=["/".join(names(int(f))) for f in S],product=[product(int(f)) for f in S])
open(f"{OUT}/index.js","w").write("window.OF_PROBS_INDEX="+json.dumps(idx,separators=(",",":"))+";\n")
c=np.array(check)
print(f"\nsearchable families {len(S)}  rank knots {len(RK)}")
print(f"real-gene check on {len(c)} positions: rank identical {np.mean(c[:,0]==c[:,1])*100:.1f}%  "
      f"max |p - stored p| {np.max(np.abs(c[:,2]-c[:,3])):.2e}")
