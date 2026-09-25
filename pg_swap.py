"""Causal test of allelic contingency: swap alleles between genomes that share the same
family path, and see whether the model's branch prediction follows the donor.

At a fork, take a genome whose branch is b_from. Replace the L proteins immediately
upstream by those of a donor genome whose upstream FAMILY PATH over those L genes is
identical, so the pangenome graph sees exactly the same path before and after the swap;
only the alleles change. Two donors:

  same-branch     donor also took b_from   -> the control: the swap changes alleles but
                  not the branch information, so the prediction should stay
  cross-branch    donor took b_to          -> if the prediction moves to b_to, the allele
                  carries the branch, and it does so causally

Both prefixes are made of real proteins of the right families, in the same order, so a
move cannot be blamed on the input being unusual (the answer to the modal-protein
objection). Reported: mean P(b_to) under each swap, and how often the top-1 branch flips.

    python3 pg_swap.py [--L 10] [--forks 60]
"""
import sys, json, argparse, collections, warnings, numpy as np, torch
from scipy import sparse
from transformers import AutoModelForCausalLM
warnings.simplefilter("ignore")
ap=argparse.ArgumentParser(); ap.add_argument("--L",type=int,default=10); ap.add_argument("--forks",type=int,default=60)
ap.add_argument("--ctx",type=int,default=120); ap.add_argument("--max_pairs",type=int,default=60); A=ap.parse_args()
dev="cuda:0"; CLS,PROT=2,4; rng=np.random.default_rng(0)
Z=np.load("pgb/chrom.npz"); off=Z["offsets"]; FAM=Z["fam"]; PID=Z["pid"]; RGP=Z["rgp"]; NG=len(off)-1; NP=off[-1]
J=json.load(open("pgb/chrom_genomes.json")); FN=J["families"]
name_of=np.concatenate([np.array(n,dtype=object) for n in J["gene_names"]])
NPROT=sum(1 for _ in open("pgb/chrom_prot.txt"))
EMB=np.memmap("pgb/chrom_emb.f16",dtype=np.float16,mode="r",shape=(NPROT,480))
TF=np.memmap("pgb/calls_top15_f.i32",dtype=np.int32,mode="r",shape=(NP,15)); TP=np.memmap("pgb/calls_top15_p.f16",dtype=np.float16,mode="r",shape=(NP,15))
SITES=np.load("pgb/fork_sites.npy"); gen_of=np.repeat(np.arange(NG),np.diff(off)); first=np.zeros(NP,bool); first[off[:-1]]=True
cm=AutoModelForCausalLM.from_pretrained("macwiatrak/bacformer-causal-complete-genomes",
     trust_remote_code=True).to(torch.bfloat16).eval().to(dev)
fams=np.unique(FAM); FI=np.full(FAM.max()+1,-1); FI[fams]=np.arange(len(fams))
GH=np.arange(NG)%2
def decoder(hh):
    m=np.where((GH[gen_of]!=hh)&~first)[0]
    Amat=sparse.csr_matrix((np.asarray(TP[m],np.float32).ravel(),(np.repeat(FI[FAM[m]],15),np.asarray(TF[m]).ravel())),shape=(len(fams),50001))
    return Amat.tocsr(),np.asarray(Amat.sum(0)).ravel()
DEC={hh:decoder(hh) for hh in (0,1)}; print("decoders built",flush=True)
def bweights(br,hh):
    Amat,tt=DEC[hh]; rows=[Amat.getrow(FI[b]) for b in br]
    cl=np.array(sorted({int(c) for r in rows for c in r.indices}),dtype=np.int64)
    pos={c:i for i,c in enumerate(cl.tolist())}; M=np.zeros((len(cl)+1,len(br)))
    for j,r in enumerate(rows):
        for c,v in zip(r.indices.tolist(),r.data.tolist()): M[pos[c],j]=v/max(tt[c],1e-12)
    return cl,M
@torch.no_grad()
def branch_p(prefixes,cl,M,bs=8):
    """prefixes: list of pid arrays (may differ in length) -> (n, branches) normalised"""
    out=[]
    for i in range(0,len(prefixes),bs):
        ch=prefixes[i:i+bs]; n=max(len(x) for x in ch)
        x=torch.zeros((len(ch),n+1,480),device=dev,dtype=torch.bfloat16)
        stm=torch.zeros((len(ch),n+1),dtype=torch.long,device=dev)
        for j,pf in enumerate(ch):                       # left-pad: the last position is the call
            x[j,n+1-len(pf):]=torch.tensor(np.asarray(EMB[pf],np.float32),device=dev).to(torch.bfloat16)
            stm[j,n-len(pf)]=CLS; stm[j,n+1-len(pf):]=PROT
        lg=cm(protein_embeddings=x,special_tokens_mask=stm,
              token_type_ids=torch.zeros((len(ch),n+1),dtype=torch.long,device=dev),return_dict=True).logits[:,-1].float()
        p=torch.softmax(lg,-1).cpu().numpy()
        q=np.zeros((len(ch),M.shape[1]))
        for j in range(len(ch)):
            nz=np.where(p[j]>1e-9)[0]; k=np.clip(np.searchsorted(cl,nz),0,len(cl)-1)
            k=np.where(cl[k]==nz,k,len(cl)); q[j]=(p[j][nz][:,None]*M[k]).sum(0)
        s=q.sum(1,keepdims=True); out.append(np.where(s>0,q/np.maximum(s,1e-12),1.0/M.shape[1]))
    return np.concatenate(out,0)

succ=collections.defaultdict(collections.Counter); occ=collections.Counter()
for p in SITES: succ[int(FAM[p-1])][int(FAM[p])]+=1; occ[int(FAM[p-1])]+=1
by_fork=collections.defaultdict(list)
for p in SITES: by_fork[int(FAM[p-1])].append(int(p))
AL=json.load(open("pgb/allelic.json"))
fidx={FN[F]:F for F in by_fork}
cand=sorted(AL["per_fork"],key=lambda r:r["bits"]["family"]-r["bits"]["model"],reverse=True)
L=A.L; rows=[]; done=0
for rec in cand:
    if done>=A.forks: break
    F=fidx.get(rec["fork"])
    if F is None: continue
    br=[b for b,c in succ[F].most_common(8) if c>=max(0.10*occ[F],20)]
    if len(br)<2: continue
    ps=[p for p in by_fork[F] if FAM[p] in br and (p-off[gen_of[p]])>A.ctx]
    if len(ps)<40: continue
    key=lambda p: tuple(FAM[p-L:p].tolist())
    bykey=collections.defaultdict(lambda: collections.defaultdict(list))
    for p in ps: bykey[key(p)][br.index(FAM[p])].append(p)
    pairs=[]
    for k,d in bykey.items():
        for i in d:
            for j in d:
                if i==j: continue
                for p in d[i]:
                    donors_x=[q for q in d[j] if gen_of[q]!=gen_of[p]]
                    donors_s=[q for q in d[i] if gen_of[q]!=gen_of[p]]
                    if donors_x and donors_s: pairs.append((p,i,j,rng.choice(donors_x),rng.choice(donors_s)))
    if len(pairs)<10: continue
    if len(pairs)>A.max_pairs: pairs=[pairs[i] for i in rng.choice(len(pairs),A.max_pairs,replace=False)]
    for hh in (0,1):
        sel=[x for x in pairs if GH[gen_of[x[0]]]==hh]
        if not sel: continue
        cl,M=bweights(br,hh)
        pre=lambda p,src=None: np.concatenate([PID[max(off[gen_of[p]],p-A.ctx):p-L], PID[src-L:src] if src is not None else PID[p-L:p]])
        P0=branch_p([pre(p) for p,_,_,_,_ in sel],cl,M)
        PX=branch_p([pre(p,dx) for p,_,_,dx,_ in sel],cl,M)
        PS=branch_p([pre(p,ds) for p,_,_,_,ds in sel],cl,M)
        for (p,i,j,dx,ds),q0,qx,qs in zip(sel,P0,PX,PS):
            rows.append(dict(fork=FN[F],cls=rec["cls"],rgp=bool(RGP[p]),site=int(p),i=int(i),j=int(j),
                p0_from=float(q0[i]),p0_to=float(q0[j]),px_from=float(qx[i]),px_to=float(qx[j]),
                ps_from=float(qs[i]),ps_to=float(qs[j]),
                top0=int(q0.argmax()),topx=int(qx.argmax()),tops=int(qs.argmax())))
    done+=1
    if done%10==0: print(f"  {done}/{A.forks} forks, {len(rows)} swaps",flush=True)
json.dump(dict(L=L,ctx=A.ctx,rows=rows),open("pgb/swap.json","w"))
R=rows; n=len(R)
f=lambda k: np.mean([r[k] for r in R])
flip_x=np.mean([r["topx"]==r["j"] for r in R]); flip_s=np.mean([r["tops"]==r["j"] for r in R]); keep0=np.mean([r["top0"]==r["i"] for r in R])
print(f"\n{n} swaps over {done} forks, {L} proteins replaced, identical family path, {A.ctx} genes of context")
print(f"  no swap          P(own branch) {f('p0_from'):.3f}   P(other branch) {f('p0_to'):.3f}   top-1 = own branch {keep0:.1%}")
print(f"  same-branch swap P(own branch) {f('ps_from'):.3f}   P(other branch) {f('ps_to'):.3f}   top-1 = other branch {flip_s:.1%}")
print(f"  cross-branch swap P(own branch) {f('px_from'):.3f}  P(other branch) {f('px_to'):.3f}   top-1 = other branch {flip_x:.1%}")
from scipy.stats import wilcoxon
d=np.array([r["px_to"]-r["ps_to"] for r in R]); print(f"  cross minus same, P(other branch): {d.mean():+.3f}, Wilcoxon p={wilcoxon(d).pvalue:.2e}")
byc=collections.defaultdict(list)
for r in R: byc[r["cls"]].append(r["px_to"]-r["ps_to"])
print("  by class:", {c:round(float(np.mean(v)),3) for c,v in sorted(byc.items(),key=lambda x:-np.mean(x[1]))})
ir=[r["px_to"]-r["ps_to"] for r in R if r["rgp"]]; orr=[r["px_to"]-r["ps_to"] for r in R if not r["rgp"]]
if ir and orr: print(f"  in a region of plasticity {np.mean(ir):+.3f} vs outside {np.mean(orr):+.3f}")
