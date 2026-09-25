"""What upstream decides the branch at the forks where Bacformer beats both the genome's
nearest relatives and the graph's family paths (fork_rules.py)?

For every genome at the fork, each of the 30 proteins upstream is deleted in turn and
the model's probability of the branch the genome actually takes is read again (branch
probabilities through the cross-validated decoder, as in fork_rules.py). A protein
whose deletion drops that probability carries the information. Per upstream family:
mean drop in log2 probability and how often the deletion flips the call.

Then, for the strongest determinant, an allele swap: in genomes of the minority branch,
its protein is replaced by the most common allele of the same family in genomes of the
majority branch. If the call flips, what decides is the sequence of a family that both
branches carry, which the pangenome graph (one node per family) cannot see.

    python3 fork_ablation.py     # -> pgb/fork_ablation.json
"""
import json, collections, numpy as np, torch
from scipy import sparse
from transformers import AutoModelForCausalLM
dev="cuda:0"; CLS,PROT=2,4; UP=30
TARGETS=[(46,"DUF957"),(75,"UDP-galactose"),(78,"waaZ"),(53,"yfjQ")]      # (column, label prefix)
cm=AutoModelForCausalLM.from_pretrained("macwiatrak/bacformer-causal-complete-genomes",
     trust_remote_code=True).to(torch.bfloat16).eval().to(dev)
W=json.load(open("pgb/window.json")); EMB=np.load("pgb/window_emb.npy").astype(np.float32)
P=json.load(open("pgb/graph_pgb.json")); node={int(n["id"][1:]):n for n in P["nodes"]}; G=W["genomes"]
C={k:v for k,v in np.load("pgb/model_calls.npz").items()}
FI={int(f):i for i,f in enumerate(sorted(map(int,W["families"])))}; NF=len(FI)
fam_at=np.array([FI[G[int(g)]["genes"][int(l)-G[int(g)]["start"]][1]] for g,l in zip(C["genome"],C["locus"])])
def decoder(mask):
    A=sparse.csr_matrix((C["top_p"][mask].ravel(),(np.repeat(fam_at[mask],C["top_f"].shape[1]),C["top_f"][mask].ravel())),shape=(NF,50001))
    return A,np.asarray(A.sum(0)).ravel()
DEC={h:decoder(C["genome"]%2!=h) for h in (0,1)}
succ=collections.defaultdict(collections.Counter)
for g in G:
    L=g["genes"]
    for k in range(len(L)-1): succ[L[k][1]][L[k+1][1]]+=1
def fork_of(col,prefix):
    c=[f for f,n in node.items() if n["locus"]==col and n["label"].startswith(prefix)]
    return max(c,key=lambda f:node[f]["n"])
def weights(br,h):
    A,t=DEC[h]; rows=[A.getrow(FI[b]) for b in br]
    cl=sorted({int(c) for r in rows for c in r.indices}); ci={c:i for i,c in enumerate(cl)}
    M=np.zeros((max(len(cl),1),len(br)),np.float32)
    for j,r in enumerate(rows):
        for c,v in zip(r.indices,r.data): M[ci[int(c)],j]=v/t[c]
    return torch.tensor(cl or [0],device=dev,dtype=torch.long),torch.tensor(M,device=dev)
@torch.no_grad()
def branch_q(prefixes,cl,M):
    """prefixes: list of equal-length pid lists -> (n, branches) normalised branch probabilities"""
    n=len(prefixes[0]); x=torch.tensor(EMB[np.array(prefixes)],device=dev).to(torch.bfloat16)
    x=torch.cat([torch.zeros((len(prefixes),1,480),device=dev,dtype=x.dtype),x],1)
    lg=cm(protein_embeddings=x,special_tokens_mask=torch.tensor([[CLS]+[PROT]*n]*len(prefixes),device=dev),
          token_type_ids=torch.zeros((len(prefixes),n+1),dtype=torch.long,device=dev),return_dict=True).logits[:,-1].float()
    s=(torch.softmax(lg,-1)[:,cl][:,:,None]*M[None]).sum(1).cpu().numpy()+1e-12
    return s/s.sum(1,keepdims=True)
lab=lambda f: node[f]["label"] if f in node else str(f)

out=[]
for col,prefix in TARGETS:
    f=fork_of(col,prefix); br=[b for b,c in succ[f].most_common(6) if c>=20]; Wh={h:weights(br,h) for h in (0,1)}
    eff=collections.defaultdict(list); flips=collections.Counter(); seen=collections.Counter(); sites=[]
    for gi,g in enumerate(G):
        L=g["genes"]
        for k in range(len(L)-1):
            if L[k][1]!=f or L[k+1][1] not in br: continue
            t=br.index(L[k+1][1]); cl,M=Wh[gi%2]; pids=[y[3] for y in L[:k+1]]
            q0=branch_q([pids],cl,M)[0]; js=list(range(max(0,k-UP),k))
            if not js: continue
            qd=branch_q([pids[:j]+pids[j+1:] for j in js],cl,M)
            for j,q in zip(js,qd):
                fam=L[j][1]; eff[fam].append(float(np.log2(q0[t])-np.log2(q[t])))
                flips[fam]+=int(q.argmax()!=q0.argmax()); seen[fam]+=1
            sites.append((gi,k,t,q0))
    ranked=sorted(eff,key=lambda fam:-np.mean(eff[fam]))
    top=[dict(family=lab(fam),fam=node[fam]["fam"] if fam in node else "",genomes=seen[fam],
              mean_drop_bits=round(float(np.mean(eff[fam])),3),flip_rate=round(flips[fam]/seen[fam],3))
         for fam in ranked if seen[fam]>=10][:6]
    rec=dict(fork=lab(f),column=col,branches=[[lab(b),succ[f][b]] for b in br],genomes=len(sites),
             model_acc=round(float(np.mean([q.argmax()==t for _,_,t,q in sites])),3),determinants=top)
    # allele swap for the strongest determinant carried by genomes of both branches
    for d in [fam for fam in ranked if seen[fam]>=10][:3]:
        by=collections.defaultdict(collections.Counter)
        for gi,k,t,q in sites:
            for y in G[gi]["genes"][max(0,k-UP):k]:
                if y[1]==d: by[t][y[3]]+=1
        if len(by)<2 or 0 not in by: continue
        maj=by[0].most_common(1)[0][0]; tried=fl=0
        for gi,k,t,q in sites:
            if t==0: continue
            L=G[gi]["genes"]; pids=[y[3] for y in L[:k+1]]
            js=[j for j in range(max(0,k-UP),k) if L[j][1]==d and pids[j]!=maj]
            if not js: continue
            cl,M=Wh[gi%2]; sw=list(pids); sw[js[-1]]=maj
            q=branch_q([sw],cl,M)[0]; tried+=1; fl+=int(q.argmax()==0)
        if tried:
            rec["allele_swap"]=dict(family=lab(d),genomes=tried,to_majority_branch=round(fl/tried,3),
                                    note=f"minority-branch genomes given the majority-branch allele of {lab(d)}")
            break
    out.append(rec)
    print(f"\n{rec['fork']} (column {col}) {rec['genomes']} genomes, model right {rec['model_acc']:.0%} | branches "
          +" / ".join(f"{a}:{b}" for a,b in rec["branches"]))
    for d in top: print(f"   delete {d['family'][:34]:34s} in {d['genomes']:4d} genomes: -{d['mean_drop_bits']:.2f} bits on the true branch, flips {d['flip_rate']:.0%}")
    if "allele_swap" in rec: print(f"   allele swap {rec['allele_swap']['family']}: {rec['allele_swap']['to_majority_branch']:.0%} of {rec['allele_swap']['genomes']} minority genomes switch to the majority branch")
json.dump(out,open("pgb/fork_ablation.json","w"),indent=1)
