"""ESM-2 embeddings of a species' distinct proteins, then Bacformer's call at every gene of every genome.

Input: out/<id>/ from extract.py (genes.npz, meta.json, proteins.txt). Output, in out/<id>/:
  emb.u8, emb_range.npy    mean-pooled ESM-2 t12 35M embedding of each distinct protein, as in the
                           E. coli run, stored in 8 bits with one affine scale per dimension (the
                           species' range of each dimension): a few dimensions are far larger than the
                           rest, which a scale per protein would crush (on E. coli: top-1 unchanged in
                           99.8 % of calls, against 97.6 % with a scale per protein); --check measures it
  calls_f.u16, calls_p.f16 the model's 15 most probable clusters and their softmax probabilities, in
                           the call for each gene, made from everything before it in its genome, from
                           the full-precision embeddings (the 8-bit ones serve later, live queries)
  calls_ent.f16            the entropy of that call, in bits
  gpu.json, gpu.ok         timings and counts
A genome is read as Bacformer's own preprocessing reads it (bacformer.pp.protein_embeddings_to_inputs):
[CLS] contig 1 [SEP] contig 2 [SEP] ... [END], token type = contig index, SEP being the CLS id as in that
function. A genome longer than the model's 6,000 positions is read in windows of 6,000 every 3,000,
each call keeping at least 3,000 positions of context. proteins.txt is removed once embedded.

    python gpu.py out/11587 [--check 20]
    python gpu.py --loop                 # a worker: takes extracted species until none is left
"""
import os, json, time, glob, argparse, warnings
import numpy as np, torch
warnings.simplefilter("ignore")
ROOT = os.environ.get("PGG_ROOT", ".")
dev = "cuda:0"
ESM = "facebook/esm2_t12_35M_UR50D"; BAC = "macwiatrak/bacformer-causal-complete-genomes"
CLS, PROT, END = 2, 4, 5; SEP = CLS                     # the SEP default of bacformer's preprocessing
W, S, K, SHARD = 6000, 3000, 15, 50000
BUDGET = int(os.environ.get("ESM_BUDGET", 120000))       # residues per ESM-2 batch (48 GB card)
_esm = _tok = _bac = None


def _esm_sdpa(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
              encoder_attention_mask=None, past_key_value=None, output_attentions=False):
    """ESM-2's self-attention, the same steps (query scaled, rotary positions, additive mask, softmax), in
    PyTorch's fused scaled_dot_product_attention: transformers 4.53 has no fused path for ESM"""
    q = self.transpose_for_scores(self.query(hidden_states)) * self.attention_head_size ** -0.5
    k = self.transpose_for_scores(self.key(hidden_states)); v = self.transpose_for_scores(self.value(hidden_states))
    q, k = self.rotary_embeddings(q, k)
    c = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, scale=1.0)
    return (c.permute(0, 2, 1, 3).reshape(c.size(0), c.size(2), self.all_head_size),)


def models():
    global _esm, _tok, _bac
    if _bac is None:
        from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
        import transformers.models.esm.modeling_esm as me
        me.EsmSelfAttention.forward = _esm_sdpa
        _tok = AutoTokenizer.from_pretrained(ESM)
        _esm = AutoModel.from_pretrained(ESM).to(torch.float16).eval().to(dev)
        _bac = AutoModelForCausalLM.from_pretrained(BAC, trust_remote_code=True).to(torch.bfloat16).eval().to(dev)
    return _tok, _esm, _bac


_lut = None
def encode(prots):
    """ESM-2 token ids of each protein (<= 1,022 residues), one table lookup per residue: the same ids as
    the ESM tokenizer (one token per letter, <unk> otherwise), without its slow Python loop"""
    global _lut
    if _lut is None:
        tok = models()[0]; v = tok.get_vocab(); _lut = np.full(256, v["<unk>"], np.int64)
        for k, i in v.items():
            if len(k) == 1: _lut[ord(k)] = i
    return [_lut[np.frombuffer(p[:1022].encode(), np.uint8)] for p in prots]


@torch.no_grad()
def esm_embed(ids):
    """mean over residues of ESM-2's last layer, CLS and EOS left out, as in the E. coli run (float16 model)"""
    tok, esm, _ = models(); lens = np.array([len(x) for x in ids]); L = lens.max() + 2
    a = np.full((len(ids), L), tok.pad_token_id, np.int64); a[:, 0] = tok.cls_token_id
    for r, x in enumerate(ids): a[r, 1:len(x) + 1] = x; a[r, len(x) + 1] = tok.eos_token_id
    ar = np.arange(L); am = torch.tensor(ar[None] < (lens + 2)[:, None], device=dev)
    h = esm(input_ids=torch.tensor(a, device=dev), attention_mask=am.long()).last_hidden_state
    pm = torch.tensor((ar[None] >= 1) & (ar[None] <= lens[:, None]), device=dev).unsqueeze(-1).to(h.dtype)
    return ((h * pm).sum(1) / pm.sum(1).clamp(min=1)).float().cpu().numpy()


def embed(d, prots):
    """ESM-2 embeddings in float16 to a temporary file, resumable by shards of SHARD proteins (shortest
    first, in batches of BUDGET residues); then 8 bits with one affine scale per dimension"""
    N = len(prots); tmp = f"{d}/emb.tmp.f16"
    T = np.memmap(tmp, dtype=np.float16, mode="r+" if os.path.exists(tmp) else "w+", shape=(N, 480))
    done_p = f"{d}/emb.done.json"; done = set(json.load(open(done_p))) if os.path.exists(done_p) else set()
    order = np.argsort([len(p) for p in prots], kind="stable")
    for s0 in range(0, N, SHARD):
        if s0 in done: continue
        idx = order[s0:s0 + SHARD]; ids = encode([prots[k] for k in idx]); i = 0
        while i < len(idx):
            j = i
            while j < len(idx) and (j - i + 1) * (len(ids[j]) + 2) <= BUDGET: j += 1
            j = max(j, i + 1); T[idx[i:j]] = esm_embed(ids[i:j]).astype(np.float16); i = j
        T.flush(); done.add(s0); json.dump(sorted(done), open(done_p, "w"))
    lo = np.full(480, np.inf, np.float32); hi = np.full(480, -np.inf, np.float32)
    for s0 in range(0, N, 200000):
        x = np.asarray(T[s0:s0 + 200000], np.float32); lo = np.minimum(lo, x.min(0)); hi = np.maximum(hi, x.max(0))
    sc = np.maximum(hi - lo, 1e-6) / 255.0; np.save(f"{d}/emb_range.npy", np.stack([lo, hi]))
    Q = np.memmap(f"{d}/emb.u8", dtype=np.uint8, mode="w+", shape=(N, 480))
    for s0 in range(0, N, 200000):
        x = np.asarray(T[s0:s0 + 200000], np.float32); Q[s0:s0 + 200000] = np.clip(np.rint((x - lo) / sc), 0, 255).astype(np.uint8)
    Q.flush()
    return Q, (lambda p: Q[p].astype(np.float32) * sc + lo), T


def genome_tokens(G, g):
    """token arrays of genome g: special ids, protein index (-1 for special), token type, gene index (-1)"""
    c0, c1 = np.searchsorted(G["cgenome"], g), np.searchsorted(G["cgenome"], g, "right")
    sp, pr, tt, gi = [CLS], [-1], [0], [-1]
    for c in range(c0, c1):
        a, b = G["cstart"][c], G["cstart"][c + 1]; ci = min(c - c0, 999)
        sp += [PROT] * (b - a) + [SEP]; pr += list(G["pid"][a:b]) + [-1]; tt += [ci] * (b - a + 1); gi += list(range(a, b)) + [-1]
    sp.append(END); pr.append(-1); tt.append(tt[-1]); gi.append(-1)
    return np.array(sp), np.array(pr), np.array(tt), np.array(gi)


@torch.no_grad()
def calls_for(emb_of, sp, pr, tt, gi):
    """top-K clusters, their probabilities and the entropy of the call on each protein token"""
    _, _, bac = models(); n = len(sp); out = {}
    starts = [0] if n <= W else list(range(0, n - W, S)) + [n - W]
    for s in starts:
        if s == 0: sl = np.arange(0, min(W, n)); keep_from = 1
        else: sl = np.r_[0, np.arange(s + 1, s + W)]; keep_from = max(S, 1)   # CLS, then the window
        x = torch.zeros((1, len(sl), 480), dtype=torch.float32)
        m = pr[sl] >= 0; x[0, torch.tensor(np.flatnonzero(m))] = torch.tensor(emb_of(pr[sl][m]))
        lg = bac(protein_embeddings=x.to(dev, torch.bfloat16), special_tokens_mask=torch.tensor(sp[sl][None], device=dev),
                 token_type_ids=torch.tensor(tt[sl][None], device=dev), return_dict=True).logits[0]
        for p in range(keep_from, len(sl), 1024):                 # the call on position p comes from position p - 1
            q = np.arange(p, min(p + 1024, len(sl))); q = q[pr[sl][q] >= 0]
            q = q[[sl[x_] not in out for x_ in q]] if s else q
            if not len(q): continue
            pb = torch.softmax(lg[torch.tensor(q - 1, device=dev)].float(), -1)
            tp, tf = pb.topk(K, 1); ent = -(pb.clamp(min=1e-12) * pb.clamp(min=1e-12).log2()).sum(1)
            tf, tp, ent = tf.cpu().numpy(), tp.cpu().numpy(), ent.cpu().numpy()
            for r, x_ in enumerate(q): out[int(sl[x_])] = (tf[r], tp[r], ent[r])
        del lg
    tok_idx = np.array(sorted(out)); g = gi[tok_idx]
    return g, np.stack([out[t][0] for t in tok_idx]), np.stack([out[t][1] for t in tok_idx]), np.array([out[t][2] for t in tok_idx])


def process(d, check=0):
    t0 = time.time(); G = dict(np.load(f"{d}/genes.npz")); M = json.load(open(f"{d}/meta.json"))
    prots = open(f"{d}/proteins.txt").read().split("\n")[:M["n_proteins"]]
    Q, deq, T = embed(d, prots); t1 = time.time()
    n = M["n_genes"]; mode = "w+"
    CF = np.memmap(f"{d}/calls_f.u16", dtype=np.uint16, mode=mode, shape=(n, K))
    CP = np.memmap(f"{d}/calls_p.f16", dtype=np.float16, mode=mode, shape=(n, K))
    CE = np.memmap(f"{d}/calls_ent.f16", dtype=np.float16, mode=mode, shape=(n,))
    report = {}
    if check:                                      # what 8-bit storage changes: the same genomes from the float16 embeddings
        agree = []; dlp = []
        for g in range(min(check, M["n_genomes"])):
            sp, pr, tt, gi = genome_tokens(G, g)
            _, f1, p1, _ = calls_for(lambda p: np.asarray(T[p], np.float32), sp, pr, tt, gi)
            _, f2, p2, _ = calls_for(deq, sp, pr, tt, gi)
            agree.append(float((f1[:, 0] == f2[:, 0]).mean())); dlp.append(float(np.abs(np.log(p1[:, 0].astype(np.float64) + 1e-9) - np.log(p2[:, 0].astype(np.float64) + 1e-9)).mean()))
        report["check"] = dict(genomes=len(agree), top1_same=round(float(np.mean(agree)), 5), mean_abs_dlog_p_top1=round(float(np.mean(dlp)), 5))
        print("check 8-bit embeddings:", report["check"], flush=True)
    for g in range(M["n_genomes"]):
        sp, pr, tt, gi = genome_tokens(G, g)
        gg, f, p, e = calls_for(lambda q: np.asarray(T[q], np.float32), sp, pr, tt, gi)
        CF[gg] = f.astype(np.uint16); CP[gg] = p.astype(np.float16); CE[gg] = e.astype(np.float16)
        if g % 200 == 0: print(f"  {os.path.basename(d)}: genome {g + 1}/{M['n_genomes']}, {time.time() - t1:.0f} s", flush=True)
    for a in (CF, CP, CE): a.flush()
    report.update(genes=n, proteins=len(prots), genomes=M["n_genomes"], embed_s=round(t1 - t0, 1), calls_s=round(time.time() - t1, 1))
    json.dump(report, open(f"{d}/gpu.json", "w"))
    del T; os.remove(f"{d}/emb.tmp.f16"); os.remove(f"{d}/proteins.txt"); os.remove(f"{d}/emb.done.json")
    open(f"{d}/gpu.ok", "w").write(str(time.time()))
    return report


def loop():
    out = os.path.join(ROOT, "out"); me = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    while True:
        todo = sorted(glob.glob(os.path.join(out, "*", "extract.ok")), key=os.path.getmtime)
        todo = [t for t in todo if not os.path.exists(os.path.join(os.path.dirname(t), "gpu.ok"))]
        if not todo:
            if os.path.exists(os.path.join(ROOT, "state", "download.finished")) and not glob.glob(os.path.join(ROOT, "raw", "*")): return
            time.sleep(60); continue
        ok = todo[0]; d = os.path.dirname(ok); claim = os.path.join(d, f"gpu.claim.{me}")
        try: os.rename(ok, claim)
        except OSError: continue
        try:
            r = process(d); os.rename(claim, ok)
            print(f"{os.path.basename(d)}: {r['genomes']} genomes, {r['genes']} genes, {r['proteins']} proteins, embed {r['embed_s']} s, calls {r['calls_s']} s", flush=True)
        except Exception as e:
            print(f"{os.path.basename(d)}: FAILED {type(e).__name__}: {e}", flush=True); os.rename(claim, os.path.join(d, "gpu.failed"))
            torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dir", nargs="?"); ap.add_argument("--loop", action="store_true")
    ap.add_argument("--check", type=int, default=0)
    a = ap.parse_args()
    if a.loop: loop()
    else: print(json.dumps(process(a.dir, a.check)))
