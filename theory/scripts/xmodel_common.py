"""Shared machinery for the cross-model theory tests (any HF causal LM with Llama-style MLPs).
Populations from a recitation scan of the 9216 dolmino windows (no ground truth); coupled K-FAC bases of the
chosen layers' gate/up matrices from sampled-label gradients on 1152 dolmino sequences; block edits; evaluation."""
import os, math, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
S = "/tmp/claude-1002/-home-guillaume-memorization-kfac/a39940c3-dcc5-4714-8566-58fa7889e391/scratchpad"
OLMO = "/home/guillaume/.cache/huggingface/hub/models--allenai--OLMo-2-1124-7B/snapshots/7df9a82518afdecae4e8c026b27adccc8c1f0032"
dev = "cuda"; L = 112; PRE = 64; FLAT = 0.6; PROJ = ("gate_proj", "up_proj")
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

def load(path, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(path); model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    return tok, model

def data(tok):
    gen = torch.load(f"{S}/dolmino_seqs_2304.pt"); pile = torch.load(f"{S}/pile_seqs_2304.pt")
    otok = AutoTokenizer.from_pretrained(OLMO)
    if tok.get_vocab() != otok.get_vocab():
        print("re-tokenising through text", flush=True)
        def retok(ids, T):
            out = []
            for row in ids:
                t = tok(otok.decode(row.tolist()), add_special_tokens=False)["input_ids"][:T]; out.append(t + [tok.pad_token_id or 0] * (T - len(t)))
            return torch.tensor(out)
        gen, pile = retok(gen, 512), retok(pile, 512)
    windows = gen[:, :(512 // L) * L].reshape(-1, L)
    return gen, pile, windows

def scan(model, tok, windows, tag):
    f = f"{S}/xmodel_{tag}_scan.pt"
    if os.path.exists(f): return torch.load(f)["acc"]
    acc = []
    with torch.no_grad():
        for s in range(0, len(windows), 64):
            x = windows[s:s + 64].to(dev); out = model.generate(x[:, :PRE], max_new_tokens=L - PRE, do_sample=False, pad_token_id=tok.pad_token_id or 0)
            acc.append((out[:, PRE:L] == x[:, PRE:]).float().mean(1).cpu())
    acc = torch.cat(acc); torch.save({"acc": acc, "recited": (acc == 1).nonzero()[:, 0]}, f); return acc

def populations(acc, n_never=1200):
    recited = (acc == 1).nonzero()[:, 0]
    if len(recited) < 40: recited = (acc >= 0.95).nonzero()[:, 0]; print("relaxed recitation threshold to 0.95", flush=True)
    never = (acc < 0.75).nonzero()[:, 0]; g = torch.Generator().manual_seed(0); never = never[torch.randperm(len(never), generator=g)[:n_never]]
    wm = torch.zeros(L - 1, dtype=torch.bool); wm[PRE - 1:] = True
    return recited, never, wm

def modules(model, layers):
    return {(l, p): getattr(model.model.layers[l].mlp, p) for l in layers for p in PROJ}

def coupled_bases(model, mods, gen, tag, n=1152, bs=4):
    """Coupled K-FAC factors with sampled labels; eigenbases sorted by ascending eigenvalue (bulk first)."""
    f = f"{S}/xmodel_{tag}_bases.pt"
    if os.path.exists(f):
        B = torch.load(f); return {k: tuple(t.to(dev) for t in v) for k, v in B.items()}
    cap = {}
    def hook(k):
        def h(mod, inp, out):
            cap[(k, "a")] = inp[0].detach(); out.requires_grad_(True); out.register_hook(lambda gg: cap.__setitem__((k, "g"), gg.detach()))
        return h
    hs = [m.register_forward_hook(hook(k)) for k, m in mods.items()]
    covA = {k: torch.zeros(m.weight.shape[1], m.weight.shape[1], device=dev) for k, m in mods.items()}
    covG = {k: torch.zeros(m.weight.shape[0], m.weight.shape[0], device=dev) for k, m in mods.items()}
    torch.manual_seed(0)
    for s in range(0, n, bs):
        x = gen[s:s + bs].to(dev); logits = model(input_ids=x).logits[:, :-1].float()
        with torch.no_grad(): y = torch.distributions.Categorical(logits=logits).sample()
        torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum").backward(); del logits
        with torch.no_grad():
            for k in mods:
                a = cap[(k, "a")].float()[:, :-1].reshape(-1, covA[k].shape[0]); gg = cap[(k, "g")].float()[:, :-1].reshape(-1, covG[k].shape[0])
                covA[k] += (a * (gg ** 2).sum(-1, keepdim=True)).T @ a; covG[k] += (gg * (a ** 2).sum(-1, keepdim=True)).T @ gg
        cap.clear()
    for h in hs: h.remove()
    Q = {}
    for k in mods:
        eG, QG = torch.linalg.eigh(covG[k] / n); eA, QA = torch.linalg.eigh(covA[k] / n); Q[k] = (QG, QA, eG, eA)
        print(f"[bases] {k}: G p90/p10 {float(eG[int(.9 * len(eG))] / eG[int(.1 * len(eG))]):.2f}, A p90/p10 {float(eA[int(.9 * len(eA))] / eA[int(.1 * len(eA))]):.2f}", flush=True)
    torch.save({k: tuple(t.cpu() for t in v) for k, v in Q.items()}, f); del covA, covG
    return Q

def band_slices(Q, k, band):
    O, I = Q[k][0].shape[1], Q[k][1].shape[1]
    if band == "bulk": return slice(0, int(FLAT * O)), slice(0, int(FLAT * I))
    return slice(int(0.8 * O), O), slice(int(0.8 * I), I)

def block_delta(kind, band, Q, W0, alpha=1.0, seed=0, norm=None):
    """Full-matrix perturbation for removal (-alpha * block) or matched-norm noise in the block. Returns {k: dW}, per-matrix norms."""
    out, norms = {}, []
    for k in W0:
        gs, as_ = band_slices(Q, k, band); Gs, As = Q[k][0][:, gs], Q[k][1][:, as_]; Wt = Gs.T @ W0[k] @ As; nrm = float(Wt.norm()) if norm is None else norm
        if kind == "remove": N = -alpha * Wt
        else:
            g = torch.Generator(device=dev).manual_seed(1000 * seed + sum(map(ord, str(k)))); N = torch.randn(Wt.shape, device=dev, generator=g); N = N * (alpha * nrm / N.norm())
        out[k] = Gs @ N @ As.T; norms.append(float(N.norm()))
    return out, norms

def set_weights(mods, W0, delta=None):
    with torch.no_grad():
        for k, m in mods.items(): m.weight.copy_((W0[k] + (delta[k] if delta is not None else 0)).to(m.weight.dtype))

def token_stats(model, seqs, mask, bs=16, want_margins=False):
    """Per population: mean masked loss, strict recitation; optionally per-token margins (N, L-1) at masked positions."""
    tot, cnt, ok, M = 0.0, 0, 0, []
    with torch.no_grad():
        for s in range(0, len(seqs), bs):
            x = seqs[s:s + bs].to(dev); mk = mask[s:s + bs].to(dev); logits = model(input_ids=x).logits[:, :-1].float(); y = x[:, 1:]
            l = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none").view(y.shape)
            tot += float((l * mk).sum()); cnt += int(mk.sum()); ok += int((((logits.argmax(-1) == y) & mk).sum(1) == mk.sum(1)).sum())
            if want_margins:
                tgt = logits.gather(-1, y[..., None])[..., 0]; other = logits.scatter(-1, y[..., None], -1e9).max(-1).values; M.append((tgt - other).cpu())
            del logits
    r = dict(loss=tot / cnt, strict=ok / len(seqs))
    if want_margins: r["margins"] = torch.cat(M)
    return r
