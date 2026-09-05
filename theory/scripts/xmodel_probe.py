"""Second-model replication with no ground truth. Usage: python xmodel_probe.py <model_path> <tag> <layers a-b>
(1) recitation scan of the 9216 dolmino windows (prefix 64 / suffix 48, greedy) -> recited (acc == 1) and never (acc < 0.75);
(2) coupled covariances (1152 dolmino sequences, sampled labels) for gate/up of the given layers -> eigenbases, spectrum;
(3) per-direction activation and margin-gradient energies, recited vs never -> half-whitening exponents, product-law coupling ratio;
(4) two-halves cosine of the recited set's bulk-projected margin gradient, cosine with never;
(5) edits: bulk deletion (alpha = 1) vs matched noise; head-quintile block removal vs noise -> recited strict, never loss, Pile loss.
Assumes the OLMo-2 tokenizer (token ids reused); re-tokenises through text otherwise."""
import sys, math, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
S = "/tmp/claude-1002/-home-guillaume-memorization-kfac/a39940c3-dcc5-4714-8566-58fa7889e391/scratchpad"; OLMO = "/home/guillaume/.cache/huggingface/hub/models--allenai--OLMo-2-1124-7B/snapshots/7df9a82518afdecae4e8c026b27adccc8c1f0032"
path, tag = sys.argv[1], sys.argv[2]; lo, hi = map(int, sys.argv[3].split("-")); LAYERS = tuple(range(lo, hi + 1)); PROJ = ("gate_proj", "up_proj"); dev = "cuda"; L = 112; PRE = 64; FLAT = 0.6
torch.backends.cuda.matmul.allow_tf32 = False
tok = AutoTokenizer.from_pretrained(path); model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
gen = torch.load(f"{S}/dolmino_seqs_2304.pt"); pile = torch.load(f"{S}/pile_seqs_2304.pt")[:384]
otok = AutoTokenizer.from_pretrained(OLMO)
if tok.get_vocab() != otok.get_vocab():
    print("re-tokenising through text", flush=True)
    def retok(ids, T):
        out = []
        for row in ids:
            t = tok(otok.decode(row.tolist()), add_special_tokens=False)["input_ids"][:T]; out.append(t + [tok.pad_token_id or 0] * (T - len(t)))
        return torch.tensor(out)
    gen = retok(gen, 512); pile = retok(pile, 512)
CORPUS = sys.argv[4] if len(sys.argv) > 4 else "dolmino"
scan_src = pile_full if False else None
if CORPUS == "pile":
    pile_all = torch.load(f"{S}/pile_seqs_2304.pt")
    if tok.get_vocab() != otok.get_vocab(): pile_all = retok(pile_all, 512)
    windows = pile_all[:, :(512 // L) * L].reshape(-1, L); print("scanning Pile windows for recitation", flush=True)
else: windows = gen[:, :(512 // L) * L].reshape(-1, L)
def scan():
    acc = []
    with torch.no_grad():
        for s in range(0, len(windows), 64):
            x = windows[s:s + 64].to(dev); out = model.generate(x[:, :PRE], max_new_tokens=L - PRE, do_sample=False, pad_token_id=tok.pad_token_id or 0)
            acc.append((out[:, PRE:L] == x[:, PRE:]).float().mean(1).cpu())
    return torch.cat(acc)
acc = scan(); recited = (acc == 1).nonzero()[:, 0]; never = (acc < 0.75).nonzero()[:, 0]
if len(recited) < 40: recited = (acc >= 0.95).nonzero()[:, 0]; print("relaxed recitation threshold to 0.95", flush=True)
g = torch.Generator().manual_seed(0); never = never[torch.randperm(len(never), generator=g)[:1200]]
print(f"[{tag}] recitation scan: mean suffix acc {acc.mean():.3f}, recited (acc==1) {int((acc == 1).sum())}, >=0.95 {int((acc >= 0.95).sum())}, using {len(recited)} recited and {len(never)} never", flush=True)
wm = torch.zeros(L - 1, dtype=torch.bool); wm[PRE - 1:] = True
P = {"recited": (windows[recited], wm[None].expand(len(recited), -1)), "never": (windows[never], wm[None].expand(len(never), -1)), "pile": (pile, torch.ones(len(pile), pile.shape[1] - 1, dtype=torch.bool))}
# (2) coupled covariances
mods = {(l, p): getattr(model.model.layers[l].mlp, p) for l in LAYERS for p in PROJ}; W0 = {k: m.weight.detach().float().clone() for k, m in mods.items()}
cap = {}
def hook(k):
    def h(mod, inp, out):
        cap[(k, "a")] = inp[0].detach(); out.requires_grad_(True); out.register_hook(lambda gg: cap.__setitem__((k, "g"), gg.detach()))
    return h
hs = [m.register_forward_hook(hook(k)) for k, m in mods.items()]
covA = {k: torch.zeros(m.weight.shape[1], m.weight.shape[1], device=dev) for k, m in mods.items()}; covG = {k: torch.zeros(m.weight.shape[0], m.weight.shape[0], device=dev) for k, m in mods.items()}
torch.manual_seed(0)
for s in range(0, 1152, 4):
    x = gen[s:s + 4].to(dev); logits = model(input_ids=x).logits[:, :-1].float()
    with torch.no_grad(): y = torch.distributions.Categorical(logits=logits).sample()
    torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum").backward(); del logits
    with torch.no_grad():
        for k in mods:
            a = cap[(k, "a")].float()[:, :-1].reshape(-1, covA[k].shape[0]); gg = cap[(k, "g")].float()[:, :-1].reshape(-1, covG[k].shape[0])
            covA[k] += (a * (gg ** 2).sum(-1, keepdim=True)).T @ a; covG[k] += (gg * (a ** 2).sum(-1, keepdim=True)).T @ gg
    cap.clear()
Q = {}
for k in mods:
    evA, Qa = torch.linalg.eigh(covA.pop(k)); evG, Qg = torch.linalg.eigh(covG.pop(k)); Q[k] = (Qg, Qa, evG.clamp_min(1e-30), evA.clamp_min(1e-30))
    q = lambda v, x: float(torch.quantile(v.log10(), x)); print(f"[{tag}] {k}: G spectrum log10 quantiles (1,10,50,90,99%) {[round(q(evG.clamp_min(1e-30), x), 2) for x in (0.01, 0.1, 0.5, 0.9, 0.99)]} | p90/p10 {float(evG.quantile(0.9) / evG.quantile(0.1)):.2f} | head >10x median {float((evG > 10 * evG.median()).float().mean()):.4f} || A: {[round(q(evA.clamp_min(1e-30), x), 2) for x in (0.01, 0.1, 0.5, 0.9, 0.99)]}", flush=True)
torch.cuda.empty_cache()
# (3) energies with margin gradients
def energies(seqs, mask):
    acc_ = {k: [torch.zeros(Q[k][0].shape[1], device=dev), torch.zeros(Q[k][1].shape[1], device=dev)] for k in mods}; ntok = 0
    for s in range(0, len(seqs), 32):
        x = seqs[s:s + 32].to(dev); m = mask[s:s + 32].to(dev).float(); logits = model(input_ids=x).logits[:, :-1].float(); y = x[:, 1:]
        tgt = logits.gather(-1, y[..., None])[..., 0]; other = logits.scatter(-1, y[..., None], -1e9).max(-1).values; ((tgt - other) * m).sum().backward(); del logits
        with torch.no_grad():
            for k in mods:
                a = cap[(k, "a")].float()[:, :-1]; gg = cap[(k, "g")].float()[:, :-1]
                acc_[k][0] += (((gg @ Q[k][0]) ** 2) * m[..., None]).sum((0, 1)); acc_[k][1] += (((a @ Q[k][1]) ** 2) * m[..., None]).sum((0, 1))
        ntok += int(m.sum()); cap.clear()
    return {k: (acc_[k][0] / ntok, acc_[k][1] / ntok) for k in mods}
def fit(x, y):
    x, y = np.log(np.clip(x, 1e-30, None)), np.log(np.clip(y, 1e-30, None)); X = np.vstack([np.ones_like(x), x]).T; b = np.linalg.lstsq(X, y, rcond=None)[0]; r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum(); return float(b[1]), float(r2)
Er, En = energies(*P["recited"]), energies(*P["never"])
ratios = []
for k in mods:
    sG, sA = fit(En[k][0].cpu().numpy(), (Er[k][0] / En[k][0]).cpu().numpy()), fit(En[k][1].cpu().numpy(), (Er[k][1] / En[k][1]).cpu().numpy())
    nb_g, nb_a = int(FLAT * len(Q[k][2])), int(FLAT * len(Q[k][3])); bulkG = float(Er[k][0][:nb_g].sum() / En[k][0][:nb_g].sum()); bulkA = float(Er[k][1][:nb_a].sum() / En[k][1][:nb_a].sum()); ratios.append(bulkG * bulkA)
    print(f"[{tag}] {k}: half-whitening exponent G {sG[0]:+.3f} (R2 {sG[1]:.2f}), A {sA[0]:+.3f} (R2 {sA[1]:.2f}) | bulk energy recited/never G {bulkG:.2f} A {bulkA:.2f} | bulk share of energy: never G {float(En[k][0][:nb_g].sum() / En[k][0].sum()):.2f} rec {float(Er[k][0][:nb_g].sum() / Er[k][0].sum()):.2f}, A never {float(En[k][1][:nb_a].sum() / En[k][1].sum()):.2f} rec {float(Er[k][1][:nb_a].sum() / Er[k][1].sum()):.2f}", flush=True)
print(f"[{tag}] product-law coupling ratio recited/never (mean over matrices): {np.mean(ratios):.2f}", flush=True)
# (4) two-halves cosine
for h in hs: h.remove()
def margin_grad(seqs, mask):
    for m in mods.values(): m.weight.requires_grad_(True)
    g_ = {k: torch.zeros_like(W0[k]) for k in mods}
    for s in range(0, len(seqs), 16):
        x = seqs[s:s + 16].to(dev); mk = mask[s:s + 16].to(dev).float(); logits = model(input_ids=x).logits[:, :-1].float(); y = x[:, 1:]
        tgt = logits.gather(-1, y[..., None])[..., 0]; other = logits.scatter(-1, y[..., None], -1e9).max(-1).values; ((tgt - other) * mk).sum().backward()
        with torch.no_grad():
            for k, m in mods.items(): g_[k] += m.weight.grad.float(); m.weight.grad = None
        del logits
    for m in mods.values(): m.weight.requires_grad_(False)
    return g_
sel = lambda k: (Q[k][0][:, : int(FLAT * Q[k][0].shape[1])], Q[k][1][:, : int(FLAT * Q[k][1].shape[1])])
def proj(gr): return {k: sel(k)[0] @ (sel(k)[0].T @ gr[k] @ sel(k)[1]) @ sel(k)[1].T for k in mods}
def cos(a, b): return float(sum((a[k] * b[k]).sum() for k in mods) / (sum((a[k] ** 2).sum() for k in mods).sqrt() * sum((b[k] ** 2).sum() for k in mods).sqrt()))
h1, h2 = recited[: len(recited) // 2], recited[len(recited) // 2:]
GA, GB, GN = margin_grad(windows[h1], wm[None].expand(len(h1), -1)), margin_grad(windows[h2], wm[None].expand(len(h2), -1)), margin_grad(windows[never[:600]], wm[None].expand(600, -1))
print(f"[{tag}] cosines (full | bulk): recited halves {cos(GA, GB):+.3f} | {cos(proj(GA), proj(GB)):+.3f}; recited vs never {cos(GA, GN):+.3f} | {cos(proj(GA), proj(GN)):+.3f}", flush=True)
# (5) edits
def evaluate():
    out = {}
    with torch.no_grad():
        for n, (seqs, mask) in P.items():
            tot, cnt, ok = 0.0, 0, 0; bs = 16 if n == "pile" else 64
            for s in range(0, len(seqs), bs):
                x = seqs[s:s + bs].to(dev); mk = mask[s:s + bs].to(dev); logits = model(input_ids=x).logits[:, :-1].float(); y = x[:, 1:]
                l = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none").view(y.shape)
                tot += float((l * mk).sum()); cnt += int(mk.sum()); ok += int((((logits.argmax(-1) == y) & mk).sum(1) == mk.sum(1)).sum()); del logits
            out[n] = dict(loss=tot / cnt, strict=ok / len(seqs))
    return out
base = evaluate(); print(f"[{tag}] base:", {n: (round(v["loss"], 3), round(v["strict"], 3)) for n, v in base.items()}, flush=True)
def apply(kind, band):
    norms = []
    with torch.no_grad():
        for k, m in mods.items():
            Qg, Qa = Q[k][0], Q[k][1]; O, I = Qg.shape[1], Qa.shape[1]
            gs, as_ = (slice(0, int(FLAT * O)), slice(0, int(FLAT * I))) if band == "bulk" else (slice(int(0.8 * O), O), slice(int(0.8 * I), I))
            Gs, As = Qg[:, gs], Qa[:, as_]; Wt = Gs.T @ W0[k] @ As; nrm = float(Wt.norm()) * (1.0 if band == "bulk" else 1.0)
            if kind == "remove": N = -Wt
            else: gg = torch.Generator(device=dev).manual_seed(hash(str(k)) % 997); N = torch.randn(Wt.shape, device=dev, generator=gg); N = N * (nrm / N.norm())
            norms.append(nrm); m.weight.copy_((W0[k] + Gs @ N @ As.T).to(torch.bfloat16))
    return norms
print(f"[{tag}] edit (block = flattest 60% x 60% 'bulk' or top 20% x 20% 'head', full removal vs matched noise): recited strict | never d | pile d")
for band in ("bulk", "head"):
    for kind in ("remove", "noise"):
        norms = apply(kind, band); r = evaluate()
        print(f"[{tag}]   {band:4s} {kind:6s} (norm {np.mean(norms):5.1f}): {r['recited']['strict']:.3f} | +{r['never']['loss'] - base['never']['loss']:.4f} | +{r['pile']['loss'] - base['pile']['loss']:.4f}", flush=True)
with torch.no_grad():
    for k, m in mods.items(): m.weight.copy_(W0[k].to(torch.bfloat16))
torch.save({"acc": acc, "recited": recited}, f"{S}/xmodel_{tag}_scan.pt"); print("done")
