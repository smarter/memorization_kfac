"""Causal test of minimum-interference storage. Usage: python imprint_experiment.py <optimizer: adam|sgd|ng|adam_full> <lr> [max_steps]
OLMo-2 1B. Memorize 96 never-recited dolmino windows (from sequences outside the reference pool) by training the gate/up
matrices of layers 9-13 (or all parameters for adam_full) on mixed batches (8 item windows + 4 reference sequences) until
the items are recited. Optimizers: AdamW, SGD with momentum, and natural gradient in the population K-FAC basis
(ng: dW = -lr Q [D / (lambda mu + damping)] P^T, the minimum-interference direction of Prop. 3). Measurements: imprint energy
profile per direction against the population eigenvalue (exponents), imprint bulk share; the items' activation profile at
the inputs of layers 11-13 before and after (bulk share, exponent vs ordinary; total energy); collateral on held-out
ordinary text and Pile; then edits of the final model in the band: bulk and head block removal of the imprint only, and
of the full weights, with items' recitation remaining and ordinary cost."""
import os, sys, math, time, numpy as np, torch
sys.path.insert(0, "/tmp/claude-1002/-home-guillaume-memorization-kfac/a39940c3-dcc5-4714-8566-58fa7889e391/scratchpad")
from xmodel_common import *
OPT, LR = sys.argv[1], float(sys.argv[2]); MAX_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 600
ITEM_W = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0; NI, NR = (int(v) for v in (sys.argv[5].split(",") if len(sys.argv) > 5 else "8,4".split(",")))
RUN = OPT + (f"_w{ITEM_W:g}" if len(sys.argv) > 4 else "")
P1B = "/home/guillaume/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B/snapshots"
import glob; P1B = glob.glob(P1B + "/*")[0]
TRAIN_LAYERS = (9, 10, 11, 12, 13); PROBE_LAYERS = (11, 12, 13); NITEM = 96; tag = "olmo1b_band"
tok, model = load(P1B, torch.float32); gen, pile, windows = data(tok); acc = scan(model, tok, windows, "olmo1b"); recited, never, wm = populations(acc)
mods = modules(model, TRAIN_LAYERS); W0 = {k: m.weight.detach().clone() for k, m in mods.items()}
Q = coupled_bases(model, mods, gen[:1152], tag, n=576)
# items: never-recited windows from sequences >= 1152 (outside the reference pool); ordinary eval: never windows from sequences < 1152
WPS = 512 // L; g = torch.Generator().manual_seed(1)
cand = [int(i) for i in never.tolist() if i // WPS >= 1152]; items_idx = torch.tensor(cand)[torch.randperm(len(cand), generator=g)[:NITEM]]
ord_idx = torch.tensor([int(i) for i in never.tolist() if i // WPS < 1152])[:400]
items = windows[items_idx]; ordinary = windows[ord_idx]; item_seqs = set((items_idx // WPS).tolist())
ref_pool = torch.tensor([s for s in range(1152, 2304) if s not in item_seqs]); REF = gen
if os.path.exists(f"{S}/dolmino_pool_big.pt"): REF = torch.load(f"{S}/dolmino_pool_big.pt"); ref_pool = torch.arange(len(REF)); print(f"using big pool {len(REF)}", flush=True); print(f"[{OPT}] items {len(items)} ordinary {len(ordinary)} ref pool {len(ref_pool)}", flush=True)
IM = wm[None].expand(NITEM, -1); OM = wm[None].expand(len(ordinary), -1); PM = torch.ones(128, 511, dtype=torch.bool)
def evaluate():
    r = token_stats(model, items, IM); o = token_stats(model, ordinary, OM); p = token_stats(model, pile[:128], PM, bs=8)
    return dict(item_strict=r["strict"], item_loss=r["loss"], ord_loss=o["loss"], pile_loss=p["loss"])
# activation / margin-gradient profiles at the probe layers (inputs of gate_proj; output gradients of the margin)
pm = modules(model, PROBE_LAYERS); cap = {}
def hook(k):
    def h(mod, inp, out):
        cap[(k, "a")] = inp[0].detach(); out.requires_grad_(True); out.register_hook(lambda gg: cap.__setitem__((k, "g"), gg.detach()))
    return h
def profiles(seqs, mask):
    hs = [m.register_forward_hook(hook(k)) for k, m in pm.items()]; EA = {k: 0 for k in pm}; EG = {k: 0 for k in pm}; cnt = 0
    for s in range(0, len(seqs), 16):
        x = seqs[s:s + 16].to(dev); mk = mask[s:s + 16].to(dev); logits = model(input_ids=x).logits[:, :-1].float(); y = x[:, 1:]
        tgt = logits.gather(-1, y[..., None])[..., 0]; other = logits.scatter(-1, y[..., None], -1e9).max(-1).values; ((tgt - other) * mk.float()).sum().backward()
        with torch.no_grad():
            for k in pm:
                a = cap[(k, "a")][:, :-1][mk].float(); gg = cap[(k, "g")][:, :-1][mk].float(); EA[k] = EA[k] + ((a @ Q[k][1]) ** 2).sum(0); EG[k] = EG[k] + ((gg @ Q[k][0]) ** 2).sum(0)
        cnt += int(mk.sum()); cap.clear(); del logits
    for h in hs: h.remove()
    for m in model.parameters(): m.grad = None
    return {k: (EA[k] / cnt, EG[k] / cnt) for k in pm}
def fit(x, y):
    x, y = np.log(np.asarray(x)), np.log(np.asarray(y)); X = np.stack([np.ones_like(x), x]).T; b = np.linalg.lstsq(X, y, rcond=None)[0]; r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum(); return float(b[1]), float(r2)
def report_profiles(label, Pi, Po):
    for k in pm:
        nbA = int(FLAT * len(Q[k][3])); nbG = int(FLAT * len(Q[k][2])); EAi, EGi = Pi[k]; EAo, EGo = Po[k]
        bA, r2A = fit(EAo.cpu(), (EAi / EAo).cpu()); bG, r2G = fit(EGo.cpu(), (EGi / EGo).cpu())
        print(f"[{OPT}] {label} {k}: A exponent {bA:+.3f} (R2 {r2A:.2f}) bulk share items {float(EAi[:nbA].sum() / EAi.sum()):.3f} ordinary {float(EAo[:nbA].sum() / EAo.sum()):.3f} total energy ratio {float(EAi.sum() / EAo.sum()):.3f} | G exponent {bG:+.3f} (R2 {r2G:.2f}) bulk share items {float(EGi[:nbG].sum() / EGi.sum()):.3f} ordinary {float(EGo[:nbG].sum() / EGo.sum()):.3f}", flush=True)
for p in model.parameters(): p.requires_grad_(False)
Pi0 = profiles(items, IM); Po = profiles(ordinary, OM); report_profiles("before", Pi0, Po)
base = evaluate(); print(f"[{OPT}] base: {base}", flush=True)
# ---- training
if OPT == "adam_full": params = [p for p in model.parameters()]
else: params = [m.weight for m in mods.values()]
for p in params: p.requires_grad_(True)
if OPT in ("adam", "adam_full"): opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
elif OPT == "sgd": opt = torch.optim.SGD(params, lr=LR, momentum=0.9)
else: opt = None
PRE_ = {}
if OPT == "ng":
    for k in mods:
        lam = Q[k][2][:, None] * Q[k][3][None, :]; PRE_[k] = lam.mean() / (lam + 0.05 * lam.mean())
gi = torch.Generator().manual_seed(2); step = 0; t0 = time.time(); order = torch.randperm(NITEM, generator=gi); ptr = 0
while step < MAX_STEPS:
    if ptr + NI > NITEM: order = torch.randperm(NITEM, generator=gi); ptr = 0
    xi = items[order[ptr:ptr + NI]].to(dev); ptr += NI; xr = REF[ref_pool[torch.randint(len(ref_pool), (NR,), generator=gi)]].to(dev)
    li = torch.nn.functional.cross_entropy(model(input_ids=xi).logits[:, :-1].reshape(-1, model.config.vocab_size), xi[:, 1:].reshape(-1))
    lr_ = torch.nn.functional.cross_entropy(model(input_ids=xr).logits[:, :-1].reshape(-1, model.config.vocab_size), xr[:, 1:].reshape(-1))
    (ITEM_W * li + lr_).backward()
    if opt is not None: opt.step(); opt.zero_grad()
    else:
        with torch.no_grad():
            for k, m in mods.items():
                D = Q[k][0].T @ m.weight.grad @ Q[k][1]; m.weight -= LR * (Q[k][0] @ (D * PRE_[k]) @ Q[k][1].T); m.weight.grad = None
    step += 1
    if step % 50 == 0 or step == 1:
        for p in params: p.requires_grad_(False)
        ev = evaluate(); upd = math.sqrt(sum(float(((m.weight - W0[k]) ** 2).sum()) for k, m in mods.items()))
        print(f"[{OPT}] step {step:4d} ({time.time() - t0:5.0f}s): item loss {li.item():.3f} ref loss {lr_.item():.3f} | items recited {ev['item_strict']:.3f} (loss {ev['item_loss']:.3f}) | ordinary d {ev['ord_loss'] - base['ord_loss']:+.4f} pile d {ev['pile_loss'] - base['pile_loss']:+.4f} | band update norm {upd:.1f}", flush=True)
        for p in params: p.requires_grad_(True)
        if ev["item_strict"] >= 0.95: break
for p in model.parameters(): p.requires_grad_(False)
final = evaluate(); print(f"[{OPT}] final after {step} steps: {final}", flush=True)
# ---- imprint profile
print(f"[{OPT}] imprint profile: matrix | exponent of imprint energy vs eigenvalue (G side, A side) | imprint bulk share (block) | imprint norm", flush=True)
D = {}
with torch.no_grad():
    for k, m in mods.items():
        dW = m.weight - W0[k]; D[k] = Q[k][0].T @ dW @ Q[k][1]; Eo = (D[k] ** 2).sum(1); Ei = (D[k] ** 2).sum(0)
        bo, r2o = fit(Q[k][2].cpu(), Eo.cpu()); bi, r2i = fit(Q[k][3].cpu(), Ei.cpu()); gs, as_ = band_slices(Q, k, "bulk")
        print(f"[{OPT}]   {k}: G {bo:+.3f} (R2 {r2o:.2f}) A {bi:+.3f} (R2 {r2i:.2f}) | bulk {float((D[k][gs, as_] ** 2).sum() / (D[k] ** 2).sum()):.3f} (0.36 if uniform) | {float(dW.norm()):.2f}", flush=True)
Pi1 = profiles(items, IM); report_profiles("after (vs ordinary before)", Pi1, Po); Po1 = profiles(ordinary, OM); report_profiles("after (vs ordinary after) ", Pi1, Po1)
# ---- edits of the final model
Wf = {k: m.weight.detach().clone() for k, m in mods.items()}
def edit_test(label, dW):
    set_weights(mods, Wf, dW); ev = evaluate(); set_weights(mods, Wf)
    print(f"[{OPT}]   {label}: items recited {ev['item_strict']:.3f} (loss {ev['item_loss']:.3f}) | ordinary d {ev['ord_loss'] - final['ord_loss']:+.4f} | pile d {ev['pile_loss'] - final['pile_loss']:+.4f}", flush=True)
print(f"[{OPT}] edits of the final model (band {TRAIN_LAYERS}):", flush=True)
for band in ("bulk", "head"):
    dWi = {}
    for k in mods:
        gs, as_ = band_slices(Q, k, band); Gs, As = Q[k][0][:, gs], Q[k][1][:, as_]; dWi[k] = -Gs @ D[k][gs, as_] @ As.T
    edit_test(f"remove imprint's {band} block", dWi)
    edit_test(f"remove full {band} block", {k: block_delta("remove", band, Q, Wf)[0][k] for k in mods})
torch.save({"D": {k: v.cpu() for k, v in D.items()}, "Wf": {k: v.cpu() for k, v in Wf.items()}, "final": final, "base": base, "steps": step, "items_idx": items_idx, "Pi0": {k: (a.cpu(), b.cpu()) for k, (a, b) in Pi0.items()}, "Pi1": {k: (a.cpu(), b.cpu()) for k, (a, b) in Pi1.items()}, "Po": {k: (a.cpu(), b.cpu()) for k, (a, b) in Po.items()}, "Po1": {k: (a.cpu(), b.cpu()) for k, (a, b) in Po1.items()}}, f"{S}/imprint_{RUN}.pt"); print("done", flush=True)
