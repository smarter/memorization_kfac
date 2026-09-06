"""Phase 2 of the imprint experiment: selective decay under population training. Usage: python imprint_decay.py <run> <optimizer> <lr> [steps]
Load the memorized band weights of a finished run (imprint_<run>.pt: Wf, or D to rebuild Wf = W0 + Q D P^T), then train the
same band on reference sequences only (no items), same optimizer and lr, tracking every 50 steps: items recited, item loss,
ordinary loss, the imprint's energy (relative to the original weights) in the bulk and head blocks of the population basis,
and the items' activation profile at the input of layer 13. Prediction: the head part of the imprint decays faster than the
bulk part; recitation that survives rests on the bulk; the items' representation flattens further."""
import sys, math, time, glob, numpy as np, torch
sys.path.insert(0, "/tmp/claude-1002/-home-guillaume-memorization-kfac/a39940c3-dcc5-4714-8566-58fa7889e391/scratchpad")
from xmodel_common import *
RUN, OPT, LR = sys.argv[1], sys.argv[2], float(sys.argv[3]); STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 400
P1B = glob.glob("/home/guillaume/.cache/huggingface/hub/models--allenai--OLMo-2-0425-1B/snapshots/*")[0]
TRAIN_LAYERS = (9, 10, 11, 12, 13); tag = "olmo1b_band"
tok, model = load(P1B, torch.float32); gen, pile, windows = data(tok); acc = scan(model, tok, windows, "olmo1b"); recited, never, wm = populations(acc)
mods = modules(model, TRAIN_LAYERS); W0 = {k: m.weight.detach().clone() for k, m in mods.items()}; Q = coupled_bases(model, mods, gen[:1152], tag, n=576)
R = torch.load(f"{S}/imprint_{RUN}.pt"); items = windows[R["items_idx"]]; WPS = 512 // L; item_seqs = set((R["items_idx"] // WPS).tolist())
if "Wf" in R: Wf = {k: v.to(dev) for k, v in R["Wf"].items()}
else: Wf = {k: W0[k] + Q[k][0] @ R["D"][k].to(dev) @ Q[k][1].T for k in mods}
set_weights(mods, Wf)
ord_idx = torch.tensor([int(i) for i in never.tolist() if i // WPS < 1152])[:400]; ordinary = windows[ord_idx]
ref_pool = torch.tensor([s for s in range(1152, 2304) if s not in item_seqs])
IM = wm[None].expand(len(items), -1); OM = wm[None].expand(len(ordinary), -1)
def imprint_energy():
    out = {"bulk": 0.0, "head": 0.0, "total": 0.0}
    with torch.no_grad():
        for k, m in mods.items():
            D = Q[k][0].T @ (m.weight - W0[k]) @ Q[k][1]; out["total"] += float((D ** 2).sum())
            for band in ("bulk", "head"): gs, as_ = band_slices(Q, k, band); out[band] += float((D[gs, as_] ** 2).sum())
    return out
pm = modules(model, (13,)); cap = {}
def hook(k):
    def h(mod, inp, out): cap[k] = inp[0].detach()
    return h
def bulk_share(seqs, mask):
    hs = [m.register_forward_hook(hook(k)) for k, m in pm.items()]; E = 0; cnt = 0
    with torch.no_grad():
        for s in range(0, len(seqs), 16):
            x = seqs[s:s + 16].to(dev); mk = mask[s:s + 16].to(dev); model(input_ids=x)
            a = cap[(13, "gate_proj")][:, :-1][mk].float(); E = E + ((a @ Q[(13, "gate_proj")][1]) ** 2).sum(0); cnt += int(mk.sum())
    for h in hs: h.remove()
    nb = int(FLAT * len(E)); return float(E[:nb].sum() / E.sum())
def status(step, t0):
    r = token_stats(model, items, IM); o = token_stats(model, ordinary, OM); e = imprint_energy()
    print(f"[{RUN} decay] step {step:4d} ({time.time() - t0:4.0f}s): items recited {r['strict']:.3f} (loss {r['loss']:.3f}) | ordinary {o['loss']:.4f} | imprint energy total {e['total']:.2f} bulk {e['bulk']:.2f} head {e['head']:.2f} (head/bulk {e['head'] / max(e['bulk'], 1e-9):.2f}) | items bulk share @13 {bulk_share(items, IM):.3f} ordinary {bulk_share(ordinary, OM):.3f}", flush=True)
t0 = time.time(); status(0, t0)
params = [m.weight for m in mods.values()]
for p in params: p.requires_grad_(True)
if OPT == "adam": opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
elif OPT == "sgd": opt = torch.optim.SGD(params, lr=LR, momentum=0.9)
else:
    opt = None; PRE_ = {}
    for k in mods: lam = Q[k][2][:, None] * Q[k][3][None, :]; PRE_[k] = lam.mean() / (lam + 0.05 * lam.mean())
gi = torch.Generator().manual_seed(3)
for step in range(1, STEPS + 1):
    xr = gen[ref_pool[torch.randint(len(ref_pool), (8,), generator=gi)]].to(dev)
    torch.nn.functional.cross_entropy(model(input_ids=xr).logits[:, :-1].reshape(-1, model.config.vocab_size), xr[:, 1:].reshape(-1)).backward()
    if opt is not None: opt.step(); opt.zero_grad()
    else:
        with torch.no_grad():
            for k, m in mods.items(): D = Q[k][0].T @ m.weight.grad @ Q[k][1]; m.weight -= LR * (Q[k][0] @ (D * PRE_[k]) @ Q[k][1].T); m.weight.grad = None
    if step % 50 == 0:
        for p in params: p.requires_grad_(False)
        status(step, t0)
        for p in params: p.requires_grad_(True)
print("done", flush=True)
