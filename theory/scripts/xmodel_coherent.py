"""Coherent edit on any model. Usage: python xmodel_coherent.py <model_path> <tag> <a-b>
Split the recited set in halves (seed 0); direction = -(bulk-projected summed margin gradient of the train half), per-matrix
normalised; steps at norms 2.5, 5, 10, 20 per matrix; report train/held-out recitation, ordinary and Pile loss, and the
cosine of the direction with the ordinary population's summed margin gradient (the alignment that prices the collateral)."""
import sys, numpy as np, torch
sys.path.insert(0, "/tmp/claude-1002/-home-guillaume-memorization-kfac/a39940c3-dcc5-4714-8566-58fa7889e391/scratchpad")
from xmodel_common import *
path, tag = sys.argv[1], sys.argv[2]; lo, hi = map(int, sys.argv[3].split("-")); LAYERS = tuple(range(lo, hi + 1))
tok, model = load(path); gen, pile, windows = data(tok); acc = scan(model, tok, windows, tag); recited, never, wm = populations(acc)
mods = modules(model, LAYERS); W0 = {k: m.weight.detach().float().clone() for k, m in mods.items()}; Q = coupled_bases(model, mods, gen, tag)
g = torch.Generator().manual_seed(0); perm = recited[torch.randperm(len(recited), generator=g)]; tr, te = perm[: len(perm) // 2], perm[len(perm) // 2:]
def margin_grad(seqs, mask):
    for m in mods.values(): m.weight.requires_grad_(True)
    G = {k: torch.zeros_like(W0[k]) for k in mods}
    for s in range(0, len(seqs), 16):
        x = seqs[s:s + 16].to(dev); mk = mask[s:s + 16].to(dev).float(); logits = model(input_ids=x).logits[:, :-1].float(); y = x[:, 1:]
        tgt = logits.gather(-1, y[..., None])[..., 0]; other = logits.scatter(-1, y[..., None], -1e9).max(-1).values; ((tgt - other) * mk).sum().backward()
        with torch.no_grad():
            for k, m in mods.items(): G[k] += m.weight.grad.float(); m.weight.grad = None
        del logits
    for m in mods.values(): m.weight.requires_grad_(False)
    return G
def proj(G):
    out = {}
    for k in mods:
        gs, as_ = band_slices(Q, k, "bulk"); Gs, As = Q[k][0][:, gs], Q[k][1][:, as_]; out[k] = Gs @ (Gs.T @ G[k] @ As) @ As.T
    return out
def cos(a, b): return float(sum((a[k] * b[k]).sum() for k in mods) / (sum((a[k] ** 2).sum() for k in mods).sqrt() * sum((b[k] ** 2).sum() for k in mods).sqrt()))
Gtr = proj(margin_grad(windows[tr], wm[None].expand(len(tr), -1))); Gte = proj(margin_grad(windows[te], wm[None].expand(len(te), -1))); Gn = proj(margin_grad(windows[never[:600]], wm[None].expand(600, -1)))
print(f"[{tag}] bulk cosines: train/held-out halves {cos(Gtr, Gte):+.3f}; train vs ordinary {cos(Gtr, Gn):+.3f}", flush=True)
POP = {"train": (windows[tr], wm[None].expand(len(tr), -1)), "held-out": (windows[te], wm[None].expand(len(te), -1)), "ordinary": (windows[never[:600]], wm[None].expand(600, -1)), "pile": (pile[:160], torch.ones(160, 511, dtype=torch.bool))}
BS = {"train": 16, "held-out": 16, "ordinary": 16, "pile": 4}
base = {n: token_stats(model, *POP[n], bs=BS[n]) for n in POP}; print(f"[{tag}] base: " + ", ".join(f"{n} {v['strict']:.3f}/{v['loss']:.3f}" for n, v in base.items()), flush=True)
print(f"[{tag}] coherent edit: norm | train recited | held-out recited | ordinary d | pile d", flush=True)
for nu in (2.5, 5.0, 10.0, 20.0):
    dW = {k: -nu * Gtr[k] / Gtr[k].norm() for k in mods}; set_weights(mods, W0, dW); r = {n: token_stats(model, *POP[n], bs=BS[n]) for n in POP}
    print(f"[{tag}]   {nu:5.1f} | {r['train']['strict']:.3f} | {r['held-out']['strict']:.3f} | {r['ordinary']['loss'] - base['ordinary']['loss']:+.4f} | {r['pile']['loss'] - base['pile']['loss']:+.4f}", flush=True)
set_weights(mods, W0); print("done", flush=True)
