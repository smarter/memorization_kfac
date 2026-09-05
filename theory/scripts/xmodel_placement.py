"""Layer map and its test. Usage: python xmodel_placement.py <model_path> <tag>
Map: per layer, C_l = mean over target tokens of sum_s ||a_{l,s}||^2 (||dm_t/dh_{l,s}||^2 + ||dm_t/du_{l,s}||^2) by the
random-sign trick, for recited suffix tokens and ordinary suffix tokens. Prediction: isotropic noise of per-entry variance
sigma^2 on the gate/up matrices of a band shifts margins with variance sigma^2 sum_{l in band} C_l. Test: noise of
per-matrix norm NU on five 3-layer bands; measured margin-shift variances (both populations), strict recitation,
ordinary and Pile loss."""
import sys, numpy as np, torch
sys.path.insert(0, "/tmp/claude-1002/-home-guillaume-memorization-kfac/a39940c3-dcc5-4714-8566-58fa7889e391/scratchpad")
from xmodel_common import *
path, tag = sys.argv[1], sys.argv[2]; K = 4; NU = 60.0
tok, model = load(path); gen, pile, windows = data(tok); acc = scan(model, tok, windows, tag); recited, never, wm = populations(acc)
NL = len(model.model.layers); R, Rm = windows[recited], wm[None].expand(len(recited), -1); N_, Nm = windows[never[:400]], wm[None].expand(400, -1)
print(f"[{tag}] layers {NL}, recited {len(recited)}", flush=True)
cap = {}
def mk(l):
    mlp = model.model.layers[l].mlp
    def fwd(x):
        h = mlp.gate_proj(x); u = mlp.up_proj(x)
        if torch.is_grad_enabled():
            h.requires_grad_(); u.requires_grad_(); cap[(l, "a2")] = (x.detach().float() ** 2).sum(-1); cap[(l, "h")] = h; cap[(l, "u")] = u
        return mlp.down_proj(mlp.act_fn(h) * u)
    return fwd
for l in range(NL): model.model.layers[l].mlp.forward = mk(l)
def coupling(seqs, mask, bs=8):
    acc_ = torch.zeros(NL, device=dev); ntar = 0
    for s in range(0, len(seqs), bs):
        x = seqs[s:s + bs].to(dev); mk_ = mask[s:s + bs].to(dev).float()
        for _ in range(K):
            with torch.enable_grad():
                logits = model(input_ids=x).logits[:, :-1].float(); y = x[:, 1:]
                tgt = logits.gather(-1, y[..., None])[..., 0]; other = logits.scatter(-1, y[..., None], -1e9).max(-1).values; m = tgt - other
                eps = (torch.randint(0, 2, m.shape, device=dev).float() * 2 - 1) * mk_
                grads = torch.autograd.grad((m * eps).sum(), [cap[(l, k)] for l in range(NL) for k in ("h", "u")])
            with torch.no_grad():
                for l in range(NL):
                    gh, gu = grads[2 * l][:, :-1].float(), grads[2 * l + 1][:, :-1].float(); a2 = cap[(l, "a2")][:, :-1]
                    acc_[l] += ((gh ** 2).sum(-1) * a2).sum() / K + ((gu ** 2).sum(-1) * a2).sum() / K
            del logits, grads
        ntar += int(mk_.sum()); cap.clear()
    return (acc_ / ntar).cpu()
import os
mf = f"{S}/xmodel_{tag}_placement_map.pt"
if os.path.exists(mf): M = torch.load(mf); Cr, Cn = M["Cr"], M["Cn"]; print("loaded map", flush=True)
else: Cr = coupling(R, Rm); Cn = coupling(N_, Nm); torch.save({"Cr": Cr, "Cn": Cn}, mf)
print(f"\n[{tag}] layer | C recited (x1e3) | C ordinary (x1e3) | ratio")
for l in range(NL): print(f"[{tag}] {l:5d} | {1e3 * Cr[l]:9.3f} | {1e3 * Cn[l]:9.3f} | {Cr[l] / Cn[l]:6.2f}")
# test: isotropic noise on five bands
m0 = token_stats(model, R, Rm, want_margins=True)["margins"][:, PRE - 1:]; n0 = token_stats(model, N_, Nm, want_margins=True)["margins"][:, PRE - 1:]
basep = token_stats(model, pile[:160], torch.ones(160, 511, dtype=torch.bool), bs=4)["loss"]; basen = token_stats(model, N_, Nm)["loss"]
bands = [(int(l), int(l) + 2) for l in np.linspace(3, NL - 4, 5).round().astype(int)]
print(f"\n[{tag}] band noise (per-matrix norm {NU}): band | predicted var recited, ordinary | measured var recited, ordinary | pred ratio | meas ratio | strict recited | ordinary d | pile d", flush=True)
for lo, hi in bands:
    ms = modules(model, range(lo, hi + 1)); W0 = {k: m.weight.detach().clone() for k, m in ms.items()}; O, I = next(iter(W0.values())).shape; sig2 = NU ** 2 / (O * I)
    g = torch.Generator(device=dev).manual_seed(int(lo)); dW = {}
    for k in W0: n = torch.randn(W0[k].shape, device=dev, generator=g, dtype=torch.float32); dW[k] = n * (NU / n.norm())
    set_weights(ms, W0, dW)
    r = token_stats(model, R, Rm, want_margins=True); rn = token_stats(model, N_, Nm, want_margins=True); rp = token_stats(model, pile[:160], torch.ones(160, 511, dtype=torch.bool), bs=4)
    d = r["margins"][:, PRE - 1:] - m0; dn = rn["margins"][:, PRE - 1:] - n0; pr, pn = float(sig2 * Cr[lo:hi + 1].sum()), float(sig2 * Cn[lo:hi + 1].sum())
    print(f"[{tag}]   {lo:2d}-{hi:2d} | {pr:.3f}, {pn:.3f} | {float(d.var()):.3f}, {float(dn.var()):.3f} | {pr / pn:.2f} | {float(d.var() / dn.var()):.2f} | {r['strict']:.3f} | {rn['loss'] - basen:+.4f} | {rp['loss'] - basep:+.4f}", flush=True)
    set_weights(ms, W0)
torch.save({"Cr": Cr, "Cn": Cn}, f"{S}/xmodel_{tag}_placement.pt"); print("done", flush=True)
