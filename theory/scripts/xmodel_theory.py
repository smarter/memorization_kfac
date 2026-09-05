"""Quantitative tests of the two-order theory on any model. Usage: python xmodel_theory.py <model_path> <tag> <a-b>
(A) Collateral: for bulk removal, bulk noise, head removal, head noise of layers a-b gate/up, the first-order logit shift
    dz_t (central differences in fp32) gives the predicted loss change  E[(p-e_y).dz] + 1/2 E_y~p[(dz_y - p.dz)^2]  per
    population (ordinary dolmino, Pile, recited), compared with the measured change after applying the full edit.
(B) Forgetting: per-token margins of the recited items under partial bulk removal (alpha) -> linearity, first-order
    prediction of strict recitation; under bulk noise at four norms x two seeds -> mean shift ~ n^2, spread ~ n,
    mean/variance constant, two-coefficient prediction of strict recitation at the largest norm from the three smaller.
(C) Coupling law: recited/ordinary ratio of margin-shift variances under bulk noise (measured) against the product of
    bulk energies (margin-gradient side x activation side) of the two populations (predicted)."""
import sys, math, numpy as np, torch
sys.path.insert(0, "/tmp/claude-1002/-home-guillaume-memorization-kfac/a39940c3-dcc5-4714-8566-58fa7889e391/scratchpad")
from xmodel_common import *
path, tag = sys.argv[1], sys.argv[2]; lo, hi = map(int, sys.argv[3].split("-")); LAYERS = tuple(range(lo, hi + 1)); EPS = 0.02
tok, model = load(path, torch.float32); gen, pile, windows = data(tok); acc = scan(model, tok, windows, tag); recited, never, wm = populations(acc)
print(f"[{tag}] recited {len(recited)} never {len(never)}", flush=True)
mods = modules(model, LAYERS); W0 = {k: m.weight.detach().clone() for k, m in mods.items()}
Q = coupled_bases(model, mods, gen, tag)
full = lambda n, T: torch.ones(n, T - 1, dtype=torch.bool)
POP = {"ordinary": (windows[never[:600]], full(600, L)), "pile": (pile[:160], full(160, 512)), "recited": (windows[recited], wm[None].expand(len(recited), -1))}
BS = {"ordinary": 16, "pile": 4, "recited": 16}
base = {n: token_stats(model, *POP[n], bs=BS[n]) for n in POP}; print(f"[{tag}] base:", {n: (round(v['loss'], 4), round(v['strict'], 3)) for n, v in base.items()}, flush=True)
def two_order(dW, seqs, mask, bs):
    """Per-token first-order logit shift by central differences; returns mean first-order, second-order (Fisher) loss terms and mean ||dz||_inf."""
    f1 = f2 = inf = 0.0; cnt = 0
    with torch.no_grad():
        for s in range(0, len(seqs), bs):
            x = seqs[s:s + bs].to(dev); mk = mask[s:s + bs].to(dev).float(); y = x[:, 1:]
            set_weights(mods, W0, {k: EPS * v for k, v in dW.items()}); Zp = model(input_ids=x).logits[:, :-1].float()
            set_weights(mods, W0, {k: -EPS * v for k, v in dW.items()}); Zm = model(input_ids=x).logits[:, :-1].float()
            set_weights(mods, W0); Z0 = model(input_ids=x).logits[:, :-1].float()
            dz = (Zp - Zm) / (2 * EPS); p = Z0.softmax(-1); pdz = (p * dz).sum(-1)
            first = pdz - dz.gather(-1, y[..., None])[..., 0]; second = 0.5 * ((p * dz ** 2).sum(-1) - pdz ** 2)
            f1 += float((first * mk).sum()); f2 += float((second * mk).sum()); inf += float((dz.abs().amax(-1) * mk).sum()); cnt += int(mk.sum()); del Zp, Zm, Z0, dz, p
    return f1 / cnt, f2 / cnt, inf / cnt
# ---------- (A) collateral
print(f"\n[{tag}] (A) two-order collateral: edit -> population: first | second | sum | measured | E||dz||_inf", flush=True)
for kind, band in (("remove", "bulk"), ("noise", "bulk"), ("remove", "head"), ("noise", "head")):
    dW, norms = block_delta(kind, band, Q, W0)
    pred = {n: two_order(dW, *POP[n], BS[n]) for n in POP}
    set_weights(mods, W0, dW); meas = {n: token_stats(model, *POP[n], bs=BS[n]) for n in POP}
    for n in POP:
        f1, f2, inf = pred[n]; extra = f"   (strict recited {meas[n]['strict']:.3f})" if n == "recited" else ""
        print(f"[{tag}]   {band} {kind:6s} (norm {np.mean(norms):5.1f}) -> {n:8s}: {f1:+.4f} | {f2:+.4f} | {f1 + f2:+.4f} | {meas[n]['loss'] - base[n]['loss']:+.4f} | {inf:.3f}{extra}", flush=True)
set_weights(mods, W0)
# ---------- (B) forgetting at the margin
R, Rm = POP["recited"]; N_, Nm = windows[never[:600]], wm[None].expand(600, -1)
rb = token_stats(model, R, Rm, want_margins=True); nb = token_stats(model, N_, Nm, want_margins=True); m0 = rb["margins"][:, PRE - 1:]; n0 = nb["margins"][:, PRE - 1:]; base_nloss = nb["loss"]
bulknorm = float(np.mean(block_delta("remove", "bulk", Q, W0)[1]))
print(f"\n[{tag}] (B) margins: recited base mean {m0.mean():.2f} (min over item, median {m0.min(1).values.median():.2f}); bulk norm {bulknorm:.1f}", flush=True)
shr = {}
for a in (0.25, 0.5, 0.75, 1.0):
    dW, _ = block_delta("remove", "bulk", Q, W0, alpha=a); set_weights(mods, W0, dW); r = token_stats(model, R, Rm, want_margins=True); shr[a] = (r["margins"][:, PRE - 1:] - m0, r["strict"])
d25 = shr[0.25][0]
for a in (0.25, 0.5, 0.75, 1.0):
    d, strict = shr[a]; pred = float(((m0 + d25 * (a / 0.25)).min(1).values > 0).float().mean()); slope = float((d * d25).sum() / (d25 ** 2).sum()) * 0.25 / a
    print(f"[{tag}]   shrink alpha {a:.2f}: mean shift {d.mean():+.3f} | slope vs linear extrapolation of 0.25: {slope:.2f} | strict measured {strict:.3f} predicted (first order from 0.25) {pred:.3f}", flush=True)
noise = {}
for f in (0.25, 0.5, 0.75, 1.0):
    for seed in (0, 1):
        dW, _ = block_delta("noise", "bulk", Q, W0, alpha=f, seed=seed); set_weights(mods, W0, dW)
        r = token_stats(model, R, Rm, want_margins=True); rn = token_stats(model, N_, Nm, want_margins=True)
        noise[(f, seed)] = (r["margins"][:, PRE - 1:] - m0, r["strict"], rn["margins"][:, PRE - 1:] - n0, rn["loss"])
set_weights(mods, W0)
print(f"[{tag}]   noise (bulk block): norm | mean shift (sd) recited | mean/var | strict | ordinary: mean (sd), loss d | var ratio recited/ordinary", flush=True)
for f in (0.25, 0.5, 0.75, 1.0):
    d = torch.stack([noise[(f, s)][0] for s in (0, 1)]); dn = torch.stack([noise[(f, s)][2] for s in (0, 1)]); strict = np.mean([noise[(f, s)][1] for s in (0, 1)])
    print(f"[{tag}]     {f * bulknorm:5.1f} | {d.mean():+.3f} ({d.std():.3f}) | {float(d.mean() / d.var()):+.3f} | {strict:.3f} | {dn.mean():+.4f} ({dn.std():.4f}), {np.mean([noise[(f, s)][3] for s in (0, 1)]) - base_nloss:+.4f} | {float(d.var() / dn.var()):.2f}", flush=True)
# two-coefficient model from f in (0.25, 0.5, 0.75) -> predict strict at all four norms
fs = [0.25, 0.5, 0.75]; n2 = torch.tensor([(f * bulknorm) ** 2 for f in fs]); means = torch.stack([torch.stack([noise[(f, s)][0] for s in (0, 1)]).mean(0) for f in fs])
beta = -(means * n2[:, None, None]).sum(0) / (n2 ** 2).sum()
var_t = torch.stack([((noise[(f, 0)][0] - noise[(f, 1)][0]) ** 2 / 2) / (f * bulknorm) ** 2 for f in fs]).mean(0); gamma = var_t.sqrt()
from torch.distributions import Normal
for f in (0.25, 0.5, 0.75, 1.0):
    n = f * bulknorm; z = (m0 - beta * n ** 2) / (gamma * n + 1e-6); surv = Normal(0, 1).cdf(z).clamp(1e-6, 1).log().sum(1).exp().mean()
    print(f"[{tag}]   two-coefficient model at norm {n:5.1f}: predicted strict {float(surv):.3f} vs measured {np.mean([noise[(f, s)][1] for s in (0, 1)]):.3f} {'(out of sample)' if f == 1.0 else '(in sample)'}", flush=True)
# ---------- (C) coupling law: product of bulk energies
cap = {}
def hook(k):
    def h(mod, inp, out):
        cap[(k, "a")] = inp[0].detach(); out.requires_grad_(True); out.register_hook(lambda gg: cap.__setitem__((k, "g"), gg.detach()))
    return h
hs = [m.register_forward_hook(hook(k)) for k, m in mods.items()]
def bulk_energies(seqs, mask):
    EG = {k: 0.0 for k in mods}; EA = {k: 0.0 for k in mods}; cnt = 0
    for s in range(0, len(seqs), 8):
        x = seqs[s:s + 8].to(dev); mk = mask[s:s + 8].to(dev); logits = model(input_ids=x).logits[:, :-1].float(); y = x[:, 1:]
        tgt = logits.gather(-1, y[..., None])[..., 0]; other = logits.scatter(-1, y[..., None], -1e9).max(-1).values; ((tgt - other) * mk).sum().backward()
        with torch.no_grad():
            for k in mods:
                gs, as_ = band_slices(Q, k, "bulk"); a = cap[(k, "a")][:, :-1][mk]; gg = cap[(k, "g")][:, :-1][mk]
                EA[k] += float(((a @ Q[k][1][:, as_]) ** 2).sum()); EG[k] += float(((gg @ Q[k][0][:, gs]) ** 2).sum())
        cnt += int(mk.sum()); cap.clear(); del logits
    return {k: EG[k] / cnt for k in mods}, {k: EA[k] / cnt for k in mods}
EGr, EAr = bulk_energies(R, Rm); EGn, EAn = bulk_energies(N_, Nm)
pred = np.mean([EGr[k] * EAr[k] / (EGn[k] * EAn[k]) for k in mods]); dv = torch.stack([noise[(1.0, s)][0] for s in (0, 1)]).var(); dnv = torch.stack([noise[(1.0, s)][2] for s in (0, 1)]).var()
print(f"\n[{tag}] (C) coupling law: predicted variance ratio recited/ordinary (product of bulk energies) {pred:.2f} | measured at norm {bulknorm:.0f}: {float(dv / dnv):.2f}; output-side ratio alone {np.mean([EGr[k] / EGn[k] for k in mods]):.2f}, input-side {np.mean([EAr[k] / EAn[k] for k in mods]):.2f}", flush=True)
torch.save({"m0": m0, "shr": {a: v[0] for a, v in shr.items()}, "noise": {k: (v[0], v[1]) for k, v in noise.items()}, "bulknorm": bulknorm}, f"{S}/xmodel_{tag}_theory.pt"); print("done", flush=True)
