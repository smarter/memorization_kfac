"""Probe v2 (bf16, any model, any band, MLP or attention projections, windows or Dolma population), saving everything the
figures and bootstrap need. Usage: python xmodel_probe2.py <model_path> <tag> <a-b> [windows|dolma] [mlp|attn]
Saves xmodel2_<tag>_<pop>_<mods>_L<a-b>.pt with: eigenvalues; per-direction energies (recited, never) both sides; per-item
energy vectors (input side, first matrix) for bootstrap; per-window activation bulk share for all 9216 windows plus the
Dolma set (detector); per-item margins at base, alpha=0.25 and alpha=1 bulk removal; edit table."""
import sys, os, numpy as np, torch
sys.path.insert(0, "/tmp/claude-1002/-home-guillaume-memorization-kfac/a39940c3-dcc5-4714-8566-58fa7889e391/scratchpad")
import xmodel_common as X
from xmodel_common import *
path, tag = sys.argv[1], sys.argv[2]; lo, hi = map(int, sys.argv[3].split("-")); LAYERS = tuple(range(lo, hi + 1))
POPN = sys.argv[4] if len(sys.argv) > 4 else "windows"; MODS = sys.argv[5] if len(sys.argv) > 5 else "mlp"
OUT = f"{S}/xmodel2_{tag}_{POPN}_{MODS}_L{lo}-{hi}.pt"; btag = f"{tag}_{MODS}_L{lo}-{hi}"
tok, model = load(path); gen, pile, windows = data(tok); acc = scan(model, tok, windows, tag); _, never, wm = populations(acc)
recited = (acc == 1).nonzero()[:, 0]
if len(recited) < 40: recited = (acc >= 0.95).nonzero()[:, 0]
dol = torch.load(f"{S}/population_sets2.pt")["mem"][0]
if POPN == "dolma":
    f = f"{S}/xmodel_{tag}_dolmascan.pt"
    if os.path.exists(f): accd = torch.load(f)
    else:
        accd = []
        with torch.no_grad():
            for s in range(0, len(dol), 64):
                x = dol[s:s + 64].to(dev); out = model.generate(x[:, :PRE], max_new_tokens=L - PRE, do_sample=False, pad_token_id=tok.pad_token_id or 0); accd.append((out[:, PRE:L] == x[:, PRE:]).float().mean(1).cpu())
        accd = torch.cat(accd); torch.save(accd, f)
    R = dol[(accd == 1).nonzero()[:, 0]]; print(f"[{tag}] Dolma set recited {len(R)} of {len(dol)}", flush=True)
else: R = windows[recited]
N_ = windows[never[:600]]; Rm = wm[None].expand(len(R), -1); Nm = wm[None].expand(len(N_), -1)
print(f"[{tag}] population {POPN}: {len(R)} recited; band {LAYERS} {MODS}", flush=True)
if MODS == "attn": mods = {(l, p): getattr(model.model.layers[l].self_attn, p) for l in LAYERS for p in ("q_proj", "k_proj", "v_proj", "o_proj")}
else: mods = modules(model, LAYERS)
W0 = {k: m.weight.detach().float().clone() for k, m in mods.items()}; Q = coupled_bases(model, mods, gen, btag)
res = {"tag": tag, "pop": POPN, "mods": MODS, "layers": LAYERS, "n_recited": len(R), "eig": {k: (Q[k][2].cpu(), Q[k][3].cpu()) for k in mods}}
# ---- per-direction energies (margin gradients, activations), per item
cap = {}
def hook(k):
    def h(mod, inp, out):
        cap[(k, "a")] = inp[0].detach(); out.requires_grad_(True); out.register_hook(lambda gg: cap.__setitem__((k, "g"), gg.detach()))
    return h
def energies(seqs, mask, per_item_key=None):
    hs = [m.register_forward_hook(hook(k)) for k, m in mods.items()]; EG = {k: 0 for k in mods}; EA = {k: 0 for k in mods}; cnt = 0; per_item = []
    for s in range(0, len(seqs), 8):
        x = seqs[s:s + 8].to(dev); mk = mask[s:s + 8].to(dev); logits = model(input_ids=x).logits[:, :-1].float(); y = x[:, 1:]
        tgt = logits.gather(-1, y[..., None])[..., 0]; other = logits.scatter(-1, y[..., None], -1e9).max(-1).values; ((tgt - other) * mk.float()).sum().backward()
        with torch.no_grad():
            for k in mods:
                a = cap[(k, "a")][:, :-1].float(); gg = cap[(k, "g")][:, :-1].float(); pa = ((a @ Q[k][1]) ** 2) * mk[..., None]; pg = ((gg @ Q[k][0]) ** 2) * mk[..., None]
                EA[k] = EA[k] + pa.sum((0, 1)); EG[k] = EG[k] + pg.sum((0, 1))
                if k == per_item_key: per_item.append(pa.sum(1).cpu())
        cnt += int(mk.sum()); cap.clear(); del logits
    for h in hs: h.remove()
    for m in mods.values(): m.weight.grad = None
    return {k: (EA[k] / cnt).cpu() for k in mods}, {k: (EG[k] / cnt).cpu() for k in mods}, (torch.cat(per_item) if per_item else None)
k0 = next(iter(mods)); EAr, EGr, PIr = energies(R, Rm, k0); EAn, EGn, PIn = energies(N_, Nm, k0)
res.update(EAr=EAr, EGr=EGr, EAn=EAn, EGn=EGn, per_item_A_recited=PIr, per_item_A_never=PIn)
def fit(x, y):
    x, y = np.log(np.asarray(x, dtype=np.float64) + 1e-30), np.log(np.asarray(y, dtype=np.float64) + 1e-30); X_ = np.stack([np.ones_like(x), x]).T; b = np.linalg.lstsq(X_, y, rcond=None)[0]; r2 = 1 - ((y - X_ @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum(); return float(b[1]), float(r2)
rng = np.random.default_rng(0); nbA = int(FLAT * PIr.shape[1]); En0 = EAn[k0].numpy()
boots = []
for _ in range(200):
    idx = rng.integers(0, len(PIr), len(PIr)); Em = PIr[idx].mean(0).numpy() / (Rm[0].sum().item()); boots.append((fit(En0, Em / En0)[0], float(Em[:nbA].sum() / Em.sum())))
boots = np.array(boots); res["boot_exponent_ci"] = (float(np.percentile(boots[:, 0], 2.5)), float(np.percentile(boots[:, 0], 97.5))); res["boot_bulkshare_ci"] = (float(np.percentile(boots[:, 1], 2.5)), float(np.percentile(boots[:, 1], 97.5)))
for k in mods:
    nb = int(FLAT * len(EAn[k])); bA, r2A = fit(EAn[k], EAr[k] / EAn[k]); bG, r2G = fit(EGn[k], EGr[k] / EGn[k])
    print(f"[{tag}] {k}: A exponent {bA:+.3f} (R2 {r2A:.2f}) bulk share never {float(EAn[k][:nb].sum() / EAn[k].sum()):.3f} recited {float(EAr[k][:nb].sum() / EAr[k].sum()):.3f} | G exponent {bG:+.3f} (R2 {r2G:.2f}) bulk share never {float(EGn[k][:int(FLAT * len(EGn[k]))].sum() / EGn[k].sum()):.3f} recited {float(EGr[k][:int(FLAT * len(EGr[k]))].sum() / EGr[k].sum()):.3f}", flush=True)
print(f"[{tag}] bootstrap 95% CI (first matrix, input side): exponent {res['boot_exponent_ci']}, recited bulk share {res['boot_bulkshare_ci']}", flush=True)
# ---- detector: per-window activation bulk share (first matrix input) for all windows and the Dolma set
def bulk_shares(seqs, mask, bs=32):
    hs = [mods[k0].register_forward_hook(hook(k0))]; out = []
    with torch.no_grad():
        for s in range(0, len(seqs), bs):
            x = seqs[s:s + bs].to(dev); mk = mask[s:s + bs].to(dev); model(input_ids=x); a = cap[(k0, "a")][:, :-1].float(); pa = ((a @ Q[k0][1]) ** 2) * mk[..., None]; e = pa.sum(1); out.append((e[:, :nbA].sum(1) / e.sum(1)).cpu()); cap.clear()
    for h in hs: h.remove()
    return torch.cat(out)
allm = wm[None].expand(len(windows), -1); bs_all = bulk_shares(windows, allm); bs_dol = bulk_shares(dol, wm[None].expand(len(dol), -1))
res.update(bulkshare_all_windows=bs_all, acc_all_windows=acc, bulkshare_dolma=bs_dol)
lab = (acc == 1).numpy(); sc = bs_all.numpy(); order = np.argsort(sc); ranks = np.empty(len(sc)); ranks[order] = np.arange(len(sc)); auc = (ranks[lab].sum() - lab.sum() * (lab.sum() - 1) / 2) / (lab.sum() * (~lab).sum())
print(f"[{tag}] detector AUC (bulk share of a window's suffix activations -> recited): {auc:.3f}; mean share recited {sc[lab].mean():.3f} vs not {sc[~lab].mean():.3f}", flush=True); res["detector_auc"] = float(auc)
# ---- edits and margins
POP = {"recited": (R, Rm), "never": (N_, Nm), "pile": (pile[:256], torch.ones(256, 511, dtype=torch.bool))}; BS = {"recited": 16, "never": 16, "pile": 4}
base = {n: token_stats(model, *POP[n], bs=BS[n], want_margins=(n == "recited")) for n in POP}; m0 = base["recited"]["margins"][:, PRE - 1:]
res["margins_base"] = m0; print(f"[{tag}] base: recited strict {base['recited']['strict']:.3f}, margins mean {m0.mean():.2f} median-min {m0.min(1).values.median():.2f}", flush=True)
table = {}
for kind, band, alpha in (("remove", "bulk", 0.25), ("remove", "bulk", 1.0), ("noise", "bulk", 1.0), ("remove", "head", 1.0), ("noise", "head", 1.0)):
    dW, norms = block_delta(kind, band, Q, W0, alpha=alpha); set_weights(mods, W0, dW); r = {n: token_stats(model, *POP[n], bs=BS[n], want_margins=(n == "recited")) for n in POP}
    key = f"{band}_{kind}_{alpha:g}"; table[key] = dict(norm=float(np.mean(norms)), strict=r["recited"]["strict"], never_d=r["never"]["loss"] - base["never"]["loss"], pile_d=r["pile"]["loss"] - base["pile"]["loss"], margins=r["recited"]["margins"][:, PRE - 1:])
    print(f"[{tag}]   {key:18s} (norm {np.mean(norms):5.1f}): recited {r['recited']['strict']:.3f} | never {table[key]['never_d']:+.4f} | pile {table[key]['pile_d']:+.4f}", flush=True)
set_weights(mods, W0)
d25 = table["bulk_remove_0.25"]["margins"] - m0; pred = float(((m0 + 4 * d25).min(1).values > 0).float().mean())
print(f"[{tag}] first-order deletability: predicted recited after full bulk removal {pred:.3f} (from alpha=0.25) vs measured {table['bulk_remove_1']['strict']:.3f}; head/bulk removal-noise cost ratios {table['head_remove_1']['never_d'] / max(table['head_noise_1']['never_d'], 1e-6):.2f} / {table['bulk_remove_1']['never_d'] / max(table['bulk_noise_1']['never_d'], 1e-6):.2f}", flush=True)
res.update(table=table, pred_first_order=pred, base=base); torch.save(res, OUT); print("done", flush=True)
