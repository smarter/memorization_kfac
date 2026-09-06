"""Edits for the capability suite, matched on held-out forgetting. Usage:
  python xmodel_unlearn.py <model_path> <tag> <a-b> coherent <norm>
  python xmodel_unlearn.py <model_path> <tag> <a-b> ga|npo <lr> <target_heldout_recited> [max_steps] [retain_weight] [beta]
All methods edit the gate/up matrices of layers a-b only. The recited dolmino windows are split in halves (seed 0, as in
xmodel_coherent.py); GA/NPO train on the train half with a retain cross-entropy on reference sequences and stop when the
held-out half's strict recitation falls to the target (checked every 10 steps). Reports train/held-out recitation, ordinary
and Pile loss; saves the edited model to $S/models/<tag>_<method>[_<param>] for the suite."""
import sys, os, math, copy, numpy as np, torch
sys.path.insert(0, "/tmp/claude-1002/-home-guillaume-memorization-kfac/a39940c3-dcc5-4714-8566-58fa7889e391/scratchpad")
from xmodel_common import *
path, tag = sys.argv[1], sys.argv[2]; lo, hi = map(int, sys.argv[3].split("-")); LAYERS = tuple(range(lo, hi + 1)); METHOD = sys.argv[4]
POPN = os.environ.get("POP", "windows")
tok, model = load(path); gen, pile, windows = data(tok); acc = scan(model, tok, windows, tag); recited, never, wm = populations(acc)
mods = modules(model, LAYERS); W0 = {k: m.weight.detach().float().clone() for k, m in mods.items()}
if POPN == "dolma":
    dol = torch.load(f"{S}/population_sets2.pt")["mem"][0]; sp = torch.load(f"{S}/targeted_split.pt"); windows_src = dol; tr, te = sp["train_idx"], sp["heldout_idx"]; print(f"[{tag}] Dolma population: train {len(tr)} held-out {len(te)}", flush=True)
else:
    g = torch.Generator().manual_seed(0); perm = recited[torch.randperm(len(recited), generator=g)]; tr, te = perm[: len(perm) // 2], perm[len(perm) // 2:]; windows_src = windows
POP = {"train": (windows_src[tr], wm[None].expand(len(tr), -1)), "held-out": (windows_src[te], wm[None].expand(len(te), -1)), "ordinary": (windows[never[:600]], wm[None].expand(600, -1)), "pile": (pile[:160], torch.ones(160, 511, dtype=torch.bool))}
BS = {"train": 16, "held-out": 16, "ordinary": 16, "pile": 4}
def evaluate(): return {n: token_stats(model, *POP[n], bs=BS[n]) for n in POP}
base = evaluate(); print(f"[{tag} {METHOD}] base: " + ", ".join(f"{n} {v['strict']:.3f}/{v['loss']:.3f}" for n, v in base.items()), flush=True)
def report(label, r): print(f"[{tag} {METHOD}] {label}: train {r['train']['strict']:.3f} | held-out {r['held-out']['strict']:.3f} | ordinary {r['ordinary']['loss'] - base['ordinary']['loss']:+.4f} | pile {r['pile']['loss'] - base['pile']['loss']:+.4f}", flush=True)
if METHOD == "coherent":
    NU = float(sys.argv[5]); Q = coupled_bases(model, mods, gen, tag)
    for m in mods.values(): m.weight.requires_grad_(True)
    G = {k: torch.zeros_like(W0[k]) for k in mods}
    for s in range(0, len(tr), 16):
        x = windows_src[tr[s:s + 16]].to(dev); mk = wm[None].expand(min(16, len(tr) - s), -1).to(dev).float(); logits = model(input_ids=x).logits[:, :-1].float(); y = x[:, 1:]
        tgt = logits.gather(-1, y[..., None])[..., 0]; other = logits.scatter(-1, y[..., None], -1e9).max(-1).values; ((tgt - other) * mk).sum().backward()
        with torch.no_grad():
            for k, m in mods.items(): G[k] += m.weight.grad.float(); m.weight.grad = None
        del logits
    for m in mods.values(): m.weight.requires_grad_(False)
    dW = {}
    for k in mods:
        gs, as_ = band_slices(Q, k, "bulk"); Gs, As = Q[k][0][:, gs], Q[k][1][:, as_]; Pg = Gs @ (Gs.T @ G[k] @ As) @ As.T; dW[k] = -NU * Pg / Pg.norm()
    set_weights(mods, W0, dW); r = evaluate(); report(f"coherent norm {NU:g}", r); name = f"{tag}_{POPN}_coherent{NU:g}"
else:
    LR = float(sys.argv[5]); TARGET = float(sys.argv[6]); MAX = int(sys.argv[7]) if len(sys.argv) > 7 else 300; RW = float(sys.argv[8]) if len(sys.argv) > 8 else 1.0; BETA = float(sys.argv[9]) if len(sys.argv) > 9 else 0.1
    ref = None; SNAP = [0.6, 0.35]
    if METHOD == "npo": ref = copy.deepcopy(model).eval()
    params = [m.weight for m in mods.values()]
    for p in params: p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.0); gi = torch.Generator().manual_seed(1); step = 0; refpool = torch.arange(1152, 2304)
    def seq_logp(mdl, x, mk):
        logits = mdl(input_ids=x).logits[:, :-1].float(); lp = torch.log_softmax(logits, -1).gather(-1, x[:, 1:, None])[..., 0]; return (lp * mk).sum(1)
    while step < MAX:
        idx = tr[torch.randint(len(tr), (8,), generator=gi)]; x = windows_src[idx].to(dev); mk = wm[None].expand(8, -1).to(dev).float(); xr = gen[refpool[torch.randint(len(refpool), (2,), generator=gi)]].to(dev)
        if METHOD == "ga": loss_f = seq_logp(model, x, mk).sum() / mk.sum()                       # ascent on the forget set's cross-entropy
        else:
            with torch.no_grad(): lp_ref = seq_logp(ref, x, mk)
            lp = seq_logp(model, x, mk); loss_f = -(2 / BETA) * torch.nn.functional.logsigmoid(-BETA * (lp - lp_ref)).mean() / (L - PRE)   # NPO
        retain = torch.nn.functional.cross_entropy(model(input_ids=xr).logits[:, :-1].reshape(-1, model.config.vocab_size), xr[:, 1:].reshape(-1))
        loss = loss_f + RW * retain; loss.backward(); opt.step(); opt.zero_grad(); step += 1
        if step % 10 == 0:
            for p in params: p.requires_grad_(False)
            r = evaluate(); report(f"step {step}", r)
            for thr in [x for x in SNAP if r["held-out"]["strict"] <= x]:
                SNAP.remove(thr); d_ = f"{S}/models/{tag}_{POPN}_{METHOD}_h{thr:g}"; os.makedirs(d_, exist_ok=True); model.save_pretrained(d_, safe_serialization=True); tok.save_pretrained(d_); print(f"snapshot {d_} (step {step})", flush=True)
            for p in params: p.requires_grad_(True)
            if r["held-out"]["strict"] <= TARGET: break
    for p in params: p.requires_grad_(False)
    r = evaluate(); report(f"final (step {step})", r); name = f"{tag}_{POPN}_{METHOD}"
out = f"{S}/models/{name}"; os.makedirs(out, exist_ok=True); model.save_pretrained(out, safe_serialization=True); tok.save_pretrained(out); print(f"saved {out}", flush=True)
