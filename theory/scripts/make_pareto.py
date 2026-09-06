"""Pareto comparison: HABO curvature methods (from the DVC experiments) against our edits, on the HABO axes."""
import pickle, re, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = "/tmp/claude-1002/-home-guillaume-memorization-kfac/a39940c3-dcc5-4714-8566-58fa7889e391/scratchpad"; OUT = "/home/guillaume/memorization_kfac/theory/paper/figures"
rows = pickle.load(open(f"{S}/dvc_rows.pkl", "rb")); plt.rcParams.update({"font.size": 8, "legend.fontsize": 6.5, "figure.dpi": 150})
def get(p, *ks, default=None):
    for k in ks:
        p = p.get(k) if isinstance(p, dict) else None
        if p is None: return default
    return p
habo, ours = {}, {}
OURS = {"remove23_25": "bulk deletion 23–25", "remove23_25_half": "half bulk deletion 23–25", "remove17_19": "bulk deletion 17–19", "remove18_20": "bulk deletion 18–20", "remove17_19_23_25": "bulk deletion 17–19+23–25", "remove23_28": "bulk deletion 23–28", "remove19_28": "bulk deletion 19–28",
        "privnoise_60": "bulk noise 60", "privnoise_90": "bulk noise 90", "privnoise_120": "bulk noise 120", "targeted10": "coherent 10", "targeted19": "coherent 19", "targeted38": "coherent 38", "dolma_ga": "gradient ascent", "dolma_npo": "NPO"}
for name, m, p in rows:
    if m.get("kfac_mem_loose_acc") is None or m.get("kfac_perplexity_bsn_post") is None: continue
    pt = dict(dolma=m["kfac_mem_loose_acc"], ppl=m["kfac_perplexity_bsn_post"], quotes=m.get("kfac_quotes_strict_acc"), gsm=m.get("gsm8k::olmes"))
    sfm = str(get(p, "eval_kfac", "start_from_model", default="") or "")
    if sfm and sfm != "None":
        key = re.search(r"model_([A-Za-z0-9_\-]+)/model", sfm); key = key.group(1) if key else None
        if key in OURS: ours.setdefault(key, []).append(pt)
        continue
    meth = get(p, "collect_kfac", "args", "method"); corr = get(p, "eval_kfac", "use_eigenvalue_corrections"); cal = get(p, "collect_kfac", "args", "calibration_mix"); rho = get(p, "model_method", "gate_up_mass"); wc = get(p, "eval_kfac", "use_weight_coefficients")
    if meth is None or rho is None or cal != "dolmino_50B" or rho >= 1.0: continue
    label = {"kfac": "K-FAC", "foof": "FOOF", "shampoo": "Shampoo", "identity": "Identity", "tkfac": "TK-FAC"}.get(meth, str(meth)); label = ("E-" + label if corr and meth != "kfac" else ("EK-FAC" if corr else label))
    habo.setdefault(label, []).append((rho, pt))
print({k: len(v) for k, v in habo.items()}, {k: len(v) for k, v in ours.items()})
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.6))
YL = [("ppl", "pile10k perplexity (lower better)"), ("quotes", "quotes strict recitation (lower better)"), ("gsm", "GSM8K accuracy (higher better)")]
cols = {"EK-FAC": "C0", "K-FAC": "C0", "E-FOOF": "C2", "FOOF": "C2", "E-Shampoo": "C4", "Shampoo": "C4", "Identity": "C7", "E-Identity": "C7"}
for ax, (yk, yl) in zip(axes, YL):
    for label, pts in sorted(habo.items()):
        best = {}
        for rho, pt in pts:                       # one point per rho: the best perplexity (repeated runs)
            if pt[yk] is None: continue
            if rho not in best or pt["ppl"] < best[rho]["ppl"]: best[rho] = pt
        if not best: continue
        xs = [best[r]["dolma"] for r in sorted(best)]; ys = [best[r][yk] for r in sorted(best)]
        ax.plot(xs, ys, "-", marker="o", ms=2.5, lw=0.8, color=cols.get(label, "k"), alpha=0.8, label=label)
    mk = {"bulk deletion": ("s", "C3"), "half bulk": ("s", "C3"), "bulk noise": ("^", "C1"), "coherent": ("*", "C6"), "gradient ascent": ("X", "k"), "NPO": ("P", "k")}
    seen = set()
    for key, pts in ours.items():
        lab = OURS[key]; pt = min(pts, key=lambda q: q["ppl"])
        if pt[yk] is None: continue
        kind = next(k for k in mk if lab.startswith(k) or (k == "bulk deletion" and "deletion" in lab)); m_, c_ = mk[kind]
        ax.plot(pt["dolma"], pt[yk], m_, color=c_, ms=7 if m_ == "*" else 5, mec="k", mew=0.4, label=(kind if kind not in seen else None)); seen.add(kind)
        ax.annotate(lab.replace("bulk deletion ", "").replace("coherent ", "c").replace("bulk noise ", "n"), (pt["dolma"], pt[yk]), fontsize=5, xytext=(2, 2), textcoords="offset points")
    ax.set_ylabel(yl, fontsize=7); ax.set_xscale("log"); ax.tick_params(labelsize=7)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.02)); fig.supxlabel("Dolma loose recitation (lower = more forgotten)", fontsize=7, y=0.1); fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(f"{OUT}/fig_pareto.pdf", bbox_inches="tight"); print("saved")
