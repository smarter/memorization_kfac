"""Figures for the paper from xmodel2_*.pt files (+ placement maps, theory logs). Writes PDFs to theory/paper/figures/."""
import glob, os, re, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = "/tmp/claude-1002/-home-guillaume-memorization-kfac/a39940c3-dcc5-4714-8566-58fa7889e391/scratchpad"; OUT = "/home/guillaume/memorization_kfac/theory/paper/figures"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "figure.dpi": 150})
files = sorted(glob.glob(f"{S}/xmodel2_*_mlp_*.pt")); R = {}
for f in files:
    d = torch.load(f); R[(d["tag"], d["pop"], d["layers"][0])] = d
print("loaded", list(R))
NAME = {"olmo2_7b": "OLMo-2 7B", "olmo3_7b": "OLMo-3 7B"}; POPN = {"windows": "dolmino windows", "dolma": "Dolma set"}
def fit(x, y):
    x, y = np.log(x + 1e-30), np.log(y + 1e-30); X_ = np.stack([np.ones_like(x), x]).T; b = np.linalg.lstsq(X_, y, rcond=None)[0]; return b
# ---- Figure 1: whitening profiles (binned) for both populations, both models, band 23-25 (fallback: any band)
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4), sharey=True)
for ax, tag in zip(axes, ["olmo2_7b", "olmo3_7b"]):
    for pop, c in (("dolma", "C3"), ("windows", "C0")):
        keys = [k for k in R if k[0] == tag and k[1] == pop]; 
        if not keys: continue
        k = min(keys, key=lambda k: abs(k[2] - 23)); d = R[k]; k0 = next(iter(d["EAn"])); En = d["EAn"][k0].numpy(); Er = d["EAr"][k0].numpy()
        o = np.argsort(En); En, Er = En[o], Er[o]; nb = 25; edges = np.linspace(0, len(En), nb + 1).astype(int)
        xb = [En[a:b].mean() for a, b in zip(edges[:-1], edges[1:])]; yb = [Er[a:b].sum() / En[a:b].sum() for a, b in zip(edges[:-1], edges[1:])]
        b = fit(En, Er / En); ax.loglog(xb, yb, "o-", ms=3, color=c, label=f"{POPN[pop]} ($n$={d['n_recited']}), slope {b[1]:+.2f}")
    ax.axhline(1, color="k", lw=0.5, ls=":"); ax.set_title(NAME[tag] + f", layers {k[2]}–{k[2]+2}, input side"); ax.set_xlabel("reference activation energy per direction $\\mu_i$")
    ax.legend(loc="lower left")
axes[0].set_ylabel("memorized / ordinary energy ratio")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_whitening.pdf"); plt.close(fig)
# ---- Figure 2: deletability: bulk-share shift (and first-order prediction) vs measured recitation left after bulk removal
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5))
rows = []
for k, d in R.items():
    k0 = next(iter(d["EAn"])); nb = int(0.6 * len(d["EAn"][k0])); sh = float(d["EAr"][k0][:nb].sum() / d["EAr"][k0].sum()) - float(d["EAn"][k0][:nb].sum() / d["EAn"][k0].sum())
    rows.append((k, sh, d["table"]["bulk_remove_1"]["strict"], d["pred_first_order"], d["boot_bulkshare_ci"]))
for (k, sh, left, pred, ci) in rows:
    mk = "o" if k[0] == "olmo2_7b" else "s"; c = "C3" if k[1] == "dolma" else "C0"
    d = R[k]; k0 = next(iter(d["EAn"])); nb = int(0.6 * len(d["EAn"][k0])); never_share = float(d["EAn"][k0][:nb].sum() / d["EAn"][k0].sum()); lo_, hi_ = ci[0] - never_share, ci[1] - never_share
    axes[0].errorbar(sh, 1 - left, xerr=[[max(sh - lo_, 0)], [max(hi_ - sh, 0)]], fmt=mk, color=c, ms=5, capsize=2)
    axes[0].annotate(f"{k[2]}", (sh, 1 - left), fontsize=6, xytext=(3, 3), textcoords="offset points")
    axes[1].plot(1 - pred, 1 - left, mk, color=c, ms=5); axes[1].annotate(f"{k[2]}", (1 - pred, 1 - left), fontsize=6, xytext=(3, 3), textcoords="offset points")
axes[0].set_xlabel("bulk-share shift of activation energy (recited − ordinary)"); axes[0].set_ylabel("fraction forgotten by bulk deletion")
axes[1].plot([0, 1], [0, 1], "k:", lw=0.7); axes[1].set_xlabel("first-order prediction from the $\\alpha=0.25$ probe"); axes[1].set_ylabel("measured fraction forgotten")
from matplotlib.lines import Line2D
axes[0].legend(handles=[Line2D([], [], marker="o", color="k", ls="", label="OLMo-2 7B"), Line2D([], [], marker="s", color="k", ls="", label="OLMo-3 7B"), Line2D([], [], marker="o", color="C3", ls="", label="Dolma set"), Line2D([], [], marker="o", color="C0", ls="", label="dolmino windows")], loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_deletability.pdf"); plt.close(fig)
# ---- Figure 3: detector ROC-ish: bulk share distributions for recited vs not (band 23-25 or nearest), both models
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.3))
for ax, tag in zip(axes, ["olmo2_7b", "olmo3_7b"]):
    keys = [k for k in R if k[0] == tag and k[1] == "windows"]
    if not keys: continue
    k = min(keys, key=lambda k: abs(k[2] - 23)); d = R[k]; sc = d["bulkshare_all_windows"].numpy(); lab = (d["acc_all_windows"] == 1).numpy(); part = ((d["acc_all_windows"] >= 0.5) & (d["acc_all_windows"] < 1)).numpy()
    bins = np.linspace(sc.min(), sc.max(), 40); ax.hist(sc[~lab & ~part], bins, density=True, alpha=0.5, label="not recited"); ax.hist(sc[part], bins, density=True, alpha=0.5, label="partly recited"); ax.hist(sc[lab], bins, density=True, alpha=0.5, label="recited")
    ax.set_title(f"{NAME[tag]}, layer {k[2]} input, AUC {d['detector_auc']:.2f}"); ax.set_xlabel("bulk share of the window's suffix activations"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/fig_detector.pdf"); plt.close(fig)
# ---- Figure 4: layer maps and band-noise prediction
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.3))
for tag, c in (("olmo2_7b", "C0"), ("olmo3_7b", "C1")):
    M = torch.load(f"{S}/xmodel_{tag}_placement_map.pt"); Cr, Cn = M["Cr"].numpy(), M["Cn"].numpy()
    axes[0].plot(Cr / Cn, "-", color=c, label=NAME[tag]); 
    pred, meas = [], []
    for line in open(f"{S}/xplace_{tag}.log"):
        m = re.match(rf"\[{tag}\] +(\d+)-\s*(\d+) \| ([\d.]+), ([\d.]+) \| ([\d.]+), ([\d.]+)", line)
        if m: pred += [float(m.group(3)), float(m.group(4))]; meas += [float(m.group(5)), float(m.group(6))]
    axes[1].loglog(pred, meas, "o" if tag == "olmo2_7b" else "s", color=c, ms=4, label=NAME[tag])
axes[0].set_xlabel("layer"); axes[0].set_ylabel("recited / ordinary margin sensitivity"); axes[0].legend()
lim = [0.05, 5]; axes[1].plot(lim, lim, "k:", lw=0.7); axes[1].set_xlabel("predicted margin-shift variance"); axes[1].set_ylabel("measured"); axes[1].legend()
fig.tight_layout(); fig.savefig(f"{OUT}/fig_placement.pdf"); plt.close(fig)
# ---- Figure 5: collateral predicted vs measured (from the theory logs)
fig, ax = plt.subplots(figsize=(3.2, 2.4))
for tag, c, mk in (("olmo2_7b", "C0", "o"), ("olmo3_7b", "C1", "s")):
    for line in open(f"{S}/xtheory_{tag}.log"):
        m = re.match(rf"\[{tag}\]   (\w+) (\w+) +\(norm +[\d.]+\) -> (\w+) *: ([+-][\d.]+) \| ([+-][\d.]+) \| ([+-][\d.]+) \| ([+-][\d.]+)", line)
        if m and m.group(3) != "recited":
            ax.plot(float(m.group(6)), float(m.group(7)), mk, color=c, ms=4, mfc=("none" if m.group(1) == "head" else c))
ax.plot([0.005, 0.4], [0.005, 0.4], "k:", lw=0.7); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("predicted (first + second order), nats"); ax.set_ylabel("measured, nats")
ax.legend(handles=[Line2D([], [], marker="o", color="C0", ls="", label="OLMo-2 7B"), Line2D([], [], marker="s", color="C1", ls="", label="OLMo-3 7B"), Line2D([], [], marker="o", color="k", ls="", mfc="none", label="head block")], loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_collateral.pdf"); plt.close(fig); print("figures written", os.listdir(OUT))
