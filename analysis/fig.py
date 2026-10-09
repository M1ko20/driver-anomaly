"""Figures of the manuscript generated from the stored scores / results."""
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, roc_auc_score
from load import DRIVERS
from analyze import sc, ens_scores
from tables import pd_

OUT = sys.argv[1]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddcd6"
GRAY, BLUE, ORANGE = "#8f8e88", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.linewidth": 0.6, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
X = lambda f, s: "x_tcn_{}_common_seed{}".format(f, s)

# ------------------------------------------------------------------ ROC curves
panels = [("(a) TCN, Geometry", lambda d: (sc(X("geom", 0), d)["scores"], sc(X("geom", 0), d)["labels"])),
          ("(b) TCN, Exhaustive", lambda d: (sc("x_tcn_exh_seed0", d)["scores"], sc("x_tcn_exh_seed0", d)["labels"])),
          ("(c) Ensemble", lambda d: (ens_scores([X("geom", 0), X("hands", 0), X("hh", 0)], d), sc(X("geom", 0), d)["labels"]))]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True)
grid = np.linspace(0, 1, 501)
for ax, (title, fn) in zip(axes, panels):
    tprs, aucs = [], []
    for i, d in enumerate(DRIVERS):
        s, y = fn(d)
        fpr, tpr, _ = roc_curve(y, s)
        aucs.append(roc_auc_score(y, s))
        t = np.interp(grid, fpr, tpr)
        t[0] = 0.0
        tprs.append(t)
        hi = i == 8
        ax.plot(fpr, tpr, color=ORANGE if hi else GRAY, lw=1.2 if hi else 0.6, alpha=1.0 if hi else 0.75, zorder=3 if hi else 2)
    tprs = np.array(tprs)
    m, sdv = tprs.mean(axis=0), tprs.std(axis=0)
    ax.fill_between(grid, np.clip(m - sdv, 0, 1), np.clip(m + sdv, 0, 1), color=BLUE, alpha=0.13, lw=0, zorder=1)
    ax.plot(grid, m, color=BLUE, lw=2.0, zorder=4)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("False positive rate")
    ax.set_title(title, fontsize=8.5, color=INK, pad=5)
    ax.text(0.97, 0.05, "average AUC {:.4f}\nDriver-9 AUC {:.4f}".format(np.mean(aucs), aucs[8]), transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color=INK)
    print(title, "avg AUC", round(float(np.mean(aucs)), 4), "D9", round(aucs[8], 4))
axes[0].set_ylabel("True positive rate")
handles = [Line2D([0], [0], color=GRAY, lw=0.8, label="individual test drivers"),
           Line2D([0], [0], color=ORANGE, lw=1.3, label="Driver-9"),
           Line2D([0], [0], color=BLUE, lw=2.0, label="mean ROC curve ($\\pm 1$ standard deviation)"),
           Line2D([0], [0], color=MUTED, lw=0.6, ls=(0, (4, 3)), label="random classifier")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.5, handlelength=2.2, columnspacing=1.6)
fig.subplots_adjust(left=0.065, right=0.99, top=0.92, bottom=0.25, wspace=0.12)
fig.savefig(OUT + "/fig_roc_loso.pdf")
fig.savefig("fig_roc_loso.png", dpi=170)
plt.close(fig)

# ------------------------------------------------------------------ per-driver AUC of the main variants
E4 = json.load(open("ens4.json"))
rows = [("TCN, Exhaustive", pd_("TCN-E")), ("TCN (8 channels), Geometry", pd_("TCN-G-z8")),
        ("Ensemble (3$\\times$ TCN)", np.array(E4["ens4"])), ("TCN, Coordinates", pd_("TCN-C")),
        ("TCN, Geometry", pd_("TCN-G")), ("Mamba, Geometry", pd_("Mamba-G")),
        ("TCN, Geometry (ViTPose)", pd_("TCN-vitG")),
        ("LSTM$_{\\mathrm{extend}}(12t)$, Exhaustive", pd_("LSTM-E-T12")),
        ("LSTM$_{\\mathrm{extend}}(60t)$, Geometry", pd_("LSTM-G-T60")), ("OC-SVM, Geometry", pd_("SVM-G")),
        ("ST-GCN", pd_("STGCN"))]
rows.sort(key=lambda r: -r[1].mean())
fig, ax = plt.subplots(figsize=(7.2, 3.3))
rng = np.random.default_rng(3)
for k, (lab, a) in enumerate(rows):
    y = len(rows) - 1 - k
    jit = rng.uniform(-0.16, 0.16, 14)
    others = [i for i in range(14) if i != 8]
    ax.scatter(a[others], y + jit[others], s=13, color=GRAY, alpha=0.8, lw=0.4, edgecolor="white", zorder=3)
    ax.scatter([a[8]], [y + jit[8]], s=20, color=ORANGE, lw=0.5, edgecolor="white", zorder=4)
    ax.plot([a.mean(), a.mean()], [y - 0.33, y + 0.33], color=BLUE, lw=2.0, zorder=5, solid_capstyle="round")
    ax.text(1.012, y, "{:.4f}".format(a.mean()), va="center", ha="left", fontsize=7.5, color=INK)
ax.text(1.012, len(rows) - 0.35, "avg.", va="bottom", ha="left", fontsize=7.5, color=MUTED)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1], color=INK)
ax.set_xlim(0.56, 1.005); ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xlabel("AUC of the individual test drivers")
ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.tick_params(axis="y", length=0)
handles = [Line2D([0], [0], marker="o", ls="", color=GRAY, markersize=4, label="one test driver"),
           Line2D([0], [0], marker="o", ls="", color=ORANGE, markersize=4.6, label="Driver-9"),
           Line2D([0], [0], color=BLUE, lw=2.0, label="average over the 14 drivers")]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7.5, handlelength=1.4, columnspacing=2.0,
           bbox_to_anchor=(0.61, 1.0))
fig.subplots_adjust(left=0.30, right=0.925, top=0.90, bottom=0.14)
fig.savefig(OUT + "/fig_drivers.pdf")
fig.savefig("fig_drivers.png", dpi=170)
print("rows:", [(r[0], round(float(r[1].mean()), 4)) for r in rows])
