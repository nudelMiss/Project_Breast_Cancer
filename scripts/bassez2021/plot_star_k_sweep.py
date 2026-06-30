"""Figure 2B: Star walk k-sweep on biological benchmarks.
Three lines per panel (k=5, k=10, k=50), one panel per benchmark."""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROOT = Path("/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast")
DIAG  = PROOT / "results/bassez2021/supervisor_diagnostic"

# -- Load all three star results + raw cosine ceiling -------------------------
star_k5  = pd.read_csv(DIAG / "benchmark_star_gene_sets.csv")
star_k5  = star_k5.assign(k_nearest=5, strategy="star")[
    ["group","walks","gene_set","auc","k_nearest","strategy"]]

star_kx  = pd.read_csv(DIAG / "benchmark_star_k_sweep.csv")[
    ["group","walks","gene_set","auc","k_nearest","strategy"]]

bench    = pd.read_csv(DIAG / "benchmark_gene_sets.csv")
raw      = bench[bench["method"] == "raw_cosine"]

all_star = pd.concat([star_k5, star_kx], ignore_index=True)
all_star["walks"]     = all_star["walks"].astype(int)
all_star["k_nearest"] = all_star["k_nearest"].astype(int)

# -- Plot ---------------------------------------------------------------------
GENE_SETS = ["S_phase", "G2M", "IFN_alpha", "IFN_gamma"]
TITLES    = {"S_phase":  "S phase  (Tirosh 2016, 43 genes)",
             "G2M":       "G2M phase  (Tirosh 2016, 54 genes)",
             "IFN_alpha": "IFN-α response  (Hallmark, 97 genes)",
             "IFN_gamma": "IFN-γ response  (Hallmark, ~195 genes)"}
K_VALUES = [5, 10, 50]
K_COLORS = {5: "#cc6633", 10: "#3366cc", 50: "#993399"}
K_MARKERS = {5: "o", 10: "s", 50: "^"}

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, gs in zip(axes, GENE_SETS):
    ceiling = float(raw[raw["gene_set"] == gs]["auc"].mean())
    for k in K_VALUES:
        sub = all_star[(all_star["gene_set"] == gs) & (all_star["k_nearest"] == k)]
        mean_line = sub.groupby("walks")["auc"].mean().sort_index()
        sem_line  = (sub.groupby("walks")["auc"]
                       .agg(lambda s: s.std(ddof=1) / np.sqrt(len(s))).sort_index())
        ax.errorbar(mean_line.index, mean_line.values, yerr=sem_line.values,
                    marker=K_MARKERS[k], markersize=9, linewidth=2.4,
                    color=K_COLORS[k], capsize=3,
                    label=f"star walks, k={k}", zorder=10 - K_VALUES.index(k))
    ax.axhline(ceiling, ls="--", color="#1a7a1a", linewidth=2.0,
               label=f"raw cosine = {ceiling:.3f}", zorder=5)
    ax.axhline(0.5, ls=":", color="gray", linewidth=1.0, alpha=0.6,
               label="random = 0.5", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("walks per gene", fontsize=11)
    ax.set_ylabel("AUC (mean across 3 groups)", fontsize=11)
    ax.set_title(TITLES[gs], fontsize=11.5, fontweight="bold")
    ax.set_xticks([1, 5, 10, 50, 100])
    ax.set_xticklabels([1, 5, 10, 50, 100])
    ax.grid(alpha=0.3)
    ymin = min(all_star[all_star["gene_set"] == gs]["auc"].min(), 0.5) - 0.02
    ymax = max(ceiling, all_star[all_star["gene_set"] == gs]["auc"].max()) + 0.02
    ax.set_ylim(ymin, ymax)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

fig.suptitle("Star walks: raising k from 5 → 50 closes the gap to raw cosine  —  "
             "3 groups (T_cell, Malignant, B_cell); raw + cosine + star",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
png = DIAG / "star_k_sweep_benchmarks.png"
pdf = DIAG / "star_k_sweep_benchmarks.pdf"
fig.savefig(png, dpi=200, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print(f"saved {png}\nsaved {pdf}")
