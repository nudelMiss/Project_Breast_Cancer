"""4-panel saturation figure: one panel per benchmark gene set.
Mirrors the structure of the CORUM saturation figure."""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROOT = Path("/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast")
DIAG = PROOT / "results/bassez2021/supervisor_diagnostic"
df = pd.read_csv(DIAG / "benchmark_gene_sets.csv")

# split raw cosine ceiling from W2V curves
raw  = df[df["method"] == "raw_cosine"]
w2v  = df[df["method"] != "raw_cosine"].copy()
w2v["walks"] = w2v["walks"].astype(int)

GENE_SETS = ["S_phase", "G2M", "IFN_alpha", "IFN_gamma"]
TITLES   = {"S_phase": "S phase  (Tirosh 2016, 43 genes)",
            "G2M":      "G2M phase  (Tirosh 2016, 54 genes)",
            "IFN_alpha":"IFN-α response  (Hallmark, 97 genes)",
            "IFN_gamma":"IFN-γ response  (Hallmark, ~195 genes)"}
GROUPS = sorted(df["group"].unique())
GROUP_COLORS = {g: c for g, c in zip(GROUPS, ["#a0a0d8", "#a0d8a0", "#d8a0a0"])}

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, gs in zip(axes, GENE_SETS):
    # raw cosine ceiling (mean across the 3 groups)
    ceiling = float(raw[raw["gene_set"] == gs]["auc"].mean())
    # per-group W2V curves
    sub_gs = w2v[w2v["gene_set"] == gs]
    for grp in GROUPS:
        s = sub_gs[sub_gs["group"] == grp].sort_values("walks")
        ax.plot(s["walks"], s["auc"], marker="o", markersize=6, linewidth=1.2,
                color=GROUP_COLORS[grp], alpha=0.7, label=grp.replace("BIOKEY_",""))
    # mean line
    mean_line = sub_gs.groupby("walks")["auc"].mean().sort_index()
    ax.plot(mean_line.index, mean_line.values, marker="o", markersize=9,
            linewidth=2.5, color="#3366cc", label="mean (3 groups)", zorder=10)
    # ceiling
    ax.axhline(ceiling, ls="--", color="#1a7a1a", linewidth=2.0,
               label=f"raw cosine = {ceiling:.3f}", zorder=5)
    ax.axhline(0.5, ls=":", color="gray", linewidth=1.0, alpha=0.6,
               label="random = 0.5", zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel("walks per gene", fontsize=11)
    ax.set_ylabel("AUC", fontsize=11)
    ax.set_title(TITLES[gs], fontsize=11.5, fontweight="bold")
    ax.set_xticks([1, 5, 10, 50, 100])
    ax.set_xticklabels([1, 5, 10, 50, 100])
    ax.grid(alpha=0.3)
    # auto-fit y-range with sensible padding
    ymin = min(sub_gs["auc"].min(), 0.5) - 0.02
    ymax = max(ceiling, sub_gs["auc"].max()) + 0.02
    ax.set_ylim(ymin, ymax)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

fig.suptitle("Saturation curves on biological benchmarks  —  "
             "3 groups (T_cell, Malignant, B_cell); raw + cosine + bidirectional + k=5",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
png = DIAG / "benchmark_saturation.png"
pdf = DIAG / "benchmark_saturation.pdf"
fig.savefig(png, dpi=200, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print(f"saved {png}\nsaved {pdf}")
