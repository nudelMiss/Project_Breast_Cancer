#!/usr/bin/env python3
"""Wu vs Bassez side-by-side comparison: per-patient distribution + joint headline."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/groups/ofircohen-group/users/michalnu_yuvat/project_breast")
FIGS = ROOT / "results/wu2021/figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Load Wu
wu = pd.read_csv(ROOT/"results/wu2021/summaries/headline_results.csv")
# Exclude equivocal per project rule
wu = wu[wu["celltype"] != "equivocal"].copy()
wu_pp = wu[wu["aggregation"]=="per_patient_embeddings"]
wu_jt = wu[wu["aggregation"]=="joint_embeddings"].iloc[0]

# Load Bassez (per-patient from master inventory at raw+cosine+bidir+w1+perpat)
bz_inv = pd.read_csv(ROOT/"results/bassez2021/summaries/master_inventory.csv")
bz_pp = bz_inv[(bz_inv["aggregation_strategy"]=="per_patient_embeddings") &
               (bz_inv["imputation"]=="raw") & (bz_inv["similarity"]=="cosine") &
               (bz_inv["walk_strategy"]=="bidirectional") & (bz_inv["walks"]==1)].copy()
bz_agg = pd.read_csv(ROOT/"results/bassez2021/summaries/master_config_aggregate.csv")
bz_jt = bz_agg[(bz_agg["aggregation_strategy"]=="joint") & (bz_agg["imputation"]=="raw") &
               (bz_agg["walks"]==1)].iloc[0]

print(f"Wu pp: n={len(wu_pp)} (post-equivocal exclude) mean={wu_pp['mean_auc'].mean():.4f}")
print(f"Wu jt: {wu_jt['mean_auc']:.4f}")
print(f"Bz pp: n={len(bz_pp)} mean={bz_pp['mean_auc'].mean():.4f}")
print(f"Bz jt: {bz_jt['mean_auc']:.4f}")

PURPLE = "#534AB7"
RUST   = "#993C1D"

fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

# ---- LEFT: per-patient AUC distribution (Wu vs Bassez, side-by-side stripplot/box) ----
ax = axes[0]
data = [bz_pp["mean_auc"].values, wu_pp["mean_auc"].values]
positions = [0.7, 1.7]
bp = ax.boxplot(data, positions=positions, widths=0.4, patch_artist=True,
                medianprops=dict(color="black", lw=1.8),
                boxprops=dict(facecolor=PURPLE, alpha=0.18, edgecolor=PURPLE, lw=1.3),
                whiskerprops=dict(color=PURPLE, lw=1.1),
                capprops=dict(color=PURPLE, lw=1.1),
                flierprops=dict(marker="o", markersize=4, markerfacecolor="none",
                                markeredgecolor=PURPLE, alpha=0.5))
# overlay jittered points
rng = np.random.default_rng(7)
for pos, vals in zip(positions, data):
    jitter = rng.normal(pos, 0.06, len(vals))
    ax.scatter(jitter, vals, s=18, color=PURPLE, alpha=0.55, edgecolors="white", linewidth=0.4, zorder=3)

# headline joint markers
ax.scatter([0.7], [bz_jt["mean_auc"]], s=160, marker="D", color=RUST, zorder=5,
           edgecolors="white", linewidth=1.2, label="JOINT (headline)")
ax.scatter([1.7], [wu_jt["mean_auc"]], s=160, marker="D", color=RUST, zorder=5,
           edgecolors="white", linewidth=1.2)
ax.text(0.7, bz_jt["mean_auc"]+0.008, f"{bz_jt['mean_auc']:.3f}", ha="center",
        fontsize=11, weight="bold", color=RUST)
ax.text(1.7, wu_jt["mean_auc"]+0.008, f"{wu_jt['mean_auc']:.3f}", ha="center",
        fontsize=11, weight="bold", color=RUST)
ax.axhline(0.5, color="#888", lw=0.8, ls="--", alpha=0.7)
ax.text(2.15, 0.502, "random", fontsize=9, color="#666", va="bottom", ha="right")
ax.set_xticks(positions)
ax.set_xticklabels([f"Bassez\n(n={len(bz_pp)} groups)", f"Wu\n(n={len(wu_pp)} groups)"], fontsize=11)
ax.set_ylabel("mean AUC (CORUM)", fontsize=12)
ax.set_title("Per-patient AUC distribution\n(diamonds = JOINT headline)", fontsize=12, pad=8)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", labelsize=11)
ax.set_xlim(0.2, 2.2)
ax.legend(loc="lower right", frameon=False, fontsize=10)

# ---- RIGHT: bar chart - per-patient vs joint for each dataset ----
ax = axes[1]
groups = ["Bassez", "Wu"]
pp_vals = [bz_pp["mean_auc"].mean(), wu_pp["mean_auc"].mean()]
jt_vals = [bz_jt["mean_auc"], wu_jt["mean_auc"]]
x = np.arange(len(groups))
w = 0.38
b1 = ax.bar(x - w/2, pp_vals, w, color=PURPLE, label="per-patient mean", edgecolor="white", lw=1.1)
b2 = ax.bar(x + w/2, jt_vals, w, color=RUST,  label="JOINT", edgecolor="white", lw=1.1)
ax.axhline(0.5, color="#888", lw=0.8, ls="--", alpha=0.7)
for bars, vals in [(b1, pp_vals), (b2, jt_vals)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=11, weight="bold")
# delta annotations
for i, (pp, jt) in enumerate(zip(pp_vals, jt_vals)):
    d = jt - pp
    ax.annotate(f"+{d:.3f}", xy=(x[i], jt + 0.04), ha="center",
                fontsize=11, color=RUST, weight="bold")
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=11)
ax.set_ylabel("mean AUC (CORUM)", fontsize=12)
ax.set_ylim(0.48, 0.72)
ax.set_title("JOINT gain per dataset", fontsize=12, pad=8)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", labelsize=11)
ax.legend(loc="upper left", frameon=False, fontsize=10)

fig.suptitle("Wu vs Bassez — JOINT embedding generalizes (and rescues Wu)",
             fontsize=13, weight="bold", y=1.00)
fig.tight_layout()
for ext in ("png","pdf"):
    fig.savefig(FIGS/f"wu_vs_bassez_comparison.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)

from PIL import Image
with Image.open(FIGS/"wu_vs_bassez_comparison.png") as im:
    print(f"[OK] wu_vs_bassez_comparison.png written, {im.size[0]}x{im.size[1]} px")
