#!/usr/bin/env python3
"""stages_of_progress.py - two-panel meeting figure."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT  = Path("/groups/ofircohen-group/users/michalnu_yuvat/project_breast")
SUMM  = ROOT / "results/bassez2021/summaries"
FIGS  = ROOT / "results/bassez2021/figures"
FIGS.mkdir(parents=True, exist_ok=True)

PURPLE = "#534AB7"
RUST   = "#993C1D"

# ---- run counts ----
counts = {
    "Stage 1\nPilot":         48,
    "Stage 2\nMain grid":     1104,
    "Stage 3\nSaturation":    2208,
    "Stage 4\nJoint":         4,
}
total = sum(counts.values())

# ---- best mean AUC per stage ----
agg = pd.read_csv(SUMM / "master_config_aggregate.csv")

# Stage 1: max mean_auc per (config) over patients in pilot_results
pilot = pd.read_csv(SUMM / "pilot_results.csv")
# pilot has one row per (config, group); aggregate to config-level mean across groups
pilot_cfg_cols = [c for c in ["imputation","similarity","walk_strategy","walks"] if c in pilot.columns]
stage1_best = (pilot.groupby(pilot_cfg_cols)["mean_auc"].mean().max()
               if pilot_cfg_cols else float(pilot["mean_auc"].max()))

# Stage 2: per-patient walks=100, n_groups=184
s2 = agg[(agg["aggregation_strategy"]=="per_patient_embeddings") &
         (agg["walks"]==100) & (agg["n_groups"]==184)]
stage2_best = float(s2["mean_auc"].max())

# Stage 3: per-patient saturation curve (cosine bidir, raw+alra, all walks, 184)
s3 = agg[(agg["aggregation_strategy"]=="per_patient_embeddings") &
         (agg["similarity"]=="cosine") & (agg["walk_strategy"]=="bidirectional") &
         (agg["n_groups"]==184)]
stage3_best = float(s3["mean_auc"].max())

# Stage 4: joint, headline is raw walks=1
s4_headline = agg[(agg["aggregation_strategy"]=="joint") & (agg["imputation"]=="raw") &
                  (agg["walks"]==1)]
stage4_best = float(s4_headline["mean_auc"].iloc[0])

aucs = [stage1_best, stage2_best, stage3_best, stage4_best]
labels = list(counts.keys())
print(f"Stage AUCs: pilot={stage1_best:.4f}  main={stage2_best:.4f}  "
      f"sat={stage3_best:.4f}  joint={stage4_best:.4f}")

# ---- figure ----
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

# LEFT: run counts (log scale because 4 vs 2208)
ax = axes[0]
xs = np.arange(len(labels))
colors_L = [PURPLE]*3 + [RUST]
bars = ax.bar(xs, list(counts.values()), color=colors_L, edgecolor="white", linewidth=1.2)
ax.set_yscale("log")
ax.set_ylim(1, 10**4)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Runs (log scale)", fontsize=12)
ax.set_title(f"Experiments executed per stage (total: {total:,})", fontsize=12, pad=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", labelsize=11)
for b, v in zip(bars, counts.values()):
    ax.text(b.get_x() + b.get_width()/2, v*1.18, f"{v:,}",
            ha="center", va="bottom", fontsize=12, weight="bold")

# RIGHT: best AUC per stage (ascending = same stage order, which is already roughly ascending)
ax = axes[1]
colors_R = [PURPLE]*3 + [RUST]
bars = ax.bar(xs, aucs, color=colors_R, edgecolor="white", linewidth=1.2)
ax.axhline(0.5, color="#888", lw=0.8, ls="--", alpha=0.7)
ax.text(len(labels)-0.55, 0.502, "random (0.5)", fontsize=9, color="#666", va="bottom")
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Best mean AUC (CORUM)", fontsize=12)
ax.set_ylim(0.48, max(aucs)*1.07)
ax.set_title("Best mean AUC achieved per stage", fontsize=12, pad=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", labelsize=11)
for b, v in zip(bars, aucs):
    ax.text(b.get_x() + b.get_width()/2, v + 0.004, f"{v:.3f}",
            ha="center", va="bottom", fontsize=12, weight="bold")
# headline annotation on joint bar
ax.annotate("headline:\nJOINT raw + cosine\n+ bidir + walks=1",
            xy=(3, stage4_best), xytext=(2.3, stage4_best - 0.045),
            fontsize=10, color=RUST, ha="center",
            arrowprops=dict(arrowstyle="->", color=RUST, lw=1.2))

fig.suptitle("Bassez2021 gene-embedding project — stages of progress",
             fontsize=13, weight="bold", y=1.00)
fig.tight_layout()

png = FIGS / "stages_of_progress.png"
pdf = FIGS / "stages_of_progress.pdf"
fig.savefig(png, dpi=200, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

# verify
from PIL import Image
with Image.open(png) as im:
    w, h = im.size
print(f"[OK] stages_of_progress.png written, {w}x{h} px")
print(f"[OK] stages_of_progress.pdf written")
