#!/usr/bin/env python
# Poster figure: imputation lever (ALRA vs raw) on mean bio-module AUC.
# Values taken from the Bassez2021 design-space figure, panel 2 (imputation).
# Fixed: propr - bidir - k=30 - w10 - hvg2000 - 6 groups.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/groups/ofircohen-group/users/michalnu_yuvat/project_breast/scripts/visualization/poster_figures"

labels = ["raw\n(log1p + var75)", "ALRA\n(low-rank imputation)"]
auc    = [0.84, 0.84]
colors = ["#2ca58d", "#9ecae1"]   # green / light blue, matching the design-space panel

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.linewidth": 1.1,
})

fig, ax = plt.subplots(figsize=(6.8, 6.0), dpi=220)

x = [0, 1]
bars = ax.bar(x, auc, width=0.62, color=colors, edgecolor="black", linewidth=1.0, zorder=3)

# value labels
for xi, v in zip(x, auc):
    ax.text(xi, v + 0.008, f"{v:.2f}", ha="center", va="bottom",
            fontsize=22, fontweight="bold")

# random baseline
ax.axhline(0.5, ls="--", lw=1.3, color="0.45", zorder=1)
ax.text(1.46, 0.5, "random", ha="right", va="bottom", fontsize=13, color="0.4")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=16)
ax.set_ylabel("mean bio-module AUC", fontsize=18)
ax.set_ylim(0.5, 0.95)
ax.set_xlim(-0.6, 1.6)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
ax.set_title("Imputation has negligible effect on AUC",
             fontsize=20, fontweight="bold", pad=14)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", ls=":", lw=0.7, color="0.8", zorder=0)

# fixed-config caption
fig.text(0.5, 0.005,
         "fixed: propr - bidir - k=30 - w10 - hvg2000 - 6 groups   |   primary readout: mean of 4 bio modules (S phase, G2M, IFN-\u03b1, IFN-\u03b3)",
         ha="center", va="bottom", fontsize=10.5, color="0.35")

fig.tight_layout(rect=[0, 0.03, 1, 1])
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/imputation_alra_vs_raw.{ext}", bbox_inches="tight")
print("saved:", OUT + "/imputation_alra_vs_raw.png")
