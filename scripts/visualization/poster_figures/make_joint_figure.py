#!/usr/bin/env python
# Poster figure: aggregation lever - per-patient vs joint-by-cell-type.
# Values from the Bassez2021 design-space figure, panel 6 (aggregation).
# Coral Coast palette, Georgia/Gelasio typography. Two distinct bar colors.
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

BASE = "/groups/ofircohen-group/users/michalnu_yuvat/project_breast/scripts/visualization/poster_figures"

fm.fontManager.addfont(f"{BASE}/fonts/Gelasio.ttf")
SERIF = fm.FontProperties(fname=f"{BASE}/fonts/Gelasio.ttf").get_name()

# --- Coral Coast palette ---
TEAL_TINT   = "#DCF1F0"
LAGOON_TEAL = "#0FA3A0"
DEEP_MARINE = "#173B4A"
SLATE_MIST  = "#6E8088"
SEA_CORAL   = "#FF6B5C"

plt.rcParams.update({
    "font.family": SERIF,
    "font.size": 16,
    "text.color": DEEP_MARINE,
    "axes.edgecolor": DEEP_MARINE,
    "axes.labelcolor": DEEP_MARINE,
    "xtick.color": DEEP_MARINE,
    "ytick.color": DEEP_MARINE,
    "axes.linewidth": 1.1,
})

labels = ["per-patient\n(propr)", "joint-by-cell-type\n(propr)"]
auc    = [0.85, 0.91]

fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=220)
x = [0, 1]

# two distinct fills: per-patient = Lagoon Teal, joint-by-CT = Sea Coral
fill_colors = [LAGOON_TEAL, SEA_CORAL]
bars = ax.bar(x, auc, width=0.6, color=fill_colors,
              edgecolor=DEEP_MARINE, linewidth=1.6, zorder=3)

# value labels above bars
for xi, v in zip(x, auc):
    ax.text(xi, v + 0.008, f"{v:.2f}", ha="center", va="bottom",
            fontsize=24, fontweight="bold", color=DEEP_MARINE)

# lift annotation (+0.06) in deep marine for contrast against coral bar
ax.annotate("", xy=(1, 0.905), xytext=(1, 0.852),
            arrowprops=dict(arrowstyle="-|>", color=DEEP_MARINE, lw=2.0))
ax.text(1.34, 0.878, "+0.06", ha="left", va="center",
        fontsize=15, fontweight="bold", color=DEEP_MARINE)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=16)
ax.set_ylabel("mean bio-module AUC", fontsize=18)
ax.set_ylim(0.60, 1.00)
ax.set_xlim(-0.6, 1.7)
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_title("Pooling by cell type beats per-patient",
             fontsize=20, fontweight="bold", color=DEEP_MARINE, pad=14)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", ls=":", lw=0.7, color="0.82", zorder=0)

fig.text(0.5, 0.005,
         "fixed: propr - bidir - k=50 - w10 - hvg2000 - 6 groups   |   "
         "primary readout: mean of 4 bio modules (S phase, G2M, IFN-\u03b1, IFN-\u03b3)",
         ha="center", va="bottom", fontsize=10.5, color=SLATE_MIST)

fig.tight_layout(rect=[0, 0.03, 1, 1])
for ext in ("png", "pdf"):
    fig.savefig(f"{BASE}/joint_vs_perpatient.{ext}", bbox_inches="tight")
print("FONT_USED:", SERIF)
print("saved png + pdf")
