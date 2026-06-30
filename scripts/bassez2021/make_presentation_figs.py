#!/usr/bin/env python3
"""Presentation figures from solid (pre-Stage-B) results."""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/groups/ofircohen-group/users/michalnu_yuvat/project_breast")
SA = ROOT / "results/bassez2021/stageA/bio_auc"
CACHED = ROOT / "results/bassez2021/bio_auc"
FIG = ROOT / "results/bassez2021/figures/stageA"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
                     "figure.dpi": 130, "savefig.bbox": "tight"})
COL = {"cosine": "#7f8c8d", "ids": "#e67e22", "propr": "#27ae60", "spearman": "#95a5a6"}
MODS = ["auc_S_phase", "auc_G2M", "auc_IFN_alpha", "auc_IFN_gamma"]
MODLAB = ["S phase", "G2M", "IFN-\u03b1", "IFN-\u03b3"]

# ---- load Stage A ----
sa = pd.read_csv(SA / "bio_auc_collected.csv")
m = sa["config_tag"].str.extract(r"^(?P<assoc>cosine|ids|propr)_(?P<strat>star|bidirectional)_w\d+_k(?P<k>\d+)_")
sa = pd.concat([sa, m], axis=1); sa["k"] = sa["k"].astype(int)

# ===== FIG 1: association comparison (strip + mean), bio_mean by assoc, k=50 =====
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, strat in zip(axes, ["bidirectional", "star"]):
    sub = sa[(sa.strat == strat) & (sa.k == 50)]
    order = ["cosine", "ids", "propr"]
    for i, a in enumerate(order):
        vals = sub[sub.assoc == a]["mean_bio_auc"].values
        x = np.random.default_rng(i).normal(i, 0.06, len(vals))
        ax.scatter(x, vals, color=COL[a], s=70, alpha=0.8, edgecolor="white", zorder=3)
        ax.hlines(vals.mean(), i - 0.28, i + 0.28, color=COL[a], lw=3, zorder=4)
        ax.text(i, vals.mean() + 0.012, f"{vals.mean():.3f}", ha="center", fontweight="bold", color=COL[a])
    ax.axhline(0.5, ls=":", color="gray", lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(["cosine", "IDS", "propr (\u03c1p)"])
    ax.set_title(f"{strat}, k=50"); ax.set_ylim(0.55, 0.92); ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("mean bio-module AUC")
fig.suptitle("Stage A screening: proportionality wins (each dot = 1 of 6 cell-type groups)", y=1.02)
fig.savefig(FIG / "fig1_assoc_comparison_k50.png"); plt.close(fig)

# ===== FIG 2: negative control (real vs random) =====
nc = pd.read_csv(SA / "propr_negcontrol.csv")
ncp = nc[nc.config == "propr_bidirectional_w10_k50_var75_hvg2000"]
g = ncp.groupby("module").agg(real=("real_auc", "mean"), rand=("rand_mean", "mean"),
                              rp95=("rand_p95", "mean")).reindex([m_.replace("auc_", "") for m_ in MODS])
fig, ax = plt.subplots(figsize=(9, 5.2))
x = np.arange(len(g)); w = 0.38
ax.bar(x - w/2, g["real"], w, label="real module", color="#27ae60", zorder=3)
ax.bar(x + w/2, g["rand"], w, label="random sets (matched size)", color="#bdc3c7", zorder=3)
ax.errorbar(x + w/2, g["rand"], yerr=[np.zeros(len(g)), g["rp95"] - g["rand"]],
            fmt="none", ecolor="#7f8c8d", capsize=4, zorder=4)
for xi, rv in zip(x, g["real"]):
    ax.text(xi - w/2, rv + 0.01, f"{rv:.2f}", ha="center", fontweight="bold")
ax.axhline(0.5, ls=":", color="gray"); ax.set_xticks(x); ax.set_xticklabels(MODLAB)
ax.set_ylabel("AUC"); ax.set_ylim(0.4, 1.0); ax.legend(frameon=False)
ax.set_title("Negative control: propr signal is real\n(random gene-sets stay at 0.5; bars = p95)")
fig.savefig(FIG / "fig2_negative_control.png"); plt.close(fig)

# ===== FIG 3: per-module, assoc comparison at k=50 (bidir), mean over groups =====
fig, ax = plt.subplots(figsize=(10, 5.2))
sub = sa[(sa.strat == "bidirectional") & (sa.k == 50)]
x = np.arange(len(MODS)); w = 0.26
for j, a in enumerate(["cosine", "ids", "propr"]):
    means = [sub[sub.assoc == a][mod].mean() for mod in MODS]
    ax.bar(x + (j-1)*w, means, w, label={"cosine":"cosine","ids":"IDS","propr":"propr (\u03c1p)"}[a],
           color=COL[a], zorder=3)
ax.axhline(0.5, ls=":", color="gray"); ax.set_xticks(x); ax.set_xticklabels(MODLAB)
ax.set_ylabel("mean AUC over 6 groups"); ax.set_ylim(0.45, 1.0); ax.legend(frameon=False, ncol=3)
ax.set_title("Per-module AUC by association metric (bidirectional, k=50)"); ax.grid(axis="y", alpha=0.3)
fig.savefig(FIG / "fig3_per_module_k50.png"); plt.close(fig)

# ===== FIG 4: saturation (per-patient pilots) + joint dominance, from cached cosine =====
cc = pd.read_csv(CACHED / "bio_auc_collected.csv")
cm = cc["config_tag"].str.extract(r"^raw_cosine_(?P<strat>star|bidirectional)_w(?P<w>\d+)_k\d+_wl3_(?P<agg>perpat|joint)")
cc = pd.concat([cc, cm], axis=1)
pilots = ["BIOKEY_18_T_cell", "BIOKEY_30_Malignant", "BIOKEY_4_B_cell"]
pp = cc[(cc.agg == "perpat") & (cc.strat == "bidirectional") & (cc.group.isin(pilots))].copy()
pp["w"] = pp["w"].astype(int)
sat = pp.groupby("w")["mean_bio_auc"].mean().sort_index()
joint = cc[(cc.agg == "joint") & (cc.group == "ALL")]["mean_bio_auc"].max()
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(sat.index, sat.values, "o-", color="#2980b9", lw=2.5, ms=9, label="per-patient (pilots mean)")
peak = sat.idxmax()
ax.scatter([peak], [sat[peak]], s=220, facecolor="none", edgecolor="#c0392b", lw=2.5, zorder=5)
ax.annotate(f"peak @ w={peak}", (peak, sat[peak]), textcoords="offset points", xytext=(8, -18), color="#c0392b")
ax.axhline(joint, ls="--", color="#27ae60", lw=2.2, label=f"JOINT (all patients) = {joint:.2f}")
ax.set_xscale("log"); ax.set_xticks(sat.index); ax.set_xticklabels(sat.index)
ax.set_xlabel("walks per gene"); ax.set_ylabel("mean bio-module AUC")
ax.set_ylim(0.6, 1.0); ax.legend(frameon=False, loc="center right"); ax.grid(alpha=0.3)
ax.set_title("Walk-count saturation & joint dominance (raw+cosine, bidir)\nwalks=1 is the worst, not best; joint pooling tops every per-patient config")
fig.savefig(FIG / "fig4_saturation_and_joint.png"); plt.close(fig)

print("Wrote 4 figures to", FIG)
for f in sorted(FIG.glob("*.png")):
    print(" ", f.name)
