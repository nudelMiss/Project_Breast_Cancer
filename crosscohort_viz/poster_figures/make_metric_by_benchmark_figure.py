#!/usr/bin/env python
# Poster figure: per-benchmark AUC by association metric (cosine / IDS / propr).
# Reuses the exact FIG3 data logic from make_presentation_figs_v2.py.
# Coral Coast palette + Georgia/Gelasio typography, distinct color per metric.
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt

ROOT = Path("/groups/ofircohen-group/users/michalnu_yuvat/project_breast")
SA   = ROOT/"results/bassez2021/stageA/bio_auc"
BASE = ROOT/"crosscohort_viz/poster_figures"

fm.fontManager.addfont(str(BASE/"fonts/Gelasio.ttf"))
SERIF = fm.FontProperties(fname=str(BASE/"fonts/Gelasio.ttf")).get_name()

# --- Coral Coast palette ---
TEAL_TINT="#DCF1F0"; LAGOON_TEAL="#0FA3A0"; DEEP_MARINE="#173B4A"
SLATE_MIST="#6E8088"; SEA_CORAL="#FF6B5C"; AMBER="#F5A623"
# distinct fill per metric (mirrors original grey/orange/green semantics)
COL={"cosine":AMBER, "ids":SEA_CORAL, "propr":LAGOON_TEAL}

plt.rcParams.update({
    "font.family":SERIF, "font.size":13,
    "text.color":DEEP_MARINE, "axes.edgecolor":DEEP_MARINE,
    "axes.labelcolor":DEEP_MARINE, "xtick.color":DEEP_MARINE,
    "ytick.color":DEEP_MARINE, "axes.linewidth":1.1,
})

MODS=["auc_S_phase","auc_G2M","auc_IFN_alpha","auc_IFN_gamma"]
MODLAB=["S phase","G2M",r"IFN-$\alpha$",r"IFN-$\gamma$"]
sub6=["BIOKEY_18_T_cell","BIOKEY_30_Malignant","BIOKEY_4_B_cell",
      "BIOKEY_10_Fibroblast","BIOKEY_12_Endothelial","BIOKEY_10_Myeloid"]

sa=pd.read_csv(SA/"bio_auc_collected.csv")
m=sa["config_tag"].str.extract(r"^(?P<assoc>cosine|ids|propr)_(?P<strat>star|bidirectional)_w\d+_k(?P<k>\d+)_")
sa=pd.concat([sa,m],axis=1)
sa=sa.dropna(subset=["assoc","strat","k"]).copy()
sa["k"]=sa["k"].astype(int)
s=sa[(sa.strat=="bidirectional")&(sa.k==50)&(sa.group.isin(sub6))]

cats=MODS+["corum_mean_auc"]; catlab=MODLAB+["CORUM\n(secondary)"]
x=np.arange(len(cats)); w=.26

fig,ax=plt.subplots(figsize=(10.5,5.8), dpi=220)
printvals={}
for j,a in enumerate(["cosine","ids","propr"]):
    means=[s[s.assoc==a][c].mean() for c in cats]
    printvals[a]=means
    ax.bar(x+(j-1)*w, means, w,
           label={"cosine":"cosine","ids":"IDS","propr":r"propr ($\rho$p)"}[a],
           color=COL[a], edgecolor=DEEP_MARINE, linewidth=0.8, zorder=3)

ax.axhline(.5, ls=":", color=SLATE_MIST, zorder=1)
ax.axvline(3.5, ls="-", color=TEAL_TINT, lw=2, zorder=1)
ax.set_xticks(x); ax.set_xticklabels(catlab, fontsize=13)
ax.set_ylabel("mean AUC over 6 groups", fontsize=15)
ax.set_ylim(.45,1.0)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", ls=":", lw=0.7, color="0.85", zorder=0)
leg=ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=13)
for t in leg.get_texts(): t.set_color(DEEP_MARINE)
ax.set_title("Per-benchmark AUC by association metric\n"
             "(4 bio modules = primary; CORUM = secondary method-limit reference)",
             fontsize=16, fontweight="bold", color=DEEP_MARINE, pad=12)
fig.text(0.5,-0.02,
         "raw counts - bidirectional - walks=10 - walk_length=3 - k=50 - hvg2000 - 6 cell-type groups",
         ha="center", va="top", fontsize=10, style="italic", color=SLATE_MIST)

fig.tight_layout()
for ext in ("png","pdf"):
    fig.savefig(BASE/f"metric_by_benchmark.{ext}", bbox_inches="tight")
print("FONT_USED:", SERIF)
for a in printvals: print(a, [f"{v:.3f}" for v in printvals[a]])
