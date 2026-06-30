#!/usr/bin/env python3
"""Stage B summary + figure: per-patient vs joint-by-celltype, propr vs cosine, by cell type."""
import re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path("/groups/ofircohen-group/users/michalnu_yuvat/project_breast")
PP = ROOT / "results/bassez2021/stageA/bio_auc/bio_auc_collected.csv"
JCT = ROOT / "results/bassez2021/models_joint_by_celltype/bio_auc/bio_auc_collected.csv"
FIG = ROOT / "results/bassez2021/figures/stageA"
MODS = ["auc_S_phase", "auc_G2M", "auc_IFN_alpha", "auc_IFN_gamma"]
CTS = ["T_cell", "B_cell", "Malignant", "Myeloid", "Fibroblast", "Endothelial"]
STRAT = "bidirectional"
plt.rcParams.update({"font.size":13,"axes.titlesize":15,"axes.titleweight":"bold","figure.dpi":130,"savefig.bbox":"tight"})

def celltype_of(group):  # BIOKEY_18_T_cell -> T_cell
    return re.sub(r"^BIOKEY_\d+_", "", group)

# ---- per-patient (mean over patients per celltype) ----
pp = pd.read_csv(PP)
tag = f"_{STRAT}_w10_k50_var75_hvg2000"
pp = pp[pp.config_tag.str.endswith(tag)].copy()
pp["assoc"] = pp.config_tag.str.split("_").str[0]
pp = pp[pp.assoc.isin(["propr","cosine"])]
pp["ct"] = pp.group.map(celltype_of)
pp_ct = pp.groupby(["ct","assoc"])[["mean_bio_auc"]+MODS].mean().reset_index()

# ---- joint-by-celltype ----
jct = pd.read_csv(JCT)
jct = jct[jct.config_tag.str.contains(f"_{STRAT}_") & jct.config_tag.str.endswith("_jointct")].copy()
jct["assoc"] = jct.config_tag.str.split("_").str[0]
jct["ct"] = jct.group.str.replace("celltype=","",regex=False)

# ---- combined summary ----
rows = []
for ct in CTS:
    for assoc in ["propr","cosine"]:
        p = pp_ct[(pp_ct.ct==ct)&(pp_ct.assoc==assoc)]
        j = jct[(jct.ct==ct)&(jct.assoc==assoc)]
        rows.append(dict(celltype=ct, assoc=assoc,
            perpat_bio=round(float(p.mean_bio_auc.iloc[0]),3) if len(p) else np.nan,
            jointct_bio=round(float(j.mean_bio_auc.iloc[0]),3) if len(j) else np.nan,
            jointct_S=round(float(j.auc_S_phase.iloc[0]),3) if len(j) else np.nan,
            jointct_G2M=round(float(j.auc_G2M.iloc[0]),3) if len(j) else np.nan,
            jointct_IFNa=round(float(j.auc_IFN_alpha.iloc[0]),3) if len(j) else np.nan,
            jointct_IFNg=round(float(j.auc_IFN_gamma.iloc[0]),3) if len(j) else np.nan))
summ = pd.DataFrame(rows)
summ.to_csv(ROOT/"results/bassez2021/stageB_summary.csv", index=False)
print(summ.to_string(index=False))

# ---- FIG 5 ----
fig, ax = plt.subplots(figsize=(12,5.6))
x = np.arange(len(CTS)); w = 0.2
series = [("perpat","cosine","#bdc3c7","per-patient cosine"),
          ("perpat","propr","#7fcfa0","per-patient propr"),
          ("jointct","cosine","#7f8c8d","joint-by-CT cosine"),
          ("jointct","propr","#1e8449","joint-by-CT propr")]
for i,(lvl,assoc,c,lab) in enumerate(series):
    col = f"{lvl}_bio"
    vals = [summ[(summ.celltype==ct)&(summ.assoc==assoc)][col].iloc[0] for ct in CTS]
    ax.bar(x+(i-1.5)*w, vals, w, color=c, label=lab, zorder=3)
gj = 0.894
ax.axhline(gj, ls="--", color="#c0392b", lw=2, label=f"global JOINT (all cells, cosine) = {gj:.2f}")
ax.axhline(0.5, ls=":", color="gray")
ax.set_xticks(x); ax.set_xticklabels(CTS, rotation=15)
ax.set_ylabel("mean bio-module AUC"); ax.set_ylim(0.6,1.0)
ax.legend(frameon=False, ncol=2, fontsize=11, loc="lower center")
ax.set_title("Stage B (184 groups): propr holds at scale; joint-by-cell-type adds another lift\n(bidirectional, k=50, hvg2000)")
ax.grid(axis="y", alpha=0.3)
fig.savefig(FIG/"fig5_stageB_perpatient_vs_jointct.png"); plt.close(fig)
print("\nWrote fig5_stageB_perpatient_vs_jointct.png + stageB_summary.csv")
