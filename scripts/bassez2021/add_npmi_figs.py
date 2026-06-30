import re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path("/groups/ofircohen-group/users/michalnu_yuvat/project_breast")
SA=pd.read_csv(ROOT/"results/bassez2021/stageA/bio_auc/bio_auc_collected.csv")
FIG=ROOT/"results/bassez2021/figures/stageA"
# include npmi in the metric regex
m=SA["config_tag"].str.extract(r"^(?P<assoc>cosine|spearman|ids|propr|npmi)_(?P<strat>star|bidirectional)_w\d+_k(?P<k>\d+)_(?P<meth>var75|alra)_hvg2000")
SA=pd.concat([SA,m],axis=1); SA["k"]=pd.to_numeric(SA["k"],errors="coerce")
sub6=["BIOKEY_18_T_cell","BIOKEY_30_Malignant","BIOKEY_4_B_cell","BIOKEY_10_Fibroblast","BIOKEY_12_Endothelial","BIOKEY_10_Myeloid"]
COL={"cosine":"#7f8c8d","spearman":"#16a085","ids":"#e67e22","propr":"#27ae60","npmi":"#8e44ad"}
LBL={"cosine":"cosine","spearman":"spearman","ids":"IDS","propr":"propr (\u03c1p)","npmi":"nPMI (co-detect)"}
ORDER=["cosine","ids","spearman","propr","npmi"]
plt.rcParams.update({"font.size":12.5,"axes.titlesize":14.5,"axes.titleweight":"bold","figure.dpi":130,"savefig.bbox":"tight"})

# ===== FIG1 with npmi added (5 metrics) =====
fig,axes=plt.subplots(1,2,figsize=(15,5.4),sharey=True)
for ax,strat in zip(axes,["bidirectional","star"]):
    s=SA[(SA.strat==strat)&(SA.k==50)&(SA.meth=="var75")&(SA.group.isin(sub6))]
    for i,a in enumerate(ORDER):
        v=s[s.assoc==a]["mean_bio_auc"].values
        if len(v)==0: continue
        ax.scatter(np.random.default_rng(i).normal(i,0.06,len(v)),v,color=COL[a],s=80,alpha=.85,edgecolor="white",zorder=3)
        ax.hlines(v.mean(),i-.30,i+.30,color=COL[a],lw=3,zorder=4)
        ax.text(i,v.mean()+.016,f"{v.mean():.3f}",ha="center",fontweight="bold",color=COL[a],fontsize=11)
    ax.axhline(.5,ls=":",color="gray"); ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([LBL[a] for a in ORDER],rotation=12)
    ax.set_title(f"{strat}, k=50"); ax.set_ylim(.55,.95); ax.grid(axis="y",alpha=.3)
axes[0].set_ylabel("mean bio-module AUC  (avg of 4 modules)")
fig.suptitle("Stage A screening: which association metric wins?\nEach dot = 1 of 6 cell-type groups; bar = mean. nPMI (magnitude-free co-detection) joins propr/spearman at the top.",y=1.06)
fig.text(0.5,-0.04,"raw counts \u00b7 walks=10 \u00b7 walk_length=3 \u00b7 top-k=50 \u00b7 universe = top-2000 HVG \u222a benchmark genes \u00b7 benchmark = 4 bio modules",
         ha="center",va="top",fontsize=9.5,style="italic",color="#555")
fig.savefig(FIG/"fig1_assoc_comparison_k50.png"); plt.close(fig)

# ===== FIG8 k-sweep saturation panel (k5/50/100) =====
fig,axes=plt.subplots(1,2,figsize=(14,5.2),sharey=True)
ks=[5,50,100]
for ax,strat in zip(axes,["bidirectional","star"]):
    s=SA[(SA.strat==strat)&(SA.meth=="var75")&(SA.group.isin(sub6))]
    for a in ["cosine","spearman","propr","npmi"]:
        ys=[]
        for k in ks:
            v=s[(s.assoc==a)&(s.k==k)]["mean_bio_auc"].values
            ys.append(v.mean() if len(v) else np.nan)
        ax.plot(range(len(ks)),ys,"-o",color=COL[a],lw=2.4,ms=8,label=LBL[a],zorder=3)
    ax.axhline(.5,ls=":",color="gray"); ax.set_xticks(range(len(ks))); ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_title(f"{strat}"); ax.grid(axis="y",alpha=.3)
axes[0].set_ylim(.62,.90); axes[0].set_ylabel("mean bio-module AUC")
axes[1].legend(frameon=False,loc="lower right")
fig.suptitle("Neighbourhood-size sweep: signal saturates by k=50 (k=100 adds nothing)\nMagnitude (cosine) keeps climbing toward the rank/co-detection metrics' ceiling; winners are flat past k=50.",y=1.06)
fig.text(0.5,-0.03,"6 screening groups \u00b7 raw \u00b7 walks=10 \u00b7 wl=3 \u00b7 hvg2000 \u00b7 spearman/npmi k=5 not run (k=5 was the early cosine/propr pilot grid)",
         ha="center",va="top",fontsize=9.5,style="italic",color="#555")
fig.savefig(FIG/"fig8_k_sweep_saturation.png"); plt.close(fig)
print("wrote fig1 (5 metrics, +npmi) + fig8 (k5/50/100 saturation)")
