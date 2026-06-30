import re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path("/groups/ofircohen-group/users/michalnu_yuvat/project_breast")
SA=pd.read_csv(ROOT/"results/bassez2021/stageA/bio_auc/bio_auc_collected.csv")
JT=pd.read_csv(ROOT/"results/bassez2021/models_joint_by_celltype/bio_auc/bio_auc_collected.csv")
FIG=ROOT/"results/bassez2021/figures/stageA"
m=SA["config_tag"].str.extract(r"^(?P<assoc>cosine|spearman|ids|propr)_(?P<strat>star|bidirectional)_w\d+_k(?P<k>\d+)_(?P<meth>var75|alra)_hvg2000")
SA=pd.concat([SA,m],axis=1); SA["k"]=pd.to_numeric(SA["k"],errors="coerce")
sub6=["BIOKEY_18_T_cell","BIOKEY_30_Malignant","BIOKEY_4_B_cell","BIOKEY_10_Fibroblast","BIOKEY_12_Endothelial","BIOKEY_10_Myeloid"]
COL={"cosine":"#7f8c8d","spearman":"#16a085","ids":"#e67e22","propr":"#27ae60"}
LBL={"cosine":"cosine","spearman":"spearman","ids":"IDS","propr":"propr (\u03c1p)"}
ORDER=["cosine","spearman","ids","propr"]
plt.rcParams.update({"font.size":12.5,"axes.titlesize":14.5,"axes.titleweight":"bold","figure.dpi":130,"savefig.bbox":"tight"})

# ===== FIG1 (slide 50) now with spearman =====
fig,axes=plt.subplots(1,2,figsize=(14,5.4),sharey=True)
for ax,strat in zip(axes,["bidirectional","star"]):
    s=SA[(SA.strat==strat)&(SA.k==50)&(SA.meth=="var75")&(SA.group.isin(sub6))]
    for i,a in enumerate(ORDER):
        v=s[s.assoc==a]["mean_bio_auc"].values
        ax.scatter(np.random.default_rng(i).normal(i,0.06,len(v)),v,color=COL[a],s=80,alpha=.85,edgecolor="white",zorder=3)
        ax.hlines(v.mean(),i-.30,i+.30,color=COL[a],lw=3,zorder=4)
        ax.text(i,v.mean()+.016,f"{v.mean():.3f}",ha="center",fontweight="bold",color=COL[a],fontsize=11)
    ax.axhline(.5,ls=":",color="gray"); ax.set_xticks(range(4)); ax.set_xticklabels([LBL[a] for a in ORDER])
    ax.set_title(f"{strat}, k=50"); ax.set_ylim(.55,.92); ax.grid(axis="y",alpha=.3)
axes[0].set_ylabel("mean bio-module AUC  (avg of 4 modules)")
fig.suptitle("Stage A screening: which association metric wins?\nEach dot = 1 of the 6 cell-type groups; bar = mean. Rank/proportionality (spearman, propr) win.",y=1.05)
fig.text(0.5,-0.02,"raw counts \u00b7 walks=10 \u00b7 walk_length=3 \u00b7 top-k=50 \u00b7 universe = top-2000 HVG \u222a benchmark genes (hvg2000) \u00b7 benchmark = 4 bio modules",
         ha="center",va="top",fontsize=9.5,style="italic",color="#555")
fig.savefig(FIG/"fig1_assoc_comparison_k50.png"); plt.close(fig)

# ===== FIG7 propr vs spearman head-to-head at Stage B scale =====
def ct_of(g): return re.sub(r"^BIOKEY_\d+_","",g)
CTS=["T_cell","B_cell","Malignant","Myeloid","Fibroblast","Endothelial"]
pp=SA[(SA.strat=="bidirectional")&(SA.k==50)&(SA.meth=="var75")&(SA.assoc.isin(["propr","spearman"]))].copy()
pp["ct"]=pp.group.map(ct_of); ppm=pp.groupby(["ct","assoc"])["mean_bio_auc"].mean()
jt=JT.copy(); jt=jt[jt.config_tag.str.contains("_bidirectional_")&jt.config_tag.str.endswith("_jointct")]
jt["assoc"]=jt.config_tag.str.split("_").str[0]; jt["ct"]=jt.group.str.replace("celltype=","",regex=False)
jtm=jt.groupby(["ct","assoc"])["mean_bio_auc"].mean()
fig,axes=plt.subplots(1,2,figsize=(15,5.4),sharey=True); x=np.arange(len(CTS)); w=.38
for ax,(src,ttl) in zip(axes,[(ppm,"per-patient (mean over patients)"),(jtm,"joint-by-cell-type")]):
    for j,a in enumerate(["spearman","propr"]):
        vals=[src.get((ct,a),np.nan) for ct in CTS]
        ax.bar(x+(j-.5)*w,vals,w,color=COL[a],label=LBL[a],zorder=3)
    ax.axhline(.5,ls=":",color="gray"); ax.set_xticks(x); ax.set_xticklabels(CTS,rotation=14)
    ax.set_title(ttl); ax.grid(axis="y",alpha=.3); ax.legend(frameon=False)
axes[0].set_ylim(.6,1.0); axes[0].set_ylabel("mean bio-module AUC")
fig.suptitle("propr vs spearman head-to-head (Stage B, 184 groups): the two top metrics are tied",y=1.03)
fig.text(0.5,-0.02,"raw counts \u00b7 bidirectional \u00b7 walks=10 \u00b7 k=50 \u00b7 hvg2000 \u00b7 benchmark = 4 bio modules",ha="center",va="top",fontsize=9.5,style="italic",color="#555")
fig.savefig(FIG/"fig7_propr_vs_spearman_stageB.png"); plt.close(fig)

# print head-to-head
print("=== per-patient (mean over cell types) ===")
print("  propr   ", round(pp[pp.assoc=='propr']['mean_bio_auc'].mean(),3), " spearman", round(pp[pp.assoc=='spearman']['mean_bio_auc'].mean(),3))
print("=== joint-by-CT (mean over cell types) ===")
print("  propr   ", round(jt[jt.assoc=='propr']['mean_bio_auc'].mean(),3), " spearman", round(jt[jt.assoc=='spearman']['mean_bio_auc'].mean(),3))
print("wrote fig1 (4 metrics) + fig7 (propr vs spearman)")
