#!/usr/bin/env python3
"""Presentation figures v2: properly labeled, CORUM added, fig5 legend fixed."""
import re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path("/groups/ofircohen-group/users/michalnu_yuvat/project_breast")
SA = ROOT/"results/bassez2021/stageA/bio_auc"; CACHED = ROOT/"results/bassez2021/bio_auc"
JCT = ROOT/"results/bassez2021/models_joint_by_celltype/bio_auc"
FIG = ROOT/"results/bassez2021/figures/stageA"
MODS=["auc_S_phase","auc_G2M","auc_IFN_alpha","auc_IFN_gamma"]; MODLAB=["S phase","G2M","IFN-\u03b1","IFN-\u03b3"]
COL={"cosine":"#7f8c8d","ids":"#e67e22","propr":"#27ae60"}
plt.rcParams.update({"font.size":12.5,"axes.titlesize":14.5,"axes.titleweight":"bold","figure.dpi":130,"savefig.bbox":"tight"})
def foot(fig, txt): fig.text(0.5, -0.02, txt, ha="center", va="top", fontsize=9.5, style="italic", color="#555")

sa=pd.read_csv(SA/"bio_auc_collected.csv")
m=sa["config_tag"].str.extract(r"^(?P<assoc>cosine|ids|propr)_(?P<strat>star|bidirectional)_w\d+_k(?P<k>\d+)_")
sa=pd.concat([sa,m],axis=1); sa["k"]=sa["k"].astype(int)
sub6=["BIOKEY_18_T_cell","BIOKEY_30_Malignant","BIOKEY_4_B_cell","BIOKEY_10_Fibroblast","BIOKEY_12_Endothelial","BIOKEY_10_Myeloid"]

# FIG1
fig,axes=plt.subplots(1,2,figsize=(13,5.4),sharey=True)
for ax,strat in zip(axes,["bidirectional","star"]):
    s=sa[(sa.strat==strat)&(sa.k==50)&(sa.group.isin(sub6))]
    for i,a in enumerate(["cosine","ids","propr"]):
        v=s[s.assoc==a]["mean_bio_auc"].values
        ax.scatter(np.random.default_rng(i).normal(i,0.06,len(v)),v,color=COL[a],s=85,alpha=.85,edgecolor="white",zorder=3)
        ax.hlines(v.mean(),i-.28,i+.28,color=COL[a],lw=3,zorder=4)
        ax.text(i,v.mean()+.013,f"{v.mean():.3f}",ha="center",fontweight="bold",color=COL[a])
    ax.axhline(.5,ls=":",color="gray"); ax.set_xticks(range(3)); ax.set_xticklabels(["cosine","IDS","propr (\u03c1p)"])
    ax.set_title(f"{strat}, k=50"); ax.set_ylim(.55,.92); ax.grid(axis="y",alpha=.3)
axes[0].set_ylabel("mean bio-module AUC  (avg of 4 modules)")
fig.suptitle("Stage A screening: which association metric wins?\nEach dot = 1 of the 6 cell-type groups; black bar = mean",y=1.04)
foot(fig,"raw counts · walks=10 · walk_length=3 · top-k=50 · universe = top-2000 HVG \u222a benchmark genes (hvg2000) · benchmark = 4 bio co-expression modules")
fig.savefig(FIG/"fig1_assoc_comparison_k50.png"); plt.close(fig)

# FIG2
nc=pd.read_csv(SA/"propr_negcontrol.csv"); ncp=nc[nc.config=="propr_bidirectional_w10_k50_var75_hvg2000"]
g=ncp.groupby("module").agg(real=("real_auc","mean"),rand=("rand_mean","mean"),rp95=("rand_p95","mean")).reindex([x.replace("auc_","") for x in MODS])
fig,ax=plt.subplots(figsize=(9,5.2)); x=np.arange(len(g)); w=.38
ax.bar(x-w/2,g["real"],w,label="real module",color="#27ae60",zorder=3)
ax.bar(x+w/2,g["rand"],w,label="random gene-sets (same size)",color="#bdc3c7",zorder=3)
ax.errorbar(x+w/2,g["rand"],yerr=[np.zeros(len(g)),g["rp95"]-g["rand"]],fmt="none",ecolor="#7f8c8d",capsize=4,zorder=4)
for xi,rv in zip(x,g["real"]): ax.text(xi-w/2,rv+.01,f"{rv:.2f}",ha="center",fontweight="bold")
ax.axhline(.5,ls=":",color="gray"); ax.set_xticks(x); ax.set_xticklabels(MODLAB)
ax.set_ylabel("AUC"); ax.set_ylim(.4,1.0); ax.legend(frameon=False)
ax.set_title("Negative control: is propr's signal real?\nReal modules score high; random gene-sets stay at 0.5 (whisker = p95)")
foot(fig,"propr · bidirectional · walks=10 · k=50 · hvg2000 · 6 groups · 40 random draws per module size (43/54/98/198 genes)")
fig.savefig(FIG/"fig2_negative_control.png"); plt.close(fig)

# FIG3  (+ CORUM)
fig,ax=plt.subplots(figsize=(11,5.4)); s=sa[(sa.strat=="bidirectional")&(sa.k==50)&(sa.group.isin(sub6))]
cats=MODS+["corum_mean_auc"]; catlab=MODLAB+["CORUM\n(secondary)"]; x=np.arange(len(cats)); w=.26
for j,a in enumerate(["cosine","ids","propr"]):
    means=[s[s.assoc==a][c].mean() for c in cats]
    ax.bar(x+(j-1)*w,means,w,label={"cosine":"cosine","ids":"IDS","propr":"propr (\u03c1p)"}[a],color=COL[a],zorder=3)
ax.axhline(.5,ls=":",color="gray"); ax.axvline(3.5,ls="-",color="#ccc",lw=1)
ax.set_xticks(x); ax.set_xticklabels(catlab); ax.set_ylabel("mean AUC over 6 groups"); ax.set_ylim(.45,1.0)
ax.legend(frameon=False,ncol=3,loc="upper right"); ax.grid(axis="y",alpha=.3)
ax.set_title("Per-benchmark AUC by association metric\n(4 bio modules = primary; CORUM = secondary method-limit reference)")
foot(fig,"raw counts · bidirectional · walks=10 · walk_length=3 · k=50 · hvg2000 · 6 cell-type groups")
fig.savefig(FIG/"fig3_per_module_k50.png"); plt.close(fig)

# FIG4
cc=pd.read_csv(CACHED/"bio_auc_collected.csv")
m=cc["config_tag"].str.extract(r"^raw_cosine_(?P<strat>star|bidirectional)_w(?P<w>\d+)_k\d+_wl3_(?P<agglvl>perpat|joint)")
cc=pd.concat([cc,m],axis=1); pilots=["BIOKEY_18_T_cell","BIOKEY_30_Malignant","BIOKEY_4_B_cell"]
pp=cc[(cc["agglvl"]=="perpat")&(cc["strat"]=="bidirectional")&(cc["group"].isin(pilots))].copy(); pp["w"]=pp["w"].astype(int)
sat=pp.groupby("w")["mean_bio_auc"].mean().sort_index(); joint=cc[(cc["agglvl"]=="joint")&(cc["group"]=="ALL")]["mean_bio_auc"].max()
fig,ax=plt.subplots(figsize=(9,5.2))
ax.plot(sat.index,sat.values,"o-",color="#2980b9",lw=2.5,ms=9,label="per-patient (mean of 3 pilots)")
pk=sat.idxmax(); ax.scatter([1],[sat[1]],s=200,facecolor="none",edgecolor="#c0392b",lw=2.5,zorder=5)
ax.annotate("walks=1 is WORST\n(old CORUM-era 'optimum')",(1,sat[1]),textcoords="offset points",xytext=(14,-4),color="#c0392b",fontsize=10)
ax.scatter([pk],[sat[pk]],s=200,facecolor="none",edgecolor="#27ae60",lw=2.5,zorder=5)
ax.annotate(f"peak @ w={pk}",(pk,sat[pk]),textcoords="offset points",xytext=(6,8),color="#1e8449")
ax.axhline(joint,ls="--",color="#8e44ad",lw=2.2,label=f"global JOINT (all patients pooled) = {joint:.2f}")
ax.set_xscale("log"); ax.set_xticks(sat.index); ax.set_xticklabels(sat.index)
ax.set_xlabel("walks per gene"); ax.set_ylabel("mean bio-module AUC"); ax.set_ylim(.6,1.0)
ax.legend(frameon=False,loc="center right"); ax.grid(alpha=.3)
ax.set_title("Why walks=5\u201310 and why pool patients\nMore walks help up to ~10; pooling patients beats every per-patient setting")
foot(fig,"raw counts · cosine · bidirectional · k=5 · universe=var75 (~16.9k genes) · benchmark = 4 bio modules")
fig.savefig(FIG/"fig4_saturation_and_joint.png"); plt.close(fig)

# FIG5 (legend below, labeled)
def ct_of(gr): return re.sub(r"^BIOKEY_\d+_","",gr)
CTS=["T_cell","B_cell","Malignant","Myeloid","Fibroblast","Endothelial"]; STRAT="bidirectional"
ppf=sa[sa.config_tag.str.endswith(f"_{STRAT}_w10_k50_var75_hvg2000")].copy()
ppf=ppf[ppf.assoc.isin(["propr","cosine"])]; ppf["ct"]=ppf.group.map(ct_of)
ppm=ppf.groupby(["ct","assoc"])["mean_bio_auc"].mean()
jt=pd.read_csv(JCT/"bio_auc_collected.csv"); jt=jt[jt.config_tag.str.contains(f"_{STRAT}_")&jt.config_tag.str.endswith("_jointct")].copy()
jt["assoc"]=jt.config_tag.str.split("_").str[0]; jt["ct"]=jt.group.str.replace("celltype=","",regex=False)
jtm=jt.groupby(["ct","assoc"])["mean_bio_auc"].mean()
fig,ax=plt.subplots(figsize=(12,6.0)); x=np.arange(len(CTS)); w=.2
series=[("pp","cosine","#bdc3c7","per-patient cosine"),("pp","propr","#7fcfa0","per-patient propr"),
        ("jct","cosine","#7f8c8d","joint-by-cell-type cosine"),("jct","propr","#1e8449","joint-by-cell-type propr")]
for i,(lvl,a,c,lab) in enumerate(series):
    src=ppm if lvl=="pp" else jtm
    vals=[src.get((ct,a),np.nan) for ct in CTS]
    ax.bar(x+(i-1.5)*w,vals,w,color=c,label=lab,zorder=3)
ax.axhline(.894,ls="--",color="#c0392b",lw=2,label="old global JOINT (all cells, cosine) = 0.89")
ax.axhline(.5,ls=":",color="gray"); ax.set_xticks(x); ax.set_xticklabels(CTS,rotation=12)
ax.set_ylabel("mean bio-module AUC"); ax.set_ylim(.6,1.0)
ax.legend(frameon=False,ncol=3,fontsize=10.5,loc="upper center",bbox_to_anchor=(0.5,-0.14))
ax.set_title("Stage B (all 184 groups): propr holds at scale; joint-by-cell-type adds another lift")
foot(fig,"raw counts · bidirectional · walks=10 · walk_length=3 · k=50 · universe=hvg2000 (top-2000 HVG \u222a benchmark genes) · benchmark = 4 bio modules")
fig.savefig(FIG/"fig5_stageB_perpatient_vs_jointct.png"); plt.close(fig)
print("regenerated figs 1-5"); [print(" ",f.name) for f in sorted(FIG.glob("*.png"))]
