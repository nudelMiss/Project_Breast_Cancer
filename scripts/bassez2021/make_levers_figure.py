#!/usr/bin/env python3
"""Consolidated design-space lever figure (7 panels: one per variable category)."""
import re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=Path("/groups/ofircohen-group/users/michalnu_yuvat/project_breast")
SA=pd.read_csv(ROOT/"results/bassez2021/stageA/bio_auc/bio_auc_collected.csv")
CC=pd.read_csv(ROOT/"results/bassez2021/bio_auc/bio_auc_collected.csv")
JT=pd.read_csv(ROOT/"results/bassez2021/models_joint_by_celltype/bio_auc/bio_auc_collected.csv")
FIG=ROOT/"results/bassez2021/figures/stageA"
sub6=["BIOKEY_18_T_cell","BIOKEY_30_Malignant","BIOKEY_4_B_cell","BIOKEY_10_Fibroblast","BIOKEY_12_Endothelial","BIOKEY_10_Myeloid"]
pilots=["BIOKEY_18_T_cell","BIOKEY_30_Malignant","BIOKEY_4_B_cell"]
plt.rcParams.update({"font.size":11,"axes.titlesize":12.5,"axes.titleweight":"bold","figure.dpi":130,"savefig.bbox":"tight"})
m=SA["config_tag"].str.extract(r"^(?P<assoc>cosine|spearman|ids|propr|npmi)_(?P<strat>star|bidirectional)_w\d+_k(?P<k>\d+)_(?P<meth>var75|alra)_hvg2000")
SA=pd.concat([SA,m],axis=1); SA["k"]=SA["k"].astype(int)
GR="#27ae60";GY="#7f8c8d";OR="#e67e22";BL="#2980b9";PU="#8e44ad";SP="#16a085";RD="#c0392b"
def sub(ax,t): ax.text(0.5,-0.24,t,transform=ax.transAxes,ha="center",va="top",fontsize=8.0,style="italic",color="#666",clip_on=False)
def vbars(ax,labels,vals,colors,title,subtxt,ymin=.45):
    x=np.arange(len(labels)); ax.bar(x,vals,color=colors,zorder=3,width=.62)
    for xi,v in zip(x,vals):
        if not np.isnan(v): ax.text(xi,v+.008,f"{v:.2f}",ha="center",fontweight="bold",fontsize=9.5)
    ax.axhline(.5,ls=":",color="gray"); ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=9.5)
    ax.set_ylim(ymin,1.0); ax.set_title(title); ax.grid(axis="y",alpha=.3); ax.set_ylabel("mean bio-AUC"); sub(ax,subtxt)

fig,axes=plt.subplots(2,4,figsize=(20,9.6)); A=axes.ravel()

# 1 Association metric (now with spearman) -- raw(var75), bidir, k50
s=SA[(SA.strat=="bidirectional")&(SA.k==50)&(SA.meth=="var75")&(SA.group.isin(sub6))]
vals=[s[s.assoc==a]["mean_bio_auc"].mean() for a in ["cosine","ids","spearman","propr","npmi"]]
vbars(A[0],["cosine","IDS","spearman","propr","nPMI"],vals,[GY,OR,SP,GR,PU],"1. Association metric","fixed: raw · bidir · k=50 · w10 · hvg2000 · 6 groups")

# 2 Imputation raw vs ALRA -- propr, bidir, k50
raw=SA[(SA.assoc=="propr")&(SA.strat=="bidirectional")&(SA.k==50)&(SA.meth=="var75")&(SA.group.isin(sub6))]["mean_bio_auc"].mean()
alra=SA[(SA.assoc=="propr")&(SA.strat=="bidirectional")&(SA.k==50)&(SA.meth=="alra")&(SA.group.isin(sub6))]["mean_bio_auc"].mean()
vbars(A[1],["raw\n(log1p+var75)","ALRA"],[raw,alra],[GR,"#a9cce3"],"2. Imputation","fixed: propr · bidir · k=50 · w10 · hvg2000 · 6 groups",ymin=.6)

# 3 top-k
A[2].set_title("3. Neighbors per gene (top-k)"); A[2].set_ylabel("mean bio-AUC")
for a,c,lab in [("cosine",GY,"cosine"),("propr",GR,"propr")]:
    y=[SA[(SA.assoc==a)&(SA.strat=="bidirectional")&(SA.k==kk)&(SA.meth=="var75")&(SA.group.isin(sub6))]["mean_bio_auc"].mean() for kk in [5,50]]
    A[2].plot([5,50],y,"o-",color=c,lw=2.5,ms=9,label=lab)
A[2].axhline(.5,ls=":",color="gray"); A[2].set_xticks([5,50]); A[2].set_xlim(-3,58); A[2].set_ylim(.6,1.0)
A[2].grid(alpha=.3); A[2].legend(frameon=False,fontsize=9); sub(A[2],"fixed: raw · bidir · w10 · hvg2000 · 6 groups")

# 4 walk strategy
A[3].set_title("4. Walk strategy"); A[3].set_ylabel("mean bio-AUC"); x=np.arange(2); w=.35
for j,(a,c,lab) in enumerate([("cosine",GY,"cosine"),("propr",GR,"propr")]):
    y=[SA[(SA.assoc==a)&(SA.strat==st)&(SA.k==50)&(SA.meth=="var75")&(SA.group.isin(sub6))]["mean_bio_auc"].mean() for st in ["bidirectional","star"]]
    A[3].bar(x+(j-.5)*w,y,w,color=c,label=lab,zorder=3)
A[3].axhline(.5,ls=":",color="gray"); A[3].set_xticks(x); A[3].set_xticklabels(["bidirectional","star"]); A[3].set_ylim(.6,1.0)
A[3].grid(axis="y",alpha=.3); A[3].legend(frameon=False,fontsize=9); sub(A[3],"fixed: raw · k=50 · w10 · hvg2000 · 6 groups")

# 5 walks per gene (cached cosine var75)
mm=CC["config_tag"].str.extract(r"^raw_cosine_(?P<strat>star|bidirectional)_w(?P<w>\d+)_k\d+_wl3_(?P<agg>perpat|joint)")
cc=pd.concat([CC,mm],axis=1); ps=cc[(cc["agg"]=="perpat")&(cc["strat"]=="bidirectional")&(cc["group"].isin(pilots))].copy()
ps["w"]=ps["w"].astype(int); sat=ps.groupby("w")["mean_bio_auc"].mean().sort_index()
A[4].plot(sat.index,sat.values,"o-",color=BL,lw=2.5,ms=9); A[4].set_xscale("log"); A[4].set_xticks(sat.index); A[4].set_xticklabels(sat.index)
A[4].axhline(.5,ls=":",color="gray"); A[4].set_ylim(.6,1.0); A[4].grid(alpha=.3); A[4].set_ylabel("mean bio-AUC")
A[4].set_title("5. Walks per gene"); A[4].axvspan(5,10,color=GR,alpha=.12); sub(A[4],"fixed: cosine · bidir · k=5 · var75 · 3 pilots")

# 6 aggregation
ppm=SA[(SA.assoc=="propr")&(SA.strat=="bidirectional")&(SA.k==50)&(SA.meth=="var75")]["mean_bio_auc"].mean()
jt=JT.copy(); jt["assoc"]=jt.config_tag.str.split("_").str[0]
jctm=jt[jt.config_tag.str.contains("_bidirectional_")&jt.config_tag.str.endswith("_jointct")&(jt.assoc=="propr")]["mean_bio_auc"].mean()
gj=cc[(cc["agg"]=="joint")&(cc["group"]=="ALL")]["mean_bio_auc"].max()
vbars(A[5],["per-patient\n(propr)","global JOINT\n(cosine)","joint-by-CT\n(propr)"],[ppm,gj,jctm],[GR,PU,"#1e8449"],
      "6. Aggregation","propr unless noted · bidir · k=50 (global JOINT=var75)",ymin=.6)

# 7 gene universe (cosine k5 w10 bidir)
cv=cc[(cc["agg"]=="perpat")&(cc["strat"]=="bidirectional")&(cc["group"].isin(pilots))&(cc["config_tag"].str.contains("_w10_"))]["mean_bio_auc"].mean()
ch=SA[(SA.assoc=="cosine")&(SA.strat=="bidirectional")&(SA.k==5)&(SA.meth=="var75")&(SA.group.isin(pilots))]["mean_bio_auc"].mean()
vbars(A[6],["var75\n(~16.9k)","hvg2000\n(~2.2k)"],[cv,ch],[GY,BL],"7. Gene universe","fixed: cosine · bidir · k=5 · w10 · 3 pilots",ymin=.6)

A[7].axis("off")
A[7].text(0.5,0.5,"Primary readout:\nmean of 4 bio modules\n(S phase, G2M, IFN-\u03b1, IFN-\u03b3)\n\nAUC=0.5 is random\n\nWINNERS:\npropr · k=50 · bidir\u2248star\njoint-by-cell-type",
          ha="center",va="center",fontsize=11.5,color="#333",
          bbox=dict(boxstyle="round",fc="#f4f9f4",ec="#27ae60"))
fig.suptitle("Bassez2021 design-space levers \u2014 effect of each variable on bio-module AUC",fontsize=16,fontweight="bold",y=1.005)
fig.subplots_adjust(hspace=0.55, bottom=0.08); fig.tight_layout(rect=[0,0.02,1,0.99]); fig.savefig(FIG/"fig6_design_space_levers.png"); plt.close(fig)
print(f"metric: cosine={vals[0]:.3f} ids={vals[1]:.3f} spearman={vals[2]:.3f} propr={vals[3]:.3f} npmi={vals[4]:.3f}")
print(f"imputation: raw={raw:.3f} alra={alra:.3f}")
print("wrote fig6_design_space_levers.png (7 panels)")
