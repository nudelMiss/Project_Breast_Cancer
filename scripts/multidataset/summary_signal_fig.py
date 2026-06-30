#!/usr/bin/env python3
"""Per-dataset signal summary: per-patient propr combined-AUC by cell type (boxplot+strip),
with joint-by-celltype overlaid as a diamond where available."""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/groups/ofircohen-group/users/michalnu_yuvat/project_breast")
PP_TAG = "propr_bidirectional_w10_k50_var75_hvg2000"
J_TAG  = PP_TAG + "_jointct"
BCOLS = ["auc_S_phase","auc_G2M","auc_IFN_alpha","auc_IFN_gamma","corum_mean_auc"]
EXP = {"bassez2021":"exports_bassez","wu2021":"exports_wu_counts","griffiths2021":"exports_griffiths",
       "qian2020":"exports_qian","azizi2018":"exports_azizi","gao2021":"exports_gao","pal2021":"exports_pal2021"}
COLOR = {'Malignant':'#D62728','Epithelial':'#BCBD22','T_cell':'#2CA02C','B_cell':'#1F77B4',
         'Plasmablast':'#1F77B4','NK_cell':'#17BECF','Myeloid':'#9467BD','Macrophage':'#9467BD',
         'Monocyte':'#9467BD','DC':'#9467BD','Fibroblast':'#8C564B','Pericyte':'#8C564B',
         'Endothelial':'#FF7F0E','HSC':'#E377C2','Lymphoid':'#2CA02C'}
ORDER = list(COLOR)

def celltypes_for(ds):
    p=REPO/EXP.get(ds,"")
    cts=set()
    if p.exists():
        for d in p.glob("patient=*"):
            m=re.search(r"celltype=(.+)$", d.name)
            if m: cts.add(m.group(1))
    return sorted(cts, key=len, reverse=True)

def assign_ct(g, cts):
    for ct in cts:
        if g==ct or g.endswith("_"+ct): return ct
    return None

def load_pp(ds):
    f=REPO/f"results/{ds}/stageA/bio_auc/bio_auc_collected.csv"
    if not f.exists(): return None
    df=pd.read_csv(f); df=df[df.config_tag==PP_TAG].copy()
    if df.empty: return None
    df=df.drop_duplicates(subset="group", keep="last")
    cts=celltypes_for(ds); df["celltype"]=df["group"].map(lambda g: assign_ct(g,cts))
    df=df[df.celltype.notna()].copy(); df["combined"]=df[BCOLS].mean(axis=1)
    return df

def load_joint(ds):
    for f in [REPO/f"results/{ds}/jointct_bio_auc/bio_auc_collected.csv",
              REPO/f"results/{ds}/models_joint_by_celltype/bio_auc/bio_auc_collected.csv"]:
        if f.exists():
            df=pd.read_csv(f); df=df[df.config_tag==J_TAG].copy()
            if df.empty: continue
            df["celltype"]=df["group"].map(lambda g: g.replace("celltype=","")); df["combined"]=df[BCOLS].mean(axis=1)
            return df
    return None

def make_fig(ds):
    pp=load_pp(ds)
    if pp is None or pp.empty: print(f"[skip] {ds}: no per-patient propr scores"); return
    j=(None if "--no-joint" in sys.argv else load_joint(ds))
    cts=sorted(pp.celltype.unique(), key=lambda c:(ORDER.index(c) if c in ORDER else 99, c))
    fig,ax=plt.subplots(figsize=(max(6,1.15*len(cts)+2),5.5))
    rng=np.random.default_rng(0)
    for i,ct in enumerate(cts):
        vals=pp[pp.celltype==ct]["combined"].values; col=COLOR.get(ct,"#999999")
        bp=ax.boxplot(vals, positions=[i], widths=0.55, patch_artist=True, showfliers=False,
                      medianprops=dict(color='black'))
        for b in bp['boxes']: b.set(facecolor=col, alpha=0.30, edgecolor=col)
        ax.scatter(rng.normal(i,0.06,size=len(vals)),vals,c=col,s=22,alpha=0.8,edgecolors='none',zorder=3)
        if j is not None:
            jv=j[j.celltype==ct]["combined"].values
            if len(jv): ax.scatter([i],[jv[0]],marker='D',s=150,c=col,edgecolors='black',linewidths=1.6,zorder=5)
    ax.axhline(0.5,ls='--',c='grey',lw=1)
    ax.set_xticks(range(len(cts))); ax.set_xticklabels([f"{c}\n(n={int((pp.celltype==c).sum())})" for c in cts],rotation=35,ha='right',fontsize=9)
    ax.set_ylabel("Combined benchmark AUC  (S/G2M/IFN-a/IFN-g/CORUM)"); ax.set_ylim(0.45,1.0)
    t=f"{ds} — propr per-cell-type signal  (n={pp.group.nunique()} groups)"
    if j is not None: t+="    \u25C6 = joint-by-celltype"
    ax.set_title(t)
    out=REPO/f"results/{ds}/figures"; out.mkdir(parents=True,exist_ok=True)
    fp=out/"signal_summary.png"; fig.tight_layout(); fig.savefig(fp,dpi=150); plt.close(fig)
    print(f"[OK] {ds}: {fp}  cts={len(cts)} joint={'y' if j is not None else 'n'}")

if __name__=="__main__":
    dss=[a for a in sys.argv[1:] if not a.startswith("--")] or ["bassez2021","wu2021","griffiths2021","qian2020"]
    for ds in dss: make_fig(ds)
