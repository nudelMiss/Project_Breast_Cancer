#!/usr/bin/env python3
"""Supervisor figures v3. Headline y = COMBINED benchmark AUC (mean of 4 bio modules + CORUM)."""
import re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

ROOT = Path("/groups/ofircohen-group/users/michalnu_yuvat/project_breast")
FIG  = ROOT/"results/bassez2021/figures/stageA"
SA = pd.read_csv(ROOT/"results/bassez2021/stageA/bio_auc/bio_auc_collected.csv")
NC = pd.read_csv(ROOT/"results/bassez2021/stageA/group_ncells.csv")
m = SA["config_tag"].str.extract(r"^(?P<assoc>cosine|spearman|ids|propr|npmi)_(?P<strat>star|bidirectional)_w\d+_k(?P<k>\d+)_(?P<meth>var75|alra)_hvg2000")
SA = pd.concat([SA, m], axis=1); SA["k"]=pd.to_numeric(SA["k"],errors="coerce")
BIO=["auc_S_phase","auc_G2M","auc_IFN_alpha","auc_IFN_gamma"]
for c in BIO+["corum_mean_auc"]: SA[c]=pd.to_numeric(SA[c],errors="coerce")
SA["combined"]=SA[BIO+["corum_mean_auc"]].mean(axis=1)
SUB6=["BIOKEY_18_T_cell","BIOKEY_30_Malignant","BIOKEY_4_B_cell","BIOKEY_10_Fibroblast","BIOKEY_12_Endothelial","BIOKEY_10_Myeloid"]
def ct_of(g): return re.sub(r"^BIOKEY_\d+_","",g)
SA["ct"]=SA["group"].map(ct_of)
CT_MARK={"T_cell":"o","Malignant":"s","B_cell":"^","Myeloid":"D","Fibroblast":"v","Endothelial":"P"}
# NEW clearly-separated metric palette: grey / orange / blue / green / purple
MCOL={"cosine":"#6c757d","ids":"#e67e22","spearman":"#2980b9","propr":"#27ae60","npmi":"#8e44ad"}
MLBL={"cosine":"cosine","ids":"IDS","spearman":"spearman","propr":"propr (\u03c1p)","npmi":"nPMI"}
ORDER=["cosine","ids","spearman","propr","npmi"]
YLAB="combined benchmark AUC\n(mean of S phase, G2M, IFN-\u03b1, IFN-\u03b3, CORUM)"
plt.rcParams.update({"font.size":12,"axes.titlesize":13.5,"axes.titleweight":"bold","figure.dpi":130,"savefig.bbox":"tight"})
def ci95(v):
    v=np.asarray([x for x in v if x==x],float)
    if len(v)<2: return (np.nan,)*3
    mu=v.mean(); h=stats.t.ppf(0.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)); return mu,mu-h,mu+h
def swarm_x(center, yvals, half=0.22):
    """deterministic rank-based horizontal spread so points never overlap."""
    yvals=np.asarray(yvals,float); n=len(yvals)
    if n==1: return np.array([center])
    order=np.argsort(yvals); offs=np.linspace(-half,half,n); x=np.empty(n)
    for slot,idx in enumerate(order): x[idx]=center+offs[slot]
    return x

def fig1():
    fig,axes=plt.subplots(1,2,figsize=(15.5,6.2),sharey=True)
    for ax,strat in zip(axes,["bidirectional","star"]):
        s=SA[(SA.strat==strat)&(SA.k==50)&(SA.meth=="var75")&(SA.group.isin(SUB6))]
        box=[s[s.assoc==a]["combined"].dropna().values for a in ORDER]
        bp=ax.boxplot(box,positions=range(len(ORDER)),widths=0.62,patch_artist=True,showfliers=False,zorder=2)
        for patch,a in zip(bp["boxes"],ORDER):
            patch.set_facecolor(MCOL[a]); patch.set_alpha(0.28); patch.set_edgecolor(MCOL[a]); patch.set_linewidth(1.6)
        for el in ["whiskers","caps"]:
            for ln in bp[el]: ln.set_color("#888"); ln.set_linewidth(1.0)
        for med,a in zip(bp["medians"],ORDER): med.set_color(MCOL[a]); med.set_linewidth(2.2)
        for i,a in enumerate(ORDER):
            sa=s[s.assoc==a].copy()
            xs=swarm_x(i, sa["combined"].values, half=0.22)
            for (xx,(_,r)) in zip(xs, sa.iterrows()):
                ax.scatter(xx, r["combined"], marker=CT_MARK.get(r["ct"],"o"),
                           color="#2b2b2b", s=70, edgecolor="white", linewidth=0.9, zorder=4)
            mu,lo,hi=ci95(sa["combined"].values)
            if mu==mu:
                ax.errorbar(i+0.40,mu,yerr=[[mu-lo],[hi-mu]],fmt="_",color=MCOL[a],capsize=4,lw=2.0,zorder=5,ms=16)
        ax.axhline(0.5,ls=":",color="gray")
        ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([MLBL[a] for a in ORDER],rotation=12)
        ax.set_xlim(-0.6,len(ORDER)-0.1); ax.set_ylim(0.55,0.88); ax.grid(axis="y",alpha=0.3); ax.set_title(f"{strat}, k=50")
    axes[0].set_ylabel(YLAB)
    shp=[Line2D([0],[0],marker=CT_MARK[c],color="#2b2b2b",ls="",ms=9,mec="white") for c in CT_MARK]
    fig.legend(shp,list(CT_MARK.keys()),title="cell type (dot shape)",ncol=6,loc="lower center",bbox_to_anchor=(0.5,-0.02),frameon=False)
    fig.suptitle("Stage A: association-metric comparison \u2014 all benchmarks combined\n"
                 "Box = spread over 6 groups; dots = groups (shape = cell type); colored bar = 95% CI of the mean.",y=1.03)
    fig.text(0.5,-0.065,"raw \u00b7 walks=10 \u00b7 wl=3 \u00b7 k=50 \u00b7 hvg2000 \u00b7 y = equally-weighted mean of 4 bio modules + CORUM",ha="center",va="top",fontsize=9.5,style="italic",color="#555")
    fig.tight_layout(rect=[0,0.03,1,0.99]); fig.savefig(FIG/"fig1_assoc_comparison_k50.png"); plt.close(fig); print("fig1 OK")

def fig8():
    SCOL={"cosine":"#8d99ae","spearman":"#2a9d8f","propr":"#264653","npmi":"#e76f51"}
    SLBL={"cosine":"cosine","spearman":"spearman","propr":"propr (\u03c1p)","npmi":"nPMI"}
    metrics=["cosine","spearman","propr","npmi"]; ks=[5,50,100]; xpos={5:0,50:1,100:2}
    fig,axes=plt.subplots(1,2,figsize=(14.5,5.6),sharey=True)
    for ax,strat in zip(axes,["bidirectional","star"]):
        s=SA[(SA.strat==strat)&(SA.meth=="var75")&(SA.group.isin(SUB6))]
        for mi,a in enumerate(metrics):
            jx=(mi-1.5)*0.06; mus=[];los=[];his=[];xs=[]
            for k in ks:
                v=s[(s.assoc==a)&(s.k==k)]["combined"].values
                if len(v)==0: mus.append(np.nan);los.append(np.nan);his.append(np.nan);xs.append(xpos[k]);continue
                mu,lo,hi=ci95(v); mus.append(mu);los.append(lo);his.append(hi);xs.append(xpos[k])
                ax.scatter(np.full(len(v),xpos[k]+jx),v,color=SCOL[a],s=22,alpha=0.45,edgecolor="none",zorder=2)
            xs=np.array(xs,float)+jx; good=[i for i,mm in enumerate(mus) if mm==mm]
            ax.plot(xs[good],[mus[i] for i in good],"-o",color=SCOL[a],lw=2.6,ms=8,label=SLBL[a],zorder=4)
        ax.axhline(0.5,ls=":",color="gray"); ax.set_xticks(list(xpos.values())); ax.set_xticklabels([f"k={k}" for k in ks])
        ax.set_xlim(-0.4,2.4); ax.set_title(strat); ax.grid(axis="y",alpha=0.3)
    axes[0].set_ylim(0.58,0.86); axes[0].set_ylabel(YLAB); axes[1].legend(frameon=False,loc="lower right")
    fig.suptitle("Neighbourhood-size sweep (all benchmarks combined): signal saturates by k=50\n"
                 "Bold line = mean over 6 groups; faint dots = individual groups (dispersion).",y=1.05)
    fig.text(0.5,-0.03,"raw \u00b7 walks=10 \u00b7 wl=3 \u00b7 hvg2000 \u00b7 6 groups \u00b7 spearman from k=5 \u00b7 y = 4 bio modules + CORUM",ha="center",va="top",fontsize=9.5,style="italic",color="#555")
    fig.savefig(FIG/"fig8_k_sweep_saturation.png"); plt.close(fig); print("fig8 OK")

def fig4_single(with_reg):
    THRESH=500   # candidate PCoA inclusion cutoff (cells); set None to hide
    CFG="propr_bidirectional_w10_k50_var75_hvg2000"
    d=SA[SA.config_tag==CFG][["group","combined"]].merge(NC,left_on="group",right_on="group_tag")
    d=d[d.celltype!="Mast"]
    CTS=["T_cell","Malignant","B_cell","Myeloid","Fibroblast","Endothelial"]
    CCOL={"T_cell":"#2a9d8f","Malignant":"#264653","B_cell":"#c9a227","Myeloid":"#f4a261","Fibroblast":"#8d99ae","Endothelial":"#e76f51"}
    HALF=0.40
    fig,ax=plt.subplots(figsize=(17,6.2))
    msg=[]; ndrop_tot=0
    for ci_,ct in enumerate(CTS):
        sub=d[d.celltype==ct].sort_values("n_cells")
        n=sub["n_cells"].astype(float).values; lx=np.log10(n); y=sub["combined"].values
        if len(lx)==0: continue
        lo,hi=lx.min(),lx.max(); rng=(hi-lo) if hi>lo else 1.0
        norm=(lx-lo)/rng; xpos=ci_-HALF+norm*(2*HALF)
        keep=n>=THRESH if THRESH else np.ones(len(n),bool); drop=~keep
        # threshold line + shaded drop zone (left of cutoff within this column)
        if THRESH:
            nT=(np.log10(THRESH)-lo)/rng; xT=ci_-HALF+np.clip(nT,0,1)*(2*HALF)
            ax.axvspan(ci_-HALF, xT, color="#c0392b", alpha=0.05, zorder=0)
            if 0<=nT<=1: ax.plot([xT,xT],[0.50,0.90],ls="--",color="#c0392b",lw=1.2,zorder=5)
            ndrop_tot+=int(drop.sum())
        ax.scatter(xpos[keep],y[keep],color=CCOL[ct],s=52,alpha=0.85,edgecolor="white",linewidth=0.6,zorder=3)
        ax.scatter(xpos[drop],y[drop],facecolor="none",edgecolor="#b0b0b0",s=46,linewidth=1.1,zorder=3)
        if with_reg and len(lx)>=3:
            b,a0=np.polyfit(lx,y,1); xx=np.linspace(0,1,60); lxx=lo+xx*rng; yh=b*lxx+a0
            resid=y-(b*lx+a0); se=np.sqrt(np.sum(resid**2)/(len(lx)-2)); Sxx=np.sum((lx-lx.mean())**2)
            tv=stats.t.ppf(0.975,len(lx)-2); band=tv*se*np.sqrt(1/len(lx)+(lxx-lx.mean())**2/Sxx)
            xb=ci_-HALF+xx*(2*HALF)
            ax.plot(xb,yh,color=CCOL[ct],lw=2.2,zorder=4); ax.fill_between(xb,yh-band,yh+band,color=CCOL[ct],alpha=0.15,zorder=1)
            r,pp=stats.pearsonr(lx,y); msg.append(f"{ct}:r={r:.2f}(p={pp:.2g})")
        ax.text(ci_,0.515,f"{int(10**lo)}\u2013{int(10**hi)} cells",ha="center",va="bottom",fontsize=8.5,color="#666")
        if THRESH: ax.text(ci_,0.905,f"drop {int(drop.sum())}/{len(n)}",ha="center",va="bottom",fontsize=8,color="#c0392b")
        if ci_>0: ax.axvline(ci_-0.5,color="#ddd",lw=1,zorder=0)
    ax.axhline(0.5,ls=":",color="gray")
    ax.set_xticks(range(len(CTS))); ax.set_xticklabels(CTS,fontsize=12,fontweight="bold")
    ax.set_xlim(-0.6,len(CTS)-0.4); ax.set_ylim(0.50,0.92); ax.set_ylabel(YLAB); ax.grid(axis="y",alpha=0.25)
    tag="with log\u2081\u2080-n linear fit (+95% CI)" if with_reg else "no fit"
    thr_txt=f" \u00b7 red dashed = {THRESH}-cell cutoff (hollow dots dropped, {ndrop_tot} total)" if THRESH else ""
    fig.suptitle(f"Per-patient embedding quality vs. depth \u2014 one panel, cell types as columns \u2014 {tag}\n"
                 "Each dot = one patient\u00d7cell-type embedding (winning config: propr \u00b7 bidir \u00b7 k=50). y = combined benchmark AUC."+thr_txt,y=1.02)
    fig.text(0.5,-0.035,"within each column: fewer cells (left) \u2192 more cells (right)",ha="center",va="top",fontsize=9,style="italic",color="#555")
    if with_reg: fig.text(0.5,-0.02," | ".join(msg),ha="center",va="top",fontsize=9,color="#444")
    fig.tight_layout(rect=[0,0.01,1,0.98])
    out=FIG/("fig4_perpatient_vs_ncells_"+("withreg" if with_reg else "noreg")+".png")
    fig.savefig(out); plt.close(fig); print(f"fig4 ({tag}) OK n={len(d)} dropped@{THRESH}={ndrop_tot} |"," ".join(msg))

for fn,a in [(fig1,()),(fig8,()),(fig4_single,(False,)),(fig4_single,(True,))]:
    try: fn(*a)
    except Exception as e:
        import traceback; print(f"[ERR] {fn.__name__}: {e}"); traceback.print_exc()
