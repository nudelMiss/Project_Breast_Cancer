#!/usr/bin/env python3
"""Batch-corrected pseudobulk PCoA: within-cohort gene centering + cell-type ellipses.
Writes pcoa_pseudobulk_batchcorrected_strong.png (new file)."""
import re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.io import mmread
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

REPO=Path("/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast")
EXP={"bassez2021":"exports_bassez_v2","wu2021":"exports_wu_v2","griffiths2021":"exports_griffiths",
     "qian2020":"exports_qian","pal2021":"exports_pal2021","azizi2018":"exports_azizi","gao2021":"exports_gao"}
ALL_DS=list(EXP)
COLOR={'Malignant':'#D62728','T_cell':'#2CA02C','Lymphoid':'#2CA02C','B_cell':'#1F77B4',
       'Plasmablast':'#1F77B4','Myeloid':'#9467BD','Macrophage':'#9467BD','Monocyte':'#9467BD',
       'Fibroblast':'#8C564B','Pericyte':'#8C564B','Endothelial':'#FF7F0E'}
BROAD={'#D62728':'Malignant','#BCBD22':'Epithelial','#2CA02C':'T / Lymphoid','#1F77B4':'B / Plasma','#17BECF':'NK',
       '#9467BD':'Myeloid','#8C564B':'Fibroblast / Stroma','#FF7F0E':'Endothelial'}
MK={"bassez2021":"o","wu2021":"s","griffiths2021":"^","qian2020":"D","pal2021":"v","azizi2018":"P","gao2021":"X"}
DSLAB={"bassez2021":"Bassez2021","wu2021":"Wu2021","griffiths2021":"Griffiths2021","qian2020":"Qian2020",
       "pal2021":"Pal2021","azizi2018":"Azizi2018","gao2021":"Gao2021"}
OUT=REPO/"crosscohort_viz/pcoa_v2"

def cts_for(ds):
    s=set()
    for d in (REPO/EXP[ds]).glob("patient=*"):
        m=re.search(r"celltype=(.+)$",d.name)
        if m: s.add(m.group(1))
    return sorted(s,key=len,reverse=True)
def assign(g,cts):
    for c in cts:
        if g==c or g.endswith("_"+c): return c
    return None
def ell(ax,pts,color,nstd=2.0):
    if len(pts)<4: return
    cov=np.cov(pts.T); mu=pts.mean(0); vals,vecs=np.linalg.eigh(cov); o=vals.argsort()[::-1]
    vals,vecs=vals[o],vecs[:,o]; ang=np.degrees(np.arctan2(vecs[1,0],vecs[0,0])); ww,hh=2*nstd*np.sqrt(np.clip(vals,0,None))
    ax.add_patch(Ellipse(mu,ww,hh,angle=ang,fill=False,edgecolor=color,ls='--',lw=2,alpha=0.85,zorder=1))
def pcoa(D):
    n=D.shape[0]; J=np.eye(n)-np.ones((n,n))/n; B=-0.5*J@(D**2)@J
    w,V=np.linalg.eigh(B); o=np.argsort(w)[::-1]; w=w[o]; V=V[:,o]
    return V[:,:2]*np.sqrt(np.clip(w[:2],0,None)), (np.clip(w,0,None)[:2]/np.clip(w,0,None).sum()*100)
def vexp(coords,labels):
    gm=coords.mean(0); tot=((coords-gm)**2).sum(); b=0.0
    for u in set(labels):
        idx=[i for i,l in enumerate(labels) if l==u]; m=coords[idx].mean(0); b+=len(idx)*((m-gm)**2).sum()
    return b/tot*100

recs=[]; pbd=[]
for ds in ALL_DS:
    cts=cts_for(ds)
    for gd in sorted((REPO/EXP[ds]).glob("patient=*")):
        ct=assign(gd.name.split("celltype=")[-1],cts)
        if ct not in COLOR: continue
        genes=[l.strip() for l in open(gd/"genes.csv")]
        M=mmread(str(gd/"expr.mtx")).tocsr(); tot=np.asarray(M.sum(1)).ravel().astype(float)
        n=min(len(genes),M.shape[0]); genes=genes[:n]; tot=tot[:n]
        cpm=np.log1p(tot/(tot.sum()+1e-9)*1e6)
        recs.append(dict(ds=ds,ct=ct,color=COLOR[ct])); pbd.append(dict(zip(genes,cpm)))
shared=set(pbd[0])
for d in pbd[1:]: shared&=set(d)
shared=sorted(shared)
X=np.array([[d[g] for g in shared] for d in pbd])
print(f"{len(recs)} groups, shared genes={len(shared)}",flush=True)
# ---- within-cohort batch correction (center each gene per cohort) ----
ds_arr=np.array([r['ds'] for r in recs]); Xc=X.copy()
for ds in set(ds_arr):
    idx=np.where(ds_arr==ds)[0]; Xc[idx]-=Xc[idx].mean(0,keepdims=True)
v=Xc.var(0); top=np.argsort(v)[::-1][:2000]; Xh=Xc[:,top]
C=np.corrcoef(Xh); D=1-C; np.fill_diagonal(D,0)
coords,ve=pcoa(D)
print(f"[VAR after BC] cell type={vexp(coords,[r['color'] for r in recs]):.1f}%  dataset={vexp(coords,[r['ds'] for r in recs]):.1f}%  (PCo1={ve[0]:.1f}% PCo2={ve[1]:.1f}%)",flush=True)
meta=pd.DataFrame([{'ds':r['ds'],'ct':r['ct']} for r in recs]); meta['PCo1']=coords[:,0]; meta['PCo2']=coords[:,1]
meta.to_csv(OUT/"pcoa_pseudobulk_bc_strong_coords.csv",index=False)

fig,ax=plt.subplots(figsize=(12,8))
for col in {r['color'] for r in recs}:
    pts=np.array([coords[i] for i,r in enumerate(recs) if r['color']==col]); ell(ax,pts,col)
for i,r in enumerate(recs):
    ax.scatter(coords[i,0],coords[i,1],c=r['color'],marker=MK[r['ds']],s=58,alpha=0.82,edgecolors='white',linewidths=0.4,zorder=3)
ax.set_xlabel(f"PCoA 1 ({ve[0]:.1f}%)"); ax.set_ylabel(f"PCoA 2 ({ve[1]:.1f}%)")
ax.set_title("PCoA \u2014 batch-corrected pseudobulk, strong cell types (sample \u00d7 cell type), 7 cohorts")
seen={}
for r in recs: seen[BROAD.get(r['color'],r['ct'])]=r['color']
h1=[Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=11,label=l) for l,c in seen.items()]
dss=[d for d in ALL_DS if any(r['ds']==d for r in recs)]
h2=[Line2D([0],[0],marker=MK[d],color='w',markerfacecolor='grey',markeredgecolor='k',markersize=10,label=DSLAB[d]) for d in dss]
leg1=ax.legend(handles=h1,title="Cell type",loc='upper left',bbox_to_anchor=(1.02,1.0),fontsize=10,frameon=True); ax.add_artist(leg1)
y2=1.0-0.075*(len(h1)+1.8)
leg2=ax.legend(handles=h2,title="Cohort",loc='upper left',bbox_to_anchor=(1.02,y2),fontsize=10,frameon=True)
fig.savefig(OUT/"pcoa_pseudobulk_batchcorrected_strong.png",dpi=160,bbox_inches='tight',bbox_extra_artists=(leg1,leg2))
plt.close(fig); print("[FIG] pcoa_pseudobulk_batchcorrected_strong.png",flush=True); print("DONE",flush=True)
