#!/usr/bin/env python3
"""Top-K embedding PCoA (batch-corrected): select the top-K (patient x celltype) groups by
cell count, within each cohort x strong cell type, then run the SAME embedding-PCoA method as
build_pcoa_emb_bc.py (gene x gene cosine profile on 100%-intersection anchor -> within-cohort
feature centering (batch correction) -> classical MDS). Tests whether per-patient embeddings
fail only because of low cell counts. Writes pcoa_embedding_topk_batchcorrected.png (NEW file)."""
import re
from pathlib import Path
import numpy as np, pandas as pd
from gensim.models import Word2Vec
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

import sys
K=int(sys.argv[1]) if len(sys.argv)>1 else 5
REPO=Path("/groups/ofircohen-group/users/michalnu_yuvat/project_breast")
TAG="propr_bidirectional_w10_k50_var75_hvg2000"
DATASETS=["bassez2021","wu2021","qian2020","pal2021"]
EXP={"bassez2021":"exports_bassez","wu2021":"exports_wu_counts","qian2020":"exports_qian","pal2021":"exports_pal2021"}
BROAD={'Malignant':('Malignant','#D62728'),'T_cell':('T / Lymphoid','#2CA02C'),'Lymphoid':('T / Lymphoid','#2CA02C'),
       'B_cell':('B cells','#1F77B4'),'Plasmablast':('B cells','#1F77B4'),
       'Myeloid':('Myeloid','#9467BD'),'Macrophage':('Myeloid','#9467BD'),'Monocyte':('Myeloid','#9467BD'),
       'Fibroblast':('Fibroblast / Stroma','#8C564B'),'Pericyte':('Fibroblast / Stroma','#8C564B'),
       'Endothelial':('Endothelial','#FF7F0E')}
MK={"bassez2021":"o","wu2021":"s","qian2020":"^","pal2021":"D"}
DSLAB={"bassez2021":"Bassez2021","wu2021":"Wu2021","qian2020":"Qian2020","pal2021":"Pal2021"}
OUT=REPO/"results/multidataset/pcoa"

def ncells(d):
    cf=d/"cells.csv"
    return sum(1 for _ in open(cf)) if cf.exists() else None
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

# 1) candidate strong-celltype groups with a model + cell count
cand=[]
for ds in DATASETS:
    for d in sorted((REPO/EXP[ds]).glob("patient=*__celltype=*")):
        m=re.match(r"patient=(.+)__celltype=(.+)$",d.name)
        if not m: continue
        patient,ct=m.group(1),m.group(2)
        if ct not in BROAD: continue
        broad,color=BROAD[ct]
        md=REPO/f"results/{ds}/stageA/models/{patient}_{ct}/{TAG}/gene_embeddings.model"
        if not md.exists(): continue
        cand.append(dict(ds=ds,patient=patient,ct=ct,broad=broad,color=color,
                         ncells=ncells(d),model=str(md)))
cd=pd.DataFrame(cand)
# 2) top-K within cohort x broad cell type (take all if fewer than K)
sel=(cd.sort_values('ncells',ascending=False)
       .groupby(['ds','broad'],group_keys=False).head(K)
       .reset_index(drop=True))
sel.drop(columns=['model']).to_csv(OUT/f"pcoa_embedding_top{K}_selected.csv",index=False)
print(f"K={K}: selected {len(sel)} of {len(cd)} candidate per-patient groups",flush=True)
print(sel.pivot_table(index='ds',columns='broad',values='patient',aggfunc='count',fill_value=0).to_string(),flush=True)

# 3) load embeddings (same method as build_pcoa_emb_bc.py)
recs=[]; KV={}
for _,r in sel.iterrows():
    kv=Word2Vec.load(r['model']).wv; KV[len(recs)]=kv
    recs.append(dict(ds=r['ds'],ct=r['ct'],broad=r['broad'],color=r['color'],
                     ncells=int(r['ncells']),vocab=set(kv.index_to_key)))
anchor=sorted(set.intersection(*[r['vocab'] for r in recs]))
print(f"{len(recs)} models, anchor(100% intersection)={len(anchor)} genes",flush=True)
iu=np.triu_indices(len(anchor),k=1)
P=np.zeros((len(recs),len(iu[0])),dtype=np.float32)
for i in range(len(recs)):
    kv=KV[i]; M=np.array([kv[g] for g in anchor],dtype=np.float64); M/=(np.linalg.norm(M,axis=1,keepdims=True)+1e-12)
    P[i]=(M@M.T)[iu]
# pre-BC reference
co0,_=pcoa(1-np.corrcoef(P))
pre_ct=vexp(co0,[r['broad'] for r in recs]); pre_ds=vexp(co0,[r['ds'] for r in recs])
print(f"[pre-BC] cell type={pre_ct:.1f}%  dataset={pre_ds:.1f}%",flush=True)
# within-cohort batch correction
ds_arr=np.array([r['ds'] for r in recs]); Pc=P.astype(np.float64).copy()
for ds in set(ds_arr):
    idx=np.where(ds_arr==ds)[0]; Pc[idx]-=Pc[idx].mean(0,keepdims=True)
C=np.corrcoef(Pc); D=1-C; np.fill_diagonal(D,0); coords,ve=pcoa(D)
post_ct=vexp(coords,[r['broad'] for r in recs]); post_ds=vexp(coords,[r['ds'] for r in recs])
print(f"[post-BC] cell type={post_ct:.1f}%  dataset={post_ds:.1f}%  (PCo1={ve[0]:.1f}% PCo2={ve[1]:.1f}%)",flush=True)
meta=pd.DataFrame([{'ds':r['ds'],'ct':r['ct'],'broad':r['broad'],'ncells':r['ncells']} for r in recs])
meta['PCo1']=coords[:,0]; meta['PCo2']=coords[:,1]
meta.to_csv(OUT/f"pcoa_embedding_top{K}_bc_coords.csv",index=False)

# 4) figure
fig,ax=plt.subplots(figsize=(12,8))
for col in {r['color'] for r in recs}:
    pts=np.array([coords[i] for i,r in enumerate(recs) if r['color']==col]); ell(ax,pts,col)
for i,r in enumerate(recs):
    ax.scatter(coords[i,0],coords[i,1],c=r['color'],marker=MK[r['ds']],s=58,alpha=0.82,edgecolors='white',linewidths=0.4,zorder=3)
ax.set_xlabel(f"PCoA 1 ({ve[0]:.1f}%)"); ax.set_ylabel(f"PCoA 2 ({ve[1]:.1f}%)")
ax.set_title(f"PCoA \u2014 batch-corrected propr embeddings (top-{K} per cohort\u00d7cell type by cell count, n={len(recs)})")
seen={}
for r in recs: seen[r['broad']]=r['color']
h1=[Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=11,label=l) for l,c in seen.items()]
h2=[Line2D([0],[0],marker=MK[d],color='w',markerfacecolor='grey',markeredgecolor='k',markersize=10,label=DSLAB[d]) for d in DATASETS]
leg1=ax.legend(handles=h1,title="Cell type",loc='upper left',bbox_to_anchor=(1.02,1.0),fontsize=10); ax.add_artist(leg1)
leg2=ax.legend(handles=h2,title="Cohort",loc='upper left',bbox_to_anchor=(1.02,1.0-0.075*(len(h1)+1.8)),fontsize=10)
fig.savefig(OUT/f"pcoa_embedding_top{K}_batchcorrected.png",dpi=160,bbox_inches='tight',bbox_extra_artists=(leg1,leg2))
plt.close(fig); print(f"[FIG] pcoa_embedding_top{K}_batchcorrected.png",flush=True)
print(f"SUMMARY pre-BC ct={pre_ct:.1f}/ds={pre_ds:.1f}  post-BC ct={post_ct:.1f}/ds={post_ds:.1f}  n={len(recs)} anchor={len(anchor)}",flush=True)
print("DONE",flush=True)
