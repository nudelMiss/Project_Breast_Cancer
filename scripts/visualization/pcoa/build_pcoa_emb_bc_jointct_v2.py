#!/usr/bin/env python3
"""Embedding PCoA (batch-corrected): per-patient propr models, strong cell types, 4 multi-type cohorts.
Distance = 1 - corr of gene x gene cosine profiles on shared anchor genes, with within-cohort feature
centering (batch correction). Writes pcoa_embedding_jointct_batchcorrected.png (new file)."""
import re
from pathlib import Path
import numpy as np, pandas as pd
from gensim.models import Word2Vec
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

REPO=Path("/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast")
TAG="propr_bidirectional_w10_k50_var75_hvg2000"
DATASETS=["bassez2021","wu2021","qian2020","pal2021"]
EXP={"bassez2021":"exports_bassez_v2","wu2021":"exports_wu_v2","qian2020":"exports_qian","pal2021":"exports_pal2021"}
KEEP={'Malignant':('Malignant','#D62728'),'T_cell':('T / Lymphoid','#2CA02C'),'Lymphoid':('T / Lymphoid','#2CA02C'),
      'B_cell':('B cells','#1F77B4'),'Plasmablast':('B cells','#1F77B4'),
      'Myeloid':('Myeloid','#9467BD'),'Macrophage':('Myeloid','#9467BD'),'Monocyte':('Myeloid','#9467BD'),
      'Fibroblast':('Fibroblast / Stroma','#8C564B'),'Pericyte':('Fibroblast / Stroma','#8C564B'),
      'Endothelial':('Endothelial','#FF7F0E')}
MK={"bassez2021":"o","wu2021":"s","qian2020":"^","pal2021":"D"}
DSLAB={"bassez2021":"Bassez2021","wu2021":"Wu2021","qian2020":"Qian2020","pal2021":"Pal2021"}
OUT=REPO/"scripts/visualization/pcoa_v2"

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

JTAG=TAG+"_jointct"
recs=[]; KV={}
for ds in DATASETS:
    for md in sorted((REPO/(f"results/{ds}_v2/models_joint_by_celltype" if ds in ("bassez2021","wu2021") else f"results/{ds}/models_joint_by_celltype")).glob(f"celltype=*/{JTAG}/gene_embeddings.model")):
        ct=md.parent.parent.name.replace("celltype=","")
        if ct in KEEP:
            kv=Word2Vec.load(str(md)).wv; KV[len(recs)]=kv
            recs.append(dict(ds=ds,ct=ct,broad=KEEP[ct][0],color=KEEP[ct][1],vocab=set(kv.index_to_key)))
anchor=sorted(set.intersection(*[r['vocab'] for r in recs]))
print(f"{len(recs)} models, anchor(100% intersection)={len(anchor)} genes",flush=True)
iu=np.triu_indices(len(anchor),k=1)
P=np.zeros((len(recs),len(iu[0])),dtype=np.float32)
for i in range(len(recs)):
    kv=KV[i]; M=np.array([kv[g] for g in anchor],dtype=np.float64); M/=(np.linalg.norm(M,axis=1,keepdims=True)+1e-12)
    P[i]=(M@M.T)[iu]
# raw (pre-BC) split for reference
co0,_=pcoa(1-np.corrcoef(P)); 
print(f"[pre-BC] cell type={vexp(co0,[r['broad'] for r in recs]):.1f}%  dataset={vexp(co0,[r['ds'] for r in recs]):.1f}%",flush=True)
# within-cohort batch correction on profile features
ds_arr=np.array([r['ds'] for r in recs]); Pc=P.astype(np.float64).copy()
for ds in set(ds_arr):
    idx=np.where(ds_arr==ds)[0]; Pc[idx]-=Pc[idx].mean(0,keepdims=True)
C=np.corrcoef(Pc); D=1-C; np.fill_diagonal(D,0); coords,ve=pcoa(D)
print(f"[post-BC] cell type={vexp(coords,[r['broad'] for r in recs]):.1f}%  dataset={vexp(coords,[r['ds'] for r in recs]):.1f}%  (PCo1={ve[0]:.1f}% PCo2={ve[1]:.1f}%)",flush=True)
meta=pd.DataFrame([{'ds':r['ds'],'ct':r['ct'],'broad':r['broad']} for r in recs]); meta['PCo1']=coords[:,0]; meta['PCo2']=coords[:,1]
meta.to_csv(OUT/"pcoa_embedding_jointct_bc_coords.csv",index=False)

fig,ax=plt.subplots(figsize=(12,8))
for col in {r['color'] for r in recs}:
    pts=np.array([coords[i] for i,r in enumerate(recs) if r['color']==col]); ell(ax,pts,col)
for i,r in enumerate(recs):
    ax.scatter(coords[i,0],coords[i,1],c=r['color'],marker=MK[r['ds']],s=58,alpha=0.82,edgecolors='white',linewidths=0.4,zorder=3)
ax.set_xlabel(f"PCoA 1 ({ve[0]:.1f}%)"); ax.set_ylabel(f"PCoA 2 ({ve[1]:.1f}%)")
ax.set_title("PCoA \u2014 batch-corrected propr embeddings (joint-by-cell-type, strong cell types, 4 cohorts)")
seen={}
for r in recs: seen[r['broad']]=r['color']
h1=[Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=11,label=l) for l,c in seen.items()]
h2=[Line2D([0],[0],marker=MK[d],color='w',markerfacecolor='grey',markeredgecolor='k',markersize=10,label=DSLAB[d]) for d in DATASETS]
leg1=ax.legend(handles=h1,title="Cell type",loc='upper left',bbox_to_anchor=(1.02,1.0),fontsize=10); ax.add_artist(leg1)
leg2=ax.legend(handles=h2,title="Cohort",loc='upper left',bbox_to_anchor=(1.02,1.0-0.075*(len(h1)+1.8)),fontsize=10)
fig.savefig(OUT/"pcoa_embedding_jointct_batchcorrected.png",dpi=160,bbox_inches='tight',bbox_extra_artists=(leg1,leg2))
plt.close(fig); print("[FIG] pcoa_embedding_jointct_batchcorrected.png",flush=True); print("DONE",flush=True)
