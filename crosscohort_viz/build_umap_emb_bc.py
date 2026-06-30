#!/usr/bin/env python3
"""UMAP of per-(patient x cell-type) propr embeddings -- representation B
(flattened anchor gene-gene cosine profile per group), all 7 cohorts.
Two panels: pre-batch-correction vs within-cohort feature centering.
Color = broad cell type; marker shape = cohort (batch-effect check).
Outputs to crosscohort_viz/umap/ (yuvat cannot write michalnu's results/multidataset)."""
import re
from pathlib import Path
import numpy as np, pandas as pd
from gensim.models import Word2Vec
import umap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

REPO=Path("/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast")
OUT=REPO/"crosscohort_viz/umap"; OUT.mkdir(parents=True, exist_ok=True)
TAG="propr_bidirectional_w10_k50_var75_hvg2000"
DATASETS=["bassez2021","wu2021","qian2020","pal2021","griffiths2021","gao2021","azizi2018"]
EXP={"bassez2021":"exports_bassez","wu2021":"exports_wu_counts","qian2020":"exports_qian",
     "pal2021":"exports_pal2021","griffiths2021":"exports_griffiths","gao2021":"exports_gao","azizi2018":"exports_azizi"}
KEEP={'Malignant':('Malignant','#D62728'),'T_cell':('T / Lymphoid','#2CA02C'),'Lymphoid':('T / Lymphoid','#2CA02C'),
      'B_cell':('B cells','#1F77B4'),'Plasmablast':('B cells','#1F77B4'),
      'Myeloid':('Myeloid','#9467BD'),'Macrophage':('Myeloid','#9467BD'),'Monocyte':('Myeloid','#9467BD'),
      'Fibroblast':('Fibroblast / Stroma','#8C564B'),'Pericyte':('Fibroblast / Stroma','#8C564B'),
      'Endothelial':('Endothelial','#FF7F0E')}
MK={"bassez2021":"o","wu2021":"s","qian2020":"^","pal2021":"D","griffiths2021":"v","gao2021":"P","azizi2018":"X"}
DSLAB={"bassez2021":"Bassez2021","wu2021":"Wu2021","qian2020":"Qian2020","pal2021":"Pal2021",
       "griffiths2021":"Griffiths2021","gao2021":"Gao2021","azizi2018":"Azizi2018"}
SEED=42; N_NEIGHBORS=15; MIN_DIST=0.1

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
def sep_pct(coords,labels):
    gm=coords.mean(0); tot=((coords-gm)**2).sum(); b=0.0
    for u in set(labels):
        idx=[i for i,l in enumerate(labels) if l==u]; m=coords[idx].mean(0); b+=len(idx)*((m-gm)**2).sum()
    return b/tot*100 if tot>0 else 0.0

recs=[]; KV={}
for ds in DATASETS:
    cts=cts_for(ds); nkept=0
    for md in sorted((REPO/f"results/{ds}/stageA/models").glob(f"*/{TAG}/gene_embeddings.model")):
        ct=assign(md.parent.parent.name,cts)
        if ct in KEEP:
            kv=Word2Vec.load(str(md)).wv; KV[len(recs)]=kv
            recs.append(dict(ds=ds,ct=ct,broad=KEEP[ct][0],color=KEEP[ct][1],vocab=set(kv.index_to_key)))
            nkept+=1
    print(f"{ds}: {nkept} kept groups",flush=True)
print(f"TOTAL points={len(recs)}",flush=True)

anchor=sorted(set.intersection(*[r['vocab'] for r in recs]))
print(f"ANCHOR (100% intersection, all cohorts) = {len(anchor)} genes",flush=True)
for ds in DATASETS:
    others=[r['vocab'] for r in recs if r['ds']!=ds]
    if others:
        print(f"  anchor WITHOUT {ds}: {len(set.intersection(*others))} genes",flush=True)

iu=np.triu_indices(len(anchor),k=1)
P=np.zeros((len(recs),len(iu[0])),dtype=np.float32)
for i in range(len(recs)):
    kv=KV[i]; M=np.array([kv[g] for g in anchor],dtype=np.float64); M/=(np.linalg.norm(M,axis=1,keepdims=True)+1e-12)
    P[i]=(M@M.T)[iu]

broad=[r['broad'] for r in recs]; dsl=[r['ds'] for r in recs]
def run_umap(X):
    return umap.UMAP(n_neighbors=N_NEIGHBORS,min_dist=MIN_DIST,metric='correlation',random_state=SEED).fit_transform(X)

co_raw=run_umap(P.astype(np.float64))
print(f"[pre-BC]  cell-type sep={sep_pct(co_raw,broad):.1f}%  dataset sep={sep_pct(co_raw,dsl):.1f}%",flush=True)
Pc=P.astype(np.float64).copy(); ds_arr=np.array(dsl)
for ds in set(ds_arr):
    idx=np.where(ds_arr==ds)[0]; Pc[idx]-=Pc[idx].mean(0,keepdims=True)
co_bc=run_umap(Pc)
print(f"[post-BC] cell-type sep={sep_pct(co_bc,broad):.1f}%  dataset sep={sep_pct(co_bc,dsl):.1f}%",flush=True)

meta=pd.DataFrame({'ds':dsl,'ct':[r['ct'] for r in recs],'broad':broad,
    'UMAP1_raw':co_raw[:,0],'UMAP2_raw':co_raw[:,1],'UMAP1_bc':co_bc[:,0],'UMAP2_bc':co_bc[:,1]})
meta.to_csv(OUT/"umap_embedding_coords.csv",index=False)

fig,axes=plt.subplots(1,2,figsize=(20,8))
for ax,coords,ttl in [(axes[0],co_raw,"Pre-correction (raw profiles)"),(axes[1],co_bc,"Within-cohort centered (batch-corrected)")]:
    for col in {r['color'] for r in recs}:
        pts=np.array([coords[i] for i,r in enumerate(recs) if r['color']==col]); ell(ax,pts,col)
    for i,r in enumerate(recs):
        ax.scatter(coords[i,0],coords[i,1],c=r['color'],marker=MK[r['ds']],s=58,alpha=0.82,edgecolors='white',linewidths=0.4,zorder=3)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.set_title(ttl)
seen={}
for r in recs: seen[r['broad']]=r['color']
h1=[Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=11,label=l) for l,c in seen.items()]
h2=[Line2D([0],[0],marker=MK[d],color='w',markerfacecolor='grey',markeredgecolor='k',markersize=10,label=DSLAB[d]) for d in DATASETS]
leg1=axes[1].legend(handles=h1,title="Cell type",loc='upper left',bbox_to_anchor=(1.02,1.0),fontsize=10); axes[1].add_artist(leg1)
leg2=axes[1].legend(handles=h2,title="Cohort",loc='upper left',bbox_to_anchor=(1.02,1.0-0.075*(len(h1)+1.8)),fontsize=10)
fig.suptitle(f"UMAP of per-patient propr embeddings (B: gene-gene cosine profile) -- {len(recs)} groups, {len(DATASETS)} cohorts, anchor={len(anchor)} genes",fontsize=13)
fig.savefig(OUT/"umap_embedding_batchcorrected.png",dpi=160,bbox_inches='tight',bbox_extra_artists=(leg1,leg2))
plt.close(fig)
print("[FIG] crosscohort_viz/umap/umap_embedding_batchcorrected.png",flush=True); print("DONE",flush=True)
