#!/usr/bin/env python3
"""UMAP of per-(patient x cell-type) propr embeddings -- representation B
(flattened anchor gene-gene cosine profile per group), all 7 cohorts.
Two panels: pre-batch-correction vs within-cohort feature centering.
Color = broad cell type; marker shape = cohort (batch-effect check).
Outputs to scripts/visualization/umap/ (yuvat cannot write michalnu's results/multidataset)."""
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
OUT=REPO/"scripts/visualization/umap"; OUT.mkdir(parents=True, exist_ok=True)
TAG="propr_bidirectional_w10_k50_var75_hvg2000"
DATASETS=["bassez2021","wu2021","qian2020","pal2021","gao2021","azizi2018"]
EXP={"bassez2021":"exports_bassez_v2","wu2021":"exports_wu_v2","qian2020":"exports_qian",
     "pal2021":"exports_pal2021","gao2021":"exports_gao","azizi2018":"exports_azizi"}
KEEP={'Malignant':('Malignant','#D62728'),'T_cell':('T / Lymphoid','#2CA02C'),'Lymphoid':('T / Lymphoid','#2CA02C'),
      'B_cell':('B cells','#1F77B4'),'Plasmablast':('B cells','#1F77B4'),
      'Myeloid':('Myeloid','#9467BD'),'Macrophage':('Myeloid','#9467BD'),'Monocyte':('Myeloid','#9467BD'),
      'Fibroblast':('Fibroblast / Stroma','#8C564B'),'Pericyte':('Fibroblast / Stroma','#8C564B'),
      'Endothelial':('Endothelial','#FF7F0E')}
MK={"bassez2021":"o","wu2021":"s","qian2020":"^","pal2021":"D","gao2021":"P","azizi2018":"X"}
DSLAB={"bassez2021":"Bassez2021","wu2021":"Wu2021","qian2020":"Qian2020","pal2021":"Pal2021",
       "gao2021":"Gao2021","azizi2018":"Azizi2018"}
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
    for md in sorted((REPO/(f"results/{ds}_v2/models" if ds in ("bassez2021","wu2021") else f"results/{ds}/stageA/models")).glob(f"*/{TAG}/gene_embeddings.model")):
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

# ===== LDA projection (SUPERVISED; separation is guaranteed, CV accuracy is the evidence) =====
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

broad=np.array([r['broad'] for r in recs]); dsl=[r['ds'] for r in recs]
N_PCA=min(50, len(recs)-1)
nsplits=max(2, min(5, min(Counter(broad).values())))
print(f"class counts: {dict(Counter(broad))}  | PCA={N_PCA}  CV folds={nsplits}", flush=True)

Pc=P.astype(np.float64).copy(); dsa=np.array(dsl)
for ds in set(dsa):
    idx=np.where(dsa==ds)[0]; Pc[idx]-=Pc[idx].mean(0,keepdims=True)

def lda_project(X):
    Z=PCA(n_components=N_PCA,random_state=42).fit_transform(StandardScaler().fit_transform(X))
    return LDA(n_components=2).fit_transform(Z, broad)
def cv_report(X, tag):
    pipe=Pipeline([('sc',StandardScaler()),('pca',PCA(n_components=N_PCA,random_state=42)),('lda',LDA())])
    skf=StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=42)
    acc=cross_val_score(pipe, X, broad, cv=skf, scoring='accuracy')
    rng=np.random.RandomState(0); yb=broad.copy(); rng.shuffle(yb)
    accs=cross_val_score(pipe, X, yb, cv=skf, scoring='accuracy')
    pipe.fit(X, broad); tr=pipe.score(X, broad)
    print(f"[{tag}] train_acc={tr:.3f}  cv_acc={acc.mean():.3f}+/-{acc.std():.3f}  cv_shuffled={accs.mean():.3f}  chance={1/len(set(broad)):.3f}", flush=True)
    return acc.mean(), accs.mean()

ld_raw=lda_project(P.astype(np.float64)); a_raw=cv_report(P.astype(np.float64),"raw")
ld_bc =lda_project(Pc);                   a_bc =cv_report(Pc,"batch-corrected")

def make_lda(draw_ellipse,name):
    fig,axes=plt.subplots(1,2,figsize=(20,8))
    for ax,co,ttl,cva in [(axes[0],ld_raw,"RAW profiles",a_raw),(axes[1],ld_bc,"Batch-corrected",a_bc)]:
        if draw_ellipse:
            for col in {r['color'] for r in recs}:
                pts=np.array([co[i] for i,r in enumerate(recs) if r['color']==col]); ell(ax,pts,col)
        for i,r in enumerate(recs):
            ax.scatter(co[i,0],co[i,1],c=r['color'],marker=MK[r['ds']],s=58,alpha=0.82,edgecolors='white',linewidths=0.4,zorder=3)
        ax.set_xlabel("LD 1"); ax.set_ylabel("LD 2")
        ax.set_title(f"{ttl}   5-fold CV acc={cva[0]:.2f}  (shuffled label ctrl={cva[1]:.2f})")
    seen={}
    for r in recs: seen[r['broad']]=r['color']
    h1=[Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=11,label=l) for l,c in seen.items()]
    h2=[Line2D([0],[0],marker=MK[d],color='w',markerfacecolor='grey',markeredgecolor='k',markersize=10,label=DSLAB[d]) for d in DATASETS]
    l1=axes[1].legend(handles=h1,title="Cell type",loc='upper left',bbox_to_anchor=(1.02,1.0),fontsize=10); axes[1].add_artist(l1)
    axes[1].legend(handles=h2,title="Cohort",loc='upper left',bbox_to_anchor=(1.02,0.45),fontsize=10)
    fig.suptitle(f"LDA (SUPERVISED) of per-patient propr embeddings -- {len(recs)} groups, 6 cohorts, anchor={len(anchor)} genes, PCA={N_PCA}",fontsize=13)
    fig.savefig(OUT/name,dpi=160,bbox_inches='tight'); plt.close(fig); print("[FIG]",name,flush=True)

make_lda(True ,"lda_emb_6cohort_v2_ellipse.png")
make_lda(False,"lda_emb_6cohort_v2_noellipse.png")
pd.DataFrame({'ds':dsl,'ct':[r['ct'] for r in recs],'broad':list(broad),
  'LD1_raw':ld_raw[:,0],'LD2_raw':ld_raw[:,1],'LD1_bc':ld_bc[:,0],'LD2_bc':ld_bc[:,1]}).to_csv(OUT/"lda_emb_6cohort_v2_coords.csv",index=False)
print("DONE_LDA",flush=True)
