#!/usr/bin/env python3
"""V2 rerun: PCoA + UMAP + LDA of per-(patient x cell-type) propr embeddings.
Bassez & Wu use V2 objects/models; Qian/Pal/Gao/Azizi unchanged. Representation B
(gene-gene cosine profile on shared anchor). Two panels each: raw vs within-cohort centered.
Outputs to crosscohort_viz/multidataset_v2/."""
import re, numpy as np, pandas as pd
from pathlib import Path
from gensim.models import Word2Vec
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import umap
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

REPO=Path("/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast")
OUT=REPO/"crosscohort_viz/multidataset_v2"; OUT.mkdir(parents=True, exist_ok=True)
TAG="propr_bidirectional_w10_k50_var75_hvg2000"
DATASETS=["bassez2021_v2","wu2021_v2","qian2020","pal2021","gao2021","azizi2018"]
EXP={"bassez2021_v2":"exports_bassez_v2","wu2021_v2":"exports_wu_v2","qian2020":"exports_qian",
     "pal2021":"exports_pal2021","gao2021":"exports_gao","azizi2018":"exports_azizi"}
KEEP={'Malignant':('Malignant','#D62728'),'T_cell':('T / Lymphoid','#2CA02C'),'Lymphoid':('T / Lymphoid','#2CA02C'),
      'B_cell':('B cells','#1F77B4'),'Plasmablast':('B cells','#1F77B4'),
      'Myeloid':('Myeloid','#9467BD'),'Macrophage':('Myeloid','#9467BD'),'Monocyte':('Myeloid','#9467BD'),
      'Fibroblast':('Fibroblast / Stroma','#8C564B'),'Pericyte':('Fibroblast / Stroma','#8C564B'),
      'Endothelial':('Endothelial','#FF7F0E')}
MK={"bassez2021_v2":"o","wu2021_v2":"s","qian2020":"^","pal2021":"D","gao2021":"P","azizi2018":"X"}
DSLAB={"bassez2021_v2":"Bassez2021 (V2)","wu2021_v2":"Wu2021 (V2)","qian2020":"Qian2020","pal2021":"Pal2021",
       "gao2021":"Gao2021","azizi2018":"Azizi2018"}
SEED=42; N_NEIGHBORS=15; MIN_DIST=0.1

def mdir(ds):
    a=REPO/f"results/{ds}/models"
    return a if a.exists() else REPO/f"results/{ds}/stageA/models"
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
    cov=np.cov(pts.T);mu=pts.mean(0);vals,vecs=np.linalg.eigh(cov);o=vals.argsort()[::-1];vals,vecs=vals[o],vecs[:,o]
    ang=np.degrees(np.arctan2(vecs[1,0],vecs[0,0]));ww,hh=2*nstd*np.sqrt(np.clip(vals,0,None))
    ax.add_patch(Ellipse(mu,ww,hh,angle=ang,fill=False,edgecolor=color,ls='--',lw=2,alpha=0.85,zorder=1))
def sep_pct(coords,labels):
    gm=coords.mean(0); tot=((coords-gm)**2).sum(); b=0.0
    for u in set(labels):
        idx=[i for i,l in enumerate(labels) if l==u]; m=coords[idx].mean(0); b+=len(idx)*((m-gm)**2).sum()
    return b/tot*100 if tot>0 else 0.0

recs=[]; KV={}
for ds in DATASETS:
    cts=cts_for(ds); nk=0
    for md in sorted(mdir(ds).glob(f"*/{TAG}/gene_embeddings.model")):
        ct=assign(md.parent.parent.name,cts)
        if ct in KEEP:
            kv=Word2Vec.load(str(md)).wv; KV[len(recs)]=kv
            recs.append(dict(ds=ds,ct=ct,broad=KEEP[ct][0],color=KEEP[ct][1],vocab=set(kv.index_to_key))); nk+=1
    print(f"{ds}: {nk} kept groups",flush=True)
print(f"TOTAL points={len(recs)}",flush=True)
anchor=sorted(set.intersection(*[r['vocab'] for r in recs]))
print(f"ANCHOR (6 cohorts, V2 bassez+wu) = {len(anchor)} genes",flush=True)
iu=np.triu_indices(len(anchor),k=1)
P=np.zeros((len(recs),len(iu[0])))
for i in range(len(recs)):
    kv=KV[i]; M=np.array([kv[g] for g in anchor]); M/=(np.linalg.norm(M,axis=1,keepdims=True)+1e-12); P[i]=(M@M.T)[iu]
broad=np.array([r['broad'] for r in recs]); dsl=[r['ds'] for r in recs]
Pc=P.copy(); dsa=np.array(dsl)
for ds in set(dsa):
    idx=np.where(dsa==ds)[0]; Pc[idx]-=Pc[idx].mean(0,keepdims=True)
COLORS=sorted({r['color'] for r in recs})

def two_panel(cr,cb,title,fname,draw_ellipse,xlab,ylab,subtitles):
    fig,axes=plt.subplots(1,2,figsize=(20,8))
    for ax,co,st in [(axes[0],cr,subtitles[0]),(axes[1],cb,subtitles[1])]:
        if draw_ellipse:
            for col in COLORS:
                pts=np.array([co[i] for i,r in enumerate(recs) if r['color']==col]); ell(ax,pts,col)
        for i,r in enumerate(recs):
            ax.scatter(co[i,0],co[i,1],c=r['color'],marker=MK[r['ds']],s=58,alpha=0.82,edgecolors='white',linewidths=0.4,zorder=3)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(st)
    seen={}
    for r in recs: seen[r['broad']]=r['color']
    h1=[Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=11,label=l) for l,c in seen.items()]
    h2=[Line2D([0],[0],marker=MK[d],color='w',markerfacecolor='grey',markeredgecolor='k',markersize=10,label=DSLAB[d]) for d in DATASETS]
    l1=axes[1].legend(handles=h1,title="Cell type",loc='upper left',bbox_to_anchor=(1.02,1.0),fontsize=10); axes[1].add_artist(l1)
    axes[1].legend(handles=h2,title="Cohort",loc='upper left',bbox_to_anchor=(1.02,0.45),fontsize=10)
    fig.suptitle(title,fontsize=13)
    fig.savefig(OUT/fname,dpi=160,bbox_inches='tight'); plt.close(fig); print("[FIG]",fname,flush=True)

def pcoa(D):
    n=len(D); J=np.eye(n)-1.0/n; B=-0.5*J@(D**2)@J
    w,v=np.linalg.eigh(B); o=w.argsort()[::-1]; w,v=w[o],v[:,o]
    L=np.sqrt(np.clip(w[:2],0,None)); pos=w[w>0].sum()
    return v[:,:2]*L, (w[:2]/pos*100 if pos>0 else np.array([0.0,0.0]))
def cdist(X): return 1-np.corrcoef(X)
pc_raw,vr=pcoa(cdist(P)); pc_bc,vb=pcoa(cdist(Pc))
print(f"[PCoA raw] ct={sep_pct(pc_raw,broad):.1f}% ds={sep_pct(pc_raw,dsl):.1f}% PCo1/2={vr[0]:.1f}/{vr[1]:.1f}%",flush=True)
print(f"[PCoA bc ] ct={sep_pct(pc_bc,broad):.1f}% ds={sep_pct(pc_bc,dsl):.1f}% PCo1/2={vb[0]:.1f}/{vb[1]:.1f}%",flush=True)
sr=f"Pre-correction (PCo1={vr[0]:.1f}%, PCo2={vr[1]:.1f}%)"; sb=f"Batch-corrected (PCo1={vb[0]:.1f}%, PCo2={vb[1]:.1f}%)"
two_panel(pc_raw,pc_bc,f"PCoA V2 -- {len(recs)} groups, 6 cohorts, anchor={len(anchor)} genes","pcoa_v2_ellipse.png",True,"PCo 1","PCo 2",[sr,sb])
two_panel(pc_raw,pc_bc,f"PCoA V2 (no ellipse) -- {len(recs)} groups, anchor={len(anchor)} genes","pcoa_v2_noellipse.png",False,"PCo 1","PCo 2",[sr,sb])

def run_umap(X): return umap.UMAP(n_neighbors=N_NEIGHBORS,min_dist=MIN_DIST,metric='correlation',random_state=SEED).fit_transform(X)
um_raw=run_umap(P); um_bc=run_umap(Pc)
print(f"[UMAP raw] ct={sep_pct(um_raw,broad):.1f}% ds={sep_pct(um_raw,dsl):.1f}%",flush=True)
print(f"[UMAP bc ] ct={sep_pct(um_bc,broad):.1f}% ds={sep_pct(um_bc,dsl):.1f}%",flush=True)
two_panel(um_raw,um_bc,f"UMAP V2 -- {len(recs)} groups, 6 cohorts, anchor={len(anchor)} genes","umap_v2_ellipse.png",True,"UMAP 1","UMAP 2",["Pre-correction","Batch-corrected"])
two_panel(um_raw,um_bc,f"UMAP V2 (no ellipse) -- {len(recs)} groups, anchor={len(anchor)} genes","umap_v2_noellipse.png",False,"UMAP 1","UMAP 2",["Pre-correction","Batch-corrected"])

N_PCA=min(50,len(recs)-1); nsp=max(2,min(5,min(Counter(broad).values())))
print(f"LDA: classes={dict(Counter(broad))} PCA={N_PCA} folds={nsp}",flush=True)
def lda_proj(X):
    Z=PCA(n_components=N_PCA,random_state=42).fit_transform(StandardScaler().fit_transform(X)); return LDA(n_components=2).fit_transform(Z,broad)
def cvr(X,tag):
    pipe=Pipeline([('sc',StandardScaler()),('pca',PCA(n_components=N_PCA,random_state=42)),('lda',LDA())])
    skf=StratifiedKFold(n_splits=nsp,shuffle=True,random_state=42)
    acc=cross_val_score(pipe,X,broad,cv=skf); rng=np.random.RandomState(0); yb=broad.copy(); rng.shuffle(yb)
    accs=cross_val_score(pipe,X,yb,cv=skf); pipe.fit(X,broad); tr=pipe.score(X,broad)
    print(f"[LDA {tag}] train={tr:.3f} cv={acc.mean():.3f}+/-{acc.std():.3f} shuffled={accs.mean():.3f} chance={1/len(set(broad)):.3f}",flush=True)
    return acc.mean(),accs.mean()
ld_raw=lda_proj(P); ar=cvr(P,"raw"); ld_bc=lda_proj(Pc); ab=cvr(Pc,"bc")
lr=f"RAW  5-fold CV acc={ar[0]:.2f} (shuffled {ar[1]:.2f})"; lb=f"Batch-corr  CV acc={ab[0]:.2f} (shuffled {ab[1]:.2f})"
two_panel(ld_raw,ld_bc,f"LDA (SUPERVISED) V2 -- {len(recs)} groups, anchor={len(anchor)}, PCA={N_PCA}","lda_v2_ellipse.png",True,"LD 1","LD 2",[lr,lb])
two_panel(ld_raw,ld_bc,f"LDA (SUPERVISED) V2 (no ellipse) -- anchor={len(anchor)}, PCA={N_PCA}","lda_v2_noellipse.png",False,"LD 1","LD 2",[lr,lb])

pd.DataFrame({'ds':dsl,'ct':[r['ct'] for r in recs],'broad':list(broad),
  'PCoA1_raw':pc_raw[:,0],'PCoA2_raw':pc_raw[:,1],'PCoA1_bc':pc_bc[:,0],'PCoA2_bc':pc_bc[:,1],
  'UMAP1_raw':um_raw[:,0],'UMAP2_raw':um_raw[:,1],'UMAP1_bc':um_bc[:,0],'UMAP2_bc':um_bc[:,1],
  'LD1_raw':ld_raw[:,0],'LD2_raw':ld_raw[:,1],'LD1_bc':ld_bc[:,0],'LD2_bc':ld_bc[:,1]}).to_csv(OUT/"all3_v2_coords.csv",index=False)
print("ALL_DONE",flush=True)
