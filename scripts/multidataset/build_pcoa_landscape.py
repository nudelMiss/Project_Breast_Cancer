#!/usr/bin/env python3
"""(A) pseudobulk-expression PCoA across cohorts; (B) marker-panel embedding PCoA.
Writes NEW files; does not touch existing pcoa_*."""
import re
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from scipy.io import mmread
from gensim.models import Word2Vec
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

REPO=Path("/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast")
TAG="propr_bidirectional_w10_k50_var75_hvg2000"
EXP={"bassez2021":"exports_bassez","wu2021":"exports_wu_counts","griffiths2021":"exports_griffiths",
     "qian2020":"exports_qian","pal2021":"exports_pal2021","azizi2018":"exports_azizi","gao2021":"exports_gao"}
ALL_DS=list(EXP); MULTI=["bassez2021","wu2021","qian2020","pal2021"]
COLOR={'Malignant':'#D62728','Epithelial':'#BCBD22','T_cell':'#2CA02C','Lymphoid':'#2CA02C','B_cell':'#1F77B4',
       'Plasmablast':'#1F77B4','NK_cell':'#17BECF','Myeloid':'#9467BD','Macrophage':'#9467BD','Monocyte':'#9467BD',
       'Fibroblast':'#8C564B','Pericyte':'#8C564B','Endothelial':'#FF7F0E'}
BROAD={'#D62728':'Malignant','#BCBD22':'Epithelial','#2CA02C':'T / Lymphoid','#1F77B4':'B / Plasma','#17BECF':'NK',
       '#9467BD':'Myeloid','#8C564B':'Fibroblast / Stroma','#FF7F0E':'Endothelial'}
MK={"bassez2021":"o","wu2021":"s","griffiths2021":"^","qian2020":"D","pal2021":"v","azizi2018":"P","gao2021":"X"}
OUT=REPO/"results/multidataset/pcoa"; OUT.mkdir(parents=True,exist_ok=True)
MARKERS=['EPCAM','KRT8','KRT18','KRT19','ELF3','CD3D','CD3E','CD2','IL7R','CD8A','TRAC','CD79A','CD79B','MS4A1','CD19',
         'IGHM','NKG7','GNLY','KLRD1','LYZ','CD68','CD14','AIF1','FCGR3A','C1QA','COL1A1','COL1A2','DCN','LUM','PDGFRB',
         'PECAM1','VWF','CLDN5','CDH5','PLVAP']

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
    C=V[:,:2]*np.sqrt(np.clip(w[:2],0,None)); vp=np.clip(w,0,None); return C, vp[:2]/vp.sum()*100
def vexp(coords,labels):
    gm=coords.mean(0); tot=((coords-gm)**2).sum(); b=0.0
    for u in set(labels):
        idx=[i for i,l in enumerate(labels) if l==u]; m=coords[idx].mean(0); b+=len(idx)*((m-gm)**2).sum()
    return b/tot*100
def plot(coords,ve,recs,fname,title):
    fig,ax=plt.subplots(figsize=(11,8))
    bc={r['color']:None for r in recs}
    for col in bc:
        pts=np.array([[coords[i,0],coords[i,1]] for i,r in enumerate(recs) if r['color']==col])
        ell(ax,pts,col)
    for i,r in enumerate(recs):
        ax.scatter(coords[i,0],coords[i,1],c=r['color'],marker=MK[r['ds']],s=58,alpha=0.82,edgecolors='white',linewidths=0.4,zorder=3)
    ax.set_xlabel(f"PCoA 1 ({ve[0]:.1f}%)"); ax.set_ylabel(f"PCoA 2 ({ve[1]:.1f}%)"); ax.set_title(title)
    seen={}
    for r in recs: seen[BROAD.get(r['color'],r['ct'])]=r['color']
    h1=[Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=11,label=l) for l,c in seen.items()]
    dss=[d for d in ALL_DS if any(r['ds']==d for r in recs)]
    h2=[Line2D([0],[0],marker=MK[d],color='w',markerfacecolor='grey',markeredgecolor='k',markersize=10,label=d) for d in dss]
    l1=ax.legend(handles=h1,title="Cell type",loc='upper left',bbox_to_anchor=(1.01,1.0),fontsize=10); ax.add_artist(l1)
    ax.legend(handles=h2,title="Cohort",loc='lower left',bbox_to_anchor=(1.01,0.0),fontsize=10)
    fig.savefig(OUT/fname,dpi=160,bbox_inches='tight'); plt.close(fig); print(f"[FIG] {fname}",flush=True)

# ---------- (A) PSEUDOBULK ----------
print("=== (A) pseudobulk ===",flush=True)
recsA=[]; pbd=[]
for ds in ALL_DS:
    cts=cts_for(ds)
    for gd in sorted((REPO/EXP[ds]).glob("patient=*")):
        ct=assign(gd.name.split("celltype=")[-1],cts)
        if ct not in COLOR: continue
        genes=[l.strip() for l in open(gd/"genes.csv")]
        M=mmread(str(gd/"expr.mtx")).tocsr()           # genes x cells
        tot=np.asarray(M.sum(1)).ravel().astype(float)
        n=min(len(genes),M.shape[0]); genes=genes[:n]; tot=tot[:n]
        cpm=np.log1p(tot/(tot.sum()+1e-9)*1e6)
        recsA.append(dict(ds=ds,ct=ct,color=COLOR[ct])); pbd.append(dict(zip(genes,cpm)))
shared=set(pbd[0])
for d in pbd[1:]: shared&=set(d)
shared=sorted(shared)
print(f"[A] {len(recsA)} pseudobulk groups, shared genes={len(shared)}",flush=True)
Xmat=np.array([[d[g] for g in shared] for d in pbd])
v=Xmat.var(0); top=np.argsort(v)[::-1][:2000]; Xh=Xmat[:,top]
C=np.corrcoef(Xh); D=1-C; np.fill_diagonal(D,0)
coords,ve=pcoa(D)
labs_ct=[r['color'] for r in recsA]; labs_ds=[r['ds'] for r in recsA]
print(f"[A][VAR] cell type={vexp(coords,labs_ct):.1f}%  dataset={vexp(coords,labs_ds):.1f}%",flush=True)
meta=pd.DataFrame([{'ds':r['ds'],'ct':r['ct']} for r in recsA]); meta['PCo1']=coords[:,0]; meta['PCo2']=coords[:,1]
meta.to_csv(OUT/"pcoa_pseudobulk_coords.csv",index=False)
plot(coords,ve,recsA,"pcoa_pseudobulk_expression.png","PCoA \u2014 pseudobulk expression per (sample \u00d7 cell type), 7 cohorts")

# ---------- (B) MARKER EMBEDDING ----------
print("=== (B) marker embedding ===",flush=True)
recsB=[]; KV={}
for ds in MULTI:
    cts=cts_for(ds)
    for md in sorted((REPO/f"results/{ds}/stageA/models").glob(f"*/{TAG}/gene_embeddings.model")):
        ct=assign(md.parent.parent.name,cts)
        if ct in COLOR:
            kv=Word2Vec.load(str(md)).wv; KV[len(recsB)]=kv
            recsB.append(dict(ds=ds,ct=ct,color=COLOR[ct],vocab=set(kv.index_to_key)))
panel=[m for m in MARKERS if all(m in r['vocab'] for r in recsB)]
print(f"[B] {len(recsB)} models; markers present in ALL: {len(panel)} -> {panel}",flush=True)
if len(panel)>=8:
    iu=np.triu_indices(len(panel),k=1)
    P=np.zeros((len(recsB),len(iu[0])))
    for i in range(len(recsB)):
        kv=KV[i]; Mk=np.array([kv[g] for g in panel]); Mk/= (np.linalg.norm(Mk,axis=1,keepdims=True)+1e-12)
        P[i]=(Mk@Mk.T)[iu]
    C=np.corrcoef(P); D=1-C; np.fill_diagonal(D,0); coords,ve=pcoa(D)
    print(f"[B][VAR] cell type={vexp(coords,[r['color'] for r in recsB]):.1f}%  dataset={vexp(coords,[r['ds'] for r in recsB]):.1f}%",flush=True)
    meta=pd.DataFrame([{'ds':r['ds'],'ct':r['ct']} for r in recsB]); meta['PCo1']=coords[:,0]; meta['PCo2']=coords[:,1]
    meta.to_csv(OUT/"pcoa_marker_coords.csv",index=False)
    plot(coords,ve,recsB,"pcoa_marker_embedding.png",f"PCoA \u2014 marker-panel embedding geometry ({len(panel)} markers, per-patient)")
else:
    print(f"[B] ABORT: only {len(panel)} markers shared across all models - too thin for a figure",flush=True)
print("DONE",flush=True)
