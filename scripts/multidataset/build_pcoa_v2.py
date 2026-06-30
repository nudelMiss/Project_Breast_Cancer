#!/usr/bin/env python3
"""PCoA v2: per-patient propr embeddings, strong cell types, multi-type cohorts only,
robust anchor, cell-type ellipses (reference-style)."""
import re
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd
from gensim.models import Word2Vec
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

REPO = Path("/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast")
TAG  = "propr_bidirectional_w10_k50_var75_hvg2000"
DATASETS = ["bassez2021","wu2021","qian2020","pal2021"]        # multi-type cohorts only
EXP = {"bassez2021":"exports_bassez","wu2021":"exports_wu_counts","qian2020":"exports_qian","pal2021":"exports_pal2021"}
# strong cell types -> (broad label, color). Anything not here is dropped (NK, Plasmablast, normal Epithelial).
KEEP = {'Malignant':('Malignant','#D62728'),
        'T_cell':('T / Lymphoid','#2CA02C'),'Lymphoid':('T / Lymphoid','#2CA02C'),
        'B_cell':('B cells','#1F77B4'),
        'Myeloid':('Myeloid','#9467BD'),'Macrophage':('Myeloid','#9467BD'),'Monocyte':('Myeloid','#9467BD'),
        'Fibroblast':('Fibroblast / Stroma','#8C564B'),'Pericyte':('Fibroblast / Stroma','#8C564B'),
        'Endothelial':('Endothelial','#FF7F0E')}
DS_MARKER = {"bassez2021":"o","wu2021":"s","qian2020":"^","pal2021":"D"}
DS_LABEL  = {"bassez2021":"Bassez2021","wu2021":"Wu2021","qian2020":"Qian2020","pal2021":"Pal2021"}

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

records=[]
for ds in DATASETS:
    cts=cts_for(ds)
    for md in sorted((REPO/f"results/{ds}/stageA/models").glob(f"*/{TAG}/gene_embeddings.model")):
        ct=assign(md.parent.parent.name,cts)
        if ct in KEEP:
            records.append(dict(name=f"{ds}:{md.parent.parent.name}",ds=ds,ct=ct,
                                broad=KEEP[ct][0],color=KEEP[ct][1],path=str(md)))
print(f"[LOAD] {len(records)} per-patient models (strong types, {len(DATASETS)} cohorts)",flush=True)

vsets=[]; KV={}
for r in records:
    kv=Word2Vec.load(r["path"]).wv; KV[r["name"]]=kv; vsets.append(set(kv.index_to_key))
nmod=len(records); present=Counter()
for v in vsets: present.update(v)
anchor=sorted([g for g,c in present.items() if c==nmod]); mode="intersection(100%)"
if len(anchor)<300:
    thr=int(np.ceil(0.95*nmod)); anchor=sorted([g for g,c in present.items() if c>=thr])
    keep=[i for i,v in enumerate(vsets) if set(anchor)<=v]
    records=[records[i] for i in keep]; vsets=[vsets[i] for i in keep]
    mode=f"95%-presence (kept {len(keep)}/{nmod} models)"
print(f"[ANCHOR] {len(anchor)} genes via {mode}",flush=True)

iu=np.triu_indices(len(anchor),k=1)
P=np.zeros((len(records),len(iu[0])),dtype=np.float32)
for i,r in enumerate(records):
    kv=KV[r["name"]]; M=np.array([kv[g] for g in anchor],dtype=np.float64)
    M/=(np.linalg.norm(M,axis=1,keepdims=True)+1e-12); P[i]=(M@M.T)[iu]
C=np.corrcoef(P); D=1.0-C; np.fill_diagonal(D,0.0)
n=len(records); J=np.eye(n)-np.ones((n,n))/n; B=-0.5*J@(D**2)@J
w,V=np.linalg.eigh(B); o=np.argsort(w)[::-1]; w=w[o]; V=V[:,o]
coords=V[:,:2]*np.sqrt(np.clip(w[:2],0,None)); vp=np.clip(w,0,None); ve=vp[:2]/vp.sum()*100

meta=pd.DataFrame([{k:r[k] for k in ('name','ds','ct','broad')} for r in records])
meta["PCo1"]=coords[:,0]; meta["PCo2"]=coords[:,1]
# variance explained by factor
def vexp(col):
    X=coords; gm=X.mean(0); tot=((X-gm)**2).sum(); bss=0.0
    for _,idx in meta.groupby(col).groups.items():
        m=X[list(idx)].mean(0); bss+=len(idx)*((m-gm)**2).sum()
    return bss/tot*100
print(f"[VAR] cell type={vexp('broad'):.1f}%  dataset={vexp('ds'):.1f}%  (PCo1={ve[0]:.1f}% PCo2={ve[1]:.1f}%)",flush=True)

OUT=REPO/"results/multidataset/pcoa"; OUT.mkdir(parents=True,exist_ok=True)
meta.to_csv(OUT/"pcoa_v2_coords.csv",index=False)

def ell(ax,pts,color,nstd=2.0):
    if len(pts)<4: return
    cov=np.cov(pts.T); mu=pts.mean(0); vals,vecs=np.linalg.eigh(cov); ordr=vals.argsort()[::-1]
    vals,vecs=vals[ordr],vecs[:,ordr]; ang=np.degrees(np.arctan2(vecs[1,0],vecs[0,0]))
    ww,hh=2*nstd*np.sqrt(np.clip(vals,0,None))
    ax.add_patch(Ellipse(mu,ww,hh,angle=ang,fill=False,edgecolor=color,ls='--',lw=2,alpha=0.85,zorder=1))

fig,ax=plt.subplots(figsize=(11,8))
broad_color={r['broad']:r['color'] for r in records}
for b,c in broad_color.items():
    sub=meta[meta.broad==b]; ell(ax,sub[["PCo1","PCo2"]].values,c)
for i,r in enumerate(records):
    ax.scatter(coords[i,0],coords[i,1],c=r["color"],marker=DS_MARKER[r["ds"]],s=55,alpha=0.8,edgecolors='white',linewidths=0.4,zorder=3)
ax.set_xlabel(f"PCoA 1 ({ve[0]:.1f}%)"); ax.set_ylabel(f"PCoA 2 ({ve[1]:.1f}%)")
ax.set_title("PCoA — propr per-patient gene embeddings (strong cell types, 4 cohorts)")
h_ct=[Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=11,label=b) for b,c in broad_color.items()]
h_ds=[Line2D([0],[0],marker=DS_MARKER[d],color='w',markerfacecolor='grey',markeredgecolor='k',markersize=10,label=DS_LABEL[d]) for d in DATASETS]
l1=ax.legend(handles=h_ct,title="Cell type",loc='upper left',bbox_to_anchor=(1.01,1.0),fontsize=10); ax.add_artist(l1)
ax.legend(handles=h_ds,title="Cohort",loc='lower left',bbox_to_anchor=(1.01,0.0),fontsize=10)
fig.savefig(OUT/"pcoa_v2_strong_perpatient.png",dpi=160,bbox_inches='tight'); plt.close(fig)
print(f"[FIG] {OUT/'pcoa_v2_strong_perpatient.png'}",flush=True); print("DONE",flush=True)
