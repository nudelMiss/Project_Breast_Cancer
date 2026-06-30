#!/usr/bin/env python3
"""PCoA of propr gene-embedding geometry across cohorts. Distance = 1 - corr of
per-model gene x gene cosine-similarity profiles on shared (anchor) genes."""
import re
from pathlib import Path
import numpy as np, pandas as pd
from gensim.models import Word2Vec
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path("/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast")
TAG  = "propr_bidirectional_w10_k50_var75_hvg2000"
JTAG = TAG + "_jointct"
DATASETS = ["bassez2021","wu2021","griffiths2021","qian2020","pal2021","azizi2018","gao2021"]
EXP = {"bassez2021":"exports_bassez","wu2021":"exports_wu_counts","griffiths2021":"exports_griffiths",
       "qian2020":"exports_qian","pal2021":"exports_pal2021","azizi2018":"exports_azizi","gao2021":"exports_gao"}
COLOR = {'Malignant':'#D62728','Epithelial':'#BCBD22','T_cell':'#2CA02C','Lymphoid':'#2CA02C',
         'B_cell':'#1F77B4','Plasmablast':'#1F77B4','NK_cell':'#17BECF','Myeloid':'#9467BD',
         'Macrophage':'#9467BD','Monocyte':'#9467BD','DC':'#9467BD','Fibroblast':'#8C564B',
         'Pericyte':'#8C564B','Endothelial':'#FF7F0E','HSC':'#E377C2'}
BROAD = {'#D62728':'Malignant','#BCBD22':'Epithelial (normal)','#2CA02C':'T / Lymphoid','#1F77B4':'B / Plasma',
         '#17BECF':'NK','#9467BD':'Myeloid','#8C564B':'Fibroblast / Stroma','#FF7F0E':'Endothelial','#E377C2':'HSC'}
DS_MARKER = {"bassez2021":"o","wu2021":"s","griffiths2021":"^","qian2020":"D","pal2021":"v","azizi2018":"P","gao2021":"X"}

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
        if ct: records.append(dict(name=f"{ds}:{md.parent.parent.name}",ds=ds,ct=ct,agg="perpat",path=str(md)))
    for md in sorted((REPO/f"results/{ds}/models_joint_by_celltype").glob(f"celltype=*/{JTAG}/gene_embeddings.model")):
        ct=md.parent.parent.name.replace("celltype=","")
        records.append(dict(name=f"{ds}:JOINT:{ct}",ds=ds,ct=ct,agg="jointct",path=str(md)))
npp=sum(r['agg']=='perpat' for r in records); nj=sum(r['agg']=='jointct' for r in records)
print(f"[LOAD] {len(records)} models = {npp} per-patient + {nj} joint",flush=True)

vocabs=[]; KV={}
for r in records:
    kv=Word2Vec.load(r["path"]).wv; KV[r["name"]]=kv; vocabs.append(set(kv.index_to_key))
anchor=sorted(set.intersection(*vocabs))
print(f"[ANCHOR] full-vocab intersection = {len(anchor)} genes",flush=True)
if len(anchor)<200:
    bdf=pd.read_csv(REPO/"resources/genesets/bio_modules_benchmark.tsv",sep="\t")
    biog=set(pd.unique(bdf.values.ravel().astype(str)))
    anchor=sorted(set.intersection(*vocabs,biog)); print(f"[ANCHOR fallback bio] = {len(anchor)} genes",flush=True)

iu=np.triu_indices(len(anchor),k=1)
P=np.zeros((len(records),len(iu[0])),dtype=np.float32)
for i,r in enumerate(records):
    kv=KV[r["name"]]; M=np.array([kv[g] for g in anchor],dtype=np.float64)
    M/=(np.linalg.norm(M,axis=1,keepdims=True)+1e-12)
    P[i]=(M@M.T)[iu]
C=np.corrcoef(P); D=1.0-C; np.fill_diagonal(D,0.0)
n=len(records); J=np.eye(n)-np.ones((n,n))/n; B=-0.5*J@(D**2)@J
w,V=np.linalg.eigh(B); o=np.argsort(w)[::-1]; w=w[o]; V=V[:,o]
coords=V[:,:2]*np.sqrt(np.clip(w[:2],0,None))
vp=np.clip(w,0,None); ve=vp[:2]/vp.sum()*100
print(f"[PCoA] PCo1={ve[0]:.1f}%  PCo2={ve[1]:.1f}%",flush=True)

OUT=REPO/"results/multidataset/pcoa"; OUT.mkdir(parents=True,exist_ok=True)
meta=pd.DataFrame([{k:r[k] for k in ('name','ds','ct','agg')} for r in records])
meta["PCo1"]=coords[:,0]; meta["PCo2"]=coords[:,1]; meta["anchor_genes"]=len(anchor)
meta.to_csv(OUT/"pcoa_coords.csv",index=False)

def render(with_joint,fname):
    fig,ax=plt.subplots(figsize=(11,8.5))
    for i,r in enumerate(records):
        if r["agg"]=="jointct" and not with_joint: continue
        col=COLOR.get(r["ct"],"#999999"); mk=DS_MARKER.get(r["ds"],"o")
        if r["agg"]=="perpat":
            ax.scatter(coords[i,0],coords[i,1],c=col,marker=mk,s=42,alpha=0.68,edgecolors='none',zorder=2)
        else:
            ax.scatter(coords[i,0],coords[i,1],c=col,marker=mk,s=380,alpha=1,edgecolors='black',linewidths=2.2,zorder=5)
    ax.set_xlabel(f"PCo1 ({ve[0]:.1f}%)"); ax.set_ylabel(f"PCo2 ({ve[1]:.1f}%)")
    ax.set_title("propr gene-embedding geometry across 7 breast-cancer cohorts"+(" \u2014 with joint-by-celltype centroids" if with_joint else ""))
    seen={}
    for r in records: seen[BROAD.get(COLOR.get(r["ct"],"#999999"),r["ct"])]=COLOR.get(r["ct"],"#999999")
    h1=[Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=11,label=l) for l,c in seen.items()]
    h2=[Line2D([0],[0],marker=DS_MARKER[d],color='w',markerfacecolor='grey',markeredgecolor='k',markersize=10,label=d) for d in DATASETS]
    l1=ax.legend(handles=h1,title="Cell type",loc='upper left',bbox_to_anchor=(1.01,1.0),fontsize=9); ax.add_artist(l1)
    ax.legend(handles=h2,title="Dataset",loc='lower left',bbox_to_anchor=(1.01,0.0),fontsize=9)
    fig.savefig(OUT/fname,dpi=160,bbox_inches='tight'); plt.close(fig); print(f"[FIG] {OUT/fname}",flush=True)

render(False,"pcoa_perpatient_only.png")
render(True,"pcoa_with_joint_centroids.png")
print("DONE",flush=True)
