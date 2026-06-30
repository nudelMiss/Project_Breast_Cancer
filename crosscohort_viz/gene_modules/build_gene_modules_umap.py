#!/usr/bin/env python3
"""Gene-module map for ONE W2V embedding (within-model view): kNN -> SNN(Jaccard) -> Louvain
(Seurat RNA_snn-style), UMAP layout. Panel A = Louvain modules; Panel B = known programs
(G2M/S/IFN-a/IFN-g); Panel C = hypergeometric enrichment of modules vs programs. Read-only on models."""
import sys, ast
from pathlib import Path
import numpy as np, pandas as pd
from gensim.models import Word2Vec
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from scipy.stats import hypergeom
import igraph as ig
import umap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DS  = sys.argv[1] if len(sys.argv)>1 else "bassez2021"
CT  = sys.argv[2] if len(sys.argv)>2 else "Malignant"
K   = 20            # Seurat default kNN
PRUNE = 1/15        # Seurat default SNN prune
PROJ=Path("/groups/ofircohen-group/users/michalnu_yuvat/project_breast")
TAG="propr_bidirectional_w10_k50_var75_hvg2000_jointct"
OUT=Path("/groups/ofircohen-group/users/michalnu_yuvat/crosscohort_viz/gene_modules")
MODEL=PROJ/f"results/{DS}/models_joint_by_celltype/celltype={CT}/{TAG}/gene_embeddings.model"
print(f"model={MODEL}\nexists={MODEL.exists()}",flush=True)

# --- known programs (extract literals only; avoids the module's heavy imports) ---
src=open(PROJ/"scripts/benchmark_gene_sets.py").read(); tree=ast.parse(src); ns={}
want={"S_GENES","G2M_GENES","IFN_A","IFN_G","GENE_SETS"}
for node in tree.body:
    if isinstance(node,ast.Assign):
        names={t.id for t in node.targets if isinstance(t,ast.Name)}
        if names & want:
            exec(compile(ast.Module(body=[node],type_ignores=[]),"<gs>","exec"),ns)
GENE_SETS={k:set(v) for k,v in ns["GENE_SETS"].items()}   # S_phase,G2M,IFN_alpha,IFN_gamma
PROG_ORDER=["G2M","S_phase","IFN_alpha","IFN_gamma"]
PROG_COL={"G2M":"#D62728","S_phase":"#2CA02C","IFN_alpha":"#1F77B4","IFN_gamma":"#FF7F0E","other":"#CCCCCC"}
print("programs:",{k:len(v) for k,v in GENE_SETS.items()},flush=True)

# --- embedding ---
wv=Word2Vec.load(str(MODEL)).wv
genes=list(wv.index_to_key); N=len(genes)
V=np.array([wv[g] for g in genes],dtype=np.float32)
V/=(np.linalg.norm(V,axis=1,keepdims=True)+1e-12)
print(f"{N} genes, dim={V.shape[1]}",flush=True)

# --- kNN -> SNN (Jaccard) ---
nn=NearestNeighbors(n_neighbors=K+1,metric="cosine").fit(V)
_,idx=nn.kneighbors(V)
rows=np.repeat(np.arange(N),K+1); cols=idx.ravel()
A=sp.csr_matrix((np.ones(len(rows),dtype=np.float32),(rows,cols)),shape=(N,N))
A=((A+A.T)>0).astype(np.float32)            # symmetric kNN union
inter=(A@A.T).tocoo()                        # shared-neighbour counts
deg=np.asarray(A.sum(1)).ravel()
ei,ej,ev=[],[],[]
for i,j,c in zip(inter.row,inter.col,inter.data):
    if i<j:
        jac=c/(deg[i]+deg[j]-c)
        if jac>=PRUNE: ei.append(i); ej.append(j); ev.append(float(jac))
print(f"SNN edges after prune: {len(ev)}",flush=True)
g=ig.Graph(n=N,edges=list(zip(ei,ej))); g.es["weight"]=ev
comm=g.community_multilevel(weights="weight")      # Louvain (modularity, res=1)
memb=np.array(comm.membership)
# relabel communities by size (0 = largest), keep sizable ones
order={c:r for r,(c,_) in enumerate(sorted(pd.Series(memb).value_counts().items(),key=lambda x:-x[1]))}
memb=np.array([order[c] for c in memb]); ncl=memb.max()+1
print(f"Louvain modules: {ncl}  sizes={np.bincount(memb).tolist()}",flush=True)

# --- UMAP layout ---
emb=umap.UMAP(n_neighbors=15,min_dist=0.3,metric="cosine",random_state=42).fit_transform(V)
print("UMAP done",flush=True)

# --- program label per gene (first match in priority order) ---
gset=set(genes)
prog=np.array(["other"]*N,dtype=object)
for p in PROG_ORDER:
    inp=GENE_SETS[p]&gset
    for gname in inp:
        gi=genes.index(gname)
        if prog[gi]=="other": prog[gi]=p

# --- hypergeometric enrichment: module x program ---
rows_en=[]
for c in range(ncl):
    csize=int((memb==c).sum())
    for p in PROG_ORDER:
        Kp=len(GENE_SETS[p]&gset)
        k=int(((memb==c)&(prog==p)).sum())
        pval=hypergeom.sf(k-1,N,Kp,csize) if Kp>0 and csize>0 else 1.0
        rows_en.append(dict(module=c,program=p,module_size=csize,prog_in_vocab=Kp,overlap=k,pval=pval))
en=pd.DataFrame(rows_en)
# BH across all tests
pv=en["pval"].values; n=len(pv); o=np.argsort(pv); ro=np.empty(n,dtype=int); ro[o]=np.arange(n)
qraw=pv*n/(ro+1); qs=np.minimum.accumulate(qraw[o][::-1])[::-1]; qfull=np.empty(n); qfull[o]=qs
en["qval"]=np.clip(qfull,0,1); en["neglog10q"]=-np.log10(en["qval"]+1e-300)
en.to_csv(OUT/f"gene_modules_{DS}_{CT}_enrichment.csv",index=False)
top=(en.sort_values("qval").groupby("module").first().reset_index()[["module","program","overlap","qval"]])
print("top program per module:\n"+top.to_string(index=False),flush=True)

pd.DataFrame({"gene":genes,"module":memb,"program":prog,"umap1":emb[:,0],"umap2":emb[:,1]}).to_csv(
    OUT/f"gene_modules_{DS}_{CT}_assignments.csv",index=False)

# --- figure ---
fig,axes=plt.subplots(1,3,figsize=(21,6.5))
cmap=plt.get_cmap("tab10")
axA=axes[0]
for c in range(ncl):
    m=memb==c; axA.scatter(emb[m,0],emb[m,1],s=6,color=cmap(c%10),alpha=0.7,linewidths=0)
    cx,cy=emb[m,0].mean(),emb[m,1].mean(); axA.text(cx,cy,str(c),fontsize=12,fontweight="bold",ha="center",va="center")
axA.set_title(f"Louvain modules on SNN of W2V embedding\n{DS} {CT}  ({N} genes, {ncl} modules)")
axA.set_xlabel("UMAP 1"); axA.set_ylabel("UMAP 2")
axB=axes[1]
m=prog=="other"; axB.scatter(emb[m,0],emb[m,1],s=5,color=PROG_COL["other"],alpha=0.45,linewidths=0)
for p in PROG_ORDER:
    m=prog==p; axB.scatter(emb[m,0],emb[m,1],s=22,color=PROG_COL[p],alpha=0.95,edgecolors="k",linewidths=0.2,label=f"{p} (n={int(m.sum())})")
axB.set_title("Same UMAP, colored by known program"); axB.set_xlabel("UMAP 1"); axB.set_ylabel("UMAP 2")
axB.legend(fontsize=9,loc="best")
axC=axes[2]
H=en.pivot(index="module",columns="program",values="neglog10q").reindex(columns=PROG_ORDER)
im=axC.imshow(H.values,cmap="magma",aspect="auto")
axC.set_xticks(range(len(PROG_ORDER))); axC.set_xticklabels(PROG_ORDER,rotation=40,ha="right",fontsize=9)
axC.set_yticks(range(ncl)); axC.set_yticklabels([f"mod {c}" for c in range(ncl)],fontsize=9)
for (yy,xx),v in np.ndenumerate(H.values):
    if v>=1.3: axC.text(xx,yy,f"{v:.0f}",ha="center",va="center",color="white",fontsize=9)
axC.set_title("Module x program enrichment\n(-log10 q, hypergeometric; >=1.3 = q<0.05)")
fig.colorbar(im,ax=axC,fraction=0.046,pad=0.04)
fig.tight_layout()
fig.savefig(OUT/f"gene_modules_{DS}_{CT}_louvain_umap.png",dpi=160,bbox_inches="tight")
print(f"[FIG] gene_modules_{DS}_{CT}_louvain_umap.png",flush=True); print("DONE",flush=True)
