import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
OUT="/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast/scripts/visualization/umap"
df=pd.read_csv(OUT+"/umap_emb_6cohort_coords.csv")
COL={'Malignant':'#D62728','T / Lymphoid':'#2CA02C','B cells':'#1F77B4','Myeloid':'#9467BD','Fibroblast / Stroma':'#8C564B','Endothelial':'#FF7F0E'}
MK={"bassez2021":"o","wu2021":"s","qian2020":"^","pal2021":"D","gao2021":"P","azizi2018":"X"}
DSLAB={"bassez2021":"Bassez2021","wu2021":"Wu2021","qian2020":"Qian2020","pal2021":"Pal2021","gao2021":"Gao2021","azizi2018":"Azizi2018"}
DATASETS=list(MK)
def ell(ax,pts,color,nstd=2.0):
    if len(pts)<4: return
    cov=np.cov(pts.T);mu=pts.mean(0);vals,vecs=np.linalg.eigh(cov);o=vals.argsort()[::-1];vals,vecs=vals[o],vecs[:,o]
    ang=np.degrees(np.arctan2(vecs[1,0],vecs[0,0]));ww,hh=2*nstd*np.sqrt(np.clip(vals,0,None))
    ax.add_patch(Ellipse(mu,ww,hh,angle=ang,fill=False,edgecolor=color,ls='--',lw=2,alpha=0.85,zorder=1))
def make(draw_ellipse,name):
    fig,ax=plt.subplots(figsize=(11,9))
    if draw_ellipse:
        for b,c in COL.items():
            pts=df[df.broad==b][['UMAP1_raw','UMAP2_raw']].values; ell(ax,pts,c)
    for _,r in df.iterrows():
        ax.scatter(r.UMAP1_raw,r.UMAP2_raw,c=COL[r.broad],marker=MK[r.ds],s=70,alpha=0.82,edgecolors='white',linewidths=0.4,zorder=3)
    ax.set_xlabel("UMAP 1");ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP of per-patient propr embeddings -- RAW (pre-correction)\n6 cohorts, 340 groups, anchor=245 genes")
    h1=[Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=11,label=b) for b,c in COL.items()]
    h2=[Line2D([0],[0],marker=MK[d],color='w',markerfacecolor='grey',markeredgecolor='k',markersize=10,label=DSLAB[d]) for d in DATASETS]
    l1=ax.legend(handles=h1,title="Cell type",loc='upper left',bbox_to_anchor=(1.02,1.0),fontsize=10);ax.add_artist(l1)
    ax.legend(handles=h2,title="Cohort",loc='upper left',bbox_to_anchor=(1.02,0.52),fontsize=10)
    fig.savefig(OUT+"/"+name,dpi=160,bbox_inches='tight');plt.close(fig);print("[FIG]",name)
make(True,"umap_emb_6cohort_raw_ellipse.png")
make(False,"umap_emb_6cohort_raw_noellipse.png")
print("DONE")
