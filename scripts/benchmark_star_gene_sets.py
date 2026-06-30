"""Evaluate STAR-walk W2V models against the 4 biological gene sets."""
import sys
sys.path.insert(0, '/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast/scripts')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec

# Reuse gene sets from sibling script by importing
sys.path.insert(0, '/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast/scripts')
from benchmark_gene_sets import GENE_SETS, auc_for_gene_set, cosine_matrix_from_embedding

PROOT = Path("/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast")
OUT = PROOT / "results/bassez2021/supervisor_diagnostic/benchmark_star_gene_sets.csv"
GROUPS = ["BIOKEY_18_T_cell", "BIOKEY_30_Malignant", "BIOKEY_4_B_cell"]

def model_path(grp, w):
    # diagnostic models for w in {1,5,10,50}; pipeline model for w=100
    if w == 100:
        return PROOT / f"results/bassez2021/models/{grp}/raw_cosine_star_w100_k5_wl3_perpat/gene_embeddings.model"
    return PROOT / f"results/bassez2021/supervisor_diagnostic/models/{grp}/star_w{w}/gene_embeddings.model"

rows = []
for grp in GROUPS:
    print(f"\n=== {grp} ===", flush=True)
    for w in [1, 5, 10, 50, 100]:
        mp = model_path(grp, w)
        C_emb, emb_genes = cosine_matrix_from_embedding(mp)
        print(f"  star w={w}: {len(emb_genes)} genes", flush=True)
        for gs_name, gs in GENE_SETS.items():
            auc, n_in_set, n_pos, n_neg = auc_for_gene_set(C_emb, emb_genes, gs)
            print(f"    {gs_name}: AUC={auc:.3f}", flush=True)
            rows.append(dict(group=grp, method=f"star_w{w}", walks=w,
                             gene_set=gs_name, n_in_set=n_in_set,
                             n_pos=n_pos, n_neg=n_neg, auc=auc))
        del C_emb

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"\nwrote {OUT}\n")
print("=== STAR walks: mean AUC across 3 groups ===")
print(df.groupby(["walks","gene_set"])["auc"].mean().round(3).unstack())
