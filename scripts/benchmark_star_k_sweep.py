"""Evaluate STAR-walk W2V at k=10 and k=50 on the 4 biological gene sets."""
import sys
sys.path.insert(0, '/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast/scripts')
import numpy as np, pandas as pd
from pathlib import Path
from benchmark_gene_sets import GENE_SETS, auc_for_gene_set, cosine_matrix_from_embedding

PROOT = Path("/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast")
OUT   = PROOT / "results/bassez2021/supervisor_diagnostic/benchmark_star_k_sweep.csv"
GROUPS = ["BIOKEY_18_T_cell", "BIOKEY_30_Malignant", "BIOKEY_4_B_cell"]

rows = []
for grp in GROUPS:
    print(f"\n=== {grp} ===", flush=True)
    for k in [10, 50]:
        for w in [1, 5, 10, 50, 100]:
            mp = PROOT / f"results/bassez2021/supervisor_diagnostic/models/{grp}/star_k{k}_w{w}/gene_embeddings.model"
            if not mp.exists():
                print(f"  MISSING: {mp}", flush=True); continue
            C, emb_genes = cosine_matrix_from_embedding(mp)
            print(f"  star k={k} w={w}: {len(emb_genes)} genes", flush=True)
            for gs_name, gs in GENE_SETS.items():
                auc, n_in_set, n_pos, n_neg = auc_for_gene_set(C, emb_genes, gs)
                print(f"    {gs_name}: AUC={auc:.3f}", flush=True)
                rows.append(dict(group=grp, method=f"star_k{k}_w{w}",
                                 strategy="star", k_nearest=k, walks=w,
                                 gene_set=gs_name, auc=auc, n_in_set=n_in_set,
                                 n_pos=n_pos, n_neg=n_neg))
            del C

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"\nwrote {OUT}\n")
print("=== STAR k-sweep: mean AUC across 3 groups ===")
print(df.groupby(["k_nearest","walks","gene_set"])["auc"].mean().round(3).unstack())
