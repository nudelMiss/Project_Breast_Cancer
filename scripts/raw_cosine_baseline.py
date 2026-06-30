"""Raw pairwise cosine CORUM AUC for all 6 pilot groups -> CSV."""
import sys, csv
sys.path.insert(0, '/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast/scripts')
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.io import mmread
from pathlib import Path
from sklearn.metrics import roc_auc_score
from train_model_new import (
    normalize_cells_log1p, remove_invalid_genes, filter_by_top_variance,
    _safe_load_genes, _align_expr_and_genes,
)

PROOT = Path("/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast")
GROUPS = [
    ("BIOKEY_10_Myeloid",      "patient=BIOKEY_10__celltype=Myeloid"),
    ("BIOKEY_13_Endothelial",  "patient=BIOKEY_13__celltype=Endothelial"),
    ("BIOKEY_18_T_cell",       "patient=BIOKEY_18__celltype=T_cell"),
    ("BIOKEY_30_Malignant",    "patient=BIOKEY_30__celltype=Malignant"),
    ("BIOKEY_3_Fibroblast",    "patient=BIOKEY_3__celltype=Fibroblast"),
    ("BIOKEY_4_B_cell",        "patient=BIOKEY_4__celltype=B_cell"),
]
CORUM = PROOT / "resources/corum_core_complexes.tsv"
OUT = PROOT / "results/bassez2021/supervisor_diagnostic/baseline/raw_cosine_baseline.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

def load_corum(p):
    df = pd.read_csv(p, sep="\t")
    return [(str(cid), str(cid), set(g["gene"].astype(str)))
            for cid, g in df.groupby("complex_id")]

def load_group(d):
    g = Path(d)
    mat = mmread(str(g / "expr.mtx")).tocsr()
    genes = _safe_load_genes(g / "genes.csv")
    mat, genes = _align_expr_and_genes(mat, genes)
    mat = normalize_cells_log1p(mat)
    mat, genes = remove_invalid_genes(mat, genes)
    mat, genes = filter_by_top_variance(mat, genes, 0.75)
    return mat, genes

def cos_mat(mat):
    X = mat.toarray().astype(np.float32) if sp.issparse(mat) else np.asarray(mat, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    Xn = X / n
    return Xn @ Xn.T

def bench(C, gene_list, complexes, min_sz=3, seed=42):
    rng = np.random.default_rng(seed)
    gidx = {g: i for i, g in enumerate(gene_list)}
    allg = set(gene_list)
    aucs, ws = [], []
    for cid, cname, cgenes in complexes:
        shared = cgenes & allg
        if len(shared) < min_sz: continue
        idx = np.array([gidx[g] for g in sorted(shared)])
        ia, ja = np.triu_indices(len(idx), k=1)
        pos = C[idx[ia], idx[ja]]
        if len(pos) == 0: continue
        non = np.array([gidx[g] for g in sorted(allg - shared)])
        n_neg = max(200, len(pos))
        a = rng.choice(idx, size=n_neg, replace=True)
        b = rng.choice(non, size=n_neg, replace=True)
        neg = C[a, b]
        lbl = np.concatenate([np.ones(len(pos)), np.zeros(n_neg)])
        sc = np.concatenate([pos, neg])
        if np.std(sc) < 1e-10: continue
        aucs.append(roc_auc_score(lbl, sc)); ws.append(len(pos))
    aucs = np.array(aucs); ws = np.array(ws)
    return dict(n_complexes=len(aucs), mean_auc=float(aucs.mean()),
                median_auc=float(np.median(aucs)),
                weighted_mean_auc=float((aucs*ws).sum()/ws.sum()))

print("Loading CORUM...", flush=True)
complexes = load_corum(CORUM)
print(f"  {len(complexes)} complexes", flush=True)

rows = []
for tag, gdir in GROUPS:
    print(f"\n=== {tag} ===", flush=True)
    mat, genes = load_group(PROOT / "exports_bassez" / gdir)
    print(f"  preprocessed: mat={mat.shape}", flush=True)
    C = cos_mat(mat)
    s = bench(C, genes, complexes)
    s["group_tag"] = tag
    rows.append(s)
    print("  ", s, flush=True)

df = pd.DataFrame(rows)
df = df[["group_tag", "n_complexes", "mean_auc", "median_auc", "weighted_mean_auc"]]
df.to_csv(OUT, index=False)
print(f"\nWrote {OUT}")
print(df.to_string(index=False))
