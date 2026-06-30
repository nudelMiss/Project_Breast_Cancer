"""
Benchmark expansion: raw cosine + W2V at walks={1,5,10,50,100}
evaluated against CORUM, S phase, G2M, IFN-alpha, IFN-gamma
on 3 representative groups.
"""
import sys, csv
sys.path.insert(0, '/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast/scripts')
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.io import mmread
from pathlib import Path
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec
from train_model_new import (
    normalize_cells_log1p, remove_invalid_genes, filter_by_top_variance,
    _safe_load_genes, _align_expr_and_genes,
)

PROOT = Path("/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast")
GROUPS = {
    "BIOKEY_18_T_cell":    "patient=BIOKEY_18__celltype=T_cell",
    "BIOKEY_30_Malignant": "patient=BIOKEY_30__celltype=Malignant",
    "BIOKEY_4_B_cell":     "patient=BIOKEY_4__celltype=B_cell",
}
WALKS    = [1, 5, 10, 50, 100]
OUT      = PROOT / "results/bassez2021/supervisor_diagnostic/benchmark_genesets.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---- Gene sets ---------------------------------------------------------------
# Seurat cc.genes — Tirosh et al. Science 2016 (DOI 10.1126/science.aad0501)
S_PHASE = {
    "MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6",
    "CDCA7","DTL","PRIM1","UHRF1","MLF1IP","HELLS","RFC2","RPA2","NASP",
    "RAD51AP1","GMNN","WDR76","SLBP","CCNE2","UBR7","POLD3","MSH2","ATAD2",
    "RAD51","RRM2","CDC45","CDC6","EXO1","TIPIN","DSCC1","BLM","CASP8AP2",
    "USP1","CLSPN","POLA1","CHAF1B","BRIP1","E2F8",
}
G2M = {
    "HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2",
    "NUF2","CKS1B","MKI67","TMPO","CENPF","TACC3","FAM64A","SMC4","CCNB2",
    "CKAP2L","CKAP2","AURKB","BUB1","KIF11","ANP32E","TUBB4B","GTSE1",
    "KIF20B","HJURP","CDCA3","HN1","CDC20","TTK","CDC25C","KIF2C","RANGAP1",
    "NCAPD2","DLGAP5","CDCA2","CDCA8","ECT2","KIF23","HMMR","AURKA","PSRC1",
    "ANLN","LBR","CKAP5","CENPE","CTCF","NEK2","G2E3","GAS2L3","CBX5","CENPA",
}

# IFN sets parsed from MSigDB Hallmark gmt
def parse_gmt(path, name_match):
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts and name_match.lower() in parts[0].lower():
                return set(parts[2:])
    return set()

HALL = PROOT / "resources/genesets/MSigDB_Hallmark_2020.gmt"
IFN_A = parse_gmt(HALL, "Interferon Alpha Response")
IFN_G = parse_gmt(HALL, "Interferon Gamma Response")

# CORUM — combine all complex genes into one big set (proteins that participate in some complex)
# But for fair comparison with single-pathway AUC, we'll evaluate CORUM in its native multi-complex form
# AND also lump it as a single set, for a like-for-like comparison.
corum_df = pd.read_csv(PROOT / "resources/corum_core_complexes.tsv", sep="\t")
CORUM_COMPLEXES = [(str(cid), set(g["gene"].astype(str))) for cid, g in corum_df.groupby("complex_id")]

GENESETS = {
    "S_phase": S_PHASE,
    "G2M":     G2M,
    "IFN_a":   IFN_A,
    "IFN_g":   IFN_G,
}

print(f"Gene set sizes: S={len(S_PHASE)}, G2M={len(G2M)}, "
      f"IFNa={len(IFN_A)}, IFNg={len(IFN_G)}, CORUM={len(CORUM_COMPLEXES)} complexes")

# ---- Eval functions ---------------------------------------------------------
def single_set_auc(C, gene_to_idx, gene_set, seed=42, n_neg_min=200):
    """AUC for a single gene set against all other genes (one big 'complex')."""
    allg = set(gene_to_idx.keys())
    shared = gene_set & allg
    n_shared = len(shared)
    if n_shared < 3:
        return float("nan"), n_shared
    idx = np.array([gene_to_idx[g] for g in sorted(shared)])
    i_arr, j_arr = np.triu_indices(len(idx), k=1)
    pos = np.asarray(C[idx[i_arr], idx[j_arr]]).ravel()
    n_pos = len(pos)
    non = np.array([gene_to_idx[g] for g in sorted(allg - shared)])
    n_neg = max(n_neg_min, n_pos)
    rng = np.random.default_rng(seed)
    a = rng.choice(idx, size=n_neg, replace=True)
    b = rng.choice(non, size=n_neg, replace=True)
    neg = np.asarray(C[a, b]).ravel()
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    scores = np.concatenate([pos, neg])
    if np.std(scores) < 1e-10:
        return 0.5, n_shared
    return float(roc_auc_score(labels, scores)), n_shared

def corum_aucs(C, gene_to_idx, complexes, min_sz=3, seed=42):
    """Same as the project's standard CORUM eval (mean over complexes, weighted by n_pos)."""
    rng = np.random.default_rng(seed)
    allg = set(gene_to_idx.keys())
    aucs, ws = [], []
    for cid, cgenes in complexes:
        shared = cgenes & allg
        if len(shared) < min_sz: continue
        idx = np.array([gene_to_idx[g] for g in sorted(shared)])
        i_arr, j_arr = np.triu_indices(len(idx), k=1)
        pos = np.asarray(C[idx[i_arr], idx[j_arr]]).ravel()
        n_pos = len(pos)
        if n_pos == 0: continue
        non = np.array([gene_to_idx[g] for g in sorted(allg - shared)])
        n_neg = max(200, n_pos)
        a = rng.choice(idx, size=n_neg, replace=True)
        b = rng.choice(non, size=n_neg, replace=True)
        neg = np.asarray(C[a, b]).ravel()
        labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
        scores = np.concatenate([pos, neg])
        if np.std(scores) < 1e-10: continue
        aucs.append(roc_auc_score(labels, scores))
        ws.append(n_pos)
    aucs = np.array(aucs); ws = np.array(ws)
    return float(aucs.mean()), float((aucs*ws).sum()/ws.sum())

# ---- Compute matrices --------------------------------------------------------
def load_group(d):
    mat = mmread(str(d / "expr.mtx")).tocsr()
    genes = _safe_load_genes(d / "genes.csv")
    mat, genes = _align_expr_and_genes(mat, genes)
    mat = normalize_cells_log1p(mat)
    mat, genes = remove_invalid_genes(mat, genes)
    mat, genes = filter_by_top_variance(mat, genes, 0.75)
    return mat, genes

def cos_matrix(X):
    X = X.toarray().astype(np.float32) if sp.issparse(X) else np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True); norms[norms == 0] = 1.0
    Xn = X / norms
    return Xn @ Xn.T

def w2v_cos_matrix(model_path):
    """Load W2V model, build embedding cosine matrix in vocab order."""
    m = Word2Vec.load(str(model_path))
    genes = list(m.wv.key_to_index.keys())
    V = np.array([m.wv[g] for g in genes], dtype=np.float32)
    norms = np.linalg.norm(V, axis=1, keepdims=True); norms[norms == 0] = 1.0
    Vn = V / norms
    return Vn @ Vn.T, genes

# ---- Main loop ---------------------------------------------------------------
rows = []
for grp_tag, grp_name in GROUPS.items():
    print(f"\n=== {grp_tag} ===", flush=True)
    mat, genes = load_group(PROOT / "exports_bassez" / grp_name)
    print(f"  preprocessed mat={mat.shape} genes={len(genes)}", flush=True)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    # 1) Raw cosine
    print("  computing raw cosine matrix...", flush=True)
    C_raw = cos_matrix(mat)
    print("  ...done. Evaluating gene sets:", flush=True)
    for name, gset in GENESETS.items():
        auc, n_sh = single_set_auc(C_raw, gene_to_idx, gset)
        print(f"    raw cosine | {name:10s}: AUC={auc:.4f}  (n_shared={n_sh})", flush=True)
        rows.append(dict(group=grp_tag, method="raw_cosine", walks=None, geneset=name,
                         auc=auc, n_shared=n_sh))
    mean_c, wt_c = corum_aucs(C_raw, gene_to_idx, CORUM_COMPLEXES)
    print(f"    raw cosine | CORUM_meanAUC = {mean_c:.4f}  weighted = {wt_c:.4f}", flush=True)
    rows.append(dict(group=grp_tag, method="raw_cosine", walks=None, geneset="CORUM_mean",
                     auc=mean_c, n_shared=-1))
    rows.append(dict(group=grp_tag, method="raw_cosine", walks=None, geneset="CORUM_wt",
                     auc=wt_c, n_shared=-1))
    del C_raw

    # 2) W2V at each walks count
    for w in WALKS:
        mp = (PROOT / "results/bassez2021/models" / grp_tag /
              f"raw_cosine_bidirectional_w{w}_k5_wl3_perpat" / "gene_embeddings.model")
        if not mp.exists():
            print(f"  W2V w={w} MISSING: {mp}")
            continue
        print(f"  W2V walks={w}: loading + cosine...", flush=True)
        C_w, w_genes = w2v_cos_matrix(mp)
        w_idx = {g: i for i, g in enumerate(w_genes)}
        for name, gset in GENESETS.items():
            auc, n_sh = single_set_auc(C_w, w_idx, gset)
            print(f"    W2V w={w:4d} | {name:10s}: AUC={auc:.4f}  (n_shared={n_sh})", flush=True)
            rows.append(dict(group=grp_tag, method="w2v", walks=w, geneset=name,
                             auc=auc, n_shared=n_sh))
        mean_c, wt_c = corum_aucs(C_w, w_idx, CORUM_COMPLEXES)
        print(f"    W2V w={w:4d} | CORUM_meanAUC = {mean_c:.4f}  weighted = {wt_c:.4f}", flush=True)
        rows.append(dict(group=grp_tag, method="w2v", walks=w, geneset="CORUM_mean",
                         auc=mean_c, n_shared=-1))
        rows.append(dict(group=grp_tag, method="w2v", walks=w, geneset="CORUM_wt",
                         auc=wt_c, n_shared=-1))
        del C_w

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"\nWrote {len(df)} rows to {OUT}")
print("\n=== Summary table (mean across 3 groups) ===")
piv = (df.groupby(["method","walks","geneset"])["auc"].mean()
         .unstack("geneset").round(3))
print(piv)
