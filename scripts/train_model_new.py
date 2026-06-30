#!/usr/bin/env python3
"""
train_model_new.py

New graph-construction experiments for the breast cancer gene-embedding pipeline.
Fixed default setup for the current phase:
  - T-cells only
  - Spearman graph
  - top-k graph
  - star-walk sentences
  - Word2Vec embeddings

Supported graph methods:
  1. expr      : baseline expression-frequency filter
  2. var75     : drop 25% lowest-variance genes, keep top 75%
  3. alra      : ALRA-like low-rank imputation before Spearman
  4. rhosig    : keep only statistically significant top-k Spearman rho edges

Notes:
  - The ALRA implementation here is a Python-native ALRA-like low-rank imputation
    fallback, not a wrapper around the original R ALRA package.
  - sNET is exposed as a CLI option, but intentionally raises an error until a
    concrete Python/R implementation is connected.
"""

import argparse
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import igraph as ig
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy import sparse
from scipy.io import mmread
from scipy.stats import rankdata, t as student_t
from tqdm import tqdm


# ============================================================
# Defaults for the new phase
# ============================================================

TARGET_GROUP = "T-cells"
SIMILARITY_METRIC = "spearman"
EDGE_MODE = "topk"
K_NEAREST = 5
WALK_LENGTH = 4
WALK_STRATEGY = "star"
WALKS_PER_GENE = 50
VECTOR_DIM = 64
EPOCHS = 20
WINDOW = 5
MIN_COUNT = 1
SEED = 42

DEFAULT_MIN_EXPR_FRAC = 0.003  # 0.3%; change with --min_expr_frac if needed
DEFAULT_VARIANCE_KEEP_FRAC = 0.75


# ============================================================
# General helpers
# ============================================================

def _get_slurm_cpus(default: int = 4) -> int:
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"):
        value = os.environ.get(key)
        if not value:
            continue
        value = value.split("(")[0]
        try:
            return int(value)
        except ValueError:
            pass
    return default


def _safe_load_genes(genes_csv: Path) -> List[str]:
    s = pd.read_csv(genes_csv, header=None).iloc[:, 0]
    s = s.dropna().astype(str).str.strip()
    return s[s != ""].tolist()


def _align_expr_and_genes(mat, genes: List[str]):
    """Ensure rows correspond to genes."""
    if mat.shape[0] != len(genes) and mat.shape[1] == len(genes):
        print("[LOAD] Transposing expression matrix so rows are genes", flush=True)
        mat = mat.T.tocsr()

    if mat.shape[0] != len(genes):
        diff = len(genes) - mat.shape[0]
        print(
            f"[WARN] gene/matrix mismatch: mat_rows={mat.shape[0]} "
            f"n_genes={len(genes)} diff={diff}",
            flush=True,
        )
        if abs(diff) <= 5:
            new_n = min(mat.shape[0], len(genes))
            print(f"[WARN] Aligning by truncation to n={new_n}", flush=True)
            mat = mat[:new_n, :]
            genes = genes[:new_n]
        else:
            raise ValueError(
                f"After transpose: mat.shape={mat.shape}, n_genes={len(genes)}, diff={diff}"
            )
    return mat, genes


def clean_token(value: object) -> str:
    """Folder-safe token: no '=', no spaces, no dots, minimal punctuation."""
    s = str(value)
    s = s.replace("T-cells", "Tcells")
    s = s.replace("B-cell", "Bcell")
    s = s.replace("=", "")
    s = s.replace(".", "")
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    s = s.replace("-", "")
    return s


def make_group_tag(group_name: str) -> str:
    """
    Convert:
        patient=CID3586__celltype=T-cells
    To:
        CID3586_Tcells
    """
    patient = None
    celltype = None
    for part in group_name.split("__"):
        if part.startswith("patient="):
            patient = part.replace("patient=", "")
        elif part.startswith("celltype="):
            celltype = part.replace("celltype=", "")

    if patient and celltype:
        return f"{clean_token(patient)}_{clean_token(celltype)}"
    return clean_token(group_name)


def expr_frac_tag(x: float) -> str:
    # 0.003 -> expr003, 0.03 -> expr0030 would be confusing, so use percent-like token.
    if x < 0.01:
        return f"expr{x:.3f}".replace("0.", "").replace(".", "")
    return f"expr{x:g}".replace(".", "")


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction, implemented without statsmodels."""
    p_values = np.asarray(p_values, dtype=np.float64)
    n = p_values.size
    if n == 0:
        return p_values

    order = np.argsort(p_values)
    ranked = p_values[order]
    q_ranked = ranked * n / np.arange(1, n + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)

    q = np.empty_like(q_ranked)
    q[order] = q_ranked
    return q


# ============================================================
# Preprocessing and filtering
# ============================================================

def normalize_cells_log1p(mat, scale_factor: float = 10000.0):
    """Library-size normalize each cell/column and apply log1p to non-zero values."""
    print("[PREPROCESS] Library-size normalization + log1p", flush=True)
    mat = mat.tocsr().astype(np.float32)

    cell_sums = np.asarray(mat.sum(axis=0)).ravel().astype(np.float32)
    scale = np.ones_like(cell_sums, dtype=np.float32)
    nonzero_cells = cell_sums > 0
    scale[nonzero_cells] = scale_factor / cell_sums[nonzero_cells]

    mat = mat @ sparse.diags(scale, offsets=0, format="csr")
    mat.data = np.log1p(mat.data)

    print(
        f"[PREPROCESS] done: shape={mat.shape}, nnz={mat.nnz}, "
        f"cell_sum_mean={float(cell_sums.mean()):.6g}",
        flush=True,
    )
    return mat.tocsr()


def remove_invalid_genes(mat, genes: List[str]):
    valid_idx = [i for i, g in enumerate(genes) if g and g.lower() != "x"]
    mat = mat[valid_idx, :]
    genes = [genes[i] for i in valid_idx]
    print(f"[CLEAN] kept valid genes: shape={mat.shape}, n_genes={len(genes)}", flush=True)
    return mat, genes


def filter_by_expression_fraction(mat, genes: List[str], min_expr_frac: float):
    before = len(genes)
    nonzero_frac = np.asarray((mat > 0).sum(axis=1)).ravel() / mat.shape[1]
    keep_mask = nonzero_frac >= min_expr_frac
    mat = mat[keep_mask, :]
    genes = [g for g, keep in zip(genes, keep_mask) if keep]
    print(
        f"[FILTER] expression >= {min_expr_frac:.4%} cells: kept {len(genes)}/{before}",
        flush=True,
    )
    return mat, genes


def filter_by_top_variance(mat, genes: List[str], keep_frac: float):
    before = len(genes)
    if not (0.0 < keep_frac <= 1.0):
        raise ValueError(f"keep_frac must be in (0, 1], got {keep_frac}")

    row_mean = np.asarray(mat.mean(axis=1)).ravel()
    row_mean_sq = np.asarray(mat.power(2).mean(axis=1)).ravel()
    gene_var = row_mean_sq - np.square(row_mean)

    n_keep = max(1, int(np.ceil(before * keep_frac)))
    keep_idx = np.argpartition(gene_var, -n_keep)[-n_keep:]
    keep_idx = np.sort(keep_idx)

    mat = mat[keep_idx, :]
    genes = [genes[i] for i in keep_idx]
    print(
        f"[FILTER] variance keep top {keep_frac:.0%}: kept {len(genes)}/{before}",
        flush=True,
    )
    return mat, genes


def alra_like_lowrank_impute(mat, rank: Optional[int] = None, preserve_observed: bool = True):
    """
    ALRA-style low-rank imputation with two critical fixes over the v1 benchmark:

    1. DATA-DRIVEN RANK via Marchenko-Pastur (MP):
       For an (m x n) random matrix with iid entries of variance sigma^2, the
       largest singular value converges to sigma * sqrt(n) * (1 + sqrt(m/n)).
       We select rank = number of real singular values that exceed this noise edge.
       This replaces the old fixed rank=50 default.

    2. PER-GENE ADAPTIVE THRESHOLDING (the defining ALRA step):
       After low-rank reconstruction, for each gene the most-negative
       reconstructed value reflects the noise floor.  Everything below
       |min_value| per gene is zeroed, restoring biological sparsity.
       This replaces the old np.maximum(X_hat, 0) which produced a dense
       matrix of small positives and destroyed Spearman's ability to
       distinguish real co-expression from low-rank smoothing artefacts.
    """
    X = mat.astype(np.float32).toarray() if sparse.issparse(mat) else np.asarray(mat, dtype=np.float32)
    n_genes, n_cells = X.shape
    orig_sparsity_pct = (X == 0).sum() / X.size * 100

    # Center genes before SVD
    gene_means = X.mean(axis=1, keepdims=True)
    X_centered = X - gene_means

    # Economy SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # ---- Data-driven rank selection (Marchenko-Pastur) ----
    if rank is None:
        row_vars = np.var(X_centered, axis=1)
        sigma = np.sqrt(float(np.mean(row_vars)))
        gamma = n_genes / n_cells
        mp_edge = sigma * np.sqrt(float(n_cells)) * (1.0 + np.sqrt(gamma))
        rank = int(np.sum(S > mp_edge))
        rank = max(2, min(rank, min(n_genes, n_cells) - 1))
        print(f"[IMPUTE] MP noise edge = {mp_edge:.2f}, "
              f"auto rank = {rank} (of {len(S)} components)", flush=True)
    else:
        rank = int(rank)
        rank = max(1, min(rank, min(n_genes, n_cells) - 1))

    print(f"[IMPUTE] ALRA imputation: rank={rank}, "
          f"matrix=({n_genes}, {n_cells})", flush=True)

    # Low-rank reconstruction
    X_hat = (U[:, :rank] * S[:rank]) @ Vt[:rank, :]
    X_hat = X_hat + gene_means

    # ---- Per-gene adaptive thresholding ----
    # For each gene row, |most-negative value| is the noise floor.
    # Zero out everything below that threshold.
    row_mins = X_hat.min(axis=1, keepdims=True)            # (n_genes, 1)
    thresholds = np.abs(np.minimum(row_mins, 0.0))         # |min| if < 0, else 0
    below = X_hat < thresholds
    n_zeroed = int(below.sum())
    X_hat[below] = 0.0

    post_sparsity_pct = (X_hat == 0).sum() / X_hat.size * 100
    print(f"[IMPUTE] Adaptive threshold: zeroed {n_zeroed:,} entries, "
          f"sparsity {orig_sparsity_pct:.1f}% -> {post_sparsity_pct:.1f}%",
          flush=True)

    # Preserve original observed non-zero values
    if preserve_observed:
        X_hat[X > 0] = X[X > 0]

    print("[IMPUTE] done", flush=True)
    return np.asarray(X_hat, dtype=np.float32)


def snet_impute_placeholder(mat):
    raise NotImplementedError(
        "sNET imputation is not connected yet. Use --graph_method alra for the first "
        "imputation benchmark, or plug the sNET implementation into snet_impute_placeholder()."
    )


# ============================================================
# Spearman edge construction
# ============================================================

def rank_standardize_rows(expression_matrix):
    dense = (
        expression_matrix.astype(np.float32).toarray()
        if sparse.issparse(expression_matrix)
        else np.asarray(expression_matrix, dtype=np.float32)
    )
    n, m = dense.shape
    print(f"[Spearman] dense shape={dense.shape}", flush=True)
    print("[Spearman] ranking rows", flush=True)
    R = np.apply_along_axis(rankdata, 1, dense).astype(np.float32)

    print("[Spearman] standardizing ranks", flush=True)
    R -= R.mean(axis=1, keepdims=True)
    denom = R.std(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    R /= denom
    return R, m


def spearman_topk_edges(
    expression_matrix,
    gene_names: Sequence[str],
    k: int,
    chunk_rows: int,
    chunk_cols: int,
    pvalue_filter: bool = False,
    rho_alpha: float = 0.05,
    rho_pvalue_mode: str = "fdr",
    rho_min: float = 0.0,
):
    """
    Compute top-k Spearman rho edges per gene. Optionally keep only statistically
    significant rho edges using an approximate t-test p-value and optional BH-FDR.

    Returns a DataFrame with at least: src, dst, weight.
    For rhosig runs, also includes: p_value, q_value, significant_score.
    """
    R, n_cells = rank_standardize_rows(expression_matrix)
    n = R.shape[0]
    scale = float(n_cells)

    src_idx_all: List[int] = []
    dst_idx_all: List[int] = []
    rho_all: List[float] = []

    print(
        f"[Spearman] top-k edges: n={n}, k={k}, rows={chunk_rows}, cols={chunk_cols}",
        flush=True,
    )

    for rs in tqdm(range(0, n, chunk_rows), desc="spearman_rows"):
        re = min(rs + chunk_rows, n)
        A = R[rs:re]
        br = re - rs

        best_vals = np.full((br, k), -np.inf, dtype=np.float32)
        best_idx = np.full((br, k), -1, dtype=np.int32)

        for cs in range(0, n, chunk_cols):
            ce = min(cs + chunk_cols, n)
            B = R[cs:ce]
            corr = (A @ B.T) / scale

            # Mask self-correlation.
            for local_i, global_i in enumerate(range(rs, re)):
                if cs <= global_i < ce:
                    corr[local_i, global_i - cs] = -np.inf

            local_k = min(k, ce - cs)
            loc_vals = np.partition(corr, -local_k, axis=1)[:, -local_k:]
            loc_idx = np.argpartition(corr, -local_k, axis=1)[:, -local_k:] + cs

            merged_vals = np.concatenate([best_vals, loc_vals], axis=1)
            merged_idx = np.concatenate([best_idx, loc_idx], axis=1)
            take = np.argpartition(merged_vals, -k, axis=1)[:, -k:]
            rows = np.arange(br)[:, None]
            best_vals = merged_vals[rows, take]
            best_idx = merged_idx[rows, take]
            order = np.argsort(best_vals, axis=1)[:, ::-1]
            best_vals = best_vals[rows, order]
            best_idx = best_idx[rows, order]

        for local_i in range(br):
            src_global = rs + local_i
            for dst_global, rho in zip(best_idx[local_i], best_vals[local_i]):
                if dst_global >= 0 and np.isfinite(rho):
                    src_idx_all.append(src_global)
                    dst_idx_all.append(int(dst_global))
                    rho_all.append(float(rho))

    edges = pd.DataFrame(
        {
            "src": [gene_names[i] for i in src_idx_all],
            "dst": [gene_names[j] for j in dst_idx_all],
            "weight": rho_all,
        }
    )

    if not pvalue_filter:
        return edges

    print("[RHO-SIG] Computing approximate Spearman p-values for top-k candidate edges", flush=True)
    rho = np.asarray(edges["weight"], dtype=np.float64)
    rho_clipped = np.clip(rho, -0.999999, 0.999999)
    df = max(1, n_cells - 2)
    t_stat = rho_clipped * np.sqrt(df / np.maximum(1e-12, 1.0 - rho_clipped**2))
    p_values = 2.0 * student_t.sf(np.abs(t_stat), df=df)
    q_values = bh_fdr(p_values)

    edges["p_value"] = p_values
    edges["q_value"] = q_values

    if rho_pvalue_mode == "raw":
        keep = (edges["p_value"] <= rho_alpha) & (edges["weight"] >= rho_min)
        edges["significant_score"] = edges["p_value"]
    elif rho_pvalue_mode == "fdr":
        keep = (edges["q_value"] <= rho_alpha) & (edges["weight"] >= rho_min)
        edges["significant_score"] = edges["q_value"]
    else:
        raise ValueError(f"Unknown rho_pvalue_mode={rho_pvalue_mode}")

    before = len(edges)
    edges = edges.loc[keep].copy()
    print(
        f"[RHO-SIG] kept {len(edges)}/{before} top-k candidate edges "
        f"using {rho_pvalue_mode} <= {rho_alpha} and rho >= {rho_min}",
        flush=True,
    )
    return edges


# ============================================================
# Cosine similarity (Bassez2021 extension)
# ============================================================

def cosine_topk_edges(
    expression_matrix,
    gene_names,
    k,
    chunk_rows,
    chunk_cols,
):
    """
    Compute top-k cosine-similarity edges per gene over cells.

    `expression_matrix` is (genes x cells) (post `_align_expr_and_genes`).
    Returns a DataFrame with columns: src, dst, weight  (weight = cosine).
    """
    from sklearn.preprocessing import normalize

    if sparse.issparse(expression_matrix):
        X = expression_matrix.astype(np.float32)
        # L2-normalize each gene (row) so dot product == cosine.
        X = normalize(X, norm="l2", axis=1)
        # Densify for fast chunked matmul; genes x cells, both modest at this scale.
        X = np.asarray(X.todense(), dtype=np.float32)
    else:
        X = np.asarray(expression_matrix, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms

    n = X.shape[0]
    print(f"[Cosine] top-k edges: n={n}, k={k}, rows={chunk_rows}, cols={chunk_cols}", flush=True)

    src_idx_all, dst_idx_all, w_all = [], [], []

    for rs in tqdm(range(0, n, chunk_rows), desc="cosine_rows"):
        re = min(rs + chunk_rows, n)
        A = X[rs:re]
        br = re - rs

        best_vals = np.full((br, k), -np.inf, dtype=np.float32)
        best_idx = np.full((br, k), -1, dtype=np.int32)

        for cs in range(0, n, chunk_cols):
            ce = min(cs + chunk_cols, n)
            B = X[cs:ce]
            corr = A @ B.T  # already L2-normalized => cosine

            # Mask self-similarity.
            for local_i, global_i in enumerate(range(rs, re)):
                if cs <= global_i < ce:
                    corr[local_i, global_i - cs] = -np.inf

            local_k = min(k, ce - cs)
            loc_vals = np.partition(corr, -local_k, axis=1)[:, -local_k:]
            loc_idx = np.argpartition(corr, -local_k, axis=1)[:, -local_k:] + cs

            merged_vals = np.concatenate([best_vals, loc_vals], axis=1)
            merged_idx = np.concatenate([best_idx, loc_idx], axis=1)
            take = np.argpartition(merged_vals, -k, axis=1)[:, -k:]
            rows = np.arange(br)[:, None]
            best_vals = merged_vals[rows, take]
            best_idx = merged_idx[rows, take]
            order = np.argsort(best_vals, axis=1)[:, ::-1]
            best_vals = best_vals[rows, order]
            best_idx = best_idx[rows, order]

        for local_i in range(br):
            src_global = rs + local_i
            for dst_global, w in zip(best_idx[local_i], best_vals[local_i]):
                if dst_global >= 0 and np.isfinite(w):
                    src_idx_all.append(src_global)
                    dst_idx_all.append(int(dst_global))
                    w_all.append(float(w))

    edges = pd.DataFrame({
        "src": [gene_names[i] for i in src_idx_all],
        "dst": [gene_names[j] for j in dst_idx_all],
        "weight": w_all,
    })
    return edges


# ============================================================
# HVG-capped universe + dense-matrix association backends
# ============================================================

def load_benchmark_universe(tsv_path):
    if not tsv_path:
        return set()
    df = pd.read_csv(tsv_path, sep="\t")
    col = "gene" if "gene" in df.columns else df.columns[-1]
    return set(df[col].astype(str).str.strip().str.upper())


def restrict_to_hvg_union(mat, genes, cap, bench_genes):
    """Keep top-`cap` genes by variance over cells UNION present benchmark genes.
    mat is (genes x cells). Returns (mat_sub, genes_sub)."""
    if cap is None or cap <= 0 or mat.shape[0] <= cap:
        return mat, genes
    M = mat.tocsr() if sparse.issparse(mat) else np.asarray(mat)
    if sparse.issparse(M):
        mean = np.asarray(M.mean(axis=1)).ravel()
        sqmean = np.asarray(M.multiply(M).mean(axis=1)).ravel()
        var = sqmean - mean ** 2
    else:
        var = M.var(axis=1)
    order = np.argsort(var)[::-1]
    keep = set(order[:cap].tolist())
    n_hvg = len(keep)
    bench_upper = {g.upper() for g in bench_genes}
    for i, g in enumerate(genes):
        if g.upper() in bench_upper:
            keep.add(i)
    keep_idx = sorted(keep)
    sub = M[keep_idx, :]
    sub_genes = [genes[i] for i in keep_idx]
    print(f"[HVG] {len(genes)} -> {len(sub_genes)} genes "
          f"(top-{cap} variance + {len(sub_genes) - n_hvg} forced benchmark)", flush=True)
    return sub, sub_genes


def dense_topk_edges(S, gene_names, k):
    """Top-k edges per row of a dense (n x n) score matrix (higher = closer)."""
    n = S.shape[0]
    S = np.array(S, dtype=np.float32, copy=True)
    np.fill_diagonal(S, -np.inf)
    kk = min(k, n - 1)
    src_all, dst_all, w_all = [], [], []
    for i in range(n):
        row = S[i]
        idx = np.argpartition(row, -kk)[-kk:]
        idx = idx[np.argsort(row[idx])[::-1]]
        for j in idx:
            w = row[j]
            if np.isfinite(w):
                src_all.append(i); dst_all.append(int(j)); w_all.append(float(w))
    return pd.DataFrame({
        "src": [gene_names[i] for i in src_all],
        "dst": [gene_names[j] for j in dst_all],
        "weight": w_all,
    })


def ids_topk_edges(expression_matrix, gene_names, k, num_terms=6, p_norm="2"):
    """IDS (Radhakrishnan et al.) association -> top-k edges. mat is (genes x cells)."""
    from ids.numpy_dependence import compute_IDS_numpy as _ids
    M = expression_matrix
    X = M.toarray() if sparse.issparse(M) else np.asarray(M)
    X = X.T.astype(np.float64)  # cells x genes
    X = X - X.min(axis=0, keepdims=True)
    mx = X.max(axis=0, keepdims=True); mx[mx == 0] = 1.0
    X = X / mx * 8.0
    pn = int(p_norm) if str(p_norm) in {"1", "2"} else p_norm
    print(f"[IDS] {X.shape[1]}x{X.shape[1]} dependence (cells={X.shape[0]}, "
          f"num_terms={num_terms}, p_norm={pn})...", flush=True)
    C = np.asarray(_ids(X, num_terms=num_terms, p_norm=pn), dtype=np.float32)
    return dense_topk_edges(C, gene_names, k)


def propr_topk_edges(expression_matrix, gene_names, k):
    """Proportionality rho_p on CLR profiles -> top-k edges. mat is (genes x cells)."""
    M = expression_matrix
    X = M.toarray() if sparse.issparse(M) else np.asarray(M)
    X = X.astype(np.float64)
    Xp = X + 1.0
    L = np.log(Xp)
    L = L - L.mean(axis=0, keepdims=True)   # CLR: center each cell (column)
    v = L.var(axis=1)
    Lc = L - L.mean(axis=1, keepdims=True)
    cov = (Lc @ Lc.T) / (L.shape[1] - 1)
    denom = v[:, None] + v[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = 2.0 * cov / denom            # = 1 - var(li-lj)/(var li+var lj)
    rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return dense_topk_edges(rho, gene_names, k)


def npmi_topk_edges(expression_matrix, gene_names, k, normalize=True):
    """Normalized PMI (Bouma 2009) on binarized co-detection -> top-k edges.

    mat is (genes x cells). Scores gene on/off co-occurrence (magnitude-free),
    orthogonal to the magnitude-based metrics. Mirrors propr_topk_edges:
    sparse B @ B.T (co-detection counts) -> dense nPMI matrix -> dense_topk_edges.

    Definitions (N = n_cells, B = (M > 0)):
        P(a)   = det_a / N                       det_a = row-sum of B
        P(a,b) = (B @ B.T)_ab / N
        PMI    = log2( P(a,b) / (P(a) P(b)) )
        nPMI   = PMI / (-log2 P(a,b))   in [-1, 1]
    nPMI = -1 where P(a,b) = 0 (never co-detected); = +1 where P(a,b) = 1
    (always co-detected; guards the 0/0). normalize=False returns raw PMI.
    Binarization is invariant to log1p (zero pattern identical), so this fits
    the existing log1p flow unchanged.
    """
    M = expression_matrix
    B = (M > 0)
    if sparse.issparse(B):
        B = B.astype(np.float64).tocsr()
    else:
        B = sparse.csr_matrix(np.asarray(B, dtype=np.float64))
    N = float(B.shape[1])  # n_cells
    det = np.asarray(B.sum(axis=1)).ravel()                    # per-gene detection counts
    Pa = det / N                                               # P(a)
    cooc = np.asarray((B @ B.T).todense(), dtype=np.float64)   # genes x genes co-detection
    Pab = cooc / N                                             # P(a,b)
    print(f"[nPMI] {B.shape[0]}x{B.shape[0]} co-detection "
          f"(cells={int(N)}, normalize={normalize})...", flush=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log2(Pab / (Pa[:, None] * Pa[None, :]))
        if normalize:
            score = pmi / (-np.log2(Pab))
            score[Pab == 0.0] = -1.0    # never co-detected
            score[Pab >= 1.0] = 1.0     # always co-detected (guard -log2(1)=0)
            score = np.nan_to_num(score, nan=-1.0, posinf=1.0, neginf=-1.0)
        else:
            score = pmi
            score[Pab == 0.0] = -np.inf  # dense_topk_edges skips non-finite weights
            score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=-np.inf)
    score = score.astype(np.float32)
    return dense_topk_edges(score, gene_names, k)


# ============================================================
# Graph and walks (star + bidirectional)
# ============================================================

def build_graph_from_edges(genes: Sequence[str], edges_df: pd.DataFrame, combine_edges: str) -> ig.Graph:
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    edge_tuples = []
    edge_weights = []

    for row in edges_df.itertuples(index=False):
        src = row.src
        dst = row.dst
        weight = float(row.weight)
        if src in gene_to_idx and dst in gene_to_idx:
            edge_tuples.append((gene_to_idx[src], gene_to_idx[dst]))
            edge_weights.append(weight)

    g = ig.Graph(n=len(genes), edges=edge_tuples, directed=False)
    g.vs["name"] = list(genes)
    g.es["weight"] = edge_weights

    combine = {"mean": "mean", "max": "max", "sum": "sum"}[combine_edges]
    g.simplify(combine_edges={"weight": combine})
    print(f"[GRAPH] |V|={g.vcount():,} |E|={g.ecount():,}", flush=True)
    return g


def build_weight_cache(graph: ig.Graph):
    nb_cache = []
    w_cache = []
    for v in range(graph.vcount()):
        nbs = graph.neighbors(v)
        if not nbs:
            nb_cache.append([])
            w_cache.append([])
            continue
        weights = []
        for nb in nbs:
            eid = graph.get_eid(v, nb, directed=False, error=False)
            w = float(graph.es[eid]["weight"]) if eid != -1 else 0.0
            weights.append(max(1e-4, w))  # floor, not clamp: keep all top-k neighbors reachable
        nb_cache.append(nbs)
        w_cache.append(weights)
    return nb_cache, w_cache


def star_walk(graph: ig.Graph, central_idx: int, walk_length: int, nb_cache, w_cache):
    central_gene = graph.vs[central_idx]["name"]
    nbs = nb_cache[central_idx]
    if not nbs:
        return [central_gene]

    weights = w_cache[central_idx]
    if sum(weights) <= 0.0:
        left = [random.choice(nbs) for _ in range(walk_length)]
        right = [random.choice(nbs) for _ in range(walk_length)]
    else:
        left = random.choices(nbs, weights=weights, k=walk_length)
        right = random.choices(nbs, weights=weights, k=walk_length)

    return [graph.vs[i]["name"] for i in left] + [central_gene] + [graph.vs[i]["name"] for i in right]


def bidirectional_walk(graph, central_idx, half_length, nb_cache, w_cache):
    """
    Single sentence of length 2*half_length+1: a half_length random walk to the
    left of the center, the center, and a half_length random walk to the right.
    Each step is sampled from the neighbors of the CURRENT node (not the center).
    """
    central_gene = graph.vs[central_idx]["name"]

    def _one_side(start_idx, length):
        path = []
        cur = start_idx
        for _ in range(length):
            nbs = nb_cache[cur]
            if not nbs:
                break
            weights = w_cache[cur]
            if sum(weights) <= 0.0:
                nxt = random.choice(nbs)
            else:
                nxt = random.choices(nbs, weights=weights, k=1)[0]
            path.append(nxt)
            cur = nxt
        return path

    left = _one_side(central_idx, half_length)
    right = _one_side(central_idx, half_length)
    # left appears center-first in the path; reverse so center sits in the middle
    return [graph.vs[i]["name"] for i in left[::-1]] + [central_gene] + [graph.vs[i]["name"] for i in right]


class BidirectionalWalkCorpus:
    """Same epoch-seed pattern as StarWalkCorpus."""
    def __init__(self, graph, half_length, walks_per_gene, seed, nb_cache, w_cache):
        self.graph = graph
        self.half_length = half_length
        self.walks_per_gene = walks_per_gene
        self.seed = seed
        self.nb_cache = nb_cache
        self.w_cache = w_cache
        self.total_examples = graph.vcount() * walks_per_gene
        self._iter_count = 0

    def __iter__(self):
        epoch_seed = self.seed + self._iter_count
        self._iter_count += 1
        random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        for v_idx in range(self.graph.vcount()):
            for _ in range(self.walks_per_gene):
                sentence = bidirectional_walk(
                    self.graph, v_idx, self.half_length, self.nb_cache, self.w_cache,
                )
                if len(sentence) > 1:
                    yield sentence


class StarWalkCorpus:
    def __init__(self, graph: ig.Graph, walk_length: int, walks_per_gene: int, seed: int, nb_cache, w_cache):
        self.graph = graph
        self.walk_length = walk_length
        self.walks_per_gene = walks_per_gene
        self.seed = seed
        self.nb_cache = nb_cache
        self.w_cache = w_cache
        self.total_examples = graph.vcount() * walks_per_gene
        self._iter_count = 0  # incremented each __iter__ call for epoch diversity

    def __iter__(self):
        # Use a different seed per epoch so Word2Vec sees fresh walks each time.
        # build_vocab gets epoch 0; training epochs get 1, 2, ..., 20.
        epoch_seed = self.seed + self._iter_count
        self._iter_count += 1
        random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        for v_idx in range(self.graph.vcount()):
            for _ in range(self.walks_per_gene):
                sentence = star_walk(self.graph, v_idx, self.walk_length, self.nb_cache, self.w_cache)
                if len(sentence) > 1:
                    yield sentence


def print_marker_neighbors(graph: ig.Graph, marker_genes: Sequence[str], top_k: int = 15):
    for gene in marker_genes:
        try:
            idx = graph.vs.find(name=gene).index
        except ValueError:
            print(f"[MARKER] {gene}: not found in graph", flush=True)
            continue

        rows = []
        for nb in graph.neighbors(idx):
            eid = graph.get_eid(idx, nb, directed=False)
            rows.append((graph.vs[nb]["name"], float(graph.es[eid]["weight"])))
        rows = sorted(rows, key=lambda x: -x[1])[:top_k]
        print(f"\n[MARKER] Neighbors for {gene}", flush=True)
        for name, weight in rows:
            print(f"  {name}\t{weight:.4f}", flush=True)


# ============================================================
# Configuration and naming
# ============================================================

@dataclass
class RunConfig:
    graph_method: str
    sim: str
    edge_mode: str
    k_nearest: int
    walks_per_gene: int
    walk_length: int
    walk_strategy: str
    vector_dim: int
    epochs: int
    window: int
    min_count: int
    min_expr_frac: float
    variance_keep_frac: float
    alra_rank: Optional[int]
    rho_alpha: float
    rho_pvalue_mode: str
    rho_min: float
    seed: int
    combine_edges: str
    materialize: bool
    spearman_chunk_rows: int
    spearman_chunk_cols: int
    graph_cache_dir: Optional[str] = None
    hvg_cap: int = 0
    benchmark_genes_tsv: Optional[str] = None
    precomputed_edges: Optional[str] = None
    ids_num_terms: int = 6
    ids_p_norm: str = "2"

    def method_tag(self) -> str:
        if self.graph_method == "expr":
            return expr_frac_tag(self.min_expr_frac)
        if self.graph_method == "var75":
            return "var75"
        if self.graph_method == "alra":
            return "alra"
        if self.graph_method == "snet":
            return "snet"
        if self.graph_method == "rhosig":
            alpha_tag = f"{self.rho_alpha:g}".replace("0.", "").replace(".", "")
            return f"rhosig_{self.rho_pvalue_mode}{alpha_tag}"
        return clean_token(self.graph_method)

    def tag(self) -> str:
        # New short folder name, no '='.
        base = (
            f"{clean_token(self.sim)}_"
            f"{clean_token(self.walk_strategy)}_"
            f"w{self.walks_per_gene}_"
            f"k{self.k_nearest}_"
            f"{self.method_tag()}"
        )
        if self.hvg_cap:
            base += f"_hvg{self.hvg_cap}"
        return base


# ============================================================
# One group run
# ============================================================

T_CELL_MARKERS = [
    "CD3D", "CD3E", "CD3G", "TRAC", "TRDC", "TRBC1", "TRBC2", "CD247",
    "LCK", "LAT", "IL7R", "TCF7", "LEF1", "NKG7", "GZMB", "PRF1", "CD8A", "CD8B",
]


def prepare_matrix_for_method(mat, genes: List[str], cfg: RunConfig):
    """Apply the method-specific graph preprocessing before Spearman."""
    if cfg.graph_method == "expr":
        mat, genes = filter_by_expression_fraction(mat, genes, cfg.min_expr_frac)

    elif cfg.graph_method == "var75":
        mat, genes = filter_by_top_variance(mat, genes, cfg.variance_keep_frac)

    elif cfg.graph_method == "alra":
        # Keep a minimal expression filter before imputation to avoid imputing genes
        # that are almost never observed.
        mat, genes = filter_by_expression_fraction(mat, genes, cfg.min_expr_frac)
        mat = alra_like_lowrank_impute(mat, rank=cfg.alra_rank)

    elif cfg.graph_method == "snet":
        mat, genes = filter_by_expression_fraction(mat, genes, cfg.min_expr_frac)
        mat = snet_impute_placeholder(mat)

    elif cfg.graph_method == "rhosig":
        # Use the expression filter as the base candidate gene set, then filter
        # the top-k Spearman edges by p-value/FDR.
        mat, genes = filter_by_expression_fraction(mat, genes, cfg.min_expr_frac)

    else:
        raise ValueError(f"Unknown graph_method={cfg.graph_method}")

    return mat, genes


def run_one_group(group_dir: Path, out_root: Path, cfg: RunConfig, save_edges: bool, skip_w2v: bool, debug_markers: bool):
    group_tag = make_group_tag(group_dir.name)
    out_dir = out_root / group_tag / cfg.tag()
    model_path = out_dir / "gene_embeddings.model"

    print(f"\n=== GROUP: {group_dir.name} -> {group_tag} ===", flush=True)
    print(f"[OUT] {out_dir}", flush=True)

    if model_path.exists():
        print(f"[SKIP] Existing model found: {model_path}", flush=True)
        return

    mat = mmread(str(group_dir / "expr.mtx")).tocsr()
    genes = _safe_load_genes(group_dir / "genes.csv")
    print(f"[LOAD] expr shape={mat.shape}, n_genes={len(genes)}", flush=True)
    mat, genes = _align_expr_and_genes(mat, genes)

    overall_nonzero_pct = (mat.nnz / (mat.shape[0] * mat.shape[1])) * 100
    print(f"[SPARSITY] overall non-zero % = {overall_nonzero_pct:.4f}", flush=True)

    mat, genes = remove_invalid_genes(mat, genes)
    mat = normalize_cells_log1p(mat)
    mat, genes = prepare_matrix_for_method(mat, genes, cfg)
    if cfg.hvg_cap and cfg.hvg_cap > 0:
        _bench = load_benchmark_universe(cfg.benchmark_genes_tsv)
        mat, genes = restrict_to_hvg_union(mat, genes, cfg.hvg_cap, _bench)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # ---- Graph construction with optional disk cache --------------------
    # Cache key: (group_tag, graph_method, sim). Walk strategy & walk count do
    # NOT change the graph and are NOT in the cache key.
    cached_edges_path = None
    if cfg.graph_cache_dir is not None:
        cache_root = Path(cfg.graph_cache_dir)
        # graph_method == 'rhosig' depends on rho params, so only cache non-rhosig.
        if cfg.graph_method != "rhosig":
            cached_edges_path = (
                cache_root / group_tag / f"{cfg.graph_method}_{cfg.sim}" / "edges.tsv"
            )

    if cfg.precomputed_edges:
        _pe = Path(cfg.precomputed_edges)
        if not _pe.exists():
            raise FileNotFoundError(f"--precomputed_edges not found: {_pe}")
        print(f"[PRECOMP] loading edges from {_pe}", flush=True)
        edges_df = pd.read_csv(_pe, sep="\t")
    elif cached_edges_path is not None and cached_edges_path.exists():
        print(f"[CACHE] loading edges from {cached_edges_path}", flush=True)
        edges_df = pd.read_csv(cached_edges_path, sep="\t")
    else:
        if cfg.sim == "spearman":
            pvalue_filter = cfg.graph_method == "rhosig"
            edges_df = spearman_topk_edges(
                mat,
                genes,
                k=cfg.k_nearest,
                chunk_rows=cfg.spearman_chunk_rows,
                chunk_cols=cfg.spearman_chunk_cols,
                pvalue_filter=pvalue_filter,
                rho_alpha=cfg.rho_alpha,
                rho_pvalue_mode=cfg.rho_pvalue_mode,
                rho_min=cfg.rho_min,
            )
        elif cfg.sim == "cosine":
            if cfg.graph_method == "rhosig":
                raise ValueError("cosine + rhosig is not supported (no p-values for cosine)")
            edges_df = cosine_topk_edges(
                mat,
                genes,
                k=cfg.k_nearest,
                chunk_rows=cfg.spearman_chunk_rows,
                chunk_cols=cfg.spearman_chunk_cols,
            )
        elif cfg.sim == "ids":
            edges_df = ids_topk_edges(
                mat, genes, k=cfg.k_nearest,
                num_terms=cfg.ids_num_terms, p_norm=cfg.ids_p_norm,
            )
        elif cfg.sim == "propr":
            edges_df = propr_topk_edges(mat, genes, k=cfg.k_nearest)
        elif cfg.sim == "npmi":
            edges_df = npmi_topk_edges(mat, genes, k=cfg.k_nearest, normalize=True)
        elif cfg.sim == "pmi":
            edges_df = npmi_topk_edges(mat, genes, k=cfg.k_nearest, normalize=False)
        elif cfg.sim == "cscore":
            raise ValueError("sim=cscore requires --precomputed_edges (compute in R).")
        else:
            raise ValueError(f"Unknown similarity metric: {cfg.sim}")

        if cached_edges_path is not None:
            cached_edges_path.parent.mkdir(parents=True, exist_ok=True)
            edges_df.to_csv(cached_edges_path, sep="\t", index=False)
            print(f"[CACHE] saved edges to {cached_edges_path}", flush=True)
    # --------------------------------------------------------------------

    print(f"[EDGES] final directed edges: {len(edges_df):,}", flush=True)
    if len(edges_df) == 0:
        raise RuntimeError("No edges remained after graph construction. Relax thresholds or check the data.")

    if save_edges:
        edges_df.to_csv(out_dir / "edges.tsv", sep="\t", index=False)
        print(f"[SAVE] edges.tsv", flush=True)

    graph = build_graph_from_edges(genes, edges_df, combine_edges=cfg.combine_edges)

    if debug_markers:
        print_marker_neighbors(graph, T_CELL_MARKERS)

    nb_cache, w_cache = build_weight_cache(graph)
    if cfg.walk_strategy == "star":
        corpus = StarWalkCorpus(
            graph, cfg.walk_length, cfg.walks_per_gene, cfg.seed, nb_cache, w_cache,
        )
    elif cfg.walk_strategy == "bidirectional":
        # walk_length here is interpreted as half-length per the Bassez2021 spec:
        # final sentence = (walk_length steps left) + center + (walk_length steps right).
        corpus = BidirectionalWalkCorpus(
            graph, cfg.walk_length, cfg.walks_per_gene, cfg.seed, nb_cache, w_cache,
        )
    else:
        raise ValueError(f"Unknown walk_strategy: {cfg.walk_strategy}")

    if skip_w2v:
        print("[STOP] --skip_w2v was used; stopping after graph construction.", flush=True)
        return

    if cfg.materialize:
        print("[WALKS] materializing corpus", flush=True)
        corpus_for_w2v = list(corpus)
        total_examples = len(corpus_for_w2v)
    else:
        corpus_for_w2v = corpus
        total_examples = corpus.total_examples

    print("[W2V] initializing", flush=True)
    model = Word2Vec(
        vector_size=cfg.vector_dim,
        window=cfg.window,
        min_count=cfg.min_count,
        sg=1,
        workers=_get_slurm_cpus(default=8),
        seed=cfg.seed,
    )

    print("[W2V] building vocab", flush=True)
    model.build_vocab(corpus_for_w2v)

    print("[W2V] training", flush=True)
    model.train(corpus_for_w2v, total_examples=total_examples, epochs=cfg.epochs)

    model.save(str(model_path))
    print(f"[SAVE] model: {model_path}", flush=True)


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    # IO
    p.add_argument("--in_root", type=str, default="exports_by_patient_celltype")
    p.add_argument("--out_root", "--output_dir", dest="out_root", type=str, default="results/models_spearman_star")
    p.add_argument("--only_group", type=str, default=None)
    p.add_argument("--celltype", type=str, default=TARGET_GROUP)
    p.add_argument("--save_edges", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip_w2v", action="store_true")
    p.add_argument("--debug_markers", action="store_true")

    # New method flag
    p.add_argument(
        "--graph_method",
        choices=["expr", "var75", "alra", "snet", "rhosig"],
        required=True,
        help="Graph construction method to benchmark.",
    )

    # Fixed core parameters, still configurable for controlled experiments
    p.add_argument("--sim", choices=["spearman", "cosine", "ids", "cscore", "propr", "npmi", "pmi"], default=SIMILARITY_METRIC)
    p.add_argument("--hvg_cap", type=int, default=0,
                   help="If >0, restrict to top-N variance genes UNION benchmark genes before association.")
    p.add_argument("--benchmark_genes_tsv", type=str, default=None,
                   help="TSV with a 'gene' column force-kept in the universe when --hvg_cap is set.")
    p.add_argument("--precomputed_edges", type=str, default=None,
                   help="Load edges.tsv from this path; skip association computation (used for cscore from R).")
    p.add_argument("--ids_num_terms", type=int, default=6)
    p.add_argument("--ids_p_norm", type=str, default="2")
    p.add_argument("--edge_mode", choices=["topk"], default=EDGE_MODE)
    p.add_argument("--k_nearest", type=int, default=K_NEAREST)
    p.add_argument("--walks", type=int, default=WALKS_PER_GENE)
    p.add_argument("--walk_length", type=int, default=WALK_LENGTH)
    p.add_argument("--walk_strategy", choices=["star", "bidirectional"], default=WALK_STRATEGY)

    # Filtering and imputation
    p.add_argument("--min_expr_frac", type=float, default=DEFAULT_MIN_EXPR_FRAC)
    p.add_argument("--variance_keep_frac", type=float, default=DEFAULT_VARIANCE_KEEP_FRAC)
    p.add_argument("--alra_rank", type=int, default=None)

    # Rho significance filtering
    p.add_argument("--rho_alpha", type=float, default=0.05)
    p.add_argument("--rho_pvalue_mode", choices=["raw", "fdr"], default="fdr")
    p.add_argument("--rho_min", type=float, default=0.0)

    # Word2Vec
    p.add_argument("--vector_dim", type=int, default=VECTOR_DIM)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--min_count", type=int, default=MIN_COUNT)
    p.add_argument("--seed", type=int, default=SEED)

    # Graph/compute settings
    p.add_argument("--combine_edges", choices=["mean", "max", "sum"], default="mean")
    p.add_argument("--materialize", action="store_true")
    p.add_argument("--spearman_chunk_rows", type=int, default=256)
    p.add_argument("--spearman_chunk_cols", type=int, default=2048)

    # Graph caching (Bassez2021 extension)
    p.add_argument("--graph_cache_dir", type=str, default=None,
                   help="If set, load/save edges.tsv under <dir>/<group_tag>/<imputation>_<sim>/edges.tsv")

    return p.parse_args()


def main():
    args = parse_args()

    if isinstance(args.celltype, str) and args.celltype.strip().lower() in {"none", "all", ""}:
        args.celltype = None

    cfg = RunConfig(
        graph_method=args.graph_method,
        sim=args.sim,
        edge_mode=args.edge_mode,
        k_nearest=args.k_nearest,
        walks_per_gene=args.walks,
        walk_length=args.walk_length,
        walk_strategy=args.walk_strategy,
        vector_dim=args.vector_dim,
        epochs=args.epochs,
        window=args.window,
        min_count=args.min_count,
        min_expr_frac=args.min_expr_frac,
        variance_keep_frac=args.variance_keep_frac,
        alra_rank=args.alra_rank,
        rho_alpha=args.rho_alpha,
        rho_pvalue_mode=args.rho_pvalue_mode,
        rho_min=args.rho_min,
        seed=args.seed,
        combine_edges=args.combine_edges,
        materialize=args.materialize,
        spearman_chunk_rows=args.spearman_chunk_rows,
        spearman_chunk_cols=args.spearman_chunk_cols,
        graph_cache_dir=args.graph_cache_dir,
        hvg_cap=args.hvg_cap,
        benchmark_genes_tsv=args.benchmark_genes_tsv,
        precomputed_edges=args.precomputed_edges,
        ids_num_terms=args.ids_num_terms,
        ids_p_norm=args.ids_p_norm,
    )

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=== RUN CONFIGURATION ===", flush=True)
    print(f"graph_method: {cfg.graph_method}", flush=True)
    print(f"similarity_metric: {cfg.sim}", flush=True)
    print(f"walk_strategy: {cfg.walk_strategy}", flush=True)
    print(f"walks_per_gene: {cfg.walks_per_gene}", flush=True)
    print(f"walk_length: {cfg.walk_length}", flush=True)
    print(f"k_nearest: {cfg.k_nearest}", flush=True)
    print(f"vector_dim: {cfg.vector_dim}", flush=True)
    print(f"config_tag: {cfg.tag()}", flush=True)
    print(f"output_root: {out_root}", flush=True)
    print("=========================", flush=True)

    group_dirs = sorted([p for p in in_root.iterdir() if p.is_dir()])
    if args.only_group:
        group_dirs = [p for p in group_dirs if p.name == args.only_group]
    elif args.celltype:
        suffix = f"__celltype={args.celltype}"
        group_dirs = [p for p in group_dirs if p.name.endswith(suffix)]

    print(f"Groups to run: {len(group_dirs)}", flush=True)
    if not group_dirs:
        raise SystemExit("No groups matched the requested filters.")

    for group_dir in group_dirs:
        run_one_group(
            group_dir=group_dir,
            out_root=out_root,
            cfg=cfg,
            save_edges=args.save_edges,
            skip_w2v=args.skip_w2v,
            debug_markers=args.debug_markers,
        )

    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
