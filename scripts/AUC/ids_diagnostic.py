#!/usr/bin/env python3
"""
InterDependence Score (IDS) diagnostic
======================================
Same Spearman -> per-set AUC harness as the metacell/MAGIC/norm gates, but
replaces the similarity metric: instead of Spearman rho, the gene-gene score
is the IDS (Radhakrishnan et al. 2025) -- a universal dependence measure that
captures nonlinear/non-monotonic relationships Spearman misses.

Everything else is identical to the prior diagnostics, so the AUC is directly
comparable to the Spearman tables.

IDS input must be (n_samples x d_variables) = (cells x genes).
Output is a (genes x genes) matrix in [0,1] used in place of the rho matrix.

Reports per-set AUC + within/between IDS for each benchmark gene set.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread
from sklearn.metrics import roc_auc_score
from ids.numpy_dependence import compute_IDS_numpy as ids_np


def load_genes(p):
    s = pd.read_csv(p, header=None).iloc[:, 0]
    return s.dropna().astype(str).str.strip().tolist()


def align(mat, genes):
    if mat.shape[0] != len(genes) and mat.shape[1] == len(genes):
        mat = mat.T.tocsr()
    if mat.shape[0] != len(genes):
        n = min(mat.shape[0], len(genes)); mat, genes = mat[:n, :], genes[:n]
    return mat, genes


def remove_invalid(mat, genes):
    idx = [i for i, g in enumerate(genes) if g and g.upper() != "X"]
    return mat[idx], [genes[i] for i in idx]


def normalize_log1p(mat, scale=10000.0):
    m = mat.tocsr().astype(np.float32)
    sums = np.asarray(m.sum(axis=0)).ravel().astype(np.float32)
    sv = np.where(sums > 0, scale / sums, 1.0).astype(np.float32)
    m = m @ sparse.diags(sv, format="csr")
    m.data = np.log1p(m.data)
    return m.tocsr()


def filter_expr_mask(mat, min_frac=0.003):
    nz = np.asarray((mat > 0).sum(axis=1)).ravel() / mat.shape[1]
    return nz >= min_frac


def load_benchmark(path):
    df = pd.read_csv(path, sep="\t")
    sets = {}
    has = "complex_name" in df.columns
    for cid, grp in df.groupby("complex_id"):
        name = str(grp["complex_name"].iloc[0]) if has else str(cid)
        sets[str(cid)] = (name, set(grp["gene"].astype(str).str.strip().str.upper()))
    return sets


def ids_matrix(norm_genes_by_cells, gene_names, bench_genes, num_terms=6, p_norm="max"):
    """Compute IDS among benchmark-shared genes. Returns (shared, IDS matrix)."""
    gu = [g.upper() for g in gene_names]
    shared = sorted(set(gu) & bench_genes)
    pos = {g: i for i, g in enumerate(gu)}
    idx = [pos[g] for g in shared]

    sub = norm_genes_by_cells[idx, :]
    sub = sub.toarray().astype(np.float64) if sparse.issparse(sub) else np.asarray(sub, np.float64)
    # IDS wants cells x genes; min-max scale each gene to [0,8] (paper preprocessing)
    X = sub.T
    X = X - X.min(axis=0, keepdims=True)
    mx = X.max(axis=0, keepdims=True); mx[mx == 0] = 1.0
    X = X / mx * 8.0
    print(f"  IDS on {X.shape[1]} genes x {X.shape[0]} cells...", flush=True)
    C = ids_np(X, num_terms=num_terms, p_norm=p_norm)
    C = np.asarray(C, dtype=np.float32)
    np.fill_diagonal(C, 0.0)
    return shared, C


def evaluate(gene_names, score, sets, min_size=3, n_neg_min=200, seed=42):
    rng = np.random.default_rng(seed)
    gset = set(gene_names); g2i = {g: i for i, g in enumerate(gene_names)}
    out = []
    for cid, (cname, cg) in sets.items():
        present = sorted(cg & gset)
        if len(present) < min_size:
            continue
        pv = [score[g2i[a], g2i[b]] for i, a in enumerate(present) for b in present[i + 1:]]
        if not pv:
            continue
        non = sorted(gset - cg)
        if not non:
            continue
        nn = max(n_neg_min, len(pv))
        nv = [score[g2i[rng.choice(present)], g2i[rng.choice(non)]] for _ in range(nn)]
        lab = np.concatenate([np.ones(len(pv)), np.zeros(nn)])
        sc = np.concatenate([pv, nv])
        auc = 0.5 if np.std(sc) < 1e-10 else roc_auc_score(lab, sc)
        out.append({"complex_id": cid, "complex_name": cname,
                    "n_genes_present": len(present), "n_pos_pairs": len(pv),
                    "auc": auc, "mean_within": float(np.mean(pv)),
                    "mean_between": float(np.mean(nv))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group_dir", required=True)
    ap.add_argument("--bench_path", required=True)
    ap.add_argument("--bench_tag", required=True)
    ap.add_argument("--output_root", default="results/diagnostics/ids_spearman")
    ap.add_argument("--min_expr_frac", type=float, default=0.003)
    ap.add_argument("--num_terms", type=int, default=6)
    ap.add_argument("--p_norm", default="max")
    args = ap.parse_args()
    if str(args.p_norm) in {"1", "2"}:
        args.p_norm = int(args.p_norm)

    gd = Path(args.group_dir); gn = gd.name
    patient = gn.split("__")[0].replace("patient=", "")
    celltype = next((p.replace("celltype=", "").replace("-", "").replace(" ", "")
                     for p in gn.split("__") if p.startswith("celltype=")), "unknown")
    tag = f"{patient}_{celltype}"
    print(f"=== IDS diagnostic: {tag} | bench={args.bench_tag} | p_norm={args.p_norm} ===", flush=True)

    mat = mmread(str(gd / "expr.mtx")).tocsr()
    genes = load_genes(gd / "genes.csv")
    mat, genes = align(mat, genes); mat, genes = remove_invalid(mat, genes)
    n_cells = mat.shape[1]
    norm = normalize_log1p(mat)
    mask = filter_expr_mask(norm, args.min_expr_frac)
    genes = [g for g, k in zip(genes, mask) if k]
    norm = norm[mask]
    print(f"Universe after filter: {len(genes)} genes x {n_cells} cells", flush=True)

    sets = load_benchmark(Path(args.bench_path))
    allb = set()
    for _, (_, cg) in sets.items():
        allb.update(cg)

    shared, C = ids_matrix(norm, genes, allb, num_terms=args.num_terms, p_norm=args.p_norm)
    res = evaluate(shared, C, sets)

    out = Path(args.output_root) / f"{tag}__{args.bench_tag}"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(res)
    df.to_csv(out / "per_set.csv", index=False)
    summary = {
        "patient": patient, "celltype": celltype, "metric": "IDS",
        "p_norm": args.p_norm, "n_cells": n_cells, "n_shared_genes": len(shared),
        "n_sets": len(res),
        "mean_auc": float(df.auc.mean()), "median_auc": float(df.auc.median()),
        "mean_within": float(df.mean_within.mean()),
        "mean_between": float(df.mean_between.mean()),
        "separation": float(df.mean_within.mean() - df.mean_between.mean()),
    }
    pd.DataFrame([summary]).to_csv(out / "summary.csv", index=False)
    print(f"\nSaved to {out}", flush=True)
    print(f"  mean_auc={summary['mean_auc']:.4f}  within={summary['mean_within']:.4f}  "
          f"between={summary['mean_between']:.4f}  sep={summary['separation']:.4f}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
