#!/usr/bin/env python
"""
auc_compare_v3.py — extends v2 to isolate the metric difference (cosine vs -Z-Euclidean).

Adds two axes to v2:
  - anisotropy: 0 (isotropic, same as v2) and norm-2 shared direction (W2V-like)
  - metric: cosine vs -Z-Euclidean

Output: 2x3 grid (anisotropy x signal regime), each cell shows 2 bars
(cosine_q05 vs negZEuc_q05). Full CSV with all 4x2 condition combinations.
"""
import os, json, argparse
import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import cdist
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_synthetic(rng, n_genes=2000, dim=64, n_complexes=100,
                   size_min=3, size_max=30, signal=1.0, anisotropy=0.0):
    """anisotropy: norm of a single shared direction added to every embedding."""
    emb = rng.standard_normal((n_genes, dim)).astype(np.float32)

    # add shared direction (anisotropy)
    if anisotropy > 0:
        shared = rng.standard_normal(dim).astype(np.float32)
        shared = shared / (np.linalg.norm(shared) + 1e-12) * anisotropy
        emb += shared[None, :]

    # plant complexes
    avail = list(range(n_genes))
    rng.shuffle(avail)
    complexes = []
    idx = 0
    for _ in range(n_complexes):
        if idx >= len(avail):
            break
        size = int(rng.integers(size_min, size_max + 1))
        if idx + size > len(avail):
            break
        members = avail[idx:idx + size]
        idx += size
        complexes.append(members)
    for members in complexes:
        center = rng.standard_normal(dim).astype(np.float32)
        center /= np.linalg.norm(center) + 1e-12
        emb[members] += signal * center
    return emb, complexes


# --- metric setup: precompute full pairwise score matrix once per embedding+metric ---

def cosine_matrix(emb):
    n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return n @ n.T


def neg_z_euclidean_matrix(emb):
    """For each row i: z-score Euclidean distances, then negate. Symmetrize."""
    D = cdist(emb, emb, metric="euclidean").astype(np.float32)
    # z-score each row over off-diagonal entries
    # (diagonal is 0 by construction; including it barely shifts mean/std with n=2000)
    mu = D.mean(axis=1, keepdims=True)
    sd = D.std(axis=1, keepdims=True) + 1e-12
    Z = (D - mu) / sd
    # symmetrize and negate
    S = -(Z + Z.T) / 2.0
    return S


def score_pairs(S, i_idx, j_idx):
    return S[i_idx, j_idx]


def within_pair_indices(members, rng, max_pairs=500):
    m = np.array(members)
    if len(m) < 2:
        return np.array([]), np.array([])
    i, j = np.triu_indices(len(m), k=1)
    if len(i) > max_pairs:
        sel = rng.choice(len(i), size=max_pairs, replace=False)
        i, j = i[sel], j[sel]
    return m[i], m[j]


def negative_pair_indices(n_pairs, rng, allowed_pool):
    pool = np.asarray(allowed_pool)
    if len(pool) < 2:
        return np.array([]), np.array([])
    i = rng.choice(pool, size=n_pairs, replace=True)
    j = rng.choice(pool, size=n_pairs, replace=True)
    same = i == j
    while same.any():
        j[same] = rng.choice(pool, size=int(same.sum()), replace=True)
        same = i == j
    return i, j


def score_one_complex(S, members, all_corum_genes, n_genes, rng,
                      neg_pool_mode="full", n_neg_floor=200):
    pos_i, pos_j = within_pair_indices(members, rng)
    if len(pos_i) < 1:
        return np.nan, np.nan, 0
    pos = score_pairs(S, pos_i, pos_j)
    n_neg = max(n_neg_floor, len(pos))
    if neg_pool_mode == "full":
        pool = np.arange(n_genes)
    elif neg_pool_mode == "corum":
        pool = np.setdiff1d(all_corum_genes, np.array(members), assume_unique=False)
    else:
        raise ValueError(neg_pool_mode)
    neg_i, neg_j = negative_pair_indices(n_neg, rng, pool)
    neg = score_pairs(S, neg_i, neg_j)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    try:
        auc = roc_auc_score(y, s)
    except Exception:
        auc = np.nan
    try:
        _, p = mannwhitneyu(pos, neg, alternative="greater")
    except Exception:
        p = np.nan
    return auc, p, len(pos)


def evaluate(S, complexes, n_genes, neg_pool_mode, rng):
    all_corum_genes = np.unique(np.concatenate([np.array(c) for c in complexes]))
    rows = []
    for k, members in enumerate(complexes):
        auc, p, n_pos = score_one_complex(
            S, members, all_corum_genes, n_genes, rng, neg_pool_mode=neg_pool_mode
        )
        rows.append({"complex_id": k, "n_pos": n_pos, "auc": auc, "pvalue": p})
    df = pd.DataFrame(rows)
    valid = df["pvalue"].notna()
    df["qvalue"] = np.nan
    if valid.sum() > 0:
        _, q, _, _ = multipletests(df.loc[valid, "pvalue"].values, method="fdr_bh")
        df.loc[valid, "qvalue"] = q
    return df


def summarize(df, q_filter=False, q_thresh=0.05):
    sub = df[df["qvalue"] < q_thresh] if q_filter else df
    return {
        "n_complexes_used": int(sub.shape[0]),
        "n_complexes_total": int(df.shape[0]),
        "mean_auc": float(sub["auc"].mean()) if len(sub) else float("nan"),
        "frac_sig_q05": float((df["qvalue"] < 0.05).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_genes", type=int, default=2000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--n_complexes", type=int, default=100)
    ap.add_argument("--anisotropy_strong", type=float, default=2.0,
                    help="norm of shared direction in the anisotropic condition")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    regimes = [("weak", 0.3), ("medium", 0.7), ("strong", 1.5)]
    anisos = [("isotropic", 0.0), ("anisotropic", args.anisotropy_strong)]
    metrics = ["cosine", "neg_z_euclidean"]

    rows = []
    for aniso_name, aniso in anisos:
        for regime_name, signal in regimes:
            rng = default_rng(args.seed)
            emb, complexes = make_synthetic(
                rng, n_genes=args.n_genes, dim=args.dim,
                n_complexes=args.n_complexes, signal=signal, anisotropy=aniso,
            )
            for metric in metrics:
                if metric == "cosine":
                    S = cosine_matrix(emb)
                else:
                    S = neg_z_euclidean_matrix(emb)
                df_full = evaluate(S, complexes, args.n_genes, "full",
                                   default_rng(args.seed + 1))
                df_corum = evaluate(S, complexes, args.n_genes, "corum",
                                    default_rng(args.seed + 2))
                for cond_name, df, qf in [
                    ("OURS",     df_full,  False),
                    ("+NEGPOOL", df_corum, False),
                    ("+QFILTER", df_full,  True),
                    ("BOTH",     df_corum, True),
                ]:
                    s = summarize(df, q_filter=qf)
                    s.update({
                        "aniso": aniso_name,
                        "aniso_norm": aniso,
                        "regime": regime_name,
                        "signal": signal,
                        "metric": metric,
                        "condition": cond_name,
                    })
                    rows.append(s)

    res = pd.DataFrame(rows)[[
        "aniso", "aniso_norm", "regime", "signal", "metric", "condition",
        "mean_auc", "frac_sig_q05", "n_complexes_used", "n_complexes_total",
    ]]
    csv_path = os.path.join(args.out_dir, "results.csv")
    res.to_csv(csv_path, index=False)
    print("\n=== full results ===")
    print(res.to_string(index=False))
    print(f"\nsaved: {csv_path}")

    # --- Figure: 2x3 grid, anisotropy (rows) x regime (cols).
    # Each cell: 2 bars = cosine vs neg_z_euclidean, both with q<0.05 filter applied.
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharey=True)
    regime_order = ["weak", "medium", "strong"]
    aniso_order = ["isotropic", "anisotropic"]
    metric_order = ["cosine", "neg_z_euclidean"]
    metric_labels = ["cosine", "−Z-Euclidean"]
    metric_colors = ["#4C8DC9", "#E08A3C"]

    for i, aniso in enumerate(aniso_order):
        for j, regime in enumerate(regime_order):
            ax = axes[i, j]
            vals = []
            for m in metric_order:
                sub = res[
                    (res.aniso == aniso) & (res.regime == regime)
                    & (res.metric == m) & (res.condition == "+QFILTER")
                ]
                v = sub["mean_auc"].values[0] if len(sub) else float("nan")
                vals.append(v)
            x = np.arange(2)
            bars = ax.bar(x, vals, color=metric_colors, width=0.6)
            for xi, v in enumerate(vals):
                if not np.isnan(v):
                    ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
                else:
                    ax.text(xi, 0.505, "n.s.", ha="center", fontsize=8, color="gray",
                            style="italic")
            ax.set_xticks(x)
            ax.set_xticklabels(metric_labels, fontsize=9)
            ax.axhline(0.5, color="k", linewidth=0.5, linestyle="--")
            ax.set_ylim(0.45, 0.85)
            if j == 0:
                ax.set_ylabel(f"{aniso}\nmean AUC", fontsize=10)
            if i == 0:
                ax.set_title(regime, fontsize=11)
    fig.suptitle("Metric comparison (cosine vs −Z-Euclidean) under q<0.05 filter\n"
                 "Rows: embedding anisotropy. Cols: planted signal strength.",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(args.out_dir, f"metric_comparison.{ext}"), dpi=200)
    print(f"saved: {os.path.join(args.out_dir, 'metric_comparison.png')}")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(res.to_dict(orient="records"), f, indent=2)


if __name__ == "__main__":
    main()
