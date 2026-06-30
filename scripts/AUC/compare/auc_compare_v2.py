#!/usr/bin/env python
"""
auc_compare_v2.py — synthetic test isolating two reporting/sampling differences
between our CORUM benchmark and Yaniv's R script.

Cosine similarity is held CONSTANT. We vary only:
  (A) negative pool: random pairs from full gene universe (OURS) vs
                     random pairs from other-CORUM genes only (YANIV)
  (B) per-complex aggregation: mean over all complexes (OURS) vs
                               mean over q<0.05 complexes after BH (YANIV)

Four conditions per signal regime: OURS, +NEGPOOL, +QFILTER, BOTH.
Three signal regimes (planted within-complex similarity strength): weak, medium, strong.
"""
import os, json, argparse
import numpy as np
import pandas as pd
from numpy.random import default_rng
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_synthetic(rng, n_genes=2000, dim=64, n_complexes=100,
                   size_min=3, size_max=30, signal=1.0):
    emb = rng.standard_normal((n_genes, dim)).astype(np.float32)
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


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return (a * b).sum(axis=-1)


def within_pairs(emb, members, rng, max_pairs=500):
    m = np.array(members)
    if len(m) < 2:
        return np.array([])
    i, j = np.triu_indices(len(m), k=1)
    if len(i) > max_pairs:
        sel = rng.choice(len(i), size=max_pairs, replace=False)
        i, j = i[sel], j[sel]
    return cosine_sim(emb[m[i]], emb[m[j]])


def negative_pairs(emb, n_pairs, rng, allowed_pool=None):
    pool = np.arange(emb.shape[0]) if allowed_pool is None else np.asarray(allowed_pool)
    if len(pool) < 2:
        return np.array([])
    i = rng.choice(pool, size=n_pairs, replace=True)
    j = rng.choice(pool, size=n_pairs, replace=True)
    same = i == j
    while same.any():
        j[same] = rng.choice(pool, size=int(same.sum()), replace=True)
        same = i == j
    return cosine_sim(emb[i], emb[j])


def score_one_complex(emb, members, all_corum_genes, rng,
                      neg_pool_mode="full", n_neg_floor=200):
    pos = within_pairs(emb, members, rng)
    if len(pos) < 1:
        return np.nan, np.nan, 0
    n_neg = max(n_neg_floor, len(pos))
    if neg_pool_mode == "full":
        neg = negative_pairs(emb, n_neg, rng, allowed_pool=None)
    elif neg_pool_mode == "corum":
        other = np.setdiff1d(all_corum_genes, np.array(members), assume_unique=False)
        neg = negative_pairs(emb, n_neg, rng, allowed_pool=other)
    else:
        raise ValueError(neg_pool_mode)
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


def evaluate_embedding(emb, complexes, neg_pool_mode, rng):
    all_corum_genes = np.unique(np.concatenate([np.array(c) for c in complexes]))
    rows = []
    for k, members in enumerate(complexes):
        auc, p, n_pos = score_one_complex(
            emb, members, all_corum_genes, rng, neg_pool_mode=neg_pool_mode
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
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    regimes = [("weak", 0.3), ("medium", 0.7), ("strong", 1.5)]
    rows = []

    for regime_name, signal in regimes:
        rng = default_rng(args.seed)
        emb, complexes = make_synthetic(
            rng, n_genes=args.n_genes, dim=args.dim,
            n_complexes=args.n_complexes, signal=signal
        )
        df_full = evaluate_embedding(emb, complexes, "full",
                                     default_rng(args.seed + 1))
        df_corum = evaluate_embedding(emb, complexes, "corum",
                                      default_rng(args.seed + 2))
        for cond_name, df, qf in [
            ("OURS (full pool, no q filter)",        df_full,  False),
            ("+NEGPOOL (corum pool, no q filter)",   df_corum, False),
            ("+QFILTER (full pool, q<0.05)",         df_full,  True),
            ("BOTH (corum pool, q<0.05)",            df_corum, True),
        ]:
            s = summarize(df, q_filter=qf)
            s["regime"] = regime_name
            s["signal"] = signal
            s["condition"] = cond_name
            rows.append(s)
        df_full.to_csv(os.path.join(args.out_dir, f"per_complex_full_{regime_name}.csv"), index=False)
        df_corum.to_csv(os.path.join(args.out_dir, f"per_complex_corum_{regime_name}.csv"), index=False)

    res = pd.DataFrame(rows)[["regime", "signal", "condition",
                              "mean_auc", "frac_sig_q05",
                              "n_complexes_used", "n_complexes_total"]]
    csv_path = os.path.join(args.out_dir, "results.csv")
    res.to_csv(csv_path, index=False)
    print("\n=== results ===")
    print(res.to_string(index=False))
    print(f"\nsaved: {csv_path}")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    regime_order = ["weak", "medium", "strong"]
    cond_order = [
        "OURS (full pool, no q filter)",
        "+NEGPOOL (corum pool, no q filter)",
        "+QFILTER (full pool, q<0.05)",
        "BOTH (corum pool, q<0.05)",
    ]
    colors = ["#888888", "#4C8DC9", "#E08A3C", "#7BAE5C"]
    x = np.arange(len(regime_order))
    width = 0.2
    for i, cond in enumerate(cond_order):
        vals = [res[(res.regime == r) & (res.condition == cond)]["mean_auc"].values[0]
                for r in regime_order]
        ax.bar(x + (i - 1.5) * width, vals, width, label=cond, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(regime_order)
    ax.set_ylabel("mean AUC over complexes used")
    ax.set_xlabel("planted signal regime")
    ax.axhline(0.5, color="k", linewidth=0.5, linestyle="--")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("AUC pipeline comparison (cosine held constant)\n"
                 "Isolating negative-pool choice and q<0.05 reporting filter")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(args.out_dir, f"comparison.{ext}"), dpi=200)
    print(f"saved: {os.path.join(args.out_dir, 'comparison.png')}")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(res.to_dict(orient="records"), f, indent=2)


if __name__ == "__main__":
    main()
