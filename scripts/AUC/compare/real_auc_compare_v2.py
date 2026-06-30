#!/usr/bin/env python
"""
Compare cosine vs -Z-Euclidean on a real joint W2V embedding,
under our and Yaniv's reporting conventions.

Reports for each metric:
  mean_auc_all   - average per-complex AUC over all complexes (our convention)
  mean_auc_q05   - average over complexes with q<0.05 (Yaniv's convention)
  frac_sig_q05   - fraction of complexes that pass q<0.05

Negative pool: full vocab (random pairs), n_neg = max(200, n_pos).
"""
import os, json, argparse, time
import numpy as np
import pandas as pd
from numpy.random import default_rng
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
from scipy.spatial.distance import cdist
from statsmodels.stats.multitest import multipletests


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_corum(path, vocab_set):
    df = pd.read_csv(path, sep="\t")
    complexes = {}
    for cid, sub in df.groupby("complex_id"):
        genes = [g for g in sub["gene"].astype(str).tolist() if g in vocab_set]
        if len(genes) >= 3:
            complexes[int(cid)] = genes
    return complexes


def within_pair_idx(idxs, rng, max_pairs=500):
    n = len(idxs)
    if n < 2:
        return np.array([]), np.array([])
    i, j = np.triu_indices(n, k=1)
    if len(i) > max_pairs:
        sel = rng.choice(len(i), size=max_pairs, replace=False)
        i, j = i[sel], j[sel]
    return np.asarray(idxs)[i], np.asarray(idxs)[j]


def neg_pair_idx(n_pairs, n_vocab, rng):
    i = rng.integers(0, n_vocab, size=n_pairs)
    j = rng.integers(0, n_vocab, size=n_pairs)
    same = i == j
    while same.any():
        j[same] = rng.integers(0, n_vocab, size=int(same.sum()))
        same = i == j
    return i, j


def score_pairs(S, i, j):
    return S[i, j]


def evaluate_metric(S, complexes, gene2idx, n_vocab, rng_seed, metric_name):
    log(f"  scoring under {metric_name}...")
    rng = default_rng(rng_seed)
    rows = []
    for cid, genes in complexes.items():
        idxs = [gene2idx[g] for g in genes]
        pi, pj = within_pair_idx(idxs, rng)
        if len(pi) < 1:
            continue
        pos = score_pairs(S, pi, pj)
        n_neg = max(200, len(pos))
        ni, nj = neg_pair_idx(n_neg, n_vocab, rng)
        neg = score_pairs(S, ni, nj)
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
        rows.append({"complex_id": cid, "n_pos": len(pos),
                     "auc": auc, "pvalue": p, "metric": metric_name})
    df = pd.DataFrame(rows)
    valid = df["pvalue"].notna()
    df["qvalue"] = np.nan
    if valid.sum():
        _, q, _, _ = multipletests(df.loc[valid, "pvalue"].values, method="fdr_bh")
        df.loc[valid, "qvalue"] = q
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corum", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    log(f"loading model: {args.model}")
    m = Word2Vec.load(args.model)
    V = m.wv.vectors.astype(np.float32)
    n_vocab, dim = V.shape
    log(f"  vocab={n_vocab}, dim={dim}")
    gene2idx = {g: i for i, g in enumerate(m.wv.index_to_key)}

    log(f"loading CORUM: {args.corum}")
    complexes = load_corum(args.corum, set(gene2idx))
    log(f"  complexes with >=3 in-vocab genes: {len(complexes)}")

    # --- cosine similarity matrix ---
    log("computing cosine similarity matrix...")
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    S_cos = (Vn @ Vn.T).astype(np.float32)
    log(f"  S_cos shape={S_cos.shape}, MB={S_cos.nbytes/1024**2:.0f}")
    del Vn

    df_cos = evaluate_metric(S_cos, complexes, gene2idx, n_vocab,
                             args.seed + 1, "cosine")
    df_cos.to_csv(os.path.join(args.out_dir, "per_complex_cosine.csv"), index=False)
    del S_cos
    log("  freed S_cos")

    # --- -Z-Euclidean similarity matrix ---
    log("computing pairwise Euclidean distances...")
    D = cdist(V, V, metric="euclidean").astype(np.float32)
    log(f"  D MB={D.nbytes/1024**2:.0f}, z-scoring rows...")
    mu = D.mean(axis=1, keepdims=True)
    sd = D.std(axis=1, keepdims=True) + 1e-12
    Z = (D - mu) / sd
    del D, mu, sd
    log("  symmetrizing and negating...")
    S_euc = -0.5 * (Z + Z.T)
    del Z
    log(f"  S_euc shape={S_euc.shape}, MB={S_euc.nbytes/1024**2:.0f}")

    df_euc = evaluate_metric(S_euc, complexes, gene2idx, n_vocab,
                             args.seed + 1, "neg_z_euclidean")
    df_euc.to_csv(os.path.join(args.out_dir, "per_complex_neg_z_euclidean.csv"), index=False)
    del S_euc

    # --- summary ---
    summary = {}
    for name, df in [("cosine", df_cos), ("neg_z_euclidean", df_euc)]:
        sub_q05 = df[df["qvalue"] < 0.05]
        summary[name] = {
            "n_complexes_total": int(df.shape[0]),
            "mean_auc_all": float(df["auc"].mean()),
            "mean_auc_q05": float(sub_q05["auc"].mean()) if len(sub_q05) else float("nan"),
            "frac_sig_q05": float((df["qvalue"] < 0.05).mean()),
            "n_sig_q05": int((df["qvalue"] < 0.05).sum()),
        }

    log("=== SUMMARY ===")
    for k, v in summary.items():
        log(f"{k}:")
        for kk, vv in v.items():
            log(f"  {kk}: {vv}")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"saved summary to {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()
