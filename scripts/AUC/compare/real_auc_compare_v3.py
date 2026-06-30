#!/usr/bin/env python
"""
Score a single W2V model under all 8 reporting conditions:
  2 metrics (cosine, -Z-Euclidean) x 2 neg pools (full, CORUM-only) x 2 filters (none, q<0.05)
Writes summary.json with all 8 numbers + per-complex CSVs.
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
    out = {}
    for cid, sub in df.groupby("complex_id"):
        g = [x for x in sub["gene"].astype(str).tolist() if x in vocab_set]
        if len(g) >= 3:
            out[int(cid)] = g
    return out


def within_idx(idxs, rng, max_pairs=500):
    n = len(idxs)
    if n < 2:
        return np.array([]), np.array([])
    i, j = np.triu_indices(n, k=1)
    if len(i) > max_pairs:
        sel = rng.choice(len(i), size=max_pairs, replace=False)
        i, j = i[sel], j[sel]
    arr = np.asarray(idxs)
    return arr[i], arr[j]


def neg_idx(n_pairs, pool, rng):
    pool = np.asarray(pool)
    if len(pool) < 2:
        return np.array([]), np.array([])
    i = rng.choice(pool, size=n_pairs, replace=True)
    j = rng.choice(pool, size=n_pairs, replace=True)
    same = i == j
    while same.any():
        j[same] = rng.choice(pool, size=int(same.sum()), replace=True)
        same = i == j
    return i, j


def evaluate(S, complexes, gene2idx, n_vocab, neg_pool_mode, rng_seed, metric, all_corum_idx):
    rng = default_rng(rng_seed)
    rows = []
    for cid, genes in complexes.items():
        idxs = [gene2idx[g] for g in genes]
        pi, pj = within_idx(idxs, rng)
        if len(pi) < 1:
            continue
        pos = S[pi, pj]
        n_neg = max(200, len(pos))
        if neg_pool_mode == "full":
            pool = np.arange(n_vocab)
        else:  # corum
            pool = np.setdiff1d(all_corum_idx, np.asarray(idxs), assume_unique=False)
            if len(pool) < 2:
                continue
        ni, nj = neg_idx(n_neg, pool, rng)
        neg = S[ni, nj]
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
        rows.append({"complex_id": cid, "n_pos": len(pos), "auc": auc,
                     "pvalue": p, "metric": metric, "neg_pool": neg_pool_mode})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
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
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    log(f"loading model: {args.model}")
    m = Word2Vec.load(args.model)
    V = m.wv.vectors.astype(np.float32)
    n_vocab, dim = V.shape
    log(f"  vocab={n_vocab}, dim={dim}")
    gene2idx = {g: i for i, g in enumerate(m.wv.index_to_key)}

    complexes = load_corum(args.corum, set(gene2idx))
    log(f"  complexes with >=3 in-vocab genes: {len(complexes)}")
    if len(complexes) == 0:
        log("  NO COMPLEXES — abort")
        return
    all_corum_idx = np.array(sorted({gene2idx[g] for genes in complexes.values() for g in genes}))

    # cosine sim
    log("S_cos ...")
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    S_cos = (Vn @ Vn.T).astype(np.float32)
    del Vn

    df_cos_full  = evaluate(S_cos, complexes, gene2idx, n_vocab, "full",  args.seed+1, "cosine", all_corum_idx)
    df_cos_corum = evaluate(S_cos, complexes, gene2idx, n_vocab, "corum", args.seed+2, "cosine", all_corum_idx)
    del S_cos

    # -Z-Euc sim
    log("S_euc ...")
    D = cdist(V, V, metric="euclidean").astype(np.float32)
    mu = D.mean(axis=1, keepdims=True); sd = D.std(axis=1, keepdims=True) + 1e-12
    Z = (D - mu) / sd
    del D, mu, sd
    S_euc = -0.5 * (Z + Z.T); del Z

    df_euc_full  = evaluate(S_euc, complexes, gene2idx, n_vocab, "full",  args.seed+1, "neg_z_euclidean", all_corum_idx)
    df_euc_corum = evaluate(S_euc, complexes, gene2idx, n_vocab, "corum", args.seed+2, "neg_z_euclidean", all_corum_idx)
    del S_euc

    all_df = pd.concat([df_cos_full, df_cos_corum, df_euc_full, df_euc_corum], ignore_index=True)
    all_df.to_csv(os.path.join(args.out_dir, "per_complex_all.csv"), index=False)

    summary = {"label": args.label, "n_vocab": int(n_vocab),
               "n_complexes": int(len(complexes)), "conditions": {}}
    for metric, neg_pool, df in [
        ("cosine",          "full",  df_cos_full),
        ("cosine",          "corum", df_cos_corum),
        ("neg_z_euclidean", "full",  df_euc_full),
        ("neg_z_euclidean", "corum", df_euc_corum),
    ]:
        if df.empty:
            continue
        sub_q05 = df[df["qvalue"] < 0.05]
        key = f"{metric}__{neg_pool}"
        summary["conditions"][key] = {
            "n_complexes_total": int(df.shape[0]),
            "mean_auc_all": float(df["auc"].mean()),
            "mean_auc_q05": float(sub_q05["auc"].mean()) if len(sub_q05) else float("nan"),
            "frac_sig_q05": float((df["qvalue"] < 0.05).mean()),
            "n_sig_q05": int((df["qvalue"] < 0.05).sum()),
        }

    log("=== SUMMARY ===")
    for k, v in summary["conditions"].items():
        log(f"{k}: all={v['mean_auc_all']:.4f}  q05={v['mean_auc_q05']:.4f}  frac_sig={v['frac_sig_q05']:.3f}")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
