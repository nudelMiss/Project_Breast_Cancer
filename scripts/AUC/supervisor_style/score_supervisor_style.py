#!/usr/bin/env python
"""
Re-score a trained gensim Word2Vec gene-embedding model using the supervisor's
R-pipeline AUC procedure, end-to-end, in Python.

Mirrors:
  1. Similarity = -Z-score of pairwise Euclidean distance
  2. Between-pairs = OTHER CORUM genes only (not random genome-wide)
  3. Per-complex AUC = ROC-AUC of within vs between sims
  4. Per-complex p-value = one-sided Mann-Whitney U (within > between)
  5. BH correction PER EMBEDDING
  6. Reports BOTH unfiltered (your convention) AND q<0.05 filtered (his) AUCs,
     plus the global pooled AUC (his "global_poc")
  7. min_complex_size = 3 (his default)
"""

import argparse, os, json
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests


def neg_zscore_similarity_matrix(E):
    sq = np.sum(E * E, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (E @ E.T)
    np.maximum(D2, 0.0, out=D2)
    D = np.sqrt(D2)
    iu = np.triu_indices_from(D, k=1)
    mu = D[iu].mean()
    sd = D[iu].std(ddof=1) or 1.0
    S = -(D - mu) / sd
    np.fill_diagonal(S, np.nan)
    return S


def load_embedding(model_path):
    m = Word2Vec.load(model_path)
    kv = m.wv
    return list(kv.index_to_key), np.asarray(kv.vectors, dtype=np.float64)


def load_corum(tsv_path, min_complex_size=3):
    df = pd.read_csv(tsv_path, sep="\t")
    out = {}
    for cid, sub in df.groupby("complex_id"):
        g = sorted(set(sub["gene"].astype(str)))
        if len(g) >= min_complex_size:
            out[str(cid)] = g
    return out


def evaluate(model_path, corum_path, out_dir, min_complex_size=3, label="model"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Loading embedding: {model_path}")
    genes, E = load_embedding(model_path)
    print(f"  genes={len(genes)}  dims={E.shape[1]}")

    print("Computing -Z-score similarity matrix...")
    S = neg_zscore_similarity_matrix(E)
    gene2idx = {g: i for i, g in enumerate(genes)}
    gene_set = set(genes)

    complexes_all = load_corum(corum_path, min_complex_size=min_complex_size)
    complexes = {}
    for cid, glist in complexes_all.items():
        kept = [g for g in glist if g in gene_set]
        if len(kept) >= min_complex_size:
            complexes[cid] = kept
    print(f"  complexes loaded: {len(complexes_all)}, evaluable: {len(complexes)}")

    corum_universe = sorted({g for glist in complexes.values() for g in glist})
    corum_idx = np.array([gene2idx[g] for g in corum_universe])
    print(f"  CORUM-in-embedding universe: {len(corum_universe)} genes")

    rows, all_within, all_between = [], [], []
    for cid, glist in complexes.items():
        in_idx = np.array([gene2idx[g] for g in glist])
        out_idx = np.setdiff1d(corum_idx, in_idx, assume_unique=False)
        sub = S[np.ix_(in_idx, in_idx)]
        iu = np.triu_indices_from(sub, k=1)
        within = sub[iu]
        between = S[np.ix_(in_idx, out_idx)].ravel()
        within = within[np.isfinite(within)]
        between = between[np.isfinite(between)]
        if len(within) == 0 or len(between) == 0:
            continue
        y = np.concatenate([np.ones_like(within), np.zeros_like(between)])
        s = np.concatenate([within, between])
        try:    auc = roc_auc_score(y, s)
        except Exception: auc = np.nan
        try:    _, p = mannwhitneyu(within, between, alternative="greater")
        except Exception: p = np.nan
        rows.append({"complex": cid, "n_genes": len(glist),
                     "n_within_pairs": len(within), "n_between_pairs": len(between),
                     "auc": auc, "p_value": p,
                     "within_mean": float(np.mean(within)),
                     "between_mean": float(np.mean(between))})
        all_within.append(within); all_between.append(between)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("No complexes evaluated."); return
    mask = df["p_value"].notna()
    df["p_adj"] = np.nan
    df.loc[mask, "p_adj"] = multipletests(df.loc[mask, "p_value"], method="fdr_bh")[1]

    aw = np.concatenate(all_within); ab = np.concatenate(all_between)
    y = np.concatenate([np.ones_like(aw), np.zeros_like(ab)])
    s = np.concatenate([aw, ab])
    global_auc = roc_auc_score(y, s)
    _, global_p = mannwhitneyu(aw, ab, alternative="greater")
    sig = df[df["p_adj"] < 0.05]
    summary = {
        "embedding": label,
        "n_complexes_evaluated": int(len(df)),
        "n_significant_q05": int(len(sig)),
        "frac_significant_q05": float(len(sig) / len(df)),
        "unfiltered_mean_auc": float(df["auc"].mean()),
        "unfiltered_median_auc": float(df["auc"].median()),
        "unfiltered_frac_above_0.5": float((df["auc"] > 0.5).mean()),
        "filtered_q05_mean_auc": float(sig["auc"].mean()) if len(sig) else None,
        "filtered_q05_median_auc": float(sig["auc"].median()) if len(sig) else None,
        "global_pooled_auc": float(global_auc),
        "global_pooled_p_value": float(global_p),
        "n_within_pairs_total": int(len(aw)),
        "n_between_pairs_total": int(len(ab)),
    }
    df.to_csv(os.path.join(out_dir, "per_complex.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===")
    for k, v in summary.items(): print(f"  {k}: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corum", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_complex_size", type=int, default=3)
    ap.add_argument("--label", default="model")
    a = ap.parse_args()
    evaluate(a.model, a.corum, a.out_dir, a.min_complex_size, a.label)


if __name__ == "__main__":
    main()
