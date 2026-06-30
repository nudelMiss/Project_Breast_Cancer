#!/usr/bin/env python3
"""
CORUM MCC benchmark for gene embeddings.

For a trained Word2Vec model:
  1. Build the universe of "shared" genes = (model vocab) ∩ (any gene in any CORUM complex).
  2. Positives = all gene pairs (i, j) where i and j are both in the shared universe
                 AND co-occur in at least one CORUM complex.
  3. Negatives = a random sample of gene pairs (i, j) within the shared universe
                 that DO NOT co-occur in any CORUM complex.
                 Size: max(200, n_positives), capped at --max_negatives.
  4. Score = cosine similarity between embeddings.
  5. For each candidate threshold in a fine grid, compute MCC, precision, recall, F1,
     and confusion matrix entries (TP, FP, TN, FN).
  6. Output:
       - mcc_best  + threshold_best  + accompanying metrics
       - mcc_at_0.5 + same accompanying metrics  (fixed-threshold comparability)
       - n_pos, n_neg, n_shared_genes, n_complexes_used, embedding_path, ...
     Plus a full threshold-sweep CSV.

Usage:
    python scripts/MCC/benchmark_corum_mcc.py \
        --embedding_path .../gene_embeddings.model \
        --corum_path resources/corum_core_complexes.tsv \
        --output_dir .../mcc/<group>/<tag>/ \
        --min_complex_size 3 --random_seed 42
"""
from __future__ import annotations
import argparse, csv, json, sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics import matthews_corrcoef


# ============================================================
# Loaders
# ============================================================

def load_embedding(path: Path):
    model = Word2Vec.load(str(path))
    genes = [g.upper().strip() for g in model.wv.index_to_key]
    vectors = np.array([model.wv[g] for g in model.wv.index_to_key], dtype=np.float32)
    # L2-normalize so dot product == cosine similarity.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    return genes, vectors


def load_corum(path: Path, min_complex_size: int):
    df = pd.read_csv(path, sep="\t")
    if not {"complex_id", "gene"}.issubset(df.columns):
        raise ValueError(f"CORUM TSV must have columns complex_id, gene; got {df.columns.tolist()}")
    df["gene"] = df["gene"].astype(str).str.upper().str.strip()
    complexes = {}
    for cid, grp in df.groupby("complex_id"):
        gs = set(grp["gene"]) - {""}
        if len(gs) >= min_complex_size:
            complexes[str(cid)] = gs
    return complexes


# ============================================================
# Pair construction
# ============================================================

def build_positive_pairs(complexes_in_vocab: dict[str, set]):
    """Return a set of frozenset({g_i, g_j}) for all within-complex pairs."""
    pos = set()
    for cid, genes in complexes_in_vocab.items():
        gs = sorted(genes)
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                pos.add(frozenset((gs[i], gs[j])))
    return pos


def sample_negative_pairs(universe_sorted, positive_pairs, n_target, rng, hard_cap=200_000):
    """
    Sample up to n_target negative pairs (no replacement in spirit) by random index pairs.
    Cap at hard_cap to keep runtime sane on huge complexes.
    """
    target = min(int(n_target), int(hard_cap))
    n_u = len(universe_sorted)
    if n_u < 2 or target <= 0:
        return []

    negs = set()
    # Try up to ~5x oversample to fill target
    attempts = 0
    max_attempts = target * 8 + 1000
    while len(negs) < target and attempts < max_attempts:
        i = int(rng.integers(0, n_u))
        j = int(rng.integers(0, n_u))
        if i == j:
            attempts += 1; continue
        pair = frozenset((universe_sorted[i], universe_sorted[j]))
        if pair in positive_pairs or pair in negs:
            attempts += 1; continue
        negs.add(pair)
        attempts += 1
    return list(negs)


def pair_cosines(pairs, gene_to_idx, vectors):
    """vectors must be L2-normalized."""
    out = np.empty(len(pairs), dtype=np.float32)
    for k, pr in enumerate(pairs):
        g1, g2 = tuple(pr) if len(pr) == 2 else (next(iter(pr)),) * 2
        i = gene_to_idx[g1]; j = gene_to_idx[g2]
        out[k] = float(np.dot(vectors[i], vectors[j]))
    return out


# ============================================================
# Threshold sweep
# ============================================================

def confusion_at_threshold(scores, labels, thresh):
    """labels: 1 for positive, 0 for negative."""
    pred = (scores >= thresh).astype(np.int8)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    p_denom = tp + fp
    r_denom = tp + fn
    precision = tp / p_denom if p_denom else 0.0
    recall = tp / r_denom if r_denom else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=precision, recall=recall, f1=f1)


def mcc_from_cm(cm):
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    denom = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denom == 0.0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


def sweep_thresholds(scores, labels, grid):
    rows = []
    for t in grid:
        cm = confusion_at_threshold(scores, labels, float(t))
        cm["threshold"] = float(t)
        cm["mcc"] = mcc_from_cm(cm)
        rows.append(cm)
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="CORUM MCC benchmark for gene embeddings")
    p.add_argument("--embedding_path", required=True, type=Path)
    p.add_argument("--corum_path", type=Path, default=Path("resources/corum_core_complexes.tsv"))
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--min_complex_size", type=int, default=3)
    p.add_argument("--max_negatives", type=int, default=100_000,
                   help="Hard cap on number of negative pairs sampled.")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--fixed_threshold", type=float, default=0.5,
                   help="Threshold for the reported 'mcc_at_fixed' metric (default 0.5).")
    p.add_argument("--threshold_step", type=float, default=0.01,
                   help="Threshold grid resolution for the sweep.")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.embedding_path.exists():
        sys.exit(f"FATAL: embedding not found: {args.embedding_path}")
    if not args.corum_path.exists():
        sys.exit(f"FATAL: CORUM not found: {args.corum_path}")

    print(f"[MCC] embedding: {args.embedding_path}", flush=True)
    print(f"[MCC] corum    : {args.corum_path}", flush=True)
    genes, vectors = load_embedding(args.embedding_path)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    print(f"[MCC] embedding genes: {len(genes):,}", flush=True)

    complexes = load_corum(args.corum_path, args.min_complex_size)
    print(f"[MCC] CORUM complexes (size >= {args.min_complex_size}): {len(complexes):,}", flush=True)

    # Restrict each complex to the genes present in the embedding vocabulary
    cv = {cid: (g & set(genes)) for cid, g in complexes.items()}
    cv = {cid: g for cid, g in cv.items() if len(g) >= 2}
    print(f"[MCC] complexes with >=2 shared genes: {len(cv):,}", flush=True)

    # Universe of genes that appear in any kept complex
    universe = set().union(*cv.values()) if cv else set()
    universe_sorted = sorted(universe)
    print(f"[MCC] shared-gene universe: {len(universe_sorted):,}", flush=True)

    positives = build_positive_pairs(cv)
    print(f"[MCC] positive pairs: {len(positives):,}", flush=True)
    if len(positives) == 0:
        sys.exit("FATAL: no positive pairs after intersecting CORUM with the model vocabulary.")

    n_target_neg = max(200, len(positives))
    rng = np.random.default_rng(args.random_seed)
    negatives = sample_negative_pairs(universe_sorted, positives, n_target_neg, rng, args.max_negatives)
    print(f"[MCC] negative pairs sampled: {len(negatives):,} (target {n_target_neg:,})", flush=True)

    all_pairs = list(positives) + negatives
    labels = np.array([1] * len(positives) + [0] * len(negatives), dtype=np.int8)
    scores = pair_cosines(all_pairs, gene_to_idx, vectors)

    # Threshold grid: cosine in [-1, 1].
    grid = np.round(np.arange(-1.0, 1.0 + args.threshold_step / 2, args.threshold_step), 4)
    sweep = sweep_thresholds(scores, labels, grid)
    sweep_csv = args.output_dir / "mcc_threshold_sweep.csv"
    sweep.to_csv(sweep_csv, index=False)
    print(f"[WRITE] {sweep_csv}", flush=True)

    # Best MCC
    best_row = sweep.iloc[int(sweep["mcc"].idxmax())].to_dict()
    # Fixed threshold (0.5 by default), choose the row closest on the grid
    fixed = float(args.fixed_threshold)
    sweep["_d"] = (sweep["threshold"] - fixed).abs()
    fixed_row = sweep.iloc[int(sweep["_d"].idxmin())].to_dict()
    fixed_row.pop("_d", None)
    sweep.drop(columns=["_d"], inplace=True)

    summary = {
        "embedding_path": str(args.embedding_path),
        "corum_path": str(args.corum_path),
        "n_embedding_genes": len(genes),
        "n_shared_genes": len(universe_sorted),
        "n_complexes_total": len(complexes),
        "n_complexes_used": len(cv),
        "n_positives": int(len(positives)),
        "n_negatives": int(len(negatives)),
        "random_seed": int(args.random_seed),
        "min_complex_size": int(args.min_complex_size),
        "max_negatives": int(args.max_negatives),
        "threshold_step": float(args.threshold_step),
        "mcc_best": float(best_row["mcc"]),
        "threshold_best": float(best_row["threshold"]),
        "precision_best": float(best_row["precision"]),
        "recall_best": float(best_row["recall"]),
        "f1_best": float(best_row["f1"]),
        "tp_best": int(best_row["tp"]),
        "fp_best": int(best_row["fp"]),
        "tn_best": int(best_row["tn"]),
        "fn_best": int(best_row["fn"]),
        "mcc_at_fixed": float(fixed_row["mcc"]),
        "fixed_threshold": float(fixed_row["threshold"]),
        "precision_at_fixed": float(fixed_row["precision"]),
        "recall_at_fixed": float(fixed_row["recall"]),
        "f1_at_fixed": float(fixed_row["f1"]),
        "tp_at_fixed": int(fixed_row["tp"]),
        "fp_at_fixed": int(fixed_row["fp"]),
        "tn_at_fixed": int(fixed_row["tn"]),
        "fn_at_fixed": int(fixed_row["fn"]),
    }

    summary_csv = args.output_dir / "mcc_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    print(f"[WRITE] {summary_csv}", flush=True)

    summary_json = args.output_dir / "mcc_summary.json"
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[WRITE] {summary_json}", flush=True)

    # One-line headline
    print("[RESULT] "
          f"mcc_best={summary['mcc_best']:.4f} @ t={summary['threshold_best']:.2f}  "
          f"mcc@{summary['fixed_threshold']:.2f}={summary['mcc_at_fixed']:.4f}  "
          f"n_pos={summary['n_positives']:,}  n_neg={summary['n_negatives']:,}", flush=True)


if __name__ == "__main__":
    main()
