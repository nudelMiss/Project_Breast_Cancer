#!/usr/bin/env python3
"""
CORUM AUC Benchmark for Gene Embeddings.

Computes AUC scores measuring how well embedding distances separate
genes within CORUM protein complexes from genes across complexes.

Algorithm:
1. Load embedding and CORUM complexes
2. Compute pairwise cosine distances for shared genes
3. Convert to similarity via negative z-score
4. For each complex: compute AUC (within-complex vs sampled between-complex pairs)
5. Output per-complex and summary statistics

Usage:
    python scripts/benchmark_corum_auc.py \
        --embedding_path results/results_by_patient_celltype_100_walks/patient=CID3586__celltype=T-cells/.../gene_embeddings.model \
        --corum_path resources/corum_core_complexes.tsv \
        --output_dir results/auc_benchmarks/walks_100 \
        --min_complex_size 3 \
        --random_seed 42
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu, zscore
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu


# ============================================================
# Data Classes
# ============================================================


@dataclass
class ComplexAUCResult:
    """Result for a single complex AUC computation."""

    complex_id: str
    complex_name: str
    complex_size: int  # Total genes in complex (from CORUM)
    n_genes_used: int  # Genes found in embedding intersection
    n_positive_pairs: int  # Within-complex pairs
    n_negative_pairs: int  # Between-complex pairs sampled
    auc: float
    p_value: float = np.nan
    adjusted_p_value: float = np.nan


@dataclass
class BenchmarkSummary:
    """Summary statistics across all complexes."""

    embedding_path: str
    n_embedding_genes: int
    n_shared_genes: int
    n_complexes_total: int
    n_complexes_used: int
    mean_auc: float
    median_auc: float
    weighted_mean_auc: float  # Weighted by n_positive_pairs
    p_value: float = np.nan

# ============================================================
# Loading Functions
# ============================================================


def load_embedding(embedding_path: Path) -> Tuple[List[str], np.ndarray]:
    """
    Load gene embedding from Word2Vec model.

    Args:
        embedding_path: Path to .model file

    Returns:
        Tuple of (gene_names, embedding_matrix) where matrix is (n_genes, dim)
    """
    model = Word2Vec.load(str(embedding_path))
    genes = list(model.wv.index_to_key)
    vectors = np.array([model.wv[g] for g in genes], dtype=np.float32)
    return genes, vectors


def load_corum(corum_path: Path) -> Dict[str, Tuple[str, Set[str]]]:
    """
    Load CORUM complexes from TSV.

    Args:
        corum_path: Path to TSV file with columns: complex_id, gene
            Optionally supports an extra complex_name column.

    Returns:
        Dict mapping complex_id -> (complex_name, set of genes)
    """
    df = pd.read_csv(corum_path, sep='\t')
    required = {'complex_id', 'gene'}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CORUM TSV must contain columns {sorted(required)}. Found: {list(df.columns)}"
        )

    has_name = 'complex_name' in df.columns
    complexes = {}
    for cid, group in df.groupby('complex_id'):
        name = str(group['complex_name'].iloc[0]) if has_name else str(cid)
        genes = set(group['gene'].str.strip().str.upper())
        complexes[str(cid)] = (name, genes)
    return complexes


def normalize_gene_names(genes: List[str]) -> List[str]:
    """Normalize gene names: strip whitespace, uppercase."""
    return [g.strip().upper() for g in genes]


# ============================================================
# Core Computation Functions
# ============================================================


def compute_pairwise_distances(vectors: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine distances for all gene vectors.

    Args:
        vectors: (n_genes, dim) embedding matrix

    Returns:
        Condensed distance vector from scipy.pdist (n*(n-1)/2 elements)
    """
    # Normalize for cosine distance
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Avoid division by zero
    normalized = vectors / norms

    # pdist with cosine returns 1 - cos_sim, range [0, 2]
    return pdist(normalized, metric='cosine')


def distances_to_similarity(distances: np.ndarray) -> np.ndarray:
    """
    Convert distances to similarity scores using negative z-score.

    Lower distance -> higher similarity -> higher score for positive class.

    Args:
        distances: Condensed distance array

    Returns:
        Similarity scores (higher = more similar)
    """
    # Z-score normalize, then negate (small distance = high similarity)
    with np.errstate(invalid='ignore'):
        z = zscore(distances)

    # Handle case where all distances are identical (zero variance)
    if np.all(np.isnan(z)):
        z = np.zeros_like(distances)

    z = np.nan_to_num(z, nan=0.0)
    return -z


def get_pair_index(i: int, j: int, n: int) -> int:
    """
    Get index into condensed distance array for pair (i, j) where i < j.

    For condensed array from pdist, index for (i,j) where i<j is:
    n*i + j - ((i+2)*(i+1))//2
    """
    if i > j:
        i, j = j, i
    return n * i + j - ((i + 2) * (i + 1)) // 2


def compute_complex_auc(
    complex_id: str,
    complex_name: str,
    complex_genes: Set[str],
    gene_to_idx: Dict[str, int],
    similarity_scores: np.ndarray,
    n_genes: int,
    min_complex_size: int,
    rng: np.random.Generator,
    all_embedding_genes: Set[str],
) -> Optional[ComplexAUCResult]:
    """
    Compute AUC for a single complex.

    Args:
        complex_id: Complex identifier
        complex_name: Complex name
        complex_genes: Set of genes in this complex (from CORUM)
        gene_to_idx: Mapping from gene name to index in embedding
        similarity_scores: Similarity scores (condensed form)
        n_genes: Total number of genes in embedding
        min_complex_size: Minimum genes required in complex
        rng: Random number generator
        all_embedding_genes: Set of all genes in embedding

    Returns:
        ComplexAUCResult or None if complex doesn't meet criteria
    """
    # Get genes that are both in complex AND in embedding
    genes_in_complex = complex_genes & all_embedding_genes

    if len(genes_in_complex) < min_complex_size:
        return None

    genes_list = sorted(genes_in_complex)

    # Within-complex pairs (positive class)
    within_pairs = []
    for i, g1 in enumerate(genes_list):
        for g2 in genes_list[i + 1:]:
            idx1, idx2 = gene_to_idx[g1], gene_to_idx[g2]
            pair_idx = get_pair_index(idx1, idx2, n_genes)
            within_pairs.append(pair_idx)

    n_positive = len(within_pairs)

    if n_positive == 0:
        return None

    # Sample between-complex pairs (negative class)
    # Pairs between genes IN the complex and genes OUTSIDE the complex
    non_complex_genes = all_embedding_genes - genes_in_complex
    non_complex_list = sorted(non_complex_genes)

    if len(non_complex_list) < 1:
        return None

    # Sample negative pairs: use at least 200 to stabilise AUC for small complexes
    n_neg_target = max(200, n_positive)
    between_pairs = []
    genes_in_list = list(genes_in_complex)

    for _ in range(n_neg_target):
        g1 = rng.choice(genes_in_list)
        g2 = rng.choice(non_complex_list)
        idx1, idx2 = gene_to_idx[g1], gene_to_idx[g2]
        pair_idx = get_pair_index(idx1, idx2, n_genes)
        between_pairs.append(pair_idx)

    n_negative = len(between_pairs)

    if n_negative == 0:
        return None

    # Compute AUC
    positive_scores = similarity_scores[within_pairs]
    negative_scores = similarity_scores[between_pairs]

    labels = np.concatenate([np.ones(n_positive), np.zeros(n_negative)])
    scores = np.concatenate([positive_scores, negative_scores])

    # Handle edge case: all scores identical
    if np.std(scores) < 1e-10:
        auc = 0.5
    else:
        auc = roc_auc_score(labels, scores)

    # One-sided test for enrichment of within-complex similarities.
    try:
        p_value = float(
            mannwhitneyu(positive_scores, negative_scores, alternative='two-sided').pvalue
        )
    except Exception:
        p_value = np.nan

    return ComplexAUCResult(
        complex_id=complex_id,
        complex_name=complex_name,
        complex_size=len(complex_genes),
        n_genes_used=len(genes_in_complex),
        n_positive_pairs=n_positive,
        n_negative_pairs=n_negative,
        auc=auc,
        p_value=p_value,
    )


def run_benchmark(
    embedding_path: Path,
    corum_path: Path,
    min_complex_size: int,
    random_seed: int,
) -> Tuple[List[ComplexAUCResult], BenchmarkSummary]:
    """
    Run the full CORUM AUC benchmark.

    Args:
        embedding_path: Path to embedding .model file
        corum_path: Path to CORUM TSV
        min_complex_size: Minimum genes per complex
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (per_complex_results, summary)
    """
    rng = np.random.default_rng(random_seed)

    # Load data
    print(f"Loading embedding: {embedding_path}", flush=True)
    genes, vectors = load_embedding(embedding_path)
    genes = normalize_gene_names(genes)
    n_embedding_genes = len(genes)

    print(f"Loading CORUM: {corum_path}", flush=True)
    complexes = load_corum(corum_path)
    n_complexes_total = len(complexes)

    # Find shared genes
    embedding_gene_set = set(genes)
    corum_gene_set = set()
    for _, (_, cg) in complexes.items():
        corum_gene_set.update(cg)

    shared_genes = embedding_gene_set & corum_gene_set

    print(f"Embedding genes: {n_embedding_genes}", flush=True)
    print(f"CORUM genes: {len(corum_gene_set)}", flush=True)
    print(f"Shared genes: {len(shared_genes)}", flush=True)

    if len(shared_genes) < 2:
        raise ValueError(
            f"Need at least 2 shared genes to compute pairwise distances, found {len(shared_genes)}"
        )

    # Restrict to shared genes only, as required by the benchmark design.
    shared_mask = np.array([g in shared_genes for g in genes], dtype=bool)
    genes = [g for g, keep in zip(genes, shared_mask) if keep]
    vectors = vectors[shared_mask]
    n_genes = len(genes)
    embedding_gene_set = set(genes)

    # Build gene index
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    # Compute pairwise distances
    print("Computing pairwise cosine distances...", flush=True)
    distances = compute_pairwise_distances(vectors)

    print("Converting to similarity scores via -zscore...", flush=True)
    similarity = distances_to_similarity(distances)

    # Process each complex
    print(f"Evaluating {n_complexes_total} complexes...", flush=True)
    results = []
    n_skipped = 0

    for cid, (cname, cgenes) in complexes.items():
        result = compute_complex_auc(
            complex_id=cid,
            complex_name=cname,
            complex_genes=cgenes,
            gene_to_idx=gene_to_idx,
            similarity_scores=similarity,
            n_genes=n_genes,
            min_complex_size=min_complex_size,
            rng=rng,
            all_embedding_genes=embedding_gene_set,
        )

        if result is not None:
            results.append(result)
        else:
            n_skipped += 1

    # BH correction across valid p-values from all evaluated complexes.
    if results:
        p_values = np.array([r.p_value for r in results], dtype=float)
        valid_mask = np.isfinite(p_values)
        if np.any(valid_mask):
            _, adjusted_vals, _, _ = multipletests(p_values[valid_mask], method='fdr_bh')
            adj_idx = 0
            for i, r in enumerate(results):
                if valid_mask[i]:
                    r.adjusted_p_value = float(adjusted_vals[adj_idx])
                    adj_idx += 1
                else:
                    r.adjusted_p_value = np.nan

    # Compute summary
    if results:
        aucs = np.array([r.auc for r in results])
        weights = np.array([r.n_positive_pairs for r in results])

        summary = BenchmarkSummary(
            embedding_path=str(embedding_path),
            n_embedding_genes=n_embedding_genes,
            n_shared_genes=len(shared_genes),
            n_complexes_total=n_complexes_total,
            n_complexes_used=len(results),
            mean_auc=float(np.mean(aucs)),
            median_auc=float(np.median(aucs)),
            weighted_mean_auc=float(np.average(aucs, weights=weights)),
            p_value=float(np.nanmedian(p_values)),
        )
    else:
        summary = BenchmarkSummary(
            embedding_path=str(embedding_path),
            n_embedding_genes=n_embedding_genes,
            n_shared_genes=len(shared_genes),
            n_complexes_total=n_complexes_total,
            n_complexes_used=0,
            mean_auc=np.nan,
            median_auc=np.nan,
            weighted_mean_auc=np.nan,
        )

    return results, summary


# ============================================================
# Output Functions
# ============================================================


def save_results(
    results: List[ComplexAUCResult],
    summary: BenchmarkSummary,
    output_dir: Path,
    embedding_path: Path,
    filter_significant: bool,
):
    """
    Save benchmark results to CSV files.

    Preserves the same directory structure as embeddings:
    output_dir / group_name / config_tag / corum_auc_*.csv
    """
    # Extract group and config tag from embedding path
    config_tag = embedding_path.parent.name
    group_name = embedding_path.parent.parent.name

    out_subdir = output_dir / group_name / config_tag
    out_subdir.mkdir(parents=True, exist_ok=True)

    # Per-complex results
    per_complex_path = out_subdir / "corum_auc_per_complex.csv"
    df_results = pd.DataFrame(
        [
            {
                'complex_id': r.complex_id,
                'complex_name': r.complex_name,
                'complex_size': r.complex_size,
                'n_genes_used': r.n_genes_used,
                'n_positive_pairs': r.n_positive_pairs,
                'n_negative_pairs': r.n_negative_pairs,
                'auc': r.auc,
                'p_value': r.p_value,
                'adjusted_p_value': r.adjusted_p_value,
            }
            for r in results
        ]
    )

    if filter_significant:
        before_filter = len(df_results)

        df_results = df_results[
            df_results['adjusted_p_value'].notna() &
            (df_results['adjusted_p_value'] < 0.05)
        ].copy()

        after_filter = len(df_results)

        print(
            f"Significance filter: kept {after_filter}/{before_filter} complexes "
            f"with adjusted_p_value < 0.05",
            flush=True,
        )

    df_results.to_csv(per_complex_path, index=False)

    # Summary
    summary_path = out_subdir / "corum_auc_summary.csv"
    df_summary = pd.DataFrame(
        [
            {
                'embedding_path': summary.embedding_path,
                'n_embedding_genes': summary.n_embedding_genes,
                'n_shared_genes': summary.n_shared_genes,
                'n_complexes_total': summary.n_complexes_total,
                'n_complexes_used': summary.n_complexes_used,
                'mean_auc': summary.mean_auc,
                'median_auc': summary.median_auc,
                'weighted_mean_auc': summary.weighted_mean_auc,
                'p_value': summary.p_value,
            }
        ]
    )
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}", flush=True)


# ============================================================
# Main
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description='CORUM AUC Benchmark for Gene Embeddings'
    )
    parser.add_argument(
        '--embedding_path',
        type=str,
        required=True,
        help='Path to gene_embeddings.model file',
    )
    parser.add_argument(
        '--corum_path',
        type=str,
        default='resources/corum_core_complexes.tsv',
        help='Path to CORUM complexes TSV',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/auc_benchmarks/',
        help='Output directory for results',
    )
    parser.add_argument(
        '--min_complex_size',
        type=int,
        default=3,
        help='Minimum genes in complex to evaluate',
    )
    parser.add_argument(
        '--random_seed', type=int, default=42, help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--filter_significant',
        action='store_true',
        help='If set, keep only per-complex rows with adjusted_p_value < 0.05.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    embedding_path = Path(args.embedding_path)
    corum_path = Path(args.corum_path)
    output_dir = Path(args.output_dir)

    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding not found: {embedding_path}")

    if not corum_path.exists():
        raise FileNotFoundError(f"CORUM file not found: {corum_path}")

    results, summary = run_benchmark(
        embedding_path=embedding_path,
        corum_path=corum_path,
        min_complex_size=args.min_complex_size,
        random_seed=args.random_seed,
    )

    save_results(
        results,
        summary,
        output_dir,
        embedding_path,
        filter_significant=args.filter_significant,
    )

    print("\n=== SUMMARY ===", flush=True)
    print(f"Complexes used: {summary.n_complexes_used}", flush=True)
    print(f"Complexes skipped: {summary.n_complexes_total - summary.n_complexes_used}", flush=True)
    print(f"Mean AUC: {summary.mean_auc:.4f}", flush=True)
    print(f"Median AUC: {summary.median_auc:.4f}", flush=True)
    print(f"Weighted Mean AUC: {summary.weighted_mean_auc:.4f}", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
