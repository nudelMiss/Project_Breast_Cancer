#!/usr/bin/env python3
import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import igraph as ig
import numpy as np
import pandas as pd
from scipy.io import mmread
from tqdm import tqdm

# GPU cosine
import torch
import torch.nn.functional as F

# Spearman
from scipy.stats import rankdata

# Word2Vec
from gensim.models import Word2Vec

# ============================================================
# Global constants and settings
# ============================================================

WALK_LENGTH = 6

# ============================================================
# Helpers
# ============================================================

def _get_slurm_cpus(default: int = 4) -> int:
    for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"):
        v = os.environ.get(k)
        if not v:
            continue
        v = v.split("(")[0]  # e.g. "6(x1)"
        try:
            return int(v)
        except ValueError:
            pass
    return default


def _safe_load_genes(genes_csv: Path) -> List[str]:
    s = pd.read_csv(genes_csv, header=None).iloc[:, 0]
    s = s.dropna().astype(str).str.strip()
    genes = s[s != ""].tolist()
    return genes


def _align_expr_and_genes(mat, genes: List[str]):
    # Ensure rows = genes
    if mat.shape[0] != len(genes) and mat.shape[1] == len(genes):
        print("Transposing expr matrix so rows are genes", flush=True)
        mat = mat.T.tocsr()

    if mat.shape[0] != len(genes):
        diff = len(genes) - mat.shape[0]
        print(f"[WARN] gene/matrix mismatch: mat_rows={mat.shape[0]} n_genes={len(genes)} diff={diff}", flush=True)
        if abs(diff) <= 5:
            new_n = min(mat.shape[0], len(genes))
            print(f"[WARN] Aligning by truncation to n={new_n}", flush=True)
            genes = genes[:new_n]
            mat = mat[:new_n, :]
        else:
            raise ValueError(f"After transpose: mat.shape={mat.shape} but n_genes={len(genes)} diff={diff}")
    return mat, genes


# ============================================================
# Similarity backends
# ============================================================

def compute_topk_cosine_gpu(expression_matrix, gene_names, k: int, batch_size: int = 1024) -> List[Tuple[str, str, float]]:
    """
    Returns directed edges (src, dst, weight) from topk neighbors per src gene.
    We'll later build an undirected igraph and simplify with combine_edges.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available, cannot run cosine_gpu backend.")

    device = torch.device("cuda")

    print("[GPU] Converting sparse -> dense float32 (CPU) ...", flush=True)
    dense = expression_matrix.astype(np.float32).toarray()

    print("[GPU] Moving dense matrix to GPU ...", flush=True)
    X = torch.from_numpy(dense).to(device)

    print("[GPU] X shape:", tuple(X.shape), flush=True)

    X = F.normalize(X, p=2, dim=1)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n = X.shape[0]
    edges = []

    print(f"[GPU] Cosine batches (n={n}, batch={batch_size}, k={k}) ...", flush=True)

    for start in tqdm(range(0, n, batch_size), desc="cosine_gpu"):
        end = min(start + batch_size, n)

        sims = X[start:end] @ X.T  # (b, n)

        # mask self-similarity for rows that correspond to the same global index
        rows = torch.arange(end - start, device=device)
        cols = torch.arange(start, end, device=device)
        sims[rows, cols] = -float("inf")

        vals, idx = torch.topk(sims, k=k, dim=1)

        vals = vals.detach().cpu().numpy()
        idx = idx.detach().cpu().numpy()

        for i in range(end - start):
            gi = gene_names[start + i]
            for j, w in zip(idx[i], vals[i]):
                if np.isfinite(w):
                    edges.append((gi, gene_names[int(j)], float(w)))

        del sims

    return edges


def compute_spearman_edges_cpu(
    expression_matrix,
    gene_names: List[str],
    mode: str,               # "topk" or "threshold"
    k: int,
    threshold: float,
    chunk_rows: int = 256,
    chunk_cols: int = 2048,
) -> List[Tuple[str, str, float]]:
    """
    Spearman correlation between gene vectors (rows).
    Implemented as Pearson on ranks, computed in chunks to avoid O(n^2) RAM blowups.

    Returns directed edges (src, dst, weight) selected by mode.
    """
    # Convert sparse to dense float32
    dense = expression_matrix.astype(np.float32).toarray()  # (n_genes, n_cells)
    n, m = dense.shape
    print(f"[Spearman CPU] dense shape={dense.shape}", flush=True)

    # rank-transform each row (gene) across cells
    print("[Spearman CPU] Ranking rows...", flush=True)
    # rankdata along axis=1; use average ranks for ties (default)
    R = np.apply_along_axis(rankdata, 1, dense).astype(np.float32)

    # standardize rows: subtract mean, divide std (for Pearson)
    print("[Spearman CPU] Standardizing ranks...", flush=True)
    R -= R.mean(axis=1, keepdims=True)
    denom = R.std(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    R /= denom

    edges: List[Tuple[str, str, float]] = []
    print(f"[Spearman CPU] Computing correlations in chunks (chunk_rows={chunk_rows}, chunk_cols={chunk_cols})", flush=True)

    # We compute corr block-wise: Corr(A,B) = A @ B.T / (m-1) but after standardization it’s ~ mean of products
    # Since we standardized to unit std, dot/(m-1) approximates Pearson; we can use dot/m for stability.
    scale = float(m)  # use m instead of (m-1) for simplicity

    for rs in tqdm(range(0, n, chunk_rows), desc="spearman_rows"):
        re = min(rs + chunk_rows, n)
        A = R[rs:re]  # (br, m)

        # We will scan columns in blocks, but we still need selection per row.
        # For topk mode: keep a running topk for each row.
        if mode == "topk":
            best_vals = np.full((re - rs, k), -np.inf, dtype=np.float32)
            best_idx = np.full((re - rs, k), -1, dtype=np.int32)

        for cs in range(0, n, chunk_cols):
            ce = min(cs + chunk_cols, n)
            B = R[cs:ce]  # (bc, m)
            corr = (A @ B.T) / scale  # (br, bc)

            # mask self for overlapping region
            if rs <= cs < re or cs <= rs < ce:
                # overlap exists; mask diagonal elements that correspond to same gene
                for i in range(re - rs):
                    gi = rs + i
                    if cs <= gi < ce:
                        corr[i, gi - cs] = -np.inf if mode == "topk" else 0.0

            if mode == "threshold":
                # collect edges above threshold
                hits = np.where(corr >= threshold)
                for i, j in zip(hits[0], hits[1]):
                    src = gene_names[rs + i]
                    dst = gene_names[cs + j]
                    w = float(corr[i, j])
                    edges.append((src, dst, w))
            else:
                # topk: merge block topk into running topk
                # For each row, get local topk from this block
                local_k = min(k, ce - cs)
                loc_vals = np.partition(corr, -local_k, axis=1)[:, -local_k:]
                # indices of those values
                loc_idx = np.argpartition(corr, -local_k, axis=1)[:, -local_k:]

                # Now merge (best_vals,best_idx) with (loc_vals, loc_idx+cs)
                merged_vals = np.concatenate([best_vals, loc_vals], axis=1)
                merged_idx = np.concatenate([best_idx, loc_idx + cs], axis=1)

                # pick topk from merged
                take = np.argpartition(merged_vals, -k, axis=1)[:, -k:]
                # sort within topk descending (optional)
                row_ids = np.arange(re - rs)[:, None]
                best_vals = merged_vals[row_ids, take]
                best_idx = merged_idx[row_ids, take]
                order = np.argsort(best_vals, axis=1)[:, ::-1]
                best_vals = best_vals[row_ids, order]
                best_idx = best_idx[row_ids, order]

        if mode == "topk":
            for i in range(re - rs):
                src = gene_names[rs + i]
                for j, w in zip(best_idx[i], best_vals[i]):
                    if j >= 0 and np.isfinite(w):
                        edges.append((src, gene_names[int(j)], float(w)))

    return edges


# ============================================================
# Edge selection wrapper
# ============================================================

def compute_edges(
    mat,
    genes: List[str],
    sim: str,
    edge_mode: str,
    k: int,
    threshold: float,
    cos_batch: int,
    spearman_chunk_rows: int,
    spearman_chunk_cols: int,
) -> List[Tuple[str, str, float]]:
    if sim == "cosine":
        if edge_mode != "topk":
            raise ValueError("cosine backend currently supports edge_mode=topk only (use spearman for threshold mode).")
        return compute_topk_cosine_gpu(mat, genes, k=k, batch_size=cos_batch)

    if sim == "spearman":
        return compute_spearman_edges_cpu(
            mat, genes,
            mode=edge_mode,
            k=k,
            threshold=threshold,
            chunk_rows=spearman_chunk_rows,
            chunk_cols=spearman_chunk_cols,
        )

    raise ValueError(f"Unknown sim={sim}")


# ============================================================
# Random walk optimization (cache neighbors + weights)
# ============================================================

def build_weight_cache(graph: ig.Graph):
    nb_cache = []
    w_cache = []
    for v in range(graph.vcount()):
        nbs = graph.neighbors(v)
        if not nbs:
            nb_cache.append([])
            w_cache.append([])
            continue

        ws = []
        for nb in nbs:
            eid = graph.get_eid(v, nb, directed=False, error=False)
            ws.append(float(graph.es[eid]["weight"]) if eid != -1 else 0.0)

        nb_cache.append(nbs)
        w_cache.append(ws)
    return nb_cache, w_cache


def weighted_next_vertex_cached(graph: ig.Graph, current_idx: int, nb_cache, w_cache):
    nbs = nb_cache[current_idx]
    if not nbs:
        return None
    ws = w_cache[current_idx]
    total = float(sum(ws))
    if total <= 0.0:
        return graph.vs[random.choice(nbs)]
    chosen_nb = random.choices(nbs, weights=ws, k=1)[0]
    return graph.vs[chosen_nb]


def bidirectional_walk_cached(graph: ig.Graph, central_idx: int, walk_length: int, nb_cache, w_cache):
    start_v = graph.vs[central_idx]
    central_gene = start_v["name"]

    left = []
    cur_idx = central_idx
    for _ in range(walk_length):
        nxt = weighted_next_vertex_cached(graph, cur_idx, nb_cache, w_cache)
        if nxt is None:
            break
        left.append(nxt["name"])
        cur_idx = nxt.index

    right = []
    cur_idx = central_idx
    for _ in range(walk_length):
        nxt = weighted_next_vertex_cached(graph, cur_idx, nb_cache, w_cache)
        if nxt is None:
            break
        right.append(nxt["name"])
        cur_idx = nxt.index

    return left[::-1] + [central_gene] + right


class WalkCorpus:
    def __init__(self, graph, walk_length, walks_per_gene, seed, nb_cache, w_cache):
        self.graph = graph
        self.walk_length = walk_length
        self.walks_per_gene = walks_per_gene
        self.seed = seed
        self.nb_cache = nb_cache
        self.w_cache = w_cache
        self.total_examples = len(graph.vs) * walks_per_gene

    def __iter__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        for v_idx in range(self.graph.vcount()):
            for _ in range(self.walks_per_gene):
                sent = bidirectional_walk_cached(
                    self.graph, v_idx, self.walk_length, self.nb_cache, self.w_cache
                )
                if len(sent) > 1:
                    yield sent


# ============================================================
# Config + Naming
# ============================================================

@dataclass
class RunConfig:
    sim: str
    edge_mode: str
    k_nearest: int
    edge_threshold: float
    walks_per_gene: int
    walk_length: int
    vector_dim: int
    epochs: int
    window: int
    min_count: int
    seed: int
    combine_edges: str
    materialize: bool
    cos_batch: int
    spearman_chunk_rows: int
    spearman_chunk_cols: int

    def tag(self) -> str:
        # short but informative directory name
        parts = [
            f"sim={self.sim}",
            f"emode={self.edge_mode}",
            f"k={self.k_nearest}" if self.edge_mode == "topk" else f"thr={self.edge_threshold}",
            f"walks={self.walks_per_gene}",
            f"wlen={self.walk_length}",
            f"dim={self.vector_dim}",
            f"ep={self.epochs}",
            f"win={self.window}",
            f"mc={self.min_count}",
            f"seed={self.seed}",
        ]
        return "__".join(parts)


# ============================================================
# Run One Group
# ============================================================

def run_one_group(group_dir: Path, out_root: Path, cfg: RunConfig, *, save_edges: bool, skip_w2v: bool):
    print(f"\n=== GROUP: {group_dir.name} ===", flush=True)

    out_dir = out_root / group_dir.name / cfg.tag()
    model_path = out_dir / "gene_embeddings.model"
    if model_path.exists():
        print(f"[SKIP] Existing model found: {model_path}", flush=True)
        return

    mat = mmread(str(group_dir / "expr.mtx")).tocsr()
    genes = _safe_load_genes(group_dir / "genes.csv")

    print("expr shape:", mat.shape, "n_genes:", len(genes), flush=True)
    mat, genes = _align_expr_and_genes(mat, genes)

    # =========================
    # 🔴 DEBUG: sparsity check
    # =========================
    import numpy as np

    nonzero_frac = (mat > 0).sum(axis=1) / mat.shape[1]
    overall_nonzero_pct = (mat.nnz / (mat.shape[0] * mat.shape[1])) * 100

    print("\n=== NON-ZERO FRACTION STATS ===")
    print("min:", float(nonzero_frac.min()))
    print("mean:", float(nonzero_frac.mean()))
    print("max:", float(nonzero_frac.max()))
    print("Overall Matrix non-zero %:", overall_nonzero_pct)
    # =========================

    # =========================
    # 🔴 FIX: remove invalid gene 'x'
    # =========================
    valid_idx = [i for i, g in enumerate(genes) if g and g.lower() != "x"]

    genes = [genes[i] for i in valid_idx]
    mat = mat[valid_idx, :]

    print(f"[CLEAN] removed 'x' genes, new shape: {mat.shape}, n_genes: {len(genes)}", flush=True)
    # =========================

    # =========================
    # FILTER: keep genes expressed in at least 3% of cells
    # =========================
    min_expr_frac = 0.03

    nonzero_frac = np.array((mat > 0).sum(axis=1)).flatten() / mat.shape[1]
    keep_mask = nonzero_frac >= min_expr_frac

    before_n = len(genes)
    mat = mat[keep_mask, :]
    genes = [g for g, keep in zip(genes, keep_mask) if keep]

    print(
        f"[FILTER] expression filter >= {min_expr_frac:.2%} cells: "
        f"kept {len(genes)}/{before_n} genes, new shape={mat.shape}",
        flush=True,
    )
    # =========================

    # --- compute edges ---
    print(f"Computing edges: sim={cfg.sim} edge_mode={cfg.edge_mode} ...", flush=True)
    edges = compute_edges(
        mat, genes,
        sim=cfg.sim,
        edge_mode=cfg.edge_mode,
        k=cfg.k_nearest,
        threshold=cfg.edge_threshold,
        cos_batch=cfg.cos_batch,
        spearman_chunk_rows=cfg.spearman_chunk_rows,
        spearman_chunk_cols=cfg.spearman_chunk_cols,
    )
    print(f"Edges computed: {len(edges):,}", flush=True)

    # --- output directory for this config + group ---
    out_dir.mkdir(parents=True, exist_ok=True)

    # save config
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # optional: save edges to TSV
    if save_edges:
        df = pd.DataFrame(edges, columns=["src", "dst", "weight"])
        df.to_csv(out_dir / "edges.tsv", sep="\t", index=False)

    # --- build graph ---
    print("Building igraph ...", flush=True)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    # edges can include genes outside mapping if something weird happened; guard:
    edge_tuples = []
    edge_weights = []
    for s, t, w in edges:
        if s in gene_to_idx and t in gene_to_idx:
            edge_tuples.append((gene_to_idx[s], gene_to_idx[t]))
            edge_weights.append(float(w))

    g = ig.Graph(n=len(genes), edges=edge_tuples, directed=False)
    g.vs["name"] = genes
    g.es["weight"] = edge_weights

    combine = {"mean": "mean", "max": "max", "sum": "sum"}[cfg.combine_edges]
    g.simplify(combine_edges={"weight": combine})

    print(f"Graph: |V|={g.vcount():,} |E|={g.ecount():,}", flush=True)

    # =========================
    # 🔴 DEBUG: GRAPH NEIGHBORS + SIMILARITY
    # =========================
    def debug_graph_neighbors(g, gene, top_k=20):
        try:
            idx = g.vs.find(name=gene).index
        except:
            print(f"{gene} not found in graph")
            return

        neighbors = g.neighbors(idx)

        rows = []
        for nb in neighbors:
            eid = g.get_eid(idx, nb)
            w = g.es[eid]["weight"]
            rows.append((g.vs[nb]["name"], w))

        rows = sorted(rows, key=lambda x: -x[1])[:top_k]

        print(f"\n=== GRAPH NEIGHBORS for {gene} ===")
        for name, weight in rows:
            print(f"{name}: {weight:.4f}")


    # run on same genes as walk debug
    example_genes = [
        # Core T-cell markers
        "CD3D",
        "CD3E",
        "CD3G",
        "TRAC",
        "IL7R",
        "LTB",

        # TCR signaling
        "LCK",
        "LAT",
        "PTPRC",
        "SKAP1",
        "ITK",

        # Cytotoxic / CD8-like
        "NKG7",
        "CCL5",
        "GZMK",
        "GZMB",
        "PRF1",

        # Activation / immune regulation
        "LTB",
        "MALAT1",
        "MHC2TA",
        "TIGIT",
        "PDCD1",
        "CTLA4",

        # Naive / memory-like
        "MAL",
        "LTB",
        "IL32",
        "TCF7",
        "LEF1",
    ]

    for gene in example_genes:
        debug_graph_neighbors(g, gene)
    # =========================

    # =========================
    # 🔴 DEBUG WALKS (moved before skip_w2v)
    # =========================
    print("Caching neighbors+weights ...", flush=True)
    nb_cache, w_cache = build_weight_cache(g)

    corpus = WalkCorpus(g, cfg.walk_length, cfg.walks_per_gene, cfg.seed, nb_cache, w_cache)

    print("Materializing walks into RAM (DEBUG)...", flush=True)
    walks = list(corpus)
    print(f"Materialized {len(walks):,} walks", flush=True)

    # --- print sample walks ---
    print("\n=== SAMPLE WALKS ===")
    for w in walks[:5]:
        print(w)

    # --- frequency debug ---
    from collections import Counter

    def debug_gene_walks(gene, walks, top_k=20):
        gene_walks = [w for w in walks if w[len(w)//2] == gene]
        all_genes = [g for w in gene_walks for g in w if g != gene]
        counts = Counter(all_genes)

        print(f"\n=== WALK ANALYSIS for {gene} ===")
        for g, c in counts.most_common(top_k):
            print(f"{g}: {c}")

    example_genes = [
        # Core T-cell markers
        "CD3D",
        "CD3E",
        "CD3G",
        "TRAC",
        "IL7R",
        "LTB",

        # TCR signaling
        "LCK",
        "LAT",
        "PTPRC",
        "SKAP1",
        "ITK",

        # Cytotoxic / CD8-like
        "NKG7",
        "CCL5",
        "GZMK",
        "GZMB",
        "PRF1",

        # Activation / immune regulation
        "LTB",
        "MALAT1",
        "MHC2TA",
        "TIGIT",
        "PDCD1",
        "CTLA4",

        # Naive / memory-like
        "MAL",
        "LTB",
        "IL32",
        "TCF7",
        "LEF1",
    ]
    for gene in example_genes:
        debug_gene_walks(gene, walks)

    # =========================

    if skip_w2v:
        print("skip_w2v=True => stopping after debug.", flush=True)
        return

    # --- respect materialize flag for training ---
    if cfg.materialize:
        corpus_for_w2v = walks
        total_examples = len(walks)
    else:
        corpus_for_w2v = corpus
        total_examples = corpus.total_examples
    # Initialize the Word2Vec model before building vocab
    print("Initializing Word2Vec model...", flush=True)
    model = Word2Vec(
        vector_size=cfg.vector_dim,
        window=5,
        min_count=1,
        sg=1,
        workers=8
    )
    print("Building vocab ...", flush=True)
    model.build_vocab(corpus_for_w2v)

    print("Training Word2Vec ...", flush=True)
    model.train(
        corpus_for_w2v,
        total_examples=total_examples,
        epochs=cfg.epochs,
    )

    model.save(str(model_path))
    print(f"Saved model: {model_path}", flush=True)


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    # IO
    p.add_argument("--in_root", type=str, default="exports_by_patient_celltype")
    p.add_argument("--out_root", "--output_dir", dest="out_root", type=str, default="results_by_patient_celltype")
    p.add_argument("--only_group", type=str, default=None)  # e.g. bc1000_..._T-cells
    p.add_argument("--save_edges", action="store_true")
    p.add_argument("--skip_w2v", action="store_true")

    # Variant knobs
    p.add_argument("--sim", choices=["cosine", "spearman"], default="cosine")
    p.add_argument("--edge_mode", choices=["topk", "threshold"], default="topk")
    p.add_argument("--k_nearest", type=int, default=50)
    p.add_argument("--edge_threshold", type=float, default=0.3, help="Used when edge_mode=threshold")

    # Random walks
    p.add_argument("--walks", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)

    # Word2Vec
    p.add_argument("--vector_dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--min_count", type=int, default=5)

    # Graph simplify combine
    p.add_argument("--combine_edges", choices=["mean", "max", "sum"], default="mean")

    # Performance
    p.add_argument("--cos_batch", type=int, default=1024)
    p.add_argument("--materialize", action="store_true")

    # Spearman chunking
    p.add_argument("--spearman_chunk_rows", type=int, default=256)
    p.add_argument("--spearman_chunk_cols", type=int, default=2048)

    return p.parse_args()


def main():
    args = parse_args()

    # Validate combos
    if args.sim == "cosine" and args.edge_mode != "topk":
        raise SystemExit("cosine supports only edge_mode=topk (use spearman for threshold).")

    cfg = RunConfig(
        sim=args.sim,
        edge_mode=args.edge_mode,
        k_nearest=args.k_nearest,
        edge_threshold=args.edge_threshold,
        walks_per_gene=args.walks,
        walk_length=WALK_LENGTH,
        vector_dim=args.vector_dim,
        epochs=args.epochs,
        window=args.window,
        min_count=args.min_count,
        seed=args.seed,
        combine_edges=args.combine_edges,
        materialize=args.materialize,
        cos_batch=args.cos_batch,
        spearman_chunk_rows=args.spearman_chunk_rows,
        spearman_chunk_cols=args.spearman_chunk_cols,
    )

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    edge_value = args.k_nearest if args.edge_mode == "topk" else args.edge_threshold
    edge_label = "number_of_edges" if args.edge_mode == "topk" else "edge_threshold"
    print("=== RUN CONFIGURATION ===", flush=True)
    print(f"similarity_metric: {args.sim}", flush=True)
    print(f"{edge_label}: {edge_value}", flush=True)
    print(f"number_of_walks: {args.walks}", flush=True)
    print(f"walk_length: {WALK_LENGTH}", flush=True)
    print(f"vector_dimension: {args.vector_dim}", flush=True)
    print(f"output_directory: {out_root}", flush=True)
    print("=========================", flush=True)

    group_dirs = sorted([p for p in in_root.iterdir() if p.is_dir()])
    if args.only_group:
        group_dirs = [p for p in group_dirs if p.name == args.only_group]

    print(f"Groups to run: {len(group_dirs)}", flush=True)
    print(f"Config tag: {cfg.tag()}", flush=True)

    for group_dir in group_dirs:
        run_one_group(
            group_dir,
            out_root,
            cfg,
            save_edges=args.save_edges,
            skip_w2v=args.skip_w2v,
        )

    print("DONE.", flush=True)


if __name__ == "__main__":
    main()