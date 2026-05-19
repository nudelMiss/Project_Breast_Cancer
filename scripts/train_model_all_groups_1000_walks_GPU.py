import argparse
import os
import random
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from scipy.io import mmread
from tqdm import tqdm

# GPU
import torch
import torch.nn.functional as F

# Word2Vec
from gensim.models import Word2Vec

# =======================================
# Global Configuration
# =======================================

K_NEAREST = 50
WALK_LENGTH = 7
VECTOR_DIM = 64
EPOCHS = 20

# =======================================
# CLI Args
# =======================================

parser = argparse.ArgumentParser()
parser.add_argument("--walks", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--in_root", type=str, default="exports_by_patient_celltype")
parser.add_argument("--out_root", type=str, default="results_by_patient_celltype")
parser.add_argument("--only_group", type=str, default=None)
parser.add_argument("--cos_batch", type=int, default=1024)

# ADDED: optional materialization to remove single-thread generator bottleneck
parser.add_argument(
    "--materialize",
    action="store_true",
    help="Materialize all walks into RAM before Word2Vec (much faster, uses more memory).",
)

args = parser.parse_args()

WALKS_PER_GENE = args.walks
SEED = args.seed
COS_BATCH = args.cos_batch

# ADDED: robust CPU detection across Slurm variants
def _get_slurm_cpus(default: int = 4) -> int:
    for k in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"):
        v = os.environ.get(k)
        if not v:
            continue
        # SLURM_JOB_CPUS_PER_NODE can look like "6(x1)"
        v = v.split("(")[0]
        try:
            return int(v)
        except ValueError:
            pass
    return default


SLURM_CPUS = _get_slurm_cpus(default=4)
W2V_WORKERS = max(1, SLURM_CPUS)


# =======================================
# GPU Cosine Similarity
# =======================================

def compute_topk_cosine_gpu(expression_matrix, gene_names, k, batch_size=1024):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")

    device = torch.device("cuda")

    # expression_matrix expected shape: (n_genes, n_cells)
    print("[GPU] Converting sparse -> dense float32 (CPU) ...", flush=True)
    dense = expression_matrix.astype(np.float32).toarray()

    print("[GPU] Moving dense matrix to GPU ...", flush=True)
    X = torch.from_numpy(dense).to(device)

    print("[GPU] X shape:", tuple(X.shape), flush=True)

    X = F.normalize(X, p=2, dim=1)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n = X.shape[0]
    edges = []

    print(f"[GPU] Computing cosine similarity in batches (n={n}, batch={batch_size}, k={k}) ...", flush=True)

    for start in tqdm(range(0, n, batch_size), desc="GPU cosine"):
        end = min(start + batch_size, n)

        sims = X[start:end] @ X.T

        # mask self-similarity on the block diagonal
        rows = torch.arange(end - start, device=device)
        cols = torch.arange(start, end, device=device)
        sims[rows, cols] = -float("inf")

        vals, idx = torch.topk(sims, k=k, dim=1)

        vals = vals.detach().cpu().numpy()
        idx = idx.detach().cpu().numpy()

        # build edges (same logic as before)
        for i in range(end - start):
            gi = gene_names[start + i]
            for j, w in zip(idx[i], vals[i]):
                if np.isfinite(w):
                    edges.append((gi, gene_names[int(j)], float(w)))

        del sims  # IMPORTANT: let GPU memory be reused; don't empty_cache()

    return edges


# =======================================
# Random Walk Optimization (cache neighbors+weights)
# =======================================

def build_weight_cache(graph: ig.Graph):
    """
    Precompute neighbors and their weights for each vertex once.
    This keeps the *same walk logic* but avoids expensive graph.get_eid()
    calls inside the inner loop.
    """
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
    """
    Same logic as your bidirectional_walk, but uses integer vertex indices and cached weights.
    Returns a list of gene names (strings), same as before.
    """
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


# =======================================
# Random Walk Corpus
# =======================================

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

        # iterate by index (faster) instead of vertex object
        for v_idx in range(self.graph.vcount()):
            for _ in range(self.walks_per_gene):
                sent = bidirectional_walk_cached(
                    self.graph,
                    v_idx,
                    self.walk_length,
                    self.nb_cache,
                    self.w_cache,
                )
                if len(sent) > 1:
                    yield sent


# =======================================
# Run One Group
# =======================================

def run_one_group(group_dir: Path, out_dir: Path):
    print(f"\n=== GROUP: {group_dir.name} ===", flush=True)

    mat = mmread(str(group_dir / "expr.mtx")).tocsr()

    # robust gene loading (strip + drop empty)
    genes_series = pd.read_csv(group_dir / "genes.csv", header=None).iloc[:, 0]
    genes_series = genes_series.dropna().astype(str).str.strip()
    genes = genes_series[genes_series != ""].tolist()

    print("expr shape:", mat.shape, "n_genes:", len(genes), flush=True)

    # Ensure rows = genes (so we compute gene-gene cosine, not cell-cell)
    if mat.shape[0] != len(genes) and mat.shape[1] == len(genes):
        print("Transposing expr matrix so rows are genes", flush=True)
        mat = mat.T.tocsr()

    # handle small mismatches safely
    if mat.shape[0] != len(genes):
        diff = len(genes) - mat.shape[0]
        print(
            f"[WARN] gene/matrix mismatch after transpose: mat_rows={mat.shape[0]} "
            f"n_genes={len(genes)} diff={diff}",
            flush=True,
        )

        # If mismatch is tiny (like your case: 1), align by truncation
        if abs(diff) <= 5:
            new_n = min(mat.shape[0], len(genes))
            print(f"[WARN] Aligning by truncation to n={new_n}", flush=True)
            genes = genes[:new_n]
            mat = mat[:new_n, :]
        else:
            raise ValueError(
                f"After possible transpose: mat.shape={mat.shape} but n_genes={len(genes)} (diff={diff})"
            )

    print("Computing cosine on GPU...", flush=True)
    edges = compute_topk_cosine_gpu(mat, genes, K_NEAREST, COS_BATCH)

    print("Building graph (faster: integer vertex ids)...", flush=True)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    edge_tuples = [(gene_to_idx[s], gene_to_idx[t]) for (s, t, _) in edges]
    edge_weights = [w for (_, _, w) in edges]

    gene_graph = ig.Graph(n=len(genes), edges=edge_tuples, directed=False)
    gene_graph.vs["name"] = genes
    gene_graph.es["weight"] = edge_weights
    gene_graph.simplify(combine_edges={"weight": "mean"})

    print("Caching neighbors+weights for fast walks...", flush=True)
    nb_cache, w_cache = build_weight_cache(gene_graph)

    print("Preparing corpus...", flush=True)
    corpus = WalkCorpus(gene_graph, WALK_LENGTH, WALKS_PER_GENE, SEED, nb_cache, w_cache)

    print(f"SLURM_CPUS={SLURM_CPUS} W2V_WORKERS={W2V_WORKERS} materialize={args.materialize}", flush=True)

    # Word2Vec
    model = Word2Vec(
        vector_size=VECTOR_DIM,
        window=4,
        min_count=5,
        sg=1,
        workers=W2V_WORKERS,
    )

    # ADDED: optional materialization to avoid single-thread generator bottleneck
    if args.materialize:
        print("Materializing walks into RAM (this can take time/memory)...", flush=True)
        walks = list(corpus)
        print(f"Materialized {len(walks)} walks", flush=True)
        corpus_for_w2v = walks
        total_examples = len(walks)
    else:
        corpus_for_w2v = corpus
        total_examples = corpus.total_examples

    print("Building vocab...", flush=True)
    model.build_vocab(corpus_for_w2v)
    print("Vocab built. Starting train...", flush=True)
    model.train(
        corpus_for_w2v,
        total_examples=total_examples,
        epochs=EPOCHS,
    )
    print("Train done.", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir / f"gene_embeddings_w{WALKS_PER_GENE}_weighted.model"))

    print("Saved model.", flush=True)


# =======================================
# Main
# =======================================

def main():
    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    group_dirs = sorted([p for p in in_root.iterdir() if p.is_dir()])

    if args.only_group:
        group_dirs = [p for p in group_dirs if p.name == args.only_group]

    for group_dir in group_dirs:
        run_one_group(group_dir, out_root / group_dir.name)

    print("DONE.", flush=True)


if __name__ == "__main__":
    main()