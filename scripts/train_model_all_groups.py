import argparse
import json
import random
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.io import mmread
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# =======================================
# Global Configuration
# =======================================

K_NEAREST = 50
WALK_LENGTH = 5
VECTOR_DIM = 64
EPOCHS = 20
SUBSET_SIZE = None  # set to an int for quick tests, else None

# =======================================
# CLI Args
# =======================================

parser = argparse.ArgumentParser()
parser.add_argument("--walks", type=int, default=10, help="Number of random walks per gene")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--in_root", type=str, default="exports_by_patient_celltype",
                    help="Root folder containing group subfolders (each has expr.mtx, genes.csv)")
parser.add_argument("--out_root", type=str, default="results_by_patient_celltype",
                    help="Root folder for saving outputs per group")
parser.add_argument("--skip_existing", action="store_true",
                    help="Skip groups that already have a saved model in output folder")
args = parser.parse_args()

WALKS_PER_GENE = args.walks
SEED = args.seed

# =======================================
# 2. Compute Cosine Similarity (Top-k ONLY, no filtering)
# =======================================

def compute_topk_cosine(expression_matrix, gene_names, k):
    """
    Computes the top-k nearest neighbors for each gene based on cosine similarity.
    Does NOT filter negative similarities.
    Returns a list of weighted edges (source, target, weight).
    """
    topk_edges = []
    num_genes = expression_matrix.shape[0]

    for i in tqdm(range(num_genes), desc="Cosine similarity", unit="gene", leave=False):
        gene_vec = expression_matrix[i].toarray()
        sims = cosine_similarity(gene_vec, expression_matrix).flatten()
        sims[i] = -np.inf  # exclude self

        # pick top-k by similarity (highest values)
        if num_genes - 1 > k:
            topk_idx = np.argpartition(-sims, k - 1)[:k]
        else:
            topk_idx = np.where(np.isfinite(sims))[0]

        gi = gene_names[i]
        for j in topk_idx:
            if not np.isfinite(sims[j]):
                continue
            topk_edges.append((gi, gene_names[j], float(sims[j])))

    return topk_edges

# =======================================
# 4. Random Walks ("Sentences") - WEIGHTED (no max(0,w) clamp)
# =======================================

def weighted_next_vertex(graph, current_idx, neighbors):
    """
    Choose a neighbor with probability proportional to edge weight.
    Assumes all weights are non-negative.
    """
    if not neighbors:
        return None

    weights = []
    for nb in neighbors:
        eid = graph.get_eid(current_idx, nb, directed=False, error=False)
        if eid == -1:
            w = 0.0
        else:
            w = float(graph.es[eid]["weight"])
        weights.append(w)

    total = sum(weights)
    if total <= 0.0:
        # fall back to uniform
        return graph.vs[random.choice(neighbors)]

    chosen_nb = random.choices(neighbors, weights=weights, k=1)[0]
    return graph.vs[chosen_nb]


def bidirectional_walk(graph, central_gene, walk_length):
    """
    Performs a bidirectional weighted random walk starting from 'central_gene'.
    Returns a list of genes representing a 'sentence'.
    """
    start_v = graph.vs.find(name=central_gene)

    left = []
    current = start_v
    for _ in range(walk_length):
        neighbors = graph.neighbors(current.index)
        if not neighbors:
            break
        next_v = weighted_next_vertex(graph, current.index, neighbors)
        if next_v is None:
            break
        left.append(next_v["name"])
        current = next_v

    right = []
    current = start_v
    for _ in range(walk_length):
        neighbors = graph.neighbors(current.index)
        if not neighbors:
            break
        next_v = weighted_next_vertex(graph, current.index, neighbors)
        if next_v is None:
            break
        right.append(next_v["name"])
        current = next_v

    return left[::-1] + [central_gene] + right

# =======================================
# Run one group
# =======================================

def run_one_group(group_dir: Path, out_dir: Path):
    mtx_path = group_dir / "expr.mtx"
    genes_path = group_dir / "genes.csv"
    meta_path = group_dir / "meta.json"

    if not mtx_path.exists() or not genes_path.exists():
        print(f"[SKIP] Missing expr.mtx or genes.csv in {group_dir}")
        return

    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}

    print(f"\n=== GROUP: {group_dir.name} ===")
    print("Loading data...")

    mat = mmread(str(mtx_path)).tocsr()
    genes = pd.read_csv(genes_path, header=None).iloc[:, 0].astype(str).tolist()

    if SUBSET_SIZE:
        print(f"Subsetting to first {SUBSET_SIZE} genes...")
        mat = mat[:SUBSET_SIZE]
        genes = genes[:SUBSET_SIZE]

    # Cosine edges
    print(f"Computing top-{K_NEAREST} cosine edges for {mat.shape[0]} genes...")
    edges = compute_topk_cosine(mat, genes, K_NEAREST)
    print(f"Computed {len(edges)} edges.")

    # Build graph
    print("Building igraph object...")
    gene_graph = ig.Graph()
    gene_graph.add_vertices(genes)

    edge_tuples = [(s, t) for (s, t, w) in edges]
    edge_weights = [w for (s, t, w) in edges]

    gene_graph.add_edges(edge_tuples)
    gene_graph.es["weight"] = edge_weights
    gene_graph.simplify(combine_edges={"weight": "mean"})

    print(gene_graph.summary())

    if len(gene_graph.es) == 0:
        print("[SKIP] Graph has 0 edges.")
        return

    w_min = float(min(gene_graph.es["weight"]))
    w_max = float(max(gene_graph.es["weight"]))
    print("Min weight:", w_min)
    print("Max weight:", w_max)

    # IMPORTANT: since you asked to remove clamping, enforce non-negative weights
    if w_min < 0:
        raise ValueError(
            f"Found negative edge weights (min={w_min}). "
            f"With clamping removed, weighted sampling requires non-negative weights."
        )

    # Walks
    print("Generating random walk sentences...")
    random.seed(SEED)
    np.random.seed(SEED)

    all_sentences = []
    for vertex in tqdm(gene_graph.vs, desc="Random walks", unit="gene"):
        gene_name = vertex["name"]
        for _ in range(WALKS_PER_GENE):
            sentence = bidirectional_walk(gene_graph, gene_name, WALK_LENGTH)
            if len(sentence) > 1:
                all_sentences.append(sentence)

    print(f"Generated {len(all_sentences)} sentences.")

    # Word2Vec
    print("Training Word2Vec model...")
    model = Word2Vec(
        sentences=all_sentences,
        vector_size=VECTOR_DIM,
        window=4,
        min_count=5,
        sg=1,
        workers=4,
        epochs=EPOCHS
    )

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    model_filename = out_dir / f"gene_embeddings_w{WALKS_PER_GENE}_weighted.model"
    model.save(str(model_filename))

    summary = {
        "group": group_dir.name,
        "patient": meta.get("patient"),
        "celltype": meta.get("celltype"),
        "n_cells": meta.get("n_cells"),
        "n_genes": meta.get("n_genes"),
        "k_nearest": K_NEAREST,
        "walk_length": WALK_LENGTH,
        "walks_per_gene": WALKS_PER_GENE,
        "vector_dim": VECTOR_DIM,
        "epochs": EPOCHS,
        "n_edges": int(gene_graph.ecount()),
        "n_sentences": int(len(all_sentences)),
        "seed": SEED,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved model: {model_filename}")
    print(f"Saved summary: {out_dir / 'run_summary.json'}")

# =======================================
# Main loop over all groups
# =======================================

def main():
    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not in_root.exists():
        raise FileNotFoundError(f"in_root does not exist: {in_root}")

    group_dirs = sorted([p for p in in_root.iterdir() if p.is_dir()])
    print(f"Found {len(group_dirs)} groups under {in_root}")

    for group_dir in group_dirs:
        out_dir = out_root / group_dir.name

        if args.skip_existing:
            expected = out_dir / f"gene_embeddings_w{WALKS_PER_GENE}_weighted.model"
            if expected.exists():
                print(f"[SKIP] Exists: {expected}")
                continue

        try:
            run_one_group(group_dir, out_dir)
        except Exception as e:
            print(f"[ERROR] Group {group_dir.name} failed: {e}")

    print("\nDONE.")

if __name__ == "__main__":
    main()
