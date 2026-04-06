#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
import argparse
import numpy as np
import pandas as pd
from scipy.io import mmread
import igraph as ig

import train_model_general as tmg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--in_root", default="exports_by_patient_celltype")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--walks", type=int, default=100)
    ap.add_argument("--cos_batch", type=int, default=1024)
    args = ap.parse_args()

    group_dir = Path(args.in_root) / args.group
    mat = mmread(str(group_dir / "expr.mtx")).tocsr()
    genes = tmg._safe_load_genes(group_dir / "genes.csv")
    mat, genes = tmg._align_expr_and_genes(mat, genes)

    print(f"group={args.group}")
    print(f"aligned genes={len(genes)} cells={mat.shape[1]}")

    edges = tmg.compute_edges(
        mat=mat,
        genes=genes,
        sim="cosine",
        edge_mode="topk",
        k=args.k,
        threshold=0.0,
        cos_batch=args.cos_batch,
        spearman_chunk_rows=256,
        spearman_chunk_cols=2048,
    )

    df = pd.DataFrame(edges, columns=["src", "dst", "weight"])

    print("\nEDGE CHECKS")
    print("unique src:", df["src"].nunique())
    print("total edge rows:", len(df))
    print("expected rows:", len(genes) * args.k)
    print("self edges:", int((df["src"] == df["dst"]).sum()))
    print("weight min/mean/max:", df["weight"].min(), df["weight"].mean(), df["weight"].max())

    src_counts = df.groupby("src").size()
    bad_src = (src_counts != args.k).sum()
    print("genes with != 50 outgoing edges:", int(bad_src))

    gene_to_idx = {g: i for i, g in enumerate(genes)}
    edge_tuples = [(gene_to_idx[s], gene_to_idx[t]) for s, t, _ in edges]
    edge_weights = [float(w) for _, _, w in edges]

    g = ig.Graph(n=len(genes), edges=edge_tuples, directed=False)
    g.vs["name"] = genes
    g.es["weight"] = edge_weights
    g.simplify(combine_edges={"weight": "mean"})

    print("\nGRAPH CHECKS")
    deg = np.array(g.degree())
    print("nodes:", g.vcount(), "edges:", g.ecount())
    print("degree min/mean/max:", deg.min(), deg.mean(), deg.max())

    nb_cache, w_cache = tmg.build_weight_cache(g)

    print("\nWALK CHECKS")
    total = 0
    len_counter = Counter()

    for v_idx in range(g.vcount()):
        for _ in range(args.walks):
            sent = tmg.bidirectional_walk_cached(g, v_idx, tmg.WALK_LENGTH, nb_cache, w_cache)
            if len(sent) > 1:
                total += 1
                len_counter[len(sent)] += 1

    print("walks requested:", g.vcount() * args.walks)
    print("walks yielded:", total)
    print("sentence length distribution:", dict(sorted(len_counter.items())))
    print("expected full length:", 2 * tmg.WALK_LENGTH + 1)

if __name__ == "__main__":
    main()