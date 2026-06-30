#!/usr/bin/env python3
"""
Joint-embeddings trainer for Bassez2021.

Strategy:
  1. For each (group in dataset_summary) build/reuse the per-patient graph
     using train_model_new.py's --skip_w2v with the shared --graph_cache_dir.
     The cached edges.tsv files are produced as a side effect.
  2. Build the graph object in memory from each cached edges.tsv, run walks,
     concatenate walks from ALL groups, and train ONE shared Word2Vec model.

Output goes to row['model_dir']/gene_embeddings.model (canonical path).

Idempotent: skips if the joint model exists.
"""
from __future__ import annotations
import argparse, csv, json, os, random, subprocess, sys
from pathlib import Path
import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from train_model_new import clean_token

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import igraph as ig

# Make the train_model_new module importable so we reuse its helpers.
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import train_model_new as tm  # type: ignore


def read_row(manifest_path: Path, row_index: int) -> dict:
    rows = list(csv.DictReader(open(manifest_path), delimiter="\t"))
    if not (0 <= row_index < len(rows)):
        sys.exit(f"FATAL: row {row_index} out of range")
    return rows[row_index]


def ensure_graph_for_group(group_row: dict, manifest_row: dict, in_root: Path):
    """Run train_model_new.py with --skip_w2v if the edges.tsv cache is missing.

    group_row keys: group (dir name like 'patient=X__celltype=Y'), patient, celltype, ...
    Cache is keyed by the canonical short tag '<patient>_<celltype>'.
    """
    method = manifest_row["graph_method"]
    sim = manifest_row["similarity"]
    cache_dir = Path(manifest_row["graph_cache_dir"])
    canonical_tag = f"{clean_token(group_row['patient'])}_{clean_token(group_row['celltype'])}"
    edges_path = cache_dir / canonical_tag / f"{method}_{sim}" / "edges.tsv"
    if edges_path.exists():
        return edges_path
    # Trigger build via the main script with --skip_w2v.
    scratch = Path(manifest_row["model_dir"] + f"__joint_pre__{canonical_tag}")
    cmd = [
        sys.executable, "-u", "scripts/train_model_new.py",
        "--in_root", manifest_row["in_root"],
        "--only_group", group_row["group"],
        "--out_root", str(scratch),
        "--graph_method", method,
        "--sim", sim,
        "--walk_strategy", manifest_row["walk_strategy"],
        "--walks", "1",
        "--walk_length", manifest_row["walk_length"],
        "--k_nearest", "5",
        "--variance_keep_frac", "0.75",
        "--graph_cache_dir", str(cache_dir),
        "--seed", "42",
        "--skip_w2v",
    ]
    print(f"[GRAPH] building for {canonical_tag}", flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f"graph build failed for {canonical_tag} (rc={rc})")
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    if not edges_path.exists():
        raise RuntimeError(f"graph cache missing after build: {edges_path}")
    return edges_path


def build_igraph_from_edges(edges_df: pd.DataFrame, combine_edges: str = "mean") -> ig.Graph:
    genes = sorted(set(edges_df["src"]).union(edges_df["dst"]))
    return tm.build_graph_from_edges(genes, edges_df, combine_edges)


def walk_corpus_for_graph(graph: ig.Graph, strategy: str, walks: int,
                          walk_length: int, seed: int):
    nb_cache, w_cache = tm.build_weight_cache(graph)
    if strategy == "star":
        return tm.StarWalkCorpus(graph, walk_length, walks, seed, nb_cache, w_cache)
    elif strategy == "bidirectional":
        return tm.BidirectionalWalkCorpus(graph, walk_length, walks, seed, nb_cache, w_cache)
    else:
        raise ValueError(f"bad strategy: {strategy}")


class ChainedCorpus:
    """Chain multiple per-graph corpora into one iterator for Word2Vec."""
    def __init__(self, corpora):
        self.corpora = corpora
    def __iter__(self):
        for c in self.corpora:
            for sentence in c:
                yield sentence


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--row_index", type=int, required=True)
    p.add_argument("--summary_csv",
                   default="results/bassez2021/summaries/dataset_summary.csv")
    args = p.parse_args()

    row = read_row(Path(args.manifest), args.row_index)
    if row["aggregation_strategy"] != "joint_embeddings":
        sys.exit("FATAL: row is not aggregation_strategy=joint_embeddings")

    model_path = Path(row["model_path"])
    if model_path.exists():
        print(f"[SKIP] joint model exists: {model_path}", flush=True)
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Iterate all per-patient groups.
    groups = list(csv.DictReader(open(args.summary_csv)))
    print(f"[JOINT] {len(groups)} groups to walk", flush=True)

    # Ensure all graph caches exist.
    edge_paths = []
    for g in groups:
        ep = ensure_graph_for_group(g, row, Path(row["in_root"]))
        edge_paths.append(ep)

    # Build corpora per graph.
    walks = int(row["walks"]); walk_length = int(row["walk_length"])
    strategy = row["walk_strategy"]; seed = 42
    print(f"[JOINT] strategy={strategy} walks={walks} walk_length={walk_length}", flush=True)

    corpora = []
    for ep in edge_paths:
        ed = pd.read_csv(ep, sep="\t")
        g = build_igraph_from_edges(ed)
        corpora.append(walk_corpus_for_graph(g, strategy, walks, walk_length, seed))

    chained = ChainedCorpus(corpora)
    print("[W2V] training joint model", flush=True)
    w2v = Word2Vec(sentences=chained, vector_size=64, window=5, min_count=1,
                   sg=1, workers=4, epochs=20, seed=seed)
    w2v.save(str(model_path))
    print(f"[SAVE] {model_path}", flush=True)

    # Save config sidecar
    with open(Path(row["model_dir"]) / "run_config.json", "w") as f:
        json.dump({k: row[k] for k in row}, f, indent=2)


if __name__ == "__main__":
    main()
