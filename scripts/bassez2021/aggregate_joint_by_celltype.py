#!/usr/bin/env python3
"""
Joint-by-cell-type embeddings (Stage B).
Pools all per-patient groups OF ONE CELL TYPE into a single Word2Vec, by chaining
their cached per-patient edges.tsv walk-corpora (same edges as the per-patient run).

Reuses results/bassez2021/stageA/models/<group>/<sim>_<strat>_w{W}_k{K}_var75_hvg2000/edges.tsv.
Output: results/bassez2021/models_joint_by_celltype/celltype=<CT>/<sim>_<strat>_w{W}_k{K}_var75_hvg2000_jointct/
Does NOT touch the existing global JOINT under models/ALL/.
"""
import argparse, json, sys
from pathlib import Path
import pandas as pd
from gensim.models import Word2Vec

REPO = Path("/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast")
sys.path.insert(0, str(REPO / "scripts"))
import train_model_new as tm

PPMODELS = REPO / "results/bassez2021/stageA/models"
OUTROOT = REPO / "results/bassez2021/models_joint_by_celltype"
W, K, WL = 10, 50, 3


class ChainedCorpus:
    def __init__(self, corpora): self.corpora = corpora
    def __iter__(self):
        for c in self.corpora:
            for s in c: yield s


def corpus_for_edges(edges_path, strat, seed=42):
    ed = pd.read_csv(edges_path, sep="\t")
    genes = sorted(set(ed["src"]).union(ed["dst"]))
    g = tm.build_graph_from_edges(genes, ed, "mean")
    nb, wc = tm.build_weight_cache(g)
    if strat == "star":
        return tm.StarWalkCorpus(g, WL, W, seed, nb, wc)
    return tm.BidirectionalWalkCorpus(g, WL, W, seed, nb, wc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celltype", required=True)
    ap.add_argument("--sim", required=True, choices=["propr", "cosine", "spearman"])
    ap.add_argument("--strat", required=True, choices=["star", "bidirectional"])
    args = ap.parse_args()

    cfg_tag = f"{args.sim}_{args.strat}_w{W}_k{K}_var75_hvg2000"
    out_tag = cfg_tag + "_jointct"
    out_dir = OUTROOT / f"celltype={args.celltype}" / out_tag
    model_path = out_dir / "gene_embeddings.model"
    if model_path.exists():
        print(f"[SKIP] {model_path}", flush=True); return

    # gather per-patient edges for this cell type + config
    grp_dirs = sorted(PPMODELS.glob(f"*_{args.celltype}"))
    edge_paths = []
    for gd in grp_dirs:
        ep = gd / cfg_tag / "edges.tsv"
        if ep.exists():
            edge_paths.append(ep)
    if not edge_paths:
        sys.exit(f"FATAL: no per-patient edges for celltype={args.celltype} cfg={cfg_tag}")
    print(f"[JOINT-CT] {args.celltype} {cfg_tag}: pooling {len(edge_paths)} patient groups", flush=True)

    corpora = [corpus_for_edges(ep, args.strat) for ep in edge_paths]
    print("[W2V] training joint-by-celltype model", flush=True)
    w2v = Word2Vec(sentences=ChainedCorpus(corpora), vector_size=64, window=5,
                   min_count=1, sg=1, workers=4, epochs=20, seed=42)
    out_dir.mkdir(parents=True, exist_ok=True)
    w2v.save(str(model_path))
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(dict(celltype=args.celltype, sim=args.sim, strat=args.strat,
                       walks=W, k=K, walk_length=WL, hvg_cap=2000,
                       aggregation="joint_by_celltype", n_patients_pooled=len(edge_paths)), f, indent=2)
    print(f"[SAVE] {model_path}  (pooled {len(edge_paths)} patients)", flush=True)


if __name__ == "__main__":
    main()
