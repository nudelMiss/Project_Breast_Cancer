#!/usr/bin/env python3
"""Joint-by-cell-type embeddings (generalized, path-parameterized).
Pools all per-patient groups of ONE cell type into one Word2Vec by chaining their
cached per-patient edges.tsv walk-corpora. Mirrors bassez aggregate_joint_by_celltype.py
but takes --ppmodels_root / --outroot so it works on any dataset (v2)."""
import argparse, json, sys
from pathlib import Path
import pandas as pd
from gensim.models import Word2Vec
REPO = Path("/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast")
sys.path.insert(0, str(REPO / "scripts"))
import train_model_new as tm
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
    ap.add_argument("--ppmodels_root", required=True)
    ap.add_argument("--outroot", required=True)
    ap.add_argument("--celltype", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--strat", required=True, choices=["star", "bidirectional"])
    a = ap.parse_args()
    PP = Path(a.ppmodels_root); OUT = Path(a.outroot)
    cfg_tag = f"{a.sim}_{a.strat}_w{W}_k{K}_var75_hvg2000"
    out_dir = OUT / f"celltype={a.celltype}" / (cfg_tag + "_jointct")
    if (out_dir / "gene_embeddings.model").exists():
        print(f"[SKIP] {out_dir}"); return
    grp_dirs = sorted(PP.glob(f"*_{a.celltype}"))
    eps = [gd/cfg_tag/"edges.tsv" for gd in grp_dirs if (gd/cfg_tag/"edges.tsv").exists()]
    if not eps: sys.exit(f"FATAL: no per-patient edges for celltype={a.celltype} cfg={cfg_tag} under {PP}")
    print(f"[JOINT-CT] {a.celltype} {cfg_tag}: pooling {len(eps)} groups", flush=True)
    corpora = [corpus_for_edges(ep, a.strat) for ep in eps]
    w2v = Word2Vec(sentences=ChainedCorpus(corpora), vector_size=64, window=5,
                   min_count=1, sg=1, workers=4, epochs=20, seed=42)
    out_dir.mkdir(parents=True, exist_ok=True)
    w2v.save(str(out_dir / "gene_embeddings.model"))
    json.dump(dict(celltype=a.celltype, sim=a.sim, strat=a.strat, walks=W, k=K, walk_length=WL,
                   hvg_cap=2000, aggregation="joint_by_celltype", n_patients_pooled=len(eps)),
              open(out_dir/"run_config.json","w"), indent=2)
    print(f"[SAVE] {out_dir} (pooled {len(eps)})", flush=True)

if __name__ == "__main__":
    main()
