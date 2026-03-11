#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import mmread
from gensim.models import Word2Vec

RESULTS_ROOT = Path("results/results_by_patient_celltype_100_walks")
EXPORTS_ROOT = Path("exports_by_patient_celltype")

EXPECTED = {
    "sim": "cosine",
    "edge_mode": "topk",
    "k_nearest": 50,
    "walks_per_gene": 100,
    "walk_length": 6,
    "vector_dim": 64,
    "epochs": 20,
    "window": 4,
    "min_count": 5,
    "seed": 42,
}

def safe_load_genes(path: Path):
    s = pd.read_csv(path, header=None).iloc[:, 0]
    s = s.dropna().astype(str).str.strip()
    return s[s != ""].tolist()

def aligned_gene_count(group_dir: Path):
    mat = mmread(str(group_dir / "expr.mtx")).tocsr()
    genes = safe_load_genes(group_dir / "genes.csv")

    if mat.shape[0] != len(genes) and mat.shape[1] == len(genes):
        mat = mat.T.tocsr()

    if mat.shape[0] != len(genes):
        new_n = min(mat.shape[0], len(genes))
        genes = genes[:new_n]
        mat = mat[:new_n, :]
    return mat.shape[0]

all_ok = True
rows = []

group_dirs = sorted([p for p in RESULTS_ROOT.iterdir() if p.is_dir()])

for group_dir in group_dirs:
    tag_dirs = sorted([p for p in group_dir.iterdir() if p.is_dir()])
    if len(tag_dirs) != 1:
        print(f"[WARN] {group_dir.name}: expected 1 tag dir, found {len(tag_dirs)}")
    for tag_dir in tag_dirs:
        cfg_path = tag_dir / "run_config.json"
        model_path = tag_dir / "gene_embeddings.model"
        edges_path = tag_dir / "edges.tsv"

        status = {
            "group": group_dir.name,
            "tag_dir": tag_dir.name,
            "config_ok": False,
            "model_ok": False,
            "edges_ok": None,
            "expected_genes": None,
            "model_vocab": None,
            "vector_dim": None,
            "edge_rows": None,
            "unique_src": None,
            "min_w": None,
            "mean_w": None,
            "max_w": None,
        }

        expected_n = aligned_gene_count(EXPORTS_ROOT / group_dir.name)
        status["expected_genes"] = expected_n

        # config
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            config_ok = True
            for k, v in EXPECTED.items():
                if cfg.get(k) != v:
                    print(f"[BAD CONFIG] {group_dir.name}: {k}={cfg.get(k)} expected {v}")
                    config_ok = False
            status["config_ok"] = config_ok
            all_ok &= config_ok
        else:
            print(f"[MISSING] {group_dir.name}: run_config.json")
            all_ok = False

        # model
        if model_path.exists():
            model = Word2Vec.load(str(model_path))
            vocab_size = len(model.wv)
            dim = model.vector_size
            status["model_vocab"] = vocab_size
            status["vector_dim"] = dim

            model_ok = (dim == EXPECTED["vector_dim"] and vocab_size == expected_n)
            if not model_ok:
                print(f"[BAD MODEL] {group_dir.name}: vocab={vocab_size} expected_genes={expected_n}, dim={dim}")
            status["model_ok"] = model_ok
            all_ok &= model_ok
        else:
            print(f"[MISSING] {group_dir.name}: gene_embeddings.model")
            all_ok = False

        # edges.tsv
        if edges_path.exists():
            df = pd.read_csv(edges_path, sep="\t")
            status["edge_rows"] = len(df)
            status["unique_src"] = df["src"].nunique()
            status["min_w"] = float(df["weight"].min())
            status["mean_w"] = float(df["weight"].mean())
            status["max_w"] = float(df["weight"].max())

            counts = df.groupby("src").size()
            edges_ok = True

            if df["src"].nunique() != expected_n:
                print(f"[BAD EDGES] {group_dir.name}: unique src={df['src'].nunique()} expected {expected_n}")
                edges_ok = False

            if not (counts == EXPECTED["k_nearest"]).all():
                bad = counts[counts != EXPECTED["k_nearest"]]
                print(f"[BAD EDGES] {group_dir.name}: {len(bad)} genes do not have exactly 50 outgoing edges")
                edges_ok = False

            if (df["src"] == df["dst"]).any():
                print(f"[BAD EDGES] {group_dir.name}: self-edges found")
                edges_ok = False

            if not np.isfinite(df["weight"]).all():
                print(f"[BAD EDGES] {group_dir.name}: non-finite weights found")
                edges_ok = False

            if (df["weight"] < -1).any() or (df["weight"] > 1).any():
                print(f"[BAD EDGES] {group_dir.name}: weights outside [-1,1]")
                edges_ok = False

            status["edges_ok"] = edges_ok
            all_ok &= edges_ok
        else:
            print(f"[WARN] {group_dir.name}: no edges.tsv (cannot validate edge counts / cosine distribution)")
            status["edges_ok"] = None

        rows.append(status)

summary = pd.DataFrame(rows)
summary.to_csv("walk100_validation_summary.csv", index=False)

print("\n====================")
print("Validation finished")
print("====================")
print(f"Groups checked: {len(summary)}")
print(f"All checks passed: {all_ok}")

if "edges_ok" in summary.columns:
    print("\nEdge weight summary across groups:")
    print(summary[["group", "min_w", "mean_w", "max_w"]].dropna().head(10).to_string(index=False))

print("\nSaved: walk100_validation_summary.csv")
