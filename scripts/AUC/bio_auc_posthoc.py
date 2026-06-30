#!/usr/bin/env python3
"""
Post-hoc bio-module AUC on cached W2V models (no retraining).
============================================================
Closes the documented gap: cached AUC was CORUM-only; bio-module AUC
existed only for the pilot. This scores ANY existing gene_embeddings.model
against the headline bio modules (S_phase, G2M, IFN_alpha, IFN_gamma) and,
optionally, CORUM -- using the project-standard eval protocol:

  positives = all within-set gene pairs
  negatives = max(200, n_pos) pairs of (in-set gene, out-of-set gene)
  score     = cosine similarity of W2V embeddings
  seed      = 42  (matches existing CORUM benchmark + IDS gate)

Outputs (NEVER overwrites a model dir):
  <out_root>/<group>/<config_tag>/bio_auc_per_module.csv
  <out_root>/<group>/<config_tag>/bio_auc_summary.csv

Usage:
  python bio_auc_posthoc.py --models_root results/bassez2021/models \
      --bench resources/genesets/bio_modules_benchmark.tsv \
      --out_root results/bassez2021/bio_auc \
      --groups BIOKEY_18_T_cell,BIOKEY_30_Malignant,BIOKEY_4_B_cell \
      --configs raw_cosine_bidirectional_w1_k5_wl3_perpat,raw_cosine_star_w100_k5_wl3_perpat \
      [--corum resources/corum_core_complexes.tsv]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec


def load_sets_tsv(path):
    """Read a benchmark TSV (complex_id, complex_name, gene) -> {name: set(GENES_UPPER)}."""
    df = pd.read_csv(path, sep="\t")
    sets = {}
    for cid, grp in df.groupby("complex_id"):
        name = str(grp["complex_name"].iloc[0]) if "complex_name" in df.columns else str(cid)
        sets[name] = set(grp["gene"].astype(str).str.strip().str.upper())
    return sets


def load_corum(path):
    df = pd.read_csv(path, sep="\t")
    return [(str(cid), set(g["gene"].astype(str).str.strip().str.upper()))
            for cid, g in df.groupby("complex_id")]


def embedding_cosine(model_path):
    m = Word2Vec.load(str(model_path))
    genes = list(m.wv.key_to_index.keys())
    V = np.array([m.wv[g] for g in genes], dtype=np.float32)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Vn = V / norms
    return Vn @ Vn.T, [g.upper() for g in genes]


def set_auc(C, g2i, gene_set, seed=42, min_sz=3, n_neg_min=200):
    allg = set(g2i.keys())
    shared = sorted(gene_set & allg)
    if len(shared) < min_sz:
        return np.nan, len(shared), 0
    idx = np.array([g2i[g] for g in shared])
    i_arr, j_arr = np.triu_indices(len(idx), k=1)
    pos = np.asarray(C[idx[i_arr], idx[j_arr]]).ravel()
    n_pos = len(pos)
    if n_pos == 0:
        return np.nan, len(shared), 0
    non = np.array([g2i[g] for g in sorted(allg - set(shared))])
    if non.size == 0:
        return np.nan, len(shared), n_pos
    n_neg = max(n_neg_min, n_pos)
    rng = np.random.default_rng(seed)
    a = rng.choice(idx, size=n_neg, replace=True)
    b = rng.choice(non, size=n_neg, replace=True)
    neg = np.asarray(C[a, b]).ravel()
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    scores = np.concatenate([pos, neg])
    if np.std(scores) < 1e-10:
        return 0.5, len(shared), n_pos
    return float(roc_auc_score(labels, scores)), len(shared), n_pos


def corum_mean_weighted(C, g2i, complexes, seed=42, min_sz=3):
    aucs, ws = [], []
    for _, cg in complexes:
        a, n_sh, n_pos = set_auc(C, g2i, cg, seed=seed, min_sz=min_sz)
        if np.isnan(a) or n_pos == 0:
            continue
        aucs.append(a); ws.append(n_pos)
    if not aucs:
        return np.nan, np.nan, 0
    aucs = np.array(aucs); ws = np.array(ws)
    return float(aucs.mean()), float((aucs * ws).sum() / ws.sum()), len(aucs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_root", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--groups", required=True, help="comma-separated group tags")
    ap.add_argument("--configs", required=True, help="comma-separated config tags")
    ap.add_argument("--corum", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sets = load_sets_tsv(args.bench)
    corum = load_corum(args.corum) if args.corum else None
    models_root = Path(args.models_root)
    out_root = Path(args.out_root)
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    print(f"Modules: {{k: len(v) for k,v in sets.items()}}".replace("{k: len(v) for k,v in sets.items()}",
          str({k: len(v) for k, v in sets.items()})), flush=True)
    summary_rows = []
    for grp in groups:
        for cfg in configs:
            mp = models_root / grp / cfg / "gene_embeddings.model"
            if not mp.exists():
                print(f"[MISS] {grp}/{cfg}", flush=True)
                continue
            C, genes = embedding_cosine(mp)
            g2i = {g: i for i, g in enumerate(genes)}
            per = []
            for name, gset in sets.items():
                auc, n_sh, n_pos = set_auc(C, g2i, gset, seed=args.seed)
                per.append(dict(module=name, auc=auc, n_genes_present=n_sh, n_pos_pairs=n_pos))
            outdir = out_root / grp / cfg
            outdir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(per).to_csv(outdir / "bio_auc_per_module.csv", index=False)
            valid = [r["auc"] for r in per if not np.isnan(r["auc"])]
            srow = dict(group=grp, config_tag=cfg, vocab=len(genes),
                        mean_bio_auc=float(np.mean(valid)) if valid else np.nan)
            for r in per:
                srow[f"auc_{r['module']}"] = r["auc"]
            if corum is not None:
                cm, cw, ncx = corum_mean_weighted(C, g2i, corum, seed=args.seed)
                srow["corum_mean_auc"] = cm
                srow["corum_weighted_auc"] = cw
                srow["corum_n_complexes"] = ncx
            pd.DataFrame([srow]).to_csv(outdir / "bio_auc_summary.csv", index=False)
            summary_rows.append(srow)
            msg = "  ".join(f"{r['module']}={r['auc']:.3f}" for r in per)
            print(f"[OK] {grp}/{cfg}  vocab={len(genes)}  {msg}", flush=True)

    if summary_rows:
        allout = out_root / "bio_auc_collected.csv"
        # append-or-create without clobbering prior collected rows for other configs
        new = pd.DataFrame(summary_rows)
        if allout.exists():
            old = pd.read_csv(allout)
            comb = pd.concat([old, new], ignore_index=True)
            comb = comb.drop_duplicates(subset=["group", "config_tag"], keep="last")
        else:
            comb = new
        comb.to_csv(allout, index=False)
        print(f"\nWrote {len(summary_rows)} summaries; collected -> {allout} ({len(comb)} rows)", flush=True)


if __name__ == "__main__":
    main()
