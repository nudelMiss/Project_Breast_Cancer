#!/usr/bin/env python3
"""
Negative control for the propr screening win.
For each model, draw N random gene-sets of the SAME sizes as the real bio modules
(S_phase 43, G2M 54, IFN_a 98, IFN_g 198) from the model vocab, score each with the
SAME embedding-cosine AUC protocol, and compare the random-set AUC distribution to the
real-module AUC. Real signal => random ~0.5, real modules high. Artifact => random inflated too.
"""
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts/AUC")
from bio_auc_posthoc import embedding_cosine, set_auc, load_sets_tsv

ROOT = Path("/groups/ofircohen-group/users/michalnu_yuvat/project_breast")
MODELS = ROOT / "results/bassez2021/stageA/models"
BENCH = ROOT / "resources/genesets/bio_modules_benchmark.tsv"
SIZES = {"S_phase": 43, "G2M": 54, "IFN_alpha": 98, "IFN_gamma": 198}


def random_auc_dist(C, genes, size, n_rep, base_seed):
    g2i = {g: i for i, g in enumerate(genes)}
    rng = np.random.default_rng(base_seed)
    vocab = np.array(genes)
    aucs = []
    for r in range(n_rep):
        rs = set(rng.choice(vocab, size=min(size, len(vocab)), replace=False))
        a, _, _ = set_auc(C, g2i, rs, seed=1000 + r)
        if not np.isnan(a):
            aucs.append(a)
    return np.array(aucs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--n_rep", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "results/bassez2021/stageA/bio_auc/propr_negcontrol.csv"))
    args = ap.parse_args()
    real_sets = load_sets_tsv(BENCH)
    rows = []
    for grp in args.groups.split(","):
        for cfg in args.configs.split(","):
            mp = MODELS / grp / cfg / "gene_embeddings.model"
            if not mp.exists():
                print(f"[MISS] {grp}/{cfg}"); continue
            C, genes = embedding_cosine(mp)
            g2i = {g: i for i, g in enumerate(genes)}
            for mod, size in SIZES.items():
                real_a, _, _ = set_auc(C, g2i, real_sets[mod], seed=42)
                rand = random_auc_dist(C, genes, size, args.n_rep, base_seed=hash((grp, cfg, mod)) % (2**31))
                rows.append(dict(group=grp, config=cfg, module=mod, size=size,
                                 real_auc=round(real_a, 3),
                                 rand_mean=round(float(rand.mean()), 3),
                                 rand_p95=round(float(np.quantile(rand, 0.95)), 3),
                                 rand_max=round(float(rand.max()), 3),
                                 gap=round(real_a - float(rand.mean()), 3)))
            print(f"[OK] {grp}/{cfg}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print("\n=== mean over groups: real vs random AUC by config x module ===")
    piv = df.groupby(["config", "module"]).agg(
        real=("real_auc", "mean"), rand=("rand_mean", "mean"),
        rand_p95=("rand_p95", "mean"), gap=("gap", "mean")).round(3)
    print(piv.to_string())
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
