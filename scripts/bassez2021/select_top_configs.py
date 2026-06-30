#!/usr/bin/env python3
"""
Read aggregated pilot results, rank (imputation, similarity, walk_strategy)
configs by a composite score, emit chosen list for build_manifest.py.
Composite = mean(mean_auc) + mean(mcc_best), averaged across pilot groups.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot_csv", required=True)
    p.add_argument("--keep", type=int, default=6)
    p.add_argument("--out_ranking", required=True)
    p.add_argument("--out_chosen", required=True)
    return p.parse_args()

def main():
    a = parse_args()
    df = pd.read_csv(a.pilot_csv)
    needed = {"imputation","similarity","walk_strategy","mean_auc","mcc_best"}
    miss = needed - set(df.columns)
    if miss: raise SystemExit(f"pilot_csv missing columns: {miss}")

    df = df.dropna(subset=["mean_auc","mcc_best"])
    if df.empty: raise SystemExit("No rows with both AUC and MCC present in pilot_csv")

    g = (df.groupby(["imputation","similarity","walk_strategy"])
           .agg(n=("mean_auc","size"),
                mean_auc=("mean_auc","mean"),
                median_auc=("mean_auc","median"),
                mean_mcc_best=("mcc_best","mean"),
                median_mcc_best=("mcc_best","median"),
                mean_mcc_at_fixed=("mcc_at_fixed","mean"))
           .reset_index())
    g["composite"] = g["mean_auc"] + g["mean_mcc_best"]
    g = g.sort_values("composite", ascending=False).round(4)

    Path(a.out_ranking).parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(a.out_ranking, index=False)
    print(f"[WRITE] {a.out_ranking}")
    print(g.to_string(index=False))

    top = g.head(a.keep)
    triples = [f"{r.imputation}:{r.similarity}:{r.walk_strategy}" for r in top.itertuples()]
    chosen_str = ",".join(triples)
    Path(a.out_chosen).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_chosen).write_text(chosen_str + "\n")
    print(f"\n[WRITE] {a.out_chosen}")
    print(f"chosen (top {a.keep}): {chosen_str}")

if __name__ == "__main__":
    main()
