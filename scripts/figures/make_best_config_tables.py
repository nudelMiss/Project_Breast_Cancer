#!/usr/bin/env python3
"""Make best-config summary tables from an aggregated results CSV."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--prefix", default="results")
    p.add_argument("--top_n", type=int, default=10)
    return p.parse_args()

def main():
    a = parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.in_csv)

    keys = ["imputation","similarity","walk_strategy"]
    if "walks" in df.columns and df["walks"].nunique() > 1:
        keys = keys + ["walks"]
    if "aggregation_strategy" in df.columns and df["aggregation_strategy"].nunique() > 1:
        keys = keys + ["aggregation_strategy"]

    base = (df.dropna(subset=["mean_auc","mcc_best"])
              .groupby(keys)
              .agg(n=("mean_auc","size"),
                   mean_auc=("mean_auc","mean"),
                   median_auc=("mean_auc","median"),
                   mean_mcc_best=("mcc_best","mean"),
                   median_mcc_best=("mcc_best","median"),
                   mean_mcc_at_fixed=("mcc_at_fixed","mean"))
              .reset_index())
    base["composite"] = base["mean_auc"] + base["mean_mcc_best"]
    base = base.round(4)

    for sort_col, suffix in [("mean_auc","by_mean_auc"),
                              ("median_auc","by_median_auc"),
                              ("mean_mcc_best","by_mcc"),
                              ("composite","by_composite")]:
        top = base.sort_values(sort_col, ascending=False).head(a.top_n)
        path = out / f"{a.prefix}_top_{suffix}.csv"
        top.to_csv(path, index=False)
        print(f"[WRITE] {path}")
        print(top.to_string(index=False)); print()

if __name__ == "__main__":
    main()
