#!/usr/bin/env python3
"""
Standard axis-comparison figures and tables for an aggregated Bassez2021 run.

Inputs:  --in_csv  = output of aggregate_pilot.py (or any stage's aggregator).
Outputs (under --out_dir): PNG and PDF for each axis comparison + summary CSVs.

Comparisons emitted (each one panel per metric, metrics = mean_auc and mcc_best):
  - raw vs ALRA                      (axis: imputation)
  - Spearman vs cosine               (axis: similarity)
  - star vs bidirectional            (axis: walk_strategy)
  - per-patient vs joint             (axis: aggregation_strategy, if multi)
  - by walk count                    (axis: walks)
  - by cell type                     (per group)
  - top configs by mean_auc / median_auc / mcc_best (tables)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = [("mean_auc", "Mean CORUM AUC", 0.5),
           ("mcc_best", "Best CORUM MCC", 0.0)]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--prefix", default="pilot",
                   help="Output filename prefix (e.g. 'pilot', 'stage2').")
    return p.parse_args()


def derive_celltype(df: pd.DataFrame) -> pd.Series:
    """group_dir = 'patient=X__celltype=Y' -> Y; falls back to splitting group."""
    if "celltype" in df.columns and df["celltype"].notna().any():
        return df["celltype"]
    if "group_dir" in df.columns:
        return (df["group_dir"].astype(str)
                .str.extract(r"celltype=(.+)$", expand=False))
    # Fallback for "BIOKEY_30_Malignant"
    return df["group"].astype(str).str.split("_", n=2).str[2]


def _strip_plot(ax, data, x_col, y_col, ref_line=None, title=""):
    """Categorical strip + mean dot."""
    cats = sorted(data[x_col].dropna().unique())
    for i, c in enumerate(cats):
        vals = data.loc[data[x_col] == c, y_col].dropna().values
        rng = np.random.default_rng(42 + i)
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals,
                   s=22, alpha=0.55, edgecolors="none")
        if len(vals) > 0:
            ax.scatter([i], [vals.mean()], marker="D", s=70,
                       facecolor="black", zorder=5)
    if ref_line is not None:
        ax.axhline(ref_line, ls="--", color="red", alpha=0.7,
                   label=f"chance={ref_line}")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_xlabel(x_col)
    ax.set_title(title)
    if ref_line is not None: ax.legend(loc="best", fontsize=8)


def axis_figure(df, axis, out_path_stem):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (col, label, ref) in zip(axes, METRICS):
        if col not in df.columns or df[col].dropna().empty:
            ax.text(0.5, 0.5, f"{col} missing", ha="center", va="center")
            continue
        _strip_plot(ax, df.dropna(subset=[col]), axis, col, ref_line=ref,
                    title=f"{label} by {axis}")
        ax.set_ylabel(label)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_path_stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[WRITE] {out_path_stem}.png / .pdf")


def best_config_table(df, out_csv, group_keys, top_n=10):
    needed = set(group_keys) | {"mean_auc","mcc_best"}
    if not needed.issubset(df.columns):
        return None
    tbl = (df.dropna(subset=["mean_auc","mcc_best"])
             .groupby(list(group_keys))
             .agg(n=("mean_auc","size"),
                  mean_auc=("mean_auc","mean"),
                  median_auc=("mean_auc","median"),
                  mean_mcc_best=("mcc_best","mean"),
                  median_mcc_best=("mcc_best","median"),
                  mean_mcc_at_fixed=("mcc_at_fixed","mean"))
             .reset_index())
    tbl["composite"] = tbl["mean_auc"] + tbl["mean_mcc_best"]
    tbl = tbl.sort_values("composite", ascending=False).round(4).head(top_n)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out_csv, index=False)
    print(f"[WRITE] {out_csv}")
    return tbl


def main():
    a = parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.in_csv)
    df["celltype"] = derive_celltype(df)

    # Standard axis comparisons (only if axis has >1 value).
    for axis in ["imputation","similarity","walk_strategy",
                 "aggregation_strategy","walks","celltype"]:
        if axis in df.columns and df[axis].nunique(dropna=True) > 1:
            stem = str(out / f"{a.prefix}_by_{axis}")
            axis_figure(df, axis, stem)

    # Best-config tables.
    best_config_table(df, out / f"{a.prefix}_top_by_imp_sim_strat.csv",
                      ("imputation","similarity","walk_strategy"))
    best_config_table(df, out / f"{a.prefix}_top_by_celltype.csv",
                      ("celltype","imputation","similarity","walk_strategy"))

    # Per-cell-type signal table (which celltypes have the strongest mean AUC across configs).
    if "celltype" in df.columns:
        ct = (df.dropna(subset=["mean_auc"])
                .groupby("celltype")
                .agg(n=("mean_auc","size"),
                     mean_auc=("mean_auc","mean"),
                     median_auc=("mean_auc","median"),
                     mean_mcc_best=("mcc_best","mean"))
                .sort_values("mean_auc", ascending=False).round(4).reset_index())
        ct.to_csv(out / f"{a.prefix}_celltype_signal.csv", index=False)
        print(f"[WRITE] {out / f'{a.prefix}_celltype_signal.csv'}")

    print("[DONE] figures written to", out)


if __name__ == "__main__":
    main()
