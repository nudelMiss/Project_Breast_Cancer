#!/usr/bin/env python3
"""
Aggregate Stage-1 pilot (or any per-patient) results across all manifest rows.

Joins every row of the manifest with its AUC summary + MCC summary on disk.
Writes a combined CSV at the chosen path.

Each output row contains:
  manifest columns + AUC fields (mean_auc, median_auc, weighted_mean_auc, p_value,
  n_complexes_used, n_shared_genes) + MCC fields (mcc_best, threshold_best,
  precision/recall/f1/TP/FP/TN/FN at best, and the same at the fixed threshold).
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import pandas as pd

AUC_NAME = "corum_auc_summary.csv"
MCC_NAME = "mcc_summary.csv"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--require_both", action="store_true",
                   help="Skip rows missing either AUC or MCC. Default: emit a row with NaNs.")
    return p.parse_args()

def main():
    a = parse_args()
    rows = list(csv.DictReader(open(a.manifest), delimiter="\t"))
    print(f"[AGG] manifest rows: {len(rows)}")

    out_rows = []
    n_missing_auc = n_missing_mcc = 0
    for r in rows:
        auc_path = Path(r["auc_dir"]) / AUC_NAME
        mcc_path = Path(r["mcc_dir"]) / MCC_NAME

        auc_present = auc_path.exists()
        mcc_present = mcc_path.exists()
        if not auc_present: n_missing_auc += 1
        if not mcc_present: n_missing_mcc += 1
        if a.require_both and not (auc_present and mcc_present):
            continue

        out = dict(r)
        if auc_present:
            auc_df = pd.read_csv(auc_path)
            for col in ("mean_auc","median_auc","weighted_mean_auc","p_value",
                        "n_complexes_used","n_shared_genes"):
                if col in auc_df.columns:
                    out[col] = auc_df[col].iloc[0]
        else:
            for col in ("mean_auc","median_auc","weighted_mean_auc","p_value",
                        "n_complexes_used","n_shared_genes"):
                out[col] = None

        if mcc_present:
            mcc_df = pd.read_csv(mcc_path)
            mcc_cols = [c for c in mcc_df.columns if c not in ("embedding_path","corum_path")]
            for col in mcc_cols:
                out[col] = mcc_df[col].iloc[0]
        else:
            for col in ("mcc_best","threshold_best","mcc_at_fixed","fixed_threshold",
                        "precision_best","recall_best","f1_best",
                        "tp_best","fp_best","tn_best","fn_best",
                        "precision_at_fixed","recall_at_fixed","f1_at_fixed",
                        "tp_at_fixed","fp_at_fixed","tn_at_fixed","fn_at_fixed",
                        "n_positives","n_negatives"):
                out[col] = None

        out["auc_present"] = auc_present
        out["mcc_present"] = mcc_present
        out_rows.append(out)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out_rows)
    df.to_csv(a.out, index=False)
    print(f"[WRITE] {a.out}  rows={len(df)}")
    print(f"[STATS] missing AUC: {n_missing_auc}   missing MCC: {n_missing_mcc}")

    # Quick console digest by config_tag.
    if "mean_auc" in df.columns and df["mean_auc"].notna().any():
        cols = ["imputation","similarity","walk_strategy"]
        digest = (df.dropna(subset=["mean_auc"])
                    .groupby(cols)[["mean_auc","mcc_best","mcc_at_fixed"]]
                    .agg(["mean","median","count"]).round(3))
        print()
        print("=== per-(imp, sim, strat) digest ===")
        print(digest)

if __name__ == "__main__":
    main()
