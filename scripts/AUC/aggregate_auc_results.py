#!/usr/bin/env python3
"""
Aggregate CORUM AUC per-embedding summary CSV files.

Searches recursively under an output root for files named
"corum_auc_summary.csv", concatenates them, adds metadata parsed from path,
and writes one combined CSV.
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate CORUM AUC summary files")
    parser.add_argument(
        "--input_root",
        type=str,
        default="results/auc_benchmarks/5_walks_all",
        help="Root directory containing per-embedding corum_auc_summary.csv files",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/auc_benchmarks/5_walks_all_summary_combined.csv",
        help="Combined output CSV path",
    )
    return parser.parse_args()


def add_path_metadata(df: pd.DataFrame, summary_path: Path, input_root: Path) -> pd.DataFrame:
    """Attach metadata columns inferred from relative path structure."""
    rel_parts = summary_path.relative_to(input_root).parts
    # Expected pattern:
    # input_root / patient=...__celltype=... / sim=... / corum_auc_summary.csv
    patient_celltype = rel_parts[0] if len(rel_parts) >= 1 else ""
    config_tag = rel_parts[1] if len(rel_parts) >= 2 else ""

    patient = ""
    celltype = ""
    if "__celltype=" in patient_celltype and patient_celltype.startswith("patient="):
        left, right = patient_celltype.split("__celltype=", 1)
        patient = left.replace("patient=", "", 1)
        celltype = right

    out = df.copy()
    out["summary_path"] = str(summary_path)
    out["patient_celltype"] = patient_celltype
    out["patient"] = patient
    out["celltype"] = celltype
    out["config_tag"] = config_tag
    return out


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_csv = Path(args.output_csv)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    summary_files = sorted(input_root.rglob("corum_auc_summary.csv"))
    if not summary_files:
        raise FileNotFoundError(f"No corum_auc_summary.csv files found under {input_root}")

    combined_frames = []
    for summary_file in summary_files:
        df = pd.read_csv(summary_file)
        df = add_path_metadata(df, summary_file, input_root)
        combined_frames.append(df)

    combined = pd.concat(combined_frames, ignore_index=True)
    combined = combined.sort_values(by=["patient_celltype", "config_tag"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    print(f"Input root: {input_root}")
    print(f"Found summaries: {len(summary_files)}")
    print(f"Wrote combined CSV: {output_csv}")


if __name__ == "__main__":
    main()
