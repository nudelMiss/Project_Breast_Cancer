#!/usr/bin/env python3
"""
Run AUC + MCC benchmark for one manifest row's trained model.
Idempotent: skips if both summaries already exist.
"""
from __future__ import annotations
import argparse, csv, subprocess, sys
from pathlib import Path


def read_row(manifest_path: Path, row_index: int) -> dict:
    rows = list(csv.DictReader(open(manifest_path), delimiter="\t"))
    if row_index < 0 or row_index >= len(rows):
        sys.exit(f"FATAL: row_index {row_index} out of range (n={len(rows)})")
    return rows[row_index]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--row_index", type=int, required=True)
    p.add_argument("--corum", default="resources/corum_core_complexes.tsv")
    p.add_argument("--force", action="store_true",
                   help="Re-run AUC/MCC even if existing summaries are present.")
    args = p.parse_args()

    row = read_row(Path(args.manifest), args.row_index)
    model_path = Path(row["model_path"])
    if not model_path.exists():
        sys.exit(f"FATAL: model not trained yet: {model_path}")

    auc_dir = Path(row["auc_dir"]); mcc_dir = Path(row["mcc_dir"])
    auc_dir.mkdir(parents=True, exist_ok=True)
    mcc_dir.mkdir(parents=True, exist_ok=True)

    # AUC -----------------------------------------------------------
    auc_summary = auc_dir / "corum_auc_summary.csv"
    if auc_summary.exists() and not args.force:
        print(f"[SKIP] AUC exists: {auc_summary}", flush=True)
    else:
        # benchmark_corum_auc.py auto-suffixes <output_dir>/<group>/<config_tag>/,
        # so we pass the GRANDPARENT of our canonical auc_dir.
        auc_root = auc_dir.parent.parent
        cmd = [
            sys.executable, "-u", "scripts/AUC/benchmark_corum_auc.py",
            "--embedding_path", str(model_path),
            "--corum_path", args.corum,
            "--output_dir", str(auc_root),
            "--min_complex_size", "3",
            "--random_seed", "42",
        ]
        print("[CMD]", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            sys.exit(f"AUC benchmark failed rc={rc}")

    # MCC -----------------------------------------------------------
    mcc_summary = mcc_dir / "mcc_summary.csv"
    if mcc_summary.exists() and not args.force:
        print(f"[SKIP] MCC exists: {mcc_summary}", flush=True)
    else:
        cmd = [
            sys.executable, "-u", "scripts/MCC/benchmark_corum_mcc.py",
            "--embedding_path", str(model_path),
            "--corum_path", args.corum,
            "--output_dir", str(mcc_dir),
            "--min_complex_size", "3",
            "--random_seed", "42",
            "--fixed_threshold", "0.5",
        ]
        print("[CMD]", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            sys.exit(f"MCC benchmark failed rc={rc}")

    print("[DONE] eval complete for row", args.row_index, flush=True)


if __name__ == "__main__":
    main()
