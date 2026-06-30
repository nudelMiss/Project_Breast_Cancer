#!/usr/bin/env python3
"""
Read one row from a manifest TSV and train the corresponding Word2Vec model.

Idempotent: if model_path already exists, skip (train_model_new.py already
handles its own skip, but we also do an explicit check before invoking).

For aggregation_strategy == 'per_patient_embeddings', delegates to
train_model_new.py with --only_group $group_dir.

For aggregation_strategy == 'joint_embeddings', invokes train_one_joint.py.
"""
from __future__ import annotations
import argparse, csv, os, subprocess, sys
from pathlib import Path


def read_row(manifest_path: Path, row_index: int) -> dict:
    """row_index is 0-based among data rows (excluding header)."""
    rows = list(csv.DictReader(open(manifest_path), delimiter="\t"))
    if row_index < 0 or row_index >= len(rows):
        sys.exit(f"FATAL: row_index {row_index} out of range (n={len(rows)})")
    return rows[row_index]


def train_per_patient(row: dict) -> int:
    model_path = Path(row["model_path"])
    if model_path.exists():
        print(f"[SKIP] model exists: {model_path}", flush=True)
        return 0
    model_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u", "scripts/train_model_new.py",
        "--in_root", row["in_root"],
        "--only_group", row["group_dir"],
        # We pass an out_root that LANDS the model at `model_path` because
        # train_model_new.py builds the path as
        #   <out_root>/<group_tag>/<cfg.tag()>/gene_embeddings.model
        # and cfg.tag() doesn't match our config_tag scheme. We therefore set
        # --out_root to a per-row scratch dir and then move the file into place.
        # Simpler approach: pass out_root = parent.parent of model_path, then
        # AFTER training, move the trained dir contents into the canonical place.
        "--out_root", row["model_dir"] + "__scratch",
        "--graph_method", row["graph_method"],
        "--sim", row["similarity"],
        "--walk_strategy", row["walk_strategy"],
        "--walks", row["walks"],
        "--walk_length", row["walk_length"],
        "--k_nearest", "5",
        "--vector_dim", "64",
        "--epochs", "20",
        "--window", "5",
        "--min_count", "1",
        "--variance_keep_frac", "0.75",
        "--graph_cache_dir", row["graph_cache_dir"],
        "--seed", "42",
    ]
    print("[CMD]", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        return rc

    # train_model_new.py writes to <scratch>/<group_tag>/<cfg.tag()>/...
    # Move/rename that to row['model_dir'].
    scratch = Path(row["model_dir"] + "__scratch")
    # Find the trained subdir.
    candidates = list(scratch.rglob("gene_embeddings.model"))
    if len(candidates) != 1:
        print(f"[FATAL] expected exactly one trained model under {scratch}, "
              f"found {len(candidates)}", file=sys.stderr)
        return 2
    src_dir = candidates[0].parent
    # Move src_dir contents into row['model_dir']
    Path(row["model_dir"]).mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        dst = Path(row["model_dir"]) / item.name
        if dst.exists():
            dst.unlink() if dst.is_file() else None
        item.rename(dst)
    # Clean up scratch.
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"[MOVED] -> {row['model_dir']}", flush=True)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--row_index", type=int, required=True,
                   help="0-based data row index (use SLURM_ARRAY_TASK_ID)")
    args = p.parse_args()

    row = read_row(Path(args.manifest), args.row_index)
    print(f"=== ROW {args.row_index} ===", flush=True)
    for k, v in row.items():
        print(f"  {k}: {v}", flush=True)

    if row["aggregation_strategy"] == "per_patient_embeddings":
        rc = train_per_patient(row)
    elif row["aggregation_strategy"] == "joint_embeddings":
        # Delegate to the joint runner (which lives next to this script).
        cmd = [
            sys.executable, "-u",
            str(Path(__file__).parent / "train_one_joint.py"),
            "--manifest", args.manifest, "--row_index", str(args.row_index),
        ]
        rc = subprocess.run(cmd).returncode
    else:
        print(f"[FATAL] unknown aggregation_strategy: {row['aggregation_strategy']}",
              file=sys.stderr)
        rc = 3
    sys.exit(rc)


if __name__ == "__main__":
    main()
