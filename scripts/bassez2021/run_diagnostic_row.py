#!/usr/bin/env python3
"""Train W2V + run CORUM AUC for one row of the diagnostic manifest."""
import sys, csv, subprocess, shutil, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--row_index", type=int, required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest), delimiter="\t"))
    r = rows[args.row_index]
    print("### ROW", args.row_index, "###", flush=True)
    for k, v in r.items(): print(f"  {k}: {v}", flush=True)

    # Layout:
    #   models/<group_tag>/<label>/gene_embeddings.model
    #   auc/<group_tag>/<label>/corum_auc_summary.csv
    base = Path("results/bassez2021/supervisor_diagnostic")
    model_dir = base / "models" / r["group_tag"] / r["label"]
    model_path = model_dir / "gene_embeddings.model"
    auc_root = base / "auc"  # benchmark adds group_tag/label/ suffix
    scratch = base / "models" / r["group_tag"] / f"{r['label']}__scratch"

    if not model_path.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        scratch.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-u", "scripts/train_model_new.py",
            "--in_root", str(Path(r["group_dir"]).parent),
            "--only_group", Path(r["group_dir"]).name,
            "--out_root", str(scratch),
            "--graph_method", r["graph_method"],
            "--sim", r["sim"],
            "--walk_strategy", r["walk_strategy"],
            "--walks", r["walks"],
            "--walk_length", r["walk_length"],
            "--k_nearest", r["k_nearest"],
            "--vector_dim", "64",
            "--epochs", "20",
            "--window", "5",
            "--min_count", "1",
            "--variance_keep_frac", "0.75",
            "--seed", "42",
        ]
        print("[CMD]", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0: sys.exit(f"train failed rc={rc}")
        cands = list(scratch.rglob("gene_embeddings.model"))
        if len(cands) != 1: sys.exit(f"want 1 model, found {len(cands)}")
        src = cands[0].parent
        for item in src.iterdir():
            dst = model_dir / item.name
            if dst.exists():
                if dst.is_file(): dst.unlink()
                else: shutil.rmtree(dst)
            item.rename(dst)
        shutil.rmtree(scratch, ignore_errors=True)
    else:
        print(f"[SKIP train] {model_path}", flush=True)

    auc_summary = auc_root / r["group_tag"] / r["label"] / "corum_auc_summary.csv"
    if auc_summary.exists():
        print(f"[SKIP AUC] {auc_summary}", flush=True)
        return
    cmd = [
        sys.executable, "-u", "scripts/AUC/benchmark_corum_auc.py",
        "--embedding_path", str(model_path),
        "--corum_path", "resources/corum_core_complexes.tsv",
        "--output_dir", str(auc_root),
        "--min_complex_size", "3",
        "--random_seed", "42",
    ]
    print("[CMD]", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0: sys.exit(f"AUC failed rc={rc}")
    print("DONE row", args.row_index, flush=True)

if __name__ == "__main__":
    main()
