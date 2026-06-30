#!/usr/bin/env python3
"""
Stage 0: Bassez2021 sanity check.

For every patient/celltype group under exports_bassez/, record:
  - n_cells (from cells.csv)
  - n_genes (from genes.csv)
  - mtx_rows, mtx_cols, mtx_nnz (from MatrixMarket header — no full load)
  - corum_overlap (genes that appear in CORUM)
  - has_complete_files (expr.mtx, genes.csv, cells.csv all present and non-empty)

Output:
  results/bassez2021/summaries/dataset_summary.csv
  results/bassez2021/summaries/dataset_summary_report.txt
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_root", default="exports_bassez")
    p.add_argument("--corum", default="resources/corum_core_complexes.tsv")
    p.add_argument("--out_csv",
                   default="results/bassez2021/summaries/dataset_summary.csv")
    p.add_argument("--out_report",
                   default="results/bassez2021/summaries/dataset_summary_report.txt")
    return p.parse_args()

def read_mtx_header(p: Path):
    """Return (rows, cols, nnz) by reading only the MatrixMarket header."""
    with open(p, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            parts = s.split()
            if len(parts) >= 3:
                return int(parts[0]), int(parts[1]), int(parts[2])
            break
    return None, None, None

def count_lines(p: Path) -> int:
    """genes.csv and cells.csv in this export have NO header — every line is data."""
    n = 0
    with open(p, "r") as f:
        for _ in f:
            n += 1
    return n

def load_corum_genes(p: Path) -> set:
    genes = set()
    with open(p, "r") as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                genes.add(parts[1].strip())
    return genes

def load_group_genes(p: Path) -> list:
    """No header. Each line is a single gene symbol (may be quoted)."""
    genes = []
    with open(p, "r") as f:
        for line in f:
            g = line.split(",")[0].strip().strip('"')
            if g:
                genes.append(g)
    return genes

def main():
    args = parse_args()
    in_root = Path(args.in_root)
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_report = Path(args.out_report)

    if not in_root.is_dir():
        print(f"FATAL: in_root does not exist: {in_root}", file=sys.stderr); sys.exit(1)
    corum_path = Path(args.corum)
    if not corum_path.is_file():
        print(f"FATAL: CORUM not found: {corum_path}", file=sys.stderr); sys.exit(1)

    corum = load_corum_genes(corum_path)
    print(f"[CORUM] unique genes: {len(corum):,}")

    rows = []
    groups = sorted(p for p in in_root.iterdir() if p.is_dir())
    print(f"[SCAN] {len(groups)} group directories under {in_root}")

    issues = []
    celltypes_seen = set()
    for gd in groups:
        name = gd.name
        # parse patient + celltype from `patient=X__celltype=Y`
        try:
            patient = name.split("__")[0].split("=", 1)[1]
            celltype = name.split("__")[1].split("=", 1)[1]
        except Exception:
            patient, celltype = "?", "?"
        celltypes_seen.add(celltype)

        expr = gd / "expr.mtx"
        gcsv = gd / "genes.csv"
        ccsv = gd / "cells.csv"
        complete = expr.exists() and gcsv.exists() and ccsv.exists()
        if not complete:
            issues.append(f"INCOMPLETE: {name}")

        mtx_rows = mtx_cols = mtx_nnz = None
        if expr.exists() and expr.stat().st_size > 0:
            mtx_rows, mtx_cols, mtx_nnz = read_mtx_header(expr)
        n_genes_csv = count_lines(gcsv) if gcsv.exists() else 0
        n_cells_csv = count_lines(ccsv) if ccsv.exists() else 0

        # CORUM overlap on this group's genes
        try:
            group_genes = set(load_group_genes(gcsv)) if gcsv.exists() else set()
        except Exception as e:
            group_genes = set()
            issues.append(f"GENES_PARSE_FAIL: {name}: {e}")
        corum_overlap = len(group_genes & corum)

        # MTX/CSV consistency
        # expected layout: rows=cells, cols=genes (because group has cells.csv and
        # train_model_new.py uses mmread().tocsr() on the matrix as cells x genes)
        # Let's verify by checking which dim matches n_cells:
        mtx_orient = ""
        if mtx_rows is not None:
            if mtx_rows == n_cells_csv and mtx_cols == n_genes_csv:
                mtx_orient = "cells_x_genes"
            elif mtx_cols == n_cells_csv and mtx_rows == n_genes_csv:
                mtx_orient = "genes_x_cells"
            else:
                mtx_orient = "MISMATCH"
                issues.append(f"DIM_MISMATCH: {name} mtx={mtx_rows}x{mtx_cols} cells={n_cells_csv} genes={n_genes_csv}")

        rows.append({
            "group": name,
            "patient": patient,
            "celltype": celltype,
            "n_cells_csv": n_cells_csv,
            "n_genes_csv": n_genes_csv,
            "mtx_rows": mtx_rows,
            "mtx_cols": mtx_cols,
            "mtx_nnz": mtx_nnz,
            "mtx_orient": mtx_orient,
            "corum_overlap": corum_overlap,
            "complete": complete,
        })

    fields = ["group","patient","celltype","n_cells_csv","n_genes_csv",
              "mtx_rows","mtx_cols","mtx_nnz","mtx_orient","corum_overlap","complete"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"[WRITE] {out_csv}")

    # --- text report ---
    rep = []
    rep.append("Bassez2021 — Stage 0 sanity report")
    rep.append("="*60)
    rep.append(f"Source: {in_root}")
    rep.append(f"CORUM:  {corum_path}  ({len(corum):,} unique genes)")
    rep.append("")
    rep.append(f"Total groups : {len(rows)}")
    rep.append(f"Cell types   : {sorted(celltypes_seen)}")
    rep.append(f"Equivocal    : {'present' if any(c.lower().startswith('equiv') for c in celltypes_seen) else 'ABSENT (good)'}")

    # per-celltype
    from collections import defaultdict
    bycell = defaultdict(list)
    for r in rows: bycell[r["celltype"]].append(r)
    rep.append("")
    rep.append("Per-celltype counts:")
    rep.append(f"  {'celltype':<14} {'groups':>7} {'mean_cells':>11} {'med_cells':>10} {'mean_corum':>11}")
    for ct in sorted(bycell):
        rs = bycell[ct]
        cells = [r["n_cells_csv"] for r in rs]
        co = [r["corum_overlap"] for r in rs]
        mean_c = sum(cells)/len(cells); med_c = sorted(cells)[len(cells)//2]
        mean_co = sum(co)/len(co)
        rep.append(f"  {ct:<14} {len(rs):>7d} {mean_c:>11.0f} {med_c:>10d} {mean_co:>11.0f}")

    rep.append("")
    rep.append("Cell-count distribution (overall):")
    all_cells = sorted(r["n_cells_csv"] for r in rows)
    rep.append(f"  min={all_cells[0]}  Q1={all_cells[len(all_cells)//4]}  "
               f"median={all_cells[len(all_cells)//2]}  "
               f"Q3={all_cells[(3*len(all_cells))//4]}  max={all_cells[-1]}")
    rep.append(f"  groups with <200 cells (likely unstable): "
               f"{sum(1 for c in all_cells if c < 200)}")

    rep.append("")
    rep.append("CORUM overlap distribution:")
    all_co = sorted(r["corum_overlap"] for r in rows)
    rep.append(f"  min={all_co[0]}  median={all_co[len(all_co)//2]}  max={all_co[-1]}")

    rep.append("")
    if issues:
        rep.append(f"Issues ({len(issues)}):")
        for s in issues[:20]:
            rep.append(f"  - {s}")
        if len(issues) > 20:
            rep.append(f"  ... and {len(issues)-20} more")
    else:
        rep.append("Issues: NONE")

    out_report.write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    print(f"\n[WRITE] {out_report}")

if __name__ == "__main__":
    main()
