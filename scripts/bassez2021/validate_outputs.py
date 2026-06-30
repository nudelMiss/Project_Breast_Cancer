#!/usr/bin/env python3
"""
Walk a manifest TSV; report missing/corrupt model/AUC/MCC outputs.
Optionally emit a delta TSV containing ONLY the rows that still need work.

Usage:
    validate_outputs.py --manifest M.tsv [--out_delta missing.tsv] [--quiet]
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

def is_valid_csv(p: Path) -> bool:
    if not p.exists(): return False
    if p.stat().st_size == 0: return False
    # Must have a header + at least one data line
    with open(p) as f:
        first = f.readline()
        second = f.readline()
    return bool(first) and bool(second)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_delta", default=None,
                   help="If set, write a TSV with only the rows missing model/AUC/MCC.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()

def main():
    a = parse_args()
    rows = list(csv.DictReader(open(a.manifest), delimiter="\t"))
    needs_model = needs_auc = needs_mcc = 0
    delta = []
    for r in rows:
        model_ok = Path(r["model_path"]).exists() and Path(r["model_path"]).stat().st_size > 0
        auc_ok = is_valid_csv(Path(r["auc_dir"]) / "corum_auc_summary.csv")
        mcc_ok = is_valid_csv(Path(r["mcc_dir"]) / "mcc_summary.csv")
        if not model_ok: needs_model += 1
        if not auc_ok:   needs_auc += 1
        if not mcc_ok:   needs_mcc += 1
        if not (model_ok and auc_ok and mcc_ok):
            if not a.quiet:
                print(f"MISS row={rows.index(r):3d}  group={r['group']:<28} "
                      f"tag={r['config_tag']:<55} "
                      f"model={'Y' if model_ok else 'N'} "
                      f"auc={'Y' if auc_ok else 'N'} "
                      f"mcc={'Y' if mcc_ok else 'N'}")
            delta.append(r)

    print()
    print(f"[VALIDATE] manifest rows: {len(rows)}")
    print(f"[VALIDATE] need model: {needs_model}   need AUC: {needs_auc}   need MCC: {needs_mcc}")
    print(f"[VALIDATE] complete: {len(rows) - len(delta)} / {len(rows)}")

    if a.out_delta:
        Path(a.out_delta).parent.mkdir(parents=True, exist_ok=True)
        cols = list(rows[0].keys())
        with open(a.out_delta, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
            w.writeheader(); w.writerows(delta)
        print(f"[WRITE] {a.out_delta}  delta_rows={len(delta)}")

    sys.exit(0 if not delta else 1)

if __name__ == "__main__":
    main()
