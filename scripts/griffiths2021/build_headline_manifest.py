#!/usr/bin/env python3
"""Griffiths2021 headline manifest: raw + cosine + bidirectional + walks=1.
Mirrors Wu builder. Excludes equivocal-like cell types (defensive duplicate;
the R exporter already skips them, but enforces it here too)."""
from __future__ import annotations
import csv
from pathlib import Path

ROOT     = Path("/groups/ofircohen-group/users/michalnu_yuvat/project_breast")
SUMMARY  = ROOT / "results/griffiths2021/summaries/dataset_summary.csv"
OUT      = ROOT / "results/griffiths2021/manifests/headline_manifest.tsv"
RR       = "results/griffiths2021"
IN_ROOT  = "exports_griffiths"

IMP, GM, SIM, STRAT, WALKS, WL = "raw", "var75", "cosine", "bidirectional", 1, 3
EXCLUDE_CELLTYPES = {"equivocal"}  # case-insensitive substring match

def cfg_tag(agg):
    a = "perpat" if agg == "per_patient_embeddings" else "joint"
    return f"{IMP}_{SIM}_{STRAT}_w{WALKS}_k5_wl{WL}_{a}"

def pp_row(group, group_dir):
    tag = cfg_tag("per_patient_embeddings")
    return dict(
        stage="headline_pp", group=group, group_dir=group_dir,
        imputation=IMP, graph_method=GM, similarity=SIM,
        walk_strategy=STRAT, walks=WALKS, walk_length=WL,
        aggregation_strategy="per_patient_embeddings",
        config_tag=tag, in_root=IN_ROOT,
        model_dir=f"{RR}/models/{group}/{tag}",
        model_path=f"{RR}/models/{group}/{tag}/gene_embeddings.model",
        graph_cache_dir=f"{RR}/graphs",
        auc_dir=f"{RR}/auc/{group}/{tag}",
        mcc_dir=f"{RR}/mcc/{group}/{tag}",
    )

def joint_row():
    tag = cfg_tag("joint_embeddings")
    return dict(
        stage="headline_joint", group="ALL", group_dir="",
        imputation=IMP, graph_method=GM, similarity=SIM,
        walk_strategy=STRAT, walks=WALKS, walk_length=WL,
        aggregation_strategy="joint_embeddings",
        config_tag=tag, in_root=IN_ROOT,
        model_dir=f"{RR}/models/ALL/{tag}",
        model_path=f"{RR}/models/ALL/{tag}/gene_embeddings.model",
        graph_cache_dir=f"{RR}/graphs",
        auc_dir=f"{RR}/auc/ALL/{tag}",
        mcc_dir=f"{RR}/mcc/ALL/{tag}",
    )

def main():
    rows, skipped = [], 0
    with open(SUMMARY) as f:
        for r in csv.DictReader(f):
            ct = r["celltype"]
            if any(ex.lower() in ct.lower() for ex in EXCLUDE_CELLTYPES):
                skipped += 1
                continue
            tag = f"{r['patient']}_{r['celltype']}"
            rows.append(pp_row(tag, r["group"]))
    rows.append(joint_row())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[OK] {OUT}")
    print(f"  per-patient rows: {sum(1 for r in rows if r['stage']=='headline_pp')}")
    print(f"  joint rows:       {sum(1 for r in rows if r['stage']=='headline_joint')}")
    print(f"  total:            {len(rows)}")
    print(f"  skipped (equivocal-like): {skipped}")

if __name__ == "__main__":
    main()
