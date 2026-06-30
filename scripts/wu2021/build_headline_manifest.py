#!/usr/bin/env python3
"""Build Wu2021 headline manifest: raw + cosine + bidirectional + walks=1
- 89 per-patient rows (one per group from dataset_summary.csv)
- 1 joint row pooling all 89 groups
TSV columns match the Bassez manifest schema so existing train/eval scripts work."""
from __future__ import annotations
import csv
from pathlib import Path

ROOT     = Path("/groups/ofircohen-group/users/michalnu_yuvat/project_breast")
SUMMARY  = ROOT / "results/wu2021/summaries/dataset_summary.csv"
OUT      = ROOT / "results/wu2021/manifests/headline_manifest.tsv"
RR       = "results/wu2021"
IN_ROOT  = "exports_wu"

# Locked headline config from Bassez results
IMP   = "raw"     # -> graph_method "var75"
GM    = "var75"
SIM   = "cosine"
STRAT = "bidirectional"
WALKS = 1
WL    = 3

def cfg_tag(agg):
    a = "perpat" if agg == "per_patient_embeddings" else "joint"
    return f"{IMP}_{SIM}_{STRAT}_w{WALKS}_k5_wl{WL}_{a}"

def per_patient_row(group, group_dir):
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
    rows = []
    # Build group -> patient_celltype tag like Bassez ("CID3586_T-cells")
    with open(SUMMARY) as f:
        for r in csv.DictReader(f):
            group_tag = f"{r['patient']}_{r['celltype']}"
            group_dir = r["group"]   # e.g. patient=CID3586__celltype=T-cells
            rows.append(per_patient_row(group_tag, group_dir))
    rows.append(joint_row())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        for r in rows: w.writerow(r)

    n_pp = sum(1 for r in rows if r["stage"] == "headline_pp")
    n_jt = sum(1 for r in rows if r["stage"] == "headline_joint")
    print(f"[OK] {OUT}")
    print(f"  per-patient rows: {n_pp}")
    print(f"  joint rows:       {n_jt}")
    print(f"  total:            {len(rows)}")

if __name__ == "__main__":
    main()
