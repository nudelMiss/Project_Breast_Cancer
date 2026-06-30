#!/usr/bin/env python3
"""
Build a Bassez2021 experiment manifest.

A manifest is a TSV where each row is one (group x config) job. The same row is
used by both the training step and the AUC/MCC evaluation step. Every path in
the row is deterministic; downstream scripts MUST NOT reconstruct paths from
parameters elsewhere — they should read the path columns.

Stages:
  pilot       : 6 pilot groups x 8 configs (raw/alra x sp/cos x star/bidir), w=100
  stage2      : <chosen> configs x all 184 groups, w=100
  saturation  : top-2 configs x all 184 groups x walks in {1,5,10,50,100,1000}
  joint       : top-2 configs, joint aggregation, one row per config

Columns (per row):
  stage, group, group_dir, imputation, similarity, walk_strategy, walks,
  walk_length, aggregation_strategy,
  config_tag, in_root,
  model_path, model_dir, graph_cache_dir,
  auc_dir, mcc_dir
"""
from __future__ import annotations
import argparse, csv, sys
from itertools import product
from pathlib import Path

# Mapping from user-facing "imputation" label to train_model_new.py's --graph_method.
# "raw" reuses var75 (top-75% variance, no imputation) -- consistent with prior Wu work.
IMP_TO_METHOD = {"raw": "var75", "alra": "alra"}

PILOT_GROUPS = [
    # Picked to span celltype, size, and sparsity. See plan.
    "BIOKEY_30_Malignant",
    "BIOKEY_18_T_cell",
    "BIOKEY_3_Fibroblast",
    "BIOKEY_4_B_cell",
    "BIOKEY_10_Myeloid",
    "BIOKEY_13_Endothelial",
]

# Default axis values.
DEFAULT_IMP = ["raw", "alra"]
DEFAULT_SIM = ["spearman", "cosine"]
DEFAULT_STRAT = ["star", "bidirectional"]
DEFAULT_WALK_LENGTH = 3        # 3 neighbors per side; sentence of length 7
DEFAULT_WALKS_STAGE12 = 100
DEFAULT_SATURATION_WALKS = [1, 5, 10, 50, 100, 1000]


def group_to_indir(group_tag: str) -> str:
    """BIOKEY_30_Malignant  ->  patient=BIOKEY_30__celltype=Malignant.

    Cell-type labels in exports_bassez have underscores (B_cell, T_cell), so we
    can't simply split on '_'. Instead we use the dataset_summary mapping when
    available.
    """
    # Read the summary file written by stage0_sanity_check.py.
    summary = Path("results/bassez2021/summaries/dataset_summary.csv")
    if not summary.exists():
        raise FileNotFoundError(
            "Run stage0_sanity_check.py first; it produces dataset_summary.csv "
            "which is used to map group_tag -> in_root subdir."
        )
    with open(summary) as f:
        for row in csv.DictReader(f):
            tag = f"{row['patient']}_{row['celltype']}"
            if tag == group_tag:
                return row["group"]
    raise KeyError(f"group_tag not found in dataset_summary.csv: {group_tag}")


def all_groups() -> list[str]:
    """All Bassez2021 group_tags from dataset_summary.csv."""
    summary = Path("results/bassez2021/summaries/dataset_summary.csv")
    rows = list(csv.DictReader(open(summary)))
    return [f"{r['patient']}_{r['celltype']}" for r in rows]


def make_config_tag(imputation, sim, strat, walks, walk_length, aggregation):
    """Folder-safe deterministic tag."""
    agg = "perpat" if aggregation == "per_patient_embeddings" else "joint"
    return (f"{imputation}_{sim}_{strat}_w{walks}_k5_wl{walk_length}_{agg}")


def row_for(stage, group, imputation, sim, strat, walks, walk_length, aggregation,
            results_root="results/bassez2021"):
    method = IMP_TO_METHOD[imputation]
    group_dir = group_to_indir(group)
    cfg_tag = make_config_tag(imputation, sim, strat, walks, walk_length, aggregation)

    # Output paths (deterministic).
    rr = Path(results_root)
    model_dir = rr / "models" / group / cfg_tag
    model_path = model_dir / "gene_embeddings.model"
    # Graph cache is shared across walk_strategy/walks/aggregation -- key on (group, imp, sim).
    graph_cache_dir = rr / "graphs"
    auc_dir = rr / "auc" / group / cfg_tag
    mcc_dir = rr / "mcc" / group / cfg_tag

    return {
        "stage": stage,
        "group": group,
        "group_dir": group_dir,
        "imputation": imputation,
        "graph_method": method,
        "similarity": sim,
        "walk_strategy": strat,
        "walks": int(walks),
        "walk_length": int(walk_length),
        "aggregation_strategy": aggregation,
        "config_tag": cfg_tag,
        "in_root": "exports_bassez",
        "model_dir": str(model_dir),
        "model_path": str(model_path),
        "graph_cache_dir": str(graph_cache_dir),
        "auc_dir": str(auc_dir),
        "mcc_dir": str(mcc_dir),
    }


def build_pilot():
    rows = []
    for grp in PILOT_GROUPS:
        for imp, sim, strat in product(DEFAULT_IMP, DEFAULT_SIM, DEFAULT_STRAT):
            rows.append(row_for(
                "pilot", grp, imp, sim, strat,
                walks=DEFAULT_WALKS_STAGE12, walk_length=DEFAULT_WALK_LENGTH,
                aggregation="per_patient_embeddings"))
    return rows


def build_stage2(configs):
    """configs: list of (imputation, similarity, walk_strategy) survivors from pilot."""
    rows = []
    for grp in all_groups():
        for (imp, sim, strat) in configs:
            rows.append(row_for(
                "stage2", grp, imp, sim, strat,
                walks=DEFAULT_WALKS_STAGE12, walk_length=DEFAULT_WALK_LENGTH,
                aggregation="per_patient_embeddings"))
    return rows


def build_saturation(top_configs):
    """top_configs: list of (imputation, similarity, walk_strategy)."""
    rows = []
    for grp in all_groups():
        for (imp, sim, strat) in top_configs:
            for w in DEFAULT_SATURATION_WALKS:
                rows.append(row_for(
                    "saturation", grp, imp, sim, strat,
                    walks=w, walk_length=DEFAULT_WALK_LENGTH,
                    aggregation="per_patient_embeddings"))
    return rows


def build_joint(top_configs):
    """One row per (imputation, similarity, walk_strategy). Joint pools all groups."""
    rows = []
    for (imp, sim, strat) in top_configs:
        # Use group="ALL" for joint rows.
        method = IMP_TO_METHOD[imp]
        cfg_tag = make_config_tag(imp, sim, strat, DEFAULT_WALKS_STAGE12,
                                  DEFAULT_WALK_LENGTH, "joint_embeddings")
        rr = Path("results/bassez2021")
        rows.append({
            "stage": "joint",
            "group": "ALL",
            "group_dir": "",
            "imputation": imp,
            "graph_method": method,
            "similarity": sim,
            "walk_strategy": strat,
            "walks": DEFAULT_WALKS_STAGE12,
            "walk_length": DEFAULT_WALK_LENGTH,
            "aggregation_strategy": "joint_embeddings",
            "config_tag": cfg_tag,
            "in_root": "exports_bassez",
            "model_dir": str(rr / "models" / "ALL" / cfg_tag),
            "model_path": str(rr / "models" / "ALL" / cfg_tag / "gene_embeddings.model"),
            "graph_cache_dir": str(rr / "graphs"),
            "auc_dir": str(rr / "auc" / "ALL" / cfg_tag),
            "mcc_dir": str(rr / "mcc" / "ALL" / cfg_tag),
        })
    return rows


def parse_configs_arg(text):
    """Parse comma-separated 'imp:sim:strat' triples, e.g. 'raw:spearman:star,alra:cosine:bidirectional'."""
    out = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk: continue
        parts = chunk.split(":")
        if len(parts) != 3:
            raise ValueError(f"bad config triple: {chunk}")
        imp, sim, strat = parts
        if imp not in IMP_TO_METHOD: raise ValueError(f"bad imp: {imp}")
        if sim not in ("spearman","cosine"): raise ValueError(f"bad sim: {sim}")
        if strat not in ("star","bidirectional"): raise ValueError(f"bad strat: {strat}")
        out.append((imp, sim, strat))
    return out


def write_manifest(rows, out_path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["stage","group","group_dir","imputation","graph_method","similarity",
            "walk_strategy","walks","walk_length","aggregation_strategy",
            "config_tag","in_root","model_dir","model_path",
            "graph_cache_dir","auc_dir","mcc_dir"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader(); w.writerows(rows)
    print(f"[WRITE] {out}  rows={len(rows)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["pilot","stage2","saturation","joint"])
    p.add_argument("--out", required=True, help="Output TSV path")
    p.add_argument("--configs", default=None,
                   help="For stage2/saturation/joint: comma-sep 'imp:sim:strat' triples")
    args = p.parse_args()

    if args.stage == "pilot":
        rows = build_pilot()
    elif args.stage == "stage2":
        if not args.configs:
            sys.exit("stage2 requires --configs (e.g. raw:spearman:star,alra:cosine:bidirectional,...)")
        rows = build_stage2(parse_configs_arg(args.configs))
    elif args.stage == "saturation":
        if not args.configs:
            sys.exit("saturation requires --configs")
        rows = build_saturation(parse_configs_arg(args.configs))
    elif args.stage == "joint":
        if not args.configs:
            sys.exit("joint requires --configs")
        rows = build_joint(parse_configs_arg(args.configs))
    else:
        sys.exit("unreachable")

    write_manifest(rows, args.out)


if __name__ == "__main__":
    main()
