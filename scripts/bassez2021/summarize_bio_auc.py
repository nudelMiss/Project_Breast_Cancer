#!/usr/bin/env python3
"""Slide-ready summary of post-hoc bio-AUC results.
Reads results/bassez2021/bio_auc/bio_auc_collected.csv and writes:
  - bio_auc_pivot_by_group.csv  : every (group,config) x module
  - bio_auc_saturation_mean.csv : pilots-mean per config (the saturation story)
"""
from pathlib import Path
import pandas as pd

ROOT = Path("/groups/ofircohen-group/users/michalnu_yuvat/project_breast")
BIO = ROOT / "results/bassez2021/bio_auc"
df = pd.read_csv(BIO / "bio_auc_collected.csv")
mods = ["auc_S_phase", "auc_G2M", "auc_IFN_alpha", "auc_IFN_gamma"]

keep = ["group", "config_tag", "mean_bio_auc"] + mods
extra = [c for c in ["corum_mean_auc", "corum_weighted_auc"] if c in df.columns]
piv = df[keep + extra].sort_values(["group", "config_tag"]).round(3)
piv.to_csv(BIO / "bio_auc_pivot_by_group.csv", index=False)

# Pilots-mean saturation view (the 3 pilots only)
pilots = ["BIOKEY_18_T_cell", "BIOKEY_30_Malignant", "BIOKEY_4_B_cell"]
sub = df[df.group.isin(pilots)].copy()
agg = (sub.groupby("config_tag")[["mean_bio_auc"] + mods + extra]
          .mean().round(3).sort_values("mean_bio_auc", ascending=False))
agg.to_csv(BIO / "bio_auc_saturation_mean.csv")

print("=== Pilots-mean by config (sorted by mean_bio_auc) ===")
print(agg.to_string())
print(f"\nWrote: {BIO/'bio_auc_pivot_by_group.csv'}")
print(f"Wrote: {BIO/'bio_auc_saturation_mean.csv'}")
