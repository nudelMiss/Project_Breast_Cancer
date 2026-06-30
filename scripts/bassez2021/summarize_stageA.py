#!/usr/bin/env python3
"""
Stage A screening summary: does a new association metric beat matched cosine on bio?
Reads results/bassez2021/stageA/bio_auc/bio_auc_collected.csv (config_tag = {assoc}_{strat}_w10_k{k}_var75_hvg2000).
Writes stageA_comparison.csv (per group×config) and prints the assoc×strat×k pivot
(mean over groups) + the STOP/GO check vs the matched cosine baseline per (strat,k) cell.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/groups/ofircohen-group/users/michalnu_yuvat/project_breast")
BIO = ROOT / "results/bassez2021/stageA/bio_auc"
df = pd.read_csv(BIO / "bio_auc_collected.csv")

pat = re.compile(r"^(?P<assoc>cosine|ids|propr|cscore)_(?P<strat>star|bidirectional)_w(?P<w>\d+)_k(?P<k>\d+)_")
m = df["config_tag"].str.extract(pat)
df = pd.concat([df, m], axis=1)
df["k"] = df["k"].astype(int)

df.sort_values(["group", "assoc", "strat", "k"]).to_csv(BIO / "stageA_comparison.csv", index=False)

print("=== mean over groups: bio_mean_auc by assoc x (strat,k) ===")
piv = df.pivot_table(index="assoc", columns=["strat", "k"], values="mean_bio_auc", aggfunc="mean").round(3)
print(piv.to_string())

print("\n=== per-module mean over groups (assoc x strat, k=50) ===")
k50 = df[df.k == 50]
for mod in ["auc_S_phase", "auc_G2M", "auc_IFN_alpha", "auc_IFN_gamma"]:
    if mod in k50.columns:
        print(f"\n{mod}:")
        print(k50.pivot_table(index="assoc", columns="strat", values=mod, aggfunc="mean").round(3).to_string())

print("\n=== STOP/GO: per (strat,k) cell, does each assoc beat cosine on bio_mean, and on how many groups? ===")
for (strat, k), g in df.groupby(["strat", "k"]):
    cos = g[g.assoc == "cosine"].set_index("group")["mean_bio_auc"]
    for assoc in ["ids", "propr", "cscore"]:
        a = g[g.assoc == assoc].set_index("group")["mean_bio_auc"]
        if a.empty:
            continue
        common = cos.index.intersection(a.index)
        if len(common) == 0:
            continue
        wins = int((a[common] > cos[common]).sum())
        delta = float((a[common] - cos[common]).mean())
        verdict = "GO" if (delta > 0 and wins >= 2) else "no"
        print(f"  {strat:13s} k={k:<2d} {assoc:6s}: mean_delta_vs_cosine={delta:+.3f}  wins={wins}/{len(common)}  -> {verdict}")
print(f"\nWrote {BIO/'stageA_comparison.csv'}")
