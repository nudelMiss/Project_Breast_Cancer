"""Aggregate diagnostic experiment results into a single CSV for plotting."""
import csv, re
import pandas as pd
from pathlib import Path

PROOT = Path("/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast")
DIAG_AUC = PROOT / "results/bassez2021/supervisor_diagnostic/auc"
EXISTING_AUC = PROOT / "results/bassez2021/auc"
OUT = PROOT / "results/bassez2021/supervisor_diagnostic/all_results.csv"

GROUPS = [
    "BIOKEY_10_Myeloid", "BIOKEY_13_Endothelial", "BIOKEY_18_T_cell",
    "BIOKEY_30_Malignant", "BIOKEY_3_Fibroblast", "BIOKEY_4_B_cell",
]

def read_summary(p):
    """Read mean/weighted_mean AUC from a corpus summary CSV."""
    df = pd.read_csv(p)
    return float(df.iloc[0]["mean_auc"]), float(df.iloc[0]["weighted_mean_auc"])

rows = []

# A) Diagnostic runs: star walks + topk sweep
for grp in GROUPS:
    for sub in (DIAG_AUC / grp).iterdir() if (DIAG_AUC / grp).is_dir() else []:
        f = sub / "corum_auc_summary.csv"
        if not f.exists(): continue
        label = sub.name  # e.g. "star_w5" or "bidir_w1_k50"
        mean_auc, wt = read_summary(f)
        if label.startswith("star_w"):
            w = int(label.split("_w")[1])
            family = "star_saturation"; walks = w; k = 5; strategy = "star"
        elif label.startswith("bidir_w1_k"):
            k = int(label.split("_k")[1])
            family = "topk_sweep"; walks = 1; strategy = "bidirectional"
        else:
            continue
        rows.append(dict(family=family, group_tag=grp, strategy=strategy,
                         walks=walks, k_nearest=k, mean_auc=mean_auc,
                         weighted_mean_auc=wt, source="diagnostic"))

# B) Existing runs we can re-use:
#    raw_cosine_bidirectional w in {1,5,10,50,100} k=5  -> walks saturation (bidir)
#    raw_cosine_star_w100_k5 -> the "star w=100" point (anchors the star saturation curve)
existing_configs = [
    ("raw_cosine_bidirectional_w1_k5_wl3_perpat",   "topk_sweep",       "bidirectional", 1,   5),
    ("raw_cosine_bidirectional_w5_k5_wl3_perpat",   "bidir_saturation", "bidirectional", 5,   5),
    ("raw_cosine_bidirectional_w10_k5_wl3_perpat",  "bidir_saturation", "bidirectional", 10,  5),
    ("raw_cosine_bidirectional_w50_k5_wl3_perpat",  "bidir_saturation", "bidirectional", 50,  5),
    ("raw_cosine_bidirectional_w100_k5_wl3_perpat", "bidir_saturation", "bidirectional", 100, 5),
    ("raw_cosine_star_w100_k5_wl3_perpat",          "star_saturation",  "star",          100, 5),
]
# Note: bidir_w1_k5 is the shared anchor for BOTH the bidir_saturation curve and the topk_sweep
# So we add it under topk_sweep (it's at k=5, walks=1) AND duplicate it as bidir_saturation walks=1.
extra_w1_anchor = [("raw_cosine_bidirectional_w1_k5_wl3_perpat", "bidir_saturation", "bidirectional", 1, 5)]
for grp in GROUPS:
    base = EXISTING_AUC / grp
    for cfg_dir, family, strategy, w, k in existing_configs + extra_w1_anchor:
        f = base / cfg_dir / "corum_auc_summary.csv"
        if not f.exists(): continue
        mean_auc, wt = read_summary(f)
        rows.append(dict(family=family, group_tag=grp, strategy=strategy,
                         walks=w, k_nearest=k, mean_auc=mean_auc,
                         weighted_mean_auc=wt, source="existing"))

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"wrote {len(df)} rows to {OUT}")
print("\n=== Summary by (family, walks, k_nearest) ===")
print(df.groupby(["family","walks","k_nearest"])
        .agg(n=("group_tag","count"),
             mean_AUC=("mean_auc","mean"),
             wt_AUC=("weighted_mean_auc","mean"))
        .round(4))
