"""Build the supervisor diagnostic figure + table."""
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROOT = Path("/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast")
DIAG = PROOT / "results/bassez2021/supervisor_diagnostic"
df  = pd.read_csv(DIAG / "all_results.csv")
bl  = pd.read_csv(DIAG / "baseline/raw_cosine_baseline.csv")

# Raw cosine ceiling — mean across the 6 pilot groups
COS_MEAN   = float(bl["mean_auc"].mean())
COS_WT     = float(bl["weighted_mean_auc"].mean())
GROUPS     = sorted(df["group_tag"].unique())
COLORS     = plt.get_cmap("tab10").colors

# --- collect curves -----------------------------------------------------------
def agg(family, x_col, fixed_col=None, fixed_val=None):
    sub = df[df["family"] == family].copy()
    if fixed_col is not None:
        sub = sub[sub[fixed_col] == fixed_val]
    return (sub.groupby(x_col)
              .agg(mean_auc=("mean_auc","mean"),
                   wt=("weighted_mean_auc","mean"),
                   sem=("mean_auc", lambda s: s.std(ddof=1)/np.sqrt(len(s))))
              .reset_index()
              .sort_values(x_col))

star_curve  = agg("star_saturation",  "walks")
bidir_curve = agg("bidir_saturation", "walks")
topk_curve  = agg("topk_sweep",       "k_nearest")

# --- figure -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# ---- Panel A: saturation curves (star + bidir) on mean AUC ------------------
ax = axes[0]
# bidir
ax.errorbar(bidir_curve["walks"], bidir_curve["mean_auc"], yerr=bidir_curve["sem"],
            marker="o", linewidth=2.2, markersize=8, color="#3366cc",
            label="bidirectional walk", capsize=3, zorder=3)
# star
ax.errorbar(star_curve["walks"], star_curve["mean_auc"], yerr=star_curve["sem"],
            marker="s", linewidth=2.2, markersize=8, color="#cc6633",
            label="star walk (supervisor's test)", capsize=3, zorder=3)
# raw cosine ceiling
ax.axhline(COS_MEAN, ls="--", color="#1a7a1a", linewidth=2.0,
           label=f"raw cosine (no W2V) = {COS_MEAN:.3f}", zorder=2)
ax.axhline(0.5, ls=":", color="gray", linewidth=1.0, alpha=0.6, label="random = 0.5", zorder=1)
ax.set_xscale("log")
ax.set_xlabel("walks per gene", fontsize=12)
ax.set_ylabel("mean CORUM AUC (avg over 6 groups)", fontsize=12)
ax.set_title("A. Saturation curves: both star and bidir decline with more walks",
             fontsize=12, fontweight="bold")
ax.set_xticks([1, 5, 10, 50, 100])
ax.set_xticklabels([1, 5, 10, 50, 100])
ax.set_ylim(0.48, 0.74)
ax.grid(alpha=0.3)
ax.legend(loc="lower left", fontsize=9, framealpha=0.95)

# ---- Panel B: top-k sweep at walks=1 ----------------------------------------
ax = axes[1]
ax.errorbar(topk_curve["k_nearest"], topk_curve["mean_auc"], yerr=topk_curve["sem"],
            marker="^", linewidth=2.2, markersize=9, color="#993399",
            label="bidir walks=1, varying top-k", capsize=3, zorder=3)
ax.axhline(COS_MEAN, ls="--", color="#1a7a1a", linewidth=2.0,
           label=f"raw cosine (no W2V) = {COS_MEAN:.3f}", zorder=2)
ax.axhline(0.5, ls=":", color="gray", linewidth=1.0, alpha=0.6, label="random = 0.5", zorder=1)
ax.set_xscale("log")
ax.set_xlabel("top-k edges per gene in graph", fontsize=12)
ax.set_ylabel("mean CORUM AUC (avg over 6 groups)", fontsize=12)
ax.set_title("B. Top-k sweep at walks=1: 20× more edges → essentially no change",
             fontsize=12, fontweight="bold")
ax.set_xticks([5, 10, 25, 50, 100])
ax.set_xticklabels([5, 10, 25, 50, 100])
ax.set_ylim(0.48, 0.74)
ax.grid(alpha=0.3)
ax.legend(loc="lower left", fontsize=9, framealpha=0.95)

fig.suptitle("Supervisor diagnostic: where is the CORUM signal being lost?\n"
             "(6 pilot groups; raw cosine + bidirectional/star walks + W2V; mean AUC)",
             fontsize=13, fontweight="bold", y=1.00)
plt.tight_layout()

png = DIAG / "supervisor_diagnostic.png"
pdf = DIAG / "supervisor_diagnostic.pdf"
fig.savefig(png, dpi=200, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print(f"saved {png}")
print(f"saved {pdf}")

# --- table ---------------------------------------------------------------------
print("\n=== TABLE: mean AUC by configuration (avg ± sem across 6 pilot groups) ===")
def fmt(row):
    return f"{row['mean_auc']:.3f} ± {row['sem']:.3f}"

print(f"\nRaw cosine ceiling (no W2V):  {COS_MEAN:.3f}  (weighted: {COS_WT:.3f})")
print("\nW2V — Bidirectional walks, k=5:")
for _, r in bidir_curve.iterrows():
    print(f"  walks={int(r['walks']):4d}, k=5:    mean AUC = {fmt(r)}  weighted = {r['wt']:.3f}")
print("\nW2V — Star walks, k=5:")
for _, r in star_curve.iterrows():
    print(f"  walks={int(r['walks']):4d}, k=5:    mean AUC = {fmt(r)}  weighted = {r['wt']:.3f}")
print("\nW2V — Top-k sweep at walks=1, bidirectional:")
for _, r in topk_curve.iterrows():
    print(f"  walks=1, k={int(r['k_nearest']):3d}:     mean AUC = {fmt(r)}  weighted = {r['wt']:.3f}")

# save the table as CSV too
table_rows = []
table_rows.append({"config": "raw cosine (no W2V)", "x":"-", "mean_auc": COS_MEAN, "sem": np.nan, "weighted_mean_auc": COS_WT})
for _, r in bidir_curve.iterrows():
    table_rows.append({"config": "W2V bidir k=5", "x": f"walks={int(r['walks'])}",
                       "mean_auc": r["mean_auc"], "sem": r["sem"], "weighted_mean_auc": r["wt"]})
for _, r in star_curve.iterrows():
    table_rows.append({"config": "W2V star k=5", "x": f"walks={int(r['walks'])}",
                       "mean_auc": r["mean_auc"], "sem": r["sem"], "weighted_mean_auc": r["wt"]})
for _, r in topk_curve.iterrows():
    table_rows.append({"config": "W2V bidir walks=1", "x": f"k={int(r['k_nearest'])}",
                       "mean_auc": r["mean_auc"], "sem": r["sem"], "weighted_mean_auc": r["wt"]})
pd.DataFrame(table_rows).to_csv(DIAG / "diagnostic_table.csv", index=False)
print(f"\nsaved {DIAG / 'diagnostic_table.csv'}")
