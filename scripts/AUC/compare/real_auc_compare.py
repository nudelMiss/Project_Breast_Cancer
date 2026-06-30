#!/usr/bin/env python
"""
Real-embedding AUC comparison: OLD vs NEW reporting/sampling conventions.

OLD: cosine sim + random negatives outside complex + mean over ALL complexes.
NEW: cosine sim + other-CORUM negatives only           + mean over q<0.05 only.

The similarity metric (cosine) is held constant -- only the negative pool
and the reporting aggregator change.

Memory-safe: builds the cosine matrix only on the CORUM-in-embedding subset
(~3k genes -> ~70 MB). OLD-method random negatives are sampled on the fly
from the full embedding (no full N x N matrix is ever materialized).

Usage (single model):
  python real_auc_compare.py \
    --model results/.../gene_embeddings.model \
    --corum resources/corum_core_complexes.tsv \
    --out_dir results/bassez2021/auc_compare/joint_raw_w1 \
    --label JOINT_raw_cosine_bidir_w1

Add --make_figure to also save PNG/PDF.
"""

import argparse, os, json
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #

def load_embedding(model_path):
    """Return (gene_names, vectors_float64) and an L2-normalized copy."""
    m = Word2Vec.load(model_path)
    genes = list(m.wv.index_to_key)
    E = np.asarray(m.wv.vectors, dtype=np.float64)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    En = E / norms
    return genes, E, En


def load_corum(tsv_path, min_complex_size=3):
    """Parse a 2-col CORUM TSV. Returns dict[complex_id -> list[gene]]."""
    df = pd.read_csv(tsv_path, sep="\t")
    out = {}
    for cid, sub in df.groupby("complex_id"):
        g = sorted(set(sub["gene"].astype(str)))
        if len(g) >= min_complex_size:
            out[str(cid)] = g
    return out


# --------------------------------------------------------------------------- #
# OLD scorer: random negatives anywhere outside the complex
# --------------------------------------------------------------------------- #

def score_old(En, gene_names, complexes, n_neg_floor=200,
              min_complex_size=3, seed=42):
    """
    Per-complex AUC with random (inside, outside-complex) negatives.
    n_neg = max(n_neg_floor, n_positive). Cosine similarities computed on
    the fly via inner products on L2-normalized vectors -- no full N x N
    matrix is allocated.
    """
    rng = np.random.default_rng(seed)
    gene2idx = {g: i for i, g in enumerate(gene_names)}
    gene_set = set(gene_names)
    n_total = len(gene_names)
    rows, all_within, all_between = [], [], []

    for cid, glist in complexes.items():
        in_genes = [g for g in glist if g in gene_set]
        if len(in_genes) < min_complex_size:
            continue
        in_idx = np.array([gene2idx[g] for g in in_genes])
        E_in = En[in_idx]                          # (k, d)

        # Positives: all within-complex pairs (upper triangle of k x k)
        Cw = E_in @ E_in.T                         # cosine, since En is unit-norm
        iu = np.triu_indices_from(Cw, k=1)
        within = Cw[iu]
        within = within[np.isfinite(within)]
        if len(within) == 0:
            continue

        # Negatives: (inside, anywhere-outside-the-complex) random pairs
        outside_pool = np.setdiff1d(np.arange(n_total), in_idx,
                                    assume_unique=False)
        if len(outside_pool) == 0:
            continue
        n_neg = max(n_neg_floor, len(within))
        a = rng.choice(in_idx, size=n_neg, replace=True)
        b = rng.choice(outside_pool, size=n_neg, replace=True)
        between = np.einsum("ij,ij->i", En[a], En[b])
        between = between[np.isfinite(between)]
        if len(between) == 0:
            continue

        y = np.concatenate([np.ones_like(within), np.zeros_like(between)])
        s = np.concatenate([within, between])
        auc = roc_auc_score(y, s)
        _, p = mannwhitneyu(within, between, alternative="greater")

        rows.append({"complex": cid, "n_genes": len(in_genes),
                     "n_within": int(len(within)), "n_between": int(len(between)),
                     "auc": auc, "p_value": p})
        all_within.append(within); all_between.append(between)

    return _finalize(rows, all_within, all_between)


# --------------------------------------------------------------------------- #
# NEW scorer: negatives restricted to other CORUM genes
# --------------------------------------------------------------------------- #

def score_new(En, gene_names, complexes, min_complex_size=3):
    """
    Per-complex AUC with negatives = (complex_gene, other_CORUM_gene). The
    cosine matrix is materialized only on the CORUM-in-embedding subset.
    """
    gene2idx = {g: i for i, g in enumerate(gene_names)}
    gene_set = set(gene_names)

    # Filter complexes to those with enough overlap with the embedding
    eligible = {}
    for cid, glist in complexes.items():
        kept = [g for g in glist if g in gene_set]
        if len(kept) >= min_complex_size:
            eligible[cid] = kept

    # CORUM universe = unique genes appearing in any eligible complex
    corum_universe = sorted({g for gl in eligible.values() for g in gl})
    corum_idx_global = np.array([gene2idx[g] for g in corum_universe])
    gene2local = {g: k for k, g in enumerate(corum_universe)}

    # Subset cosine matrix on CORUM universe only
    En_sub = En[corum_idx_global]
    S = En_sub @ En_sub.T
    np.fill_diagonal(S, np.nan)

    rows, all_within, all_between = [], [], []

    for cid, in_genes in eligible.items():
        in_idx = np.array([gene2local[g] for g in in_genes])
        all_local = np.arange(len(corum_universe))
        out_idx = np.setdiff1d(all_local, in_idx, assume_unique=False)
        if len(out_idx) == 0:
            continue

        # Positives: within-complex upper triangle
        sub = S[np.ix_(in_idx, in_idx)]
        iu = np.triu_indices_from(sub, k=1)
        within = sub[iu]
        within = within[np.isfinite(within)]

        # Negatives: full rectangle (in-complex) x (other CORUM)
        between = S[np.ix_(in_idx, out_idx)].ravel()
        between = between[np.isfinite(between)]
        if len(within) == 0 or len(between) == 0:
            continue

        y = np.concatenate([np.ones_like(within), np.zeros_like(between)])
        s = np.concatenate([within, between])
        auc = roc_auc_score(y, s)
        _, p = mannwhitneyu(within, between, alternative="greater")

        rows.append({"complex": cid, "n_genes": len(in_genes),
                     "n_within": int(len(within)), "n_between": int(len(between)),
                     "auc": auc, "p_value": p})
        all_within.append(within); all_between.append(between)

    return _finalize(rows, all_within, all_between)


# --------------------------------------------------------------------------- #
# Shared finalization: BH correction per embedding, summary numbers
# --------------------------------------------------------------------------- #

def _finalize(rows, all_within, all_between):
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df, {}
    df["p_adj"] = multipletests(df["p_value"], method="fdr_bh")[1]
    sig = df[df["p_adj"] < 0.05]
    summary = {
        "n_complexes_evaluated": int(len(df)),
        "n_significant_q05": int(len(sig)),
        "frac_significant_q05": float(len(sig) / len(df)),
        "mean_auc_all": float(df["auc"].mean()),
        "median_auc_all": float(df["auc"].median()),
        "mean_auc_q05": float(sig["auc"].mean()) if len(sig) else float("nan"),
        "median_auc_q05":
            float(sig["auc"].median()) if len(sig) else float("nan"),
    }
    return df, summary


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #

def make_figure(df_old, df_new, s_old, s_new, label, out_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: 3 bars -- the decomposition
    ax = axes[0]
    bars_labels = ["OLD\n(random neg,\nmean over ALL)",
                   "intermediate\n(CORUM neg,\nmean over ALL)",
                   "NEW\n(CORUM neg,\nmean over q<0.05)"]
    bars_vals = [s_old["mean_auc_all"], s_new["mean_auc_all"],
                 s_new["mean_auc_q05"]]
    colors = ["#888888", "#4C72B0", "#C44E52"]
    bars = ax.bar(bars_labels, bars_vals, color=colors, edgecolor="black")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_ylabel("Reported mean AUC", fontsize=13)
    ax.set_title(f"Headline AUC: {label}", fontsize=13, pad=10)
    ax.set_ylim(0.45, max(max(bars_vals) + 0.08, 0.75))
    for rect, v in zip(bars, bars_vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel B: per-complex scatter
    ax = axes[1]
    merged = df_old[["complex", "auc"]].rename(columns={"auc": "auc_old"}).merge(
        df_new[["complex", "auc", "p_adj"]].rename(
            columns={"auc": "auc_new", "p_adj": "q_new"}),
        on="complex"
    )
    sig_mask = merged["q_new"] < 0.05
    ax.scatter(merged.loc[~sig_mask, "auc_old"],
               merged.loc[~sig_mask, "auc_new"],
               s=25, c="#888888", edgecolor="black", linewidth=0.3, alpha=0.6,
               label=f"Not significant  n={(~sig_mask).sum()}")
    ax.scatter(merged.loc[sig_mask, "auc_old"],
               merged.loc[sig_mask, "auc_new"],
               s=25, c="#C44E52", edgecolor="black", linewidth=0.3, alpha=0.7,
               label=f"Significant (q<0.05)  n={sig_mask.sum()}")
    lo = min(merged["auc_old"].min(), merged["auc_new"].min()) - 0.02
    hi = max(merged["auc_old"].max(), merged["auc_new"].max()) + 0.02
    lims = [lo, hi]
    ax.plot(lims, lims, color="grey", linestyle="--", linewidth=1, label="y = x")
    ax.axvline(0.5, color="lightgrey", linewidth=1)
    ax.axhline(0.5, color="lightgrey", linewidth=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Per-complex AUC, OLD (random negatives)", fontsize=12)
    ax.set_ylabel("Per-complex AUC, NEW (CORUM-only negatives)", fontsize=12)
    ax.set_title(f"Per-complex agreement ({merged.shape[0]} complexes)",
                 fontsize=13, pad=10)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.95)
    ax.grid(linestyle=":", alpha=0.5)

    fig.suptitle(
        f"Real embedding -- AUC pipeline comparison\n{label}",
        fontsize=14, y=1.00,
    )
    fig.tight_layout()
    png = os.path.join(out_dir, "auc_comparison.png")
    pdf = os.path.join(out_dir, "auc_comparison.pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf,            bbox_inches="tight")
    plt.close(fig)
    print(f"  figure: {png}")
    print(f"  figure: {pdf}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corum", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--min_complex_size", type=int, default=3)
    ap.add_argument("--make_figure", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    print(f"Loading embedding: {a.model}")
    genes, E, En = load_embedding(a.model)
    print(f"  genes={len(genes)}  dims={E.shape[1]}")

    print(f"Loading CORUM (min_complex_size={a.min_complex_size})")
    complexes = load_corum(a.corum, min_complex_size=a.min_complex_size)
    print(f"  complexes: {len(complexes)}")

    print("Scoring OLD (random negatives, mean over ALL) ...")
    df_old, s_old = score_old(En, genes, complexes,
                              min_complex_size=a.min_complex_size)
    print(f"  evaluated={s_old['n_complexes_evaluated']}, "
          f"mean_auc_all={s_old['mean_auc_all']:.4f}")

    print("Scoring NEW (CORUM negatives, mean over q<0.05) ...")
    df_new, s_new = score_new(En, genes, complexes,
                              min_complex_size=a.min_complex_size)
    print(f"  evaluated={s_new['n_complexes_evaluated']}, "
          f"q<0.05={s_new['n_significant_q05']}, "
          f"mean_auc_q05={s_new['mean_auc_q05']:.4f}")

    df_old.to_csv(os.path.join(a.out_dir, "per_complex_old.csv"), index=False)
    df_new.to_csv(os.path.join(a.out_dir, "per_complex_new.csv"), index=False)
    out_summary = {
        "label": a.label, "model": a.model,
        "OLD": s_old, "NEW": s_new,
        "delta_headline (NEW_q05 - OLD_all)":
            s_new["mean_auc_q05"] - s_old["mean_auc_all"],
    }
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(out_summary, f, indent=2)

    print("\n=== HEADLINE COMPARISON ===")
    print(f"  OLD mean_auc_all : {s_old['mean_auc_all']:.4f}")
    print(f"  NEW mean_auc_q05 : {s_new['mean_auc_q05']:.4f}")
    print(f"  delta            : {out_summary['delta_headline (NEW_q05 - OLD_all)']:+.4f}")

    if a.make_figure:
        make_figure(df_old, df_new, s_old, s_new, a.label, a.out_dir)


if __name__ == "__main__":
    main()
