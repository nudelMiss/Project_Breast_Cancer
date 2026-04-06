#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import matplotlib.patches as mpatches
import umap


# -----------------------------
# Helpers
# -----------------------------
CELLTYPE_RE = re.compile(r"celltype=([^_]+)$")


def parse_group_name(group_name: str):
    # expected: patient=CIDxxxx__celltype=B-cell
    parts = group_name.split("__")
    celltype = None
    for p in parts:
        if p.startswith("celltype="):
            celltype = p.split("=", 1)[1]
    if celltype is None:
        celltype = "unknown"
    return celltype


def upper_triangle_flat(D: np.ndarray) -> np.ndarray:
    # flatten upper triangle (i<j)
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu]


def pairwise_distances_euclidean(X: np.ndarray) -> np.ndarray:
    # X: (m, d)
    # returns (m, m) Euclidean distance matrix
    X = X.astype(np.float32, copy=False)
    norms = np.sum(X * X, axis=1, keepdims=True)   # (m,1)
    G = X @ X.T                                    # (m,m)
    D2 = norms + norms.T - 2.0 * G                 # (m,m)
    np.maximum(D2, 0.0, out=D2)
    D = np.sqrt(D2).astype(np.float32, copy=False)
    return D


def pairwise_distances_between_rows(X: np.ndarray) -> np.ndarray:
    # X: (n, d) -> Euclidean distances between rows
    X = X.astype(np.float32, copy=False)
    norms = np.sum(X * X, axis=1, keepdims=True)
    G = X @ X.T
    D2 = norms + norms.T - 2.0 * G
    np.maximum(D2, 0.0, out=D2)
    D = np.sqrt(D2).astype(np.float64, copy=False)
    return D


def pcoa(D: np.ndarray, n_components: int = 2):
    # Classical MDS / PCoA from a distance matrix D (n,n)
    D = D.astype(np.float64, copy=False)
    n = D.shape[0]
    D2 = D * D
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    pos = eigvals > 1e-12
    eigvals = eigvals[pos]
    eigvecs = eigvecs[:, pos]
    k = min(n_components, eigvecs.shape[1])
    coords = eigvecs[:, :k] * np.sqrt(eigvals[:k])
    var_expl = eigvals[:k] / np.sum(eigvals) if eigvals.size else np.zeros(k)
    return coords, var_expl


def add_ellipse(ax, x, y, color, scale=2.5):
    if len(x) < 3:
        return
    pts = np.column_stack([x, y])
    mu = pts.mean(axis=0)
    cov = np.cov(pts.T)

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    width, height = 2 * scale * np.sqrt(np.maximum(vals, 1e-12))
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    ell = mpatches.Ellipse(
        mu,
        width,
        height,
        angle=angle,
        fill=False,
        lw=2,
        ls="--",
        edgecolor=color,
        alpha=0.9,
    )
    ax.add_patch(ell)


def plot_embedding(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_png: str,
    color_map: dict,
    default_color: str = "#7F7F7F",
):
    plt.figure(figsize=(11, 6.5))
    ax = plt.gca()

    uniq = sorted(df["celltype"].unique().tolist())

    for ct in uniq:
        sub = df[df["celltype"] == ct]
        c = color_map.get(ct, default_color)
        ax.scatter(sub[x_col], sub[y_col], s=60, alpha=0.85, label=ct, color=c)

    for ct in uniq:
        sub = df[df["celltype"] == ct]
        add_ellipse(
            ax,
            sub[x_col].to_numpy(),
            sub[y_col].to_numpy(),
            color_map.get(ct, default_color),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"[INFO] Wrote {out_png}", flush=True)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models_root",
        type=str,
        default="results/results_by_patient_celltype_5_walks",
        help="Folder that contains patient/celltype subfolders with model files inside",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="gene_embeddings.model",
        help="Model filename inside each variant dir",
    )
    ap.add_argument(
        "--sample_genes",
        type=int,
        default=500,
        help="Number of genes to sample (from intersection across available models)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out_prefix",
        type=str,
        default="results/PCoA/cosine_50_edges_5_walks/pcoa_walks5_sample500",
        help="Output prefix for csv/png. The parent folder will be created automatically.",
    )
    ap.add_argument(
        "--variant_label",
        type=str,
        default="cosine, 50 edges, 5 walks",
        help="Label to include in plot titles",
    )
    ap.add_argument(
        "--min_models",
        type=int,
        default=10,
        help="Refuse to run if fewer than this many models are found",
    )
    args = ap.parse_args()

    root = Path(args.models_root)
    if not root.exists():
        raise FileNotFoundError(f"models_root not found: {root}")

    # create output directory automatically
    out_prefix_path = Path(args.out_prefix)
    out_prefix_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect available models.
    # Expected layout:
    # root/
    #   patient=CIDxxxx__celltype=...
    #       sim=cosine__emode=.../
    #           gene_embeddings.model
    model_paths = sorted(root.glob(f"*/*/{args.pattern}"))
    if len(model_paths) < args.min_models:
        raise RuntimeError(
            f"Found only {len(model_paths)} models under {root}. Need >= {args.min_models}."
        )

    # group should be the patient/celltype folder, not the sim folder
    groups = [p.parent.parent.name for p in model_paths]
    celltypes = [parse_group_name(g) for g in groups]

    # Remove equivocal groups entirely
    filtered = [
        (p, g, ct)
        for p, g, ct in zip(model_paths, groups, celltypes)
        if ct != "equivocal"
    ]

    model_paths = [x[0] for x in filtered]
    groups = [x[1] for x in filtered]
    celltypes = [x[2] for x in filtered]

    if len(model_paths) < args.min_models:
        raise RuntimeError(
            f"Found only {len(model_paths)} non-equivocal models under {root}. Need >= {args.min_models}."
        )

    print(f"[INFO] Found {len(model_paths)} non-equivocal models", flush=True)

    # Load vocab keys (intersection)
    common = None
    for p in model_paths:
        m = Word2Vec.load(str(p))
        keys = set(m.wv.index_to_key)
        common = keys if common is None else (common & keys)

    common = sorted(common) if common else []
    print(f"[INFO] Common genes across ALL available models: {len(common)}", flush=True)
    if len(common) < 10:
        raise RuntimeError("Intersection across models is too small. Cannot compare embeddings robustly.")

    rng = np.random.default_rng(args.seed)
    n_sample = min(args.sample_genes, len(common))
    sampled = rng.choice(common, size=n_sample, replace=False).tolist()
    print(f"[INFO] Using sample_genes={n_sample}", flush=True)

    # Build signature vector for each model: all pairwise distances among sampled genes
    signatures = []
    for i, p in enumerate(model_paths):
        mobj = Word2Vec.load(str(p))
        X = np.vstack([mobj.wv[g] for g in sampled]).astype(np.float32, copy=False)
        Dg = pairwise_distances_euclidean(X)
        sig = upper_triangle_flat(Dg).astype(np.float32, copy=False)
        signatures.append(sig)
        if (i + 1) % 10 == 0 or (i + 1) == len(model_paths):
            print(f"[INFO] signatures: {i + 1}/{len(model_paths)}", flush=True)

    S = np.vstack(signatures)
    print(f"[INFO] Signature matrix: {S.shape}", flush=True)

    variant_label = args.variant_label
    color_map = {
        "Epithelial": "#D62728",   # red
        "T-cells": "#2CA02C",      # green
        "B-cell": "#1F77B4",       # blue
        "Myeloid": "#9467BD",      # purple
        "Fibroblasts": "#8C564B",  # brown
        "Endothelial": "#FF7F0E",  # orange
    }
    default_color = "#7F7F7F"

    # -----------------------------
    # UMAP
    # -----------------------------
    print("[INFO] Running UMAP...", flush=True)
    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=args.seed,
    )
    umap_coords = umap_model.fit_transform(S)

    df_umap = pd.DataFrame(
        {
            "group": groups,
            "celltype": celltypes,
            "UMAP1": umap_coords[:, 0],
            "UMAP2": umap_coords[:, 1],
        }
    )

    out_umap_csv = f"{args.out_prefix}_umap.csv"
    df_umap.to_csv(out_umap_csv, index=False)
    print(f"[INFO] Wrote {out_umap_csv}", flush=True)

    out_umap_png = f"{args.out_prefix}_umap.png"
    plot_embedding(
        df=df_umap,
        x_col="UMAP1",
        y_col="UMAP2",
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        title=f"UMAP of embedding signatures ({variant_label})",
        out_png=out_umap_png,
        color_map=color_map,
        default_color=default_color,
    )

    # -----------------------------
    # Distances between embeddings
    # -----------------------------
    D = pairwise_distances_between_rows(S)
    print("[INFO] Built embedding-distance matrix", flush=True)

    # -----------------------------
    # PCoA
    # -----------------------------
    coords, var_expl = pcoa(D, n_components=2)
    pc1_var = var_expl[0] if len(var_expl) > 0 else 0.0
    pc2_var = var_expl[1] if len(var_expl) > 1 else 0.0
    print(f"[INFO] PCoA variance explained: PC1={pc1_var:.3f}, PC2={pc2_var:.3f}", flush=True)

    df_pcoa = pd.DataFrame(
        {
            "group": groups,
            "celltype": celltypes,
            "PC1": coords[:, 0] if coords.shape[1] > 0 else np.zeros(len(groups)),
            "PC2": coords[:, 1] if coords.shape[1] > 1 else np.zeros(len(groups)),
        }
    )

    out_pcoa_csv = f"{args.out_prefix}_pcoa.csv"
    df_pcoa.to_csv(out_pcoa_csv, index=False)
    print(f"[INFO] Wrote {out_pcoa_csv}", flush=True)

    out_pcoa_png = f"{args.out_prefix}_pcoa.png"
    plot_embedding(
        df=df_pcoa,
        x_col="PC1",
        y_col="PC2",
        xlabel=f"PCoA 1 ({pc1_var * 100:.1f}%)",
        ylabel=f"PCoA 2 ({pc2_var * 100:.1f}%)",
        title=f"PCoA of embedding signatures ({variant_label})",
        out_png=out_pcoa_png,
        color_map=color_map,
        default_color=default_color,
    )

    # -----------------------------
    # QC on embedding distance matrix
    # -----------------------------
    cell = np.array(celltypes)
    same = (cell[:, None] == cell[None, :])
    diag_mask = np.eye(D.shape[0], dtype=bool)
    np.fill_diagonal(same, False)

    within = D[same]
    between = D[(~same) & (~diag_mask)]

    if within.size > 0:
        print(f"[QC] mean within-celltype distance:  {within.mean():.4f}")
    else:
        print("[QC] mean within-celltype distance:  NA (no within-celltype pairs)")

    if between.size > 0:
        print(f"[QC] mean between-celltype distance: {between.mean():.4f}")
    else:
        print("[QC] mean between-celltype distance: NA (no between-celltype pairs)")

    if within.size > 0 and between.size > 0:
        score = (between.mean() - within.mean()) / (between.std() + 1e-12)
        print(f"[QC] separation z-score (bigger=better): {score:.3f}", flush=True)
    else:
        print("[QC] separation z-score (bigger=better): NA", flush=True)


if __name__ == "__main__":
    main()