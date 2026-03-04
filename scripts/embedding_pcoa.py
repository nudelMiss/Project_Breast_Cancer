#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec

import matplotlib.patches as mpatches

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
    # Use squared distance trick: ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    X = X.astype(np.float32, copy=False)
    norms = np.sum(X * X, axis=1, keepdims=True)  # (m,1)
    G = X @ X.T                                    # (m,m)
    D2 = norms + norms.T - 2.0 * G                 # (m,m)
    np.maximum(D2, 0.0, out=D2)
    D = np.sqrt(D2, dtype=np.float32)
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


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_root", type=str, default="results_by_patient_celltype_1000_walks",
                    help="Folder that contains group subfolders with *.model inside")
    ap.add_argument("--pattern", type=str, default="gene_embeddings_w1000_weighted.model",
                    help="Model filename inside each group dir")
    ap.add_argument("--sample_genes", type=int, default=500,
                    help="Number of genes to sample (from intersection across available models)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_prefix", type=str, default="pcoa_w2v_compare",
                    help="Output prefix for csv/png")
    ap.add_argument("--min_models", type=int, default=10,
                    help="Refuse to run if fewer than this many models are found")
    args = ap.parse_args()

    root = Path(args.models_root)
    if not root.exists():
        raise FileNotFoundError(f"models_root not found: {root}")

    # Collect available models
    model_paths = sorted(root.glob(f"*/{args.pattern}"))
    if len(model_paths) < args.min_models:
        raise RuntimeError(f"Found only {len(model_paths)} models under {root}. Need >= {args.min_models}.")

    groups = [p.parent.name for p in model_paths]
    celltypes = [parse_group_name(g) for g in groups]

    print(f"[INFO] Found {len(model_paths)} models", flush=True)

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
    m = min(args.sample_genes, len(common))
    sampled = rng.choice(common, size=m, replace=False).tolist()
    print(f"[INFO] Using sample_genes={m}", flush=True)

    # Build signature vector for each model: all pairwise distances among sampled genes
    signatures = []
    for i, p in enumerate(model_paths):
        mobj = Word2Vec.load(str(p))
        X = np.vstack([mobj.wv[g] for g in sampled]).astype(np.float32, copy=False)  # (m, dim)
        Dg = pairwise_distances_euclidean(X)                                         # (m, m)
        sig = upper_triangle_flat(Dg).astype(np.float32, copy=False)                 # (m*(m-1)/2,)
        signatures.append(sig)
        if (i + 1) % 10 == 0 or (i + 1) == len(model_paths):
            print(f"[INFO] signatures: {i+1}/{len(model_paths)}", flush=True)

    S = np.vstack(signatures)  # (n_models, sig_len)
    print(f"[INFO] Signature matrix: {S.shape}", flush=True)

    # Distances between embeddings (signature vectors)
    # Euclidean between signatures
    norms = np.sum(S * S, axis=1, keepdims=True)
    G = S @ S.T
    D2 = norms + norms.T - 2.0 * G
    np.maximum(D2, 0.0, out=D2)
    D = np.sqrt(D2, dtype=np.float64)
    print("[INFO] Built embedding-distance matrix", flush=True)

    # PCoA
    coords, var_expl = pcoa(D, n_components=2)
    print(f"[INFO] PCoA variance explained: PC1={var_expl[0]:.3f}, PC2={var_expl[1]:.3f}", flush=True)

    # Save CSV
    df = pd.DataFrame({
        "group": groups,
        "celltype": celltypes,
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
    })

        # ---- Sanity: are samples closer to same celltype than other celltypes? ----
    # Use the embedding distance matrix D (between models).
    cell = np.array(celltypes)
    same = (cell[:, None] == cell[None, :])

    # ignore diagonal
    np.fill_diagonal(same, False)

    within = D[same]
    between = D[~same & ~np.eye(D.shape[0], dtype=bool)]

    print(f"[QC] mean within-celltype distance:  {within.mean():.4f}")
    print(f"[QC] mean between-celltype distance: {between.mean():.4f}")

    # simple separation score (positive is good)
    score = (between.mean() - within.mean()) / (between.std() + 1e-12)
    print(f"[QC] separation z-score (bigger=better): {score:.3f}", flush=True)

    out_csv = f"{args.out_prefix}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote {out_csv}", flush=True)

    plt.figure(figsize=(11, 6.5))

    uniq = sorted(df["celltype"].unique().tolist())
    cmap = plt.get_cmap("tab10")
    color_map = {ct: cmap(i % 10) for i, ct in enumerate(uniq)}

    # scatter
    for ct in uniq:
        sub = df[df["celltype"] == ct]
        plt.scatter(sub["PC1"], sub["PC2"], s=60, alpha=0.85, label=ct, color=color_map[ct])

    # ellipse helper (1-sigma like visual boundary)
    def add_ellipse(x, y, color):
        if len(x) < 3:
            return
        pts = np.column_stack([x, y])
        mu = pts.mean(axis=0)
        cov = np.cov(pts.T)
        # eigen-decomposition
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        # width/height = 2*sqrt(eigvals) * scale
        scale = 2.5  # controls ellipse size (2.0-3.0 usually nice)
        width, height = 2 * scale * np.sqrt(np.maximum(vals, 1e-12))
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        ell = mpatches.Ellipse(mu, width, height, angle=angle,
                            fill=False, lw=2, ls="--", edgecolor=color, alpha=0.9)
        plt.gca().add_patch(ell)

    # add ellipses per cell type
    for ct in uniq:
        sub = df[df["celltype"] == ct]
        add_ellipse(sub["PC1"].to_numpy(), sub["PC2"].to_numpy(), color_map[ct])

    plt.xlabel(f"PCoA 1 ({var_expl[0]*100:.1f}%)")
    plt.ylabel(f"PCoA 2 ({var_expl[1]*100:.1f}%)")
    plt.title("PCoA of embedding signatures (pairwise gene distances)")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()

    out_png = f"{args.out_prefix}.png"
    plt.savefig(out_png, dpi=220)
    print(f"[INFO] Wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
