#!/usr/bin/env python3
"""Saturation curves: x=walks, y=mean/median metric, one line per (imp,sim,strat) config."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

METRICS = [("mean_auc","Mean CORUM AUC",0.5),
           ("mcc_best","Best CORUM MCC",0.0)]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--prefix", default="saturation")
    p.add_argument("--reduce", choices=["mean","median"], default="mean")
    return p.parse_args()

def main():
    a = parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.in_csv).dropna(subset=["walks"])
    df["walks"] = df["walks"].astype(int)
    keys = ["imputation","similarity","walk_strategy"]
    for col, ylabel, ref in METRICS:
        if col not in df.columns or df[col].dropna().empty: continue
        fig, ax = plt.subplots(figsize=(7.5,5))
        agg = (df.dropna(subset=[col]).groupby(keys+["walks"])[col]
                 .agg(a.reduce).reset_index())
        for k, sub in agg.groupby(keys):
            sub = sub.sort_values("walks")
            ax.plot(sub["walks"], sub[col], marker="o", linewidth=1.6,
                    label=":".join(map(str,k)))
        ax.set_xscale("log"); ax.set_xlabel("walks per gene")
        ax.set_ylabel(f"{a.reduce} {ylabel}")
        ax.set_title(f"{a.reduce.capitalize()} {ylabel} vs walk saturation")
        ax.axhline(ref, ls="--", color="red", alpha=0.5)
        ax.legend(fontsize=8, loc="best"); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        stem = out / f"{a.prefix}_{col}_{a.reduce}"
        for ext in ("png","pdf"):
            fig.savefig(f"{stem}.{ext}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[WRITE] {stem}.png / .pdf")

if __name__ == "__main__":
    main()
