#!/usr/bin/env python3
"""
Build results/experiment_inventory.tsv : one row per (group, config_tag) model dir.
Validates outputs and records CORUM-AUC and BIO-AUC as SEPARATE columns.

- model_ok : gene_embeddings.model exists and non-empty
- corum_* : from results/bassez2021/auc/<g>/<tag>/corum_auc_summary.csv (mean_auc, weighted_mean_auc, n_complexes_used)
- bio_*   : from results/bassez2021/bio_auc/<g>/<tag>/bio_auc_summary.csv (mean_bio_auc + per module)
Config tag parsed as: {imp}_{sim}_{strat}_w{walks}_k{k}_wl{wl}_{agg}
"""
from pathlib import Path
import csv, re

ROOT = Path("/groups/ofircohen-group/users/michalnu_yuvat/project_breast")
MODELS = ROOT / "results/bassez2021/models"
AUC = ROOT / "results/bassez2021/auc"
BIO = ROOT / "results/bassez2021/bio_auc"
OUT = ROOT / "results/experiment_inventory.tsv"

TAG_RE = re.compile(r"^(?P<imp>[^_]+)_(?P<sim>[^_]+)_(?P<strat>[^_]+)_w(?P<walks>\d+)_k(?P<k>\d+)_wl(?P<wl>\d+)_(?P<agg>.+)$")

def read_summary(path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None

def f(d, key):
    if d is None or key not in d or d[key] in ("", None):
        return ""
    try:
        return f"{float(d[key]):.4f}"
    except (ValueError, TypeError):
        return d[key]

rows = []
n_model = n_corum = n_bio = 0
for gdir in sorted(MODELS.iterdir()):
    if not gdir.is_dir():
        continue
    group = gdir.name
    for cdir in sorted(gdir.iterdir()):
        if not cdir.is_dir():
            continue
        tag = cdir.name
        m = TAG_RE.match(tag)
        fields = m.groupdict() if m else {k: "" for k in
                  ["imp","sim","strat","walks","k","wl","agg"]}
        mp = cdir / "gene_embeddings.model"
        model_ok = mp.exists() and mp.stat().st_size > 0
        if model_ok: n_model += 1
        corum = read_summary(AUC / group / tag / "corum_auc_summary.csv")
        bio = read_summary(BIO / group / tag / "bio_auc_summary.csv")
        if corum is not None: n_corum += 1
        if bio is not None: n_bio += 1
        rows.append({
            "group": group, "config_tag": tag,
            "imputation": fields["imp"], "sim": fields["sim"],
            "walk_strategy": fields["strat"], "walks": fields["walks"],
            "k": fields["k"], "wl": fields["wl"], "aggregation": fields["agg"],
            "model_ok": int(model_ok),
            "corum_mean_auc": f(corum, "mean_auc"),
            "corum_weighted_auc": f(corum, "weighted_mean_auc"),
            "corum_n_complexes": (corum or {}).get("n_complexes_used", ""),
            "bio_mean_auc": f(bio, "mean_bio_auc"),
            "bio_S_phase": f(bio, "auc_S_phase"),
            "bio_G2M": f(bio, "auc_G2M"),
            "bio_IFN_alpha": f(bio, "auc_IFN_alpha"),
            "bio_IFN_gamma": f(bio, "auc_IFN_gamma"),
        })

cols = ["group","config_tag","imputation","sim","walk_strategy","walks","k","wl",
        "aggregation","model_ok","corum_mean_auc","corum_weighted_auc","corum_n_complexes",
        "bio_mean_auc","bio_S_phase","bio_G2M","bio_IFN_alpha","bio_IFN_gamma"]
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=cols, delimiter="\t")
    w.writeheader()
    w.writerows(rows)

print(f"[INVENTORY] {len(rows)} model dirs")
print(f"  model_ok: {n_model}   corum_auc present: {n_corum}   bio_auc present: {n_bio}")
print(f"  -> {OUT}")
