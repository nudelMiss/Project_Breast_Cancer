#!/usr/bin/env python3
"""
Stage A screening driver (per-patient, metric question).
For ONE group, trains the assoc x walk x k grid at fixed raw/w10/wl3/hvg2000,
then scores bio + CORUM AUC on each model.

Grid: assoc{cosine,ids,propr} x strat{star,bidirectional} x k{5,50}  (+cscore if --cscore_cache given)
Outputs land in results/bassez2021/stageA/{models,bio_auc}/<group_tag>/<tag>/
RunConfig.tag() = {sim}_{strat}_w10_k{k}_var75_hvg2000  (self-distinguishing, never collides w/ cached).
"""
import argparse, subprocess, sys
from pathlib import Path

PROOT = Path("/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast")
PY = sys.executable
FK = str(PROOT / "resources/genesets/stageA_force_keep_genes.tsv")
BENCH = str(PROOT / "resources/genesets/bio_modules_benchmark.tsv")
CORUM = str(PROOT / "resources/corum_core_complexes.tsv")
STAGEA = PROOT / "results/bassez2021/stageA"
MODELS = STAGEA / "models"

ASSOC = ["cosine", "ids", "propr"]
STRATS = ["star", "bidirectional"]
KS = [5, 50]
WALKS, WL = 10, 3


def cfg_tag(sim, strat, k):
    return f"{sim}_{strat}_w{WALKS}_k{k}_var75_hvg2000"


def train_one(group_name, sim, strat, k, cscore_cache=None):
    tag = cfg_tag(sim, strat, k)
    outdir = MODELS / group_name_to_tag(group_name) / tag
    if (outdir / "gene_embeddings.model").exists():
        print(f"[SKIP] {outdir}", flush=True)
        return tag
    cmd = [PY, "-u", str(PROOT / "scripts/train_model_new.py"),
           "--in_root", str(PROOT / "exports_bassez"),
           "--only_group", group_name,
           "--graph_method", "var75",
           "--sim", sim,
           "--walk_strategy", strat,
           "--k_nearest", str(k),
           "--walks", str(WALKS), "--walk_length", str(WL),
           "--hvg_cap", "2000", "--benchmark_genes_tsv", FK,
           "--vector_dim", "64", "--epochs", "20", "--window", "5",
           "--min_count", "1", "--seed", "42",
           "--out_root", str(MODELS)]
    if sim == "cscore":
        if not cscore_cache:
            print(f"[SKIP cscore] no --cscore_cache provided", flush=True)
            return None
        cmd += ["--precomputed_edges", cscore_cache]
    print("[TRAIN]", tag, flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"[FAIL rc={rc}] {tag}", flush=True)
        return None
    return tag


def group_name_to_tag(group_name):
    # patient=BIOKEY_12__celltype=Endothelial -> BIOKEY_12_Endothelial
    p = group_name.split("__")
    patient = p[0].replace("patient=", "")
    ct = next((x.replace("celltype=", "") for x in p if x.startswith("celltype=")), "NA")
    return f"{patient}_{ct}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group_name", required=True, help="exports_bassez dir name, e.g. patient=BIOKEY_12__celltype=Endothelial")
    ap.add_argument("--cscore_cache", default=None, help="path to precomputed cscore edges.tsv for this group")
    ap.add_argument("--assoc", default=",".join(ASSOC))
    ap.add_argument("--ks", default=",".join(str(k) for k in KS))
    ap.add_argument("--strats", default=",".join(STRATS))
    args = ap.parse_args()

    assoc = [a for a in args.assoc.split(",") if a]
    strats = [x for x in args.strats.split(",") if x]
    ks = [int(x) for x in args.ks.split(",") if x]
    gtag = group_name_to_tag(args.group_name)
    tags = []
    for sim in assoc:
        for strat in strats:
            for k in ks:
                t = train_one(args.group_name, sim, strat, k, args.cscore_cache)
                if t:
                    tags.append(t)
    if not tags:
        print("No models trained.", flush=True); return

    # Score bio + CORUM on all trained configs for this group.
    print(f"\n[SCORE] {gtag}: {len(tags)} configs", flush=True)
    subprocess.run([PY, "-u", str(PROOT / "scripts/AUC/bio_auc_posthoc.py"),
                    "--models_root", str(MODELS),
                    "--bench", BENCH, "--out_root", str(STAGEA / "bio_auc"),
                    "--corum", CORUM,
                    "--groups", gtag, "--configs", ",".join(tags)])
    print(f"[DONE] {gtag}", flush=True)


if __name__ == "__main__":
    main()
