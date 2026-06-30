#!/bin/bash
#SBATCH --job-name=bioauc_posthoc
#SBATCH --output=/home/michalnu/bioauc_posthoc_%j.out
#SBATCH --chdir=/home/michalnu
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=ise-cpu256-28

set -e
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast

# --- NFS warm-up retry loop (avoid ExitCode 0:53 on NFS-unreachable nodes) ---
for i in $(seq 1 30); do
  if [ -d "$PROOT/scripts/AUC" ]; then echo "[NFS] reachable (try $i)"; break; fi
  echo "[NFS] not ready, retry $i/30"; sleep 5
done
cd "$PROOT"

module load anaconda
source activate bc_walks

BENCH="$PROOT/resources/genesets/bio_modules_benchmark.tsv"
CORUM="$PROOT/resources/corum_core_complexes.tsv"
OUT="$PROOT/results/bassez2021/bio_auc"
MR="$PROOT/results/bassez2021/models"

PILOTS="BIOKEY_18_T_cell,BIOKEY_30_Malignant,BIOKEY_4_B_cell"
TIERS="BIOKEY_10_Fibroblast,BIOKEY_12_Endothelial,BIOKEY_10_Myeloid"
SAT_CFGS="raw_cosine_bidirectional_w1_k5_wl3_perpat,raw_cosine_bidirectional_w5_k5_wl3_perpat,raw_cosine_bidirectional_w10_k5_wl3_perpat,raw_cosine_bidirectional_w50_k5_wl3_perpat,raw_cosine_bidirectional_w100_k5_wl3_perpat,raw_cosine_star_w100_k5_wl3_perpat"
KEY_CFGS="raw_cosine_bidirectional_w1_k5_wl3_perpat,raw_cosine_bidirectional_w10_k5_wl3_perpat,raw_cosine_bidirectional_w100_k5_wl3_perpat"

echo "===== 1) PILOTS: full saturation + star, with CORUM ====="
python -u "$PROOT/scripts/AUC/bio_auc_posthoc.py" \
  --models_root "$MR" --bench "$BENCH" --out_root "$OUT" --corum "$CORUM" \
  --groups "$PILOTS" --configs "$SAT_CFGS"

echo "===== 2) TIER groups: key configs, bio-only ====="
python -u "$PROOT/scripts/AUC/bio_auc_posthoc.py" \
  --models_root "$MR" --bench "$BENCH" --out_root "$OUT" \
  --groups "$TIERS" --configs "$KEY_CFGS"

echo "===== 3) JOINT headline (group ALL), with CORUM ====="
python -u "$PROOT/scripts/AUC/bio_auc_posthoc.py" \
  --models_root "$MR" --bench "$BENCH" --out_root "$OUT" --corum "$CORUM" \
  --groups "ALL" --configs "raw_cosine_bidirectional_w1_k5_wl3_joint,raw_cosine_bidirectional_w5_k5_wl3_joint,alra_cosine_bidirectional_w1_k5_wl3_joint"

echo "===== DONE ====="
