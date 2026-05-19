#!/bin/bash
#SBATCH --job-name=walk100_single
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/walk100_single_%j.out
#SBATCH --error=logs/walk100_single_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

PY=~/.conda/envs/bc_walks/bin/python
OUT_ROOT="results/results_by_patient_celltype_100_walks"

GROUP="$1"

$PY -u scripts/train_model_general.py \
  --sim cosine \
  --edge_mode topk \
  --k_nearest 50 \
  --walks 100 \
  --only_group "$GROUP" \
  --out_root "$OUT_ROOT" \
  --materialize
