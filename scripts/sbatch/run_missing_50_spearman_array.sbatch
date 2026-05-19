#!/bin/bash
#SBATCH --job-name=emb50_sp
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/groups/ofircohen-group/users/michalnu_yuvat/project_breast
#SBATCH --output=logs/emb50_sp_%A_%a.out
#SBATCH --error=logs/emb50_sp_%A_%a.err

set -eo pipefail

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "Node=$SLURMD_NODENAME"
echo "Working dir=$(pwd)"
date

module load anaconda

PYTHON="/home/michalnu/.conda/envs/bc_walks/bin/python"

echo "Using Python: $PYTHON"
$PYTHON -c "import sys; print(sys.executable)"
$PYTHON -c "import igraph as ig; print('igraph ok:', ig.__version__)"

MANIFEST="missing_50_spearman_groups.txt"
GROUP=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")

if [ -z "$GROUP" ]; then
  echo "ERROR: no group found for task id $SLURM_ARRAY_TASK_ID"
  exit 1
fi

echo "Running group: $GROUP"

$PYTHON -u scripts/train_model_general.py \
  --only_group "$GROUP" \
  --sim spearman \
  --edge_mode topk \
  --k_nearest 10 \
  --walks 50 \
  --vector_dim 64 \
  --out_root results/models_spearman_filtered

echo "Finished group: $GROUP"
date