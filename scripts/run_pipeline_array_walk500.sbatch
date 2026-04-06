#!/bin/bash
#SBATCH --job-name=walk500_arr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-12:00:00
#SBATCH --array=0-88%2
#SBATCH --output=logs/walk500_%A_%a.out
#SBATCH --error=logs/walk500_%A_%a.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"

PY=~/.conda/envs/bc_walks/bin/python
GROUPS_FILE="scripts/groups_89.txt"

GROUP=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$GROUPS_FILE")

echo "JobID=$SLURM_JOB_ID ArrayTask=$SLURM_ARRAY_TASK_ID Group=$GROUP Host=$(hostname)"
echo "PWD=$(pwd)"

SIM="cosine"
EDGE_MODE="topk"
K_NEAREST="50"
WALKS="500"

OUT_ROOT="results/results_by_patient_celltype_500_walks"

WALKLEN="6"
DIM="64"
EPOCHS="20"
WINDOW="4"
MINCOUNT="5"
SEED="42"

TAG="sim=${SIM}__emode=${EDGE_MODE}__k=${K_NEAREST}__walks=${WALKS}__wlen=${WALKLEN}__dim=${DIM}__ep=${EPOCHS}__win=${WINDOW}__mc=${MINCOUNT}__seed=${SEED}"

MODEL_PATH="${OUT_ROOT}/${GROUP}/${TAG}/gene_embeddings.model"

if [ -f "$MODEL_PATH" ]; then
  echo "Already finished: $MODEL_PATH"
  exit 0
fi

$PY -u scripts/train_model_general.py \
  --sim "$SIM" \
  --edge_mode "$EDGE_MODE" \
  --k_nearest "$K_NEAREST" \
  --walks "$WALKS" \
  --only_group "$GROUP" \
  --out_root "$OUT_ROOT" \
  --materialize \
  --save_edges
