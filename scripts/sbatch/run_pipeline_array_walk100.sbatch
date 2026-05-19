#!/bin/bash
#SBATCH --job-name=walk100_arr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-88%4
#SBATCH --output=logs/walk100_%A_%a.out
#SBATCH --error=logs/walk100_%A_%a.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"

module load anaconda
source ~/.bashrc
conda activate bc_walks

GROUPS_FILE="scripts/groups_89.txt"

# --- pick group by array index ---
GROUP=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$GROUPS_FILE")
echo "JobID=$SLURM_JOB_ID ArrayTask=$SLURM_ARRAY_TASK_ID Group=$GROUP Host=$(hostname)"
echo "PWD=$(pwd)"

# --- parameters (your requested variant) ---
SIM="cosine"
EDGE_MODE="topk"
K_NEAREST="50"
WALKS="100"
OUT_ROOT="results/results_by_patient_celltype_100_walks"
# walk_length is hard-coded in python (6)

# --- deterministic tag (must match your python cfg.tag()) ---
# Update these if you change defaults in the python script.
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

python scripts/train_model_general.py \
  --sim "$SIM" \
  --edge_mode "$EDGE_MODE" \
  --k_nearest "$K_NEAREST" \
  --walks "$WALKS" \
  --only_group "$GROUP" \
  --out_root "$OUT_ROOT" \
  --save_edges