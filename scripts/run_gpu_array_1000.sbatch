#!/bin/bash
#SBATCH --job-name=bc1000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=2-12:00:00
#SBATCH --output=logs/bc1000_%A_%a.out
#SBATCH --error=logs/bc1000_%A_%a.err

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

set -euo pipefail

IN_ROOT="exports_by_patient_celltype"
OUT_ROOT="results_by_patient_celltype_1000_walks"
WALKS=1000
SEED=42
COS_BATCH=1024   # lower to 512 if GPU OOM

PY=~/.conda/envs/bc_walks/bin/python
SCRIPT="train_model_all_groups_1000_walks_GPU.py"

mkdir -p logs "$OUT_ROOT"

# Resolve the group folder for this array index
GROUP=$(find "$IN_ROOT" -mindepth 1 -maxdepth 1 -type d | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")
NAME=$(basename "$GROUP")

OUTFILE="$OUT_ROOT/$NAME/gene_embeddings_w${WALKS}_weighted.model"
if [[ -f "$OUTFILE" ]]; then
  echo "Already exists: $OUTFILE  -> skipping"
  exit 0
fi

echo "Node: ${SLURM_NODELIST}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Task: ${SLURM_ARRAY_TASK_ID}"
echo "Group: ${NAME}"
echo "Python: $($PY -c "import sys; print(sys.executable)")"

# Sanity check: torch + GPU
$PY -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# Run the actual job
$PY -u "$SCRIPT" \
  --walks "$WALKS" \
  --seed "$SEED" \
  --in_root "$IN_ROOT" \
  --out_root "$OUT_ROOT" \
  --only_group "$NAME" \
  --cos_batch "$COS_BATCH" \
  --materialize

