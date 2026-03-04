#!/bin/bash
#SBATCH --job-name=bc500
#SBATCH --partition=main
#SBATCH --time=2-12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module load anaconda
source activate bc_walks   # <-- אם שם הסביבה שלך שונה, תשני פה

LIST="remaining_500.txt"
IN_ROOT="exports_by_patient_celltype"
OUT_ROOT="results_by_patient_celltype_500_walks"

GROUP=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")

echo "Running group: $GROUP"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unset}"

python -u train_model_all_groups_500_walks.py \
  --walks 500 \
  --seed 42 \
  --in_root "$IN_ROOT" \
  --out_root "$OUT_ROOT" \
  --skip_existing \
  --only_group "$GROUP"

