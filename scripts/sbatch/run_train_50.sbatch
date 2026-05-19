#!/bin/bash
#SBATCH --job-name=train_50
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_50_%j.out
#SBATCH --error=logs/train_50_%j.err

cd "$SLURM_SUBMIT_DIR"

eval "$(conda shell.bash hook)"
conda activate bc_walks

echo "Starting training for 50 walks..."

python scripts/train_model_general.py \
  --sim spearman \
  --k_nearest 10 \
  --walks 50 \
  --vector_dim 64 \
  --out_root results/models_spearman_filtered

echo "Done!"