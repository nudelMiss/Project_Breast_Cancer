#!/bin/bash
#SBATCH --job-name=train_100
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_100_%j.out
#SBATCH --error=logs/train_100_%j.err
#SBATCH --chdir=/groups/ofircohen-group/users/michalnu_yuvat/project_breast

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs
mkdir -p results/models_spearman_filtered

module load anaconda
eval "$(conda shell.bash hook)"
conda activate bc_walks

echo "Starting training for 100 walks..."

python scripts/train_model_general.py \
  --sim spearman \
  --k_nearest 10 \
  --walks 100 \
  --vector_dim 64 \
  --out_root results/models_spearman_filtered

echo "Done!"