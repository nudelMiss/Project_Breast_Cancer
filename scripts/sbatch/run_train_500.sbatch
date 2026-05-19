#!/bin/bash
#SBATCH --job-name=train_final
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/groups/ofircohen-group/users/michalnu_yuvat/project_breast
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# הדפסת נתיב העבודה לביקורת
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "Running on node: $SLURMD_NODENAME"

# טעינת סביבה
module load anaconda
eval "$(conda shell.bash hook)"
conda activate bc_walks

# הרצה
python scripts/train_model_general.py \
  --sim spearman \
  --k_nearest 10 \
  --walks 500 \
  --vector_dim 64 \
  --out_root results/models_fix
EOF