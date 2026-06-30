#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --tasks=1
#SBATCH --gpus=0
#SBATCH --chdir=/home/michalnu
#SBATCH --output=/home/michalnu/griffiths_logs/%x_%j.out
#SBATCH --error=/home/michalnu/griffiths_logs/%x_%j.err
set -eo pipefail

PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for attempt in $(seq 1 30); do
  if [ -d "$PROOT" ]; then break; fi
  ls /mnt /mnt/new_groups /mnt/new_groups/ofircohen-group >/dev/null 2>&1 || true
  sleep 5
done
cd "$PROOT"

module load anaconda
source activate bc_r
Rscript scripts/inspect_bassez_rds.R "RDS files/Griffiths2021_Breast_10X.RDS" 2>&1
echo done=$(date)
