#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --tasks=1
#SBATCH --gpus=0
#SBATCH --chdir=/home/michalnu
#SBATCH --output=/home/michalnu/griffiths_logs/%x_%j.out
#SBATCH --error=/home/michalnu/griffiths_logs/%x_%j.err
set -eo pipefail
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for a in $(seq 1 30); do [ -d "$PROOT" ] && break; ls /mnt/new_groups/ofircohen-group >/dev/null 2>&1 || true; sleep 5; done
cd "$PROOT"
module load anaconda
source activate bc_r
Rscript scripts/griffiths2021/export_griffiths_all.R \
  --rds="RDS files/Griffiths2021_Breast_10X.RDS" \
  --output_root=exports_griffiths \
  --min_cells=200 \
  --skip_celltype=equivocal \
  ${SAMPLE_COL:+--sample_col=$SAMPLE_COL} \
  ${CELLTYPE_COL:+--celltype_col=$CELLTYPE_COL}
echo done=$(date)
