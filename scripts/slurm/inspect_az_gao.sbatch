#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --tasks=1
#SBATCH --chdir=/home/michalnu
#SBATCH --output=/home/michalnu/newdata_logs/%x_%j.out
#SBATCH --error=/home/michalnu/newdata_logs/%x_%j.err
set -eo pipefail
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for a in $(seq 1 30); do [ -d "$PROOT" ] && break; ls /mnt /mnt/new_groups /mnt/new_groups/ofircohen-group >/dev/null 2>&1 || true; sleep 5; done
cd "$PROOT"; module load anaconda; source activate bc_r
echo "############ AZIZI ############"; Rscript scripts/exports/inspect_assays.R "RDS_files/Azizi2018_Breast_10x.RDS" 2>&1
echo "############ GAO ############"; Rscript scripts/exports/inspect_assays.R "RDS_files/Data_Gao2021_Breast.RDS" 2>&1
echo done=$(date)
