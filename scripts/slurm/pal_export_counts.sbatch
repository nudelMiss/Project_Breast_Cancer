#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --tasks=1
#SBATCH --gpus=0
#SBATCH --chdir=/home/michalnu
#SBATCH --output=/home/michalnu/newdata_logs/%x_%j.out
#SBATCH --error=/home/michalnu/newdata_logs/%x_%j.err
set -eo pipefail
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for a in $(seq 1 30); do [ -d "$PROOT" ] && break; ls /mnt /mnt/new_groups /mnt/new_groups/ofircohen-group >/dev/null 2>&1 || true; sleep 5; done
cd "$PROOT"; module load anaconda; source activate bc_r
for i in 01 02 03 04 05; do
  RDS="RDS_files/Pal2021_Breast_${i}_10x.RDS"
  [ -f "$RDS" ] || { echo "MISSING $RDS"; continue; }
  echo "############ PAL FILE $i ############"
  Rscript scripts/pal2021_export.R --rds="$RDS" --output_root=exports_pal2021 --slot=counts --min_cells=200 --skip_celltype=equivocal || echo "[WARN] file $i export failed"
done
echo "exports_pal2021 groups: $(ls -d exports_pal2021/patient=*/ 2>/dev/null | wc -l)"
echo done=$(date)
