#!/bin/bash
#SBATCH --job-name=export_v2
#SBATCH --output=/home/michalnu/export_v2_%j.out
#SBATCH --chdir=/home/michalnu
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=ise-cpu256-28
set -eo pipefail
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for i in $(seq 1 30); do [ -d "$PROOT/objects" ] && break; echo "[NFS] retry $i"; sleep 5; done
cd "$PROOT"
module load anaconda
source activate bc_r

echo "############ BASSEZ v2 ############"
Rscript scripts/multidataset/export_v2.R \
  --rds=objects/Bassez2021_Breast_10x.RDS \
  --output_root=exports_bassez_v2 \
  --slot=counts --min_cells=200 --skip_celltype=equivocal \
  --inventory_out=results/bassez2021_v2/summaries/exports_inventory.tsv

echo "############ WU v2 ############"
Rscript scripts/multidataset/export_v2.R \
  --rds=objects/Wu2021_Breast_10x.RDS \
  --output_root=exports_wu_v2 \
  --slot=counts --min_cells=200 --skip_celltype=equivocal \
  --inventory_out=results/wu2021_v2/summaries/exports_inventory.tsv

echo "ALL DONE $(date)"
