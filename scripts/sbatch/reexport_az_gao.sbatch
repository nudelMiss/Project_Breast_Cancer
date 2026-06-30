#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --tasks=1
#SBATCH --chdir=/home/michalnu
#SBATCH --output=/home/michalnu/newdata_logs/%x_%j.out
#SBATCH --error=/home/michalnu/newdata_logs/%x_%j.err
set -eo pipefail
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for a in $(seq 1 30); do [ -d "$PROOT" ] && break; ls /mnt /mnt/new_groups /mnt/new_groups/ofircohen-group >/dev/null 2>&1 || true; sleep 5; done
cd "$PROOT"; module load anaconda; source activate bc_r
echo "#### AZIZI ####"; Rscript scripts/exports/export_generic_v2.R --rds="RDS_files/Azizi2018_Breast_10x.RDS" --output_root=exports_azizi --slot=counts --min_cells=200 --sample_col=sampleID --celltype_col=cell_type --inventory_out=results/azizi2018/summaries/exports_inventory.tsv
echo "#### GAO ####"; Rscript scripts/exports/export_generic_v2.R --rds="RDS_files/Data_Gao2021_Breast.RDS" --output_root=exports_gao --slot=counts --min_cells=200 --sample_col=orig.ident --celltype_col=cell_type --inventory_out=results/gao2021/summaries/exports_inventory.tsv
echo done=$(date)
