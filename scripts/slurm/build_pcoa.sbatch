#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --tasks=1
#SBATCH --chdir=/home/michalnu
#SBATCH --output=/home/michalnu/newdata_logs/%x_%j.out
#SBATCH --error=/home/michalnu/newdata_logs/%x_%j.err
set -eo pipefail
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for a in $(seq 1 30); do [ -d "$PROOT" ] && break; ls /mnt /mnt/new_groups /mnt/new_groups/ofircohen-group >/dev/null 2>&1 || true; sleep 5; done
cd "$PROOT"; module load anaconda; source activate bc_walks
python -u scripts/multidataset/build_pcoa.py
echo done=$(date)
