#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --tasks=1
#SBATCH --exclude=ise-cpu256-28,cs-cpu-10
#SBATCH --chdir=/home/yuvat
#SBATCH --output=/home/yuvat/umap_logs/%x_%j.out
#SBATCH --error=/home/yuvat/umap_logs/%x_%j.err
set -eo pipefail
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for a in $(seq 1 30); do [ -d "$PROOT" ] && break; ls /mnt /mnt/new_groups /mnt/new_groups/ofircohen-group >/dev/null 2>&1 || true; sleep 5; done
cd "$PROOT"; module load anaconda; source activate bc_walks
python -u crosscohort_viz/build_umap_emb_bc.py
echo done=$(date)
