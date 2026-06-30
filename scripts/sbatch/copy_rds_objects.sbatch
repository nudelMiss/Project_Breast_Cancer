#!/bin/bash
#SBATCH --job-name=copy_rds
#SBATCH --output=/home/yuvat/copy_rds_%j.out
#SBATCH --chdir=/home/yuvat
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --exclude=ise-cpu256-28
set -eo pipefail
SRC=/mnt/new_groups/ofircohen-group/datasets/Cancer_DB/scRNAseq_objects/3CA_Tirosh
DST=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast/objects
for a in $(seq 1 30); do [ -d "$SRC" ] && [ -d "$DST" ] && break; sleep 5; done
for f in Bassez2021_Breast_10x.RDS Wu2021_Breast_10x.RDS; do
  echo "[COPY] $f"; cp -p "$SRC/$f" "$DST/$f"
  echo "  src: $(stat -c %s "$SRC/$f")  dst: $(stat -c %s "$DST/$f")"
done
echo "=== md5 (dst) ==="; md5sum "$DST"/Bassez2021_Breast_10x.RDS "$DST"/Wu2021_Breast_10x.RDS
echo "DONE=$(date)"
