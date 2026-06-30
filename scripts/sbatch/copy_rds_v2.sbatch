#!/bin/bash
#SBATCH --job-name=copy_rds_v2
#SBATCH --output=/home/yuvat/copy_rds_v2_%j.out
#SBATCH --chdir=/home/yuvat
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --exclude=ise-cpu256-28
set -e
SRC=/mnt/new_groups/ofircohen-group/datasets/Cancer_DB/scRNAseq_objects/3CA_Tirosh
DST=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast/objects
for i in $(seq 1 30); do [ -d "$DST" ] && break; sleep 5; done
cp -v "$SRC/Bassez2021_Breast_10x.RDS" "$DST/"
cp -v "$SRC/Wu2021_Breast_10x.RDS"     "$DST/"
ls -la "$DST"/*.RDS
md5sum "$SRC/Bassez2021_Breast_10x.RDS" "$DST/Bassez2021_Breast_10x.RDS"
md5sum "$SRC/Wu2021_Breast_10x.RDS" "$DST/Wu2021_Breast_10x.RDS"
echo DONE
