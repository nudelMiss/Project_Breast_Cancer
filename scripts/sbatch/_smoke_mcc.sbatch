#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --chdir=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
#SBATCH --output=logs/bassez2021/smoke_mcc_%A.out
#SBATCH --error=logs/bassez2021/smoke_mcc_%A.err
set -eo pipefail
module load anaconda
source activate bc_walks
cd /mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
echo "host=$(hostname)  jobid=$SLURM_JOBID  start=$(date)"

python -u scripts/AUC/benchmark_corum_mcc.py \
  --embedding_path results/models_bassez_spearman_star/BIOKEY_30_Malignant/spearman_star_w100_k5_var75/gene_embeddings.model \
  --corum_path resources/corum_core_complexes.tsv \
  --output_dir results/bassez2021/_smoke_mcc/BIOKEY_30_Malignant_var75 \
  --random_seed 42

echo "done=$(date)"
