#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
#SBATCH --output=logs/bassez2021/smoke_new_modes_%A.out
#SBATCH --error=logs/bassez2021/smoke_new_modes_%A.err
set -eo pipefail
module load anaconda
source activate bc_walks
cd /mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
echo "host=$(hostname)  jobid=$SLURM_JOBID  start=$(date)"

GROUP="patient=BIOKEY_13__celltype=Myeloid"
SMOKE_OUT=results/bassez2021/_smoke
CACHE=results/bassez2021/_smoke/graphs

# 4 quick runs at walks=5 (fast). Confirm: cosine works, bidirectional works,
# graph cache is reused on the 2nd run of each (sim).
for SIM in spearman cosine; do
  for STRAT in star bidirectional; do
    echo "=========================================="
    echo "RUN sim=$SIM strat=$STRAT"
    echo "=========================================="
    python -u scripts/train_model_new.py \
      --in_root exports_bassez \
      --only_group "$GROUP" \
      --out_root "$SMOKE_OUT/models" \
      --graph_method var75 --variance_keep_frac 0.75 \
      --sim "$SIM" --edge_mode topk --k_nearest 5 \
      --walks 5 --walk_strategy "$STRAT" --walk_length 3 \
      --vector_dim 64 --epochs 5 --window 5 --min_count 1 \
      --graph_cache_dir "$CACHE"
  done
done

echo "done=$(date)"
