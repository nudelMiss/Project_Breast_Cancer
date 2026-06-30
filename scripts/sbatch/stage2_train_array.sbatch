#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --tasks=1
#SBATCH --gpus=0
#SBATCH --chdir=/home/michalnu
#SBATCH --output=/home/michalnu/newdata_logs/%x_%A_%a.out
#SBATCH --error=/home/michalnu/newdata_logs/%x_%A_%a.err
set -eo pipefail
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for a in $(seq 1 30); do [ -d "$PROOT" ] && break; ls /mnt /mnt/new_groups /mnt/new_groups/ofircohen-group >/dev/null 2>&1 || true; sleep 5; done
cd "$PROOT"; module load anaconda; source activate bc_walks
FK=resources/genesets/stageA_force_keep_genes.tsv
: "${DS:?}"; : "${IN_ROOT:?}"; : "${GLIST:?}"
mapfile -t GG < "$GLIST"
GNAME="${GG[$SLURM_ARRAY_TASK_ID]}"
[ -z "${GNAME:-}" ] && { echo "no group at idx $SLURM_ARRAY_TASK_ID"; exit 0; }
p=${GNAME#patient=}; p=${p%%__*}; c=${GNAME##*celltype=}; GTAG="${p}_${c}"
MODELS="results/$DS/stageA/models"; mkdir -p "$MODELS"
echo "[$DS] idx=$SLURM_ARRAY_TASK_ID group=$GNAME gtag=$GTAG"
for SIM in propr; do
  TAG="${SIM}_bidirectional_w10_k50_var75_hvg2000"
  if [ -f "$MODELS/$GTAG/$TAG/gene_embeddings.model" ]; then echo "[SKIP] $GTAG/$TAG"; continue; fi
  python -u scripts/train_model_new.py --in_root "$IN_ROOT" --only_group "$GNAME" \
    --graph_method var75 --sim "$SIM" --walk_strategy bidirectional --k_nearest 50 \
    --walks 10 --walk_length 3 --hvg_cap 2000 --benchmark_genes_tsv "$FK" \
    --vector_dim 64 --epochs 20 --window 5 --min_count 1 --seed 42 --out_root "$MODELS"
done
echo done=$(date)
