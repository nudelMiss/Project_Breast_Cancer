#!/bin/bash
#SBATCH --job-name=winv2_pp
#SBATCH --output=/home/michalnu/winv2_logs/%x_%A_%a.out
#SBATCH --chdir=/home/michalnu
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=ise-cpu256-28
set -eo pipefail
mkdir -p /home/michalnu/winv2_logs
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for i in $(seq 1 30); do [ -d "$PROOT/$INROOT" ] && break; echo "[NFS] retry $i"; sleep 5; done
cd "$PROOT"
module load anaconda; source activate bc_walks
PY=/home/michalnu/.conda/envs/bc_walks/bin/python

GLIST="$PROOT/results/$DS/manifests/groups.txt"
GNAME=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$GLIST")
[ -z "$GNAME" ] && { echo "no group for idx $SLURM_ARRAY_TASK_ID"; exit 0; }
p=${GNAME%%__*}; p=${p#patient=}; ct=${GNAME##*celltype=}; GTAG="${p}_${ct}"
CFG=propr_bidirectional_w10_k50_var75_hvg2000
echo "=== $DS idx=$SLURM_ARRAY_TASK_ID group=$GNAME gtag=$GTAG ==="

MODELS="$PROOT/results/$DS/models"
$PY -u scripts/train_model_new.py \
  --in_root "$PROOT/$INROOT" --only_group "$GNAME" \
  --graph_method var75 --sim propr --walk_strategy bidirectional --k_nearest 50 \
  --walks 10 --walk_length 3 --hvg_cap 2000 \
  --benchmark_genes_tsv "$PROOT/resources/genesets/stageA_force_keep_genes.tsv" \
  --vector_dim 64 --epochs 20 --window 5 --min_count 1 --seed 42 \
  --out_root "$MODELS"

$PY -u scripts/AUC/bio_auc_posthoc.py \
  --models_root "$MODELS" \
  --bench "$PROOT/resources/genesets/bio_modules_benchmark.tsv" \
  --corum "$PROOT/resources/corum_core_complexes.tsv" \
  --out_root "$PROOT/results/$DS/bio_auc" \
  --groups "$GTAG" --configs "$CFG"
echo "=== DONE $GTAG ==="
