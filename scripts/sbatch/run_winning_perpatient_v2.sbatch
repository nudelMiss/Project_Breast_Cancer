#!/bin/bash
#SBATCH --job-name=win_pp_v2
#SBATCH --output=/home/michalnu/win_pp_v2_%A_%a.out
#SBATCH --chdir=/home/michalnu
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=ise-cpu256-28
set -e
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for i in $(seq 1 30); do [ -d "$PROOT/scripts" ] && break; sleep 5; done
cd "$PROOT"
module load anaconda; source activate bc_walks
PY=python
FK=$PROOT/resources/genesets/stageA_force_keep_genes.tsv
BENCH=$PROOT/resources/genesets/bio_modules_benchmark.tsv
CORUM=$PROOT/resources/corum_core_complexes.tsv
# env-provided: DS_IN (exports dir), DS_OUT (results root for this dataset), GLIST (group-name list)
GNAME=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$GLIST")
echo "TASK $SLURM_ARRAY_TASK_ID : $GNAME  (in=$DS_IN out=$DS_OUT)"
# group tag: patient=X__celltype=Y -> X_Y
GTAG=$(echo "$GNAME" | sed -E 's/patient=//; s/__celltype=/_/')
MODELS=$PROOT/$DS_OUT/models
$PY -u scripts/train_model_new.py \
  --in_root "$PROOT/$DS_IN" --only_group "$GNAME" \
  --graph_method var75 --sim propr --walk_strategy bidirectional \
  --k_nearest 50 --walks 10 --walk_length 3 \
  --hvg_cap 2000 --benchmark_genes_tsv "$FK" \
  --vector_dim 64 --epochs 20 --window 5 --min_count 1 --seed 42 \
  --out_root "$MODELS"
$PY -u scripts/AUC/bio_auc_posthoc.py \
  --models_root "$MODELS" --bench "$BENCH" --corum "$CORUM" \
  --out_root "$PROOT/$DS_OUT/bio_auc" \
  --groups "$GTAG" --configs propr_bidirectional_w10_k50_var75_hvg2000
echo "DONE $GTAG"
