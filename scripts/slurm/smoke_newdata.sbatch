#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --tasks=1
#SBATCH --gpus=0
#SBATCH --chdir=/home/michalnu
#SBATCH --output=/home/michalnu/newdata_logs/%x_%j.out
#SBATCH --error=/home/michalnu/newdata_logs/%x_%j.err
set -eo pipefail
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for a in $(seq 1 30); do [ -d "$PROOT" ] && break; ls /mnt /mnt/new_groups /mnt/new_groups/ofircohen-group >/dev/null 2>&1 || true; sleep 5; done
cd "$PROOT"
module load anaconda
source activate bc_walks

FK=resources/genesets/stageA_force_keep_genes.tsv
BENCH=resources/genesets/bio_modules_benchmark.tsv
CORUM=resources/corum_core_complexes.tsv

declare -A INROOT_OF GNAME_OF
INROOT_OF[wu2021]=exports_wu_counts
INROOT_OF[griffiths2021]=exports_griffiths
GNAME_OF[wu2021]="patient=CID4515__celltype=Malignant"
GNAME_OF[griffiths2021]="patient=P13__celltype=Malignant"

for DS in wu2021 griffiths2021; do
  INROOT="${INROOT_OF[$DS]}"; GNAME="${GNAME_OF[$DS]}"
  MODELS="results/$DS/stageA/models"; BIO="results/$DS/stageA/bio_auc"
  mkdir -p "$MODELS" "$BIO"
  echo "############ TRAIN $DS  group=$GNAME ############"
  for SIM in propr cosine; do
    python -u scripts/train_model_new.py \
      --in_root "$INROOT" --only_group "$GNAME" \
      --graph_method var75 --sim "$SIM" \
      --walk_strategy bidirectional --k_nearest 50 \
      --walks 10 --walk_length 3 \
      --hvg_cap 2000 --benchmark_genes_tsv "$FK" \
      --vector_dim 64 --epochs 20 --window 5 --min_count 1 --seed 42 \
      --out_root "$MODELS"
  done
  GTAG=$(ls "$MODELS" | head -1)
  GRP_LIST=$(ls "$MODELS" | paste -sd, -)
  CFG_LIST=$(ls "$MODELS/$GTAG" | paste -sd, -)
  echo "############ SCORE $DS  groups=$GRP_LIST  configs=$CFG_LIST ############"
  python scripts/AUC/bio_auc_posthoc.py \
    --models_root "$MODELS" --bench "$BENCH" --out_root "$BIO" \
    --groups "$GRP_LIST" --configs "$CFG_LIST" --corum "$CORUM"
done
echo ALL_DONE=$(date)
