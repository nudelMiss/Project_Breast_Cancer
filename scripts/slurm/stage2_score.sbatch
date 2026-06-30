#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --tasks=1
#SBATCH --gpus=0
#SBATCH --chdir=/home/michalnu
#SBATCH --output=/home/michalnu/newdata_logs/%x_%j.out
#SBATCH --error=/home/michalnu/newdata_logs/%x_%j.err
set -eo pipefail
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for a in $(seq 1 30); do [ -d "$PROOT" ] && break; ls /mnt /mnt/new_groups /mnt/new_groups/ofircohen-group >/dev/null 2>&1 || true; sleep 5; done
cd "$PROOT"; module load anaconda; source activate bc_walks
BENCH=resources/genesets/bio_modules_benchmark.tsv
CORUM=resources/corum_core_complexes.tsv
: "${DS:?}"; : "${GLIST:?}"
MODELS="results/$DS/stageA/models"; BIO="results/$DS/stageA/bio_auc"; mkdir -p "$BIO"
GRP_LIST=$(while read GNAME; do [ -z "$GNAME" ] && continue; p=${GNAME#patient=}; p=${p%%__*}; c=${GNAME##*celltype=}; echo "${p}_${c}"; done < "$GLIST" | paste -sd, -)
CFG_LIST="propr_bidirectional_w10_k50_var75_hvg2000"
echo "[$DS] scoring $(echo $GRP_LIST | tr ',' '\n' | wc -l) groups x 2 configs"
python scripts/AUC/bio_auc_posthoc.py --models_root "$MODELS" --bench "$BENCH" \
  --out_root "$BIO" --groups "$GRP_LIST" --configs "$CFG_LIST" --corum "$CORUM"
echo done=$(date)
