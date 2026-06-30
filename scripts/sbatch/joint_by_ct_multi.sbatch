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
cd "$PROOT"; module load anaconda; source activate bc_walks
BENCH=resources/genesets/bio_modules_benchmark.tsv
CORUM=resources/corum_core_complexes.tsv
declare -A EXP_OF
EXP_OF[wu2021]=exports_wu_counts
EXP_OF[griffiths2021]=exports_griffiths
EXP_OF[qian2020]=exports_qian
EXP_OF[pal2021]=exports_pal2021
EXP_OF[azizi2018]=exports_azizi
EXP_OF[gao2021]=exports_gao
for DS in ${DSLIST:-wu2021 griffiths2021 qian2020}; do
  MR="results/$DS/stageA/models"; OUTR="results/$DS/models_joint_by_celltype"
  CTS=$(ls -d ${EXP_OF[$DS]}/patient=*/ | sed 's/.*celltype=//; s|/||' | sort -u)
  echo "############ $DS celltypes: $(echo $CTS | tr '\n' ' ') ############"
  for CT in $CTS; do
    python -u scripts/aggregate_joint_by_celltype_multi.py \
      --models_root "$MR" --out_root "$OUTR" --celltype "$CT" --sim propr --strat bidirectional
  done
  GRP_LIST=$(for CT in $CTS; do echo "celltype=$CT"; done | paste -sd, -)
  echo "[$DS] scoring joint models: $GRP_LIST"
  python scripts/AUC/bio_auc_posthoc.py --models_root "$OUTR" --bench "$BENCH" \
    --out_root "results/$DS/jointct_bio_auc" \
    --groups "$GRP_LIST" --configs "propr_bidirectional_w10_k50_var75_hvg2000_jointct" --corum "$CORUM"
done
echo ALL_DONE=$(date)
