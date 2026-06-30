#!/bin/bash
#SBATCH --job-name=jointct_v2
#SBATCH --output=/home/michalnu/jointct_v2_%A_%a.out
#SBATCH --chdir=/home/michalnu
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=ise-cpu256-28
set -e
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for i in $(seq 1 30); do [ -d "$PROOT/scripts" ] && break; sleep 5; done
cd "$PROOT"; module load anaconda; source activate bc_walks
BENCH=$PROOT/resources/genesets/bio_modules_benchmark.tsv
CORUM=$PROOT/resources/corum_core_complexes.tsv
# env: DS_OUT (results root), CTFILE (celltype list)
CT=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$CTFILE")
PP=$PROOT/$DS_OUT/models
OUT=$PROOT/$DS_OUT/models_joint_by_celltype
echo "JOINT-CT $CT (pp=$PP)"
python -u scripts/multidataset/aggregate_joint_by_celltype_v2.py \
  --ppmodels_root "$PP" --outroot "$OUT" --celltype "$CT" --sim propr --strat bidirectional
# score the joint-ct model: it lives under OUT/celltype=CT/<tag>_jointct/
python -u scripts/AUC/bio_auc_posthoc.py \
  --models_root "$OUT" --bench "$BENCH" --corum "$CORUM" \
  --out_root "$PROOT/$DS_OUT/bio_auc_jointct" \
  --groups "celltype=$CT" --configs propr_bidirectional_w10_k50_var75_hvg2000_jointct || echo "[warn] scoring path mismatch; will handle in summary"
echo "DONE CT=$CT"
