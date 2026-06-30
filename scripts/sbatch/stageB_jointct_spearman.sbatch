#!/bin/bash
#SBATCH --job-name=stageB_jsp
#SBATCH --output=/home/michalnu/stageB_jsp_%A_%a.out
#SBATCH --chdir=/home/michalnu
#SBATCH --array=0-11%6
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=ise-cpu256-28
set -e
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for i in $(seq 1 30); do [ -d "$PROOT/scripts" ] && break; echo "[NFS] retry $i"; sleep 5; done
cd "$PROOT"; module load anaconda; source activate bc_walks
read CT SIM STRAT <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$PROOT/results/bassez2021/manifests/stageB_jointct_spearman.txt")"
echo "===== ct=$CT sim=$SIM strat=$STRAT ====="
python -u scripts/bassez2021/aggregate_joint_by_celltype.py --celltype "$CT" --sim "$SIM" --strat "$STRAT"
TAG="${SIM}_${STRAT}_w10_k50_var75_hvg2000_jointct"
python -u scripts/AUC/bio_auc_posthoc.py --models_root "$PROOT/results/bassez2021/models_joint_by_celltype" \
  --bench "$PROOT/resources/genesets/bio_modules_benchmark.tsv" \
  --out_root "$PROOT/results/bassez2021/models_joint_by_celltype/bio_auc" \
  --corum "$PROOT/resources/corum_core_complexes.tsv" --groups "celltype=$CT" --configs "$TAG"
echo "===== DONE ====="
