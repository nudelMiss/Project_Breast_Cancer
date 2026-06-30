#!/bin/bash
#SBATCH --job-name=stageA_npmi_k100
#SBATCH --output=/home/michalnu/stageA_npmi_k100_%A_%a.out
#SBATCH --chdir=/home/michalnu
#SBATCH --array=0-5
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=ise-cpu256-28

set -e
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for i in $(seq 1 30); do [ -d "$PROOT/scripts" ] && break; echo "[NFS] retry $i"; sleep 5; done
cd "$PROOT"
module load anaconda
source activate bc_walks

GLIST="$PROOT/results/bassez2021/manifests/stageA_group_names.txt"
GNAME=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$GLIST")
echo "===== TASK $SLURM_ARRAY_TASK_ID : $GNAME ====="

# Task 1: nPMI as new metric, full k-sweep x both walk strategies
echo "--- npmi x {star,bidir} x {k5,50,100} ---"
python -u scripts/bassez2021/run_stageA_grid.py --group_name "$GNAME" \
  --assoc npmi --strats star,bidirectional --ks 5,50,100

# Task 2: k=100 saturation fill for the existing winners + cosine baseline
echo "--- k=100 fill: {propr,spearman,cosine} x {star,bidir} ---"
python -u scripts/bassez2021/run_stageA_grid.py --group_name "$GNAME" \
  --assoc propr,spearman,cosine --strats star,bidirectional --ks 100

echo "===== TASK $SLURM_ARRAY_TASK_ID DONE ====="
