#!/bin/bash
#SBATCH --job-name=stageB_pp
#SBATCH --output=/home/michalnu/stageB_pp_%A_%a.out
#SBATCH --chdir=/home/michalnu
#SBATCH --array=0-182%40
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=ise-cpu256-28

set -e
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for i in $(seq 1 30); do [ -d "$PROOT/scripts" ] && break; echo "[NFS] retry $i"; sleep 5; done
cd "$PROOT"
module load anaconda
source activate bc_walks

GLIST="$PROOT/results/bassez2021/manifests/stageB_group_names.txt"
GNAME=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$GLIST")
echo "===== TASK $SLURM_ARRAY_TASK_ID : $GNAME ====="
python -u scripts/bassez2021/run_stageA_grid.py \
  --group_name "$GNAME" \
  --assoc propr,cosine --strats star,bidirectional --ks 50
echo "===== TASK $SLURM_ARRAY_TASK_ID DONE ====="
