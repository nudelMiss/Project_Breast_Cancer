#!/bin/bash
#SBATCH --job-name=stageA_spear_k5
#SBATCH --output=/home/michalnu/stageA_spear_k5_%A_%a.out
#SBATCH --chdir=/home/michalnu
#SBATCH --array=0-5
#SBATCH --time=06:00:00
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

# spearman at k=5 (both walk strategies) to complete the k-sweep from k5
python -u scripts/bassez2021/run_stageA_grid.py --group_name "$GNAME" \
  --assoc spearman --strats star,bidirectional --ks 5

echo "===== TASK $SLURM_ARRAY_TASK_ID DONE ====="
