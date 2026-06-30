#!/bin/bash
#SBATCH --job-name=npmi_negctrl
#SBATCH --output=/home/michalnu/npmi_negctrl_%j.out
#SBATCH --chdir=/home/michalnu
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --exclude=ise-cpu256-28

set -e
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for i in $(seq 1 30); do
  if [ -d "$PROOT/scripts/AUC" ]; then echo "[NFS] reachable (try $i)"; break; fi
  echo "[NFS] not ready, retry $i/30"; sleep 5
done
cd "$PROOT"
module load anaconda
source activate bc_walks

GROUP_LIST="BIOKEY_18_T_cell,BIOKEY_30_Malignant,BIOKEY_4_B_cell,BIOKEY_10_Fibroblast,BIOKEY_12_Endothelial,BIOKEY_10_Myeloid"
CONFIGS="npmi_star_w10_k50_var75_hvg2000,npmi_bidirectional_w10_k50_var75_hvg2000"

echo "===== nPMI negative control (random matched gene-sets) ====="
python -u "$PROOT/scripts/AUC/propr_negative_control.py" \
  --groups "$GROUP_LIST" --configs "$CONFIGS" --n_rep 40 \
  --out "$PROOT/results/bassez2021/stageA/bio_auc/npmi_negcontrol.csv"
echo "===== DONE ====="
