#!/bin/bash
#SBATCH --partition=main
#SBATCH --job-name=ids_bio_gate
#SBATCH --output=/home/michalnu/logs/ids_bio_gate_%A_%a.out
#SBATCH --error=/home/michalnu/logs/ids_bio_gate_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0-02:00:00
#SBATCH --chdir=/home/michalnu
#SBATCH --exclude=ise-cpu256-28
#SBATCH --array=0-5%3

set -eo pipefail
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for a in $(seq 1 30); do [ -d "$PROOT" ] && break; ls /mnt/new_groups/ofircohen-group >/dev/null 2>&1 || true; sleep 5; done
cd "$PROOT"
mkdir -p /home/michalnu/logs

module load anaconda
set +u
source activate bc_walks
set -u

GROUP_LIST=( \
  "patient=BIOKEY_18__celltype=T_cell" \
  "patient=BIOKEY_30__celltype=Malignant" \
  "patient=BIOKEY_4__celltype=B_cell" )
PNORMS=( "max" "2" )

NG=${#GROUP_LIST[@]}
GI=$(( SLURM_ARRAY_TASK_ID % NG ))
PI=$(( SLURM_ARRAY_TASK_ID / NG ))
G="${GROUP_LIST[$GI]}"
P="${PNORMS[$PI]}"

echo "Task $SLURM_ARRAY_TASK_ID: group=$G p_norm=$P  $(date)"
python -u scripts/AUC/ids_diagnostic.py \
  --group_dir "exports_bassez/$G" \
  --bench_path "resources/genesets/bio_modules_benchmark.tsv" \
  --bench_tag "biomod_p${P}" \
  --p_norm "$P" \
  --output_root results/diagnostics/ids_bio_gate
echo "Done $(date)"
