#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tasks=1
#SBATCH --gpus=0
#SBATCH --chdir=/home/michalnu
#SBATCH --output=/home/michalnu/griffiths_logs/%x_%A_%a.out
#SBATCH --error=/home/michalnu/griffiths_logs/%x_%A_%a.err
set -eo pipefail

if [ -z "${MANIFEST:-}" ]; then
  echo "FATAL: MANIFEST env var must be set" >&2; exit 2
fi

module load anaconda
source activate bc_walks

PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for attempt in $(seq 1 30); do
  for d in /mnt /mnt/new_groups /mnt/new_groups/ofircohen-group \
           /mnt/new_groups/ofircohen-group/users \
           /mnt/new_groups/ofircohen-group/users/michalnu_yuvat ; do
    ls "$d" >/dev/null 2>&1 || true
  done
  if [ -d "$PROOT" ]; then break; fi
  echo "[NFS_WAIT] attempt $attempt -- $PROOT not yet reachable"; sleep 5
done

if [ ! -d "$PROOT" ]; then
  echo "FATAL: project dir unreachable after retries: $PROOT" >&2
  exit 11
fi
cd "$PROOT"

echo "host=$(hostname)  job=$SLURM_JOBID  arr=$SLURM_ARRAY_TASK_ID  manifest=$MANIFEST"

ROW=${SLURM_ARRAY_TASK_ID:-0}

python -u scripts/bassez2021/train_one_config.py \
  --manifest "$MANIFEST" --row_index "$ROW"

python -u scripts/bassez2021/eval_one_config.py \
  --manifest "$MANIFEST" --row_index "$ROW"

echo "done=$(date)"
