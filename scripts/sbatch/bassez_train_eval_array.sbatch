#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tasks=1
#SBATCH --gpus=0
#SBATCH --chdir=/home/michalnu
#SBATCH --output=/home/michalnu/bassez_logs/%x_%A_%a.out
#SBATCH --error=/home/michalnu/bassez_logs/%x_%A_%a.err
set -eo pipefail

if [ -z "${MANIFEST:-}" ]; then
  echo "FATAL: MANIFEST env var must be set" >&2; exit 2
fi

module load anaconda
source activate bc_walks

# Robust cd to project dir: warm NFS from parents, retry on transient failure.
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
  echo "FATAL: project dir unreachable after 10 retries: $PROOT" >&2
  exit 11
fi
cd "$PROOT"

echo "===================="
echo "host        = $(hostname)"
echo "jobid       = $SLURM_JOBID"
echo "array_task  = $SLURM_ARRAY_TASK_ID"
echo "nodelist    = $SLURM_JOB_NODELIST"
echo "manifest    = $MANIFEST"
echo "pwd         = $(pwd)"
echo "start       = $(date)"
echo "===================="

ROW=${SLURM_ARRAY_TASK_ID:-0}

python3 -c "
import csv,sys
rows=list(csv.DictReader(open('$MANIFEST'),delimiter='\t'))
r=rows[$ROW]
print('### ROW $ROW ###')
for k,v in r.items(): print(f'  {k}: {v}')
" || true

python -u scripts/bassez2021/train_one_config.py \
  --manifest "$MANIFEST" --row_index "$ROW"

if [ "${SKIP_EVAL:-0}" != "1" ]; then
  python -u scripts/bassez2021/eval_one_config.py \
    --manifest "$MANIFEST" --row_index "$ROW"
fi

echo "done=$(date)"
