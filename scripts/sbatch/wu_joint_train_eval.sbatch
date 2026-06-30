#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --tasks=1
#SBATCH --gpus=0
#SBATCH --chdir=/home/michalnu
#SBATCH --output=/home/michalnu/wu_logs/%x_%j.out
#SBATCH --error=/home/michalnu/wu_logs/%x_%j.err
set -eo pipefail

if [ -z "${MANIFEST:-}" ]; then
  echo "FATAL: MANIFEST env var must be set" >&2; exit 2
fi
if [ -z "${ROW:-}" ]; then
  echo "FATAL: ROW env var must be set (manifest row index of joint row)" >&2; exit 2
fi
if [ -z "${SUMMARY:-}" ]; then
  echo "FATAL: SUMMARY env var must be set (dataset_summary.csv path)" >&2; exit 2
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
  echo "[NFS_WAIT] attempt $attempt"; sleep 5
done
cd "$PROOT"

echo "host=$(hostname)  job=$SLURM_JOBID  row=$ROW  summary=$SUMMARY"

python -u scripts/bassez2021/train_one_joint.py \
  --manifest "$MANIFEST" --row_index "$ROW" --summary_csv "$SUMMARY"

python -u scripts/bassez2021/eval_one_config.py \
  --manifest "$MANIFEST" --row_index "$ROW"

echo "done=$(date)"
