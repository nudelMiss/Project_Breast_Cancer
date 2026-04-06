#!/bin/bash
#SBATCH --job-name=auc5walks
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/auc_5_walks_%A_%a.out
#SBATCH --error=logs/auc_5_walks_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

MANIFEST_PATH="${MANIFEST_PATH:-resources/manifests/auc_5_walk_models.txt}"
CORUM_PATH="${CORUM_PATH:-resources/corum_core_complexes.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/auc_benchmarks/5_walks_all}"
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
MIN_COMPLEX_SIZE="${MIN_COMPLEX_SIZE:-3}"
RANDOM_SEED="${RANDOM_SEED:-42}"

if [ ! -f "$MANIFEST_PATH" ]; then
  echo "ERROR: manifest file not found: $MANIFEST_PATH"
  exit 1
fi

if [ ! -f "$CORUM_PATH" ]; then
  echo "ERROR: CORUM TSV not found: $CORUM_PATH"
  exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: Python binary not executable: $PYTHON_BIN"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
mkdir -p logs

EMBEDDING_PATH=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST_PATH")

if [ -z "${EMBEDDING_PATH}" ]; then
  echo "ERROR: no manifest entry for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
  exit 1
fi

if [ ! -f "$EMBEDDING_PATH" ]; then
  echo "ERROR: embedding path does not exist: $EMBEDDING_PATH"
  exit 1
fi

echo "=============================================="
echo "JobID=$SLURM_JOB_ID TaskID=$SLURM_ARRAY_TASK_ID"
echo "Host=$(hostname)"
echo "Embedding=$EMBEDDING_PATH"
echo "Manifest=$MANIFEST_PATH"
echo "OutputRoot=$OUTPUT_ROOT"
echo "=============================================="

group_name=$(basename "$(dirname "$(dirname "$EMBEDDING_PATH")")")
config_tag=$(basename "$(dirname "$EMBEDDING_PATH")")
summary_path="$OUTPUT_ROOT/$group_name/$config_tag/corum_auc_summary.csv"

if [ -f "$summary_path" ]; then
  echo "Output already exists, skipping: $summary_path"
  exit 0
fi

"$PYTHON_BIN" scripts/AUC/benchmark_corum_auc.py \
  --embedding_path "$EMBEDDING_PATH" \
  --corum_path "$CORUM_PATH" \
  --output_dir "$OUTPUT_ROOT" \
  --min_complex_size "$MIN_COMPLEX_SIZE" \
  --random_seed "$RANDOM_SEED"

echo "DONE: $EMBEDDING_PATH"
