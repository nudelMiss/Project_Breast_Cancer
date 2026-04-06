#!/bin/bash
#SBATCH --job-name=corum_auc
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/corum_auc_%A_%a.out
#SBATCH --error=logs/corum_auc_%A_%a.err

# CORUM AUC Benchmark - SLURM Array Job
# Runs benchmark_corum_auc.py for each embedding path listed in EMBEDDINGS_LIST.
#
# Usage:
#   1. Create a text file with one embedding path per line
#   2. Set EMBEDDINGS_LIST below
#   3. Submit with array bounds matching number of lines, e.g.:
#      sbatch --array=0-88%8 scripts/submit_auc_array.sbatch

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"

PY=${PYTHON_BIN:-python}

# Required input: one embedding path per line
EMBEDDINGS_LIST=${EMBEDDINGS_LIST:-scripts/embedding_paths.txt}

# Fixed parameters
CORUM_PATH="resources/corum_core_complexes.tsv"
OUTPUT_DIR=${OUTPUT_DIR:-results/auc_benchmarks}
MIN_COMPLEX_SIZE=3
SEED=42

if [ ! -f "$EMBEDDINGS_LIST" ]; then
    echo "ERROR: Embeddings list not found: $EMBEDDINGS_LIST"
    exit 1
fi

EMBEDDING_PATH=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$EMBEDDINGS_LIST")
if [ -z "$EMBEDDING_PATH" ]; then
    echo "ERROR: No embedding path at array index $SLURM_ARRAY_TASK_ID in $EMBEDDINGS_LIST"
    exit 1
fi

echo "=============================================="
echo "JobID=$SLURM_JOB_ID ArrayTask=$SLURM_ARRAY_TASK_ID"
echo "Embedding=$EMBEDDING_PATH"
echo "Host=$(hostname)"
echo "PWD=$(pwd)"
echo "=============================================="

# Check if CORUM file exists
if [ ! -f "$CORUM_PATH" ]; then
    echo "ERROR: CORUM file not found: $CORUM_PATH"
    echo "Run: python scripts/preprocess_corum.py first"
    exit 1
fi

# Check if embedding exists
if [ ! -f "$EMBEDDING_PATH" ]; then
    echo "WARNING: Embedding not found: $EMBEDDING_PATH"
    echo "Skipping this task."
    exit 0
fi

echo "Embedding: $EMBEDDING_PATH"
echo "CORUM: $CORUM_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

# Run benchmark
$PY -u scripts/benchmark_corum_auc.py \
    --embedding_path "$EMBEDDING_PATH" \
    --corum_path "$CORUM_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --min_complex_size "$MIN_COMPLEX_SIZE" \
    --random_seed "$SEED"

echo ""
echo "DONE: $EMBEDDING_PATH"
