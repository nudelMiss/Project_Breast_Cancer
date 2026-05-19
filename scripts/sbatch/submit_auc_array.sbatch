#!/bin/bash
#SBATCH --job-name=corum_auc
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=logs/corum_auc_%A_%a.out
#SBATCH --error=logs/corum_auc_%A_%a.err

# 1. הפעלת סביבת קונדה באופן מפורש
eval "$(conda shell.bash hook)"
conda activate bc_walks

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

# 2. קבלת משתנים מבחוץ (או שימוש בברירת מחדל)
EMBEDDINGS_LIST=${EMBEDDINGS_LIST:-manifest_5_walks.txt}
OUTPUT_DIR=${OUTPUT_DIR:-results/auc_benchmarks/spearman/walks_5}

CORUM_PATH="resources/corum_core_complexes.tsv"
MIN_COMPLEX_SIZE=3
SEED=42

if [ ! -f "$EMBEDDINGS_LIST" ]; then
    echo "ERROR: Embeddings list not found: $EMBEDDINGS_LIST"
    exit 1
fi

EMBEDDING_PATH=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$EMBEDDINGS_LIST")

if [ -z "$EMBEDDING_PATH" ]; then
    echo "ERROR: No embedding path at array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Running AUC for: $EMBEDDING_PATH"

# 3. הפעלת הסקריפט מתוך התיקייה המעודכנת (scripts/AUC/)
python -u scripts/AUC/benchmark_corum_auc.py \
    --embedding_path "$EMBEDDING_PATH" \
    --corum_path "$CORUM_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --min_complex_size "$MIN_COMPLEX_SIZE" \
    --random_seed "$SEED"

echo "DONE: $EMBEDDING_PATH"