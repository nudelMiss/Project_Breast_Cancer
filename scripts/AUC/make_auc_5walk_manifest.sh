#!/bin/bash
set -euo pipefail

ROOT_DIR="${1:-results/results_by_patient_celltype_5_walks}"
MANIFEST_PATH="${2:-resources/manifests/auc_5_walk_models.txt}"

mkdir -p "$(dirname "$MANIFEST_PATH")"

find "$ROOT_DIR" -type f -name "gene_embeddings.model" | sort > "$MANIFEST_PATH"

count=$(wc -l < "$MANIFEST_PATH")
echo "Manifest written: $MANIFEST_PATH"
echo "Models listed: $count"
