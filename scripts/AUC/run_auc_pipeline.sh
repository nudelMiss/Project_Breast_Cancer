#!/bin/bash

# --- CONFIGURATION ------
WALKS="500" 
# ------------------------

SOURCE_ROOT="/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast/results/models_spearman_filtered"
OUTPUT_ROOT="results/auc_benchmarks/spearman/walks_${WALKS}"
CORUM_PATH="resources/corum_core_complexes.tsv"
MODEL_NAME="gene_embeddings.model"

echo "Starting Spearman AUC Pipeline for Walks: ${WALKS}"

mkdir -p "$OUTPUT_ROOT"

for patient_dir in ${SOURCE_ROOT}/patient=*; do
    if [ -d "$patient_dir" ]; then
        
        
        for config_dir in ${patient_dir}/sim=spearman*walks=${WALKS}*; do
            if [ -d "$config_dir" ]; then
                model_file="${config_dir}/${MODEL_NAME}"
                
                if [ -f "$model_file" ]; then
                    echo "Processing Patient: $(basename $patient_dir)"
                    
                    python scripts/AUC/benchmark_corum_auc.py \
                        --embedding_path "$model_file" \
                        --output_dir "$OUTPUT_ROOT" \
                        --corum_path "$CORUM_PATH" \
                        --min_complex_size 3
                fi
            fi
        done
    fi
done

echo "------------------------------------------------"
echo "Starting Aggregation for walks_${WALKS}..."

python scripts/AUC/aggregate_auc_results.py --input_root "$OUTPUT_ROOT" \
    --metric "spearman" \
    --walks "$WALKS"

echo "Done! Results saved in: $OUTPUT_ROOT"