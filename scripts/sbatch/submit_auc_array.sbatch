#!/bin/bash
#SBATCH --job-name=eval_auc
#SBATCH --output=auc_%j.out
#SBATCH --error=auc_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4


source ~/.bashrc
conda activate bc_walks

echo "Starting AUC Benchmark on $(hostname)"
echo "Start time: $(date)"

RESULTS_ROOT="/groups/ofircohen-group/users/michalnu_yuvat/project_breast/results_by_patient_celltype/patient=CID4513__celltype=T-cells/sim=spearman__emode=topk__k=250__walks=50__wlen=30__dim=64__ep=20__win=10__mc=5__seed=42"
CORUM_PATH="/groups/ofircohen-group/users/michalnu_yuvat/project_breast/resources/corum_core_complexes.tsv"
OUT_DIR="/groups/ofircohen-group/users/michalnu_yuvat/project_breast/results_by_patient_celltype/patient=CID4513__celltype=T-cells/sim=spearman__emode=topk__k=250__walks=50__wlen=30__dim=64__ep=20__win=10__mc=5__seed=42"

# חיפוש כל קבצי המודלים בתיקיית התוצאות שמכילים את המילה T-cells
find "$RESULTS_ROOT" -name "gene_embeddings.model" | grep "T-cells" | while read MODEL_PATH; do
    echo "========================================="
    echo "Evaluating Model: $MODEL_PATH"
    
    python scripts/AUC/benchmark_corum_auc.py \
        --embedding_path "$MODEL_PATH" \
        --corum_path "$CORUM_PATH" \
        --output_dir "$OUT_DIR"
done

echo "========================================="
echo "All AUC benchmarks completed!"
echo "End time: $(date)"