#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-03:00:00
#SBATCH --job-name=pcoa_sp10_w50
#SBATCH --output=logs/pcoa_sp10_w50_%J.out
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -eo pipefail

cd /mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast

module load anaconda
source activate bc_walks

mkdir -p logs

W=50
TAG="sim=spearman__emode=topk__k=10__walks=${W}__wlen=6__dim=64__ep=20__win=4__mc=5__seed=42"

SRC_ROOT="results/models_spearman_filtered"

LINK_ROOT="results/PCoA/_model_links/spearman_10_edges_${W}_walks"
OUT_DIR="results/PCoA/spearman_10_edges_${W}_walks"
OUT_PREFIX="${OUT_DIR}/pcoa_walks${W}_sample500"

echo "======================================"
echo "Running PCoA/UMAP for:"
echo "  walks = ${W}"
echo "  tag   = ${TAG}"
echo "  source root = ${SRC_ROOT}"
echo "======================================"

rm -rf "$LINK_ROOT"
mkdir -p "$LINK_ROOT"
mkdir -p "$OUT_DIR"

N=0

while IFS= read -r model; do
    GROUP=$(basename "$(dirname "$(dirname "$model")")")
    MODEL_TAG=$(basename "$(dirname "$model")")

    mkdir -p "${LINK_ROOT}/${GROUP}/${MODEL_TAG}"
    ln -s "$(readlink -f "$model")" "${LINK_ROOT}/${GROUP}/${MODEL_TAG}/gene_embeddings.model"

    N=$((N + 1))
done < <(find "$SRC_ROOT" -path "*/${TAG}/gene_embeddings.model" | sort)

echo "[INFO] Linked ${N} models into ${LINK_ROOT}"

if [ "$N" -lt 10 ]; then
    echo "[ERROR] Found only ${N} models for walks=${W}. Not enough for PCoA."
    exit 1
fi

python scripts/PCoA/embedding_pcoa.py \
  --models_root "$LINK_ROOT" \
  --pattern gene_embeddings.model \
  --sample_genes 500 \
  --seed 42 \
  --out_prefix "$OUT_PREFIX" \
  --variant_label "spearman, 10 edges, ${W} walks" \
  --min_models 10

echo "[DONE] Results written to ${OUT_DIR}"