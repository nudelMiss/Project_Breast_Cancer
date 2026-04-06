#!/bin/bash
#SBATCH --partition=main
#SBATCH --job-name=pcoa_w2v
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pcoa_%j.out
#SBATCH --error=logs/pcoa_%j.err

set -eo pipefail

cd ~/project_temp/project_breast
mkdir -p logs results/PCoA

module load anaconda

set +u
source activate bc_walks
set -u

python -u scripts/PCoA/embedding_pcoa.py \
  --models_root results/results_by_patient_celltype_500_walks \
  --pattern gene_embeddings.model \
  --sample_genes 500 \
  --seed 42 \
  --out_prefix results/PCoA/cosine_50_edges_500_walks/pcoa_walks500_sample500 \
  --variant_label "cosine, 50 edges, 500 walks"