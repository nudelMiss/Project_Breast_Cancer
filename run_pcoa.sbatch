#!/bin/bash
#SBATCH --job-name=pcoa_w2v
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pcoa_%j.out
#SBATCH --error=logs/pcoa_%j.err

set -euo pipefail

cd ~/project_temp/project_breast
mkdir -p logs

PY=~/.conda/envs/bc_walks/bin/python

$PY -u embedding_pcoa.py \
  --models_root results_by_patient_celltype_1000_walks \
  --pattern gene_embeddings_w1000_weighted.model \
  --sample_genes 500 \
  --seed 42 \
  --out_prefix pcoa_latest_ellipses_sample500
