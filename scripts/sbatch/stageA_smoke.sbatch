#!/bin/bash
#SBATCH --job-name=stageA_smoke
#SBATCH --output=/home/michalnu/stageA_smoke_%j.out
#SBATCH --chdir=/home/michalnu
#SBATCH --time=01:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=ise-cpu256-28

set -e
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for i in $(seq 1 30); do [ -d "$PROOT/scripts" ] && break; echo "[NFS] retry $i"; sleep 5; done
cd "$PROOT"
module load anaconda
source activate bc_walks

GRP="patient=BIOKEY_12__celltype=Endothelial"   # smallest group, 349 cells
FK="$PROOT/resources/genesets/stageA_force_keep_genes.tsv"
OUT="$PROOT/results/bassez2021/stageA/models_smoke"
COMMON="--in_root exports_bassez --only_group $GRP --graph_method var75 \
  --hvg_cap 2000 --benchmark_genes_tsv $FK --k_nearest 50 \
  --walk_strategy star --walks 10 --walk_length 3 --skip_w2v --no-save_edges \
  --out_root $OUT"

echo "##### IDS (the feasibility gate) #####"
/usr/bin/time -v python -u scripts/train_model_new.py --sim ids $COMMON 2>&1 | \
  grep -E "\[HVG\]|\[IDS\]|\[EDGES\]|Elapsed|Maximum resident" || true

echo "##### propr #####"
/usr/bin/time -v python -u scripts/train_model_new.py --sim propr $COMMON 2>&1 | \
  grep -E "\[HVG\]|\[EDGES\]|Elapsed|Maximum resident" || true

echo "##### cosine (matched-universe baseline arm) #####"
/usr/bin/time -v python -u scripts/train_model_new.py --sim cosine $COMMON 2>&1 | \
  grep -E "\[HVG\]|\[EDGES\]|Elapsed|Maximum resident" || true

echo "##### DONE #####"
