#!/bin/bash
#SBATCH --job-name=fill_levers
#SBATCH --output=/home/michalnu/fill_levers_%A_%a.out
#SBATCH --chdir=/home/michalnu
#SBATCH --array=0-5
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --exclude=ise-cpu256-28

set -e
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for i in $(seq 1 30); do [ -d "$PROOT/scripts" ] && break; echo "[NFS] retry $i"; sleep 5; done
cd "$PROOT"
module load anaconda; source activate bc_walks

GLIST="$PROOT/results/bassez2021/manifests/stageA_group_names.txt"
GNAME=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$GLIST")
GTAG=$(python -c "import sys;sys.path.insert(0,'scripts');from train_model_new import make_group_tag;print(make_group_tag('$GNAME'))")
FK="$PROOT/resources/genesets/stageA_force_keep_genes.tsv"
MR="$PROOT/results/bassez2021/stageA/models"
COMMON="--in_root $PROOT/exports_bassez --only_group $GNAME --hvg_cap 2000 --benchmark_genes_tsv $FK \
  --k_nearest 50 --walks 10 --walk_length 3 --vector_dim 64 --epochs 20 --window 5 --min_count 1 --seed 42 --out_root $MR"
echo "===== $GTAG : spearman(var75) + propr(alra), star+bidir, k50 ====="
for STRAT in star bidirectional; do
  python -u scripts/train_model_new.py --graph_method var75 --sim spearman --walk_strategy $STRAT $COMMON || echo "FAIL spearman $STRAT"
  python -u scripts/train_model_new.py --graph_method alra  --sim propr    --walk_strategy $STRAT $COMMON || echo "FAIL alra-propr $STRAT"
done
TAGS="spearman_star_w10_k50_var75_hvg2000,spearman_bidirectional_w10_k50_var75_hvg2000,propr_star_w10_k50_alra_hvg2000,propr_bidirectional_w10_k50_alra_hvg2000"
echo "[SCORE] $GTAG"
python -u scripts/AUC/bio_auc_posthoc.py --models_root "$MR" \
  --bench "$PROOT/resources/genesets/bio_modules_benchmark.tsv" \
  --out_root "$PROOT/results/bassez2021/stageA/bio_auc" \
  --corum "$PROOT/resources/corum_core_complexes.tsv" \
  --groups "$GTAG" --configs "$TAGS"
echo "===== DONE $GTAG ====="
