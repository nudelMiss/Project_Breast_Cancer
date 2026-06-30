#!/bin/bash
#SBATCH --partition=main
#SBATCH --job-name=viz_v2_all
#SBATCH --time=0-03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --tasks=1
#SBATCH --exclude=ise-cpu256-28,cs-cpu-10
#SBATCH --chdir=/home/yuvat
#SBATCH --output=/home/yuvat/umap_logs/%x_%j.out
#SBATCH --error=/home/yuvat/umap_logs/%x_%j.err
PROOT=/mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast
for a in $(seq 1 30); do [ -d "$PROOT" ] && break; ls /mnt/new_groups/ofircohen-group >/dev/null 2>&1 || true; sleep 5; done
cd "$PROOT"; module load anaconda; source activate bc_walks
for s in \
  scripts/visualization/lda/build_lda_emb_6cohort_v2.py \
  scripts/visualization/umap/build_umap_emb_6cohort_v2.py \
  scripts/visualization/pcoa/build_pcoa_emb_bc_jointct_v2.py \
  scripts/visualization/pcoa/build_pcoa_emb_bc_v2.py \
  scripts/visualization/pcoa/build_pcoa_bc_strong_v2.py \
  scripts/visualization/pcoa/build_pcoa_v2run.py ; do
  echo "######## RUN $s ########"
  python -u "$s" && echo "[OK] $s" || echo "[FAIL] $s rc=$?"
  echo
done
echo "ALL_DONE=$(date)"
