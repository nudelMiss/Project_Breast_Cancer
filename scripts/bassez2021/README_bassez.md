# Bassez2021 — Run Guide

All commands assume `cd /mnt/new_groups/ofircohen-group/users/michalnu_yuvat/project_breast`
(use `/groups/ofircohen-group/...` interchangeably).

## 0. Sanity (already done; re-run any time)

```bash
python3 scripts/bassez2021/stage0_sanity_check.py
# -> results/bassez2021/summaries/dataset_summary.csv + dataset_summary_report.txt
```

## 1. Pilot

```bash
# Build pilot manifest (6 groups x 8 configs = 48 rows)
python3 scripts/bassez2021/build_manifest.py \
    --stage pilot --out results/bassez2021/manifests/pilot.tsv

# Launch (idempotent — already-trained rows will skip)
MANIFEST=results/bassez2021/manifests/pilot.tsv sbatch \
    --job-name=bzPilot --array=0-47%20 \
    --export=ALL,MANIFEST=results/bassez2021/manifests/pilot.tsv \
    scripts/slurm/bassez_train_eval_array.sbatch

# Validate
python3 scripts/bassez2021/validate_outputs.py \
    --manifest results/bassez2021/manifests/pilot.tsv \
    --out_delta results/bassez2021/manifests/pilot_missing.tsv

# Aggregate + decide top-6 configs for Stage 2
python3 scripts/bassez2021/aggregate_pilot.py \
    --manifest results/bassez2021/manifests/pilot.tsv \
    --out      results/bassez2021/summaries/pilot_results.csv \
    --require_both

python3 scripts/bassez2021/select_top_configs.py \
    --pilot_csv   results/bassez2021/summaries/pilot_results.csv \
    --keep 6 \
    --out_ranking results/bassez2021/summaries/pilot_ranking.csv \
    --out_chosen  results/bassez2021/summaries/stage2_chosen.txt

# Figures
python3 scripts/figures/plot_axis_comparisons.py \
    --in_csv results/bassez2021/summaries/pilot_results.csv \
    --out_dir results/bassez2021/figures --prefix pilot
```

## 2. Main grid (Stage 2, per-patient, all 184 groups × top 6 configs)

```bash
CHOSEN=$(cat results/bassez2021/summaries/stage2_chosen.txt)
python3 scripts/bassez2021/build_manifest.py \
    --stage stage2 --configs "$CHOSEN" \
    --out results/bassez2021/manifests/stage2.tsv

# N = 184 * 6 = 1104 rows. Use %20.
N=$(($(wc -l < results/bassez2021/manifests/stage2.tsv) - 1))
MANIFEST=results/bassez2021/manifests/stage2.tsv sbatch \
    --job-name=bzStage2 --array=0-$((N-1))%20 \
    --export=ALL,MANIFEST=results/bassez2021/manifests/stage2.tsv \
    scripts/slurm/bassez_train_eval_array.sbatch

# Validate + aggregate
python3 scripts/bassez2021/validate_outputs.py \
    --manifest results/bassez2021/manifests/stage2.tsv \
    --out_delta results/bassez2021/manifests/stage2_missing.tsv

python3 scripts/bassez2021/aggregate_pilot.py \
    --manifest results/bassez2021/manifests/stage2.tsv \
    --out results/bassez2021/summaries/stage2_results.csv --require_both

# Pick top-2 for saturation
python3 scripts/bassez2021/select_top_configs.py \
    --pilot_csv results/bassez2021/summaries/stage2_results.csv \
    --keep 2 \
    --out_ranking results/bassez2021/summaries/stage2_ranking.csv \
    --out_chosen  results/bassez2021/summaries/saturation_chosen.txt
```

## 3. Saturation (top-2 configs × all 184 groups × walks ∈ {1,5,10,50,100,1000})

```bash
CHOSEN=$(cat results/bassez2021/summaries/saturation_chosen.txt)
python3 scripts/bassez2021/build_manifest.py \
    --stage saturation --configs "$CHOSEN" \
    --out results/bassez2021/manifests/saturation.tsv

N=$(($(wc -l < results/bassez2021/manifests/saturation.tsv) - 1))
MANIFEST=results/bassez2021/manifests/saturation.tsv sbatch \
    --job-name=bzSat --array=0-$((N-1))%20 \
    --export=ALL,MANIFEST=results/bassez2021/manifests/saturation.tsv \
    scripts/slurm/bassez_train_eval_array.sbatch

python3 scripts/bassez2021/aggregate_pilot.py \
    --manifest results/bassez2021/manifests/saturation.tsv \
    --out results/bassez2021/summaries/saturation_results.csv --require_both

python3 scripts/figures/plot_saturation_curves.py \
    --in_csv results/bassez2021/summaries/saturation_results.csv \
    --out_dir results/bassez2021/figures --prefix saturation
```

## 4. Joint embeddings (top-2 configs, 2 large single jobs)

```bash
CHOSEN=$(cat results/bassez2021/summaries/saturation_chosen.txt)
python3 scripts/bassez2021/build_manifest.py \
    --stage joint --configs "$CHOSEN" \
    --out results/bassez2021/manifests/joint.tsv

# Joint = expensive; run with longer time and more memory.
MANIFEST=results/bassez2021/manifests/joint.tsv sbatch \
    --job-name=bzJoint --array=0-1%2 \
    --time=0-24:00:00 --mem=128G --cpus-per-task=12 \
    --export=ALL,MANIFEST=results/bassez2021/manifests/joint.tsv \
    scripts/slurm/bassez_train_eval_array.sbatch
```

## 5. Final aggregation + figures

```bash
# Combine all stages into one CSV.
python3 -c "
import pandas as pd, glob
parts = []
for n in ['pilot','stage2','saturation','joint']:
    p = f'results/bassez2021/summaries/{n}_results.csv'
    try: parts.append(pd.read_csv(p))
    except FileNotFoundError: pass
pd.concat(parts, ignore_index=True).to_csv('results/bassez2021/summaries/all_results.csv', index=False)
print('all_results.csv rows:', sum(len(p) for p in parts))
"

python3 scripts/figures/plot_axis_comparisons.py \
    --in_csv results/bassez2021/summaries/all_results.csv \
    --out_dir results/bassez2021/figures --prefix all

python3 scripts/figures/make_best_config_tables.py \
    --in_csv results/bassez2021/summaries/all_results.csv \
    --out_dir results/bassez2021/summaries --prefix all
```

## Checklist before submission

1. `dataset_summary.csv` shows 184 groups, 7 celltypes, no equivocal.
2. `validate_outputs.py` on every manifest reports `complete N/N`.
3. `all_results.csv` exists and has rows for pilot+stage2+saturation+joint.
4. Figures: `pilot_by_*`, `saturation_*`, `all_by_*`, `all_top_*` CSVs.
5. `README_bassez.md` updated with any deviation.
