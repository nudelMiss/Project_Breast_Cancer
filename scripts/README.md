# Code layout

All pipeline code lives here. For the project overview and results, see the [top-level README](../README.md). Conda environment: `bc_walks`.

## Pipeline flow (what runs, in order)

1. **Export** — `exports/`: pull each cohort's expression matrices out of the source `.RDS` files.
2. **Train** — `train_model_new.py`: the core. Builds the gene-gene graph, samples random walks, trains the Word2Vec gene embeddings.
3. **Benchmark** — `AUC/` + `benchmark_gene_sets.py`: score how well the embeddings recover known gene modules (AUC and MCC metrics).
4. **Analyze** — `bassez2021/` (primary cohort) and `multidataset/` (cross-cohort): aggregation, summaries, figures.
5. **Visualize** — `visualization/`: cross-cohort UMAP / PCoA / LDA maps + poster figures.

`sbatch/` holds the SLURM job scripts that run all of the above on the cluster.

## Folders

- **exports/** — per-cohort data export + inspection R scripts. Each source `.RDS` gets its own exporter because formats and metadata differ; `export_generic_v2.R` is the shared parameterized exporter.
- **AUC/** — benchmarking: `benchmark_corum_auc.py` and `benchmark_corum_mcc.py` (AUC / MCC metrics), `bio_auc_posthoc.py` (biological-module AUC), `propr_negative_control.py`, plus aggregation/plots. `compare/` holds the AUC-method comparison.
- **bassez2021/** — the primary cohort, with the full analysis suite (staged screening, benchmarking, aggregation, summaries, presentation figures).
- **multidataset/** — cross-cohort analysis shared by every cohort (joint-by-cell-type aggregation, PCoA).
- **wu2021/, griffiths2021/** — thin per-cohort helpers (manifest + a cohort-specific comparison/inspection). The shared work lives in `multidataset/`, not duplicated here.
- **visualization/** — embedding maps (`umap/`, `pcoa/`, `lda/`) and `poster_figures/`.
- **figures/** — standalone figure scripts (config tables, axis comparisons, saturation curves).
- **sbatch/** — all SLURM job scripts.

## Key entry points

- `train_model_new.py` — the embedding pipeline (graph -> walks -> Word2Vec).
- `benchmark_gene_sets.py` — definitions of the benchmark gene modules (cell cycle, interferon, CORUM).
- `AUC/benchmark_corum_auc.py` — the main benchmark.
