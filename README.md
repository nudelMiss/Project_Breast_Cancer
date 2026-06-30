# Elucidating Cell-Type-Specific Gene Function with Single-Cell RNA-seq Using Co-Expression Graphs and Language Models

*Bioinformatics final project · Ben-Gurion University of the Negev · 202-2026-14*

**Michal Nudelman & Yuval Tal**
Advisors: Yaniv Pevzner · Dr. Ofir Cohen · Prof. Chen Keasar

## Abstract

Single-cell RNA-seq resolves gene expression by cell type, but whether co-expression encodes gene *function* is unclear. We build cell-type-specific gene–gene association graphs, sample random walks over them, and train Word2Vec embeddings that place functionally related genes near one another. Benchmarking against known biological modules (cell cycle, interferon) and CORUM complexes, we find that **graph density and walk strategy matter more than the association metric**, and that **pooling embeddings by cell type across patients gives the strongest, most reproducible signal — G2M cell-cycle AUC 0.92.**

## The idea: reading the language of the cell

To fight cancer we have to read a cell's "source code" — its gene expression, meaning which genes are switched on and how strongly. Two things make that hard. **Pleiotropy:** a gene's role depends on context, the way *heart* means something different in "heart of the city" versus "human heart" — so we analyze one cell type at a time. **Heterogeneity:** a tumor is a noisy ecosystem of cancer, immune, and stromal cells, like following one conversation in a crowded room.

Our twist is to borrow a proven NLP tool. Treat each **cell as a sentence** and each **active gene as a word**; genes used together in the same context land close together on a learned map. The model is never told which genes are related — it learns the cell's "syntax" unsupervised.

## Pipeline

1. **Preprocessing** — clean the raw matrices, filter damaged cells, strip technical noise.
2. **Graph construction** — genes become nodes; a co-expression metric sets weighted edges.
3. **Random walks** — a walker hops node to node along edge weights; each path becomes a "sentence."
4. **Word2Vec training** — tens of thousands of walk-sentences in, a gene-embedding map out.
5. **Benchmarking** — score how tightly known gene modules cluster, using AUC (0.5 = random, 1.0 = perfect).

**Final configuration:** raw / ALRA imputation → **proportionality (`propr`)** metric → **k = 50** neighbors → bidirectional / star walks → **joint-by-cell-type** pooling.

## Key results

- **It works, unsupervised.** Known programs self-organize with no labels given. Cell-cycle **G2M reaches AUC 0.92** (p < 10^-20), near the theoretical ceiling for data this sparse.
- **Metric choice:** rank/proportion-based metrics (`propr`, Spearman) beat magnitude-based ones (cosine, IDS) on every biological benchmark, so `propr` was selected.
- **Neighborhood size:** signal climbs to k ~ 50 then saturates; k = 100 adds nothing.
- **Headline — pooling by cell type:** one map per cell type (pooling patients) lifts mean bio-module AUC from **~0.84 (per-patient) to ~0.91 (joint)**, beating every per-patient approach while preserving cell-type resolution.

Benchmarks drawn from Tirosh 2016 / Seurat `cc.genes` (cell cycle) and MSigDB Hallmark (interferon). CORUM protein complexes are reported only as a secondary *method-limit* reference — complexes are regulated post-transcriptionally and co-express weakly at the mRNA level.

## Data

Pooled single-cell breast-tumor data from public atlases (Tirosh lab / Weizmann 3CA, scBaseCount, and others): a harmonized resource of 188 studies / 6.5M cells with consensus cell-type labels (SingleR + SCType + Scimilarity vote). Primary cohorts: **Bassez2021** (primary), Wu2021, Qian2020, Pal2021, Gao2021, Azizi2018.

## Repository structure

    scripts/      all pipeline code -- see scripts/README.md for the full layout and flow
    resources/        gene-set definitions and CORUM reference

Raw matrices, trained models, and per-run results live on the cluster and are excluded from version control.

## Reproducing

Conda environment `bc_walks`; main training entry point `scripts/train_model_new.py`; benchmarking under `scripts/AUC/`. Jobs run via SLURM (`sbatch`).

## Acknowledgments

Dr. Ofir Cohen for the vision and guidance; Yaniv Pevzner for hands-on mentorship; Menachem and the Ofir Cohen Lab for technical help and a warm lab atmosphere.
