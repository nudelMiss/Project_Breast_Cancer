#!/usr/bin/env Rscript
suppressPackageStartupMessages({ library(Seurat) })
rds_path <- "/groups/ofircohen-group/users/michalnu_yuvat/project_breast/RDS files/Bassez2021_Breast.RDS"
obj <- readRDS(rds_path)
md <- obj@meta.data

sample_col <- "sampleID"
ct_col <- "cell_type"
cat("Using sample col =", sample_col, ", celltype col =", ct_col, "\n")

cat("\n=== Sample counts (top 30) ===\n")
print(head(sort(table(md[[sample_col]]), decreasing=TRUE), 30))

cat("\n=== Celltype counts ===\n")
print(table(md[[ct_col]]))

cat("\n=== Sample x Celltype cross-tab (T_cell column, top 30 samples) ===\n")
ct <- table(md[[sample_col]], md[[ct_col]])
tc <- ct[, "T_cell"]
tc <- sort(tc, decreasing=TRUE)
print(head(tc, 30))

cat("\n=== Closest to ~4500 T-cells (Wu CID3586 size) ===\n")
target <- 4500
diffs <- abs(tc - target)
ord <- order(diffs)
print(head(data.frame(sample=names(tc)[ord], n_tcells=as.integer(tc[ord])), 10))
