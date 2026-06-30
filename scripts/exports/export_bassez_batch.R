#!/usr/bin/env Rscript
suppressPackageStartupMessages({ library(Seurat); library(Matrix) })
rds_path <- "/groups/ofircohen-group/users/michalnu_yuvat/project_breast/RDS files/Bassez2021_Breast.RDS"
output_root <- "/groups/ofircohen-group/users/michalnu_yuvat/project_breast/exports_bassez"
samples <- c("BIOKEY_10", "BIOKEY_12", "BIOKEY_15")
celltype_val <- "T_cell"
sample_col <- "sampleID"
celltype_col <- "cell_type"

cat("Loading RDS (one shot)...\n")
obj <- readRDS(rds_path)
md <- obj@meta.data
safe <- function(x) gsub("[^A-Za-z0-9._-]", "_", x)

for (s in samples) {
  cat("\n--- Sample:", s, "---\n")
  mask <- (md[[sample_col]] == s) & (md[[celltype_col]] == celltype_val)
  cells_keep <- rownames(md)[mask]
  cat("  cells:", length(cells_keep), "\n")
  if (length(cells_keep) == 0) next
  sub_obj <- subset(obj, cells = cells_keep)
  counts <- as(GetAssayData(sub_obj, assay = "RNA", slot = "counts"), "CsparseMatrix")
  cat("  matrix:", nrow(counts), "x", ncol(counts), ",", length(counts@x), "nz\n")
  out_dir <- file.path(output_root, paste0("patient=", safe(s), "__celltype=", safe(celltype_val)))
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  writeMM(counts, file.path(out_dir, "expr.mtx"))
  writeLines(rownames(counts), file.path(out_dir, "genes.csv"))
  writeLines(colnames(counts), file.path(out_dir, "cells.csv"))
  sub_md <- sub_obj@meta.data; sub_md$cell_id <- rownames(sub_md)
  write.csv(sub_md, file.path(out_dir, "meta.csv"), row.names = FALSE)
  cat("  done:", out_dir, "\n")
}
cat("\nAll exports done.\n")
