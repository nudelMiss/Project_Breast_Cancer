#!/usr/bin/env Rscript
suppressPackageStartupMessages({ library(Seurat); library(Matrix) })
rds_path <- "/groups/ofircohen-group/users/michalnu_yuvat/project_breast/RDS files/Bassez2021_Breast.RDS"
output_root <- "/groups/ofircohen-group/users/michalnu_yuvat/project_breast/exports_bassez"
manifest <- read.delim("bassez_groups_manifest.tsv", stringsAsFactors = FALSE)
cat("Manifest:", nrow(manifest), "groups to export\n")

cat("Loading RDS (one shot)...\n")
t0 <- Sys.time()
obj <- readRDS(rds_path)
md <- obj@meta.data
cat("Loaded in", round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1), "secs\n")

safe <- function(x) gsub("[^A-Za-z0-9._-]", "_", x)

for (i in seq_len(nrow(manifest))) {
  s <- manifest$sampleID[i]
  ct <- manifest$celltype[i]
  n_expected <- manifest$n_cells[i]
  out_dir <- file.path(output_root, paste0("patient=", safe(s), "__celltype=", safe(ct)))
  if (file.exists(file.path(out_dir, "expr.mtx"))) {
    cat(sprintf("[%d/%d] SKIP existing: %s / %s\n", i, nrow(manifest), s, ct))
    next
  }
  mask <- (md$sampleID == s) & (md$cell_type == ct)
  cells_keep <- rownames(md)[mask]
  cat(sprintf("[%d/%d] %s / %s -> %d cells\n", i, nrow(manifest), s, ct, length(cells_keep)))
  if (length(cells_keep) == 0) next
  sub_obj <- subset(obj, cells = cells_keep)
  counts <- as(GetAssayData(sub_obj, assay = "RNA", slot = "counts"), "CsparseMatrix")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  writeMM(counts, file.path(out_dir, "expr.mtx"))
  writeLines(rownames(counts), file.path(out_dir, "genes.csv"))
  writeLines(colnames(counts), file.path(out_dir, "cells.csv"))
  sub_md <- sub_obj@meta.data; sub_md$cell_id <- rownames(sub_md)
  write.csv(sub_md, file.path(out_dir, "meta.csv"), row.names = FALSE)
}
cat("\nDone.\n")
