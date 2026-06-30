#!/usr/bin/env Rscript
suppressPackageStartupMessages({ library(Seurat); library(Matrix) })

args <- commandArgs(trailingOnly = TRUE)
positional <- args[!grepl("^--", args)]
named <- args[grepl("^--", args)]
if (length(positional) < 2) {
  stop("Usage: Rscript export_bassez_celltype.R <SAMPLE> <CELLTYPE> [--flags]")
}
sample_val <- positional[1]
celltype_val <- positional[2]

get_named <- function(key, default) {
  hit <- grep(paste0("^--", key, "="), named, value = TRUE)
  if (length(hit) == 0) return(default)
  sub(paste0("^--", key, "="), "", hit[1])
}

rds_path     <- get_named("rds",
  "/groups/ofircohen-group/users/michalnu_yuvat/project_breast/RDS files/Bassez2021_Breast.RDS")
sample_col   <- get_named("sample_col", "")
celltype_col <- get_named("celltype_col", "")
assay        <- get_named("assay", "RNA")
slot         <- get_named("slot", "counts")
output_root  <- get_named("output_root",
  "/groups/ofircohen-group/users/michalnu_yuvat/project_breast/exports_bassez")

cat("Loading: ", rds_path, "\n", sep="")
obj <- readRDS(rds_path)
DefaultAssay(obj) <- assay

md <- obj@meta.data
auto_pick <- function(candidates, given, kind) {
  if (nzchar(given)) {
    if (!(given %in% colnames(md))) stop("Column '", given, "' not in metadata.")
    return(given)
  }
  hit <- intersect(candidates, colnames(md))
  if (length(hit) == 0) stop("Could not auto-detect ", kind, " column. Pass --", kind, "_col=...")
  hit[1]
}
sample_col <- auto_pick(
  c("patient_id","patient","sample","sample_id","donor","orig.ident","PatientID","Patient"),
  sample_col, "sample")
celltype_col <- auto_pick(
  c("cellType","cell_type","celltype","CellType","Cell_type","annotation",
    "Annotation","broad_celltype","cell_type_major","type"),
  celltype_col, "celltype")

cat("Using sample col = ", sample_col, ", celltype col = ", celltype_col, "\n", sep="")

mask <- (md[[sample_col]] == sample_val) & (md[[celltype_col]] == celltype_val)
cells_keep <- rownames(md)[mask]
cat("Subset: sample=", sample_val, " celltype=", celltype_val,
    " -> ", length(cells_keep), " cells\n", sep="")
if (length(cells_keep) == 0) {
  cat("Available top-25:\n")
  print(head(sort(table(md[[sample_col]], md[[celltype_col]]), decreasing=TRUE), 25))
  stop("Empty subset.")
}

sub_obj <- subset(obj, cells = cells_keep)
counts <- GetAssayData(sub_obj, assay = assay, slot = slot)
counts <- as(counts, "CsparseMatrix")
cat("Counts matrix: ", nrow(counts), " genes x ", ncol(counts), " cells, ",
    length(counts@x), " non-zero entries\n", sep="")

safe <- function(x) gsub("[^A-Za-z0-9._-]", "_", x)
out_dir <- file.path(output_root,
  paste0("patient=", safe(sample_val), "__celltype=", safe(celltype_val)))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("Writing: ", file.path(out_dir, "expr.mtx"), "\n", sep="")
writeMM(counts, file.path(out_dir, "expr.mtx"))
writeLines(rownames(counts), file.path(out_dir, "genes.csv"))
writeLines(colnames(counts), file.path(out_dir, "cells.csv"))
sub_md <- sub_obj@meta.data
sub_md$cell_id <- rownames(sub_md)
write.csv(sub_md, file.path(out_dir, "meta.csv"), row.names = FALSE)

cat("\nDone. Output dir: ", out_dir, "\n", sep="")
