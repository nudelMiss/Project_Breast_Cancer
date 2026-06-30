#!/usr/bin/env Rscript
suppressPackageStartupMessages({ library(Seurat) })

args <- commandArgs(trailingOnly = TRUE)
rds_path <- if (length(args) >= 1) args[1] else
  "/groups/ofircohen-group/users/michalnu_yuvat/project_breast/RDS files/Bassez2021_Breast.RDS"

cat("Loading: ", rds_path, "\n", sep = "")
obj <- readRDS(rds_path)

cat("\n=== Object class ===\n"); print(class(obj))
cat("Dimensions (features x cells): ", paste(dim(obj), collapse = " x "), "\n")

cat("\n=== Assays ===\n"); print(Assays(obj))
cat("Default assay: ", DefaultAssay(obj), "\n")

cat("\n=== Metadata columns ===\n"); print(colnames(obj@meta.data))

md <- obj@meta.data
candidate_sample_cols <- intersect(
  c("patient_id","patient","sample","sample_id","donor","orig.ident","PatientID","Patient"),
  colnames(md))
candidate_celltype_cols <- intersect(
  c("cellType","cell_type","celltype","CellType","Cell_type","annotation",
    "Annotation","broad_celltype","cell_type_major","type","ident","seurat_clusters"),
  colnames(md))

cat("\nCandidate sample columns: ", paste(candidate_sample_cols, collapse=", "), "\n")
cat("Candidate celltype columns: ", paste(candidate_celltype_cols, collapse=", "), "\n")

cat("\n=== head(meta.data) ===\n"); print(head(md, 3))

if (length(candidate_sample_cols) > 0 && length(candidate_celltype_cols) > 0) {
  sample_col <- candidate_sample_cols[1]
  ct_col <- candidate_celltype_cols[1]
  cat("\nUsing sample col = ", sample_col, ", celltype col = ", ct_col, "\n", sep="")

  cat("\n=== Sample counts ===\n"); print(table(md[[sample_col]]))
  cat("\n=== Celltype counts ===\n"); print(table(md[[ct_col]]))
  cat("\n=== Sample x Celltype cross-tab ===\n")
  ct <- table(md[[sample_col]], md[[ct_col]])
  print(ct)

  tcell_pattern <- "(?i)t[- _]?cell|^T$|CD4|CD8|Treg"
  tcell_celltypes <- grep(tcell_pattern, colnames(ct), value = TRUE, perl = TRUE)
  cat("\nDetected T-cell-like celltypes: ", paste(tcell_celltypes, collapse=", "), "\n")

  if (length(tcell_celltypes) > 0) {
    tcell_counts <- rowSums(ct[, tcell_celltypes, drop=FALSE])
    tcell_counts <- sort(tcell_counts, decreasing=TRUE)
    cat("\n=== T-cell counts per sample (top 25) ===\n"); print(head(tcell_counts, 25))
    target <- 4500
    diffs <- abs(tcell_counts - target)
    best <- names(tcell_counts)[which.min(diffs)]
    cat("\nClosest sample to ~", target, " T-cells: ", best,
        " (n=", tcell_counts[best], ")\n", sep="")
  }
}
cat("\nDone.\n")
