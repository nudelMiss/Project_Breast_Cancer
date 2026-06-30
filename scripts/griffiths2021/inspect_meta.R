suppressPackageStartupMessages({library(Seurat)})
args <- commandArgs(trailingOnly=TRUE)
obj <- readRDS(args[1])
md <- obj@meta.data
cat("=== meta.data columns ===\n"); print(colnames(md))
cat("\n=== n_cells ===", nrow(md), "\n")
for (cn in colnames(md)) {
  v <- md[[cn]]
  if (is.factor(v) || is.character(v)) {
    u <- unique(v)
    if (length(u) <= 40) {
      cat("\n--- ", cn, " (", length(u), " levels) ---\n", sep="")
      print(sort(table(v), decreasing=TRUE))
    } else cat("\n--- ", cn, " (", length(u), " levels: too many) ---\n", sep="")
  }
}
cat("\ndone\n")
