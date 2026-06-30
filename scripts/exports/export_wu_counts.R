#!/usr/bin/env Rscript
# Single-shot exporter for Griffiths2021_Breast_10X.RDS.
# Loads RDS once, exports every (sample, celltype) pair with >= MIN_CELLS cells,
# excluding equivocal-like labels and any specified by --skip_celltype=...
suppressPackageStartupMessages({ library(Seurat); library(Matrix) })

args <- commandArgs(trailingOnly = TRUE)
get_named <- function(key, default) {
  hit <- grep(paste0("^--", key, "="), args, value = TRUE)
  if (length(hit) == 0) return(default)
  sub(paste0("^--", key, "="), "", hit[1])
}

rds_path     <- get_named("rds", "RDS files/Griffiths2021_Breast_10X.RDS")
output_root  <- get_named("output_root", "exports_griffiths")
sample_col   <- get_named("sample_col", "")     # auto-detect if empty
celltype_col <- get_named("celltype_col", "")   # auto-detect if empty
assay        <- get_named("assay", "RNA")
slot         <- get_named("slot", "counts")
min_cells    <- as.integer(get_named("min_cells", "200"))
skip_csv     <- get_named("skip_celltype", "equivocal")  # comma-separated
skip_list    <- trimws(unlist(strsplit(skip_csv, ",")))

cat("== Griffiths export ==\n")
cat("rds=", rds_path, "\nout=", output_root, "\nmin_cells=", min_cells,
    "\nskip_celltype=", paste(skip_list, collapse="|"), "\n", sep="")

t0 <- Sys.time()
obj <- readRDS(rds_path)
cat("Loaded in ", round(as.numeric(difftime(Sys.time(), t0, units="secs")),1), " secs\n", sep="")
DefaultAssay(obj) <- assay
md <- obj@meta.data

auto_pick <- function(cands, given, kind) {
  if (nzchar(given)) {
    if (!(given %in% colnames(md))) stop("Column '", given, "' not in metadata.")
    return(given)
  }
  hit <- intersect(cands, colnames(md))
  if (length(hit) == 0) stop("Auto-detect failed for ", kind, ". Cols: ",
                              paste(colnames(md), collapse=","))
  hit[1]
}
sample_col   <- auto_pick(c("patient_id","patient","sample","sample_id","donor","orig.ident",
                            "PatientID","Patient","sampleID"), sample_col, "sample")
celltype_col <- auto_pick(c("cellType","cell_type","celltype","CellType","Cell_type",
                            "annotation","Annotation","broad_celltype","cell_type_major",
                            "type","majority_cell_type","Consensus_Cell_Type"),
                          celltype_col, "celltype")
cat("Using sample_col=", sample_col, " celltype_col=", celltype_col, "\n", sep="")

ct_tab <- table(md[[sample_col]], md[[celltype_col]])
cat("\nSample x Celltype cross-tab (first 10 rows):\n")
print(head(ct_tab, 10))

safe <- function(x) gsub("[^A-Za-z0-9._-]", "_", x)

manifest <- list()
exported <- 0; skipped_size <- 0; skipped_excl <- 0
for (s in rownames(ct_tab)) {
  for (ct in colnames(ct_tab)) {
    n <- ct_tab[s, ct]
    if (n < min_cells) { skipped_size <- skipped_size + 1; next }
    # case-insensitive equivocal-like exclusion
    if (any(tolower(ct) == tolower(skip_list)) ||
        grepl("(?i)equivocal", ct, perl=TRUE)) {
      skipped_excl <- skipped_excl + 1; next
    }
    out_dir <- file.path(output_root, paste0("patient=", safe(s), "__celltype=", safe(ct)))
    if (file.exists(file.path(out_dir, "expr.mtx"))) {
      cat("[SKIP exists] ", s, "/", ct, "\n", sep=""); next
    }
    mask <- (md[[sample_col]] == s) & (md[[celltype_col]] == ct)
    cells_keep <- rownames(md)[mask]
    if (length(cells_keep) < min_cells) { skipped_size <- skipped_size + 1; next }
    cat(sprintf("[%4d cells] %s / %s\n", length(cells_keep), s, ct))
    sub_obj <- subset(obj, cells = cells_keep)
    counts <- as(GetAssayData(sub_obj, assay=assay, slot=slot), "CsparseMatrix")
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    writeMM(counts, file.path(out_dir, "expr.mtx"))
    writeLines(rownames(counts), file.path(out_dir, "genes.csv"))
    writeLines(colnames(counts), file.path(out_dir, "cells.csv"))
    sub_md <- sub_obj@meta.data; sub_md$cell_id <- rownames(sub_md)
    write.csv(sub_md, file.path(out_dir, "meta.csv"), row.names = FALSE)
    manifest[[length(manifest)+1]] <- data.frame(
      sample=s, celltype=ct, n_cells=length(cells_keep),
      n_genes=nrow(counts), stringsAsFactors=FALSE)
    exported <- exported + 1
  }
}

if (length(manifest) > 0) {
  mf <- do.call(rbind, manifest)
  out_tsv <- file.path("results/wu2021/summaries", "exports_inventory.tsv")
  dir.create(dirname(out_tsv), recursive=TRUE, showWarnings=FALSE)
  write.table(mf, out_tsv, sep="\t", row.names=FALSE, quote=FALSE)
  cat("\n[WROTE] ", out_tsv, " (", nrow(mf), " groups)\n", sep="")
}
cat(sprintf("\nExported=%d  skipped_size=%d  skipped_excl=%d\n",
            exported, skipped_size, skipped_excl))
cat("Done.\n")
