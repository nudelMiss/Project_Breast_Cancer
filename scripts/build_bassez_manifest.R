#!/usr/bin/env Rscript
suppressPackageStartupMessages({ library(Seurat) })
obj <- readRDS("/groups/ofircohen-group/users/michalnu_yuvat/project_breast/RDS files/Bassez2021_Breast.RDS")
md <- obj@meta.data
ct <- table(md$sampleID, md$cell_type)
df <- as.data.frame.matrix(ct)
df$sampleID <- rownames(df)

# Long-form: one row per (sample, celltype, n_cells)
long <- data.frame()
for (s in rownames(ct)) {
  for (c in colnames(ct)) {
    n <- ct[s, c]
    if (n >= 200) {
      long <- rbind(long, data.frame(sampleID = s, celltype = c, n_cells = as.integer(n)))
    }
  }
}
long <- long[order(-long$n_cells), ]

write.table(long, "bassez_groups_manifest.tsv", sep = "\t", quote = FALSE, row.names = FALSE)
cat("Wrote bassez_groups_manifest.tsv with", nrow(long), "groups (>=200 cells)\n")
cat("\nBreakdown by celltype:\n")
print(table(long$celltype))

# Also write T-cell-only manifest
tcell <- long[long$celltype == "T_cell", ]
writeLines(paste0("patient=", tcell$sampleID, "__celltype=T_cell"),
           "bassez_tcells_groups.txt")
cat("\nT-cell-only manifest:", nrow(tcell), "samples written to bassez_tcells_groups.txt\n")
