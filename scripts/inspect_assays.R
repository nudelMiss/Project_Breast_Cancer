suppressPackageStartupMessages({library(Seurat)})
a<-commandArgs(trailingOnly=TRUE); obj<-readRDS(a[1])
cat("class:",paste(class(obj),collapse=","),"\n")
cat("assays:",paste(Assays(obj),collapse=",")," default:",DefaultAssay(obj),"\n")
for (as in Assays(obj)) for (sl in c("counts","data")) {
  m<-tryCatch(GetAssayData(obj,assay=as,slot=sl),error=function(e)NULL)
  if(!is.null(m)&&prod(dim(m))>0) cat(sprintf("  %s/%s: %dx%d firstval=%s\n",as,sl,nrow(m),ncol(m),ifelse(length(m@x)>0,round(m@x[1],3),"empty")))
}
md<-obj@meta.data; cat("n_cells:",nrow(md),"\nmeta cols:",paste(colnames(md),collapse=","),"\n")
for(cn in colnames(md)){v<-md[[cn]];if((is.factor(v)||is.character(v))&&length(unique(v))<=30){cat("---",cn,"(",length(unique(v)),")---\n");print(sort(table(v),decreasing=TRUE))}}
cat("done\n")
