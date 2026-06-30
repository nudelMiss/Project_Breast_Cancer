#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(Seurat); library(Matrix)})
args <- commandArgs(trailingOnly=TRUE)
getn <- function(k,d){h<-grep(paste0("^--",k,"="),args,value=TRUE); if(length(h)==0) d else sub(paste0("^--",k,"="),"",h[1])}
rds<-getn("rds",""); outroot<-getn("output_root","exports"); sample_col<-getn("sample_col",""); celltype_col<-getn("celltype_col","")
assay<-getn("assay","RNA"); slot<-getn("slot","counts"); min_cells<-as.integer(getn("min_cells","200"))
skip_list<-trimws(unlist(strsplit(getn("skip_celltype","equivocal"),","))); inv_out<-getn("inventory_out","")
obj<-readRDS(rds); DefaultAssay(obj)<-assay; md<-obj@meta.data
auto<-function(cands,given){if(nzchar(given)){if(!(given%in%colnames(md)))stop("col not found: ",given);return(given)};h<-intersect(cands,colnames(md));if(length(h)==0)stop("autodetect fail");h[1]}
sample_col<-auto(c("patient_id","patient","sample","sample_id","donor","sampleID","orig.ident"),sample_col)
celltype_col<-auto(c("cellType","cell_type","celltype","CellType","annotation"),celltype_col)
M<-as(GetAssayData(obj,assay=assay,slot=slot),"CsparseMatrix"); genes<-rownames(M)
cat("sample_col=",sample_col," celltype_col=",celltype_col," matrix=",nrow(M),"x",ncol(M),"\n",sep="")
samp<-as.character(md[[sample_col]]); ctv<-as.character(md[[celltype_col]])
stopifnot(length(samp)==ncol(M))
safe<-function(x) gsub("[^A-Za-z0-9._-]","_",x)
combos<-unique(data.frame(s=samp,ct=ctv,stringsAsFactors=FALSE)); manifest<-list(); exported<-0; sks<-0; ske<-0
for(r in seq_len(nrow(combos))){
  s<-combos$s[r]; ct<-combos$ct[r]
  if(any(tolower(ct)==tolower(skip_list))||grepl("(?i)equivocal",ct,perl=TRUE)){ske<-ske+1;next}
  mask<-(samp==s)&(ctv==ct); n<-sum(mask)
  if(n<min_cells){sks<-sks+1;next}
  od<-file.path(outroot,paste0("patient=",safe(s),"__celltype=",safe(ct)))
  if(file.exists(file.path(od,"expr.mtx"))){cat("[exists]",s,ct,"\n");next}
  m<-M[,mask,drop=FALSE]
  dir.create(od,recursive=TRUE,showWarnings=FALSE)
  writeMM(m,file.path(od,"expr.mtx")); writeLines(genes,file.path(od,"genes.csv")); writeLines(colnames(m),file.path(od,"cells.csv"))
  exported<-exported+1; manifest[[length(manifest)+1]]<-data.frame(sample=s,celltype=ct,n_cells=n,n_genes=nrow(m))
  cat(sprintf("[%5d] %s / %s\n",n,s,ct))
}
if(length(manifest)>0 && nzchar(inv_out)){d<-do.call(rbind,manifest);dir.create(dirname(inv_out),recursive=TRUE,showWarnings=FALSE);write.table(d,inv_out,sep="\t",row.names=FALSE,quote=FALSE)}
cat(sprintf("Exported=%d skipped_size=%d skipped_excl=%d\n",exported,sks,ske))
