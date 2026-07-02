#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(MethylSeekR)
  library(GenomicRanges)
  library(BiocManager)
})

option_list <- list(
  make_option(
    c("--genome"),
    type = "character",
    dest = "genome",
    help = "Genome build: hg19 or hg38"
  ),
  make_option(
    c("--sample"),
    type = "character",
    dest = "sample",
    help = "Sample ID"
  ),
  make_option(
    c("--out_dir"),
    type = "character",
    dest = "out_dir",
    default = ".",
    help = "Output directory [default: %default]"
),
 make_option(
    c("--cores"),
    type = "integer",
    dest = "cores",
    default = 6,
    help = "Number of cores for parallel processing [default: %default]"
 )
)

opt <- parse_args(OptionParser(option_list=option_list))

if (is.null(opt$genome) || is.null(opt$sample)) {
  stop("Must provide --genome and --sample")
}

sample <- opt$sample
genome <- opt$genome
out_dir <- opt$out_dir
cores <- opt$cores

log_file <- paste0( "methylseekr.log")
sample_dir <- file.path(out_dir, sample)
log_dir    <- file.path(sample_dir, "logs")
log_file <- file.path(log_dir, "job_logs", log_file)

log_con <- file(log_file, open = "wt")
sink(log_con)
sink(log_con, type = "message")

message("========================================")
message("MethylSeekR PMD segmentation started")
message("Sample: ", sample)
message("Genome: ", genome)
message("Timestamp: ", Sys.time())
message("========================================")

on.exit({
  message("========================================")
  message("MethylSeekR PMD segmentation finished")
  message("Timestamp: ", Sys.time())
  message("========================================")
  sink(type = "message")
  sink()
  close(log_con)
}, add = TRUE)

wgbs_file <- paste0("wgbs.tsv")
wgbs_file <- file.path(out_dir, sample, "prep", wgbs_file)
if (!file.exists(wgbs_file)) {
  stop("Input WGBS file not found: ", wgbs_file)
}

message("Reading WGBS input: ", wgbs_file)

meth_data <- read.table(
  wgbs_file,
  header = FALSE,
  sep = "\t",
  stringsAsFactors = FALSE
)

colnames(meth_data) <- c("chr", "start", "end", "meth", "coverage")

message("Loaded WGBS table")
message("Rows: ", nrow(meth_data))

gr <- GRanges(
  seqnames = meth_data$chr,
  ranges   = IRanges(start = meth_data$start, end = meth_data$start),
  strand   = "*",
  T        = meth_data$coverage,
  M        = meth_data$meth
)

gr <- gr[gr$T > 0 & gr$M <= gr$T]

gr <- sort(gr)

message("CpGs after filtering: ", length(gr))

if (length(gr) == 0) {
  stop("No valid CpGs after filtering")
}

if (genome == "hg19") {
  pkg <- "BSgenome.Hsapiens.UCSC.hg19"
} else if (genome == "hg38") {
  pkg <- "BSgenome.Hsapiens.UCSC.hg38"
} else {
  stop("Unsupported genome: ", genome)
}

message("Using BSgenome package: ", pkg)

if (!requireNamespace(pkg, quietly = TRUE)) {
  message("Installing BSgenome package: ", pkg)
  BiocManager::install(pkg, ask = FALSE, update = FALSE)
}

suppressPackageStartupMessages(
  library(pkg, character.only = TRUE)
)

Hsapiens <- get(pkg)
sLengths <- seqlengths(Hsapiens)

message("Loaded sequence lengths")
message("Example seqlengths:")
message(paste(capture.output(head(sLengths)), collapse = "\n"))

message("Starting PMD segmentation")
start_time <- proc.time()

PMDsegments.gr <- segmentPMDs(
  m           = gr,
  chr.sel     = "chr1",   # match notebook behavior
  seqLengths = sLengths,
  num.cores  = cores
)

elapsed <- proc.time() - start_time

message("PMD segmentation completed")
message("Runtime:")
message(paste(capture.output(elapsed), collapse = "\n"))

seg_types <- table(mcols(PMDsegments.gr)$type)

message("Segment type counts:")
message(paste(capture.output(seg_types), collapse = "\n"))

gr_out <- sprintf("methylseekr_PMDs.gr.rds")
bed_out <- sprintf("methylseekr_PMDs.bed")

gr_out <- file.path(out_dir, sample, "out", gr_out)
bed_out <- file.path(out_dir, sample, "out", bed_out)

savePMDSegments(
  PMDs            = PMDsegments.gr,
  GRangesFilename = gr_out,
  TableFilename   = bed_out
)

message("Saved GRanges: ", gr_out)
message("Saved BED: ", bed_out)

plot_file <- sprintf("methylseekr_PMDs_plot.pdf")
plot_file <- file.path(out_dir, sample, "out", plot_file)
pdf(plot_file)
plotPMDSegmentation(m = gr, segs = PMDsegments.gr)
dev.off()

message("Saved plot: ", plot_file)
