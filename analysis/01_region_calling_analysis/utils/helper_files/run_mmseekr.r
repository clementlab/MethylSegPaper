#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
  library(MMSeekR)
  library(MMSeekR.data)
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
    c("--trainChr"),
    type = "character",
    dest = "trainChr",
    default = "chr1",
    help = "Training chromosome [default: %default]"
  ),
  make_option(
    c("--nCGbin"),
    type = "integer",
    dest = "nCGbin",
    default = 201,
    help = "CpG bin size (>=101) [default: %default]"
  ),
  make_option(
    c("--num_cores"),
    type = "integer",
    dest = "num_cores",
    default = 1,
    help = "Number of cores [default: %default]"
  )
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$genome) || is.null(opt$sample)) {
  stop("Must provide --genome and --sample")
}

sample   <- opt$sample
genome   <- opt$genome
out_dir  <- opt$out_dir
trainChr <- opt$trainChr
nCGbin   <- opt$nCGbin
num_cores <- opt$num_cores

sample_dir <- file.path(out_dir, sample)
prep_dir   <- file.path(sample_dir, "prep")
out_subdir <- file.path(sample_dir, "out")
log_dir    <- file.path(sample_dir, "logs")
dir.create(out_subdir, recursive = TRUE, showWarnings = FALSE)

log_file <- file.path(log_dir, "job_logs", "mmseekr.log")
log_con  <- file(log_file, open = "wt")

sink(log_con)
sink(log_con, type = "message")

message("========================================")
message("MMSeekR segmentation started")
message("Sample: ", sample)
message("Genome: ", genome)
message("TrainChr: ", trainChr)
message("nCGbin: ", nCGbin)
message("Timestamp: ", Sys.time())
message("========================================")

on.exit({
  message("========================================")
  message("MMSeekR segmentation finished")
  message("Timestamp: ", Sys.time())
  message("========================================")
  sink(type = "message")
  sink()
  close(log_con)
}, add = TRUE)

input_file <- file.path(prep_dir, "mmseekr_input.tsv")

if (!file.exists(input_file)) {
  stop("Input file not found: ", input_file)
}

message("Using input: ", input_file)

# Output prefix (MMSeekR writes multiple files based on this)
output_index <- file.path(out_subdir, sample)

message("Output prefix: ", output_index)

message("Starting MMSeekR segmentation")
start_time <- proc.time()

runMultiModel(
  genomeVersion = genome,
  fileName      = input_file,
  outputIndex   = output_index,
  trainChr      = trainChr,
  inputFormat   = "text",
  # nCGbin        = nCGbin,
  num.cores     = num_cores
)

elapsed <- proc.time() - start_time

message("MMSeekR completed")
message("Runtime:")
message(paste(capture.output(elapsed), collapse = "\n"))

message("Expected output files (prefix-based):")
message(output_index)

message("========================================")
