#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(optparse)
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
    c("--out-bed"),
    type = "character",
    dest = "out_bed",
    help = "Output BED file"
  ),
  make_option(
    c("--chain-file"),
    type = "character",
    dest = "chain_file",
    default = NULL,
    help = "Path to hg19ToHg38 chain file (required if --genome hg38)"
  )
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$genome) || is.null(opt$out_bed)) {
  stop("Must provide --genome and --out-bed")
}

genome     <- opt$genome
out_bed    <- opt$out_bed
chain_file <- opt$chain_file

# Always use hg19 annotation (450K was designed on hg19)
anno_pkg <- "IlluminaHumanMethylation450kanno.ilmn12.hg19"

required_pkgs <- c(
  anno_pkg,
  "GenomicRanges",
  "rtracklayer"
)

for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    BiocManager::install(pkg, ask = FALSE, update = FALSE)
  }
}

suppressPackageStartupMessages({
  library(IlluminaHumanMethylation450kanno.ilmn12.hg19)
  library(GenomicRanges)
})

data("Locations")

# Build GRanges (hg19 coordinates)
gr_hg19 <- GRanges(
  seqnames = Locations$chr,
  ranges   = IRanges(start = Locations$pos, end = Locations$pos),
  probe    = rownames(Locations)
)

if (genome == "hg19") {
  
  gr_final <- gr_hg19
  
} else if (genome == "hg38") {
  
  if (is.null(chain_file)) {
    stop("Must provide --chain-file when --genome hg38")
  }
  
  if (!file.exists(chain_file)) {
    stop(paste("Chain file not found:", chain_file))
  }
  
  suppressPackageStartupMessages(library(rtracklayer))
  
  message("Using chain file: ", chain_file)
  
  chain <- import.chain(chain_file)
  
  gr_lift <- liftOver(gr_hg19, chain)
  gr_lift <- unlist(gr_lift)
  
  # Drop unmapped probes
  gr_lift <- gr_lift[!is.na(start(gr_lift))]
  
  if (length(gr_lift) == 0) {
    stop("No probes mapped during liftOver. Check chain file.")
  }
  
  gr_final <- gr_lift
  
} else {
  stop(paste("Unsupported genome:", genome))
}

# Convert to BED (0-based start, 1-based end)
locations_bed <- data.frame(
  chr   = as.character(seqnames(gr_final)),
  start = start(gr_final) - 1,
  end   = end(gr_final),
  probe = mcols(gr_final)$probe
)

write.table(
  locations_bed,
  file = out_bed,
  sep = "\t",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)

message("Wrote HM450K BED: ", out_bed)