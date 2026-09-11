#!/usr/bin/env Rscript

parse_flag_args <- function(args) {
  parsed <- list()
  idx <- 1L
  while (idx <= length(args)) {
    key <- args[[idx]]
    if (!startsWith(key, "--")) {
      stop(paste("Unexpected positional argument:", key))
    }
    if (idx == length(args)) {
      stop(paste("Missing value for argument:", key))
    }
    value <- args[[idx + 1L]]
    parsed[[sub("^--", "", key)]] <- value
    idx <- idx + 2L
  }
  parsed
}

opt <- parse_flag_args(commandArgs(trailingOnly = TRUE))

if (is.null(opt[["input-bed"]]) || is.null(opt[["output-bed"]]) || is.null(opt[["chain-file"]])) {
  stop("Must provide --input-bed, --output-bed, and --chain-file")
}

input_bed <- opt[["input-bed"]]
output_bed <- opt[["output-bed"]]
chain_file <- opt[["chain-file"]]

if (!file.exists(input_bed)) {
  stop(paste("Input BED not found:", input_bed))
}

if (!file.exists(chain_file)) {
  stop(paste("Chain file not found:", chain_file))
}

r_version_parts <- strsplit(as.character(getRversion()), ".", fixed = TRUE)[[1]]
r_series <- paste(r_version_parts[[1]], r_version_parts[[2]], sep = ".")
user_lib <- path.expand(file.path("~", "R", paste0("library-", r_series)))
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(unique(c(user_lib, .Library.site, .Library)))

package_matches_r_series <- function(pkg, expected_r_series) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    return(FALSE)
  }

  pkg_desc <- tryCatch(
    utils::packageDescription(pkg),
    error = function(err) NULL
  )
  if (is.null(pkg_desc)) {
    return(FALSE)
  }

  built_field <- pkg_desc[["Built"]]
  if (is.null(built_field) || is.na(built_field) || !nzchar(built_field)) {
    return(TRUE)
  }

  built_match <- regmatches(
    built_field,
    regexpr("[0-9]+\\.[0-9]+", built_field)
  )
  if (!length(built_match) || is.na(built_match) || !nzchar(built_match)) {
    return(TRUE)
  }

  identical(built_match, expected_r_series)
}

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org")
}

target_bioc_version <- local({
  r_version <- getRversion()
  if (r_version >= "4.5") {
    "3.22"
  } else if (r_version >= "4.4") {
    "3.20"
  } else {
    NULL
  }
})

if (!is.null(target_bioc_version)) {
  current_bioc_version <- tryCatch(
    as.character(BiocManager::version()),
    error = function(err) NA_character_
  )
  if (is.na(current_bioc_version) || current_bioc_version != target_bioc_version) {
    BiocManager::install(
      version = target_bioc_version,
      ask = FALSE,
      update = FALSE,
      force = TRUE
    )
  }
  options(repos = BiocManager::repositories(version = target_bioc_version))
}

required_pkgs <- c("GenomicRanges", "IRanges", "rtracklayer")
for (pkg in required_pkgs) {
  if (!package_matches_r_series(pkg, r_series)) {
    BiocManager::install(
      pkg,
      version = target_bioc_version,
      ask = FALSE,
      update = FALSE,
      force = TRUE
    )
  }
}

suppressPackageStartupMessages({
  library(GenomicRanges)
  library(IRanges)
  library(rtracklayer)
})

bed_df <- tryCatch(
  read.delim(
    input_bed,
    sep = "\t",
    header = FALSE,
    quote = "",
    comment.char = "",
    stringsAsFactors = FALSE
  ),
  error = function(err) {
    stop(paste("Failed reading input BED:", err$message))
  }
)

if (ncol(bed_df) < 3) {
  stop("Input BED must have at least 3 columns")
}

candidate_value_column <- NULL
if (ncol(bed_df) >= 4) {
  candidate_value_column <- suppressWarnings(as.numeric(bed_df[[4]]))
}
has_value_column <- !is.null(candidate_value_column) && any(!is.na(candidate_value_column))

if (has_value_column) {
  bed_df <- bed_df[, 1:4, drop = FALSE]
  colnames(bed_df) <- c("chrom", "start", "end", "value")
  bed_df$value <- candidate_value_column
} else {
  bed_df <- bed_df[, 1:3, drop = FALSE]
  colnames(bed_df) <- c("chrom", "start", "end")
}

bed_df$start <- as.integer(bed_df$start)
bed_df$end <- as.integer(bed_df$end)
bed_df <- bed_df[!is.na(bed_df$start) & !is.na(bed_df$end), , drop = FALSE]
bed_df <- bed_df[bed_df$end > bed_df$start, , drop = FALSE]

if (has_value_column) {
  bed_df <- bed_df[!is.na(bed_df$value), , drop = FALSE]
}

if (nrow(bed_df) == 0) {
  file.create(output_bed)
  quit(save = "no")
}

if (has_value_column) {
  gr_input <- GRanges(
    seqnames = bed_df$chrom,
    ranges = IRanges(start = bed_df$start + 1L, end = bed_df$end),
    value = bed_df$value
  )
} else {
  gr_input <- GRanges(
    seqnames = bed_df$chrom,
    ranges = IRanges(start = bed_df$start + 1L, end = bed_df$end)
  )
}

chain <- import.chain(chain_file)
gr_lifted <- unlist(liftOver(gr_input, chain), use.names = FALSE)

if (length(gr_lifted) == 0) {
  file.create(output_bed)
  quit(save = "no")
}

if (has_value_column) {
  out_df <- data.frame(
    chrom = as.character(seqnames(gr_lifted)),
    start = start(gr_lifted) - 1L,
    end = end(gr_lifted),
    value = as.numeric(mcols(gr_lifted)$value),
    stringsAsFactors = FALSE
  )
} else {
  out_df <- data.frame(
    chrom = as.character(seqnames(gr_lifted)),
    start = start(gr_lifted) - 1L,
    end = end(gr_lifted),
    stringsAsFactors = FALSE
  )
}

out_df <- out_df[!is.na(out_df$start) & !is.na(out_df$end), , drop = FALSE]
out_df <- out_df[out_df$end > out_df$start, , drop = FALSE]
if (has_value_column) {
  out_df <- out_df[!is.na(out_df$value), , drop = FALSE]
}

write.table(
  out_df,
  file = output_bed,
  sep = "\t",
  row.names = FALSE,
  col.names = FALSE,
  quote = FALSE
)
