import argparse
import concurrent.futures
import multiprocessing
import os
import shutil
import sys
import tempfile
from pathlib import Path

matplotlib_cache_dir = Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"
matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(matplotlib_cache_dir)
env_bin = str(Path(sys.executable).resolve().parent)
path_entries = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
if env_bin not in path_entries:
    os.environ["PATH"] = os.pathsep.join([env_bin] + path_entries)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pybedtools import BedTool
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT, ANALYSIS_DIR):
    import_str = str(import_path)
    if import_str not in sys.path:
        sys.path.insert(0, import_str)

from repo_paths import LAD_RESULTS_DIR, REFERENCE_DATA_DIR, REGION_CALLING_RESULTS_DIR

from lad_analysis_utils import (
    build_binary_signal_df,
    build_tool_region_path,
    clean_and_merge_intervals,
    clean_signal_df,
    ensure_chrom_sizes,
    filter_regions_for_deeptools,
    load_tool_regions,
    read_bed_intervals,
    read_bedgraph_signal,
    read_chrom_sizes,
    resolve_sample_genome,
    run_deeptools_profile,
    run_liftover_bed,
    sample_to_sample_id,
    threshold_label,
    validate_bigwig,
    write_bed,
    write_bedgraph,
    write_bigwig,
)

sns.set_theme(style="whitegrid")

DEFAULT_SEGMENTATION_RESULTS_PATH = REGION_CALLING_RESULTS_DIR
DEFAULT_OUTPUT_DIR = LAD_RESULTS_DIR
DEFAULT_SELECTED_SAMPLES = [
    "ESO26.wgbs",
    "TE5.wgbs",
    "WGBS_colon-primary-tumor_1_meth",
]
MIN_LAD_BETA = 0.0
DEFAULT_PRIMARY_WINDOW_BP = 150_000
LAD_OVERLAP_THRESHOLDS_BP = [1, 150_000]
DEFAULT_PROFILE_REGION_BODY_BP = 1_000_000
DEFAULT_PROFILE_BIN_BP = 5_000
DEFAULT_LAD_NULL_PERMUTATIONS = 50
DEFAULT_LAD_NULL_SEED = 0
DEFAULT_LAD_INTERVAL_TRACK_PATH = REFERENCE_DATA_DIR / "LAD_intervals.bed"
DEFAULT_LAMINB1_SIGNAL_TRACK_PATH = REFERENCE_DATA_DIR / "laminB1_signal.bedGraph"
DEFAULT_LIFTOVER_SCRIPT_PATH = REFERENCE_DATA_DIR / "liftover_bed.r"
DEFAULT_HG19_TO_HG38_CHAIN = REFERENCE_DATA_DIR / "hg19ToHg38.over.chain"
SAMPLE_METH_FILE_PARTS = ["methylseg", "{sample}", "prep", "wgbs.beta"]
DNMTOOLS_METHYLSEG_COMPARISONS = [
    {"comparison_tool": "dnmtools", "reference_methylseg_tool": "methylseg"},
    {"comparison_tool": "dnmtools_array", "reference_methylseg_tool": "methylseg_hm450k"},
]
NULL_MODEL_CANONICAL_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
CHROM_SIZES_PATHS = {
    "hg19": REFERENCE_DATA_DIR / "hg19.chrom.sizes",
    "hg38": REFERENCE_DATA_DIR / "hg38.chrom.sizes",
}

TOOL_REGISTRY = [
    {
        "tool": "methylseg",
        "tool_label": "MethylSeg WGBS",
        "parser_family": "methylseg",
        "tool_family": "methylseg",
        "platform": "wgbs",
        "region_type": "PMD",
        "rank_order": 0,
        "path_parts": ["methylseg", "{sample}", "out", "wgbs", "summary_files", "segments_cleaned_PMD.bed"],
    },
    {
        "tool": "methylseg_hm450k",
        "tool_label": "MethylSeg HM450K",
        "parser_family": "methylseg",
        "tool_family": "methylseg",
        "platform": "hm450k",
        "region_type": "PMD",
        "rank_order": 1,
        "path_parts": ["methylseg", "{sample}", "out", "hm450k", "summary_files", "segments_cleaned_PMD.bed"],
    },
    {
        "tool": "methylseekr",
        "tool_label": "MethylSeekR",
        "parser_family": "methylseekr",
        "tool_family": "methylseekr",
        "platform": "wgbs",
        "region_type": "PMD",
        "rank_order": 2,
        "path_parts": ["methylseekr", "{sample}", "out", "methylseekr_PMDs.bed"],
    },
    {
        "tool": "dnmtools",
        "tool_label": "DNMTools",
        "parser_family": "dnmtools",
        "tool_family": "dnmtools",
        "platform": "wgbs",
        "region_type": "PMD",
        "rank_order": 3,
        "path_parts": ["dnmtools", "{sample}", "out", "dnmtools_PMDs.bed"],
    },
    {
        "tool": "dnmtools_array",
        "tool_label": "DNMTools Array",
        "parser_family": "dnmtools",
        "tool_family": "dnmtools",
        "platform": "hm450k",
        "region_type": "PMD",
        "rank_order": 4,
        "path_parts": ["dnmtools", "{sample}", "out", "arraymode.dnmtools_PMDs.bed"],
    },
    {
        "tool": "dnmtools_pmr",
        "tool_label": "DNMTools PMR",
        "parser_family": "dnmtools",
        "tool_family": "dnmtools",
        "platform": "wgbs",
        "region_type": "PMR",
        "rank_order": 5,
        "path_parts": ["dnmtools", "{sample}", "out", "pmr.dnmtools_PMDs.bed"],
    },
    {
        "tool": "mmseekr",
        "tool_label": "MMSeekR",
        "parser_family": "mmseekr",
        "tool_family": "mmseekr",
        "platform": "wgbs",
        "region_type": "PMD",
        "rank_order": 6,
        "path_parts": ["mmseekr", "{sample}", "out", "{sample}.multiModel.PMDs.bed"],
    },
    {
        "tool": "methyl_lasso",
        "tool_label": "MethylLasso",
        "parser_family": "methyl_lasso",
        "tool_family": "methyl_lasso",
        "platform": "wgbs",
        "region_type": "PMD",
        "rank_order": 7,
        "path_parts": ["methyl_lasso", "{sample}", "out", "{sample}_pmd.tsv"],
    },
]
TOOL_CONFIG_BY_NAME = {config["tool"]: config for config in TOOL_REGISTRY}
REQUIRED_TOOLS = [config["tool"] for config in sorted(TOOL_REGISTRY, key=lambda config: config["rank_order"])]
REQUIRED_TOOL_SET = set(REQUIRED_TOOLS)
LAD_OVERLAP_METRICS = [
    "pct_regions_overlapping_lads",
    "pct_regions_overlapping_lads_gte_150kb",
    "avg_distance_to_nearest_lad",
    "avg_distance_to_nearest_lad_boundary",
    "avg_distance_to_nearest_lad_boundary_non_overlapping",
    "avg_lads_per_overlapping_pmd",
    "avg_lad_per_pmd",
    "pct_regions_with_boundary_within_150kb_of_lad_boundary",
    "pct_non_overlapping_regions_within_150kb_of_lad_boundary",
    "pct_regions_sharing_lad",
    "pmd_coverage_by_lads",
    "pct_lads_overlapping",
    "lad_coverage_by_pmds",
]
LAD_ASSOCIATION_METRICS = list(LAD_OVERLAP_METRICS)
LAD_METRIC_LABELS = {
    "pct_regions_overlapping_lads": "Fraction of regions overlapping LADs",
    "pct_regions_overlapping_lads_gte_150kb": "Fraction of regions overlapping LADs by at least 150 kb",
    "avg_distance_to_nearest_lad": "Average distance to nearest LAD (bp)",
    "avg_distance_to_nearest_lad_boundary": "Average distance to nearest LAD boundary (bp)",
    "avg_distance_to_nearest_lad_boundary_non_overlapping": "Average distance to nearest LAD boundary for non-overlapping regions (bp)",
    "avg_lads_per_overlapping_pmd": "Average LAD overlaps per overlapping PMD",
    "avg_lad_per_pmd": "Average LAD overlaps per PMD",
    "pct_regions_with_boundary_within_150kb_of_lad_boundary": "Fraction of regions with a boundary within 150 kb of a LAD boundary",
    "pct_non_overlapping_regions_within_150kb_of_lad_boundary": "Fraction of non-overlapping regions with a boundary within 150 kb of a LAD boundary",
    "pct_regions_sharing_lad": "Fraction of LAD-overlapping regions sharing a LAD",
    "pmd_coverage_by_lads": "Fraction of region bases covered by LADs",
    "pct_lads_overlapping": "Fraction of LADs overlapping regions",
    "lad_coverage_by_pmds": "Fraction of LAD bases covered by regions",
}
LAD_METRIC_MODES = {
    "pct_regions_overlapping_lads": "whole_region_overlap",
    "pct_regions_overlapping_lads_gte_150kb": "whole_region_overlap",
    "avg_distance_to_nearest_lad": "nearest_lad_distance",
    "avg_distance_to_nearest_lad_boundary": "boundary_distance",
    "avg_distance_to_nearest_lad_boundary_non_overlapping": "boundary_distance_non_overlapping",
    "avg_lads_per_overlapping_pmd": "lad_overlap_count",
    "avg_lad_per_pmd": "lad_overlap_count",
    "pct_regions_with_boundary_within_150kb_of_lad_boundary": "boundary_distance",
    "pct_non_overlapping_regions_within_150kb_of_lad_boundary": "boundary_distance_non_overlapping",
    "pct_regions_sharing_lad": "shared_lad",
    "pmd_coverage_by_lads": "whole_region_overlap",
    "pct_lads_overlapping": "lad_coverage",
    "lad_coverage_by_pmds": "lad_coverage",
}
LAD_METRIC_HIGHER_IS_BETTER = {
    "pct_regions_overlapping_lads": True,
    "pct_regions_overlapping_lads_gte_150kb": True,
    "avg_distance_to_nearest_lad": False,
    "avg_distance_to_nearest_lad_boundary": False,
    "avg_distance_to_nearest_lad_boundary_non_overlapping": False,
    "avg_lads_per_overlapping_pmd": True,
    "avg_lad_per_pmd": True,
    "pct_regions_with_boundary_within_150kb_of_lad_boundary": True,
    "pct_non_overlapping_regions_within_150kb_of_lad_boundary": True,
    "pct_regions_sharing_lad": True,
    "pmd_coverage_by_lads": True,
    "pct_lads_overlapping": True,
    "lad_coverage_by_pmds": True,
}
LAD_METRIC_THRESHOLDS_BP = {
    "pct_regions_overlapping_lads": 1,
    "pct_regions_overlapping_lads_gte_150kb": 150_000,
    "avg_distance_to_nearest_lad": 0,
    "avg_distance_to_nearest_lad_boundary": 0,
    "avg_distance_to_nearest_lad_boundary_non_overlapping": 0,
    "avg_lads_per_overlapping_pmd": 0,
    "avg_lad_per_pmd": 0,
    "pct_regions_with_boundary_within_150kb_of_lad_boundary": 150_000,
    "pct_non_overlapping_regions_within_150kb_of_lad_boundary": 150_000,
    "pct_regions_sharing_lad": 1,
    "pmd_coverage_by_lads": 0,
    "pct_lads_overlapping": 1,
    "lad_coverage_by_pmds": 0,
}
NULL_METRICS = list(LAD_OVERLAP_METRICS)
NULL_METRIC_SHORT_NAME_MAP = {
    "pct_regions_overlapping_lads": "region_pct",
    "pct_regions_overlapping_lads_gte_150kb": "region_pct_150kb",
    "avg_distance_to_nearest_lad": "dist_lad",
    "avg_distance_to_nearest_lad_boundary": "dist_boundary",
    "avg_distance_to_nearest_lad_boundary_non_overlapping": "dist_boundary_nonoverlap",
    "avg_lads_per_overlapping_pmd": "avg_lads_overlap_pmd",
    "avg_lad_per_pmd": "avg_lads_pmd",
    "pct_regions_with_boundary_within_150kb_of_lad_boundary": "boundary_pct_150kb",
    "pct_non_overlapping_regions_within_150kb_of_lad_boundary": "boundary_pct_nonoverlap_150kb",
    "pct_regions_sharing_lad": "shared_lad_pct",
    "pmd_coverage_by_lads": "region_cov",
    "pct_lads_overlapping": "lad_pct",
    "lad_coverage_by_pmds": "lad_cov",
}
NULL_METRIC_PLOT_LABEL_MAP = {
    "pct_regions_overlapping_lads": "% regions overlapping LADs",
    "pct_regions_overlapping_lads_gte_150kb": "% regions with >=150 kb LAD overlap",
    "avg_distance_to_nearest_lad": "Avg distance to LAD",
    "avg_distance_to_nearest_lad_boundary": "Avg distance to LAD boundary",
    "avg_distance_to_nearest_lad_boundary_non_overlapping": "Avg boundary distance, non-overlap",
    "avg_lads_per_overlapping_pmd": "Avg LAD overlaps per overlapping PMD",
    "avg_lad_per_pmd": "Avg LAD overlaps per PMD",
    "pct_regions_with_boundary_within_150kb_of_lad_boundary": "% boundaries within 150 kb",
    "pct_non_overlapping_regions_within_150kb_of_lad_boundary": "% non-overlap boundaries within 150 kb",
    "pct_regions_sharing_lad": "% regions sharing LAD",
    "pmd_coverage_by_lads": "Region coverage by LADs",
    "pct_lads_overlapping": "% LADs overlapping",
    "lad_coverage_by_pmds": "LAD coverage by regions",
}
NULL_STAT_PLOT_SUFFIX_MAP = {
    "enrichment_vs_null": "enrich_vs_null",
    "z_score_vs_null": "z_vs_null",
}
NULL_METRIC_NULL_COLUMN_MAP = {
    metric: f"{NULL_METRIC_SHORT_NAME_MAP[metric]}_null" for metric in NULL_METRICS
}
NULL_METRIC_OBSERVED_COLUMN_MAP = {
    metric: f"{NULL_METRIC_SHORT_NAME_MAP[metric]}_observed" for metric in NULL_METRICS
}
NULL_METRIC_RELABELED_NULL_COLUMN_MAP = {
    metric: f"{NULL_METRIC_SHORT_NAME_MAP[metric]}_null" for metric in NULL_METRICS
}
NULL_METRIC_RELABELED_OBSERVED_COLUMN_MAP = {
    metric: f"{NULL_METRIC_SHORT_NAME_MAP[metric]}_observed_value" for metric in NULL_METRICS
}
NULL_METRIC_LABEL_MAP = {
    metric: f"{NULL_METRIC_SHORT_NAME_MAP[metric]}_vs_null" for metric in NULL_METRICS
}


def _print_dataframe(title, dataframe, columns=None):
    print(f"\n=== {title} ===")
    if dataframe.empty:
        print("<empty>")
        return

    display_df = dataframe if columns is None else dataframe.loc[:, columns]
    print(display_df.to_string(index=False))


def assert_expected_tools(dataframe, dataframe_name):
    if dataframe.empty:
        raise AssertionError(f"{dataframe_name} is empty.")

    for sample, sample_df in dataframe.groupby("sample", sort=True):
        observed_tools = set(sample_df["tool"])
        missing_tools = sorted(REQUIRED_TOOL_SET - observed_tools)
        extra_tools = sorted(observed_tools - REQUIRED_TOOL_SET)
        if missing_tools or extra_tools:
            raise AssertionError(
                f"Unexpected tool membership for {dataframe_name} in {sample}. "
                f"Missing={missing_tools}; Extra={extra_tools}"
            )


def build_tool_palette(tool_df):
    palette = {}
    for row in tool_df.itertuples(index=False):
        if row.tool == "methylseg":
            palette[row.tool_label] = "#0b5394"
        elif row.tool == "methylseg_hm450k":
            palette[row.tool_label] = "#3d85c6"
        else:
            palette[row.tool_label] = "#9aa0a6"
    return palette


def clear_stale_lad_summary_pngs(figures_dir):
    figures_dir = Path(figures_dir)
    if not figures_dir.exists():
        return 0

    stale_patterns = [
        "*.lad_summary.png",
        "*.lad_boundary_summary.png",
        "*.lad_end_window_summary.png",
        "*.lad_reverse_coverage_summary.png",
        "*.lad_same_lad_summary.png",
        "lad_combined_ranking.png",
        "lad_combined_*_ranking.png",
    ]
    stale_paths = []
    for pattern in stale_patterns:
        stale_paths.extend(figures_dir.glob(pattern))

    removed = 0
    for path in sorted(set(stale_paths)):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def prepare_regions_for_lad_overlap(region_df):
    regions = region_df.copy().reset_index(drop=True)
    regions["region_id"] = np.arange(len(regions))
    regions["region_length"] = regions["end"] - regions["start"]
    return regions


def prepare_lads_for_lad_overlap(lad_df):
    lads = lad_df.copy().reset_index(drop=True)
    lads["lad_id"] = np.arange(len(lads))
    if "lad_length" not in lads.columns:
        if "length" in lads.columns:
            lads["lad_length"] = lads["length"]
        else:
            lads["lad_length"] = lads["end"] - lads["start"]
    return lads


def load_sample_methylation(sample, base_dir, path_parts_template=SAMPLE_METH_FILE_PARTS):
    path_parts = [part.format(sample=sample) for part in path_parts_template]
    meth_file_path = Path(base_dir)
    for part in path_parts:
        meth_file_path = meth_file_path / part
    if not meth_file_path.exists():
        raise FileNotFoundError(
            f"Methylation file not found for sample {sample}: {meth_file_path}"
        )

    meth_df = pd.read_csv(meth_file_path, sep="\t", header=0)
    rename_map = {"chr": "chrom"}
    meth_df = meth_df.rename(columns=rename_map)
    if "end" not in meth_df.columns and "start" in meth_df.columns:
        meth_df["end"] = pd.to_numeric(meth_df["start"], errors="coerce") + 1
    required_cols = ["chrom", "start", "end", "beta"]
    missing_cols = [column for column in required_cols if column not in meth_df.columns]
    if missing_cols:
        raise ValueError(
            f"Methylation file for {sample} is missing required columns {missing_cols}: {meth_file_path}"
        )

    meth_df["chrom"] = meth_df["chrom"].astype(str)
    meth_df["start"] = pd.to_numeric(meth_df["start"], errors="coerce")
    meth_df["end"] = pd.to_numeric(meth_df["end"], errors="coerce")
    meth_df["beta"] = pd.to_numeric(meth_df["beta"], errors="coerce")
    meth_df = meth_df.dropna(subset=required_cols).copy()
    if meth_df.empty:
        return meth_df.loc[:, required_cols]

    meth_df["start"] = meth_df["start"].astype(int)
    meth_df["end"] = meth_df["end"].astype(int)
    meth_df = meth_df.loc[meth_df["end"] > meth_df["start"]].copy()
    return meth_df.loc[:, required_cols].reset_index(drop=True)


def build_filtered_lad_reference_lookup(
    samples,
    sample_genome_lookup,
    segmentation_results_path,
    reference_lookup,
    min_lad_beta=MIN_LAD_BETA,
):
    if float(min_lad_beta) <= 0.0:
        filtered_reference_lookup = {}
        reference_summary_rows = []
        for genome, reference in reference_lookup.items():
            raw_lad_df = prepare_lads_for_lad_overlap(reference["lad_df"])
            filtered_reference_lookup[genome] = raw_lad_df.reset_index(drop=True)
            reference_summary_rows.append(
                {
                    "genome": genome,
                    "n_lad_regions_raw": int(len(raw_lad_df)),
                    "n_lad_regions_filtered": int(len(raw_lad_df)),
                    "lad_total_bp_raw": int(raw_lad_df["lad_length"].sum()) if not raw_lad_df.empty else 0,
                    "lad_total_bp_filtered": int(raw_lad_df["lad_length"].sum()) if not raw_lad_df.empty else 0,
                    "min_lad_beta": float(min_lad_beta),
                }
            )
        return (
            filtered_reference_lookup,
            pd.DataFrame(columns=["lad_id", "beta", "sample", "genome"]),
            pd.DataFrame(columns=["genome", "lad_id", "beta"]),
            pd.DataFrame(reference_summary_rows).sort_values("genome").reset_index(drop=True),
        )

    sample_lad_means = []
    for sample in samples:
        genome = sample_genome_lookup[sample]
        lad_df_sample = prepare_lads_for_lad_overlap(reference_lookup[genome]["lad_df"])
        meth_df = load_sample_methylation(sample, segmentation_results_path)

        if meth_df.empty or lad_df_sample.empty:
            continue

        lad_meth_overlap = BedTool.from_dataframe(
            meth_df[["chrom", "start", "end", "beta"]]
        ).intersect(
            BedTool.from_dataframe(lad_df_sample[["chrom", "start", "end", "lad_id"]]),
            wa=True,
            wb=True,
        )
        try:
            lad_meth_overlap_df = lad_meth_overlap.to_dataframe(
                names=[
                    "chrom",
                    "start",
                    "end",
                    "beta",
                    "lad_chrom",
                    "lad_start",
                    "lad_end",
                    "lad_id",
                ]
            )
        except pd.errors.EmptyDataError:
            lad_meth_overlap_df = pd.DataFrame()

        if lad_meth_overlap_df.empty:
            continue

        lad_meth_overlap_df["beta"] = pd.to_numeric(
            lad_meth_overlap_df["beta"], errors="coerce"
        )
        lad_meth_overlap_df["lad_id"] = pd.to_numeric(
            lad_meth_overlap_df["lad_id"], errors="coerce"
        )
        lad_meth_overlap_df = lad_meth_overlap_df.dropna(subset=["beta", "lad_id"]).copy()
        if lad_meth_overlap_df.empty:
            continue
        lad_meth_overlap_df["lad_id"] = lad_meth_overlap_df["lad_id"].astype(int)

        sample_mean_df = (
            lad_meth_overlap_df.groupby("lad_id", as_index=False)["beta"]
            .mean()
            .assign(sample=sample, genome=genome)
        )
        sample_lad_means.append(sample_mean_df)

    if sample_lad_means:
        lad_sample_means_df = pd.concat(sample_lad_means, ignore_index=True)
        filtered_lad_ids_df = (
            lad_sample_means_df.groupby(["genome", "lad_id"], as_index=False)["beta"].mean()
        )
        filtered_lad_ids_df = filtered_lad_ids_df.loc[
            filtered_lad_ids_df["beta"] >= float(min_lad_beta)
        ].copy()
    else:
        lad_sample_means_df = pd.DataFrame(columns=["lad_id", "beta", "sample", "genome"])
        filtered_lad_ids_df = pd.DataFrame(columns=["genome", "lad_id", "beta"])

    filtered_reference_lookup = {}
    reference_summary_rows = []
    for genome, reference in reference_lookup.items():
        raw_lad_df = prepare_lads_for_lad_overlap(reference["lad_df"])
        kept_lad_ids = set(
            filtered_lad_ids_df.loc[filtered_lad_ids_df["genome"] == genome, "lad_id"].astype(int)
        )
        if kept_lad_ids:
            filtered_lad_df = raw_lad_df.loc[raw_lad_df["lad_id"].isin(kept_lad_ids)].copy()
        else:
            filtered_lad_df = raw_lad_df.copy()

        filtered_reference_lookup[genome] = filtered_lad_df.reset_index(drop=True)
        reference_summary_rows.append(
            {
                "genome": genome,
                "n_lad_regions_raw": int(len(raw_lad_df)),
                "n_lad_regions_filtered": int(len(filtered_lad_df)),
                "lad_total_bp_raw": int(raw_lad_df["lad_length"].sum()) if not raw_lad_df.empty else 0,
                "lad_total_bp_filtered": int(filtered_lad_df["lad_length"].sum()) if not filtered_lad_df.empty else 0,
                "min_lad_beta": float(min_lad_beta),
            }
        )

    reference_filter_summary_df = pd.DataFrame(reference_summary_rows).sort_values("genome").reset_index(drop=True)
    return filtered_reference_lookup, lad_sample_means_df, filtered_lad_ids_df, reference_filter_summary_df


def filter_regions_by_lad_overlap(
    regions_df,
    lad_df,
):
    regions_df["region_id"] = np.arange(len(regions_df))
    regions_df["region_length"] = regions_df["end"] - regions_df["start"]
    lad_df["lad_id"] = np.arange(len(lad_df))
    lad_df["lad_length"] = lad_df["end"] - lad_df["start"]
    regions = pd.DataFrame(
        {
            "chrom": regions_df["chrom"],
            "start": regions_df["start"],
            "end": regions_df["end"],
            "region_id": regions_df["region_id"],
        }
    )

    overlap_columns = [
        "chrom",
        "region_start",
        "region_end",
        "region_id",
        "lad_chrom",
        "lad_start",
        "lad_end",
        "lad_id",
        "overlap_bp",
    ]

    if regions.empty or lad_df.empty:
        overlaps = pd.DataFrame(columns=overlap_columns)
        return overlaps

    overlaps_bed = BedTool.from_dataframe(regions).intersect(
        BedTool.from_dataframe(lad_df[["chrom", "start", "end", "lad_id"]]),
        wo=True,
    )
    try:
        overlaps = overlaps_bed.to_dataframe(names=overlap_columns)
    except pd.errors.EmptyDataError:
        overlaps = pd.DataFrame(columns=overlap_columns)
    if overlaps.empty and "overlap_bp" not in overlaps.columns:
        overlaps = pd.DataFrame(columns=overlap_columns)

    return overlaps


def build_region_lad_overlap_detail_df(regions_df, overlaps_df):
    detail_df = regions_df[
        ["region_id", "chrom", "start", "end", "region_length"]
    ].copy()
    detail_df["total_lad_overlap_bp"] = 0
    detail_df["n_lad_overlaps"] = 0

    if not overlaps_df.empty:
        overlap_summary = (
            overlaps_df.groupby("region_id", as_index=False)
            .agg(
                total_lad_overlap_bp=("overlap_bp", "sum"),
                n_lad_overlaps=("lad_id", "nunique"),
            )
        )
        detail_df = detail_df.drop(
            columns=["total_lad_overlap_bp", "n_lad_overlaps"]
        ).merge(overlap_summary, on="region_id", how="left")
        detail_df[["total_lad_overlap_bp", "n_lad_overlaps"]] = detail_df[
            ["total_lad_overlap_bp", "n_lad_overlaps"]
        ].fillna(0)

    detail_df["total_lad_overlap_bp"] = detail_df["total_lad_overlap_bp"].astype(int)
    detail_df["n_lad_overlaps"] = detail_df["n_lad_overlaps"].astype(int)
    detail_df["lad_overlap_fraction"] = np.where(
        detail_df["region_length"] > 0,
        detail_df["total_lad_overlap_bp"] / detail_df["region_length"],
        0.0,
    )
    return detail_df


def compute_region_lad_distances(overlaps_df, regions_df, lad_df):
    overlapping_regions = set(overlaps_df["region_id"].dropna())
    best_lad_per_region = (
        overlaps_df.sort_values("overlap_bp", ascending=False)
        .drop_duplicates("region_id")
        .set_index("region_id")["lad_id"]
    )
    lad_df = lad_df.sort_values(["chrom", "start"]).reset_index(drop=True)

    results = []
    for chrom, regions_chr in regions_df.groupby("chrom", sort=False):
        lads_chr = lad_df[lad_df["chrom"] == chrom]
        if lads_chr.empty:
            tmp = regions_chr.copy()
            tmp["dist_to_lad"] = np.nan
            tmp["dist_to_lad_boundary"] = np.nan
            tmp["nearest_lad_id"] = pd.Series(pd.NA, index=tmp.index, dtype="Int64")
            results.append(tmp)
            continue

        r_start = regions_chr["start"].to_numpy(dtype=np.int64)
        r_end = regions_chr["end"].to_numpy(dtype=np.int64)
        region_ids = regions_chr["region_id"].to_numpy(dtype=np.int64)

        l_start = lads_chr["start"].to_numpy(dtype=np.int64)
        l_end = lads_chr["end"].to_numpy(dtype=np.int64)
        lad_ids = lads_chr["lad_id"].to_numpy(dtype=np.int64)

        idx_start = np.searchsorted(l_start, r_start)
        idx_end = np.searchsorted(l_start, r_end)
        candidates = np.stack(
            [
                np.clip(idx_start - 1, 0, len(l_start) - 1),
                np.clip(idx_start, 0, len(l_start) - 1),
                np.clip(idx_end - 1, 0, len(l_start) - 1),
                np.clip(idx_end, 0, len(l_start) - 1),
            ],
            axis=1,
        )

        l_start_c = l_start[candidates]
        l_end_c = l_end[candidates]
        r_start_exp = r_start[:, None]
        r_end_exp = r_end[:, None]
        dist_matrix = np.stack(
            [
                np.abs(r_start_exp - l_start_c),
                np.abs(r_start_exp - l_end_c),
                np.abs(r_end_exp - l_start_c),
                np.abs(r_end_exp - l_end_c),
            ],
            axis=2,
        ).min(axis=2)

        min_idx = np.argmin(dist_matrix, axis=1)
        min_dist = dist_matrix[np.arange(len(dist_matrix)), min_idx]
        nearest_idx = candidates[np.arange(len(candidates)), min_idx]
        nearest_lad_id = lad_ids[nearest_idx]
        is_overlap = np.isin(region_ids, list(overlapping_regions))
        dist_to_lad = np.where(is_overlap, 0, min_dist)
        nearest_lad_id = np.where(
            is_overlap,
            pd.Series(region_ids).map(best_lad_per_region).to_numpy(),
            nearest_lad_id,
        )

        tmp = regions_chr.copy()
        tmp["dist_to_lad_boundary"] = min_dist
        tmp["dist_to_lad"] = dist_to_lad
        tmp["nearest_lad_id"] = pd.Series(nearest_lad_id, dtype="Int64")
        results.append(tmp)

    if not results:
        return empty_region_lad_distance_df()

    dist_df = pd.concat(results, ignore_index=True)
    dist_df["nearest_lad_id"] = dist_df["nearest_lad_id"].astype("Int64")
    return dist_df[["region_id", "dist_to_lad", "dist_to_lad_boundary", "nearest_lad_id"]]


def empty_region_lad_distance_df():
    distance_df = pd.DataFrame(
        columns=["region_id", "dist_to_lad", "dist_to_lad_boundary", "nearest_lad_id"]
    )
    distance_df["nearest_lad_id"] = distance_df["nearest_lad_id"].astype("Int64")
    return distance_df


def compute_region_lad_distances_or_empty(overlaps_df, regions_df, lad_df):
    if regions_df.empty:
        return empty_region_lad_distance_df()

    distance_parts = []
    for chrom, chrom_region_df in regions_df.groupby("chrom", sort=False):
        chrom_lad_df = lad_df.loc[lad_df["chrom"] == chrom].copy()
        if overlaps_df.empty:
            chrom_overlaps_df = overlaps_df.copy()
        else:
            chrom_overlaps_df = overlaps_df.loc[overlaps_df["chrom"] == chrom].copy()
        distance_parts.append(
            compute_region_lad_distances(
                chrom_overlaps_df,
                chrom_region_df.copy(),
                chrom_lad_df,
            )
        )

    if not distance_parts:
        return empty_region_lad_distance_df()
    return pd.concat(distance_parts, ignore_index=True)


def calculate_avg_distance_to_nearest_lad(dist_df):
    return dist_df["dist_to_lad"].mean(skipna=True)


def calculate_avg_distance_to_nearest_lad_boundary(dist_df):
    return dist_df["dist_to_lad_boundary"].mean(skipna=True)


def calculate_avg_distance_to_nearest_lad_boundary_non_overlapping(dist_df):
    non_overlapping = dist_df["dist_to_lad"] > 0
    return dist_df.loc[non_overlapping, "dist_to_lad_boundary"].mean(skipna=True)


def calculate_pct_of_regions_with_boundary_within_150kb_of_lad_boundary(dist_df):
    valid = dist_df["dist_to_lad_boundary"].notna()
    pct = (
        (dist_df.loc[valid, "dist_to_lad_boundary"] <= 150_000).mean() if valid.any() else 0
    )
    return pct


def calculate_pct_of_nonoverlapping_regions_with_boundary_within_150kb_of_lad_boundary(
    dist_df,
):
    non_overlapping = dist_df["dist_to_lad"] > 0
    valid = non_overlapping & dist_df["dist_to_lad_boundary"].notna()
    pct = (
        (dist_df.loc[valid, "dist_to_lad_boundary"] <= 150_000).mean()
        if valid.any()
        else 0
    )
    return pct


def calculate_avg_lads_per_overlapping_pmd(lad_overlap_counts):
    overlapping_counts = lad_overlap_counts.loc[lad_overlap_counts > 0]
    if overlapping_counts.empty:
        return 0.0
    return float(overlapping_counts.mean())


def calculate_avg_lad_per_pmd(lad_overlap_counts):
    if lad_overlap_counts.empty:
        return 0.0
    return float(lad_overlap_counts.mean())


def summarize_null_distribution(observed_value, perm_values):
    perm_values = pd.to_numeric(perm_values, errors="coerce").dropna()
    if perm_values.empty:
        return np.nan, np.nan, np.nan

    null_mean = float(perm_values.mean())
    if len(perm_values) == 1:
        null_sd = 0.0
        z_score = 0.0 if observed_value == null_mean else np.nan
        return null_mean, null_sd, z_score

    null_sd = float(perm_values.std(ddof=1))
    if null_sd == 0:
        z_score = 0.0 if observed_value == null_mean else np.nan
        return null_mean, null_sd, z_score

    z_score = float(
        stats.zmap([observed_value], perm_values.to_numpy(dtype=float), ddof=1, nan_policy="omit")[0]
    )
    return null_mean, null_sd, z_score


def get_overlap_scores(
    overlaps_df,
    regions_df,
    lad_df,
    dist_df=None,
):
    n_regions = len(regions_df)
    if n_regions == 0:
        return pd.DataFrame(
            {
                metric: 0.0
                for metric in LAD_OVERLAP_METRICS
            },
            index=[0],
        )

    overlapping_regions = overlaps_df["region_id"].dropna().unique()
    pct_regions_overlapping_lads = (
        len(overlapping_regions) / n_regions if n_regions > 0 else 0
    )

    overlap_per_region = overlaps_df.groupby("region_id")["overlap_bp"].sum()
    lad_overlap_counts = (
        overlaps_df.groupby("region_id")["lad_id"]
        .nunique()
        .reindex(regions_df["region_id"], fill_value=0)
        .astype(float)
    )

    pct_regions_overlapping_lads_gte_150kb = (
        (overlap_per_region >= 150_000).sum() / n_regions if n_regions > 0 else 0
    )

    if dist_df is None:
        dist_df = compute_region_lad_distances(overlaps_df, regions_df, lad_df)

    avg_distance_to_nearest_lad = calculate_avg_distance_to_nearest_lad(dist_df)

    avg_distance_to_nearest_lad_boundary = calculate_avg_distance_to_nearest_lad_boundary(dist_df)

    avg_distance_to_nearest_lad_boundary_non_overlapping = (
        calculate_avg_distance_to_nearest_lad_boundary_non_overlapping(dist_df)
    )
    avg_lads_per_overlapping_pmd = calculate_avg_lads_per_overlapping_pmd(
        lad_overlap_counts
    )
    avg_lad_per_pmd = calculate_avg_lad_per_pmd(lad_overlap_counts)

    pct_of_regions_with_boundary_within_150kb_of_lad_boundary = (
        calculate_pct_of_regions_with_boundary_within_150kb_of_lad_boundary(dist_df)
    )

    pct_non_overlapping_regions_within_150kb_of_lad_boundary = (
        calculate_pct_of_nonoverlapping_regions_with_boundary_within_150kb_of_lad_boundary(
            dist_df
        )
    )

    lad_counts = overlaps_df.groupby("lad_id")["region_id"].nunique()

    shared_lads = lad_counts[lad_counts > 1].index

    regions_in_shared_lads = overlaps_df.loc[
        overlaps_df["lad_id"].isin(shared_lads), "region_id"
    ].dropna().unique()

    overlapping_regions = overlaps_df["region_id"].dropna().unique()

    pct_regions_sharing_lad = (
        len(regions_in_shared_lads) / len(overlapping_regions)
        if len(overlapping_regions) > 0 else 0
    )

    total_pmd_bp = regions_df["region_length"].sum()
    pmd_coverage_by_lads = (
        overlaps_df["overlap_bp"].sum() / total_pmd_bp if total_pmd_bp > 0 else 0
    )

    n_lads = len(lad_df)
    pct_lads_overlapping = (
        overlaps_df["lad_id"].nunique() / n_lads if n_lads > 0 else 0
    )
    total_lad_bp = lad_df["lad_length"].sum()
    lad_coverage_by_pmds = (
        overlaps_df["overlap_bp"].sum() / total_lad_bp if total_lad_bp > 0 else 0
    )

    metrics_df = pd.DataFrame(
        {
            "pct_regions_overlapping_lads": pct_regions_overlapping_lads,
            "pct_regions_overlapping_lads_gte_150kb": pct_regions_overlapping_lads_gte_150kb,
            "avg_distance_to_nearest_lad": avg_distance_to_nearest_lad,
            "avg_distance_to_nearest_lad_boundary": avg_distance_to_nearest_lad_boundary,
            "avg_distance_to_nearest_lad_boundary_non_overlapping": avg_distance_to_nearest_lad_boundary_non_overlapping,
            "avg_lads_per_overlapping_pmd": avg_lads_per_overlapping_pmd,
            "avg_lad_per_pmd": avg_lad_per_pmd,
            "pct_regions_with_boundary_within_150kb_of_lad_boundary": pct_of_regions_with_boundary_within_150kb_of_lad_boundary,
            "pct_non_overlapping_regions_within_150kb_of_lad_boundary": pct_non_overlapping_regions_within_150kb_of_lad_boundary,
            "pct_regions_sharing_lad": pct_regions_sharing_lad,
            "pmd_coverage_by_lads": pmd_coverage_by_lads,
            "pct_lads_overlapping": pct_lads_overlapping,
            "lad_coverage_by_pmds": lad_coverage_by_pmds,
        },
        index=[0],
    )
    return metrics_df


def build_region_lad_distance_detail_df(regions_df, distance_df):
    detail_df = regions_df[
        ["region_id", "chrom", "start", "end", "region_length"]
    ].copy()
    if distance_df.empty:
        detail_df["dist_to_lad"] = np.nan
        detail_df["dist_to_lad_boundary"] = np.nan
        detail_df["nearest_lad_id"] = pd.Series(pd.NA, index=detail_df.index, dtype="Int64")
        return detail_df

    return detail_df.merge(distance_df, on="region_id", how="left")


def get_lad_metric_count_lookup(overlaps_df, regions_df, dist_df):
    n_regions = int(len(regions_df))
    if n_regions == 0:
        return {metric: 0 for metric in LAD_OVERLAP_METRICS}

    overlap_per_region = overlaps_df.groupby("region_id")["overlap_bp"].sum()
    overlapping_regions = overlaps_df["region_id"].dropna().unique()
    lad_counts = overlaps_df.groupby("lad_id")["region_id"].nunique()
    shared_lads = lad_counts[lad_counts > 1].index
    regions_in_shared_lads = overlaps_df.loc[
        overlaps_df["lad_id"].isin(shared_lads), "region_id"
    ].dropna().unique()
    boundary_valid = dist_df["dist_to_lad_boundary"].notna()
    boundary_within_150kb = (
        dist_df.loc[boundary_valid, "dist_to_lad_boundary"] <= 150_000
        if boundary_valid.any()
        else pd.Series(dtype=bool)
    )
    non_overlapping_boundary_valid = (dist_df["dist_to_lad"] > 0) & boundary_valid
    non_overlapping_boundary_within_150kb = (
        dist_df.loc[non_overlapping_boundary_valid, "dist_to_lad_boundary"] <= 150_000
        if non_overlapping_boundary_valid.any()
        else pd.Series(dtype=bool)
    )

    return {
        "pct_regions_overlapping_lads": int(len(overlapping_regions)),
        "pct_regions_overlapping_lads_gte_150kb": int(
            (overlap_per_region >= 150_000).sum()
        ),
        "avg_distance_to_nearest_lad": np.nan,
        "avg_distance_to_nearest_lad_boundary": np.nan,
        "avg_distance_to_nearest_lad_boundary_non_overlapping": np.nan,
        "avg_lads_per_overlapping_pmd": np.nan,
        "avg_lad_per_pmd": np.nan,
        "pct_regions_with_boundary_within_150kb_of_lad_boundary": int(
            boundary_within_150kb.sum()
        ),
        "pct_non_overlapping_regions_within_150kb_of_lad_boundary": int(
            non_overlapping_boundary_within_150kb.sum()
        ),
        "pct_regions_sharing_lad": int(len(regions_in_shared_lads)),
        "pmd_coverage_by_lads": np.nan,
        "pct_lads_overlapping": int(overlaps_df["lad_id"].nunique()),
        "lad_coverage_by_pmds": np.nan,
    }


def build_lad_association_metric_rows(
    metric_base,
    overlaps_df,
    regions_df,
    lad_df,
    distance_df=None,
):
    if distance_df is None:
        distance_df = compute_region_lad_distances_or_empty(overlaps_df, regions_df, lad_df)
    scores_df = get_overlap_scores(overlaps_df, regions_df, lad_df, dist_df=distance_df)
    count_lookup = get_lad_metric_count_lookup(overlaps_df, regions_df, distance_df)

    rows = []
    for metric in LAD_ASSOCIATION_METRICS:
        score = scores_df.loc[0, metric]
        rows.append(
            {
                **metric_base,
                "analysis_mode": LAD_METRIC_MODES[metric],
                "threshold_bp": int(LAD_METRIC_THRESHOLDS_BP[metric]),
                "metric": metric,
                "score": float(score) if pd.notna(score) else np.nan,
                "n_regions_passing_threshold": count_lookup[metric],
            }
        )

    return rows


def compute_region_mean_beta(regions_df, meth_df):
    if regions_df.empty or meth_df.empty:
        return pd.DataFrame(columns=["region_id", "mean_beta"])

    overlap_columns = [
        "meth_chrom",
        "meth_start",
        "meth_end",
        "beta",
        "region_chrom",
        "region_start",
        "region_end",
        "region_id",
    ]
    meth_overlap = BedTool.from_dataframe(
        meth_df[["chrom", "start", "end", "beta"]]
    ).intersect(
        BedTool.from_dataframe(regions_df[["chrom", "start", "end", "region_id"]]),
        wa=True,
        wb=True,
    )
    try:
        meth_overlap_df = meth_overlap.to_dataframe(names=overlap_columns)
    except pd.errors.EmptyDataError:
        meth_overlap_df = pd.DataFrame(columns=overlap_columns)

    if meth_overlap_df.empty:
        return pd.DataFrame(columns=["region_id", "mean_beta"])

    meth_overlap_df["beta"] = pd.to_numeric(meth_overlap_df["beta"], errors="coerce")
    meth_overlap_df["region_id"] = pd.to_numeric(
        meth_overlap_df["region_id"], errors="coerce"
    )
    meth_overlap_df = meth_overlap_df.dropna(subset=["beta", "region_id"]).copy()
    if meth_overlap_df.empty:
        return pd.DataFrame(columns=["region_id", "mean_beta"])
    meth_overlap_df["region_id"] = meth_overlap_df["region_id"].astype(int)
    return (
        meth_overlap_df.groupby("region_id", as_index=False)["beta"]
        .mean()
        .rename(columns={"beta": "mean_beta"})
    )


def classify_regions_vs_methylseg(candidate_regions_df, methylseg_regions_df):
    if candidate_regions_df.empty:
        return pd.DataFrame(columns=["region_id", "overlaps_methylseg"])

    candidate_flags = candidate_regions_df[["region_id"]].copy()
    candidate_flags["overlaps_methylseg"] = False
    if methylseg_regions_df.empty:
        return candidate_flags

    overlapping_regions = BedTool.from_dataframe(
        candidate_regions_df[["chrom", "start", "end", "region_id"]]
    ).intersect(
        BedTool.from_dataframe(methylseg_regions_df[["chrom", "start", "end"]]),
        u=True,
    )
    try:
        overlapping_regions_df = overlapping_regions.to_dataframe(
            names=["chrom", "start", "end", "region_id"]
        )
    except pd.errors.EmptyDataError:
        overlapping_regions_df = pd.DataFrame(
            columns=["chrom", "start", "end", "region_id"]
        )

    if overlapping_regions_df.empty:
        return candidate_flags

    overlapping_region_ids = set(
        pd.to_numeric(overlapping_regions_df["region_id"], errors="coerce")
        .dropna()
        .astype(int)
    )
    candidate_flags["overlaps_methylseg"] = candidate_flags["region_id"].isin(
        overlapping_region_ids
    )
    return candidate_flags


def build_unique_lad_exports(
    *,
    samples,
    sample_genome_lookup,
    segmentation_results_path,
    filtered_reference_lookup,
    output_dir,
):
    summary_rows = []
    export_rows = []
    export_root = Path(output_dir) / "unique_lads"
    export_root.mkdir(parents=True, exist_ok=True)

    for comparison in DNMTOOLS_METHYLSEG_COMPARISONS:
        comparison_tool = comparison["comparison_tool"]
        reference_methylseg_tool = comparison["reference_methylseg_tool"]

        for sample in samples:
            genome = sample_genome_lookup[sample]
            lad_df_sample = prepare_lads_for_lad_overlap(
                filtered_reference_lookup[genome]
            )
            comparison_regions_df = prepare_regions_for_lad_overlap(
                load_tool_regions(
                    TOOL_CONFIG_BY_NAME[comparison_tool],
                    sample,
                    segmentation_results_path=segmentation_results_path,
                )
            )
            methylseg_regions_df = prepare_regions_for_lad_overlap(
                load_tool_regions(
                    TOOL_CONFIG_BY_NAME[reference_methylseg_tool],
                    sample,
                    segmentation_results_path=segmentation_results_path,
                )
            )

            comparison_lad_overlaps = filter_regions_by_lad_overlap(
                comparison_regions_df.copy(), lad_df_sample.copy()
            )
            methylseg_lad_overlaps = filter_regions_by_lad_overlap(
                methylseg_regions_df.copy(), lad_df_sample.copy()
            )
            comparison_lad_ids = set(
                pd.to_numeric(
                    comparison_lad_overlaps.get("lad_id", pd.Series(dtype=float)),
                    errors="coerce",
                )
                .dropna()
                .astype(int)
            )
            methylseg_lad_ids = set(
                pd.to_numeric(
                    methylseg_lad_overlaps.get("lad_id", pd.Series(dtype=float)),
                    errors="coerce",
                )
                .dropna()
                .astype(int)
            )
            comparison_only_lad_ids = sorted(comparison_lad_ids - methylseg_lad_ids)

            comparison_only_overlap_rows = comparison_lad_overlaps.loc[
                pd.to_numeric(
                    comparison_lad_overlaps.get("lad_id", pd.Series(dtype=float)),
                    errors="coerce",
                ).isin(comparison_only_lad_ids)
            ].copy()
            comparison_only_region_ids = set(
                pd.to_numeric(
                    comparison_only_overlap_rows.get("region_id", pd.Series(dtype=float)),
                    errors="coerce",
                )
                .dropna()
                .astype(int)
            )
            summary_rows.append(
                {
                    "comparison_tool": comparison_tool,
                    "reference_methylseg_tool": reference_methylseg_tool,
                    "sample": sample,
                    "genome": genome,
                    "n_unique_lads": int(len(comparison_only_lad_ids)),
                    "n_unique_regions": int(len(comparison_only_region_ids)),
                }
            )

            export_lad_df = lad_df_sample.loc[
                lad_df_sample["lad_id"].isin(comparison_only_lad_ids),
                ["chrom", "start", "end", "lad_id"],
            ].copy()
            export_lad_df = export_lad_df.sort_values(["chrom", "start", "end"]).reset_index(
                drop=True
            )
            export_lad_df["name"] = [
                f"{comparison_tool}_only_vs_{reference_methylseg_tool}_lad_{lad_id}"
                for lad_id in export_lad_df["lad_id"].astype(int)
            ]
            sample_output_dir = export_root / sample
            sample_output_dir.mkdir(parents=True, exist_ok=True)
            export_path = (
                sample_output_dir
                / f"{sample}.{comparison_tool}_{reference_methylseg_tool}.unique_lads.bed"
            )
            export_lad_df[["chrom", "start", "end", "name"]].to_csv(
                export_path, sep="\t", header=False, index=False
            )
            export_rows.append(
                {
                    "comparison_tool": comparison_tool,
                    "reference_methylseg_tool": reference_methylseg_tool,
                    "sample": sample,
                    "genome": genome,
                    "n_unique_lads": int(len(export_lad_df)),
                    "export_path": str(export_path),
                }
            )

    unique_lad_details_df = pd.DataFrame(
        columns=[
            "comparison_tool",
            "reference_methylseg_tool",
            "sample",
            "genome",
            "region_id",
            "chrom",
            "start",
            "end",
            "region_length_bp",
            "mean_beta",
            "overlaps_methylseg",
        ]
    )
    unique_lad_summary_df = pd.DataFrame(summary_rows)
    unique_lad_exports_df = pd.DataFrame(export_rows)
    return unique_lad_details_df, unique_lad_summary_df, unique_lad_exports_df


def load_chrom_sizes_for_genome(genome):
    chrom_sizes_path = CHROM_SIZES_PATHS[str(genome)]
    chrom_sizes = {}
    with chrom_sizes_path.open() as handle:
        for line in handle:
            chrom, size = line.strip().split("\t")[:2]
            if chrom in NULL_MODEL_CANONICAL_CHROMS:
                chrom_sizes[chrom] = int(size)
    return chrom_sizes


def sample_nonoverlapping_intervals_for_chrom(lengths, chrom_size, rng):
    lengths = [int(length) for length in lengths]
    if not lengths:
        return []

    total_length = int(sum(lengths))
    if total_length > int(chrom_size):
        raise ValueError(
            f"Requested {total_length} bp but chromosome only has {chrom_size} bp of valid space"
        )

    shuffled_lengths = rng.permutation(lengths)
    remaining_space = int(chrom_size) - total_length
    gaps = rng.multinomial(
        remaining_space,
        np.full(len(shuffled_lengths) + 1, 1 / (len(shuffled_lengths) + 1)),
    )

    intervals = []
    cursor = int(gaps[0])
    for idx, length in enumerate(shuffled_lengths):
        start = cursor
        end = start + int(length)
        intervals.append((start, end, int(length)))
        cursor = end + int(gaps[idx + 1])
    return intervals


def randomize_regions_matched_null(regions_df, chrom_sizes, rng):
    if regions_df.empty:
        return regions_df.copy()

    random_rows = []
    for chrom, chrom_df in regions_df.groupby("chrom", sort=False):
        if chrom not in chrom_sizes:
            raise ValueError(f"Chromosome {chrom} is missing from chromosome sizes")
        intervals = sample_nonoverlapping_intervals_for_chrom(
            chrom_df["region_length"].astype(int).tolist(),
            chrom_sizes[chrom],
            rng,
        )
        for start, end, length in intervals:
            random_rows.append(
                {
                    "chrom": chrom,
                    "start": int(start),
                    "end": int(end),
                    "region_length": int(length),
                }
            )

    randomized_df = pd.DataFrame(random_rows)
    randomized_df = randomized_df.sort_values(["chrom", "start", "end"]).reset_index(
        drop=True
    )
    randomized_df["region_id"] = np.arange(len(randomized_df))
    return randomized_df


def compute_lad_null_metrics(overlaps_df, regions_df, lad_df):
    metrics_df = get_overlap_scores(overlaps_df, regions_df, lad_df)
    return {metric: metrics_df.iloc[0][metric] for metric in NULL_METRICS}


def regions_have_no_overlap(interval_df):
    if interval_df.empty:
        return True
    for _, chrom_df in interval_df.sort_values(["chrom", "start", "end"]).groupby(
        "chrom"
    ):
        starts = chrom_df["start"].to_numpy(dtype=np.int64)
        ends = chrom_df["end"].to_numpy(dtype=np.int64)
        if len(starts) > 1 and np.any(starts[1:] < ends[:-1]):
            return False
    return True


def _run_single_lad_null_permutation(args):
    perm_idx = args["perm_idx"]
    seed = args["seed"]
    observed_regions_df = args["observed_regions_df"]
    lad_df_sample = args["lad_df_sample"]
    chrom_sizes = args["chrom_sizes"]

    rng = np.random.default_rng(seed)
    permuted_regions_df = randomize_regions_matched_null(
        observed_regions_df.copy(), chrom_sizes, rng
    )
    permuted_overlaps_df = filter_regions_by_lad_overlap(
        permuted_regions_df.copy(), lad_df_sample.copy()
    )
    permuted_metrics = compute_lad_null_metrics(
        permuted_overlaps_df, permuted_regions_df, lad_df_sample.copy()
    )

    observed_chrom_counts = observed_regions_df["chrom"].value_counts().sort_index()
    permuted_chrom_counts = permuted_regions_df["chrom"].value_counts().sort_index()
    check_flags = {
        "same_region_count": len(permuted_regions_df) == len(observed_regions_df),
        "same_chrom_counts": observed_chrom_counts.equals(permuted_chrom_counts),
        "same_length_multiset_by_chrom": True,
        "within_chrom_bounds": (
            (permuted_regions_df["start"] >= 0).all()
            and (permuted_regions_df["end"] > permuted_regions_df["start"]).all()
            and (
                permuted_regions_df.apply(
                    lambda row: row["end"] <= chrom_sizes[row["chrom"]], axis=1
                )
            ).all()
        ),
        "strictly_positive_lengths": (permuted_regions_df["region_length"] > 0).all(),
        "non_overlapping_intervals": regions_have_no_overlap(permuted_regions_df),
    }

    for chrom, observed_chrom_df in observed_regions_df.groupby("chrom"):
        observed_lengths = sorted(observed_chrom_df["region_length"].astype(int).tolist())
        permuted_lengths = sorted(
            permuted_regions_df.loc[
                permuted_regions_df["chrom"] == chrom, "region_length"
            ]
            .astype(int)
            .tolist()
        )
        check_flags["same_length_multiset_by_chrom"] &= observed_lengths == permuted_lengths

    return {
        "perm_idx": perm_idx,
        **{
            NULL_METRIC_NULL_COLUMN_MAP[metric]: permuted_metrics[metric]
            for metric in NULL_METRICS
        },
        **check_flags,
    }


def _run_lad_null_job(args):
    tool = args["tool"]
    sample = args["sample"]
    genome = args["genome"]
    observed_regions_df = args["observed_regions_df"].copy()
    lad_df_sample = args["lad_df_sample"].copy()
    chrom_sizes = args["chrom_sizes"]
    permutation_seeds = args["permutation_seeds"]

    status = "ok"
    error_message = ""
    completed_permutations = 0
    check_flags = {
        "same_region_count": True,
        "same_chrom_counts": True,
        "same_length_multiset_by_chrom": True,
        "within_chrom_bounds": True,
        "strictly_positive_lengths": True,
        "non_overlapping_intervals": True,
    }

    if observed_regions_df.empty:
        status = "empty"
        observed_metrics = {metric: np.nan for metric in NULL_METRICS}
        permutation_rows = []
    else:
        observed_overlaps_df = filter_regions_by_lad_overlap(
            observed_regions_df.copy(), lad_df_sample.copy()
        )
        observed_metrics = compute_lad_null_metrics(
            observed_overlaps_df, observed_regions_df.copy(), lad_df_sample.copy()
        )
        permutation_rows = []
        try:
            for perm_idx, seed in enumerate(permutation_seeds):
                perm_result = _run_single_lad_null_permutation(
                    {
                        "perm_idx": perm_idx,
                        "seed": int(seed),
                        "observed_regions_df": observed_regions_df.copy(),
                        "lad_df_sample": lad_df_sample.copy(),
                        "chrom_sizes": chrom_sizes,
                    }
                )
                for flag_name in check_flags:
                    check_flags[flag_name] &= perm_result[flag_name]
                permutation_rows.append(
                    {
                        "tool": tool,
                        "sample": sample,
                        "genome": genome,
                        "perm_idx": perm_result["perm_idx"],
                        **{
                            NULL_METRIC_NULL_COLUMN_MAP[metric]: perm_result[
                                NULL_METRIC_NULL_COLUMN_MAP[metric]
                            ]
                            for metric in NULL_METRICS
                        },
                    }
                )
                completed_permutations += 1
        except Exception as exc:
            status = "failed"
            error_message = str(exc)

    observed_row = {
        "tool": tool,
        "sample": sample,
        "genome": genome,
        **{
            NULL_METRIC_OBSERVED_COLUMN_MAP[metric]: observed_metrics[metric]
            for metric in NULL_METRICS
        },
        "n_permutations_completed": completed_permutations,
        "status": status,
        "error_message": error_message,
    }
    check_row = {
        "tool": tool,
        "sample": sample,
        "genome": genome,
        "n_permutations_completed": completed_permutations,
        **check_flags,
    }
    return {
        "observed_row": observed_row,
        "permutation_rows": permutation_rows,
        "check_row": check_row,
    }


def run_lad_null_model(
    *,
    samples,
    sample_genome_lookup,
    segmentation_results_path,
    filtered_reference_lookup,
    n_permutations=DEFAULT_LAD_NULL_PERMUTATIONS,
    null_seed=DEFAULT_LAD_NULL_SEED,
):
    lad_null_observed_rows = []
    lad_null_permutation_rows = []
    lad_null_check_rows = []
    master_rng = np.random.default_rng(null_seed)
    process_pool_context = multiprocessing.get_context("fork")
    n_workers = max(1, (os.cpu_count() or 1) - 1)

    job_args = []
    for tool in REQUIRED_TOOLS:
        print(f"Queueing LAD null model for tool: {tool}")
        for sample in samples:
            genome = sample_genome_lookup[sample]
            lad_df_sample = prepare_lads_for_lad_overlap(filtered_reference_lookup[genome])
            chrom_sizes = load_chrom_sizes_for_genome(genome)
            observed_regions_df = prepare_regions_for_lad_overlap(
                load_tool_regions(
                    TOOL_CONFIG_BY_NAME[tool],
                    sample,
                    segmentation_results_path=segmentation_results_path,
                ).copy()
            )
            sample_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
            permutation_seeds = [
                int(sample_rng.integers(0, 2**32 - 1)) for _ in range(int(n_permutations))
            ]
            job_args.append(
                {
                    "tool": tool,
                    "sample": sample,
                    "genome": genome,
                    "lad_df_sample": lad_df_sample,
                    "chrom_sizes": chrom_sizes,
                    "observed_regions_df": observed_regions_df,
                    "permutation_seeds": permutation_seeds,
                }
            )

    if n_workers <= 1:
        job_results = [_run_lad_null_job(arg) for arg in job_args]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=process_pool_context,
        ) as executor:
            job_results = list(executor.map(_run_lad_null_job, job_args))

    for job_result in job_results:
        lad_null_observed_rows.append(job_result["observed_row"])
        lad_null_permutation_rows.extend(job_result["permutation_rows"])
        lad_null_check_rows.append(job_result["check_row"])

    lad_null_permutation_df = pd.DataFrame(
        lad_null_permutation_rows,
        columns=[
            "tool",
            "sample",
            "genome",
            "perm_idx",
            *[NULL_METRIC_NULL_COLUMN_MAP[metric] for metric in NULL_METRICS],
        ],
    ).rename(
        columns={
            "perm_idx": "null_perm_idx",
            **{
                NULL_METRIC_NULL_COLUMN_MAP[metric]: NULL_METRIC_RELABELED_NULL_COLUMN_MAP[
                    metric
                ]
                for metric in NULL_METRICS
            },
        }
    )
    lad_null_observed_df = pd.DataFrame(lad_null_observed_rows).rename(
        columns={
            **{
                NULL_METRIC_OBSERVED_COLUMN_MAP[metric]: NULL_METRIC_RELABELED_OBSERVED_COLUMN_MAP[
                    metric
                ]
                for metric in NULL_METRICS
            },
            "n_permutations_completed": "n_null_permutations_completed",
            "status": "null_status",
            "error_message": "null_error_message",
        }
    )
    lad_null_checks_df = pd.DataFrame(lad_null_check_rows).rename(
        columns={"n_permutations_completed": "n_null_permutations_completed"}
    )

    summary_rows = []
    for observed_row in lad_null_observed_df.itertuples(index=False):
        perm_subset = lad_null_permutation_df.loc[
            (lad_null_permutation_df["tool"] == observed_row.tool)
            & (lad_null_permutation_df["sample"] == observed_row.sample)
        ].copy()
        for metric in NULL_METRICS:
            observed_value = getattr(
                observed_row, NULL_METRIC_RELABELED_OBSERVED_COLUMN_MAP[metric]
            )
            perm_col = NULL_METRIC_RELABELED_NULL_COLUMN_MAP[metric]
            perm_values = (
                perm_subset[perm_col]
                if (not perm_subset.empty and perm_col in perm_subset.columns)
                else pd.Series(dtype=float)
            )
            null_mean, null_sd, z_score = summarize_null_distribution(
                observed_value,
                perm_values,
            )

            if pd.isna(null_mean):
                enrichment = np.nan
            elif null_mean == 0:
                enrichment = np.nan if pd.isna(observed_value) or observed_value == 0 else np.inf
            else:
                enrichment = observed_value / null_mean

            summary_rows.append(
                {
                    "tool": observed_row.tool,
                    "sample": observed_row.sample,
                    "genome": observed_row.genome,
                    "metric_vs_null": NULL_METRIC_LABEL_MAP[metric],
                    "metric": metric,
                    "observed_value": observed_value,
                    "null_mean": null_mean,
                    "null_sd": null_sd,
                    "enrichment_vs_null": enrichment,
                    "z_score_vs_null": z_score,
                    "n_null_permutations_completed": observed_row.n_null_permutations_completed,
                    "null_status": observed_row.null_status,
                    "null_error_message": observed_row.null_error_message,
                }
            )

    lad_null_summary_df = pd.DataFrame(summary_rows)
    return lad_null_observed_df, lad_null_permutation_df, lad_null_summary_df, lad_null_checks_df


def prepare_lad_reference(
    *,
    genome,
    reference_dir,
    lad_interval_track_path,
    laminb1_signal_track_path,
    liftover_script_path,
    hg19_to_hg38_chain,
    force=False,
):
    genome = str(genome)
    reference_dir = Path(reference_dir)
    chrom_sizes_path = reference_dir / f"{genome}.chrom.sizes"
    lad_clean_path = reference_dir / f"laminB1Lads.{genome}.cleaned.bed"
    occupancy_bedgraph_path = reference_dir / f"laminB1Lads.{genome}.occupancy.bedGraph"
    occupancy_bw_path = reference_dir / f"laminB1Lads.{genome}.occupancy.bw"
    signal_clean_path = reference_dir / f"laminB1.{genome}.cleaned.bedGraph"
    signal_bw_path = reference_dir / f"laminB1.{genome}.signal.bw"

    cached_paths = [
        chrom_sizes_path,
        lad_clean_path,
        occupancy_bedgraph_path,
        occupancy_bw_path,
        signal_clean_path,
        signal_bw_path,
    ]
    if not force and all(path.exists() for path in cached_paths):
        validate_bigwig(occupancy_bw_path)
        validate_bigwig(signal_bw_path)
        chrom_sizes = read_chrom_sizes(chrom_sizes_path, canonical_only=True)
        lad_df = read_bed_intervals(lad_clean_path)
        return {
            "genome": genome,
            "chrom_sizes": chrom_sizes,
            "chrom_sizes_path": chrom_sizes_path,
            "lad_df": lad_df,
            "lad_clean_path": lad_clean_path,
            "occupancy_bedgraph_path": occupancy_bedgraph_path,
            "occupancy_bw_path": occupancy_bw_path,
            "signal_clean_path": signal_clean_path,
            "signal_bw_path": signal_bw_path,
        }

    chrom_sizes, chrom_sizes_path = ensure_chrom_sizes(genome, chrom_sizes_path, force=force)

    if genome == "hg19":
        lad_source_path = Path(lad_interval_track_path)
        signal_source_path = Path(laminb1_signal_track_path)
    elif genome == "hg38":
        lad_source_path = reference_dir / "laminB1Lads.hg38.lifted_raw.bed"
        signal_source_path = reference_dir / "laminB1.hg38.lifted_raw.bedGraph"
        run_liftover_bed(
            lad_interval_track_path,
            lad_source_path,
            hg19_to_hg38_chain,
            liftover_script_path,
            force=force,
        )
        run_liftover_bed(
            laminb1_signal_track_path,
            signal_source_path,
            hg19_to_hg38_chain,
            liftover_script_path,
            force=force,
        )
    else:
        raise ValueError(f"Unsupported genome: {genome}")

    lad_df = clean_and_merge_intervals(read_bed_intervals(lad_source_path))
    if lad_df.empty:
        raise RuntimeError(f"No LAD intervals were retained for {genome} from {lad_source_path}")
    write_bed(lad_df, lad_clean_path)

    occupancy_df = build_binary_signal_df(lad_df, value=1.0)
    write_bedgraph(occupancy_df, occupancy_bedgraph_path)
    write_bigwig(occupancy_df, chrom_sizes, occupancy_bw_path)

    signal_df = clean_signal_df(
        read_bedgraph_signal(signal_source_path),
        chrom_sizes=chrom_sizes,
    )
    if signal_df.empty:
        raise RuntimeError(f"No laminB1 signal rows were retained for {genome} from {signal_source_path}")
    write_bedgraph(signal_df, signal_clean_path)
    write_bigwig(signal_df, chrom_sizes, signal_bw_path)

    return {
        "genome": genome,
        "chrom_sizes": chrom_sizes,
        "chrom_sizes_path": chrom_sizes_path,
        "lad_df": lad_df,
        "lad_clean_path": lad_clean_path,
        "occupancy_bedgraph_path": occupancy_bedgraph_path,
        "occupancy_bw_path": occupancy_bw_path,
        "signal_clean_path": signal_clean_path,
        "signal_bw_path": signal_bw_path,
    }


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Run cached LAD analysis outputs for downstream figure generation."
    )
    parser.add_argument(
        "--segmentation-results-path",
        type=Path,
        default=DEFAULT_SEGMENTATION_RESULTS_PATH,
        help="Directory containing the comparator pathway outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where LAD tables, figures, and deepTools outputs will be written. "
            "Defaults to results/04_lad_analysis."
        ),
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=list(DEFAULT_SELECTED_SAMPLES),
        help="Samples to score against LAD tracks. Defaults to the active notebook cohort.",
    )
    deeptools_group = parser.add_mutually_exclusive_group()
    deeptools_group.add_argument(
        "--skip-deeptools",
        dest="skip_deeptools",
        action="store_true",
        help="Skip LAD deepTools matrix and profile generation.",
    )
    deeptools_group.add_argument(
        "--run-deeptools",
        dest="skip_deeptools",
        action="store_false",
        help="Run LAD deepTools matrix and profile generation. Disabled by default for faster runs.",
    )
    parser.set_defaults(skip_deeptools=True)
    parser.add_argument(
        "--include-heatmaps",
        action="store_true",
        help="Generate LAD deepTools heatmaps in addition to the default profile plots.",
    )
    parser.add_argument(
        "--force-reference-rebuild",
        action="store_true",
        help="Rebuild cached LAD reference tracks even if they already exist.",
    )
    parser.add_argument(
        "--force-deeptools",
        action="store_true",
        help="Regenerate deepTools outputs even when the files already exist.",
    )
    parser.add_argument(
        "--primary-window-bp",
        type=int,
        default=DEFAULT_PRIMARY_WINDOW_BP,
        help="Flank size used for LAD deepTools profiles.",
    )
    parser.add_argument(
        "--profile-bin-bp",
        type=int,
        default=DEFAULT_PROFILE_BIN_BP,
        help="Minimum region length and bin size used for LAD deepTools profiles.",
    )
    parser.add_argument(
        "--profile-region-body-bp",
        type=int,
        default=DEFAULT_PROFILE_REGION_BODY_BP,
        help="Scaled body length used for LAD deepTools profiles.",
    )
    parser.add_argument(
        "--lad-null-permutations",
        type=int,
        default=DEFAULT_LAD_NULL_PERMUTATIONS,
        help="Number of matched-null permutations per sample/tool pair.",
    )
    parser.add_argument(
        "--lad-null-seed",
        type=int,
        default=DEFAULT_LAD_NULL_SEED,
        help="Base random seed for LAD null-model permutations.",
    )
    return parser


def run(
    segmentation_results_path,
    output_dir=None,
    samples=None,
    skip_deeptools=False,
    include_heatmaps=False,
    force_reference_rebuild=False,
    force_deeptools=False,
    primary_window_bp=DEFAULT_PRIMARY_WINDOW_BP,
    profile_bin_bp=DEFAULT_PROFILE_BIN_BP,
    profile_region_body_bp=DEFAULT_PROFILE_REGION_BODY_BP,
    lad_null_permutations=DEFAULT_LAD_NULL_PERMUTATIONS,
    lad_null_seed=DEFAULT_LAD_NULL_SEED,
):
    segmentation_results_path = Path(segmentation_results_path).resolve()
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir).resolve()
    samples = list(samples or DEFAULT_SELECTED_SAMPLES)
    profile_flank_bp = int(primary_window_bp)

    reference_dir = output_dir / "reference_tracks"
    cleaned_region_dir = output_dir / "cleaned_regions"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    deeptools_dir = output_dir / "deeptools"
    for directory in [output_dir, reference_dir, cleaned_region_dir, tables_dir, figures_dir, deeptools_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    for required_path in [
        DEFAULT_LAD_INTERVAL_TRACK_PATH,
        DEFAULT_LAMINB1_SIGNAL_TRACK_PATH,
        DEFAULT_LIFTOVER_SCRIPT_PATH,
        DEFAULT_HG19_TO_HG38_CHAIN,
    ]:
        if not Path(required_path).exists():
            raise FileNotFoundError(f"Missing required LAD reference file: {required_path}")

    missing_base_commands = [
        command for command in ["bedtools", "Rscript"] if shutil.which(command) is None
    ]
    if missing_base_commands:
        raise RuntimeError(
            "Required LAD analysis commands are not available on PATH: "
            + ", ".join(missing_base_commands)
        )

    run_deeptools = not skip_deeptools
    if run_deeptools:
        required_commands = ["computeMatrix", "plotProfile"] + (
            ["plotHeatmap"] if include_heatmaps else []
        )
        missing_deeptools = [command for command in required_commands if shutil.which(command) is None]
        if missing_deeptools:
            raise RuntimeError(
                "deepTools commands are not available on PATH: " + ", ".join(missing_deeptools)
            )

    print("LAD analysis configuration:")
    print(f"  Segmentation results: {segmentation_results_path}")
    print(f"  Output dir:           {output_dir}")
    print(f"  Samples:              {', '.join(samples)}")
    print(f"  Profile flank:        {int(profile_flank_bp):,} bp")
    print(f"  Run deepTools:        {run_deeptools}")
    print(f"  Include heatmaps:     {include_heatmaps}")
    print(f"  LAD null permutations:{int(lad_null_permutations)}")

    methylseg_results_dir = segmentation_results_path / "methylseg"
    if methylseg_results_dir.exists():
        available_sample_dirs = sorted(
            sample_dir.name for sample_dir in methylseg_results_dir.iterdir() if sample_dir.is_dir()
        )
        missing_selected_samples = sorted(set(samples) - set(available_sample_dirs))
        if missing_selected_samples:
            print(f"Requested samples missing from {methylseg_results_dir}: {missing_selected_samples}")

    sample_genome_rows = []
    for sample in samples:
        sample_genome_rows.append(
            {
                "sample": sample,
                "sample_id": sample_to_sample_id(sample),
                "genome": resolve_sample_genome(
                    sample,
                    segmentation_results_path=segmentation_results_path,
                ),
            }
        )

    sample_genome_df = (
        pd.DataFrame(sample_genome_rows).sort_values(["genome", "sample"]).reset_index(drop=True)
    )
    if sorted(sample_genome_df["sample"].tolist()) != sorted(samples):
        raise AssertionError("Resolved sample genomes did not match the requested sample list.")
    sample_genome_lookup = dict(zip(sample_genome_df["sample"], sample_genome_df["genome"]))
    lad_sample_genomes_path = tables_dir / "lad_sample_genomes.tsv"
    sample_genome_df.to_csv(lad_sample_genomes_path, sep="\t", index=False)
    _print_dataframe("Sample genomes", sample_genome_df)

    reference_lookup = {}
    reference_rows = []
    for genome in sorted(sample_genome_df["genome"].unique()):
        reference = prepare_lad_reference(
            genome=genome,
            reference_dir=reference_dir,
            lad_interval_track_path=DEFAULT_LAD_INTERVAL_TRACK_PATH,
            laminb1_signal_track_path=DEFAULT_LAMINB1_SIGNAL_TRACK_PATH,
            liftover_script_path=DEFAULT_LIFTOVER_SCRIPT_PATH,
            hg19_to_hg38_chain=DEFAULT_HG19_TO_HG38_CHAIN,
            force=force_reference_rebuild,
        )
        reference_lookup[genome] = reference
        reference_rows.append(
            {
                "genome": genome,
                "chrom_sizes_path": str(reference["chrom_sizes_path"]),
                "lad_clean_path": str(reference["lad_clean_path"]),
                "occupancy_bw_path": str(reference["occupancy_bw_path"]),
                "signal_bw_path": str(reference["signal_bw_path"]),
                "n_lad_regions": int(len(reference["lad_df"])),
                "lad_total_bp": int(reference["lad_df"]["length"].sum()),
            }
        )
    reference_summary_df = pd.DataFrame(reference_rows).sort_values("genome").reset_index(drop=True)
    _print_dataframe("LAD references", reference_summary_df)

    (
        filtered_reference_lookup,
        lad_sample_means_df,
        lad_filtered_ids_df,
        reference_filter_summary_df,
    ) = build_filtered_lad_reference_lookup(
        samples=samples,
        sample_genome_lookup=sample_genome_lookup,
        segmentation_results_path=segmentation_results_path,
        reference_lookup=reference_lookup,
        min_lad_beta=MIN_LAD_BETA,
    )
    lad_reference_summary_df = reference_summary_df.merge(
        reference_filter_summary_df,
        on="genome",
        how="left",
    )
    lad_reference_summary_path = tables_dir / "lad_reference_summary.tsv"
    lad_reference_summary_df.to_csv(lad_reference_summary_path, sep="\t", index=False)
    if not lad_sample_means_df.empty:
        lad_sample_means_df.to_csv(
            tables_dir / "lad_sample_lad_means.tsv", sep="\t", index=False
        )
    if not lad_filtered_ids_df.empty:
        lad_filtered_ids_df.to_csv(
            tables_dir / "lad_filtered_lad_ids.tsv", sep="\t", index=False
        )
    _print_dataframe("LAD reference summary", lad_reference_summary_df)

    missing_region_paths = []
    for sample in samples:
        for tool in REQUIRED_TOOLS:
            tool_config = TOOL_CONFIG_BY_NAME[tool]
            expected_path = build_tool_region_path(segmentation_results_path, sample, tool_config)
            if not expected_path.exists():
                missing_region_paths.append(
                    {
                        "sample": sample,
                        "tool": tool,
                        "tool_label": tool_config["tool_label"],
                        "expected_path": str(expected_path),
                    }
                )
    if missing_region_paths:
        missing_region_df = (
            pd.DataFrame(missing_region_paths)
            .sort_values(["sample", "tool_label"])
            .reset_index(drop=True)
        )
        _print_dataframe("Missing segmentation result files", missing_region_df)
        raise FileNotFoundError(
            "One or more expected segmentation result files were missing. "
            "Update segmentation_results_path or regenerate the missing tool outputs before rerunning LAD analysis."
        )

    region_interval_dfs = {}
    manifest_rows = []
    for sample in samples:
        sample_genome = sample_genome_lookup[sample]
        sample_id = sample_to_sample_id(sample)

        for tool in REQUIRED_TOOLS:
            tool_config = TOOL_CONFIG_BY_NAME[tool]
            raw_region_df = load_tool_regions(
                tool_config,
                sample,
                segmentation_results_path=segmentation_results_path,
            )
            clean_region_df = clean_and_merge_intervals(raw_region_df)

            clean_region_path = cleaned_region_dir / f"{sample}.{tool}.bed"
            write_bed(clean_region_df, clean_region_path)
            region_interval_dfs[(sample, tool)] = clean_region_df

            manifest_rows.append(
                {
                    "sample": sample,
                    "sample_id": sample_id,
                    "genome": sample_genome,
                    "tool": tool,
                    "tool_label": tool_config["tool_label"],
                    "tool_family": tool_config["tool_family"],
                    "platform": tool_config["platform"],
                    "region_type": tool_config["region_type"],
                    "rank_order": int(tool_config["rank_order"]),
                    "clean_region_path": str(clean_region_path),
                    "n_regions": int(len(clean_region_df)),
                    "total_bp": int(clean_region_df["length"].sum()) if not clean_region_df.empty else 0,
                }
            )

    manifest_df = (
        pd.DataFrame(manifest_rows).sort_values(["sample", "rank_order", "tool"]).reset_index(drop=True)
    )
    if len(manifest_df) != len(samples) * len(REQUIRED_TOOLS):
        raise AssertionError(
            f"Expected {len(samples) * len(REQUIRED_TOOLS)} manifest rows, found {len(manifest_df)}"
        )
    if (manifest_df.groupby(["sample", "tool"]).size() > 1).any():
        raise AssertionError("Expected exactly one manifest row per sample/tool combination.")
    assert_expected_tools(manifest_df, "manifest_df")
    lad_region_manifest_path = tables_dir / "lad_region_manifest.tsv"
    manifest_df.to_csv(lad_region_manifest_path, sep="\t", index=False)
    _print_dataframe(
        "LAD region manifest",
        manifest_df,
        ["sample", "tool_label", "platform", "region_type", "n_regions", "total_bp"],
    )

    metric_rows = []
    overlap_detail_rows = []
    distance_detail_rows = []
    for row in manifest_df.itertuples(index=False):
        region_df = prepare_regions_for_lad_overlap(region_interval_dfs[(row.sample, row.tool)])
        lad_df = prepare_lads_for_lad_overlap(filtered_reference_lookup[row.genome])
        print(
            f"Scoring LAD metrics for {row.sample} {row.tool_label} "
            f"({len(region_df):,} regions; {len(lad_df):,} LADs)"
        )
        overlaps = filter_regions_by_lad_overlap(
            region_df,
            lad_df,
        )
        overlap_detail_df = build_region_lad_overlap_detail_df(region_df, overlaps)
        distance_df = compute_region_lad_distances_or_empty(overlaps, region_df, lad_df)
        distance_detail_df = build_region_lad_distance_detail_df(region_df, distance_df)
        metric_base = {
            "sample": row.sample,
            "sample_id": row.sample_id,
            "genome": row.genome,
            "tool": row.tool,
            "tool_label": row.tool_label,
            "tool_family": row.tool_family,
            "platform": row.platform,
            "region_type": row.region_type,
            "rank_order": int(row.rank_order),
            "n_regions": int(len(region_df)),
            "n_lad_regions": int(len(lad_df)),
        }
        metric_rows.extend(
            build_lad_association_metric_rows(
                metric_base,
                overlaps,
                region_df,
                lad_df,
                distance_df=distance_df,
            )
        )

        detail_base = {
            "sample": row.sample,
            "sample_id": row.sample_id,
            "genome": row.genome,
            "tool": row.tool,
            "tool_label": row.tool_label,
            "tool_family": row.tool_family,
            "platform": row.platform,
            "region_type": row.region_type,
            "rank_order": int(row.rank_order),
        }
        overlap_detail_rows.append(overlap_detail_df.assign(**detail_base))
        distance_detail_rows.append(distance_detail_df.assign(**detail_base))

    metrics_df = (
        pd.DataFrame(metric_rows)
        .sort_values(
            [
                "sample",
                "rank_order",
                "tool",
                "analysis_mode",
                "threshold_bp",
                "metric",
            ]
        )
        .reset_index(drop=True)
    )
    assert_expected_tools(metrics_df, "metrics_df")
    expected_metric_rows = len(manifest_df) * len(LAD_ASSOCIATION_METRICS)
    if len(metrics_df) != expected_metric_rows:
        raise AssertionError(
            f"Expected {expected_metric_rows} metric rows, found {len(metrics_df)}"
        )

    overlap_detail_df = (
        pd.concat(overlap_detail_rows, ignore_index=True)
        if overlap_detail_rows
        else pd.DataFrame()
    )
    distance_detail_df = (
        pd.concat(distance_detail_rows, ignore_index=True)
        if distance_detail_rows
        else pd.DataFrame()
    )

    lad_association_metrics_path = tables_dir / "lad_association_metrics.tsv"
    lad_region_overlap_details_path = tables_dir / "lad_region_overlap_details.tsv"
    lad_region_distance_details_path = tables_dir / "lad_region_distance_details.tsv"
    lad_boundary_distance_details_path = tables_dir / "lad_boundary_distance_details.tsv"
    metrics_df.to_csv(lad_association_metrics_path, sep="\t", index=False)
    overlap_detail_df.to_csv(lad_region_overlap_details_path, sep="\t", index=False)
    distance_detail_df.to_csv(lad_region_distance_details_path, sep="\t", index=False)
    distance_detail_df.to_csv(lad_boundary_distance_details_path, sep="\t", index=False)
    _print_dataframe(
        "LAD association metrics",
        metrics_df,
        [
            "sample",
            "tool_label",
            "analysis_mode",
            "threshold_bp",
            "metric",
            "score",
            "n_regions",
            "n_regions_passing_threshold",
        ],
    )

    (
        unique_lad_details_df,
        unique_lad_summary_df,
        unique_lad_exports_df,
    ) = build_unique_lad_exports(
        samples=samples,
        sample_genome_lookup=sample_genome_lookup,
        segmentation_results_path=segmentation_results_path,
        filtered_reference_lookup=filtered_reference_lookup,
        output_dir=output_dir,
    )
    lad_unique_lad_region_details_path = tables_dir / "lad_unique_lad_region_details.tsv"
    lad_unique_lad_summary_path = tables_dir / "lad_unique_lad_summary.tsv"
    lad_unique_lad_exports_path = tables_dir / "lad_unique_lad_exports.tsv"
    unique_lad_details_df.to_csv(lad_unique_lad_region_details_path, sep="\t", index=False)
    unique_lad_summary_df.to_csv(lad_unique_lad_summary_path, sep="\t", index=False)
    unique_lad_exports_df.to_csv(lad_unique_lad_exports_path, sep="\t", index=False)

    (
        lad_null_observed_df,
        lad_null_permutation_df,
        lad_null_summary_df,
        lad_null_checks_df,
    ) = run_lad_null_model(
        samples=samples,
        sample_genome_lookup=sample_genome_lookup,
        segmentation_results_path=segmentation_results_path,
        filtered_reference_lookup=filtered_reference_lookup,
        n_permutations=lad_null_permutations,
        null_seed=lad_null_seed,
    )
    lad_null_observed_path = tables_dir / "lad_null_observed.tsv"
    lad_null_permutations_path = tables_dir / "lad_null_permutations.tsv"
    lad_null_summary_path = tables_dir / "lad_null_summary.tsv"
    lad_null_checks_path = tables_dir / "lad_null_run_checks.tsv"
    lad_null_observed_df.to_csv(lad_null_observed_path, sep="\t", index=False)
    lad_null_permutation_df.to_csv(lad_null_permutations_path, sep="\t", index=False)
    lad_null_summary_df.to_csv(lad_null_summary_path, sep="\t", index=False)
    lad_null_checks_df.to_csv(lad_null_checks_path, sep="\t", index=False)

    combined_summary_df = (
        metrics_df.groupby(
            [
                "tool",
                "tool_label",
                "tool_family",
                "platform",
                "region_type",
                "rank_order",
                "analysis_mode",
                "threshold_bp",
                "metric",
            ],
            as_index=False,
        )
        .agg(
            samples_scored=("sample", "nunique"),
            mean_n_regions=("n_regions", "mean"),
            mean_n_lad_regions=("n_lad_regions", "mean"),
            mean_n_regions_passing_threshold=("n_regions_passing_threshold", "mean"),
            mean_score=("score", "mean"),
        )
        .sort_values(
            ["analysis_mode", "metric", "threshold_bp", "mean_score", "rank_order", "tool_label"],
            ascending=[True, True, True, False, True, True],
        )
        .reset_index(drop=True)
    )
    sample_score_pivot = (
        metrics_df.pivot_table(
            index=["tool", "analysis_mode", "threshold_bp", "metric"],
            columns="sample",
            values="score",
            aggfunc="first",
        )
        .reset_index()
    )
    combined_summary_df = combined_summary_df.merge(
        sample_score_pivot,
        on=["tool", "analysis_mode", "threshold_bp", "metric"],
        how="left",
    )
    lad_combined_summary_path = tables_dir / "lad_combined_summary.tsv"
    combined_summary_df.to_csv(lad_combined_summary_path, sep="\t", index=False)
    _print_dataframe(
        "Combined LAD summary",
        combined_summary_df,
        [
            "tool_label",
            "platform",
            "region_type",
            "analysis_mode",
            "threshold_bp",
            "metric",
            "samples_scored",
            "mean_n_regions",
            "mean_n_lad_regions",
            "mean_n_regions_passing_threshold",
            "mean_score",
        ]
        + samples,
    )

    plot_palette = build_tool_palette(combined_summary_df)
    removed_stale_plots = clear_stale_lad_summary_pngs(figures_dir)
    if removed_stale_plots:
        print(f"Removed {removed_stale_plots} stale LAD summary PNGs from {figures_dir}")

    plot_rows = []
    for metric in LAD_ASSOCIATION_METRICS:
        ascending = not LAD_METRIC_HIGHER_IS_BETTER[metric]
        analysis_mode = LAD_METRIC_MODES[metric]
        threshold_bp = int(LAD_METRIC_THRESHOLDS_BP[metric])
        for sample, sample_df in metrics_df.groupby("sample", sort=True):
            plot_df_for_sample = sample_df[sample_df["metric"] == metric]
            ordered_df = plot_df_for_sample.sort_values(
                ["score", "rank_order", "tool_label"],
                ascending=[ascending, True, True],
            ).reset_index(drop=True)
            plot_order = ordered_df["tool_label"].tolist()

            fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
            sns.barplot(
                data=ordered_df,
                x="score",
                y="tool_label",
                hue="tool_label",
                order=plot_order,
                hue_order=plot_order,
                dodge=False,
                palette=plot_palette,
                legend=False,
                ax=ax,
            )
            score_max = ordered_df["score"].max()
            if pd.notna(score_max) and score_max > 0:
                ax.set_xlim(0, score_max * 1.05)
            ax.set_title(f"{sample} {LAD_METRIC_LABELS[metric]}")
            ax.set_xlabel("Score")
            ax.set_ylabel("")
            figure_path = figures_dir / f"{sample}.{metric}.lad_summary.png"
            fig.savefig(figure_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            plot_rows.append(
                {
                    "sample": sample,
                    "analysis_mode": analysis_mode,
                    "metric": metric,
                    "threshold_bp": threshold_bp,
                    "plot_type": "sample",
                    "plot_path": str(figure_path),
                    "summary_plot": str(figure_path),
                }
            )

        combined_plot_df = combined_summary_df[
            combined_summary_df["metric"] == metric
        ].sort_values(
            ["mean_score", "rank_order", "tool_label"],
            ascending=[ascending, True, True],
        ).reset_index(drop=True)
        combined_plot_order = combined_plot_df["tool_label"].tolist()
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        sns.barplot(
            data=combined_plot_df,
            x="mean_score",
            y="tool_label",
            hue="tool_label",
            order=combined_plot_order,
            hue_order=combined_plot_order,
            dodge=False,
            palette=plot_palette,
            legend=False,
            ax=ax,
        )
        score_max = combined_plot_df["mean_score"].max()
        if pd.notna(score_max) and score_max > 0:
            ax.set_xlim(0, score_max * 1.05)
        ax.set_title(f"Combined {LAD_METRIC_LABELS[metric]}")
        ax.set_xlabel("Mean score")
        ax.set_ylabel("")
        combined_rank_path = figures_dir / f"combined.{metric}.lad_summary.png"
        fig.savefig(combined_rank_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        plot_rows.append(
            {
                "sample": "combined",
                "analysis_mode": analysis_mode,
                "metric": metric,
                "threshold_bp": threshold_bp,
                "plot_type": "combined",
                "plot_path": str(combined_rank_path),
                "summary_plot": str(combined_rank_path),
            }
        )

    lad_plot_outputs_path = tables_dir / "lad_plot_outputs.tsv"
    plot_df = (
        pd.DataFrame(plot_rows)
        .sort_values(["analysis_mode", "metric", "threshold_bp", "plot_type", "sample"])
        .reset_index(drop=True)
    )
    plot_df.to_csv(lad_plot_outputs_path, sep="\t", index=False)
    _print_dataframe("LAD plot outputs", plot_df)

    deeptools_rows = []
    if run_deeptools:
        deeptools_region_rows = []
        for row in manifest_df.itertuples(index=False):
            sample_output_dir = deeptools_dir / row.sample
            sample_output_dir.mkdir(parents=True, exist_ok=True)

            filtered_region_df = filter_regions_for_deeptools(
                region_interval_dfs[(row.sample, row.tool)],
                profile_bin_bp,
            )
            deeptools_region_path = sample_output_dir / f"{row.sample}.{row.tool}.deeptools_regions.bed"
            write_bed(filtered_region_df, deeptools_region_path)
            deeptools_region_rows.append(
                {
                    "sample": row.sample,
                    "sample_id": row.sample_id,
                    "tool": row.tool,
                    "tool_label": row.tool_label,
                    "rank_order": int(row.rank_order),
                    "deeptools_region_path": str(deeptools_region_path),
                    "total_regions": int(row.n_regions),
                    "visualized_regions": int(len(filtered_region_df)),
                    "excluded_short_regions": int(row.n_regions - len(filtered_region_df)),
                    "min_region_length_bp": int(profile_bin_bp),
                }
            )

        deeptools_region_df = (
            pd.DataFrame(deeptools_region_rows)
            .sort_values(["sample", "rank_order", "tool"])
            .reset_index(drop=True)
        )
        lad_deeptools_region_manifest_path = tables_dir / "lad_deeptools_region_manifest.tsv"
        deeptools_region_df.to_csv(lad_deeptools_region_manifest_path, sep="\t", index=False)
        _print_dataframe(
            "LAD deepTools region manifest",
            deeptools_region_df,
            [
                "sample",
                "tool_label",
                "total_regions",
                "visualized_regions",
                "excluded_short_regions",
                "deeptools_region_path",
            ],
        )

        for sample in samples:
            sample_id = sample_to_sample_id(sample)
            genome = sample_genome_lookup[sample]
            sample_output_dir = deeptools_dir / sample
            sample_region_rows = [
                row for row in deeptools_region_rows if row["sample"] == sample
            ]

            occupancy_output = run_deeptools_profile(
                sample=sample,
                sample_id=sample_id,
                signal_path=reference_lookup[genome]["occupancy_bw_path"],
                signal_label="LAD occupancy",
                plot_label="LAD occupancy",
                region_rows=sample_region_rows,
                deeptools_tool_order=REQUIRED_TOOLS,
                sample_output_dir=sample_output_dir,
                file_stem="lad_occupancy",
                bin_size_bp=profile_bin_bp,
                flank_bp=profile_flank_bp,
                region_body_bp=profile_region_body_bp,
                include_heatmaps=include_heatmaps,
                force=force_deeptools,
            )
            signal_output = run_deeptools_profile(
                sample=sample,
                sample_id=sample_id,
                signal_path=reference_lookup[genome]["signal_bw_path"],
                signal_label="LaminB1 signal",
                plot_label="laminB1 signal",
                region_rows=sample_region_rows,
                deeptools_tool_order=REQUIRED_TOOLS,
                sample_output_dir=sample_output_dir,
                file_stem="laminB1_signal",
                bin_size_bp=profile_bin_bp,
                flank_bp=profile_flank_bp,
                region_body_bp=profile_region_body_bp,
                include_heatmaps=include_heatmaps,
                force=force_deeptools,
            )
            deeptools_rows.append(
                {
                    "sample": sample,
                    "sample_id": sample_id,
                    "genome": genome,
                    "occupancy_matrix_path": occupancy_output["matrix_path"],
                    "occupancy_sorted_regions_path": occupancy_output["sorted_regions_path"],
                    "occupancy_profile_path": occupancy_output["profile_path"],
                    "occupancy_heatmap_path": occupancy_output["heatmap_path"],
                    "signal_matrix_path": signal_output["matrix_path"],
                    "signal_sorted_regions_path": signal_output["sorted_regions_path"],
                    "signal_profile_path": signal_output["profile_path"],
                    "signal_heatmap_path": signal_output["heatmap_path"],
                    "include_heatmaps": bool(include_heatmaps),
                    "visualized_tools": int(signal_output["visualized_tools"]),
                }
            )
    else:
        print("Skipping LAD deepTools execution because --skip-deeptools was provided.")

    lad_profile_outputs_path = tables_dir / "lad_profile_outputs.tsv"
    if deeptools_rows:
        deeptools_outputs_df = pd.DataFrame(deeptools_rows).sort_values("sample").reset_index(drop=True)
    else:
        deeptools_outputs_df = pd.DataFrame(
            columns=[
                "sample",
                "sample_id",
                "genome",
                "occupancy_matrix_path",
                "occupancy_sorted_regions_path",
                "occupancy_profile_path",
                "occupancy_heatmap_path",
                "signal_matrix_path",
                "signal_sorted_regions_path",
                "signal_profile_path",
                "signal_heatmap_path",
                "include_heatmaps",
                "visualized_tools",
            ]
        )
    deeptools_outputs_df.to_csv(lad_profile_outputs_path, sep="\t", index=False)
    _print_dataframe("LAD profile outputs", deeptools_outputs_df)

    print("\nLAD analysis complete.")
    print(f"  Sample genomes:     {lad_sample_genomes_path}")
    print(f"  Reference summary:  {lad_reference_summary_path}")
    print(f"  Region manifest:   {lad_region_manifest_path}")
    print(f"  Metrics:           {lad_association_metrics_path}")
    print(f"  Overlap details:   {lad_region_overlap_details_path}")
    print(f"  Distance details:  {lad_region_distance_details_path}")
    print(f"  Boundary details:  {lad_boundary_distance_details_path}")
    print(f"  Unique LAD exports:{lad_unique_lad_exports_path}")
    print(f"  Null observed:     {lad_null_observed_path}")
    print(f"  Null permutations: {lad_null_permutations_path}")
    print(f"  Null summary:      {lad_null_summary_path}")
    print(f"  Null checks:       {lad_null_checks_path}")
    print(f"  Combined summary:  {lad_combined_summary_path}")
    print(f"  Plot manifest:     {lad_plot_outputs_path}")
    print(f"  Profile outputs:   {lad_profile_outputs_path}")


def main(argv=None):
    args = _build_parser().parse_args(argv)
    run(
        segmentation_results_path=args.segmentation_results_path,
        output_dir=args.output_dir,
        samples=args.samples,
        skip_deeptools=args.skip_deeptools,
        include_heatmaps=args.include_heatmaps,
        force_reference_rebuild=args.force_reference_rebuild,
        force_deeptools=args.force_deeptools,
        primary_window_bp=args.primary_window_bp,
        profile_bin_bp=args.profile_bin_bp,
        profile_region_body_bp=args.profile_region_body_bp,
        lad_null_permutations=args.lad_null_permutations,
        lad_null_seed=args.lad_null_seed,
    )


if __name__ == "__main__":
    main()
