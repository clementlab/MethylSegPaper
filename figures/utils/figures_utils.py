from __future__ import annotations

import hashlib
import json
from io import StringIO
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyBigWig
import seaborn as sns
import yaml
from plotly.subplots import make_subplots

FIGURES_DIR = Path(__file__).resolve().parents[1]
if str(FIGURES_DIR) not in sys.path:
    sys.path.insert(0, str(FIGURES_DIR))

from colors import (
    ANNOTATION_COLORS,
    LAD_TOOL_COLORS,
    NEUTRAL_COLORS,
    SYNTHETIC_INTERVAL_STYLES,
    SYNTHETIC_TRACK_COLORS,
    TOOL_DISTINCT_COLORS_BY_SLUG,
    TOOL_HIGHLIGHT_COLORS,
    TOOL_HIGHLIGHT_COLORS_BY_SLUG,
)


DATA_DIR = Path(
    "/uufs/chpc.utah.edu/common/home/u0914269/clement/projects/20260624_methylseg/data"
)
METHYLATION_DATA_DIR = DATA_DIR / "methylation_data"
TCGA_SAMPLE_REF = DATA_DIR / "reference_data/runAll.sh.samples"
METH_REF = DATA_DIR / "tcga_samples/parse450K.pl.order.lookup"
CANONICAL_CHROMS = frozenset(
    [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
)
RESULTS_DIR = Path("/uufs/chpc.utah.edu/common/home/u0914269/clement/projects/20260624_methylseg/results")
OUT_DIR = Path("/uufs/chpc.utah.edu/common/home/u0914269/clement/projects/20260624_methylseg/figures/out")
METHYL_SEG_FIGURE_OUTPUT_DIR = OUT_DIR / "methyl_seg_figures"
REGION_CALLING_FIGURE_OUTPUT_DIR = OUT_DIR / "region_calling_figures"
SYNTHETIC_FIGURE_OUTPUT_DIR = OUT_DIR / "03_synthetic_figures"
CHROMATIN_FIGURE_OUTPUT_DIR = OUT_DIR / "chromatin_figures"
REFERENCE_DATA_DIR = DATA_DIR / "reference_data"
METHYLSEG_RESULTS_DIR = RESULTS_DIR / "01_region_calling_analysis" / "methylseg"
OVERLAP_RESULTS_DIR = (
    RESULTS_DIR / "01_region_calling_analysis" / "methylseg_hm450k_overlap"
)
OVERLAP_CACHE_VERSION = "v2_union_coverage"
METHYLSEG_TUMOR_SAMPLE_IDS = [
    "ESO26.wgbs",
    "TE5.wgbs",
    "WGBS_colon-primary-tumor_1_meth",
    "WGBS_colon-primary-tumor_2_meth",
    "WGBS_colon-primary-tumor_3_meth",
]
HM450K_REGION_COLUMNS = [
    "sample_id",
    "category",
    "platform",
    "chrom",
    "start",
    "end",
    "region_length_bp",
    "max_overlap_bp",
    "max_overlap_fraction_hm450k",
    "n_qualifying_overlaps",
    "hm450k_probe_count",
    "wgbs_cpg_count",
    "probe_reduction_fraction",
    "avg_methylation",
]
WGBS_REGION_COLUMNS = [
    "sample_id",
    "category",
    "platform",
    "chrom",
    "start",
    "end",
    "region_length_bp",
    "max_overlap_bp",
    "max_overlap_fraction_hm450k",
    "n_qualifying_overlaps",
    "hm450k_probe_count",
    "wgbs_cpg_count",
    "probe_reduction_fraction",
    "avg_methylation",
]
TOOL_REGISTRY = [
    {
        "tool": "methylseg",
        "tool_label": "MethylSeg WGBS",
        "parser_family": "methylseg",
        "tool_family": "methylseg",
        "platform": "wgbs",
        "region_type": "PMD",
        "rank_order": 0,
        "path_parts": [
            "methylseg",
            "{sample}",
            "out",
            "wgbs",
            "summary_files",
            "segments_cleaned_PMD.bed",
        ],
        "columns": ["chrom", "start", "end", "label"],
        "header": False,
    },
    {
        "tool": "methylseg_hm450k",
        "tool_label": "MethylSeg HM450K",
        "parser_family": "methylseg",
        "tool_family": "methylseg",
        "platform": "hm450k",
        "region_type": "PMD",
        "rank_order": 1,
        "path_parts": [
            "methylseg",
            "{sample}",
            "out",
            "hm450k",
            "summary_files",
            "segments_cleaned_PMD.bed",
        ],
        "columns": ["chrom", "start", "end", "label"],
        "header": False,
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
        "columns": [
            "chrom",
            "start",
            "end",
        ],
        "header": True,
    },
    {
        "tool": "dnmtools",
        "tool_label": "DNMTools WGBS",
        "parser_family": "dnmtools",
        "tool_family": "dnmtools",
        "platform": "wgbs",
        "region_type": "PMD",
        "rank_order": 3,
        "path_parts": ["dnmtools", "{sample}", "out", "dnmtools_PMDs.bed"],
        "columns": ["chrom", "start", "end", "label", "score", "strand"],
        "header": False,
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
        "columns": ["chrom", "start", "end", "label", "score", "strand"],
        "header": False,
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
        "columns": ["chrom", "start", "end", "label", "score", "strand"],
        "header": False,
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
        "columns": [
            "chrom",
            "start",
            "end",
        ],
        "header": False,
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
        "columns": ["chrom", "start", "end", "num.cpgs", "meth", "std", "category"],
        "header": True,
    },
]

IGV_EXPORT_TOOL_ORDER = [
    "methylseg_wgbs",
    "methylseg_hm450k",
    "methylseekr",
    "dnmtools",
    "dnmtools_array",
    "dnmtools_pmr",
    "mmseekr",
    "methylasso",
]

TOOL_ALIASES = {
    "methylseg": "methylseg_wgbs",
    "methylseg_wgbs": "methylseg_wgbs",
    "methylseg_hm450k": "methylseg_hm450k",
    "methylseekr": "methylseekr",
    "dnmtools": "dnmtools",
    "dnmtools_array": "dnmtools_array",
    "dnmtools_pmr": "dnmtools_pmr",
    "mmseekr": "mmseekr",
    "methyl_lasso": "methylasso",
    "methylasso": "methylasso",
}

CANONICAL_TOOL_BY_EXPORT_SLUG = {
    "methylseg_wgbs": "methylseg",
    "methylseg_hm450k": "methylseg_hm450k",
    "methylseekr": "methylseekr",
    "dnmtools": "dnmtools",
    "dnmtools_array": "dnmtools_array",
    "dnmtools_pmr": "dnmtools_pmr",
    "mmseekr": "mmseekr",
    "methylasso": "methyl_lasso",
}

REGION_TYPE_BY_EXPORT_SLUG = {
    "methylseg_wgbs": "pmd",
    "methylseg_hm450k": "pmd",
    "methylseekr": "pmd",
    "dnmtools": "pmd",
    "dnmtools_array": "pmd",
    "dnmtools_pmr": "pmr",
    "mmseekr": "pmd",
    "methylasso": "pmd",
}

def wgbs_cancer_samples() -> list[str]:
    return [
        "ESO26.wgbs",
        "TE5.wgbs",
        "WGBS_colon-primary-tumor_1_meth",
        # "WGBS_colon-primary-tumor_2_meth",
        # "WGBS_colon-primary-tumor_3_meth",
    ]

def _standardize_coord_df(
    df: pd.DataFrame,
    *,
    required_cols: list[str],
    numeric_cols: list[str],
) -> pd.DataFrame:
    df = df.copy()
    df["CpG_chrm"] = df["CpG_chrm"].astype(str)
    df["CpG_chrm"] = "chr" + df["CpG_chrm"].str.replace("^chr", "", regex=True)
    df["CpG_beg"] = pd.to_numeric(df["CpG_beg"], errors="coerce")
    df["CpG_end"] = pd.to_numeric(df["CpG_end"], errors="coerce")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["CpG_chrm"].isin(CANONICAL_CHROMS)].copy()
    df = df.dropna(subset=required_cols)
    df["CpG_beg"] = df["CpG_beg"].astype(np.int64)
    df["CpG_end"] = df["CpG_end"].astype(np.int64)
    df = df.sort_values(["CpG_chrm", "CpG_beg", "CpG_end"]).reset_index(drop=True)
    return df


def _standardize_region_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["chrom"] = df["chrom"].astype(str)
    df["chrom"] = "chr" + df["chrom"].str.replace("^chr", "", regex=True)
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df = df[df["chrom"].isin(CANONICAL_CHROMS)].copy()
    df = df.dropna(subset=["chrom", "start", "end"])
    df["start"] = df["start"].astype(np.int64)
    df["end"] = df["end"].astype(np.int64)
    df = df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    return df


def _standardize_beta_track_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(
        columns={
            "chr": "chrom",
            "CpG_chrm": "chrom",
            "CpG_beg": "start",
            "CpG_start": "start",
            "CpG_end": "end",
        }
    )
    missing_cols = [col for col in ["chrom", "start", "end", "beta"] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Beta track is missing required columns {missing_cols}.")

    df["chrom"] = df["chrom"].astype(str)
    df["chrom"] = "chr" + df["chrom"].str.replace("^chr", "", regex=True)
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    df = df[df["chrom"].isin(CANONICAL_CHROMS)].copy()
    df = df.dropna(subset=["chrom", "start", "end", "beta"])
    invalid_beta = ~df["beta"].between(0.0, 1.0)
    if invalid_beta.any():
        raise ValueError(
            f"Found {int(invalid_beta.sum())} beta values outside [0, 1] while preparing an IGV track."
        )
    df["start"] = df["start"].astype(np.int64)
    df["end"] = df["end"].astype(np.int64)
    df = df[["chrom", "start", "end", "beta"]].copy()

    conflicting_dups = (
        df.groupby(["chrom", "start", "end"], sort=False)["beta"].nunique(dropna=False) > 1
    )
    if conflicting_dups.any():
        raise ValueError(
            "Found duplicated methylation intervals with conflicting beta values while preparing an IGV track."
        )

    df = df.drop_duplicates(subset=["chrom", "start", "end"]).reset_index(drop=True)
    return df


def load_tcga_sample(sample_id: str) -> pd.DataFrame:
    samples_info = pd.read_csv(TCGA_SAMPLE_REF, sep="\t")
    meth_ref = pd.read_csv(METH_REF, sep="\t")
    meth_ref = meth_ref[["CpG_chrm", "CpG_beg", "CpG_end", "probeID"]]

    sample_info = samples_info.loc[samples_info["sample"].astype(str) == str(sample_id)]
    if sample_info.empty:
        raise FileNotFoundError(
            f"No TCGA sample found for sample_id={sample_id!r} in {TCGA_SAMPLE_REF}."
        )

    meth_file = Path(sample_info["methylation_file"].iloc[0])
    meth = np.load(meth_file).astype(float)
    meth[meth == 255] = np.nan

    methylation_df = meth_ref.copy()
    methylation_df["beta"] = meth / 100.0
    methylation_df = _standardize_coord_df(
        methylation_df,
        required_cols=["CpG_chrm", "CpG_beg", "CpG_end", "beta"],
        numeric_cols=["beta"],
    )
    return methylation_df[["CpG_chrm", "CpG_beg", "CpG_end", "beta", "probeID"]]


def _find_methylation_file(sample_id: str) -> Path:
    matches = sorted(
        path
        for path in METHYLATION_DATA_DIR.iterdir()
        if path.name.startswith(sample_id)
        and (path.name.endswith(".beta") or path.name.endswith(".bed.gz"))
    )
    if not matches:
        raise FileNotFoundError(
            f"No methylation file found for sample_id={sample_id!r} in {METHYLATION_DATA_DIR}."
        )

    hg38_matches = [path for path in matches if "hg38" in path.name]
    return hg38_matches[0] if hg38_matches else matches[0]


def _infer_genome(sample_id: str) -> str:
    if sample_id.startswith("WGBS_colon") or sample_id.endswith(".wgbs"):
        return "hg38"
    return "hg19"


def _load_beta_sample(meth_file: Path, genome: str) -> pd.DataFrame:
    result = subprocess.run(
        ["wgbstools", "view", "--genome", genome, str(meth_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    df = pd.read_csv(
        StringIO(result.stdout),
        sep="\t",
        header=None,
        names=["CpG_chrm", "CpG_beg", "CpG_end", "methylated_reads", "coverage"],
    )
    df = _standardize_coord_df(
        df,
        required_cols=[
            "CpG_chrm",
            "CpG_beg",
            "CpG_end",
            "methylated_reads",
            "coverage",
        ],
        numeric_cols=["methylated_reads", "coverage"],
    )
    df["methylated_reads"] = df["methylated_reads"].round().astype(np.int64)
    df["coverage"] = df["coverage"].round().astype(np.int64)
    return df.rename(columns={"CpG_beg": "CpG_start"})[
        ["CpG_chrm", "CpG_start", "CpG_end", "methylated_reads", "coverage"]
    ]


def _load_bed_gz_sample(meth_file: Path) -> pd.DataFrame:
    preview = pd.read_csv(meth_file, sep="\t", header=None, nrows=1, compression="gzip")
    n_cols = preview.shape[1]

    if n_cols == 5:
        df = pd.read_csv(
            meth_file,
            sep="\t",
            header=None,
            compression="gzip",
            names=["CpG_chrm", "CpG_beg", "CpG_end", "coverage", "meth_percent"],
        )
        df["methylated_reads"] = (
            pd.to_numeric(df["meth_percent"], errors="coerce")
            / 100.0
            * pd.to_numeric(df["coverage"], errors="coerce")
        )
        df = _standardize_coord_df(
            df,
            required_cols=[
                "CpG_chrm",
                "CpG_beg",
                "CpG_end",
                "methylated_reads",
                "coverage",
            ],
            numeric_cols=["methylated_reads", "coverage"],
        )
        df["methylated_reads"] = df["methylated_reads"].round().astype(np.int64)
        df["coverage"] = df["coverage"].round().astype(np.int64)
        return df.rename(columns={"CpG_beg": "CpG_start"})[
            ["CpG_chrm", "CpG_start", "CpG_end", "methylated_reads", "coverage"]
        ]

    if n_cols == 6:
        df = pd.read_csv(
            meth_file,
            sep="\t",
            header=None,
            compression="gzip",
            names=["CpG_chrm", "CpG_beg", "CpG_end", "beta", "coverage", "context"],
        )
        df["methylated_reads"] = (
            pd.to_numeric(df["beta"], errors="coerce")
            * pd.to_numeric(df["coverage"], errors="coerce")
        )
        df = _standardize_coord_df(
            df,
            required_cols=[
                "CpG_chrm",
                "CpG_beg",
                "CpG_end",
                "methylated_reads",
                "coverage",
            ],
            numeric_cols=["methylated_reads", "coverage"],
        )
        df["methylated_reads"] = df["methylated_reads"].round().astype(np.int64)
        df["coverage"] = df["coverage"].round().astype(np.int64)
        return df.rename(columns={"CpG_beg": "CpG_start"})[
            ["CpG_chrm", "CpG_start", "CpG_end", "methylated_reads", "coverage"]
        ]

    raise ValueError(f"Unsupported .bed.gz format with {n_cols} columns in {meth_file}")


def load_sample(sample_id: str) -> pd.DataFrame:
    if sample_id.lower().startswith("tcga"):
        return load_tcga_sample(sample_id)

    meth_file = _find_methylation_file(sample_id)
    if meth_file.suffix == ".beta":
        return _load_beta_sample(meth_file, _infer_genome(sample_id))
    if meth_file.suffixes[-2:] == [".bed", ".gz"]:
        return _load_bed_gz_sample(meth_file)
    raise ValueError(f"Unsupported methylation file type for {meth_file}")


def _get_tool_config(tool: str) -> dict:
    for tool_config in TOOL_REGISTRY:
        if tool_config["tool"] == tool:
            return tool_config
    raise ValueError(f"Unknown tool {tool!r}.")


def igv_export_tool_slugs() -> list[str]:
    return list(IGV_EXPORT_TOOL_ORDER)


def normalize_tool_slug(tool: str) -> str:
    normalized = TOOL_ALIASES.get(str(tool))
    if normalized is None:
        raise ValueError(
            f"Unknown tool {tool!r}. Expected one of {sorted(TOOL_ALIASES)}."
        )
    return normalized


def region_type_for_tool(tool: str) -> str:
    return REGION_TYPE_BY_EXPORT_SLUG[normalize_tool_slug(tool)]


def _canonical_tool_name(tool: str) -> str:
    return CANONICAL_TOOL_BY_EXPORT_SLUG[normalize_tool_slug(tool)]


def get_methylseg_prep_dir(sample_id: str) -> Path:
    prep_dir = METHYLSEG_RESULTS_DIR / sample_id / "prep"
    if not prep_dir.is_dir():
        raise FileNotFoundError(
            f"MethylSeg prep directory not found for sample_id={sample_id!r}: {prep_dir}"
        )
    return prep_dir


def _get_prep_input_path(sample_id: str, filename: str) -> Path:
    input_path = get_methylseg_prep_dir(sample_id) / filename
    if not input_path.exists():
        raise FileNotFoundError(
            f"Expected prep file {filename!r} for sample_id={sample_id!r}: {input_path}"
        )
    return input_path


def _load_prep_config(sample_id: str) -> dict:
    config_path = _get_prep_input_path(sample_id, "config.yaml")
    with open(config_path) as fh:
        return yaml.safe_load(fh) or {}


def _get_sample_genome(sample_id: str) -> str:
    config = _load_prep_config(sample_id)
    genome = config.get("genome")
    return str(genome) if genome else _infer_genome(sample_id)


def _load_chrom_sizes(genome: str) -> list[tuple[str, int]]:
    chrom_sizes_path = REFERENCE_DATA_DIR / f"{genome}.chrom.sizes"
    if not chrom_sizes_path.exists():
        raise FileNotFoundError(
            f"Chrom sizes file not found for genome={genome!r}: {chrom_sizes_path}"
        )

    chrom_sizes = []
    with open(chrom_sizes_path) as fh:
        for line in fh:
            chrom, size = line.rstrip().split("\t")[:2]
            chrom_sizes.append((chrom, int(size)))
    return chrom_sizes


def _sort_intervals_by_chrom_order(
    df: pd.DataFrame,
    chrom_col: str,
    start_col: str,
    chrom_sizes: list[tuple[str, int]],
) -> pd.DataFrame:
    chrom_order = [chrom for chrom, _ in chrom_sizes]
    out_df = df[df[chrom_col].isin(chrom_order)].copy()
    out_df[chrom_col] = pd.Categorical(out_df[chrom_col], categories=chrom_order, ordered=True)
    out_df = out_df.sort_values([chrom_col, start_col, "end" if "end" in out_df.columns else start_col])
    out_df[chrom_col] = out_df[chrom_col].astype(str)
    return out_df.reset_index(drop=True)


def _load_prep_beta_track(sample_id: str, track: str) -> tuple[pd.DataFrame, str]:
    if track == "wgbs":
        filename = "wgbs.beta"
    elif track == "hm450k":
        filename = "450k.beta"
    else:
        raise ValueError(f"Unsupported methylation track {track!r}.")

    beta_path = _get_prep_input_path(sample_id, filename)
    with open(beta_path) as fh:
        header_fields = fh.readline().rstrip("\n").split("\t")

    has_header = header_fields[:4] == ["chrom", "start", "end", "beta"]
    if has_header:
        df = pd.read_csv(beta_path, sep="\t")
    else:
        n_cols = len(header_fields)
        if n_cols < 4:
            raise ValueError(
                f"Unsupported beta track with fewer than 4 columns: {beta_path}"
            )
        column_names = ["chrom", "start", "end", "beta"] + [
            f"extra_{idx}" for idx in range(n_cols - 4)
        ]
        df = pd.read_csv(beta_path, sep="\t", header=None, names=column_names)
    return _standardize_beta_track_df(df), _get_sample_genome(sample_id)


def _write_bigwig(
    beta_df: pd.DataFrame,
    out_path: Path,
    chrom_sizes: list[tuple[str, int]],
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_df = _sort_intervals_by_chrom_order(beta_df, "chrom", "start", chrom_sizes)
    chrom_size_map = dict(chrom_sizes)
    out_of_bounds = ordered_df["end"] > ordered_df["chrom"].map(chrom_size_map)
    if out_of_bounds.any():
        raise ValueError(
            f"Found {int(out_of_bounds.sum())} intervals extending past chromosome bounds for {out_path}."
        )

    bw = pyBigWig.open(str(out_path), "w")
    try:
        bw.addHeader(chrom_sizes)
        if not ordered_df.empty:
            bw.addEntries(
                ordered_df["chrom"].tolist(),
                ordered_df["start"].astype(int).tolist(),
                ends=ordered_df["end"].astype(int).tolist(),
                values=ordered_df["beta"].astype(float).tolist(),
            )
    finally:
        bw.close()
    return out_path


def export_methylation_bigwig(
    sample_id: str,
    track: str,
    out_path: Path,
) -> Path:
    beta_df, genome = _load_prep_beta_track(sample_id, track)
    chrom_sizes = _load_chrom_sizes(genome)
    return _write_bigwig(beta_df, out_path, chrom_sizes)


def get_methylseg_tumor_samples() -> list[str]:
    return list(METHYLSEG_TUMOR_SAMPLE_IDS)


def build_methylseg_hm450k_overlap_cache(
    *,
    sample_ids: Iterable[str] | None = None,
    threshold_req: int | float = 1,
    force: bool = False,
) -> dict[str, Path]:
    sample_ids = _normalize_overlap_sample_ids(sample_ids)
    threshold_kind, threshold_value = _normalize_overlap_threshold_req(threshold_req)
    cache_dir = _overlap_cache_dir_for(sample_ids, threshold_kind, threshold_value)
    sample_summary_path = cache_dir / "sample_summary.tsv"
    hm450k_metrics_path = cache_dir / "hm450k_region_metrics.tsv"
    wgbs_metrics_path = cache_dir / "wgbs_region_metrics.tsv"
    metadata_path = cache_dir / "metadata.json"

    if (
        not force
        and sample_summary_path.exists()
        and hm450k_metrics_path.exists()
        and wgbs_metrics_path.exists()
        and metadata_path.exists()
    ):
        return {
            "cache_dir": cache_dir,
            "sample_summary": sample_summary_path,
            "hm450k_region_metrics": hm450k_metrics_path,
            "wgbs_region_metrics": wgbs_metrics_path,
            "metadata": metadata_path,
        }

    cache_dir.mkdir(parents=True, exist_ok=True)
    hm450k_frames = []
    wgbs_frames = []
    sample_summary_rows: list[dict[str, object]] = []

    for sample_id in sample_ids:
        sample_result = _build_methylseg_overlap_sample_tables(
            sample_id=sample_id,
            threshold_kind=threshold_kind,
            threshold_value=threshold_value,
        )
        hm450k_frames.append(sample_result["hm450k_df"])
        wgbs_frames.append(sample_result["wgbs_df"])
        sample_summary_rows.append(sample_result["sample_summary"])

    hm450k_df = (
        pd.concat(hm450k_frames, ignore_index=True)
        if hm450k_frames
        else pd.DataFrame(columns=HM450K_REGION_COLUMNS)
    )
    wgbs_df = (
        pd.concat(wgbs_frames, ignore_index=True)
        if wgbs_frames
        else pd.DataFrame(columns=WGBS_REGION_COLUMNS)
    )
    sample_summary_df = pd.DataFrame(sample_summary_rows)

    hm450k_df.to_csv(hm450k_metrics_path, sep="\t", index=False)
    wgbs_df.to_csv(wgbs_metrics_path, sep="\t", index=False)
    sample_summary_df.to_csv(sample_summary_path, sep="\t", index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "cache_version": OVERLAP_CACHE_VERSION,
                "sample_ids": sample_ids,
                "threshold_kind": threshold_kind,
                "threshold_value": threshold_value,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    return {
        "cache_dir": cache_dir,
        "sample_summary": sample_summary_path,
        "hm450k_region_metrics": hm450k_metrics_path,
        "wgbs_region_metrics": wgbs_metrics_path,
        "metadata": metadata_path,
    }


def load_methylseg_hm450k_overlap_sample_summary(
    *,
    sample_ids: list[str] | None = None,
    threshold_req: int | float = 1,
    force: bool = False,
) -> pd.DataFrame:
    paths = build_methylseg_hm450k_overlap_cache(
        sample_ids=sample_ids,
        threshold_req=threshold_req,
        force=force,
    )
    return pd.read_csv(paths["sample_summary"], sep="\t")


def load_methylseg_hm450k_overlap_hm450k_region_metrics(
    *,
    sample_ids: list[str] | None = None,
    threshold_req: int | float = 1,
    force: bool = False,
) -> pd.DataFrame:
    paths = build_methylseg_hm450k_overlap_cache(
        sample_ids=sample_ids,
        threshold_req=threshold_req,
        force=force,
    )
    return pd.read_csv(paths["hm450k_region_metrics"], sep="\t")


def load_methylseg_hm450k_overlap_wgbs_region_metrics(
    *,
    sample_ids: list[str] | None = None,
    threshold_req: int | float = 1,
    force: bool = False,
) -> pd.DataFrame:
    paths = build_methylseg_hm450k_overlap_cache(
        sample_ids=sample_ids,
        threshold_req=threshold_req,
        force=force,
    )
    return pd.read_csv(paths["wgbs_region_metrics"], sep="\t")


def _normalize_overlap_sample_ids(sample_ids: Iterable[str] | None) -> list[str]:
    if sample_ids is None:
        return list(METHYLSEG_TUMOR_SAMPLE_IDS)

    normalized = [str(sample_id) for sample_id in sample_ids]
    if not normalized:
        raise ValueError("sample_ids must contain at least one sample.")
    return normalized


def _normalize_overlap_threshold_req(
    threshold_req: int | float,
) -> tuple[str, int | float]:
    if isinstance(threshold_req, bool):
        raise TypeError("threshold_req must be an int or float, not bool.")

    if isinstance(threshold_req, int):
        if threshold_req < 1:
            raise ValueError("Integer threshold_req values must be >= 1.")
        return "bp", int(threshold_req)

    threshold_float = float(threshold_req)
    if threshold_float <= 0 or threshold_float > 1:
        raise ValueError("Float threshold_req values must be in the interval (0, 1].")
    return "fraction", threshold_float


def _overlap_cache_dir_for(
    sample_ids: list[str],
    threshold_kind: str,
    threshold_value: int | float,
) -> Path:
    sample_key = hashlib.md5(
        "\n".join(sorted(sample_ids)).encode("utf-8")
    ).hexdigest()[:12]
    if threshold_kind == "bp":
        threshold_key = f"bp_{int(threshold_value)}"
    else:
        threshold_key = f"fraction_{float(threshold_value):0.4f}".replace(".", "p")
    return OVERLAP_RESULTS_DIR / (
        f"{sample_key}__{threshold_key}__{OVERLAP_CACHE_VERSION}"
    )


def _build_methylseg_overlap_sample_tables(
    *,
    sample_id: str,
    threshold_kind: str,
    threshold_value: int | float,
) -> dict[str, object]:
    prep_dir = METHYLSEG_RESULTS_DIR / sample_id / "prep"
    hm450k_regions = _load_overlap_region_df(
        METHYLSEG_RESULTS_DIR
        / sample_id
        / "out"
        / "hm450k"
        / "summary_files"
        / "segments_cleaned_PMD.bed"
    )
    wgbs_regions = _load_overlap_region_df(
        METHYLSEG_RESULTS_DIR
        / sample_id
        / "out"
        / "wgbs"
        / "summary_files"
        / "segments_cleaned_PMD.bed"
    )
    hm450k_points = _load_overlap_point_track(prep_dir / "450k_meth_ref.tsv")
    wgbs_points = _load_overlap_point_track(prep_dir / "wgbs_meth_ref.tsv")

    wgbs_by_chrom = _group_overlap_intervals_by_chrom(wgbs_regions)
    hm450k_overlap_rows = []
    for row in hm450k_regions.itertuples(index=False):
        chrom = str(row.chrom)
        q_start = int(row.start)
        q_end = int(row.end)
        q_len = max(q_end - q_start, 1)
        starts, ends, _ = wgbs_by_chrom.get(
            chrom,
            (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            ),
        )
        overlap_stats = _compute_interval_overlap_stats(q_start, q_end, starts, ends)
        max_overlap_bp = overlap_stats["max_overlap_bp"]
        max_overlap_fraction = overlap_stats["covered_bp"] / q_len
        if threshold_kind == "bp":
            is_shared = overlap_stats["covered_bp"] >= int(threshold_value)
        else:
            is_shared = (overlap_stats["covered_bp"] / q_len) >= float(
                threshold_value
            )
        hm450k_overlap_rows.append(
            {
                "max_overlap_bp": max_overlap_bp,
                "max_overlap_fraction_hm450k": float(max_overlap_fraction),
                "n_qualifying_overlaps": (
                    overlap_stats["overlap_count"] if is_shared else 0
                ),
                "is_shared": bool(is_shared),
            }
        )
    hm450k_overlap_df = pd.DataFrame(hm450k_overlap_rows)

    hm450k_signal_df = _summarize_overlap_points_by_region(hm450k_regions, hm450k_points)
    hm450k_wgbs_cpg_df = _summarize_overlap_points_by_region(hm450k_regions, wgbs_points)
    hm450k_metrics_df = hm450k_regions.copy()
    hm450k_metrics_df["sample_id"] = sample_id
    hm450k_metrics_df["category"] = np.where(
        hm450k_overlap_df["is_shared"].to_numpy(dtype=bool),
        "shared",
        "microarray_unique",
    )
    hm450k_metrics_df["platform"] = "hm450k"
    hm450k_metrics_df["region_length_bp"] = (
        hm450k_metrics_df["end"] - hm450k_metrics_df["start"]
    ).astype(np.int64)
    hm450k_metrics_df["max_overlap_bp"] = hm450k_overlap_df["max_overlap_bp"].astype(
        np.int64
    )
    hm450k_metrics_df["max_overlap_fraction_hm450k"] = hm450k_overlap_df[
        "max_overlap_fraction_hm450k"
    ].astype(float)
    hm450k_metrics_df["n_qualifying_overlaps"] = hm450k_overlap_df[
        "n_qualifying_overlaps"
    ].astype(np.int64)
    hm450k_metrics_df["hm450k_probe_count"] = hm450k_signal_df["point_count"].astype(
        np.int64
    )
    hm450k_metrics_df["wgbs_cpg_count"] = hm450k_wgbs_cpg_df["point_count"].astype(
        np.int64
    )
    hm450k_metrics_df["probe_reduction_fraction"] = _compute_probe_reduction_fraction(
        hm450k_metrics_df["hm450k_probe_count"].to_numpy(dtype=float),
        hm450k_metrics_df["wgbs_cpg_count"].to_numpy(dtype=float),
    )
    hm450k_metrics_df["avg_methylation"] = hm450k_signal_df["mean_beta"].astype(float)
    hm450k_metrics_df = hm450k_metrics_df.loc[:, HM450K_REGION_COLUMNS]

    hm450k_by_chrom = _group_overlap_intervals_by_chrom(hm450k_regions)
    wgbs_overlap_rows = []
    for row in wgbs_regions.itertuples(index=False):
        chrom = str(row.chrom)
        q_start = int(row.start)
        q_end = int(row.end)
        q_len = max(q_end - q_start, 1)
        starts, ends, _ = hm450k_by_chrom.get(
            chrom,
            (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            ),
        )
        overlap_stats = _compute_interval_overlap_stats(q_start, q_end, starts, ends)
        max_overlap_bp = overlap_stats["max_overlap_bp"]
        if threshold_kind == "bp":
            is_microarray_missed = overlap_stats["covered_bp"] < int(threshold_value)
        else:
            is_microarray_missed = (
                overlap_stats["covered_bp"] / q_len
            ) < float(threshold_value)
        wgbs_overlap_rows.append(
            {
                "max_overlap_bp": max_overlap_bp,
                "n_qualifying_overlaps": (
                    0 if is_microarray_missed else overlap_stats["overlap_count"]
                ),
                "is_microarray_missed": bool(is_microarray_missed),
            }
        )
    wgbs_overlap_df = pd.DataFrame(wgbs_overlap_rows)

    wgbs_probe_df = _summarize_overlap_points_by_region(wgbs_regions, hm450k_points)
    wgbs_signal_df = _summarize_overlap_points_by_region(wgbs_regions, wgbs_points)
    wgbs_metrics_df = wgbs_regions.copy()
    wgbs_metrics_df["sample_id"] = sample_id
    wgbs_metrics_df["category"] = np.where(
        wgbs_overlap_df["is_microarray_missed"].to_numpy(dtype=bool),
        "microarray_missed",
        "captured_by_microarray",
    )
    wgbs_metrics_df["platform"] = "wgbs"
    wgbs_metrics_df["region_length_bp"] = (
        wgbs_metrics_df["end"] - wgbs_metrics_df["start"]
    ).astype(np.int64)
    wgbs_metrics_df["max_overlap_bp"] = wgbs_overlap_df["max_overlap_bp"].astype(
        np.int64
    )
    wgbs_metrics_df["max_overlap_fraction_hm450k"] = np.nan
    wgbs_metrics_df["n_qualifying_overlaps"] = wgbs_overlap_df[
        "n_qualifying_overlaps"
    ].astype(np.int64)
    wgbs_metrics_df["hm450k_probe_count"] = wgbs_probe_df["point_count"].astype(
        np.int64
    )
    wgbs_metrics_df["wgbs_cpg_count"] = wgbs_signal_df["point_count"].astype(
        np.int64
    )
    wgbs_metrics_df["probe_reduction_fraction"] = _compute_probe_reduction_fraction(
        wgbs_metrics_df["hm450k_probe_count"].to_numpy(dtype=float),
        wgbs_metrics_df["wgbs_cpg_count"].to_numpy(dtype=float),
    )
    wgbs_metrics_df["avg_methylation"] = wgbs_signal_df["mean_beta"].astype(float)
    wgbs_metrics_df = wgbs_metrics_df.loc[
        wgbs_metrics_df["category"] == "microarray_missed",
        WGBS_REGION_COLUMNS,
    ].reset_index(drop=True)

    sample_summary = {
        "sample_id": sample_id,
        "threshold_kind": threshold_kind,
        "threshold_value": threshold_value,
        "hm450k_total_regions": int(len(hm450k_regions)),
        "hm450k_shared_regions": int(
            (hm450k_metrics_df["category"] == "shared").sum()
        ),
        "hm450k_microarray_unique_regions": int(
            (hm450k_metrics_df["category"] == "microarray_unique").sum()
        ),
        "wgbs_total_regions": int(len(wgbs_regions)),
        "wgbs_microarray_missed_regions": int(
            (wgbs_metrics_df["category"] == "microarray_missed").sum()
        ),
        "hm450k_total_bp": int(hm450k_metrics_df["region_length_bp"].sum()),
        "hm450k_shared_bp": int(
            hm450k_metrics_df.loc[
                hm450k_metrics_df["category"] == "shared",
                "region_length_bp",
            ].sum()
        ),
        "hm450k_microarray_unique_bp": int(
            hm450k_metrics_df.loc[
                hm450k_metrics_df["category"] == "microarray_unique",
                "region_length_bp",
            ].sum()
        ),
        "wgbs_total_bp": int((wgbs_regions["end"] - wgbs_regions["start"]).sum()),
        "wgbs_microarray_missed_bp": int(wgbs_metrics_df["region_length_bp"].sum()),
    }

    return {
        "hm450k_df": hm450k_metrics_df,
        "wgbs_df": wgbs_metrics_df,
        "sample_summary": sample_summary,
    }


def _load_overlap_region_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Region BED not found: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "label"],
        usecols=[0, 1, 2],
    )
    df["chrom"] = df["chrom"].astype(str)
    df["start"] = pd.to_numeric(df["start"], errors="raise").astype(np.int64)
    df["end"] = pd.to_numeric(df["end"], errors="raise").astype(np.int64)
    return df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)


def _load_overlap_point_track(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Point track not found: {path}")

    df = pd.read_csv(path, sep="\t")
    df = df.rename(
        columns={
            "CpG_chrm": "chrom",
            "CpG_beg": "start",
            "CpG_end": "end",
        }
    )
    missing_cols = [
        col for col in ["chrom", "start", "end", "beta"] if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"Point track {path} is missing required columns {missing_cols}.")

    df["chrom"] = df["chrom"].astype(str)
    df["start"] = pd.to_numeric(df["start"], errors="raise").astype(np.int64)
    df["end"] = pd.to_numeric(df["end"], errors="raise").astype(np.int64)
    df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    return df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)


def _group_overlap_intervals_by_chrom(
    df: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for chrom, chrom_df in df.groupby("chrom", sort=False):
        starts = chrom_df["start"].to_numpy(dtype=np.int64)
        ends = chrom_df["end"].to_numpy(dtype=np.int64)
        lengths = ends - starts
        grouped[str(chrom)] = (starts, ends, lengths)
    return grouped


def _group_overlap_points_by_chrom(
    df: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, chrom_df in df.groupby("chrom", sort=False):
        grouped[str(chrom)] = (
            chrom_df["start"].to_numpy(dtype=np.int64),
            chrom_df["beta"].to_numpy(dtype=float),
        )
    return grouped


def _candidate_overlap_slice(
    query_start: int,
    query_end: int,
    target_starts: np.ndarray,
    target_ends: np.ndarray,
) -> slice:
    if target_starts.size == 0:
        return slice(0, 0)
    left = int(np.searchsorted(target_ends, query_start, side="right"))
    right = int(np.searchsorted(target_starts, query_end, side="left"))
    return slice(left, right)


def _candidate_overlaps(
    query_start: int,
    query_end: int,
    target_starts: np.ndarray,
    target_ends: np.ndarray,
) -> np.ndarray:
    candidates = _candidate_overlap_slice(
        query_start,
        query_end,
        target_starts,
        target_ends,
    )
    if candidates.start >= candidates.stop:
        return np.array([], dtype=np.int64)
    overlaps = np.minimum(target_ends[candidates], query_end) - np.maximum(
        target_starts[candidates], query_start
    )
    overlaps = overlaps[overlaps > 0]
    return overlaps.astype(np.int64, copy=False)


def _compute_interval_overlap_stats(
    query_start: int,
    query_end: int,
    target_starts: np.ndarray,
    target_ends: np.ndarray,
) -> dict[str, int]:
    overlaps = _candidate_overlaps(query_start, query_end, target_starts, target_ends)
    if overlaps.size == 0:
        return {
            "overlap_count": 0,
            "max_overlap_bp": 0,
            "covered_bp": 0,
        }

    overlap_slice = _candidate_overlap_slice(
        query_start,
        query_end,
        target_starts,
        target_ends,
    )
    clipped_starts = np.maximum(target_starts[overlap_slice], query_start)
    clipped_ends = np.minimum(target_ends[overlap_slice], query_end)
    valid_mask = clipped_ends > clipped_starts
    clipped_starts = clipped_starts[valid_mask]
    clipped_ends = clipped_ends[valid_mask]

    covered_bp = 0
    current_start = int(clipped_starts[0])
    current_end = int(clipped_ends[0])
    for start, end in zip(clipped_starts[1:], clipped_ends[1:]):
        start = int(start)
        end = int(end)
        if start > current_end:
            covered_bp += current_end - current_start
            current_start = start
            current_end = end
        else:
            current_end = max(current_end, end)
    covered_bp += current_end - current_start

    return {
        "overlap_count": int(overlaps.size),
        "max_overlap_bp": int(overlaps.max()),
        "covered_bp": int(covered_bp),
    }


def _summarize_overlap_points_by_region(
    region_df: pd.DataFrame,
    point_df: pd.DataFrame,
) -> pd.DataFrame:
    point_by_chrom = _group_overlap_points_by_chrom(point_df)
    rows = []

    for row in region_df.itertuples(index=False):
        point_starts, betas = point_by_chrom.get(
            str(row.chrom),
            (np.array([], dtype=np.int64), np.array([], dtype=float)),
        )
        if point_starts.size == 0:
            rows.append({"point_count": 0, "mean_beta": np.nan})
            continue

        left = int(np.searchsorted(point_starts, int(row.start), side="left"))
        right = int(np.searchsorted(point_starts, int(row.end), side="left"))
        if left >= right:
            rows.append({"point_count": 0, "mean_beta": np.nan})
            continue

        interval_betas = betas[left:right]
        finite_mask = np.isfinite(interval_betas)
        mean_beta = (
            float(interval_betas[finite_mask].mean()) if finite_mask.any() else np.nan
        )
        rows.append(
            {
                "point_count": int(right - left),
                "mean_beta": mean_beta,
            }
        )

    return pd.DataFrame(rows)


def _compute_probe_reduction_fraction(
    hm450k_probe_count: np.ndarray,
    wgbs_cpg_count: np.ndarray,
) -> np.ndarray:
    hm450k_probe_count = np.asarray(hm450k_probe_count, dtype=float)
    wgbs_cpg_count = np.asarray(wgbs_cpg_count, dtype=float)
    result = np.full(wgbs_cpg_count.shape, np.nan, dtype=float)
    valid_mask = wgbs_cpg_count > 0
    result[valid_mask] = 1.0 - (
        hm450k_probe_count[valid_mask] / wgbs_cpg_count[valid_mask]
    )
    result[valid_mask] = np.clip(result[valid_mask], 0.0, 1.0)
    return result

## Region Comparison Results Helpers

COMPARISON_RESULTS_DIR = RESULTS_DIR / "01_region_calling_analysis" / "comparison"
HMM_TEST_RESULTS_DIR = RESULTS_DIR / "01_region_calling_analysis" / "hmm_tests"


def _iter_comparison_sample_dirs() -> list[Path]:
    sample_dirs = []
    for sample_id in wgbs_cancer_samples():
        sample_dir = COMPARISON_RESULTS_DIR / sample_id
        if sample_dir.is_dir():
            sample_dirs.append(sample_dir)
    return sample_dirs


def _load_aggregate_comparison_csv(filename: str) -> pd.DataFrame:
    frames = []
    for sample_dir in _iter_comparison_sample_dirs():
        csv_path = sample_dir / "aggregate_summaries" / filename
        if not csv_path.exists():
            continue
        frames.append(pd.read_csv(csv_path))

    if not frames:
        raise FileNotFoundError(
            f"No comparison aggregate summary files named {filename!r} were found under {COMPARISON_RESULTS_DIR}."
        )

    return pd.concat(frames, ignore_index=True)


def _load_sample_comparison_csv(sample_id: str, filename: str) -> pd.DataFrame:
    csv_path = COMPARISON_RESULTS_DIR / sample_id / "aggregate_summaries" / filename
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Comparison aggregate summary not found for sample_id={sample_id!r}: {csv_path}"
        )
    return pd.read_csv(csv_path)

def load_tool_regions(sample_id: str, tool: str) -> pd.DataFrame:
    tool_config = _get_tool_config(_canonical_tool_name(tool))
    path_parts = [part.format(sample=sample_id) for part in tool_config["path_parts"]]
    pmd_file = RESULTS_DIR / "01_region_calling_analysis" / Path(*path_parts)
    if not pmd_file.exists():
        raise FileNotFoundError(
            f"Region file not found for sample_id={sample_id!r} and tool={tool!r}: {pmd_file}"
        )

    df = pd.read_csv(
        pmd_file,
        sep="\t",
        header=0 if tool_config["header"] else None,
        names=tool_config["columns"] if not tool_config["header"] else None,
    )
    rename_map = {"chr": "chrom"}
    df = df.rename(columns=rename_map)
    missing_cols = [col for col in ["chrom", "start", "end"] if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Region file for tool={tool!r} is missing expected coordinate columns {missing_cols}: {pmd_file}"
        )

    df = _standardize_region_df(df)
    return df[["chrom", "start", "end"]]


def load_pmds(sample_id: str, tool: str) -> pd.DataFrame:
    return load_tool_regions(sample_id, tool)


def get_run_stats_df():
    return _load_aggregate_comparison_csv("all_run_stats.csv")


def get_hmm_run_times_df() -> pd.DataFrame:
    csv_path = HMM_TEST_RESULTS_DIR / "aggregate_summaries" / "all_hmm_run_times.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"HMM timing summary not found: {csv_path}"
        )
    return pd.read_csv(csv_path)


def get_region_stats_df():
    return _load_aggregate_comparison_csv("all_region_stats.csv")

def get_region_context_df():
    return _load_aggregate_comparison_csv("all_region_context_stats.csv")


def get_pairwise_missing_region_context_df() -> pd.DataFrame:
    return _load_aggregate_comparison_csv("all_pairwise_missing_region_context_stats.csv")


def load_sample_region_context_df(sample_id: str | None = None) -> pd.DataFrame:
    if sample_id is None:
        return get_region_context_df()
    return _load_sample_comparison_csv(sample_id, "all_region_context_stats.csv")


def load_sample_pairwise_missing_region_context_df(sample_id: str | None = None) -> pd.DataFrame:
    if sample_id is None:
        return get_pairwise_missing_region_context_df()
    return _load_sample_comparison_csv(
        sample_id,
        "all_pairwise_missing_region_context_stats.csv",
    )


REGION_CALLING_TOOL_ORDER = [
    "methylseg_wgbs",
    "methylseg_hm450k",
    "methylseekr",
    "dnmtools",
    "dnmtools_array",
    "dnmtools_pmr",
    "mmseekr",
    "methylasso",
]
REGION_CALLING_TOOL_LABELS = {
    "methylseg_wgbs": "MethylSeg WGBS",
    "methylseg_hm450k": "MethylSeg HM450K",
    "methylseekr": "MethylSeekR",
    "dnmtools": "DNMTools WGBS",
    "dnmtools_array": "DNMTools Array",
    "dnmtools_pmr": "DNMTools PMR",
    "mmseekr": "MMSeekR",
    "methylasso": "MethylLasso",
}
REGION_CALLING_POINT_COLORS = {
    label: TOOL_HIGHLIGHT_COLORS[label]
    for label in [
        "MethylSeg WGBS",
        "MethylSeg HM450K",
        "MethylSeekR",
        "DNMTools WGBS",
        "DNMTools Array",
        "DNMTools PMR",
        "MMSeekR",
        "MethylLasso",
    ]
}
PAIRWISE_MISSING_COMPARATOR_LABEL = "Comparator"
PAIRWISE_MISSING_PLOT_COLORS = {
    PAIRWISE_MISSING_COMPARATOR_LABEL: TOOL_HIGHLIGHT_COLORS[
        PAIRWISE_MISSING_COMPARATOR_LABEL
    ],
    "MethylSeg WGBS": TOOL_HIGHLIGHT_COLORS["MethylSeg WGBS"],
    "MethylSeg HM450K": TOOL_HIGHLIGHT_COLORS["MethylSeg HM450K"],
}


def _ordered_region_calling_tool_labels() -> list[str]:
    return [REGION_CALLING_TOOL_LABELS[tool] for tool in REGION_CALLING_TOOL_ORDER]


def normalize_region_calling_tool_name(tool_name: str) -> str:
    return TOOL_ALIASES.get(str(tool_name), str(tool_name))


def add_region_calling_tool_labels(
    df: pd.DataFrame,
    *,
    tool_col: str = "tool",
    normalized_col: str = "tool_normalized",
    label_col: str = "tool_label",
) -> pd.DataFrame:
    out_df = df.copy()
    out_df[normalized_col] = out_df[tool_col].astype(str).map(
        normalize_region_calling_tool_name
    )
    out_df[label_col] = out_df[normalized_col].map(REGION_CALLING_TOOL_LABELS)
    out_df[label_col] = pd.Categorical(
        out_df[label_col],
        categories=_ordered_region_calling_tool_labels(),
        ordered=True,
    )
    return out_df


def _comparison_label_from_id(comparison_id: str) -> str:
    parts = str(comparison_id).split(" vs ", 1)
    if len(parts) != 2:
        return str(comparison_id)

    left, right = parts
    left_norm = normalize_region_calling_tool_name(left)
    right_norm = normalize_region_calling_tool_name(right)
    left_label = REGION_CALLING_TOOL_LABELS.get(left_norm, left)
    right_label = REGION_CALLING_TOOL_LABELS.get(right_norm, right)
    return f"{left_label} vs {right_label}"


def _comparison_sort_key(comparison_id: str) -> tuple[int, int, str]:
    parts = str(comparison_id).split(" vs ", 1)
    if len(parts) != 2:
        return (len(REGION_CALLING_TOOL_ORDER), len(REGION_CALLING_TOOL_ORDER), str(comparison_id))

    left, right = parts
    left_norm = normalize_region_calling_tool_name(left)
    right_norm = normalize_region_calling_tool_name(right)
    order_lookup = {tool: idx for idx, tool in enumerate(REGION_CALLING_TOOL_ORDER)}
    return (
        order_lookup.get(left_norm, len(order_lookup)),
        order_lookup.get(right_norm, len(order_lookup)),
        str(comparison_id),
    )


def _pairwise_missing_methylseg_mask(
    df: pd.DataFrame,
    *,
    tool_col: str = "tool_normalized",
    other_tool_col: str = "other_tool_normalized",
) -> pd.Series:
    methylseg_tools = {"methylseg_wgbs", "methylseg_hm450k"}
    return df.apply(
        lambda row: len(
            {str(row[tool_col]), str(row[other_tool_col])} & methylseg_tools
        )
        == 1,
        axis=1,
    )


def _pairwise_missing_plot_label(
    row: pd.Series,
    *,
    tool_col: str = "tool_normalized",
) -> str:
    tool_name = str(row[tool_col])
    if tool_name in {"methylseg_wgbs", "methylseg_hm450k"}:
        return REGION_CALLING_TOOL_LABELS.get(tool_name, tool_name)
    return PAIRWISE_MISSING_COMPARATOR_LABEL


def _style_region_calling_axis(
    ax: plt.Axes,
    title: str,
    ylabel: str,
    *,
    title_size: int = 18,
    label_size: int = 14,
    tick_size: int = 11,
    x_label_rotation: int = 70,
) -> None:
    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.tick_params(axis="x", labelsize=tick_size)
    ax.tick_params(axis="y", labelsize=tick_size)
    for label in ax.get_xticklabels():
        label.set_rotation(x_label_rotation)
        label.set_ha("right")
        label.set_rotation_mode("anchor")


def plot_summary_table(
    table_df: pd.DataFrame,
    title: str,
    *,
    table_font_size: int = 12,
    row_height: int = 40,
    header_height: int = 44,
    min_height: int = 700,
) -> go.Figure:
    display_df = table_df.copy()
    for column in table_df.columns:
        if pd.api.types.is_numeric_dtype(table_df[column]):
            display_df[column] = table_df[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):,.2f}"
            )
    display_df = display_df.astype(object)
    if "Tool" in display_df.columns:
        methylseg_mask = display_df["Tool"].astype(str).str.startswith("MethylSeg")
        if methylseg_mask.any():
            display_df.loc[methylseg_mask, :] = display_df.loc[methylseg_mask, :].astype(str).apply(
                lambda col: col.map(lambda value: f"<b>{value}</b>")
            )

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[f"<b>{col}</b>" for col in display_df.columns],
                    fill_color=ANNOTATION_COLORS["table_header_fill"],
                    align="left",
                    font=dict(size=table_font_size),
                    height=header_height,
                ),
                cells=dict(
                    values=[display_df[col].tolist() for col in display_df.columns],
                    fill_color=ANNOTATION_COLORS["table_row_fill"],
                    align="left",
                    font=dict(size=table_font_size),
                    height=row_height,
                ),
            )
        ]
    )
    fig.update_layout(
        title=title,
        margin=dict(l=20, r=20, t=70, b=30),
        height=max(min_height, header_height + row_height * len(display_df)),
    )
    return fig


def draw_boxplot(
    ax: plt.Axes,
    data: pd.DataFrame,
    y: str,
    title: str,
    ylabel: str,
    *,
    box_fill_color: str = ANNOTATION_COLORS["table_header_fill"],
    box_width: float = 0.6,
) -> None:
    sns.boxplot(
        data=data,
        x="tool_label",
        y=y,
        order=_ordered_region_calling_tool_labels(),
        color=box_fill_color,
        width=box_width,
        fliersize=0,
        linewidth=1.2,
        ax=ax,
    )
    _style_region_calling_axis(ax, title, ylabel)


def draw_violin_with_points(
    ax: plt.Axes,
    data: pd.DataFrame,
    y: str,
    title: str,
    ylabel: str,
    *,
    violin_fill_color: str = ANNOTATION_COLORS["table_header_fill"],
    violin_width: float = 0.95,
    point_size: int = 18,
    point_alpha: float = 0.65,
    point_jitter: float = 0.12,
    point_offset: float = 0.18,
) -> None:
    ordered_labels = _ordered_region_calling_tool_labels()
    rng = np.random.default_rng(0)

    for position, tool_label in enumerate(ordered_labels):
        values = pd.to_numeric(
            data.loc[data["tool_label"] == tool_label, y],
            errors="coerce",
        ).dropna()
        if values.empty:
            continue

        violin = ax.violinplot(
            values.to_numpy(),
            positions=[position],
            widths=violin_width,
            showmeans=False,
            showextrema=False,
            showmedians=False,
        )
        body = violin["bodies"][0]
        body.set_facecolor(violin_fill_color)
        body.set_edgecolor(NEUTRAL_COLORS["dark_gray"])
        body.set_alpha(0.8)
        body.set_linewidth(1.0)
        verts = body.get_paths()[0].vertices
        verts[:, 0] = np.minimum(verts[:, 0], position)

        x_values = position + point_offset + rng.uniform(
            -point_jitter, point_jitter, size=len(values)
        )
        ax.scatter(
            x_values,
            values.to_numpy(),
            s=point_size,
            color=REGION_CALLING_POINT_COLORS[tool_label],
            alpha=point_alpha,
            linewidths=0,
        )

    ax.set_xticks(range(len(ordered_labels)))
    ax.set_xticklabels(ordered_labels)
    _style_region_calling_axis(ax, title, ylabel)


def plot_avg_methylation_hist(
    sample_id: str | None = None,
    *,
    bins: int = 50,
    col_wrap: int = 3,
) -> sns.FacetGrid:
    plot_df = load_sample_region_context_df(sample_id)
    plot_df = add_region_calling_tool_labels(plot_df, tool_col="tool")
    plot_df = plot_df.dropna(subset=["tool_label", "mean_meth"]).copy()
    col_order = [
        label
        for label in _ordered_region_calling_tool_labels()
        if label in plot_df["tool_label"].astype(str).unique()
    ]

    grid = sns.displot(
        data=plot_df,
        x="mean_meth",
        col="tool_label",
        col_order=col_order,
        col_wrap=col_wrap,
        bins=bins,
        stat="density",
        common_norm=False,
        facet_kws={"sharex": True, "sharey": True},
    )
    grid.set_axis_labels("Average methylation", "Density")
    grid.set_titles("{col_name}")
    title_prefix = sample_id if sample_id is not None else "All samples"
    grid.fig.suptitle(f"{title_prefix}: Average methylation per region by tool", y=1.02)
    return grid


def plot_pairwise_missing(
    sample_id: str | None = None,
    *,
    methyl_seg_only: bool = True,
    bins: int = 50,
    col_wrap: int = 3,
) -> sns.FacetGrid:
    plot_df = load_sample_pairwise_missing_region_context_df(sample_id)
    plot_df = add_region_calling_tool_labels(plot_df, tool_col="tool")
    plot_df = add_region_calling_tool_labels(
        plot_df,
        tool_col="other_tool",
        normalized_col="other_tool_normalized",
        label_col="other_tool_label",
    )

    if methyl_seg_only:
        plot_df = plot_df.loc[_pairwise_missing_methylseg_mask(plot_df)].copy()

    plot_df = plot_df.dropna(subset=["tool_label", "mean_meth", "comparison"]).copy()
    comparison_ids = sorted(
        plot_df["comparison"].astype(str).unique().tolist(),
        key=_comparison_sort_key,
    )
    comparison_order = [_comparison_label_from_id(comparison_id) for comparison_id in comparison_ids]
    plot_df["comparison_label"] = plot_df["comparison"].astype(str).map(
        _comparison_label_from_id
    )
    plot_df["comparison_label"] = pd.Categorical(
        plot_df["comparison_label"],
        categories=comparison_order,
        ordered=True,
    )
    plot_df["pairwise_plot_label"] = plot_df.apply(_pairwise_missing_plot_label, axis=1)

    hue_order = [
        label
        for label in [
            PAIRWISE_MISSING_COMPARATOR_LABEL,
            "MethylSeg WGBS",
            "MethylSeg HM450K",
        ]
        if label in plot_df["pairwise_plot_label"].astype(str).unique()
    ]
    palette = {label: PAIRWISE_MISSING_PLOT_COLORS[label] for label in hue_order}
    grid = sns.displot(
        data=plot_df,
        x="mean_meth",
        col="comparison_label",
        col_order=comparison_order,
        hue="pairwise_plot_label",
        hue_order=hue_order,
        bins=bins,
        stat="density",
        common_norm=False,
        col_wrap=col_wrap,
        facet_kws={"sharex": True, "sharey": True},
        palette=palette,
    )
    grid.set_axis_labels("Average methylation", "Density")
    grid.set_titles("{col_name}")
    if grid._legend is not None:
        grid._legend.set_title("Unique group")
    title_prefix = sample_id if sample_id is not None else "All samples"
    grid.fig.suptitle(
        f"{title_prefix}: Missing-region methylation by pairwise comparison",
        y=1.02,
    )
    return grid


def _load_comparison_matrix(
    sample_id: str,
    filename: str,
) -> pd.DataFrame:
    matrix_path = COMPARISON_RESULTS_DIR / sample_id / filename
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Comparison matrix not found for sample_id={sample_id!r}: {matrix_path}"
        )

    matrix_df = pd.read_csv(matrix_path, index_col=0)
    matrix_df.index = matrix_df.index.astype(str)
    matrix_df.columns = matrix_df.columns.astype(str)
    matrix_df = matrix_df.apply(pd.to_numeric, errors="coerce")
    return matrix_df


def load_jaccard_matrix(sample_id: str) -> pd.DataFrame:
    return _load_comparison_matrix(sample_id, "jaccard_matrix.csv")


def load_pct_cover_matrix(sample_id: str) -> pd.DataFrame:
    return _load_comparison_matrix(sample_id, "pct_cover_matrix.csv")


def export_region_bed(
    sample_id: str,
    tool: str,
    out_path: Path,
) -> Path:
    tool_slug = normalize_tool_slug(tool)
    genome = _get_sample_genome(sample_id)
    chrom_sizes = _load_chrom_sizes(genome)
    regions_df = load_tool_regions(sample_id, tool_slug)
    regions_df = _sort_intervals_by_chrom_order(regions_df, "chrom", "start", chrom_sizes)
    bed_df = regions_df.copy()
    bed_df["name"] = f"{tool_slug}_{region_type_for_tool(tool_slug)}"
    bed_df["score"] = 0
    bed_df["strand"] = "."

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bed_df[["chrom", "start", "end", "name", "score", "strand"]].to_csv(
        out_path,
        sep="\t",
        header=False,
        index=False,
    )
    return out_path


def export_sample_igv_tracks(
    sample_id: str,
    export_root: Path,
    *,
    export_wgbs_bigwig: bool = True,
    export_hm450k_bigwig: bool = True,
    export_region_beds: bool = True,
    region_tools: list[str] | None = None,
) -> dict[str, object]:
    export_root = Path(export_root)
    sample_root = export_root / sample_id
    methylation_paths: list[Path] = []
    region_paths: list[Path] = []

    if export_wgbs_bigwig:
        methylation_paths.append(
            export_methylation_bigwig(
                sample_id,
                "wgbs",
                sample_root / "methylation" / "wgbs.beta.bw",
            )
        )
    if export_hm450k_bigwig:
        methylation_paths.append(
            export_methylation_bigwig(
                sample_id,
                "hm450k",
                sample_root / "methylation" / "hm450k.beta.bw",
            )
        )
    if export_region_beds:
        for tool_slug in region_tools or igv_export_tool_slugs():
            normalized_tool = normalize_tool_slug(tool_slug)
            region_paths.append(
                export_region_bed(
                    sample_id,
                    normalized_tool,
                    sample_root
                    / "regions"
                    / f"{normalized_tool}.{region_type_for_tool(normalized_tool)}.bed",
                )
            )

    return {
        "sample_id": sample_id,
        "genome": _get_sample_genome(sample_id),
        "methylation_paths": methylation_paths,
        "region_paths": region_paths,
    }


## Synthetic Results Helpers

SYNTHETIC_RESULTS_DIR = RESULTS_DIR / "02_synthetic_analysis" / "synthetic_recovery"
SYNTHETIC_METRICS_DIR = SYNTHETIC_RESULTS_DIR / "metrics"
SYNTHETIC_PLOTS_DIR = SYNTHETIC_RESULTS_DIR / "plots"

SYNTHETIC_TOOL_ORDER = [
    "methylseg",
    "methylseg_hm450k",
    "methylseekr",
    "dnmtools",
    "dnmtools_array",
    "dnmtools_pmr",
    "mmseekr",
    "methyl_lasso",
]

SYNTHETIC_TOOL_LABELS = {
    "methylseg": "MethylSeg WGBS",
    "methylseg_hm450k": "MethylSeg HM450K",
    "methylseekr": "MethylSeekR",
    "dnmtools": "DNMTools WGBS",
    "dnmtools_array": "DNMTools Array",
    "dnmtools_pmr": "DNMTools PMR",
    "mmseekr": "MMSeekR",
    "methyl_lasso": "MethylLasso",
}

SYNTHETIC_TOOL_COLORS = {
    tool: TOOL_DISTINCT_COLORS_BY_SLUG[tool]
    for tool in SYNTHETIC_TOOL_ORDER
}
SYNTHETIC_BAR_HIGHLIGHT_COLORS = {
    tool: TOOL_HIGHLIGHT_COLORS_BY_SLUG[tool]
    for tool in SYNTHETIC_TOOL_ORDER
}

SYNTHETIC_TOOL_RANK = {
    tool: rank for rank, tool in enumerate(SYNTHETIC_TOOL_ORDER)
}

SYNTHETIC_EXAMPLE_TRACK_COLORS = {
    **SYNTHETIC_TRACK_COLORS,
}
SYNTHETIC_EXAMPLE_TRACK_TITLES = {
    "source": "Original source",
    "potential": "Potential PMDs",
    "cleaned": "Cleaned background",
    "injected": "Injected synthetic",
}
SYNTHETIC_EXAMPLE_INTERVAL_STYLES = {
    **SYNTHETIC_INTERVAL_STYLES,
}
SYNTHETIC_READ_CHUNK_SIZE = 1_000_000

LOWER_IS_BETTER_PATTERNS = (
    "_mabe",
    "_distance_from_1",
    "avg_false_pmd_beta",
    "mean_false_pmds_called",
)


def _read_synthetic_metrics_table(filename: str) -> pd.DataFrame:
    table_path = SYNTHETIC_METRICS_DIR / filename
    if not table_path.exists():
        raise FileNotFoundError(
            f"Synthetic metrics table not found: {table_path}"
        )
    return pd.read_csv(table_path, sep="\t")


def ordered_synthetic_tool_labels(tool_order: list[str] | None = None) -> list[str]:
    ordered_tools = tool_order or SYNTHETIC_TOOL_ORDER
    return [SYNTHETIC_TOOL_LABELS[tool] for tool in ordered_tools]


def ordered_synthetic_palette(tool_order: list[str] | None = None) -> list[str]:
    ordered_tools = tool_order or SYNTHETIC_TOOL_ORDER
    return [SYNTHETIC_TOOL_COLORS[tool] for tool in ordered_tools]


def synthetic_label_palette(tool_order: list[str] | None = None) -> dict[str, str]:
    ordered_tools = tool_order or SYNTHETIC_TOOL_ORDER
    return {
        SYNTHETIC_TOOL_LABELS[tool]: SYNTHETIC_TOOL_COLORS[tool]
        for tool in ordered_tools
    }


def lower_is_better(metric_name: str) -> bool:
    metric_name = str(metric_name)
    return any(pattern in metric_name for pattern in LOWER_IS_BETTER_PATTERNS)


def apply_synthetic_tool_order(
    df: pd.DataFrame,
    tool_col: str = "tool",
) -> pd.DataFrame:
    out_df = df.copy()
    out_df[tool_col] = pd.Categorical(
        out_df[tool_col].astype(str),
        categories=SYNTHETIC_TOOL_ORDER,
        ordered=True,
    )
    out_df = out_df.dropna(subset=[tool_col]).sort_values(tool_col).reset_index(drop=True)
    return out_df


def add_synthetic_tool_labels(
    df: pd.DataFrame,
    tool_col: str = "tool",
) -> pd.DataFrame:
    out_df = df.copy()
    out_df["tool_label"] = out_df[tool_col].astype(str).map(SYNTHETIC_TOOL_LABELS)
    out_df["tool_label"] = pd.Categorical(
        out_df["tool_label"],
        categories=ordered_synthetic_tool_labels(),
        ordered=True,
    )
    return out_df


def sort_synthetic_tools(
    df: pd.DataFrame,
    metric_col: str | None = None,
    ascending: bool | None = None,
    tool_col: str = "tool",
) -> pd.DataFrame:
    out_df = df.copy()
    out_df[tool_col] = out_df[tool_col].astype(str)
    out_df["_synthetic_tool_rank"] = out_df[tool_col].map(SYNTHETIC_TOOL_RANK)
    out_df = out_df.dropna(subset=["_synthetic_tool_rank"]).copy()
    out_df["_synthetic_tool_rank"] = out_df["_synthetic_tool_rank"].astype(int)

    if metric_col is None:
        sort_cols = ["_synthetic_tool_rank"]
        sort_ascending = [True]
    else:
        if ascending is None:
            ascending = lower_is_better(metric_col)
        sort_cols = [metric_col, "_synthetic_tool_rank"]
        sort_ascending = [ascending, True]

    out_df = out_df.sort_values(
        sort_cols,
        ascending=sort_ascending,
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    return out_df.drop(columns="_synthetic_tool_rank")


def metric_display_name(metric_name: str) -> str:
    label = str(metric_name)
    for suffix in ("_mean", "_median", "_std", "_min", "_max"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
            break
    label = label.replace("_", " ").title()
    label = label.replace("Bp", "BP")
    label = label.replace("Pmd", "PMD")
    label = label.replace("Mabe", "MABE")
    return label


def style_publication_axis(
    ax: plt.Axes,
    title: str,
    ylabel: str,
    *,
    xlabel: str = "",
    title_size: int = 18,
    label_size: int = 14,
    tick_size: int = 11,
    x_label_rotation: int = 70,
) -> None:
    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.tick_params(axis="x", labelsize=tick_size)
    ax.tick_params(axis="y", labelsize=tick_size)
    for label in ax.get_xticklabels():
        label.set_rotation(x_label_rotation)
        label.set_ha("right")
        label.set_rotation_mode("anchor")


def save_publication_figure(
    fig: plt.Figure,
    output_dir: Path,
    filename: str,
    *,
    dpi: int = 300,
    save_pdf: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if save_pdf and output_path.suffix.lower() != ".pdf":
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")


def plot_synthetic_metric_bar(
    per_tool_summary_df: pd.DataFrame,
    metric_col: str,
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    tool_col: str = "tool",
    title_size: int = 18,
    label_size: int = 14,
    tick_size: int = 11,
) -> plt.Axes:
    plot_df = sort_synthetic_tools(
        per_tool_summary_df,
        metric_col=metric_col,
        tool_col=tool_col,
    )
    plot_df = add_synthetic_tool_labels(plot_df, tool_col=tool_col)
    plot_df["metric_value"] = pd.to_numeric(plot_df[metric_col], errors="coerce")
    plot_df = plot_df.dropna(subset=["metric_value"]).reset_index(drop=True)

    if ax is None:
        _, ax = plt.subplots()

    bar_positions = np.arange(len(plot_df))
    bar_colors = [
        SYNTHETIC_BAR_HIGHLIGHT_COLORS[tool_name]
        for tool_name in plot_df[tool_col].astype(str).tolist()
    ]
    ax.bar(
        bar_positions,
        plot_df["metric_value"].to_numpy(),
        color=bar_colors,
        edgecolor=NEUTRAL_COLORS["axis_gray"],
        linewidth=0.8,
    )
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(plot_df["tool_label"].astype(str).tolist())
    style_publication_axis(
        ax,
        title or metric_display_name(metric_col),
        ylabel or metric_display_name(metric_col),
        title_size=title_size,
        label_size=label_size,
        tick_size=tick_size,
    )
    ax.grid(axis="y", alpha=0.25)
    return ax


def plot_false_positive_beta_violin(
    false_beta_df: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    per_tool_summary_df: pd.DataFrame | None = None,
    title: str = "False-Positive PMD Beta by Tool",
    ylabel: str = "False-positive region mean beta",
    tool_col: str = "tool",
    value_col: str = "mean_beta",
    violin_fill_color: str = ANNOTATION_COLORS["table_header_fill"],
    point_size: int = 9,
    point_alpha: float = 0.18,
    point_jitter: float = 0.12,
    point_offset: float = 0.2,
    violin_width: float = 0.95,
    box_width: float = 0.14,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()

    working_df = false_beta_df.copy()
    working_df[tool_col] = working_df[tool_col].astype(str)
    working_df[value_col] = pd.to_numeric(working_df[value_col], errors="coerce")
    working_df = working_df.dropna(subset=[tool_col, value_col]).reset_index(drop=True)

    if per_tool_summary_df is not None and "avg_false_pmd_beta_mean" in per_tool_summary_df.columns:
        ordered_tools = sort_synthetic_tools(
            per_tool_summary_df,
            metric_col="avg_false_pmd_beta_mean",
            tool_col=tool_col,
        )[tool_col].astype(str).tolist()
    else:
        order_df = (
            working_df.groupby(tool_col, as_index=False)[value_col]
            .median()
            .rename(columns={value_col: "value_median"})
        )
        ordered_tools = sort_synthetic_tools(
            order_df,
            metric_col="value_median",
            ascending=True,
            tool_col=tool_col,
        )[tool_col].astype(str).tolist()

    rng = np.random.default_rng(0)
    ordered_labels = [SYNTHETIC_TOOL_LABELS[tool] for tool in ordered_tools]

    for position, tool_name in enumerate(ordered_tools):
        values = (
            working_df.loc[working_df[tool_col] == tool_name, value_col]
            .dropna()
            .to_numpy()
        )
        if values.size == 0:
            continue

        violin = ax.violinplot(
            values,
            positions=[position],
            widths=violin_width,
            showmeans=False,
            showextrema=False,
            showmedians=False,
        )
        body = violin["bodies"][0]
        body.set_facecolor(violin_fill_color)
        body.set_edgecolor(NEUTRAL_COLORS["dark_gray"])
        body.set_alpha(0.8)
        body.set_linewidth(1.0)
        verts = body.get_paths()[0].vertices
        verts[:, 0] = np.minimum(verts[:, 0], position)

        box = ax.boxplot(
            values,
            positions=[position],
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": NEUTRAL_COLORS["black"], "linewidth": 1.4},
            boxprops={
                "facecolor": NEUTRAL_COLORS["white"],
                "edgecolor": NEUTRAL_COLORS["outline_gray"],
                "linewidth": 1.0,
            },
            whiskerprops={"color": NEUTRAL_COLORS["outline_gray"], "linewidth": 1.0},
            capprops={"color": NEUTRAL_COLORS["outline_gray"], "linewidth": 1.0},
        )
        for patch in box["boxes"]:
            patch.set_zorder(3)

        x_values = position + point_offset + rng.uniform(
            -point_jitter, point_jitter, size=len(values)
        )
        ax.scatter(
            x_values,
            values,
            s=point_size,
            color=SYNTHETIC_TOOL_COLORS[tool_name],
            alpha=point_alpha,
            linewidths=0,
            rasterized=True,
        )

    ax.set_xticks(range(len(ordered_labels)))
    ax.set_xticklabels(ordered_labels)
    ax.set_ylim(-0.02, 1.02)
    style_publication_axis(ax, title, ylabel)
    ax.grid(axis="y", alpha=0.25)
    return ax


def get_synthetic_results() -> pd.DataFrame:
    return _read_synthetic_metrics_table("per_sample_tool_metrics.tsv")


def get_synthetic_aggregation_run_summary_df() -> pd.DataFrame | None:
    summary_path = SYNTHETIC_METRICS_DIR / "aggregation_run_summary.tsv"
    if not summary_path.exists():
        return None
    return pd.read_csv(summary_path, sep="\t")

def get_false_positive_beta_df() -> pd.DataFrame:
    return _read_synthetic_metrics_table("false_pmd_beta_by_region.tsv")


def get_per_tool_summary_df() -> pd.DataFrame:
    return _read_synthetic_metrics_table("per_tool_summary.tsv")


def _normalize_chrom_name(chrom: str) -> str:
    chrom = str(chrom)
    return "chr" + chrom.replace("chr", "", 1)


def _synthetic_manifest_path(run_root: str | Path) -> Path:
    run_root = Path(run_root).resolve()
    manifest_path = run_root if run_root.is_file() else run_root / "synthetic_sample_manifest.tsv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Synthetic manifest not found: {manifest_path}")
    return manifest_path


def _synthetic_manifest_base_dir(manifest_path: str | Path) -> Path:
    manifest_path = Path(manifest_path).resolve()
    return manifest_path.parent.parent.parent


def _resolve_synthetic_manifest_entry(
    path_like: str | Path,
    manifest_path: str | Path,
) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (_synthetic_manifest_base_dir(manifest_path) / path).resolve()


def _load_synthetic_manifest_row(
    manifest_path: str | Path,
    sample_id: str,
    *,
    id_col: str = "synthetic_sample_id",
) -> pd.Series:
    manifest_df = pd.read_csv(_synthetic_manifest_path(manifest_path), sep="\t")
    matches = manifest_df.loc[manifest_df[id_col].astype(str) == str(sample_id)].copy()
    if matches.empty:
        raise ValueError(
            f"Sample {sample_id!r} was not found in manifest {_synthetic_manifest_path(manifest_path)}."
        )
    return matches.iloc[0]


def list_synthetic_background_samples(background_root: str | Path) -> list[str]:
    manifest_df = pd.read_csv(_synthetic_manifest_path(background_root), sep="\t")
    return manifest_df["synthetic_sample_id"].astype(str).sort_values().tolist()


def _read_table_for_chrom(
    path: str | Path,
    *,
    chrom: str,
    chrom_col: str,
    read_csv_kwargs: dict,
) -> pd.DataFrame:
    chrom = _normalize_chrom_name(chrom)
    frames = []
    seen_target = False
    for chunk in pd.read_csv(path, chunksize=SYNTHETIC_READ_CHUNK_SIZE, **read_csv_kwargs):
        chunk = chunk.copy()
        chunk_chrom = "chr" + chunk[chrom_col].astype(str).str.removeprefix("chr")
        mask = chunk_chrom == chrom
        filtered = chunk.loc[mask].copy()
        if not filtered.empty:
            seen_target = True
            frames.append(filtered)
        elif seen_target:
            break
    if not frames:
        names = read_csv_kwargs.get("names")
        return pd.DataFrame(columns=names if names is not None else None)
    return pd.concat(frames, ignore_index=True)


def _standardize_wgbs_track_df(df: pd.DataFrame) -> pd.DataFrame:
    out_df = df.copy()
    out_df["chrom"] = out_df["chrom"].astype(str)
    out_df["chrom"] = "chr" + out_df["chrom"].str.replace("^chr", "", regex=True)
    for col in ["start", "end", "meth", "coverage"]:
        out_df[col] = pd.to_numeric(out_df[col], errors="coerce")
    out_df = out_df[out_df["chrom"].isin(CANONICAL_CHROMS)].copy()
    out_df = out_df.dropna(subset=["chrom", "start", "end", "meth", "coverage"])
    out_df["start"] = out_df["start"].astype(np.int64)
    out_df["end"] = out_df["end"].astype(np.int64)
    out_df["meth"] = out_df["meth"].round().astype(np.int64)
    out_df["coverage"] = out_df["coverage"].round().astype(np.int64)
    out_df = out_df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    return out_df[["chrom", "start", "end", "meth", "coverage"]]


def _wgbs_track_to_beta_track(df: pd.DataFrame) -> pd.DataFrame:
    beta_df = df[["chrom", "start", "end"]].copy()
    beta_df["beta"] = df["meth"] / df["coverage"].replace(0, np.nan)
    beta_df["beta"] = beta_df["beta"].fillna(0.0)
    return _standardize_beta_track_df(beta_df)


def _load_synthetic_wgbs_track_for_chrom(
    path: str | Path,
    chrom: str,
) -> pd.DataFrame:
    df = _read_table_for_chrom(
        path,
        chrom=chrom,
        chrom_col="chrom",
        read_csv_kwargs={
            "sep": "\t",
            "header": None,
            "names": ["chrom", "start", "end", "meth", "coverage"],
        },
    )
    if df.empty:
        raise ValueError(f"No rows for {chrom!r} were found in {path}.")
    return _standardize_wgbs_track_df(df)


def _coverage_track_to_beta_track(
    df: pd.DataFrame,
    *,
    chrom_col: str,
    start_col: str,
    end_col: str,
) -> pd.DataFrame:
    beta_df = pd.DataFrame(
        {
            "chrom": df[chrom_col],
            "start": df[start_col],
            "end": df[end_col],
            "beta": df["methylated_reads"] / df["coverage"].replace(0, np.nan),
        }
    )
    beta_df["beta"] = beta_df["beta"].fillna(0.0)
    return _standardize_beta_track_df(beta_df)


def _load_source_beta_track_for_chrom(
    *,
    source_file: str | Path,
    source_kind: str,
    source_genome: str,
    chrom: str,
) -> pd.DataFrame:
    source_path = Path(source_file)
    chrom = _normalize_chrom_name(chrom)

    if source_kind == "beta" or source_path.suffix == ".beta":
        df = _load_beta_sample(source_path, source_genome)
        df = df.loc[
            df["CpG_chrm"].astype(str).map(_normalize_chrom_name) == chrom
        ].copy()
        if df.empty:
            raise ValueError(f"No rows for {chrom!r} were found in {source_path}.")
        return _coverage_track_to_beta_track(
            df,
            chrom_col="CpG_chrm",
            start_col="CpG_start",
            end_col="CpG_end",
        )

    if source_kind != "wgbs_bed_gz" and source_path.suffixes[-2:] != [".bed", ".gz"]:
        raise ValueError(
            f"Unsupported source kind {source_kind!r} for synthetic example track: {source_path}"
        )

    preview = pd.read_csv(
        source_path,
        sep="\t",
        header=None,
        nrows=1,
        compression="gzip",
    )
    n_cols = preview.shape[1]

    if n_cols == 5:
        df = _read_table_for_chrom(
            source_path,
            chrom=chrom,
            chrom_col="CpG_chrm",
            read_csv_kwargs={
                "sep": "\t",
                "header": None,
                "compression": "gzip",
                "names": [
                    "CpG_chrm",
                    "CpG_beg",
                    "CpG_end",
                    "coverage",
                    "meth_percent",
                ],
            },
        )
        if df.empty:
            raise ValueError(f"No rows for {chrom!r} were found in {source_path}.")
        df["methylated_reads"] = (
            pd.to_numeric(df["meth_percent"], errors="coerce")
            / 100.0
            * pd.to_numeric(df["coverage"], errors="coerce")
        )
        df = _standardize_coord_df(
            df,
            required_cols=[
                "CpG_chrm",
                "CpG_beg",
                "CpG_end",
                "methylated_reads",
                "coverage",
            ],
            numeric_cols=["methylated_reads", "coverage"],
        )
        df["methylated_reads"] = df["methylated_reads"].round().astype(np.int64)
        df["coverage"] = df["coverage"].round().astype(np.int64)
        return _coverage_track_to_beta_track(
            df,
            chrom_col="CpG_chrm",
            start_col="CpG_beg",
            end_col="CpG_end",
        )

    if n_cols == 6:
        df = _read_table_for_chrom(
            source_path,
            chrom=chrom,
            chrom_col="CpG_chrm",
            read_csv_kwargs={
                "sep": "\t",
                "header": None,
                "compression": "gzip",
                "names": [
                    "CpG_chrm",
                    "CpG_beg",
                    "CpG_end",
                    "beta",
                    "coverage",
                    "context",
                ],
            },
        )
        if df.empty:
            raise ValueError(f"No rows for {chrom!r} were found in {source_path}.")
        df["methylated_reads"] = (
            pd.to_numeric(df["beta"], errors="coerce")
            * pd.to_numeric(df["coverage"], errors="coerce")
        )
        df = _standardize_coord_df(
            df,
            required_cols=[
                "CpG_chrm",
                "CpG_beg",
                "CpG_end",
                "methylated_reads",
                "coverage",
            ],
            numeric_cols=["methylated_reads", "coverage"],
        )
        df["methylated_reads"] = df["methylated_reads"].round().astype(np.int64)
        df["coverage"] = df["coverage"].round().astype(np.int64)
        return _coverage_track_to_beta_track(
            df,
            chrom_col="CpG_chrm",
            start_col="CpG_beg",
            end_col="CpG_end",
        )

    raise ValueError(f"Unsupported source format with {n_cols} columns in {source_path}.")


def _load_truth_intervals_for_chrom(
    path: str | Path,
    chrom: str,
) -> pd.DataFrame:
    chrom = _normalize_chrom_name(chrom)
    preview = pd.read_csv(path, sep="\t", header=None, nrows=1)
    usecols = [0, 1, 2, 3] if preview.shape[1] >= 4 else [0, 1, 2]
    names = ["chrom", "start", "end", "label"][: len(usecols)]
    df = _read_table_for_chrom(
        path,
        chrom=chrom,
        chrom_col="chrom",
        read_csv_kwargs={
            "sep": "\t",
            "header": None,
            "usecols": usecols,
            "names": names,
        },
    )
    if df.empty:
        return pd.DataFrame(columns=["chrom", "start", "end", "label"])

    df = df.copy()
    df["chrom"] = "chr" + df["chrom"].astype(str).str.replace("^chr", "", regex=True)
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    if "label" not in df.columns:
        df["label"] = ""
    df["label"] = df["label"].fillna("").astype(str)
    df = df[df["chrom"].isin(CANONICAL_CHROMS)].copy()
    df = df.dropna(subset=["chrom", "start", "end"])
    df["start"] = df["start"].astype(np.int64)
    df["end"] = df["end"].astype(np.int64)
    df = df[df["end"] > df["start"]].copy()
    return df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)


def _bin_beta_track_df(beta_df: pd.DataFrame, *, chrom: str, bin_bp: int) -> pd.DataFrame:
    chrom = _normalize_chrom_name(chrom)
    bin_bp = int(bin_bp)
    if bin_bp <= 0:
        raise ValueError(f"bin_bp must be positive, got {bin_bp}.")

    chrom_df = beta_df.loc[beta_df["chrom"] == chrom].copy()
    if chrom_df.empty:
        raise ValueError(f"No beta-track rows remain for {chrom!r}.")

    chrom_df["bin_start"] = (chrom_df["start"] // bin_bp) * bin_bp
    binned_df = (
        chrom_df.groupby("bin_start", as_index=False, sort=True, observed=True)
        .agg(mean_beta=("beta", "mean"), n_cpg=("beta", "size"))
        .reset_index(drop=True)
    )
    binned_df["bin_end"] = binned_df["bin_start"] + bin_bp
    binned_df["bin_midpoint"] = binned_df["bin_start"] + (bin_bp / 2.0)
    binned_df["chrom"] = chrom
    return binned_df[
        ["chrom", "bin_start", "bin_end", "bin_midpoint", "mean_beta", "n_cpg"]
    ]


def _validate_chr_track_alignment(
    cleaned_track_df: pd.DataFrame,
    injected_track_df: pd.DataFrame,
    *,
    chrom: str,
) -> None:
    if len(cleaned_track_df) != len(injected_track_df):
        raise ValueError(
            f"Cleaned and injected tracks have different row counts on {chrom}: "
            f"{len(cleaned_track_df)} != {len(injected_track_df)}."
        )

    cleaned_coords = cleaned_track_df[["chrom", "start", "end", "coverage"]]
    injected_coords = injected_track_df[["chrom", "start", "end", "coverage"]]
    if not cleaned_coords.equals(injected_coords):
        raise ValueError(
            f"Cleaned and injected tracks are not coordinate-aligned on {chrom}."
        )


def _interval_overlay_shapes(
    intervals_df: pd.DataFrame,
    *,
    row: int,
    fillcolor: str,
    opacity: float,
) -> list[dict]:
    if intervals_df.empty:
        return []

    xref = "x" if row == 1 else f"x{row}"
    yref = "y domain" if row == 1 else f"y{row} domain"
    return [
        {
            "type": "rect",
            "xref": xref,
            "yref": yref,
            "x0": int(interval.start),
            "x1": int(interval.end),
            "y0": 0,
            "y1": 1,
            "fillcolor": fillcolor,
            "opacity": opacity,
            "line": {"width": 0},
            "layer": "below",
        }
        for interval in intervals_df.itertuples(index=False)
    ]


def _coordinate_aligned_plot_tracks(track_dfs: dict[str, pd.DataFrame]) -> bool:
    track_items = list(track_dfs.items())
    if len(track_items) < 2:
        return True

    _, reference_df = track_items[0]
    reference_coords = reference_df[["chrom", "start", "end"]]
    for _, track_df in track_items[1:]:
        if len(track_df) != len(reference_df):
            return False
        if not reference_coords.equals(track_df[["chrom", "start", "end"]]):
            return False
    return True


def _compute_plot_keep_idx(
    n_points: int,
    *,
    max_points: int | None,
    seed: int = 42,
) -> tuple[np.ndarray, bool]:
    if max_points is None or int(max_points) <= 0 or n_points <= int(max_points):
        return np.arange(n_points), False

    rng = np.random.default_rng(seed)
    keep_idx = np.sort(rng.choice(n_points, size=int(max_points), replace=False))
    return keep_idx, True


def _subset_beta_track_to_intervals(
    beta_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
) -> pd.DataFrame:
    if beta_df.empty or intervals_df.empty:
        return beta_df.iloc[0:0].copy()

    track_df = beta_df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    subset_frames = []
    for chrom_name, chrom_intervals in intervals_df.groupby("chrom", sort=False):
        chrom_track_df = track_df.loc[track_df["chrom"] == chrom_name].copy()
        if chrom_track_df.empty:
            continue

        starts = chrom_track_df["start"].to_numpy(dtype=np.int64)
        ends = chrom_track_df["end"].to_numpy(dtype=np.int64)
        keep_mask = np.zeros(len(chrom_track_df), dtype=bool)

        for interval in chrom_intervals.itertuples(index=False):
            overlap_mask = (starts < int(interval.end)) & (ends > int(interval.start))
            keep_mask |= overlap_mask

        if keep_mask.any():
            subset_frames.append(chrom_track_df.loc[keep_mask].copy())

    if not subset_frames:
        return track_df.iloc[0:0].copy()
    return pd.concat(subset_frames, ignore_index=True)


def plot_synthetic_background_example(
    *,
    background_root: str | Path,
    injected_root: str | Path,
    sample_id: str,
    injected_sample_id: str | None = None,
    chrom: str = "chr1",
    max_points: int | None = 120_000,
    point_size: float = 2.0,
    point_opacity: float = 0.55,
) -> go.Figure:
    chrom = _normalize_chrom_name(chrom)
    background_manifest = _synthetic_manifest_path(background_root)
    injected_manifest = _synthetic_manifest_path(injected_root)

    background_row = _load_synthetic_manifest_row(background_manifest, sample_id)
    injected_sample_id = injected_sample_id or f"{sample_id}_injected_pmds"
    injected_row = _load_synthetic_manifest_row(injected_manifest, injected_sample_id)

    background_sample_id = str(injected_row.get("background_sample_id", sample_id))
    if background_sample_id != str(sample_id):
        raise ValueError(
            f"Injected sample {injected_sample_id!r} belongs to {background_sample_id!r}, "
            f"not {sample_id!r}."
        )

    source_beta_df = _load_source_beta_track_for_chrom(
        source_file=background_row["source_file"],
        source_kind=str(background_row.get("source_kind", "")),
        source_genome=str(background_row.get("source_genome", "hg38")),
        chrom=chrom,
    )
    cleaned_track_df = _load_synthetic_wgbs_track_for_chrom(
        _resolve_synthetic_manifest_entry(
            background_row["normalized_source_file"],
            background_manifest,
        ),
        chrom,
    )
    injected_track_df = _load_synthetic_wgbs_track_for_chrom(
        _resolve_synthetic_manifest_entry(
            injected_row["synthetic_file"],
            injected_manifest,
        ),
        chrom,
    )
    _validate_chr_track_alignment(cleaned_track_df, injected_track_df, chrom=chrom)

    cleaned_beta_df = _wgbs_track_to_beta_track(cleaned_track_df)
    injected_beta_df = _wgbs_track_to_beta_track(injected_track_df)
    background_truth_df = _load_truth_intervals_for_chrom(
        _resolve_synthetic_manifest_entry(background_row["truth_bed"], background_manifest),
        chrom,
    )
    injected_truth_df = _load_truth_intervals_for_chrom(
        _resolve_synthetic_manifest_entry(injected_row["truth_bed"], injected_manifest),
        chrom,
    )
    raw_tracks = {
        "source": source_beta_df.copy(),
        "potential": source_beta_df.copy(),
        "cleaned": cleaned_beta_df.copy(),
        "injected": injected_beta_df.copy(),
    }
    shared_keep_idx = None
    downsampled = False
    aligned_tracks = {
        "cleaned": cleaned_beta_df.copy(),
        "injected": injected_beta_df.copy(),
    }
    if _coordinate_aligned_plot_tracks(aligned_tracks):
        shared_keep_idx, downsampled = _compute_plot_keep_idx(
            len(next(iter(aligned_tracks.values()))),
            max_points=max_points,
        )

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[
            SYNTHETIC_EXAMPLE_TRACK_TITLES["source"],
            SYNTHETIC_EXAMPLE_TRACK_TITLES["potential"],
            SYNTHETIC_EXAMPLE_TRACK_TITLES["cleaned"],
            SYNTHETIC_EXAMPLE_TRACK_TITLES["injected"],
        ],
    )

    for row_idx, track_name in enumerate(["source", "potential", "cleaned", "injected"], start=1):
        plot_df = raw_tracks[track_name].copy()
        if shared_keep_idx is not None and track_name in {"cleaned", "injected"}:
            keep_idx = shared_keep_idx
            track_downsampled = downsampled
        else:
            keep_idx, track_downsampled = _compute_plot_keep_idx(
                len(plot_df),
                max_points=max_points,
            )
            downsampled = downsampled or track_downsampled
        plot_df = plot_df.iloc[keep_idx].reset_index(drop=True)
        plot_df["point_midpoint"] = (
            plot_df["start"].to_numpy(dtype=np.float64)
            + plot_df["end"].to_numpy(dtype=np.float64)
        ) / 2.0
        fig.add_trace(
            go.Scattergl(
                x=plot_df["point_midpoint"],
                y=plot_df["beta"],
                mode="markers",
                marker={
                    "color": SYNTHETIC_EXAMPLE_TRACK_COLORS[track_name],
                    "size": point_size,
                    "opacity": point_opacity,
                },
                customdata=np.column_stack(
                    [
                        plot_df["start"].to_numpy(dtype=np.int64),
                        plot_df["end"].to_numpy(dtype=np.int64),
                    ]
                ),
                hovertemplate=(
                    "CpG: %{customdata[0]:,}-%{customdata[1]:,}<br>"
                    "Beta: %{y:.3f}<extra></extra>"
                ),
                showlegend=False,
                name=SYNTHETIC_EXAMPLE_TRACK_TITLES[track_name],
            ),
            row=row_idx,
            col=1,
        )

    overlay_shapes = []
    overlay_shapes.extend(
        _interval_overlay_shapes(
            background_truth_df,
            row=2,
            **SYNTHETIC_EXAMPLE_INTERVAL_STYLES["background"],
        )
    )
    overlay_shapes.extend(
        _interval_overlay_shapes(
            injected_truth_df,
            row=4,
            **SYNTHETIC_EXAMPLE_INTERVAL_STYLES["injected"],
        )
    )

    for row_idx in range(1, 5):
        fig.update_yaxes(
            range=[0.0, 1.0],
            fixedrange=True,
            title_text="Mean beta",
            row=row_idx,
            col=1,
        )

    x_min = min(int(df["start"].min()) for df in raw_tracks.values())
    x_max = max(int(df["end"].max()) for df in raw_tracks.values())
    fig.update_xaxes(title_text="Genomic position", row=4, col=1)
    fig.update_xaxes(
        range=[x_min, x_max],
        rangeslider={"visible": True},
        showgrid=True,
        row=4,
        col=1,
    )
    fig.update_layout(
        template="plotly_white",
        height=1050,
        hovermode="x unified",
        shapes=overlay_shapes,
        title=(
            f"Synthetic background example: {sample_id} / {injected_sample_id} "
            f"({chrom}, {'downsampled' if downsampled else 'full'})"
        ),
        margin={"l": 70, "r": 30, "t": 90, "b": 60},
    )
    return fig


## LAD Results Helpers

LAD_RESULTS_DIR = RESULTS_DIR / "04_lad_analysis"
LAD_TABLES_DIR = LAD_RESULTS_DIR / "tables"
LAD_FIGURE_OUTPUT_DIR = OUT_DIR / "05_lad"
LAD_TOOL_ORDER = [
    "methylseg",
    "methylseg_hm450k",
    "methylseekr",
    "dnmtools",
    "dnmtools_array",
    "dnmtools_pmr",
    "mmseekr",
    "methyl_lasso",
]
LAD_TOOL_LABELS = {
    "methylseg": "MethylSeg WGBS",
    "methylseg_hm450k": "MethylSeg HM450K",
    "methylseekr": "MethylSeekR",
    "dnmtools": "DNMTools WGBS",
    "dnmtools_array": "DNMTools Array",
    "dnmtools_pmr": "DNMTools PMR",
    "mmseekr": "MMSeekR",
    "methyl_lasso": "MethylLasso",
}
LAD_METRIC_LABELS = {
    "pct_regions_overlapping_lads": "Fraction of regions overlapping LADs",
    "pct_regions_overlapping_lads_gte_150kb": "Fraction of regions overlapping LADs by at least 150 kb",
    "avg_distance_to_nearest_lad": "Average distance to nearest LAD (bp)",
    "avg_distance_to_nearest_lad_boundary": "Average distance to nearest LAD boundary (bp)",
    "avg_distance_to_nearest_lad_boundary_non_overlapping": "Average distance to nearest LAD boundary for non-overlapping regions (bp)",
    "avg_lads_per_overlapping_pmd": "Average LAD overlaps per overlapping PMD",
    "avg_lad_per_pmd": "Average LAD overlaps per PMD",
    "dist_to_lad": "Distance to LAD (bp)",
    "dist_to_lad_boundary": "Distance to nearest LAD boundary (bp)",
    "total_lad_overlap_bp": "Total LAD overlap (bp)",
    "lad_overlap_fraction": "Fraction of region overlapping LADs",
    "pct_regions_with_boundary_within_150kb_of_lad_boundary": "Fraction of regions with a boundary within 150 kb of a LAD boundary",
    "pct_non_overlapping_regions_within_150kb_of_lad_boundary": "Fraction of non-overlapping regions with a boundary within 150 kb of a LAD boundary",
    "pct_regions_sharing_lad": "Fraction of LAD-overlapping regions sharing a LAD",
    "pmd_coverage_by_lads": "Fraction of region bases covered by LADs",
    "pct_lads_overlapping": "Fraction of LADs overlapping regions",
    "lad_coverage_by_pmds": "Fraction of LAD bases covered by regions",
    "n_regions": "Number of called regions",
    "n_lad_regions": "Number of LAD regions",
}
LAD_PCA_SCORE_METRICS = [
    "avg_distance_to_nearest_lad",
    "avg_distance_to_nearest_lad_boundary",
    "avg_distance_to_nearest_lad_boundary_non_overlapping",
    "avg_lads_per_overlapping_pmd",
    "avg_lad_per_pmd",
    "lad_coverage_by_pmds",
    "pct_lads_overlapping",
    "pct_non_overlapping_regions_within_150kb_of_lad_boundary",
    "pct_regions_overlapping_lads",
    "pct_regions_overlapping_lads_gte_150kb",
    "pct_regions_sharing_lad",
    "pct_regions_with_boundary_within_150kb_of_lad_boundary",
    "pmd_coverage_by_lads",
]
LAD_PCA_COUNT_FEATURES = ["n_regions", "n_lad_regions"]
LAD_PCA_METRIC_DIRECTIONS = {
    "avg_distance_to_nearest_lad": "lower_is_better",
    "avg_distance_to_nearest_lad_boundary": "lower_is_better",
    "avg_distance_to_nearest_lad_boundary_non_overlapping": "lower_is_better",
    "avg_lads_per_overlapping_pmd": "higher_is_better",
    "avg_lad_per_pmd": "higher_is_better",
    "lad_coverage_by_pmds": "higher_is_better",
    "pct_lads_overlapping": "higher_is_better",
    "pct_non_overlapping_regions_within_150kb_of_lad_boundary": "higher_is_better",
    "pct_regions_overlapping_lads": "higher_is_better",
    "pct_regions_overlapping_lads_gte_150kb": "higher_is_better",
    "pct_regions_sharing_lad": "higher_is_better",
    "pct_regions_with_boundary_within_150kb_of_lad_boundary": "higher_is_better",
    "pmd_coverage_by_lads": "higher_is_better",
}


def _read_lad_table(filename: str) -> pd.DataFrame:
    table_path = LAD_TABLES_DIR / filename
    if not table_path.exists():
        raise FileNotFoundError(f"LAD results table not found: {table_path}")
    return pd.read_csv(table_path, sep="\t")


def apply_lad_tool_order(df: pd.DataFrame, tool_col: str = "tool") -> pd.DataFrame:
    out_df = df.copy()
    out_df[tool_col] = pd.Categorical(
        out_df[tool_col].astype(str),
        categories=LAD_TOOL_ORDER,
        ordered=True,
    )
    out_df = out_df.dropna(subset=[tool_col]).sort_values(tool_col).reset_index(drop=True)
    return out_df


def add_lad_tool_labels(df: pd.DataFrame, tool_col: str = "tool") -> pd.DataFrame:
    out_df = df.copy()
    out_df["tool_label"] = out_df[tool_col].astype(str).map(LAD_TOOL_LABELS)
    out_df["tool_label"] = pd.Categorical(
        out_df["tool_label"],
        categories=[LAD_TOOL_LABELS[tool] for tool in LAD_TOOL_ORDER],
        ordered=True,
    )
    return out_df


def lad_label_palette(tool_order: list[str] | None = None) -> dict[str, str]:
    ordered_tools = tool_order or LAD_TOOL_ORDER
    return {LAD_TOOL_LABELS[tool]: LAD_TOOL_COLORS[tool] for tool in ordered_tools}


def get_lad_association_metrics_df() -> pd.DataFrame:
    return _read_lad_table("lad_association_metrics.tsv")


def get_lad_combined_summary_df() -> pd.DataFrame:
    return _read_lad_table("lad_combined_summary.tsv")


def get_lad_null_summary_df() -> pd.DataFrame:
    return _read_lad_table("lad_null_summary.tsv")


def get_lad_unique_lad_exports_df() -> pd.DataFrame:
    return _read_lad_table("lad_unique_lad_exports.tsv")


def get_lad_profile_outputs_df() -> pd.DataFrame:
    return _read_lad_table("lad_profile_outputs.tsv")


def get_lad_unique_lad_summary_df() -> pd.DataFrame:
    return _read_lad_table("lad_unique_lad_summary.tsv")


def get_lad_sample_genomes_df() -> pd.DataFrame:
    return _read_lad_table("lad_sample_genomes.tsv")


def get_lad_reference_summary_df() -> pd.DataFrame:
    return _read_lad_table("lad_reference_summary.tsv")


def get_lad_boundary_distance_details_df() -> pd.DataFrame:
    return _read_lad_table("lad_boundary_distance_details.tsv")


def get_lad_region_overlap_details_df() -> pd.DataFrame:
    return _read_lad_table("lad_region_overlap_details.tsv")


def get_lad_metric_feature_matrix_df(
    metrics_df: pd.DataFrame | None = None,
    *,
    region_type: str | None = None,
    score_metrics: list[str] | None = None,
    count_features: list[str] | None = None,
) -> pd.DataFrame:
    source_df = get_lad_association_metrics_df() if metrics_df is None else metrics_df.copy()
    if source_df.empty:
        raise ValueError("No LAD association metrics are available for PCA.")

    source_df = add_lad_tool_labels(apply_lad_tool_order(source_df))
    if region_type is not None:
        source_df = source_df.loc[
            source_df["region_type"].astype(str) == str(region_type)
        ].copy()
    if source_df.empty:
        raise ValueError(f"No LAD association metrics remain after region_type={region_type!r} filtering.")

    selected_score_metrics = list(
        LAD_PCA_SCORE_METRICS if score_metrics is None else score_metrics
    )
    selected_count_features = list(
        LAD_PCA_COUNT_FEATURES if count_features is None else count_features
    )
    available_metrics = set(source_df["metric"].astype(str))
    missing_metrics = [
        metric_name for metric_name in selected_score_metrics if metric_name not in available_metrics
    ]
    if missing_metrics:
        raise ValueError(
            "Missing LAD metrics required for PCA: " + ", ".join(missing_metrics)
        )

    missing_count_features = [
        feature_name
        for feature_name in selected_count_features
        if feature_name not in source_df.columns
    ]
    if missing_count_features:
        raise ValueError(
            "Missing LAD count features required for PCA: "
            + ", ".join(missing_count_features)
        )

    id_cols = [
        "sample",
        "sample_id",
        "tool",
        "tool_label",
        "platform",
        "region_type",
    ]
    metrics_subset_df = source_df.loc[
        source_df["metric"].astype(str).isin(selected_score_metrics),
        id_cols + ["metric", "score", *selected_count_features],
    ].copy()

    count_nunique_df = metrics_subset_df.groupby(
        id_cols,
        observed=True,
        sort=False,
    )[selected_count_features].nunique(dropna=False)
    inconsistent_counts = [
        f"{group_key}::{feature_name}"
        for group_key, row in count_nunique_df.iterrows()
        for feature_name, unique_count in row.items()
        if unique_count > 1
    ]
    if inconsistent_counts:
        raise ValueError(
            "Inconsistent LAD count features across metrics for: "
            + ", ".join(inconsistent_counts[:5])
        )

    wide_scores_df = (
        metrics_subset_df.pivot_table(
            index=id_cols,
            columns="metric",
            values="score",
            aggfunc="first",
            observed=True,
        )
        .reset_index()
    )
    wide_scores_df.columns.name = None

    counts_df = (
        metrics_subset_df[id_cols + selected_count_features]
        .drop_duplicates(subset=id_cols)
        .reset_index(drop=True)
    )
    feature_matrix_df = wide_scores_df.merge(
        counts_df,
        on=id_cols,
        how="left",
        validate="one_to_one",
    )

    missing_feature_columns = [
        feature_name
        for feature_name in [*selected_score_metrics, *selected_count_features]
        if feature_name not in feature_matrix_df.columns
    ]
    if missing_feature_columns:
        raise ValueError(
            "LAD PCA feature matrix is missing expected columns: "
            + ", ".join(missing_feature_columns)
        )

    ordered_cols = id_cols + selected_score_metrics + selected_count_features
    feature_matrix_df = feature_matrix_df.loc[:, ordered_cols].copy()
    for feature_name in selected_score_metrics + selected_count_features:
        feature_matrix_df[feature_name] = pd.to_numeric(
            feature_matrix_df[feature_name],
            errors="coerce",
        )

    feature_matrix_df = add_lad_tool_labels(
        apply_lad_tool_order(feature_matrix_df)
    )
    feature_matrix_df = feature_matrix_df.sort_values(
        ["tool", "sample"],
        kind="stable",
    ).reset_index(drop=True)
    return feature_matrix_df
