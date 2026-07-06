from __future__ import annotations

from io import StringIO
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
import seaborn as sns
import yaml


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
REFERENCE_DATA_DIR = DATA_DIR / "reference_data"
METHYLSEG_RESULTS_DIR = RESULTS_DIR / "01_region_calling_analysis" / "methylseg"
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
        "tool_label": "DNMTools",
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
        "SRR26107673",
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

## Region Comparison Results Helpers

COMPARISON_RESULTS_DIR = RESULTS_DIR / "01_region_calling_analysis" / "comparison"


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

def get_region_stats_df():
    return _load_aggregate_comparison_csv("all_region_stats.csv")

def get_region_context_df():
    return _load_aggregate_comparison_csv("all_region_context_stats.csv")

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
    "dnmtools": "DNMTools",
    "dnmtools_array": "DNMTools Array",
    "dnmtools_pmr": "DNMTools PMR",
    "mmseekr": "MMSeekR",
    "methyl_lasso": "MethylLasso",
}

SYNTHETIC_TOOL_COLORS = {
    "methylseg": "#255f85",
    "methylseg_hm450k": "#4c956c",
    "methylseekr": "#c6ac4d",
    "dnmtools": "#c97c5d",
    "dnmtools_array": "#d65f5f",
    "dnmtools_pmr": "#8f5a9c",
    "mmseekr": "#6c757d",
    "methyl_lasso": "#3f8f97",
}

SYNTHETIC_TOOL_RANK = {
    tool: rank for rank, tool in enumerate(SYNTHETIC_TOOL_ORDER)
}

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
        SYNTHETIC_TOOL_COLORS[tool_name]
        for tool_name in plot_df[tool_col].astype(str).tolist()
    ]
    ax.bar(
        bar_positions,
        plot_df["metric_value"].to_numpy(),
        color=bar_colors,
        edgecolor="#4f4f4f",
        linewidth=0.8,
    )
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(plot_df["tool_label"].astype(str).tolist())
    style_publication_axis(
        ax,
        title or metric_display_name(metric_col),
        ylabel or metric_display_name(metric_col),
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
    violin_fill_color: str = "#d9d9d9",
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
        body.set_edgecolor("#5f5f5f")
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
            medianprops={"color": "#202020", "linewidth": 1.4},
            boxprops={"facecolor": "white", "edgecolor": "#404040", "linewidth": 1.0},
            whiskerprops={"color": "#404040", "linewidth": 1.0},
            capprops={"color": "#404040", "linewidth": 1.0},
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


## LAD Results Helpers

LAD_RESULTS_DIR = RESULTS_DIR / "04_lad_analysis"
LAD_TABLES_DIR = LAD_RESULTS_DIR / "tables"
LAD_FIGURE_OUTPUT_DIR = RESULTS_DIR / "figures" / "05_lad"
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
    "dnmtools": "DNMTools",
    "dnmtools_array": "DNMTools Array",
    "dnmtools_pmr": "DNMTools PMR",
    "mmseekr": "MMSeekR",
    "methyl_lasso": "MethylLasso",
}
LAD_TOOL_COLORS = {
    "methylseg": "#0b5394",
    "methylseg_hm450k": "#3d85c6",
    "methylseekr": "#9aa0a6",
    "dnmtools": "#9aa0a6",
    "dnmtools_array": "#9aa0a6",
    "dnmtools_pmr": "#9aa0a6",
    "mmseekr": "#9aa0a6",
    "methyl_lasso": "#9aa0a6",
}
LAD_METRIC_LABELS = {
    "pct_regions_overlapping_lads": "Fraction of regions overlapping LADs",
    "pct_regions_overlapping_lads_gte_150kb": "Fraction of regions overlapping LADs by at least 150 kb",
    "avg_distance_to_nearest_lad": "Average distance to nearest LAD (bp)",
    "avg_distance_to_nearest_lad_boundary": "Average distance to nearest LAD boundary (bp)",
    "avg_distance_to_nearest_lad_boundary_non_overlapping": "Average distance to nearest LAD boundary for non-overlapping regions (bp)",
    "pct_regions_with_boundary_within_150kb_of_lad_boundary": "Fraction of regions with a boundary within 150 kb of a LAD boundary",
    "pct_non_overlapping_regions_within_150kb_of_lad_boundary": "Fraction of non-overlapping regions with a boundary within 150 kb of a LAD boundary",
    "pct_regions_sharing_lad": "Fraction of LAD-overlapping regions sharing a LAD",
    "pmd_coverage_by_lads": "Fraction of region bases covered by LADs",
    "pct_lads_overlapping": "Fraction of LADs overlapping regions",
    "lad_coverage_by_pmds": "Fraction of LAD bases covered by regions",
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
