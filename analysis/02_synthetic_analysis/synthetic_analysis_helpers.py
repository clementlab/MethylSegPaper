from __future__ import annotations

import copy
import gzip
import hashlib
import importlib
import multiprocessing as mp
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from pybedtools import BedTool

REPO_ROOT = Path(__file__).resolve().parent
while not (REPO_ROOT / "repo_paths.py").exists():
    if REPO_ROOT.parent == REPO_ROOT:
        raise ModuleNotFoundError(
            "Could not locate repo_paths.py from synthetic_analysis_helpers.py"
        )
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repo_paths import REFERENCE_DATA_DIR
AUTOSOMES = [f"chr{i}" for i in range(1, 23)]
WG_COLUMNS = ["chrom", "start", "end", "meth", "coverage"]
TRUTH_COLUMNS = ["chrom", "start", "end", "pmd_id"]

HEALTHY_PMD_TOOL_REGISTRY = [
    {
        "tool": "methylseg",
        "tool_label": "MethylSeg WGBS",
        "parser_family": "methylseg",
        "tool_family": "methylseg",
        "platform": "wgbs",
        "region_type": "PMR",
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
        "region_type": "PMR",
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
        "columns": ["chrom", "start", "end"],
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
        "columns": ["chrom", "start", "end"],
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

DEFAULT_PMD_CONFIG = {
    "global_seed": 42,
    "pmds_per_chrom_min": 5,
    "pmds_per_chrom_max": 10,
    "min_len_bp": 150_000,
    "max_len_bp": 20_000_000,
    "length_sampling": "log_normal",
    "length_mean_bp": 700_000,
    "length_std_bp": 1_000_000,
    "end_buffer_frac": 0.10,
    "center_buffer_frac": 0.10,
    "min_gap_bp": 1_000_000,
    "placement_attempts_per_chrom": 10_000,
    "biological_noise": {
        "enabled": True,
        "min_region_len_bp": 200_000,
        "spike_len_min_bp": 2_000,
        "spike_len_max_bp": 20_000,
        "spikes_per_region_min": 1,
        "spikes_per_region_max": 4,
        "high_spike_fraction": 0.5,
    },
    "manual_overrides": {
        # "all_samples": {"chr1": [(30_000_000, 30_500_000)]},
        # "synthetic_sample_id": {"chr2": [(40_000_000, 40_400_000)]},
    },
}

REFERENCE_TRACKS_DIR = REFERENCE_DATA_DIR
GENOME_SIZES_PATHS = {
    "hg19": REFERENCE_TRACKS_DIR / "hg19.chrom.sizes",
    "hg38": REFERENCE_TRACKS_DIR / "hg38.chrom.sizes",
}


def load_default_pmd_config():
    return copy.deepcopy(DEFAULT_PMD_CONFIG)


def stable_sample_seed(global_seed, synthetic_sample_id):
    digest = hashlib.sha256(
        f"{global_seed}:{synthetic_sample_id}".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16)


def stable_name_seed(name):
    digest = hashlib.sha256(str(name).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def resolve_analysis_path(path_like, root: Path = SYNTHETIC_ROOT):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def to_manifest_path(path_like, root: Path = SYNTHETIC_ROOT):
    path = Path(path_like).resolve()
    try:
        return str(path.relative_to(root.resolve()))
    except ValueError:
        return str(path)


def finalize_wgbs_table(df):
    wgbs_df = df.copy()
    wgbs_df["chrom"] = wgbs_df["chrom"].astype(str)
    for col in ["start", "end", "meth", "coverage"]:
        wgbs_df[col] = pd.to_numeric(wgbs_df[col], errors="coerce")
    wgbs_df = wgbs_df.dropna(
        subset=["chrom", "start", "end", "meth", "coverage"]
    ).copy()
    for col in ["start", "end", "meth", "coverage"]:
        wgbs_df[col] = wgbs_df[col].astype(np.int64)
    wgbs_df = wgbs_df.loc[wgbs_df["end"] > wgbs_df["start"]].copy()
    wgbs_df = wgbs_df.loc[wgbs_df["coverage"] >= 0].copy()
    wgbs_df["meth"] = np.clip(wgbs_df["meth"], 0, wgbs_df["coverage"])
    wgbs_df = wgbs_df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    return wgbs_df


def ensure_interval_df(df, autosomes=AUTOSOMES):
    if df is None:
        return pd.DataFrame(columns=["chrom", "start", "end"])
    if df.empty:
        return df.copy()
    cleaned_df = df.copy()
    cleaned_df["chrom"] = cleaned_df["chrom"].astype(str)
    cleaned_df["start"] = pd.to_numeric(cleaned_df["start"], errors="coerce")
    cleaned_df["end"] = pd.to_numeric(cleaned_df["end"], errors="coerce")
    cleaned_df = cleaned_df.dropna(subset=["chrom", "start", "end"]).copy()
    cleaned_df["start"] = cleaned_df["start"].astype(np.int64)
    cleaned_df["end"] = cleaned_df["end"].astype(np.int64)
    cleaned_df = cleaned_df.loc[cleaned_df["chrom"].isin(autosomes)].copy()
    cleaned_df = cleaned_df.loc[cleaned_df["end"] > cleaned_df["start"]].copy()
    return cleaned_df.reset_index(drop=True)


def add_beta_column(df):
    out_df = df.copy()
    out_df["beta"] = out_df["meth"] / out_df["coverage"].replace(0, np.nan)
    out_df["beta"] = out_df["beta"].fillna(0.0)
    return out_df


def load_wgbs_table(path_like, root: Path = SYNTHETIC_ROOT):
    path = resolve_analysis_path(path_like, root=root)
    df = pd.read_csv(path, sep="\t", header=None, names=WG_COLUMNS)
    return finalize_wgbs_table(df)


def load_truth_table(path_like, root: Path = SYNTHETIC_ROOT):
    path = resolve_analysis_path(path_like, root=root)
    truth_df = pd.read_csv(path, sep="\t", header=None, names=TRUTH_COLUMNS)
    truth_df = ensure_interval_df(truth_df[["chrom", "start", "end"]])
    truth_df["pmd_id"] = (
        pd.read_csv(path, sep="\t", header=None).iloc[:, 3].astype(str).to_numpy()
    )
    return truth_df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)


def normalize_source_record(record, normalized_source_dir: Path, overwrite=False):
    source_path = Path(record["source_file"])
    source_kind = record["source_kind"]
    source_genome = record.get("source_genome")
    sample_id = record["synthetic_sample_id"]
    normalized_path = normalized_source_dir / f"{sample_id}.normalized.tsv"
    if normalized_path.exists() and not overwrite:
        return normalized_path, load_wgbs_table(normalized_path, root=Path.cwd())

    if source_kind == "beta":
        if not source_genome:
            raise ValueError(
                f"{sample_id} is a beta source but source_genome is missing."
            )
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                "wgbstools",
                "view",
                "--genome",
                str(source_genome),
                str(source_path),
                "-o",
                str(normalized_path),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"wgbstools normalization failed for {sample_id} with code {result.returncode}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        df = pd.read_csv(normalized_path, sep="\t", header=None, names=WG_COLUMNS)
    elif source_kind == "wgbs_bed_gz":
        rows = []
        with gzip.open(source_path, "rt") as handle:
            for line in handle:
                chrom, start, end, coverage, meth_percent = line.rstrip("\n").split(
                    "\t"
                )[:5]
                coverage = int(coverage)
                meth = int(round(coverage * (float(meth_percent) / 100.0)))
                rows.append((chrom, int(start), int(end), meth, coverage))
        df = pd.DataFrame(rows, columns=WG_COLUMNS)
    else:
        raise ValueError(
            f"Unsupported source kind for local normalization: {source_kind}"
        )

    df = finalize_wgbs_table(df)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(normalized_path, sep="\t", header=False, index=False)
    return normalized_path, df


def write_comparator_configs(records, config_dir: Path):
    import yaml

    config_dir.mkdir(parents=True, exist_ok=True)
    config_pairs = []
    for record in records:
        config_path = config_dir / f"{record['synthetic_sample_id']}.yaml"
        payload = {
            "sample": record["synthetic_sample_id"],
            "meth_file": str(record["source_file"]),
            "genome": record["source_genome"],
        }
        with open(config_path, "w") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        config_pairs.append((record, config_path))
    return config_pairs


def get_shared_prep_path(healthy_pmd_root: Path, sample, filename):
    return healthy_pmd_root / "comparison" / sample / "shared_prep" / filename


def load_tool_regions(sample, tool_name, healthy_pmd_root: Path, tools_info_df):
    tool_row = tools_info_df.loc[tools_info_df["tool"] == tool_name].iloc[0]
    regions_bed_path = healthy_pmd_root / "/".join(tool_row["path_parts"]).replace(
        "{sample}", sample
    )
    if not regions_bed_path.exists():
        return pd.DataFrame(columns=["chrom", "start", "end", "tool"])
    regions_df = pd.read_csv(
        regions_bed_path,
        sep="\t",
        header=0 if tool_row["header"] else None,
        index_col=False,
        names=tool_row["columns"],
    )
    regions_df = ensure_interval_df(regions_df[["chrom", "start", "end"]])
    if regions_df.empty:
        return pd.DataFrame(columns=["chrom", "start", "end", "tool"])
    regions_df["tool"] = tool_name
    return regions_df


def load_all_pmd_regions(sample, healthy_pmd_root: Path, tools_info_df):
    tool_names = tools_info_df["tool"].tolist()
    region_dfs = [
        load_tool_regions(sample, tool_name, healthy_pmd_root, tools_info_df)
        for tool_name in tool_names
    ]
    non_empty = [df for df in region_dfs if not df.empty]
    if not non_empty:
        return (
            pd.DataFrame(columns=["chrom", "start", "end", "tool"]),
            pd.DataFrame(columns=["tool", "n_regions"]),
        )
    all_regions_df = pd.concat(non_empty, ignore_index=True)
    tool_summary_df = (
        all_regions_df.groupby("tool", as_index=False)
        .size()
        .rename(columns={"size": "n_regions"})
        .sort_values(["n_regions", "tool"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return all_regions_df, tool_summary_df


def annotate_union_regions_with_tools(candidate_union_pmds_df, tool_pmds_df):
    if candidate_union_pmds_df.empty:
        return pd.DataFrame(
            columns=[
                "chrom",
                "start",
                "end",
                "region_id",
                "region_length_bp",
                "contributing_tools",
                "n_contributing_tools",
            ]
        )
    annotated_df = candidate_union_pmds_df.copy()
    annotated_df["contributing_tools"] = ""
    annotated_df["n_contributing_tools"] = 0
    if tool_pmds_df.empty:
        return annotated_df

    for chrom, chrom_regions_df in annotated_df.groupby("chrom", sort=False):
        chrom_tool_df = tool_pmds_df.loc[tool_pmds_df["chrom"] == chrom].sort_values(
            ["start", "end"]
        )
        if chrom_tool_df.empty:
            continue
        tool_starts = chrom_tool_df["start"].to_numpy(dtype=np.int64)
        tool_ends = chrom_tool_df["end"].to_numpy(dtype=np.int64)
        tool_names = chrom_tool_df["tool"].to_numpy(dtype=object)
        for row in chrom_regions_df.itertuples(index=False):
            left = int(np.searchsorted(tool_ends, int(row.start), side="right"))
            right = int(np.searchsorted(tool_starts, int(row.end), side="left"))
            if right <= left:
                continue
            contributing_tools = sorted(set(tool_names[left:right]))
            annotated_df.loc[
                annotated_df["region_id"] == row.region_id,
                "contributing_tools",
            ] = ", ".join(contributing_tools)
            annotated_df.loc[
                annotated_df["region_id"] == row.region_id,
                "n_contributing_tools",
            ] = len(contributing_tools)
    return annotated_df


def merge_union_regions(regions_df):
    if regions_df.empty:
        return pd.DataFrame(
            columns=["chrom", "start", "end", "region_id", "region_length_bp"]
        )
    merged_rows = []
    for chrom, chrom_df in regions_df.sort_values(["chrom", "start", "end"]).groupby(
        "chrom", sort=False
    ):
        current_start = None
        current_end = None
        for row in chrom_df.itertuples(index=False):
            start = int(row.start)
            end = int(row.end)
            if current_start is None:
                current_start = start
                current_end = end
                continue
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged_rows.append((chrom, current_start, current_end))
                current_start = start
                current_end = end
        if current_start is not None:
            merged_rows.append((chrom, current_start, current_end))
    merged_df = pd.DataFrame(merged_rows, columns=["chrom", "start", "end"])
    merged_df = ensure_interval_df(merged_df)
    merged_df["region_id"] = np.arange(len(merged_df))
    merged_df["region_length_bp"] = merged_df["end"] - merged_df["start"]
    return merged_df


def load_sample_beta_from_shared_prep(sample, healthy_pmd_root: Path):
    beta_path = get_shared_prep_path(healthy_pmd_root, sample, "wgbs.beta")
    meth_df = pd.read_csv(beta_path, sep="\t", header=0)
    meth_df = meth_df.copy()
    meth_df["beta"] = pd.to_numeric(meth_df["beta"], errors="coerce")
    meth_df = meth_df.dropna(subset=["beta"]).copy()
    meth_df = ensure_interval_df(meth_df[["chrom", "start", "end", "beta"]])
    meth_df["beta"] = pd.to_numeric(meth_df["beta"], errors="coerce")
    return meth_df.dropna(subset=["beta"]).reset_index(drop=True)


def load_sample_wgbs_from_shared_prep(sample, healthy_pmd_root: Path):
    wgbs_path = get_shared_prep_path(healthy_pmd_root, sample, "wgbs.tsv")
    wgbs_df = pd.read_csv(wgbs_path, sep="\t", header=None, names=WG_COLUMNS)
    wgbs_df = finalize_wgbs_table(wgbs_df)
    return add_beta_column(wgbs_df)


def compute_region_mean_beta(regions_df, meth_df):
    if regions_df.empty or meth_df.empty:
        return pd.DataFrame(columns=["region_id", "mean_beta"])
    mean_rows = []
    for chrom, chrom_regions_df in regions_df.groupby("chrom", sort=False):
        chrom_meth_df = meth_df.loc[meth_df["chrom"] == chrom].sort_values(
            ["start", "end"]
        )
        if chrom_meth_df.empty:
            continue
        starts = chrom_meth_df["start"].to_numpy(dtype=np.int64)
        ends = chrom_meth_df["end"].to_numpy(dtype=np.int64)
        betas = chrom_meth_df["beta"].to_numpy(dtype=float)
        for row in chrom_regions_df.itertuples(index=False):
            left = int(np.searchsorted(ends, int(row.start), side="right"))
            right = int(np.searchsorted(starts, int(row.end), side="left"))
            if right <= left:
                continue
            mean_rows.append((int(row.region_id), float(np.mean(betas[left:right]))))
    return pd.DataFrame(mean_rows, columns=["region_id", "mean_beta"])


def filter_candidate_pmds(candidate_pmds_df, meth_df, min_beta=0.2, max_beta=0.7):
    scored_df = candidate_pmds_df.merge(
        compute_region_mean_beta(candidate_pmds_df, meth_df),
        on="region_id",
        how="left",
    )
    scored_df = scored_df.dropna(subset=["mean_beta"]).copy().reset_index(drop=True)
    in_range = scored_df["mean_beta"].between(min_beta, max_beta, inclusive="both")
    pmds_to_remove_df = scored_df.loc[in_range].copy().reset_index(drop=True)
    kept_pmds_df = scored_df.loc[~in_range].copy().reset_index(drop=True)
    pmds_to_remove_df["region_id"] = np.arange(len(pmds_to_remove_df))
    kept_pmds_df["region_id"] = np.arange(len(kept_pmds_df))
    return scored_df, pmds_to_remove_df, kept_pmds_df


def get_overlapping_row_ids(wgbs_df, regions_df):
    if wgbs_df.empty or regions_df.empty:
        return np.array([], dtype=np.int64)
    mask = np.zeros(len(wgbs_df), dtype=bool)
    grouped_wgbs = {
        chrom: chrom_df.sort_values(["start", "end"]).copy()
        for chrom, chrom_df in wgbs_df.groupby("chrom", sort=False)
    }
    for chrom, chrom_regions_df in regions_df.groupby("chrom", sort=False):
        chrom_wgbs_df = grouped_wgbs.get(chrom)
        if chrom_wgbs_df is None or chrom_wgbs_df.empty:
            continue
        starts = chrom_wgbs_df["start"].to_numpy(dtype=np.int64)
        ends = chrom_wgbs_df["end"].to_numpy(dtype=np.int64)
        row_ids = chrom_wgbs_df.index.to_numpy(dtype=np.int64)
        for row in chrom_regions_df.itertuples(index=False):
            left = int(np.searchsorted(ends, int(row.start), side="right"))
            right = int(np.searchsorted(starts, int(row.end), side="left"))
            if right <= left:
                continue
            mask[row_ids[left:right]] = True
    return np.flatnonzero(mask)


def convert_pmds_to_hmd(sample, wgbs_df, pmds_to_remove_df):
    converted_df = wgbs_df.copy()
    modified_row_ids = get_overlapping_row_ids(converted_df, pmds_to_remove_df)
    sample_seed = stable_name_seed(sample)
    if len(modified_row_ids) == 0:
        return converted_df, modified_row_ids, sample_seed
    sample_rng = np.random.default_rng(sample_seed)
    replacement_beta = sample_rng.beta(13.0, 1.5, size=len(modified_row_ids))
    converted_df.loc[modified_row_ids, "beta"] = replacement_beta
    new_meth = np.rint(
        converted_df.loc[modified_row_ids, "beta"].to_numpy()
        * converted_df.loc[modified_row_ids, "coverage"].to_numpy()
    ).astype(np.int64)
    new_meth = np.clip(
        new_meth, 0, converted_df.loc[modified_row_ids, "coverage"].to_numpy()
    )
    converted_df.loc[modified_row_ids, "meth"] = new_meth
    return add_beta_column(converted_df), modified_row_ids, sample_seed


def validate_background_conversion(sample, original_df, converted_df, modified_row_ids):
    if not original_df[["chrom", "start", "end"]].equals(
        converted_df[["chrom", "start", "end"]]
    ):
        raise AssertionError(f"Coordinate scaffold changed for {sample}")
    modified_mask = np.zeros(len(original_df), dtype=bool)
    modified_mask[modified_row_ids] = True
    if (
        not original_df.loc[~modified_mask, ["meth", "coverage"]]
        .reset_index(drop=True)
        .equals(
            converted_df.loc[~modified_mask, ["meth", "coverage"]].reset_index(
                drop=True
            )
        )
    ):
        raise AssertionError(f"Non-PMD rows changed for {sample}")
    if len(modified_row_ids) > 0:
        if (
            converted_df.loc[modified_row_ids, "beta"].mean()
            <= original_df.loc[modified_row_ids, "beta"].mean()
        ):
            raise AssertionError(
                f"Converted PMD rows did not shift upward for {sample}"
            )


def process_healthy_sample(
    record,
    healthy_pmd_root: Path,
    output_dir: Path,
    tools_info_df=None,
    min_beta=0.2,
    max_beta=0.7,
):
    sample = record["synthetic_sample_id"]
    tools_info_df = (
        tools_info_df
        if tools_info_df is not None
        else pd.DataFrame(HEALTHY_PMD_TOOL_REGISTRY)
    )
    normalized_dir = output_dir / "normalized_sources"
    truth_dir = output_dir / "truth_regions"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)

    wgbs_df = load_sample_wgbs_from_shared_prep(sample, healthy_pmd_root)
    beta_df = load_sample_beta_from_shared_prep(sample, healthy_pmd_root)
    tool_pmds_df, tool_summary_df = load_all_pmd_regions(
        sample, healthy_pmd_root, tools_info_df
    )
    candidate_union_pmds_df = annotate_union_regions_with_tools(
        merge_union_regions(tool_pmds_df),
        tool_pmds_df,
    )
    scored_pmds_df, pmds_to_remove_df, kept_pmds_df = filter_candidate_pmds(
        candidate_union_pmds_df,
        beta_df,
        min_beta=min_beta,
        max_beta=max_beta,
    )
    converted_df, modified_row_ids, sample_seed = convert_pmds_to_hmd(
        sample, wgbs_df, pmds_to_remove_df
    )
    validate_background_conversion(sample, wgbs_df, converted_df, modified_row_ids)

    normalized_path = normalized_dir / f"{sample}.normalized.tsv"
    converted_df[WG_COLUMNS].to_csv(
        normalized_path, sep="\t", header=False, index=False
    )
    truth_path = truth_dir / f"{sample}.background_removed_pmds.bed"
    truth_df = pmds_to_remove_df[["chrom", "start", "end"]].copy()
    truth_df["pmd_id"] = [
        f"background_removed_pmd_{idx:04d}" for idx in range(len(truth_df))
    ]
    truth_df.to_csv(truth_path, sep="\t", header=False, index=False)

    kept_pmds_df = (
        kept_pmds_df[
            [
                "chrom",
                "start",
                "end",
                "region_length_bp",
                "mean_beta",
                "contributing_tools",
                "n_contributing_tools",
            ]
        ]
        .sort_values(["chrom", "start", "end"])
        .reset_index(drop=True)
    )

    return {
        "summary_row": {
            "synthetic_sample_id": sample,
            "source_file": str(record["source_file"]),
            "source_genome": record["source_genome"],
            "source_kind": record["source_kind"],
            "normalized_source_file": to_manifest_path(normalized_path),
            "truth_bed": to_manifest_path(truth_path),
            "n_tools_with_output": int(tool_summary_df.shape[0]),
            "n_candidate_union_pmds": int(len(candidate_union_pmds_df)),
            "n_pmds_to_remove": int(len(pmds_to_remove_df)),
            "n_kept_pmds": int(len(kept_pmds_df)),
            "n_modified_cpg_rows": int(len(modified_row_ids)),
            "sample_seed": int(sample_seed),
        },
        "scored_pmds_df": scored_pmds_df,
        "pmds_to_remove_df": pmds_to_remove_df,
        "kept_pmds_df": kept_pmds_df,
        "tool_summary_df": tool_summary_df,
    }


def _healthy_sample_worker(kwargs):
    return process_healthy_sample(**kwargs)


def build_healthy_background_references(
    records,
    healthy_pmd_root: Path,
    output_dir: Path,
    tools_info_df=None,
    min_beta=0.2,
    max_beta=0.8,
    n_procs=None,
):
    tools_info_df = (
        tools_info_df
        if tools_info_df is not None
        else pd.DataFrame(HEALTHY_PMD_TOOL_REGISTRY)
    )
    included_records = [record for record in records if record.get("include", True)]
    worker_kwargs = [
        {
            "record": record,
            "healthy_pmd_root": healthy_pmd_root,
            "output_dir": output_dir,
            "tools_info_df": tools_info_df,
            "min_beta": min_beta,
            "max_beta": max_beta,
        }
        for record in included_records
    ]
    if n_procs is None:
        n_procs = min(len(worker_kwargs), os.cpu_count() or 1)
    manifest_rows = []
    sample_results = {}
    if n_procs <= 1:
        results = [_healthy_sample_worker(kwargs) for kwargs in worker_kwargs]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_procs) as pool:
            results = list(pool.imap_unordered(_healthy_sample_worker, worker_kwargs))
    for sample_result in results:
        sample = sample_result["summary_row"]["synthetic_sample_id"]
        sample_results[sample] = sample_result
        manifest_rows.append(sample_result["summary_row"])
    manifest_df = (
        pd.DataFrame(manifest_rows)
        .sort_values("synthetic_sample_id")
        .reset_index(drop=True)
    )
    manifest_path = output_dir / "synthetic_sample_manifest.tsv"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_path, sep="\t", index=False)
    return sample_results, manifest_df, manifest_path


def validate_background_manifest(manifest_path, healthy_pmd_root: Path):
    manifest_df = pd.read_csv(manifest_path, sep="\t")
    validation_rows = []
    for row in manifest_df.itertuples(index=False):
        original_df = load_sample_wgbs_from_shared_prep(
            row.synthetic_sample_id, healthy_pmd_root
        )
        converted_df = add_beta_column(load_wgbs_table(row.normalized_source_file))
        truth_df = load_truth_table(row.truth_bed)
        modified_row_ids = get_overlapping_row_ids(converted_df, truth_df)
        validate_background_conversion(
            row.synthetic_sample_id, original_df, converted_df, modified_row_ids
        )
        validation_rows.append(
            {
                "synthetic_sample_id": row.synthetic_sample_id,
                "n_pmds_to_remove": int(len(truth_df)),
                "n_modified_cpg_rows": int(len(modified_row_ids)),
            }
        )
    return (
        pd.DataFrame(validation_rows)
        .sort_values("synthetic_sample_id")
        .reset_index(drop=True)
    )


def get_manual_intervals(sample_id, chrom, config):
    manual_overrides = config.get("manual_overrides", {})
    intervals = []
    for key in ("all_samples", sample_id):
        for start, end in manual_overrides.get(key, {}).get(chrom, []):
            intervals.append((int(start), int(end)))
    return intervals


def compute_allowed_segments(chrom_start, chrom_end, config):
    chrom_span = chrom_end - chrom_start
    end_buffer = int(round(chrom_span * config["end_buffer_frac"]))
    center_buffer = int(round(chrom_span * config["center_buffer_frac"]))
    midpoint = chrom_start + chrom_span / 2.0
    center_start = int(round(midpoint - center_buffer / 2.0))
    center_end = int(round(midpoint + center_buffer / 2.0))
    left_segment = (chrom_start + end_buffer, center_start)
    right_segment = (center_end, chrom_end - end_buffer)
    allowed = [
        (int(start), int(end))
        for start, end in [left_segment, right_segment]
        if end - start >= config["min_len_bp"]
    ]
    if not allowed:
        raise ValueError("No allowed PMD placement space after end/center exclusions.")
    return allowed


def compute_supported_spans(chrom_df: pd.DataFrame, min_len_bp: int):
    if chrom_df.empty:
        return []
    sorted_df = chrom_df.sort_values(["start", "end"]).reset_index(drop=True)
    supported_spans = []
    current_start = int(sorted_df.loc[0, "start"])
    current_end = int(sorted_df.loc[0, "end"])
    for row in sorted_df.iloc[1:].itertuples(index=False):
        row_start = int(row.start)
        row_end = int(row.end)
        if row_start - current_end >= min_len_bp:
            supported_spans.append((current_start, current_end))
            current_start = row_start
            current_end = row_end
            continue
        current_end = max(current_end, row_end)
    supported_spans.append((current_start, current_end))
    return supported_spans


def compute_candidate_segments(chrom_df: pd.DataFrame, config):
    candidate_segments = []
    supported_spans = compute_supported_spans(chrom_df, int(config["min_len_bp"]))
    for span_start, span_end in supported_spans:
        try:
            candidate_segments.extend(
                compute_allowed_segments(span_start, span_end, config)
            )
        except ValueError:
            continue
    return candidate_segments


def max_intervals_for_segment(segment, config) -> int:
    seg_start, seg_end = segment
    usable_bp = int(seg_end) - int(seg_start)
    min_len = int(config["min_len_bp"])
    min_gap = int(config["min_gap_bp"])
    if usable_bp < min_len:
        return 0
    return int((usable_bp + min_gap) // (min_len + min_gap))


def compute_max_interval_capacity(allowed_segments, config) -> int:
    return int(
        sum(max_intervals_for_segment(segment, config) for segment in allowed_segments)
    )


def get_feasible_interval_count_bounds(allowed_segments, config):
    max_capacity = compute_max_interval_capacity(allowed_segments, config)
    lower_bound = min(int(config["pmds_per_chrom_min"]), max_capacity)
    upper_bound = min(int(config["pmds_per_chrom_max"]), max_capacity)
    return lower_bound, upper_bound


def validate_interval(interval, allowed_segments, selected_intervals, config):
    start, end = interval
    if end <= start:
        return False
    if end - start < config["min_len_bp"] or end - start > config["max_len_bp"]:
        return False
    if not any(
        start >= seg_start and end <= seg_end for seg_start, seg_end in allowed_segments
    ):
        return False
    min_gap = int(config["min_gap_bp"])
    for existing_start, existing_end in selected_intervals:
        if not (end + min_gap <= existing_start or start >= existing_end + min_gap):
            return False
    return True


def sample_interval_length(rng, config):
    min_len = int(config["min_len_bp"])
    max_len = int(config["max_len_bp"])
    if min_len > max_len:
        raise ValueError(f"min_len_bp {min_len} cannot exceed max_len_bp {max_len}.")
    sampling = str(config.get("length_sampling", "log_normal")).lower()
    if sampling == "uniform":
        return int(rng.integers(min_len, max_len + 1))
    if sampling == "normal":
        mean_len = float(config.get("length_mean_bp", (min_len + max_len) / 2.0))
        std_len = float(config.get("length_std_bp", (max_len - min_len) / 4.0))
        for _ in range(1000):
            sampled_len = int(round(rng.normal(mean_len, std_len)))
            if min_len <= sampled_len <= max_len:
                return sampled_len
        return int(np.clip(round(mean_len), min_len, max_len))
    if sampling == "log_normal":
        mean_len = float(config.get("length_mean_bp", 800_000))
        std_len = float(config.get("length_std_bp", 700_000))
        if mean_len <= 0 or std_len <= 0:
            raise ValueError(
                "length_mean_bp and length_std_bp must be positive for log_normal sampling."
            )
        variance = std_len**2
        log_sigma_sq = np.log1p(variance / (mean_len**2))
        log_sigma = float(np.sqrt(log_sigma_sq))
        log_mean = float(np.log(mean_len) - 0.5 * log_sigma_sq)
        for _ in range(1000):
            sampled_len = int(round(rng.lognormal(mean=log_mean, sigma=log_sigma)))
            if min_len <= sampled_len <= max_len:
                return sampled_len
        return int(np.clip(round(mean_len), min_len, max_len))
    raise ValueError(f"Unsupported length_sampling mode: {sampling}")


def format_segment_diagnostics(allowed_segments, max_segments=6):
    if not allowed_segments:
        return "none"
    segment_lengths = [
        (int(seg_start), int(seg_end), int(seg_end) - int(seg_start))
        for seg_start, seg_end in allowed_segments
    ]
    segment_lengths = sorted(segment_lengths, key=lambda row: row[2], reverse=True)
    preview = ", ".join(
        f"{start}-{end} ({length:,} bp)"
        for start, end, length in segment_lengths[:max_segments]
    )
    if len(segment_lengths) > max_segments:
        preview += f", ... {len(segment_lengths) - max_segments} more"
    return preview


def sample_random_interval(rng, allowed_segments, selected_intervals, config):
    interval_len = sample_interval_length(rng, config)
    feasible_segments = []
    for seg_start, seg_end in allowed_segments:
        max_start = int(seg_end) - interval_len
        if max_start >= int(seg_start):
            feasible_segments.append((int(seg_start), int(max_start)))
    if not feasible_segments:
        return None, {
            "interval_len": int(interval_len),
            "feasible_segment_count": 0,
            "allowed_segments": format_segment_diagnostics(allowed_segments),
            "reason": "sampled_length_exceeds_all_allowed_segments",
        }
    weights = np.array(
        [(max_start - seg_start + 1) for seg_start, max_start in feasible_segments],
        dtype=float,
    )
    weights = weights / weights.sum()
    seg_index = int(rng.choice(len(feasible_segments), p=weights))
    seg_start, max_start = feasible_segments[seg_index]
    start = int(rng.integers(seg_start, max_start + 1))
    end = start + interval_len
    interval = (start, end)
    if validate_interval(interval, allowed_segments, selected_intervals, config):
        return interval, {
            "interval_len": int(interval_len),
            "feasible_segment_count": int(len(feasible_segments)),
            "allowed_segments": format_segment_diagnostics(allowed_segments),
            "reason": "placed",
        }
    return None, {
        "interval_len": int(interval_len),
        "feasible_segment_count": int(len(feasible_segments)),
        "allowed_segments": format_segment_diagnostics(allowed_segments),
        "reason": "failed_gap_or_overlap_constraint",
    }


def sample_intervals_for_chromosome(allowed_segments, target_count, rng, config):
    if target_count <= 0:
        return []

    total_attempts = int(config.get("placement_attempts_per_chrom", 10_000))
    restart_attempts = max(10, min(250, total_attempts // max(target_count, 1)))
    per_interval_attempts = max(50, total_attempts // max(target_count, 1))

    best_intervals = []
    last_failure_details = None
    for _ in range(restart_attempts):
        intervals = []
        for _ in range(target_count):
            placed = False
            attempt_lengths = []
            for _ in range(per_interval_attempts):
                interval, interval_info = sample_random_interval(
                    rng, allowed_segments, intervals, config
                )
                attempt_lengths.append(int(interval_info["interval_len"]))
                if interval is not None:
                    intervals.append(interval)
                    placed = True
                    break
            if not placed:
                last_failure_details = {
                    "requested_count": int(target_count),
                    "placed_count": int(len(intervals)),
                    "failed_interval_index": int(len(intervals) + 1),
                    "attempted_lengths_bp": attempt_lengths,
                    "last_attempt_reason": interval_info["reason"],
                    "last_attempt_length_bp": int(interval_info["interval_len"]),
                    "feasible_segment_count": int(
                        interval_info["feasible_segment_count"]
                    ),
                    "allowed_segments": interval_info["allowed_segments"],
                    "placed_interval_lengths_bp": [
                        int(end - start) for start, end in intervals
                    ],
                    "min_gap_bp": int(config["min_gap_bp"]),
                }
                break
        if len(intervals) > len(best_intervals):
            best_intervals = intervals.copy()
        if len(intervals) == target_count:
            return sorted(intervals)

    if best_intervals:
        return sorted(best_intervals)

    detail_text = ""
    if last_failure_details is not None:
        detail_text = (
            f" requested={last_failure_details['requested_count']},"
            f" placed={last_failure_details['placed_count']},"
            f" failed_interval_index={last_failure_details['failed_interval_index']},"
            f" last_attempt_reason={last_failure_details['last_attempt_reason']},"
            f" last_attempt_length_bp={last_failure_details['last_attempt_length_bp']:,},"
            f" attempted_lengths_bp={last_failure_details['attempted_lengths_bp']},"
            f" placed_interval_lengths_bp={last_failure_details['placed_interval_lengths_bp']},"
            f" min_gap_bp={last_failure_details['min_gap_bp']:,},"
            f" feasible_segment_count={last_failure_details['feasible_segment_count']},"
            f" allowed_segments={last_failure_details['allowed_segments']}"
        )
    raise ValueError(
        f"Could not place any of {target_count} requested PMDs.{detail_text}"
    )


def generate_truth_regions(sample_id, wgbs_df, config, rng):
    truth_rows = []
    for chrom in AUTOSOMES:
        chrom_df = wgbs_df.loc[wgbs_df["chrom"] == chrom, ["start", "end"]]
        if chrom_df.empty:
            continue
        allowed_segments = compute_candidate_segments(chrom_df, config)
        selected_intervals = []
        manual_intervals = get_manual_intervals(sample_id, chrom, config)
        for interval in manual_intervals:
            if not validate_interval(
                interval, allowed_segments, selected_intervals, config
            ):
                raise ValueError(
                    f"Manual PMD interval {interval} is invalid for {sample_id} {chrom}."
                )
            selected_intervals.append(interval)
        if not allowed_segments and not manual_intervals:
            continue
        if manual_intervals:
            target_count = len(manual_intervals)
        else:
            lower_bound, upper_bound = get_feasible_interval_count_bounds(
                allowed_segments, config
            )
            if upper_bound == 0:
                continue
            target_count = int(rng.integers(lower_bound, upper_bound + 1))
            placed_intervals = None
            last_error = None
            for candidate_count in range(target_count, 0, -1):
                try:
                    placed_intervals = sample_intervals_for_chromosome(
                        allowed_segments=allowed_segments,
                        target_count=candidate_count,
                        rng=rng,
                        config=config,
                    )
                    break
                except ValueError as exc:
                    last_error = exc
                    continue
            if placed_intervals is None:
                raise ValueError(
                    f"Failed to place synthetic PMDs for sample={sample_id}, chrom={chrom}. "
                    f"requested_count={target_count}, lower_bound={lower_bound}, "
                    f"upper_bound={upper_bound}, allowed_segments={format_segment_diagnostics(allowed_segments)}. "
                    f"Last placement error: {last_error}"
                )
            selected_intervals.extend(placed_intervals)
        for idx, (start, end) in enumerate(sorted(selected_intervals), start=1):
            truth_rows.append(
                {
                    "chrom": chrom,
                    "start": int(start),
                    "end": int(end),
                    "pmd_id": f"{sample_id}_{chrom}_PMD_{idx:02d}",
                }
            )
    return (
        pd.DataFrame(truth_rows, columns=TRUTH_COLUMNS)
        .sort_values(["chrom", "start", "end"])
        .reset_index(drop=True)
    )


def build_pmd_mask(wgbs_df, truth_df):
    mask = np.zeros(len(wgbs_df), dtype=bool)
    for chrom in AUTOSOMES:
        chrom_index = wgbs_df.index[wgbs_df["chrom"] == chrom].to_numpy()
        if chrom_index.size == 0:
            continue
        chrom_starts = wgbs_df.loc[chrom_index, "start"].to_numpy()
        chrom_truth = truth_df.loc[truth_df["chrom"] == chrom, ["start", "end"]]
        for start, end in chrom_truth.itertuples(index=False):
            left = int(np.searchsorted(chrom_starts, start, side="left"))
            right = int(np.searchsorted(chrom_starts, end, side="left"))
            mask[chrom_index[left:right]] = True
    return mask


def sample_pmd_probabilities(rng, size):
    mixture_draw = rng.random(size)
    probabilities = np.empty(size, dtype=float)
    mid_mask = mixture_draw < 0.80
    low_mask = (mixture_draw >= 0.80) & (mixture_draw < 0.95)
    high_mask = mixture_draw >= 0.95
    probabilities[mid_mask] = rng.beta(5.0, 5.0, size=mid_mask.sum())
    probabilities[low_mask] = rng.beta(1.5, 13.0, size=low_mask.sum())
    probabilities[high_mask] = rng.beta(13.0, 1.5, size=high_mask.sum())
    return probabilities


def sample_noise_spike_probabilities(rng, size, spike_kind):
    if size <= 0:
        return np.array([], dtype=float)
    if spike_kind == "low":
        return rng.beta(1.2, 20.0, size=size)
    if spike_kind == "high":
        return rng.beta(20.0, 1.2, size=size)
    raise ValueError(f"Unsupported spike_kind: {spike_kind}")


def sample_noise_spike_intervals(region_start, region_end, rng, noise_config):
    region_start = int(region_start)
    region_end = int(region_end)
    region_len = region_end - region_start
    if region_len < int(noise_config["min_region_len_bp"]):
        return []
    min_len = int(noise_config["spike_len_min_bp"])
    max_len = min(int(noise_config["spike_len_max_bp"]), region_len)
    if min_len > max_len:
        return []
    spike_count = int(
        rng.integers(
            int(noise_config["spikes_per_region_min"]),
            int(noise_config["spikes_per_region_max"]) + 1,
        )
    )
    selected_spikes = []
    for _ in range(spike_count):
        spike_len = int(rng.integers(min_len, max_len + 1))
        max_start = region_end - spike_len
        if max_start < region_start:
            continue
        for _ in range(100):
            spike_start = int(rng.integers(region_start, max_start + 1))
            spike_end = spike_start + spike_len
            if all(
                spike_end <= existing_start or spike_start >= existing_end
                for existing_start, existing_end, _ in selected_spikes
            ):
                spike_kind = (
                    "high"
                    if rng.random() < float(noise_config["high_spike_fraction"])
                    else "low"
                )
                selected_spikes.append((spike_start, spike_end, spike_kind))
                break
    return sorted(selected_spikes)


def apply_biological_noise_to_probabilities(
    probabilities, wgbs_df, truth_df, rng, config
):
    noise_config = config.get("biological_noise", {})
    if not noise_config.get("enabled", False) or truth_df.empty:
        return probabilities
    updated_probabilities = probabilities.copy()
    pmd_row_positions = np.flatnonzero(build_pmd_mask(wgbs_df, truth_df))
    for chrom in truth_df["chrom"].unique():
        chrom_truth = truth_df.loc[truth_df["chrom"] == chrom, ["start", "end"]]
        chrom_mask = wgbs_df["chrom"] == chrom
        chrom_row_indices = wgbs_df.index[chrom_mask].to_numpy()
        if chrom_row_indices.size == 0:
            continue
        chrom_starts = wgbs_df.loc[chrom_row_indices, "start"].to_numpy(dtype=np.int64)
        for region_start, region_end in chrom_truth.itertuples(index=False):
            spike_intervals = sample_noise_spike_intervals(
                region_start, region_end, rng, noise_config
            )
            for spike_start, spike_end, spike_kind in spike_intervals:
                left = int(np.searchsorted(chrom_starts, int(spike_start), side="left"))
                right = int(np.searchsorted(chrom_starts, int(spike_end), side="left"))
                if right <= left:
                    continue
                absolute_row_indices = chrom_row_indices[left:right]
                probability_idx = np.searchsorted(
                    pmd_row_positions, absolute_row_indices
                )
                updated_probabilities[probability_idx] = (
                    sample_noise_spike_probabilities(
                        rng=rng,
                        size=probability_idx.size,
                        spike_kind=spike_kind,
                    )
                )
    return updated_probabilities


def inject_synthetic_pmds(wgbs_df, truth_df, rng, config=None):
    synthetic_df = wgbs_df.copy()
    pmd_mask = build_pmd_mask(synthetic_df, truth_df)
    coverage = synthetic_df.loc[pmd_mask, "coverage"].to_numpy(dtype=np.int64)
    probabilities = sample_pmd_probabilities(rng, size=coverage.size)
    if config is not None:
        probabilities = apply_biological_noise_to_probabilities(
            probabilities, synthetic_df, truth_df, rng, config
        )
    synthetic_meth = rng.binomial(coverage, probabilities)
    synthetic_df.loc[pmd_mask, "meth"] = synthetic_meth.astype(np.int64)
    synthetic_df["meth"] = np.clip(synthetic_df["meth"], 0, synthetic_df["coverage"])
    return synthetic_df, pmd_mask


def count_zero_overlap_truth_regions(truth_df, wgbs_df):
    zero_overlap_count = 0
    for chrom in truth_df["chrom"].unique():
        chrom_truth = truth_df.loc[truth_df["chrom"] == chrom, ["start", "end"]]
        chrom_df = (
            wgbs_df.loc[wgbs_df["chrom"] == chrom, ["start", "end"]]
            .sort_values(["start", "end"])
            .reset_index(drop=True)
        )
        if chrom_df.empty:
            zero_overlap_count += len(chrom_truth)
            continue
        chrom_starts = chrom_df["start"].to_numpy(dtype=np.int64)
        chrom_ends = chrom_df["end"].to_numpy(dtype=np.int64)
        row_idx = 0
        n_rows = len(chrom_starts)
        for truth_row in chrom_truth.itertuples(index=False):
            truth_start = int(truth_row.start)
            truth_end = int(truth_row.end)
            while row_idx < n_rows and int(chrom_ends[row_idx]) <= truth_start:
                row_idx += 1
            has_overlap = row_idx < n_rows and int(chrom_starts[row_idx]) < truth_end
            if not has_overlap:
                zero_overlap_count += 1
    return int(zero_overlap_count)


def validate_truth_regions(sample_id, truth_df, wgbs_df, config):
    issues = []
    for chrom in truth_df["chrom"].unique():
        chrom_truth = (
            truth_df.loc[truth_df["chrom"] == chrom]
            .sort_values(["start", "end"])
            .reset_index(drop=True)
        )
        chrom_df = wgbs_df.loc[wgbs_df["chrom"] == chrom, ["start", "end"]]
        if chrom_df.empty:
            issues.append(
                f"{chrom}: truth intervals exist but chromosome has no WGBS rows."
            )
            continue
        allowed_segments = compute_candidate_segments(chrom_df, config)
        for row in chrom_truth.itertuples(index=False):
            interval = (int(row.start), int(row.end))
            if not any(
                interval[0] >= seg_start and interval[1] <= seg_end
                for seg_start, seg_end in allowed_segments
            ):
                issues.append(
                    f"{chrom}: PMD {interval} falls outside the supported placement spans."
                )
        for idx in range(1, len(chrom_truth)):
            prev_end = int(chrom_truth.loc[idx - 1, "end"])
            curr_start = int(chrom_truth.loc[idx, "start"])
            if curr_start < prev_end + config["min_gap_bp"]:
                issues.append(
                    f"{chrom}: PMDs at index {idx-1} and {idx} violate the minimum gap."
                )
    zero_overlap_count = count_zero_overlap_truth_regions(truth_df, wgbs_df)
    if zero_overlap_count:
        issues.append(
            f"Found {zero_overlap_count} truth interval(s) with zero overlap against normalized WGBS rows."
        )
    return issues


def create_synthetic_outputs(
    sample_id,
    wgbs_df,
    config,
    sample_seed,
    truth_bed_path: Path,
    synthetic_path: Path,
):
    wgbs_df = finalize_wgbs_table(wgbs_df)
    rng = np.random.default_rng(sample_seed)
    truth_df = generate_truth_regions(sample_id, wgbs_df, config, rng)
    synthetic_df, pmd_mask = inject_synthetic_pmds(
        wgbs_df, truth_df, rng, config=config
    )
    synthetic_path.parent.mkdir(parents=True, exist_ok=True)
    truth_bed_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(synthetic_path, sep="\t", header=False, index=False)
    truth_df.to_csv(truth_bed_path, sep="\t", header=False, index=False)
    validation_issues = validate_truth_regions(sample_id, truth_df, wgbs_df, config)
    if validation_issues:
        raise AssertionError(
            "Truth region validation failed:\n" + "\n".join(validation_issues)
        )
    if not synthetic_df["meth"].le(synthetic_df["coverage"]).all():
        raise AssertionError(
            f"{sample_id}: found methylation counts greater than coverage."
        )
    if not synthetic_df["coverage"].equals(wgbs_df["coverage"]):
        raise AssertionError(f"{sample_id}: coverage changed during PMD injection.")
    return {
        "truth_df": truth_df,
        "synthetic_df": synthetic_df,
        "pmd_mask": pmd_mask,
        "n_truth_pmds": int(len(truth_df)),
        "n_pmd_mask_rows": int(pmd_mask.sum()),
        "n_zero_overlap_truth_regions": int(
            count_zero_overlap_truth_regions(truth_df, wgbs_df)
        ),
    }


def build_injected_sample_id(background_sample_id):
    return f"{background_sample_id}_injected_pmds"


def create_injected_sample(record, output_dir: Path, config, overwrite=False):
    synthetic_dir = output_dir / "synthetic_files"
    truth_dir = output_dir / "truth_regions"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    background_sample_id = record["synthetic_sample_id"]
    injected_sample_id = build_injected_sample_id(background_sample_id)
    synthetic_path = synthetic_dir / f"{injected_sample_id}.synthetic.tsv"
    truth_bed_path = truth_dir / f"{injected_sample_id}.truth.bed"
    sample_seed = stable_sample_seed(config["global_seed"], injected_sample_id)
    if synthetic_path.exists() and truth_bed_path.exists() and not overwrite:
        normalized_df = load_wgbs_table(record["normalized_source_file"])
        truth_df = load_truth_table(truth_bed_path, root=Path.cwd())
        synthetic_df = load_wgbs_table(synthetic_path, root=Path.cwd())
        cache_valid = (
            len(normalized_df) == len(synthetic_df)
            and normalized_df[["chrom", "start", "end"]].equals(
                synthetic_df[["chrom", "start", "end"]]
            )
            and normalized_df["coverage"].equals(synthetic_df["coverage"])
        )
        if cache_valid:
            pmd_mask = build_pmd_mask(synthetic_df, truth_df)
            return {
                "synthetic_sample_id": injected_sample_id,
                "background_sample_id": background_sample_id,
                "source_file": record["source_file"],
                "source_genome": record["source_genome"],
                "source_kind": record["source_kind"],
                "normalized_source_file": record["normalized_source_file"],
                "background_truth_bed": record["truth_bed"],
                "synthetic_file": to_manifest_path(synthetic_path),
                "truth_bed": to_manifest_path(truth_bed_path),
                "sample_seed": int(sample_seed),
                "n_truth_regions": int(len(truth_df)),
                "n_truth_cpg_rows": int(pmd_mask.sum()),
                "n_background_removed_pmds": int(record.get("n_pmds_to_remove", 0)),
            }
    wgbs_df = load_wgbs_table(record["normalized_source_file"])
    result = create_synthetic_outputs(
        sample_id=injected_sample_id,
        wgbs_df=wgbs_df,
        config=config,
        sample_seed=sample_seed,
        truth_bed_path=truth_bed_path,
        synthetic_path=synthetic_path,
    )
    return {
        "synthetic_sample_id": injected_sample_id,
        "background_sample_id": background_sample_id,
        "source_file": record["source_file"],
        "source_genome": record["source_genome"],
        "source_kind": record["source_kind"],
        "normalized_source_file": record["normalized_source_file"],
        "background_truth_bed": record["truth_bed"],
        "synthetic_file": to_manifest_path(synthetic_path),
        "truth_bed": to_manifest_path(truth_bed_path),
        "sample_seed": int(sample_seed),
        "n_truth_regions": int(result["n_truth_pmds"]),
        "n_truth_cpg_rows": int(result["n_pmd_mask_rows"]),
        "n_background_removed_pmds": int(record.get("n_pmds_to_remove", 0)),
    }


def _injected_sample_worker(kwargs):
    return create_injected_sample(**kwargs)


def build_injected_manifest(
    background_manifest,
    output_dir: Path,
    config=None,
    selected_sample_ids=None,
    overwrite=False,
    n_procs=None,
):
    manifest_df = (
        pd.read_csv(background_manifest, sep="\t")
        if not isinstance(background_manifest, pd.DataFrame)
        else background_manifest.copy()
    )
    if selected_sample_ids:
        manifest_df = manifest_df.loc[
            manifest_df["synthetic_sample_id"].isin(set(selected_sample_ids))
        ].copy()
    manifest_df = manifest_df.sort_values("synthetic_sample_id").reset_index(drop=True)
    config = copy.deepcopy(config) if config is not None else load_default_pmd_config()
    worker_kwargs = [
        {
            "record": record,
            "output_dir": output_dir,
            "config": config,
            "overwrite": overwrite,
        }
        for record in manifest_df.to_dict("records")
    ]
    if n_procs is None:
        n_procs = min(len(worker_kwargs), max(1, (os.cpu_count() or 1) - 1))
    if n_procs <= 1:
        built_rows = [_injected_sample_worker(kwargs) for kwargs in worker_kwargs]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_procs) as pool:
            built_rows = list(
                pool.imap_unordered(_injected_sample_worker, worker_kwargs)
            )
    built_manifest = (
        pd.DataFrame(built_rows)
        .sort_values("synthetic_sample_id")
        .reset_index(drop=True)
    )
    manifest_path = output_dir / "synthetic_sample_manifest.tsv"
    output_dir.mkdir(parents=True, exist_ok=True)
    built_manifest.to_csv(manifest_path, sep="\t", index=False)
    return built_manifest, manifest_path


def validate_injected_manifest(manifest_path):
    manifest_df = pd.read_csv(manifest_path, sep="\t")
    validation_rows = []
    for row in manifest_df.itertuples(index=False):
        normalized_df = load_wgbs_table(row.normalized_source_file)
        synthetic_df = load_wgbs_table(row.synthetic_file)
        truth_df = load_truth_table(row.truth_bed)
        same_row_count = len(normalized_df) == len(synthetic_df)
        same_coordinates = normalized_df[["chrom", "start", "end"]].equals(
            synthetic_df[["chrom", "start", "end"]]
        )
        same_coverage = normalized_df["coverage"].equals(synthetic_df["coverage"])
        meth_within_coverage = synthetic_df["meth"].le(synthetic_df["coverage"]).all()
        truth_mask = build_pmd_mask(synthetic_df, truth_df)
        n_truth_cpg_rows = int(truth_mask.sum())
        modified_mask = normalized_df["meth"].ne(synthetic_df["meth"])
        n_modified_rows = int(modified_mask.sum())
        if not same_row_count:
            raise AssertionError(
                f"{row.synthetic_sample_id}: row count changed after injection."
            )
        if not same_coordinates:
            raise AssertionError(
                f"{row.synthetic_sample_id}: genomic coordinates changed after injection."
            )
        if not same_coverage:
            raise AssertionError(
                f"{row.synthetic_sample_id}: coverage changed after injection."
            )
        if not meth_within_coverage:
            raise AssertionError(f"{row.synthetic_sample_id}: found meth > coverage.")
        if n_truth_cpg_rows == 0:
            raise AssertionError(
                f"{row.synthetic_sample_id}: truth regions do not overlap any CpG rows."
            )
        if n_modified_rows == 0:
            raise AssertionError(
                f"{row.synthetic_sample_id}: no CpG rows were modified by injection."
            )
        validation_rows.append(
            {
                "synthetic_sample_id": row.synthetic_sample_id,
                "n_truth_regions": int(len(truth_df)),
                "n_truth_cpg_rows": n_truth_cpg_rows,
                "n_modified_rows": n_modified_rows,
                "mean_background_beta": float(
                    add_beta_column(normalized_df)["beta"].mean()
                ),
                "mean_synthetic_beta": float(
                    add_beta_column(synthetic_df)["beta"].mean()
                ),
            }
        )
    return (
        pd.DataFrame(validation_rows)
        .sort_values("synthetic_sample_id")
        .reset_index(drop=True)
    )


def load_injected_sample_for_plot(manifest_path, sample_id=None):
    manifest_df = pd.read_csv(manifest_path, sep="\t")
    if manifest_df.empty:
        raise ValueError("Injected synthetic manifest is empty.")
    if sample_id is None:
        sample_id = manifest_df.iloc[0]["synthetic_sample_id"]
    row = manifest_df.loc[manifest_df["synthetic_sample_id"] == sample_id].iloc[0]
    background_df = add_beta_column(load_wgbs_table(row["normalized_source_file"]))
    synthetic_df = add_beta_column(load_wgbs_table(row["synthetic_file"]))
    truth_df = load_truth_table(row["truth_bed"])
    synthetic_df["is_truth_pmd"] = build_pmd_mask(synthetic_df, truth_df)
    return row, background_df, synthetic_df, truth_df


def get_methyl_tool_comparator_class(comparator_dir: Path):
    comparator_dir = Path(comparator_dir).resolve()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if str(comparator_dir) not in sys.path:
        sys.path.insert(0, str(comparator_dir))
    importlib.invalidate_caches()
    module = importlib.import_module("methyl_tool_comparator")
    return module.MethylToolComparator


def write_recovery_configs(injected_manifest, config_dir: Path):
    manifest_df = (
        pd.read_csv(injected_manifest, sep="\t")
        if not isinstance(injected_manifest, pd.DataFrame)
        else injected_manifest.copy()
    )
    records = manifest_df.to_dict("records")
    config_rows = []
    config_dir.mkdir(parents=True, exist_ok=True)
    import yaml

    for record in records:
        sample_id = record["synthetic_sample_id"]
        config_path = config_dir / f"{sample_id}.yaml"
        payload = {
            "sample": sample_id,
            "meth_file": str(resolve_analysis_path(record["synthetic_file"])),
            "genome": record["source_genome"],
        }
        with open(config_path, "w") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        config_rows.append(
            {
                "synthetic_sample_id": sample_id,
                "config_path": str(config_path),
                "truth_bed": record["truth_bed"],
                "synthetic_file": record["synthetic_file"],
                "source_genome": record["source_genome"],
            }
        )
    return (
        pd.DataFrame(
            config_rows,
            columns=[
                "synthetic_sample_id",
                "config_path",
                "truth_bed",
                "synthetic_file",
                "source_genome",
            ],
        )
        .sort_values("synthetic_sample_id")
        .reset_index(drop=True)
    )


def run_synthetic_recovery(
    injected_manifest,
    recovery_output_dir: Path,
    comparator_dir: Path,
    selected_sample_ids=None,
    force_recreate=False,
    n_jobs=10,
):
    manifest_df = (
        pd.read_csv(injected_manifest, sep="\t")
        if not isinstance(injected_manifest, pd.DataFrame)
        else injected_manifest.copy()
    )
    if selected_sample_ids:
        manifest_df = manifest_df.loc[
            manifest_df["synthetic_sample_id"].isin(set(selected_sample_ids))
        ].copy()
    manifest_df = manifest_df.sort_values("synthetic_sample_id").reset_index(drop=True)

    tool_results_dir = Path(recovery_output_dir) / "tool_results"
    config_dir = tool_results_dir / "configs"
    config_df = write_recovery_configs(manifest_df, config_dir)
    comparator_class = get_methyl_tool_comparator_class(comparator_dir)

    completed_samples = []
    for row in config_df.itertuples(index=False):
        comparator = comparator_class(
            config_file=str(row.config_path),
            out_dir=str(tool_results_dir),
            force_recreate=force_recreate,
            n_jobs=n_jobs,
            alt_params = {
                "MethylSeg" : {
                    "merge_with_intermediate" : False
                }
            }
        )
        comparator.run()
        completed_samples.append(row.synthetic_sample_id)

    return {
        "manifest_df": manifest_df,
        "config_df": config_df,
        "tool_results_dir": tool_results_dir,
        "completed_samples": completed_samples,
    }


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan


def f1_score_from_precision_recall(precision, recall):
    if pd.isna(precision) or pd.isna(recall) or precision + recall == 0:
        return np.nan
    return float(2 * precision * recall / (precision + recall))


def tool_order_key(tool_name):
    tool_name = str(tool_name)
    priority = {"methylseg": 0, "methylseg_hm450k": 1}.get(tool_name, 2)
    return (priority, tool_name)


def sort_tools_methylseg_first(df, metric_col=None, ascending=False):
    if df.empty or "tool" not in df.columns:
        return df
    out_df = df.copy()
    out_df["_tool_priority"] = out_df["tool"].map(lambda value: tool_order_key(value)[0])
    sort_cols = []
    sort_ascending = []
    if metric_col and metric_col in out_df.columns:
        sort_cols.append(metric_col)
        sort_ascending.append(ascending)
    sort_cols.extend(["_tool_priority", "tool"])
    sort_ascending.extend([True, True])
    return out_df.sort_values(sort_cols, ascending=sort_ascending).drop(
        columns=["_tool_priority"]
    ).reset_index(drop=True)


PAIRWISE_WAO_COLUMNS = [
    "a_chrom",
    "a_start",
    "a_end",
    "a_id",
    "a_len",
    "b_chrom",
    "b_start",
    "b_end",
    "b_id",
    "b_len",
    "overlap_bp",
]


def genome_sizes_path(source_genome: str | None = None) -> Path:
    genome = str(source_genome or "hg38").lower()
    if genome not in GENOME_SIZES_PATHS:
        raise ValueError(f"Unsupported genome for metric slop operations: {source_genome}")
    path = GENOME_SIZES_PATHS[genome]
    if not path.exists():
        raise FileNotFoundError(f"Missing chromosome sizes file: {path}")
    return path


def empty_bedtool() -> BedTool:
    return BedTool("", from_string=True)


def build_tagged_interval_df(interval_df, prefix: str, id_prefix: str) -> pd.DataFrame:
    clean_df = ensure_interval_df(interval_df).sort_values(["chrom", "start", "end"]).reset_index(
        drop=True
    )
    if clean_df.empty:
        return pd.DataFrame(
            columns=[
                f"{prefix}_chrom",
                f"{prefix}_start",
                f"{prefix}_end",
                f"{prefix}_id",
                f"{prefix}_len",
            ]
        )
    tagged_df = clean_df.rename(
        columns={
            "chrom": f"{prefix}_chrom",
            "start": f"{prefix}_start",
            "end": f"{prefix}_end",
        }
    )
    tagged_df[f"{prefix}_id"] = [f"{id_prefix}{i}" for i in range(1, len(tagged_df) + 1)]
    tagged_df[f"{prefix}_len"] = (
        tagged_df[f"{prefix}_end"] - tagged_df[f"{prefix}_start"]
    ).astype(int)
    return tagged_df


def tagged_df_to_bedtool(tagged_df: pd.DataFrame, prefix: str) -> BedTool:
    if tagged_df.empty:
        return empty_bedtool()
    bed_df = tagged_df.rename(
        columns={
            f"{prefix}_chrom": "chrom",
            f"{prefix}_start": "start",
            f"{prefix}_end": "end",
            f"{prefix}_id": "name",
            f"{prefix}_len": "score",
        }
    )[["chrom", "start", "end", "name", "score"]]
    return BedTool.from_dataframe(bed_df)


def cpg_df_to_bedtool(wgbs_df: pd.DataFrame) -> BedTool:
    cpg_df = ensure_interval_df(wgbs_df[["chrom", "start", "end"]]).sort_values(
        ["chrom", "start", "end"]
    )
    if cpg_df.empty:
        return empty_bedtool()
    return BedTool.from_dataframe(cpg_df)


def build_cpg_meta_df(wgbs_df: pd.DataFrame) -> pd.DataFrame:
    cpg_meta = wgbs_df[["chrom", "start", "end"]].copy()
    cpg_meta = ensure_interval_df(cpg_meta).copy()
    if cpg_meta.empty:
        return pd.DataFrame(
            columns=["cpg_chrom", "cpg_start", "cpg_end", "cpg_id", "cpg_beta"]
        )
    cpg_meta["cpg_beta"] = (
        pd.to_numeric(wgbs_df["beta"], errors="coerce")
        if "beta" in wgbs_df.columns
        else np.nan
    )
    cpg_meta = cpg_meta.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    cpg_meta = cpg_meta.rename(
        columns={"chrom": "cpg_chrom", "start": "cpg_start", "end": "cpg_end"}
    )
    cpg_meta["cpg_id"] = [f"C{i}" for i in range(1, len(cpg_meta) + 1)]
    return cpg_meta


def cpg_meta_df_to_bedtool(cpg_meta_df: pd.DataFrame) -> BedTool:
    if cpg_meta_df.empty:
        return empty_bedtool()
    bed_df = cpg_meta_df.rename(
        columns={
            "cpg_chrom": "chrom",
            "cpg_start": "start",
            "cpg_end": "end",
            "cpg_id": "name",
            "cpg_beta": "score",
        }
    )[["chrom", "start", "end", "name", "score"]]
    return BedTool.from_dataframe(bed_df)


def pairwise_wao(a: BedTool, b: BedTool) -> pd.DataFrame:
    if a.count() == 0 or b.count() == 0:
        return pd.DataFrame(columns=PAIRWISE_WAO_COLUMNS)
    overlap_bed = a.intersect(b, wao=True)
    df = overlap_bed.to_dataframe(names=PAIRWISE_WAO_COLUMNS)
    for col in ["a_start", "a_end", "a_len", "b_start", "b_end", "b_len", "overlap_bp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["overlap_bp"] > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=PAIRWISE_WAO_COLUMNS)
    for col in ["a_start", "a_end", "a_len", "b_start", "b_end", "b_len", "overlap_bp"]:
        df[col] = df[col].astype(int)
    return df.reset_index(drop=True)


def prepare_overlap_state(
    truth_df,
    predicted_df,
    boundary_tolerance_bp,
    genome_file=None,
    wgbs_df=None,
):
    truth_meta = build_tagged_interval_df(truth_df, prefix="truth", id_prefix="T")
    recall_meta = build_tagged_interval_df(predicted_df, prefix="recall", id_prefix="R")
    truth_bed = tagged_df_to_bedtool(truth_meta, prefix="truth")
    recall_bed = tagged_df_to_bedtool(recall_meta, prefix="recall")
    orig_pairs = pairwise_wao(truth_bed, recall_bed).rename(
        columns={"a_id": "truth_id", "b_id": "recall_id", "overlap_bp": "orig_overlap_bp"}
    )
    if boundary_tolerance_bp > 0:
        truth_slop_bed = truth_bed.slop(
            g=str(genome_file or genome_sizes_path()), b=int(boundary_tolerance_bp)
        )
    else:
        truth_slop_bed = truth_bed
    slop_pairs = pairwise_wao(truth_slop_bed, recall_bed).rename(
        columns={"a_id": "truth_id", "b_id": "recall_id", "overlap_bp": "slop_overlap_bp"}
    )
    cpg_meta = None
    cpg_bed = None
    if wgbs_df is not None:
        cpg_meta = build_cpg_meta_df(wgbs_df)
        cpg_bed = cpg_meta_df_to_bedtool(cpg_meta)
    return {
        "truth_meta": truth_meta,
        "recall_meta": recall_meta,
        "truth_bed": truth_bed,
        "truth_slop_bed": truth_slop_bed,
        "recall_bed": recall_bed,
        "orig_pairs": orig_pairs,
        "slop_pairs": slop_pairs,
        "cpg_meta": cpg_meta,
        "cpg_bed": cpg_bed,
        "boundary_tolerance_bp": int(boundary_tolerance_bp),
        "genome_file": str(genome_file or genome_sizes_path()),
    }


def _score_from_overlap_state(overlap_state):
    truth_meta = overlap_state["truth_meta"]
    recall_meta = overlap_state["recall_meta"]
    orig_pairs = overlap_state["orig_pairs"]
    slop_pairs = overlap_state["slop_pairs"]
    boundary_tolerance_bp = overlap_state["boundary_tolerance_bp"]

    if slop_pairs.empty:
        pair_df = pd.DataFrame(
            columns=[
                "truth_id",
                "recall_id",
                "slop_overlap_bp",
                "truth_chrom",
                "truth_start",
                "truth_end",
                "truth_len",
                "recall_chrom",
                "recall_start",
                "recall_end",
                "recall_len",
                "orig_overlap_bp",
                "left_slop_bp",
                "right_slop_bp",
                "used_slop_bp",
                "denom",
                "truth_pct_overlap",
                "start_error_bp",
                "end_error_bp",
                "mabe_bp",
                "hit_any",
                "hit_majority",
                "hit_super_majority",
            ]
        )
    else:
        pair_df = (
            slop_pairs[["truth_id", "recall_id", "slop_overlap_bp"]]
            .merge(truth_meta, on="truth_id", how="left")
            .merge(recall_meta, on="recall_id", how="left")
            .merge(
                orig_pairs[["truth_id", "recall_id", "orig_overlap_bp"]],
                on=["truth_id", "recall_id"],
                how="left",
            )
        )
        pair_df["orig_overlap_bp"] = pair_df["orig_overlap_bp"].fillna(0).astype(int)
        tol = int(boundary_tolerance_bp)
        pair_df["left_slop_bp"] = (
            pair_df["truth_start"] - pair_df["recall_start"]
        ).clip(lower=-tol, upper=tol)
        pair_df["right_slop_bp"] = (
            pair_df["recall_end"] - pair_df["truth_end"]
        ).clip(lower=-tol, upper=tol)
        pair_df["used_slop_bp"] = pair_df["left_slop_bp"] + pair_df["right_slop_bp"]
        pair_df["denom"] = pair_df["truth_len"] + pair_df["used_slop_bp"]
        pair_df["truth_pct_overlap"] = np.where(
            pair_df["denom"] > 0,
            pair_df["slop_overlap_bp"] / pair_df["denom"],
            0.0,
        )
        pair_df["start_error_bp"] = (
            pair_df["recall_start"] - pair_df["truth_start"]
        ).abs()
        pair_df["end_error_bp"] = (pair_df["recall_end"] - pair_df["truth_end"]).abs()
        pair_df["mabe_bp"] = (
            pair_df["start_error_bp"] + pair_df["end_error_bp"]
        ) / 2.0
        pair_df["hit_any"] = pair_df["slop_overlap_bp"] >= 1
        pair_df["hit_majority"] = pair_df["truth_pct_overlap"] >= 0.50
        pair_df["hit_super_majority"] = pair_df["truth_pct_overlap"] >= 0.75

    return {
        "pair_detail": pair_df,
        "truth_meta": truth_meta,
        "recall_meta": recall_meta,
        "summary": {
            "any": summarize_region_resolution(
                pair_df, truth_meta["truth_id"].tolist(), recall_meta["recall_id"].tolist(), "hit_any"
            ),
            "majority": summarize_region_resolution(
                pair_df,
                truth_meta["truth_id"].tolist(),
                recall_meta["recall_id"].tolist(),
                "hit_majority",
            ),
            "super_majority": summarize_region_resolution(
                pair_df,
                truth_meta["truth_id"].tolist(),
                recall_meta["recall_id"].tolist(),
                "hit_super_majority",
            ),
        },
    }


def score(truth_df, predicted_df, boundary_tolerance_bp, genome_file=None):
    overlap_state = prepare_overlap_state(
        truth_df=truth_df,
        predicted_df=predicted_df,
        boundary_tolerance_bp=boundary_tolerance_bp,
        genome_file=genome_file,
    )
    return _score_from_overlap_state(overlap_state)


def summarize_region_resolution(pair_df, truth_ids, recall_ids, hit_col):
    n_truth = len(truth_ids)
    n_recalled = len(recall_ids)
    if pair_df.empty:
        return {
            "truth_hits": 0,
            "recalled_hits": 0,
            "recall": 0.0 if n_truth else 0.0,
            "precision": 0.0 if n_recalled else 0.0,
            "f1": 0.0,
            "per_region_recall_fraction": (np.nan if n_truth == 0 else 0.0),
            "mabe": np.nan,
        }

    eligible = pair_df.loc[pair_df[hit_col]].copy()
    truth_hits = eligible["truth_id"].nunique() if not eligible.empty else 0
    recall_hits = eligible["recall_id"].nunique() if not eligible.empty else 0
    recall = safe_divide(truth_hits, n_truth)
    precision = safe_divide(recall_hits, n_recalled)

    if n_truth == 0:
        per_region_recall_fraction = np.nan
        mabe = np.nan
    elif eligible.empty:
        per_region_recall_fraction = 0.0
        mabe = np.nan
    else:
        truth_order = {truth_id: idx for idx, truth_id in enumerate(truth_ids)}
        best_matches = (
            eligible.sort_values(
                ["truth_id", "truth_pct_overlap", "slop_overlap_bp", "recall_id"],
                ascending=[True, False, False, True],
            )
            .drop_duplicates(subset=["truth_id"], keep="first")
            .reset_index(drop=True)
        )
        per_truth_fraction = np.zeros(n_truth, dtype=float)
        for row in best_matches.itertuples(index=False):
            per_truth_fraction[truth_order[row.truth_id]] = min(
                1.0, float(row.truth_pct_overlap)
            )
        per_region_recall_fraction = float(per_truth_fraction.mean())
        mabe = float(best_matches["mabe_bp"].mean())

    return {
        "truth_hits": int(truth_hits),
        "recalled_hits": int(recall_hits),
        "recall": recall,
        "precision": precision,
        "f1": f1_score_from_precision_recall(precision, recall),
        "per_region_recall_fraction": per_region_recall_fraction,
        "mabe": mabe,
    }


def cpg_level_metrics(
    wgbs_df,
    truth_df,
    predicted_df,
    boundary_tolerance_bp,
    genome_file=None,
):
    overlap_state = prepare_overlap_state(
        truth_df=truth_df,
        predicted_df=predicted_df,
        boundary_tolerance_bp=boundary_tolerance_bp,
        genome_file=genome_file,
        wgbs_df=wgbs_df,
    )
    return _cpg_level_metrics_from_overlap_state(overlap_state)


def _cpg_level_metrics_from_overlap_state(overlap_state):
    cpgs = overlap_state["cpg_bed"]
    truth_bed = overlap_state["truth_bed"]
    predicted_bed = overlap_state["recall_bed"]
    truth_slop = overlap_state["truth_slop_bed"]
    boundary_tolerance_bp = overlap_state["boundary_tolerance_bp"]

    if boundary_tolerance_bp > 0:
        tol_band = truth_slop.subtract(truth_bed, A=False)
        scorable_cpgs = cpgs.subtract(tol_band, A=True)
    else:
        scorable_cpgs = cpgs

    truth_pos = scorable_cpgs.intersect(truth_bed, u=True)
    pred_pos = scorable_cpgs.intersect(predicted_bed, u=True)
    tp = truth_pos.intersect(predicted_bed, u=True)

    n_cpgs = int(scorable_cpgs.count())
    n_truth_pos = int(truth_pos.count())
    n_pred_pos = int(pred_pos.count())
    n_tp = int(tp.count())
    fp = int(n_pred_pos - n_tp)
    fn = int(n_truth_pos - n_tp)
    tn = int(n_cpgs - (n_truth_pos + n_pred_pos - n_tp))

    precision = safe_divide(n_tp, n_tp + fp)
    recall = safe_divide(n_tp, n_tp + fn)
    specificity = safe_divide(tn, tn + fp)
    accuracy = safe_divide(n_tp + tn, n_cpgs)
    return {
        "bp_tp": n_tp,
        "bp_fp": fp,
        "bp_fn": fn,
        "bp_tn": tn,
        "bp_precision": precision,
        "bp_recall": recall,
        "bp_f1": f1_score_from_precision_recall(precision, recall),
        "bp_jaccard": safe_divide(n_tp, n_tp + fp + fn),
        "bp_specificity": specificity,
        "bp_accuracy": accuracy,
    }


def compute_bp_metrics(
    wgbs_df, truth_df, predicted_df, boundary_tolerance_bp, genome_file=None
):
    return cpg_level_metrics(
        wgbs_df=wgbs_df,
        truth_df=truth_df,
        predicted_df=predicted_df,
        boundary_tolerance_bp=boundary_tolerance_bp,
        genome_file=genome_file,
    )


def compute_region_metrics(
    truth_df, predicted_df, boundary_tolerance_bp, threshold, genome_file=None
):
    score_result = score(
        truth_df=truth_df,
        predicted_df=predicted_df,
        boundary_tolerance_bp=boundary_tolerance_bp,
        genome_file=genome_file,
    )
    if threshold <= 0:
        return score_result["summary"]["any"]
    if threshold >= 0.75:
        return score_result["summary"]["super_majority"]
    return score_result["summary"]["majority"]


def _compute_fragmentation_absorption_from_pairs(pair_df, truth_ids, recall_ids):
    truth_counts = np.array([], dtype=float)
    if truth_ids:
        truth_counts = np.zeros(len(truth_ids), dtype=float)
        if not pair_df.empty:
            truth_order = {truth_id: idx for idx, truth_id in enumerate(truth_ids)}
            counts = pair_df.loc[pair_df["hit_any"]].groupby("truth_id")["recall_id"].nunique()
            for truth_id, count in counts.items():
                truth_counts[truth_order[truth_id]] = float(count)
        absolute_fragmentation = float(truth_counts.mean())
        recalled_truth_counts = truth_counts[truth_counts > 0]
        fragmentation = (
            float(recalled_truth_counts.mean()) if recalled_truth_counts.size else np.nan
        )
    else:
        fragmentation = np.nan
        absolute_fragmentation = np.nan

    recall_counts = np.array([], dtype=float)
    if recall_ids:
        recall_counts = np.zeros(len(recall_ids), dtype=float)
        if not pair_df.empty:
            recall_order = {recall_id: idx for idx, recall_id in enumerate(recall_ids)}
            counts = pair_df.loc[pair_df["hit_any"]].groupby("recall_id")["truth_id"].nunique()
            for recall_id, count in counts.items():
                recall_counts[recall_order[recall_id]] = float(count)
        absolute_absorption = float(recall_counts.mean())
        overlapping_recall_counts = recall_counts[recall_counts > 0]
        absorption = (
            float(overlapping_recall_counts.mean())
            if overlapping_recall_counts.size
            else np.nan
        )
    else:
        absorption = np.nan
        absolute_absorption = np.nan

    return {
        "fragmentation": fragmentation,
        "absorption": absorption,
        "fragmentation_distance_from_1": abs(fragmentation - 1.0)
        if not pd.isna(fragmentation)
        else np.nan,
        "absorption_distance_from_1": abs(absorption - 1.0)
        if not pd.isna(absorption)
        else np.nan,
        "absolute_fragmentation": absolute_fragmentation,
        "absolute_absorption": absolute_absorption,
        "absolute_fragmentation_distance_from_1": abs(absolute_fragmentation - 1.0)
        if not pd.isna(absolute_fragmentation)
        else np.nan,
        "absolute_absorption_distance_from_1": abs(absolute_absorption - 1.0)
        if not pd.isna(absolute_absorption)
        else np.nan,
    }


def _false_positive_regions_from_overlap_state(overlap_state):
    recall_meta_df = overlap_state["recall_meta"]
    recall_bed = overlap_state["recall_bed"]
    truth_slop_bed = overlap_state["truth_slop_bed"]
    if recall_meta_df.empty:
        return pd.DataFrame(columns=["chrom", "start", "end"])
    false_bed = recall_bed.intersect(truth_slop_bed, v=True)
    if false_bed.count() == 0:
        return pd.DataFrame(columns=["chrom", "start", "end"])
    false_df = false_bed.to_dataframe(
        names=["chrom", "start", "end", "recall_id", "recall_len"]
    )
    for col in ["start", "end", "recall_len"]:
        false_df[col] = pd.to_numeric(false_df[col], errors="coerce").astype(int)
    return false_df[["chrom", "start", "end"]].reset_index(drop=True)


def compute_false_region_betas(wgbs_df, false_region_df):
    if false_region_df.empty:
        return pd.DataFrame(
            columns=[
                "chrom",
                "start",
                "end",
                "region_length_bp",
                "n_region_cpgs",
                "mean_beta",
            ]
        )
    false_meta = build_tagged_interval_df(
        false_region_df, prefix="false_region", id_prefix="F"
    )
    if false_meta.empty:
        return pd.DataFrame(
            columns=[
                "chrom",
                "start",
                "end",
                "region_length_bp",
                "n_region_cpgs",
                "mean_beta",
            ]
        )
    false_bed = tagged_df_to_bedtool(false_meta, prefix="false_region")
    cpg_meta = build_cpg_meta_df(wgbs_df)
    if cpg_meta.empty:
        result_df = false_meta.rename(
            columns={
                "false_region_chrom": "chrom",
                "false_region_start": "start",
                "false_region_end": "end",
                "false_region_len": "region_length_bp",
            }
        )[["chrom", "start", "end", "region_length_bp"]].copy()
        result_df["n_region_cpgs"] = 0
        result_df["mean_beta"] = np.nan
        return result_df.reset_index(drop=True)

    cpg_bed_df = cpg_meta.rename(
        columns={
            "cpg_chrom": "chrom",
            "cpg_start": "start",
            "cpg_end": "end",
            "cpg_id": "name",
        }
    )[["chrom", "start", "end", "name"]].copy()
    cpg_bed_df["score"] = 1
    cpg_bed = BedTool.from_dataframe(cpg_bed_df[["chrom", "start", "end", "name", "score"]])

    overlap_df = pairwise_wao(false_bed, cpg_bed).rename(
        columns={"a_id": "false_region_id", "b_id": "cpg_id"}
    )
    result_df = false_meta.rename(
        columns={
            "false_region_chrom": "chrom",
            "false_region_start": "start",
            "false_region_end": "end",
            "false_region_id": "false_region_id",
            "false_region_len": "region_length_bp",
        }
    )[["false_region_id", "chrom", "start", "end", "region_length_bp"]].copy()
    if overlap_df.empty:
        result_df["n_region_cpgs"] = 0
        result_df["mean_beta"] = np.nan
        return result_df.drop(columns=["false_region_id"]).reset_index(drop=True)
    overlap_df = overlap_df.merge(
        cpg_meta[["cpg_id", "cpg_beta"]],
        on="cpg_id",
        how="left",
    )
    summary_df = (
        overlap_df.groupby("false_region_id", as_index=False)
        .agg(
            n_region_cpgs=("cpg_id", "nunique"),
            mean_beta=("cpg_beta", "mean"),
        )
    )
    result_df = result_df.merge(summary_df, on="false_region_id", how="left")
    result_df["n_region_cpgs"] = (
        pd.to_numeric(result_df["n_region_cpgs"], errors="coerce").fillna(0).astype(int)
    )
    return result_df.drop(columns=["false_region_id"]).reset_index(drop=True)


def compute_fragmentation_absorption(
    truth_df, predicted_df, boundary_tolerance_bp, genome_file=None
):
    overlap_state = prepare_overlap_state(
        truth_df=truth_df,
        predicted_df=predicted_df,
        boundary_tolerance_bp=boundary_tolerance_bp,
        genome_file=genome_file,
    )
    score_result = _score_from_overlap_state(overlap_state)
    return _compute_fragmentation_absorption_from_pairs(
        score_result["pair_detail"],
        score_result["truth_meta"]["truth_id"].tolist(),
        score_result["recall_meta"]["recall_id"].tolist(),
    )


def false_positive_regions(
    truth_df, predicted_df, boundary_tolerance_bp, genome_file=None
):
    overlap_state = prepare_overlap_state(
        truth_df=truth_df,
        predicted_df=predicted_df,
        boundary_tolerance_bp=boundary_tolerance_bp,
        genome_file=genome_file,
    )
    return _false_positive_regions_from_overlap_state(overlap_state)


def metric_column_names():
    metric_cols = [
        "bp_precision",
        "bp_recall",
        "bp_f1",
        "bp_jaccard",
        "bp_per_region_recall_fraction",
        "bp_mabe",
        "bp_accuracy",
    ]
    for prefix in ["region_any", "region_majority", "region_super_majority"]:
        metric_cols.extend(
            [
                f"{prefix}_precision",
                f"{prefix}_recall",
                f"{prefix}_f1",
                f"{prefix}_per_region_recall_fraction",
                f"{prefix}_mabe",
            ]
        )
    metric_cols.extend(
        [
            "fragmentation",
            "absorption",
            "fragmentation_distance_from_1",
            "absorption_distance_from_1",
            "absolute_fragmentation",
            "absolute_absorption",
            "absolute_fragmentation_distance_from_1",
            "absolute_absorption_distance_from_1",
            "avg_false_pmd_beta",
            "mean_false_pmds_called",
        ]
    )
    return metric_cols


def compute_sample_tool_recovery_metrics(
    sample_id,
    tool_name,
    truth_df,
    predicted_df,
    wgbs_df,
    boundary_tolerance_bp=10_000,
    genome_file=None,
):
    truth_df = ensure_interval_df(truth_df)
    predicted_df = ensure_interval_df(predicted_df)
    print(f"Computing metrics for sample {sample_id} with tool {tool_name}...")
    print("Preparing overlap state...")
    overlap_state = prepare_overlap_state(
        truth_df=truth_df,
        predicted_df=predicted_df,
        boundary_tolerance_bp=boundary_tolerance_bp,
        genome_file=genome_file,
        wgbs_df=wgbs_df,
    )
    print("Computing scores from overlap state...")
    score_result = _score_from_overlap_state(overlap_state)
    pair_df = score_result["pair_detail"]
    truth_meta = score_result["truth_meta"]
    recall_meta = score_result["recall_meta"]
    any_metrics = score_result["summary"]["any"]
    majority_metrics = score_result["summary"]["majority"]
    super_majority_metrics = score_result["summary"]["super_majority"]
    print("Computing base-pair level metrics...")
    bp_metrics = _cpg_level_metrics_from_overlap_state(overlap_state)
    print("Computing fragmentation and absorption metrics...")
    global_metrics = _compute_fragmentation_absorption_from_pairs(
        pair_df, truth_meta["truth_id"].tolist(), recall_meta["recall_id"].tolist()
    )
    print("Computing false positive region betas...")
    false_region_df = _false_positive_regions_from_overlap_state(overlap_state)
    false_region_betas_df = compute_false_region_betas(wgbs_df, false_region_df)
    region_metric_keys = [
        "precision",
        "recall",
        "f1",
        "per_region_recall_fraction",
        "mabe",
    ]

    metric_row = {
        "synthetic_sample_id": sample_id,
        "tool": tool_name,
        "boundary_tolerance_bp": int(boundary_tolerance_bp),
        "n_truth_regions": int(len(truth_df)),
        "n_predicted_regions": int(len(predicted_df)),
        **bp_metrics,
        "bp_per_region_recall_fraction": any_metrics["per_region_recall_fraction"],
        "bp_mabe": any_metrics["mabe"],
        **{
            f"region_any_{key}": any_metrics[key]
            for key in region_metric_keys
        },
        **{
            f"region_majority_{key}": majority_metrics[key]
            for key in region_metric_keys
        },
        **{
            f"region_super_majority_{key}": super_majority_metrics[key]
            for key in region_metric_keys
        },
        **global_metrics,
        "avg_false_pmd_beta": (
            float(false_region_betas_df["mean_beta"].mean())
            if not false_region_betas_df.empty
            else np.nan
        ),
        "mean_false_pmds_called": int(len(false_region_df)),
    }
    return metric_row, false_region_betas_df


def _metric_tool_worker(kwargs):
    metric_row, false_region_betas_df = compute_sample_tool_recovery_metrics(
        sample_id=kwargs["sample_id"],
        tool_name=kwargs["tool_name"],
        truth_df=kwargs["truth_df"],
        predicted_df=kwargs["predicted_df"],
        wgbs_df=kwargs["wgbs_df"],
        boundary_tolerance_bp=kwargs["boundary_tolerance_bp"],
        genome_file=kwargs["genome_file"],
    )
    metric_row["truth_bed"] = kwargs["truth_bed"]
    metric_row["synthetic_file"] = kwargs["synthetic_file"]
    if not false_region_betas_df.empty:
        false_region_betas_df = false_region_betas_df.copy()
        false_region_betas_df["synthetic_sample_id"] = kwargs["sample_id"]
        false_region_betas_df["tool"] = kwargs["tool_name"]
    return metric_row, false_region_betas_df


def empty_false_beta_df():
    return pd.DataFrame(
        columns=[
            "chrom",
            "start",
            "end",
            "region_length_bp",
            "n_region_cpgs",
            "mean_beta",
            "synthetic_sample_id",
            "tool",
        ]
    )


def empty_tool_status_df():
    return pd.DataFrame(
        columns=[
            "synthetic_sample_id",
            "tool",
            "tool_output_path",
            "tool_status",
            "tool_status_reason",
        ]
    )


def sample_metric_shard_paths(metrics_output_dir: Path, sample_id: str):
    shard_dir = Path(metrics_output_dir) / "shards"
    return {
        "dir": shard_dir,
        "per_sample": shard_dir / f"per_sample_tool_metrics.{sample_id}.tsv",
        "false_beta": shard_dir / f"false_pmd_beta_by_region.{sample_id}.tsv",
        "sample_status": shard_dir / f"metric_sample_status.{sample_id}.tsv",
        "tool_status": shard_dir / f"metric_tool_status.{sample_id}.tsv",
    }


def compute_single_sample_metrics(
    record,
    tool_results_dir: Path,
    tools_info_df=None,
    boundary_tolerance_bp=10_000,
    n_tool_procs=1,
):
    tool_results_dir = Path(tool_results_dir)
    tools_info_df = (
        tools_info_df
        if tools_info_df is not None
        else pd.DataFrame(HEALTHY_PMD_TOOL_REGISTRY)
    )
    sample_id = record["synthetic_sample_id"]
    parsed_tools, tool_status_df = collect_sample_tool_regions(
        sample_id, tool_results_dir, tools_info_df
    )
    if tool_status_df.empty:
        tool_status_df = empty_tool_status_df()
    if not parsed_tools:
        return {
            "synthetic_sample_id": sample_id,
            "per_sample_df": pd.DataFrame(),
            "false_beta_df": empty_false_beta_df(),
            "sample_status_df": pd.DataFrame(
                [
                    {
                        "synthetic_sample_id": sample_id,
                        "sample_status": "not_called",
                        "sample_status_reason": "no expected tool outputs could be parsed",
                        "n_parsed_tools": 0,
                    }
                ]
            ),
            "tool_status_df": tool_status_df,
        }

    truth_df = load_truth_table(record["truth_bed"])
    wgbs_df = load_sample_wgbs_from_shared_prep(sample_id, tool_results_dir)
    genome_file = genome_sizes_path(record.get("source_genome"))
    worker_kwargs = [
        {
            "sample_id": sample_id,
            "tool_name": tool_name,
            "truth_df": truth_df,
            "predicted_df": predicted_df,
            "wgbs_df": wgbs_df,
            "boundary_tolerance_bp": boundary_tolerance_bp,
            "genome_file": genome_file,
            "truth_bed": record["truth_bed"],
            "synthetic_file": record["synthetic_file"],
        }
        for tool_name, predicted_df in parsed_tools.items()
    ]
    per_sample_rows = []
    false_beta_rows = []
    n_tool_procs = int(n_tool_procs or 1)
    if n_tool_procs <= 1 or len(worker_kwargs) <= 1:
        tool_results = [_metric_tool_worker(kwargs) for kwargs in worker_kwargs]
    else:
        max_workers = min(n_tool_procs, len(worker_kwargs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_metric_tool_worker, kwargs) for kwargs in worker_kwargs]
            tool_results = [future.result() for future in as_completed(futures)]
    for metric_row, false_region_betas_df in tool_results:
        per_sample_rows.append(metric_row)
        if not false_region_betas_df.empty:
            false_beta_rows.append(false_region_betas_df)

    per_sample_df = pd.DataFrame(per_sample_rows)
    if not per_sample_df.empty:
        per_sample_df = sort_tools_methylseg_first(per_sample_df)
    false_beta_df = (
        pd.concat(false_beta_rows, ignore_index=True)
        if false_beta_rows
        else empty_false_beta_df()
    )
    sample_status_df = pd.DataFrame(
        [
            {
                "synthetic_sample_id": sample_id,
                "sample_status": "called",
                "sample_status_reason": "",
                "n_parsed_tools": int(len(parsed_tools)),
            }
        ]
    )
    return {
        "synthetic_sample_id": sample_id,
        "per_sample_df": per_sample_df,
        "false_beta_df": false_beta_df,
        "sample_status_df": sample_status_df,
        "tool_status_df": tool_status_df,
    }


def write_metric_sample_shards(sample_result, metrics_output_dir: Path):
    sample_id = sample_result["synthetic_sample_id"]
    shard_paths = sample_metric_shard_paths(metrics_output_dir, sample_id)
    shard_paths["dir"].mkdir(parents=True, exist_ok=True)
    sample_result["per_sample_df"].to_csv(shard_paths["per_sample"], sep="\t", index=False)
    sample_result["false_beta_df"].to_csv(shard_paths["false_beta"], sep="\t", index=False)
    sample_result["sample_status_df"].to_csv(
        shard_paths["sample_status"], sep="\t", index=False
    )
    sample_result["tool_status_df"].to_csv(shard_paths["tool_status"], sep="\t", index=False)
    return shard_paths


def _metric_sample_worker(kwargs):
    try:
        return compute_single_sample_metrics(**kwargs)
    except Exception as exc:
        sample_id = kwargs["record"]["synthetic_sample_id"]
        return {
            "synthetic_sample_id": sample_id,
            "per_sample_df": pd.DataFrame(),
            "false_beta_df": empty_false_beta_df(),
            "sample_status_df": pd.DataFrame(
                [
                    {
                        "synthetic_sample_id": sample_id,
                        "sample_status": "failed",
                        "sample_status_reason": str(exc),
                        "n_parsed_tools": 0,
                    }
                ]
            ),
            "tool_status_df": empty_tool_status_df(),
        }


def _missing_sample_result(sample_id: str, tools_info_df=None):
    tools_info_df = (
        tools_info_df
        if tools_info_df is not None
        else pd.DataFrame(HEALTHY_PMD_TOOL_REGISTRY)
    )
    tool_status_df = pd.DataFrame(
        [
            {
                "synthetic_sample_id": sample_id,
                "tool": row["tool"],
                "tool_output_path": "",
                "tool_status": "missing_shard",
                "tool_status_reason": "metric shard files are missing",
            }
            for row in tools_info_df.to_dict("records")
        ]
    )
    return {
        "synthetic_sample_id": sample_id,
        "per_sample_df": pd.DataFrame(),
        "false_beta_df": empty_false_beta_df(),
        "sample_status_df": pd.DataFrame(
            [
                {
                    "synthetic_sample_id": sample_id,
                    "sample_status": "missing",
                    "sample_status_reason": "metric shard files are missing",
                    "n_parsed_tools": 0,
                }
            ]
        ),
        "tool_status_df": tool_status_df,
    }


def load_metric_sample_shards(sample_id: str, metrics_output_dir: Path, tools_info_df=None):
    shard_paths = sample_metric_shard_paths(metrics_output_dir, sample_id)
    required_paths = [shard_paths["sample_status"], shard_paths["tool_status"]]
    if not all(path.exists() for path in required_paths):
        return _missing_sample_result(sample_id, tools_info_df=tools_info_df)
    per_sample_df = (
        pd.read_csv(shard_paths["per_sample"], sep="\t")
        if shard_paths["per_sample"].exists() and shard_paths["per_sample"].stat().st_size > 0
        else pd.DataFrame()
    )
    false_beta_df = (
        pd.read_csv(shard_paths["false_beta"], sep="\t")
        if shard_paths["false_beta"].exists() and shard_paths["false_beta"].stat().st_size > 0
        else empty_false_beta_df()
    )
    sample_status_df = pd.read_csv(shard_paths["sample_status"], sep="\t")
    tool_status_df = pd.read_csv(shard_paths["tool_status"], sep="\t")
    return {
        "synthetic_sample_id": sample_id,
        "per_sample_df": per_sample_df,
        "false_beta_df": false_beta_df,
        "sample_status_df": sample_status_df,
        "tool_status_df": tool_status_df,
    }


def finalize_metric_outputs(sample_results, metrics_output_dir: Path):
    metrics_output_dir = Path(metrics_output_dir)
    metrics_output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_frames = [
        result["per_sample_df"]
        for result in sample_results
        if not result["per_sample_df"].empty
    ]
    false_beta_frames = [
        result["false_beta_df"]
        for result in sample_results
        if not result["false_beta_df"].empty
    ]
    sample_status_frames = [result["sample_status_df"] for result in sample_results]
    tool_status_frames = [result["tool_status_df"] for result in sample_results]

    if not per_sample_frames:
        sample_status_df = pd.concat(sample_status_frames, ignore_index=True)
        sample_status_df = sample_status_df.sort_values("synthetic_sample_id").reset_index(
            drop=True
        )
        sample_status_df.to_csv(
            metrics_output_dir / "metric_sample_status.tsv", sep="\t", index=False
        )
        raise RuntimeError("No called synthetic samples had parseable recovery outputs.")

    per_sample_df = pd.concat(per_sample_frames, ignore_index=True)
    per_sample_df = sort_tools_methylseg_first(
        per_sample_df.sort_values("synthetic_sample_id").reset_index(drop=True)
    )
    per_tool_summary_df = write_per_tool_summary(per_sample_df)
    false_beta_df = (
        pd.concat(false_beta_frames, ignore_index=True)
        if false_beta_frames
        else empty_false_beta_df()
    )
    sample_status_df = pd.concat(sample_status_frames, ignore_index=True)
    sample_status_df = sample_status_df.sort_values("synthetic_sample_id").reset_index(
        drop=True
    )
    tool_status_df = (
        pd.concat(tool_status_frames, ignore_index=True)
        if tool_status_frames
        else empty_tool_status_df()
    )

    per_sample_path = metrics_output_dir / "per_sample_tool_metrics.tsv"
    per_tool_path = metrics_output_dir / "per_tool_summary.tsv"
    false_beta_path = metrics_output_dir / "false_pmd_beta_by_region.tsv"
    sample_status_path = metrics_output_dir / "metric_sample_status.tsv"
    tool_status_path = metrics_output_dir / "metric_tool_status.tsv"
    per_sample_df.to_csv(per_sample_path, sep="\t", index=False)
    per_tool_summary_df.to_csv(per_tool_path, sep="\t", index=False)
    false_beta_df.to_csv(false_beta_path, sep="\t", index=False)
    sample_status_df.to_csv(sample_status_path, sep="\t", index=False)
    tool_status_df.to_csv(tool_status_path, sep="\t", index=False)

    return {
        "per_sample_df": per_sample_df,
        "per_tool_summary_df": per_tool_summary_df,
        "false_beta_df": false_beta_df,
        "sample_status_df": sample_status_df,
        "tool_status_df": tool_status_df,
        "per_sample_path": per_sample_path,
        "per_tool_path": per_tool_path,
        "false_beta_path": false_beta_path,
        "sample_status_path": sample_status_path,
        "tool_status_path": tool_status_path,
    }


def write_per_tool_summary(per_sample_df):
    metric_cols = [col for col in metric_column_names() if col in per_sample_df.columns]
    summary_rows = []
    for tool_name, tool_df in per_sample_df.groupby("tool", sort=False):
        row = {"tool": tool_name, "n_called_samples": int(len(tool_df))}
        for metric in metric_cols:
            values = pd.to_numeric(tool_df[metric], errors="coerce")
            valid_values = values.dropna()
            row[f"{metric}_mean"] = (
                float(valid_values.mean()) if not valid_values.empty else np.nan
            )
            row[f"{metric}_median"] = (
                float(valid_values.median()) if not valid_values.empty else np.nan
            )
            row[f"{metric}_std"] = (
                float(valid_values.std()) if len(valid_values) > 1 else np.nan
            )
            row[f"{metric}_min"] = (
                float(valid_values.min()) if not valid_values.empty else np.nan
            )
            row[f"{metric}_max"] = (
                float(valid_values.max()) if not valid_values.empty else np.nan
            )
        summary_rows.append(row)
    return sort_tools_methylseg_first(pd.DataFrame(summary_rows))


def tool_output_path(sample_id, tool_row, tool_results_dir: Path):
    return tool_results_dir / "/".join(tool_row["path_parts"]).replace(
        "{sample}", str(sample_id)
    )


def read_tool_regions_for_metrics(sample_id, tool_row, tool_results_dir: Path):
    regions_path = tool_output_path(sample_id, tool_row, tool_results_dir)
    columns = tool_row["columns"]
    try:
        regions_df = pd.read_csv(
            regions_path,
            sep="\t",
            header=0 if tool_row["header"] else None,
            index_col=False,
            names=columns,
        )
    except EmptyDataError:
        regions_df = pd.DataFrame(columns=columns)
    regions_df = ensure_interval_df(regions_df[["chrom", "start", "end"]])
    return regions_df, regions_path


def collect_sample_tool_regions(sample_id, tool_results_dir: Path, tools_info_df):
    parsed = {}
    statuses = []
    for tool_row in tools_info_df.to_dict("records"):
        tool_name = tool_row["tool"]
        path = tool_output_path(sample_id, tool_row, tool_results_dir)
        if not path.exists():
            statuses.append(
                {
                    "synthetic_sample_id": sample_id,
                    "tool": tool_name,
                    "tool_output_path": str(path),
                    "tool_status": "missing",
                    "tool_status_reason": "expected output file is missing",
                }
            )
            continue
        try:
            regions_df, _ = read_tool_regions_for_metrics(
                sample_id, tool_row, tool_results_dir
            )
            parsed[tool_name] = regions_df
            statuses.append(
                {
                    "synthetic_sample_id": sample_id,
                    "tool": tool_name,
                    "tool_output_path": str(path),
                    "tool_status": "parsed",
                    "tool_status_reason": "",
                }
            )
        except Exception as exc:
            statuses.append(
                {
                    "synthetic_sample_id": sample_id,
                    "tool": tool_name,
                    "tool_output_path": str(path),
                    "tool_status": "failed_parse",
                    "tool_status_reason": str(exc),
                }
            )
    return parsed, pd.DataFrame(statuses)


def run_metrics(
    injected_manifest,
    tool_results_dir: Path,
    metrics_output_dir: Path,
    tools_info_df=None,
    boundary_tolerance_bp=10_000,
    n_sample_procs=1,
    n_tool_procs=1,
):
    manifest_df = (
        pd.read_csv(injected_manifest, sep="\t")
        if not isinstance(injected_manifest, pd.DataFrame)
        else injected_manifest.copy()
    )
    tools_info_df = (
        tools_info_df
        if tools_info_df is not None
        else pd.DataFrame(HEALTHY_PMD_TOOL_REGISTRY)
    )
    metrics_output_dir.mkdir(parents=True, exist_ok=True)
    tool_results_dir = Path(tool_results_dir)
    worker_kwargs = [
        {
            "record": record,
            "tool_results_dir": tool_results_dir,
            "tools_info_df": tools_info_df,
            "boundary_tolerance_bp": boundary_tolerance_bp,
            "n_tool_procs": n_tool_procs,
        }
        for record in manifest_df.to_dict("records")
    ]
    if n_sample_procs is None:
        n_sample_procs = 1
    n_sample_procs = int(n_sample_procs)
    if n_sample_procs <= 1:
        sample_results = [_metric_sample_worker(kwargs) for kwargs in worker_kwargs]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(n_sample_procs, len(worker_kwargs))) as pool:
            sample_results = list(pool.imap_unordered(_metric_sample_worker, worker_kwargs))
    return finalize_metric_outputs(sample_results, metrics_output_dir)


def finalize_metrics_from_shards(
    injected_manifest,
    metrics_output_dir: Path,
    tools_info_df=None,
):
    manifest_df = (
        pd.read_csv(injected_manifest, sep="\t")
        if not isinstance(injected_manifest, pd.DataFrame)
        else injected_manifest.copy()
    )
    tools_info_df = (
        tools_info_df
        if tools_info_df is not None
        else pd.DataFrame(HEALTHY_PMD_TOOL_REGISTRY)
    )
    sample_results = [
        load_metric_sample_shards(
            record["synthetic_sample_id"],
            metrics_output_dir,
            tools_info_df=tools_info_df,
        )
        for record in manifest_df.sort_values("synthetic_sample_id").to_dict("records")
    ]
    return finalize_metric_outputs(sample_results, metrics_output_dir)


def load_synthetic_recovery_artifacts(metrics_output_dir: Path):
    metrics_output_dir = Path(metrics_output_dir)
    sample_status_path = metrics_output_dir / "metric_sample_status.tsv"
    tool_status_path = metrics_output_dir / "metric_tool_status.tsv"
    return {
        "per_sample_df": pd.read_csv(
            metrics_output_dir / "per_sample_tool_metrics.tsv", sep="\t"
        ),
        "per_tool_summary_df": pd.read_csv(
            metrics_output_dir / "per_tool_summary.tsv", sep="\t"
        ),
        "false_beta_df": pd.read_csv(
            metrics_output_dir / "false_pmd_beta_by_region.tsv", sep="\t"
        ),
        "sample_status_df": (
            pd.read_csv(sample_status_path, sep="\t")
            if sample_status_path.exists()
            else pd.DataFrame()
        ),
        "tool_status_df": (
            pd.read_csv(tool_status_path, sep="\t")
            if tool_status_path.exists()
            else pd.DataFrame()
        ),
    }
