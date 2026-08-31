from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / "repo_paths.py").exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise RuntimeError("Could not locate repo root from script path.")
    PROJECT_ROOT = PROJECT_ROOT.parent

CHROMATIN_ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "03_chromatin_analysis"
REGION_CALLING_UTILS_DIR = (
    PROJECT_ROOT / "analysis" / "01_region_calling_analysis" / "utils"
)
for import_path in [PROJECT_ROOT, CHROMATIN_ANALYSIS_DIR, REGION_CALLING_UTILS_DIR]:
    import_str = str(import_path)
    if import_str not in sys.path:
        sys.path.insert(0, import_str)

from repo_paths import REGION_CALLING_ANALYSIS_DIR, REGION_CALLING_RESULTS_DIR
from methyl_tool_comparator import SharedPrepManager
from chromatin_analysis_utils import (
    clean_interval_df,
    configure_pybedtools,
    filter_regions_for_deeptools,
    get_deeptools_processor_count,
    get_eligible_chrom_sizes,
    run_command,
    sample_to_sample_id,
    write_bed,
)
from figures.utils.figures_utils import (
    REGION_CALLING_TOOL_LABELS,
    REGION_CALLING_TOOL_ORDER,
    TOOL_REGISTRY as FIGURE_TOOL_REGISTRY,
    _canonical_tool_name,
    _load_bed_gz_sample,
    _load_beta_sample,
    _load_chrom_sizes,
    _write_bigwig,
    load_tool_regions,
    region_type_for_tool,
)


DEFAULT_CONFIGS_PATH = (
    REGION_CALLING_ANALYSIS_DIR / "slurm_code" / "configs.txt"
)
DEFAULT_OUTPUT_DIR = REGION_CALLING_ANALYSIS_DIR / "out" / "methylation_deeptools"
DEFAULT_WGBS_COMPUTE_MATRIX_BIN_SIZE = 10_000
DEFAULT_ARRAY_COMPUTE_MATRIX_BIN_SIZE = 50_000
DEFAULT_REGION_BODY_LENGTH = 500_000
DEFAULT_WGBS_FLANK_LENGTH = 250_000
DEFAULT_ARRAY_FLANK_LENGTH = 250_000
DEFAULT_AVERAGE_TYPE_BINS = "mean"
AVERAGE_TYPE_BINS_CHOICES = ["mean", "median", "min", "max", "std", "sum"]

CANONICAL_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
ARRAY_SIGNAL_TOOLS = {"methylseg_hm450k", "dnmtools_array"}
FIGURE_TOOL_REGISTRY_BY_NAME = {
    tool_config["tool"]: tool_config for tool_config in FIGURE_TOOL_REGISTRY
}
TOOL_PLATFORM_BY_NAME = {
    tool_name: tool_config["platform"]
    for tool_name, tool_config in FIGURE_TOOL_REGISTRY_BY_NAME.items()
}
REQUIRED_TOOL_SET = set(REGION_CALLING_TOOL_ORDER)


def collapse_duplicate_beta_intervals(beta_df: pd.DataFrame) -> pd.DataFrame:
    duplicate_mask = beta_df.duplicated(["chrom", "start", "end"], keep=False)
    if not duplicate_mask.any():
        return beta_df.reset_index(drop=True)

    collapsed_df = (
        beta_df.groupby(["chrom", "start", "end"], as_index=False, sort=False)["beta"]
        .mean()
        .reset_index(drop=True)
    )
    return collapsed_df


def _available_cpu_count() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(int(slurm_cpus), 1)
        except ValueError:
            pass
    return max(os.cpu_count() or 1, 1)


def run_parallel(tasks, worker, max_workers, stage_name):
    tasks = list(tasks)
    if not tasks:
        return []

    max_workers = min(max_workers, len(tasks))
    if max_workers <= 1:
        print(f"Running {stage_name} sequentially across {len(tasks)} task(s).")
        return [worker(task) for task in tasks]

    print(
        f"Running {stage_name} with {max_workers} workers across {len(tasks)} task(s)."
    )
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(worker, tasks))
    except Exception as exc:
        print(
            f"Falling back to sequential execution for {stage_name} "
            f"because multiprocessing failed: {exc}"
        )
        return [worker(task) for task in tasks]


def ensure_compute_matrix_command() -> str:
    compute_matrix = shutil.which("computeMatrix")
    if compute_matrix is None:
        env_compute_matrix = Path(sys.executable).resolve().parent / "computeMatrix"
        if env_compute_matrix.exists():
            compute_matrix = str(env_compute_matrix)
    if compute_matrix is None:
        raise RuntimeError(
            "computeMatrix was not found on PATH. Start this script from the "
            "jt_wgbs_analysis environment or add deepTools to PATH."
        )
    return compute_matrix


def maybe_run_command(command, expected_outputs, force=False):
    expected_paths = [Path(path) for path in expected_outputs]
    if not force and expected_paths and all(path.exists() for path in expected_paths):
        return True
    run_command(command, expected_outputs=expected_paths)
    return False


def join_manifest_values(values) -> str:
    return "|".join(str(value) for value in values)


def split_manifest_values(value) -> list[str]:
    if value is None:
        return []
    value = str(value).strip()
    if not value:
        return []
    return [item for item in value.split("|") if item]


def build_sample_group_key(samples) -> str:
    return "__".join(str(sample) for sample in samples)


def validate_selected_tools(selected_tools: list[str]) -> list[str]:
    deduped_tools = list(dict.fromkeys(selected_tools))
    invalid = [tool for tool in deduped_tools if tool not in REQUIRED_TOOL_SET]
    if invalid:
        raise ValueError(
            "Unsupported selected tools: " + ", ".join(invalid)
            + ". Choose from: " + ", ".join(REGION_CALLING_TOOL_ORDER)
        )
    if not deduped_tools:
        raise ValueError("At least one tool must be selected.")
    return deduped_tools


def read_config_paths(configs_path: Path) -> list[Path]:
    configs_path = Path(configs_path).expanduser().resolve()
    if not configs_path.exists():
        raise FileNotFoundError(f"Config manifest not found: {configs_path}")
    config_paths = []
    for raw_line in configs_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        config_path = Path(line).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file listed in {configs_path} does not exist: {config_path}"
            )
        config_paths.append(config_path)
    if not config_paths:
        raise ValueError(f"No config paths were found in {configs_path}")
    return config_paths


def load_sample_configs(configs_path: Path, selected_samples=None) -> pd.DataFrame:
    selected_sample_set = (
        None if selected_samples is None else {str(sample) for sample in selected_samples}
    )
    rows = []
    for config_order, config_path in enumerate(read_config_paths(configs_path)):
        with open(config_path) as handle:
            config = yaml.safe_load(handle) or {}
        sample = str(config.get("sample", "")).strip()
        meth_file = str(config.get("meth_file", "")).strip()
        genome = str(config.get("genome", "")).strip()
        if not sample or not meth_file or not genome:
            raise ValueError(
                f"Config is missing one of sample/meth_file/genome: {config_path}"
            )
        if selected_sample_set is not None and sample not in selected_sample_set:
            continue
        rows.append(
            {
                "config_order": config_order,
                "config_path": str(config_path),
                "sample": sample,
                "sample_id": sample_to_sample_id(sample),
                "meth_file": str(Path(meth_file).expanduser().resolve()),
                "genome": genome,
            }
        )
    if not rows:
        raise ValueError("No sample configs matched the current selection.")
    return pd.DataFrame(rows).sort_values("config_order").reset_index(drop=True)


def ensure_wgbstools_if_needed(sample_configs_df: pd.DataFrame) -> None:
    requires_wgbstools = sample_configs_df["meth_file"].astype(str).str.endswith(".beta").any()
    if requires_wgbstools and shutil.which("wgbstools") is None:
        raise RuntimeError(
            "Selected samples include .beta methylation files, but `wgbstools` "
            "was not found on PATH."
        )


def required_signal_tracks(selected_tools: list[str]) -> list[str]:
    signal_tracks = set()
    if any(tool not in ARRAY_SIGNAL_TOOLS for tool in selected_tools):
        signal_tracks.add("wgbs")
    if any(tool in ARRAY_SIGNAL_TOOLS for tool in selected_tools):
        signal_tracks.add("hm450k")
    return [track for track in ["wgbs", "hm450k"] if track in signal_tracks]


def signal_track_for_tool(tool: str) -> str:
    return "hm450k" if tool in ARRAY_SIGNAL_TOOLS else "wgbs"


def signal_mode_for_track(signal_track: str) -> str:
    if signal_track == "hm450k":
        return "probe_level"
    if signal_track == "wgbs":
        return "cpg_level"
    raise ValueError(f"Unsupported signal track: {signal_track}")


def flank_length_for_tool(tool: str, wgbs_flank_length: int, array_flank_length: int) -> int:
    return int(array_flank_length) if tool in ARRAY_SIGNAL_TOOLS else int(wgbs_flank_length)


def compute_matrix_bin_size_for_tool(
    tool: str,
    wgbs_compute_matrix_bin_size: int,
    array_compute_matrix_bin_size: int,
) -> int:
    return (
        int(array_compute_matrix_bin_size)
        if tool in ARRAY_SIGNAL_TOOLS
        else int(wgbs_compute_matrix_bin_size)
    )


def load_beta_track_from_beta_table(beta_path: Path) -> pd.DataFrame:
    beta_path = Path(beta_path).expanduser().resolve()
    if not beta_path.exists():
        raise FileNotFoundError(f"Missing beta track file: {beta_path}")

    with open(beta_path) as fh:
        header_fields = fh.readline().rstrip("\n").split("\t")

    has_header = header_fields[:4] == ["chrom", "start", "end", "beta"]
    if has_header:
        beta_df = pd.read_csv(beta_path, sep="\t")
    else:
        n_cols = len(header_fields)
        if n_cols < 4:
            raise ValueError(f"Unsupported beta track with fewer than 4 columns: {beta_path}")
        column_names = ["chrom", "start", "end", "beta"] + [
            f"extra_{idx}" for idx in range(n_cols - 4)
        ]
        beta_df = pd.read_csv(beta_path, sep="\t", header=None, names=column_names)

    beta_df = beta_df.loc[:, ["chrom", "start", "end", "beta"]].copy()
    beta_df["chrom"] = beta_df["chrom"].astype(str)
    beta_df["chrom"] = "chr" + beta_df["chrom"].str.replace("^chr", "", regex=True)
    beta_df["start"] = pd.to_numeric(beta_df["start"], errors="coerce")
    beta_df["end"] = pd.to_numeric(beta_df["end"], errors="coerce")
    beta_df["beta"] = pd.to_numeric(beta_df["beta"], errors="coerce")
    beta_df = beta_df.dropna(subset=["chrom", "start", "end", "beta"]).copy()
    beta_df = beta_df.loc[beta_df["chrom"].isin(CANONICAL_CHROMOSOMES)].copy()
    beta_df = beta_df.loc[beta_df["beta"].between(0.0, 1.0)].copy()
    beta_df["start"] = beta_df["start"].astype(int)
    beta_df["end"] = beta_df["end"].astype(int)
    beta_df = beta_df.loc[beta_df["end"] > beta_df["start"]].reset_index(drop=True)
    return collapse_duplicate_beta_intervals(beta_df)


def load_beta_track_from_config(config_row: dict) -> pd.DataFrame:
    meth_file = Path(config_row["meth_file"]).expanduser().resolve()
    if not meth_file.exists():
        raise FileNotFoundError(
            f"Missing methylation input for {config_row['sample']}: {meth_file}"
        )

    if meth_file.suffix == ".beta":
        raw_df = _load_beta_sample(meth_file, config_row["genome"])
    elif meth_file.suffixes[-2:] == [".bed", ".gz"]:
        raw_df = _load_bed_gz_sample(meth_file)
    else:
        raise ValueError(
            f"Unsupported methylation input format for {config_row['sample']}: {meth_file}"
        )

    beta_df = raw_df.rename(
        columns={
            "CpG_chrm": "chrom",
            "CpG_start": "start",
            "CpG_end": "end",
        }
    ).copy()
    beta_df = beta_df.loc[pd.to_numeric(beta_df["coverage"], errors="coerce") > 0].copy()
    beta_df["beta"] = (
        pd.to_numeric(beta_df["methylated_reads"], errors="coerce")
        / pd.to_numeric(beta_df["coverage"], errors="coerce")
    )
    beta_df = beta_df.dropna(subset=["chrom", "start", "end", "beta"]).copy()
    beta_df = beta_df.loc[:, ["chrom", "start", "end", "beta"]].reset_index(drop=True)
    return collapse_duplicate_beta_intervals(beta_df)


def build_shared_prep_outputs(config_row: dict, shared_prep_dir: Path, force_recreate: bool):
    manager = SharedPrepManager(
        sample_id=config_row["sample"],
        meth_file=config_row["meth_file"],
        genome=config_row["genome"],
        out_dir=shared_prep_dir,
        force_recreate=force_recreate,
        print_logs=True,
        skip_450k=False,
    )
    return manager.prepare()


def load_existing_methylseg_prep_sources(sample: str) -> dict:
    prep_config_path = (
        REGION_CALLING_RESULTS_DIR / "methylseg" / sample / "prep" / "config.yaml"
    )
    if not prep_config_path.exists():
        return {}

    with open(prep_config_path) as handle:
        prep_config = yaml.safe_load(handle) or {}
    input_paths = prep_config.get("input_paths") or {}

    resolved_paths = {}
    for key, raw_path in input_paths.items():
        if not raw_path:
            continue
        resolved_path = Path(raw_path).expanduser().resolve()
        if resolved_path.exists():
            resolved_paths[key] = str(resolved_path)
    return resolved_paths


def resolve_source_region_path(sample: str, tool: str) -> Path:
    canonical_tool = _canonical_tool_name(tool)
    tool_config = FIGURE_TOOL_REGISTRY_BY_NAME[canonical_tool]
    source_path = REGION_CALLING_RESULTS_DIR
    for part in tool_config["path_parts"]:
        source_path = source_path / part.format(sample=sample)
    return source_path


def sort_manifest_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    sort_cols = [
        col
        for col in ["config_order", "tool_order", "sample", "tool", "signal_track"]
        if col in df.columns
    ]
    if not sort_cols:
        return df.reset_index(drop=True)
    return df.sort_values(sort_cols).reset_index(drop=True)


def export_bigwig_for_sample_track(
    config_row: dict,
    signal_track: str,
    bigwig_dir: Path,
    force_rebuild_bigwigs: bool,
    shared_prep_outputs=None,
    fallback_sources=None,
):
    sample = config_row["sample"]
    signal_mode = signal_mode_for_track(signal_track)
    output_path = bigwig_dir / f"{sample}.{signal_track}.{signal_mode}.methylation.bigwig"
    cache_used = output_path.exists() and not force_rebuild_bigwigs

    if signal_track == "wgbs":
        signal_source = config_row["meth_file"]
        if not cache_used:
            beta_df = load_beta_track_from_config(config_row)
            chrom_sizes = _load_chrom_sizes(config_row["genome"])
            _write_bigwig(beta_df, output_path, chrom_sizes)
    elif signal_track == "hm450k":
        fallback_sources = fallback_sources or {}
        hm450k_beta_source = None
        if shared_prep_outputs is not None and shared_prep_outputs.hm450k_beta is not None:
            hm450k_beta_source = Path(shared_prep_outputs.hm450k_beta).expanduser().resolve()
        elif fallback_sources.get("hm450k_beta"):
            hm450k_beta_source = Path(fallback_sources["hm450k_beta"]).expanduser().resolve()
        if hm450k_beta_source is None:
            raise FileNotFoundError(
                f"HM450K shared prep beta track is missing for {sample}."
            )
        signal_source = str(hm450k_beta_source)
        if not cache_used:
            beta_df = load_beta_track_from_beta_table(signal_source)
            chrom_sizes = _load_chrom_sizes(config_row["genome"])
            _write_bigwig(beta_df, output_path, chrom_sizes)
    else:
        raise ValueError(f"Unsupported signal track: {signal_track}")

    return {
        "config_order": int(config_row["config_order"]),
        "sample": sample,
        "sample_id": config_row["sample_id"],
        "genome": config_row["genome"],
        "signal_track": signal_track,
        "signal_mode": signal_mode,
        "signal_source": str(signal_source),
        "bigwig_path": str(output_path),
        "cache_used": bool(cache_used),
    }


def build_signal_tracks_task(task: dict) -> dict:
    config_row = task["config_row"]
    signal_rows = []
    failure_rows = []
    shared_prep_outputs = None
    fallback_sources = {}
    if "hm450k" in task["required_tracks"]:
        try:
            shared_prep_outputs = build_shared_prep_outputs(
                config_row,
                shared_prep_dir=Path(task["shared_prep_dir"]),
                force_recreate=bool(task["force_rebuild_bigwigs"]),
            )
        except Exception as exc:
            fallback_sources = load_existing_methylseg_prep_sources(config_row["sample"])
            if fallback_sources.get("hm450k_beta"):
                print(
                    "Falling back to cached HM450K beta for "
                    f"{config_row['sample']} because shared prep failed: {exc}"
                )
            else:
                failure_rows.append(
                    {
                        "sample": config_row["sample"],
                        "tool": "<sample-level>",
                        "stage": "shared_prep",
                        "error": str(exc),
                    }
                )
    for signal_track in task["required_tracks"]:
        try:
            signal_rows.append(
                export_bigwig_for_sample_track(
                    config_row=config_row,
                    signal_track=signal_track,
                    bigwig_dir=Path(task["bigwig_dir"]),
                    force_rebuild_bigwigs=bool(task["force_rebuild_bigwigs"]),
                    shared_prep_outputs=shared_prep_outputs,
                    fallback_sources=fallback_sources,
                )
            )
        except Exception as exc:
            failure_rows.append(
                {
                    "sample": config_row["sample"],
                    "tool": f"<signal:{signal_track}>",
                    "stage": "signal_track_export",
                    "error": str(exc),
                }
            )
    return {"signal_rows": signal_rows, "failure_rows": failure_rows}


def build_region_manifest_row(task: dict) -> dict:
    config_row = task["config_row"]
    tool = task["tool"]
    signal_row = task["signal_row"]
    compute_matrix_bin_size = int(task["compute_matrix_bin_size"])
    flank_length = int(task["flank_length"])
    region_body_length = int(task["region_body_length"])
    average_type_bins = str(task["average_type_bins"])

    source_region_path = resolve_source_region_path(config_row["sample"], tool)
    if not source_region_path.exists():
        raise FileNotFoundError(
            f"Missing region file for {config_row['sample']} {tool}: {source_region_path}"
        )

    bigwig_path = Path(signal_row["bigwig_path"])
    chrom_sizes = get_eligible_chrom_sizes(CANONICAL_CHROMOSOMES, bigwig_path)
    if not chrom_sizes:
        raise RuntimeError(f"No eligible canonical chromosomes found in {bigwig_path}")

    raw_region_df = load_tool_regions(config_row["sample"], tool)
    prepared_region_df = clean_interval_df(raw_region_df, chrom_sizes)
    if prepared_region_df.empty:
        raise AssertionError(
            f"{config_row['sample']} {tool} produced zero retained regions after cleaning."
        )

    prepared_path = (
        Path(task["prepared_region_dir"]) / config_row["sample"] / f"{tool}.bed"
    )
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    write_bed(prepared_region_df, prepared_path)

    deeptools_region_df = filter_regions_for_deeptools(
        prepared_region_df,
        compute_matrix_bin_size,
    )
    deeptools_region_path = (
        Path(task["deeptools_dir"])
        / config_row["sample"]
        / tool
        / f"{config_row['sample']}.{tool}.deeptools_regions.bed"
    )
    deeptools_region_path.parent.mkdir(parents=True, exist_ok=True)
    write_bed(deeptools_region_df, deeptools_region_path)

    canonical_tool = _canonical_tool_name(tool)
    return {
        "config_order": int(config_row["config_order"]),
        "tool_order": int(task["tool_order"]),
        "sample": config_row["sample"],
        "sample_id": config_row["sample_id"],
        "sample_label": config_row["sample_id"],
        "member_samples": str(config_row["sample"]),
        "member_sample_ids": str(config_row["sample_id"]),
        "n_samples": 1,
        "matrix_mode": "per_sample",
        "tool": tool,
        "tool_label": REGION_CALLING_TOOL_LABELS[tool],
        "platform": TOOL_PLATFORM_BY_NAME[canonical_tool],
        "signal_track": signal_row["signal_track"],
        "signal_mode": signal_row["signal_mode"],
        "signal_source": signal_row["signal_source"],
        "region_type": region_type_for_tool(tool).upper(),
        "source_region_path": str(source_region_path),
        "prepared_region_path": str(prepared_path),
        "deeptools_region_path": str(deeptools_region_path),
        "bigwig_path": str(bigwig_path),
        "signal_sources": str(signal_row["signal_source"]),
        "bigwig_paths": str(bigwig_path),
        "total_regions": int(len(prepared_region_df)),
        "visualized_regions": int(len(deeptools_region_df)),
        "excluded_short_regions": int(len(prepared_region_df) - len(deeptools_region_df)),
        "min_region_length_bp": compute_matrix_bin_size,
        "compute_matrix_bin_bp": compute_matrix_bin_size,
        "flank_length": flank_length,
        "region_body_length": region_body_length,
        "average_type_bins": average_type_bins,
    }


def _validate_group_consistency(tool_df: pd.DataFrame, group_fields: list[str]) -> None:
    for field in group_fields:
        if tool_df[field].astype(str).nunique() > 1:
            raise ValueError(
                f"Cannot combine samples for tool {tool_df['tool'].iloc[0]} because "
                f"{field} differs across selected samples."
            )


def build_deeptools_region_manifest_df(
    region_manifest_df: pd.DataFrame,
    deeptools_dir: Path,
    combine_samples_per_tool: bool,
) -> pd.DataFrame:
    if region_manifest_df.empty:
        return pd.DataFrame()

    if not combine_samples_per_tool or region_manifest_df["sample"].astype(str).nunique() <= 1:
        return sort_manifest_df(region_manifest_df.copy())

    grouped_rows = []
    group_fields = [
        "tool_label",
        "platform",
        "region_type",
        "signal_track",
        "signal_mode",
        "tool_order",
        "compute_matrix_bin_bp",
        "flank_length",
        "region_body_length",
        "average_type_bins",
    ]
    tool_order_lookup = {
        tool: tool_order for tool_order, tool in enumerate(region_manifest_df["tool"].drop_duplicates().tolist())
    }

    for tool in region_manifest_df["tool"].drop_duplicates().tolist():
        tool_df = (
            region_manifest_df.loc[region_manifest_df["tool"].eq(tool)]
            .copy()
            .sort_values(["config_order", "sample"])
            .reset_index(drop=True)
        )
        _validate_group_consistency(tool_df, group_fields)

        member_samples = tool_df["sample"].astype(str).tolist()
        member_sample_ids = tool_df["sample_id"].astype(str).tolist()
        sample_key = build_sample_group_key(member_samples)
        sample_label = ", ".join(member_sample_ids)

        bigwig_paths = tool_df["bigwig_path"].astype(str).tolist()
        signal_sources = tool_df["signal_source"].astype(str).tolist()
        reference_row = tool_df.iloc[0].to_dict()
        chrom_sizes = get_eligible_chrom_sizes(
            CANONICAL_CHROMOSOMES,
            Path(bigwig_paths[0]),
        )
        if not chrom_sizes:
            raise RuntimeError(
                f"No eligible canonical chromosomes found in {bigwig_paths[0]}"
            )

        prepared_region_input_dfs = [
            pd.read_csv(path, sep="\t", header=None, names=["chrom", "start", "end"])
            for path in tool_df["prepared_region_path"].astype(str).tolist()
            if Path(path).exists() and Path(path).stat().st_size > 0
        ]
        combined_region_input_df = (
            pd.concat(prepared_region_input_dfs, ignore_index=True)
            if prepared_region_input_dfs
            else pd.DataFrame(columns=["chrom", "start", "end"])
        )
        combined_region_df = clean_interval_df(combined_region_input_df, chrom_sizes)
        combined_deeptools_region_df = filter_regions_for_deeptools(
            combined_region_df,
            int(reference_row["compute_matrix_bin_bp"]),
        )

        sample_output_dir = Path(deeptools_dir) / sample_key / tool
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        deeptools_region_path = sample_output_dir / f"{sample_key}.{tool}.deeptools_regions.bed"
        write_bed(combined_deeptools_region_df, deeptools_region_path)

        grouped_rows.append(
            {
                "config_order": int(tool_df["config_order"].min()),
                "tool_order": int(tool_order_lookup[tool]),
                "sample": sample_key,
                "sample_id": sample_key,
                "sample_label": sample_label,
                "member_samples": join_manifest_values(member_samples),
                "member_sample_ids": join_manifest_values(member_sample_ids),
                "n_samples": int(len(member_samples)),
                "matrix_mode": "combined_samples_per_tool",
                "tool": tool,
                "tool_label": reference_row["tool_label"],
                "platform": reference_row["platform"],
                "signal_track": reference_row["signal_track"],
                "signal_mode": reference_row["signal_mode"],
                "signal_source": join_manifest_values(signal_sources),
                "signal_sources": join_manifest_values(signal_sources),
                "region_type": reference_row["region_type"],
                "source_region_path": join_manifest_values(
                    tool_df["source_region_path"].astype(str).tolist()
                ),
                "prepared_region_path": join_manifest_values(
                    tool_df["prepared_region_path"].astype(str).tolist()
                ),
                "deeptools_region_path": str(deeptools_region_path),
                "bigwig_path": join_manifest_values(bigwig_paths),
                "bigwig_paths": join_manifest_values(bigwig_paths),
                "total_regions": int(len(combined_region_df)),
                "visualized_regions": int(len(combined_deeptools_region_df)),
                "excluded_short_regions": int(
                    len(combined_region_df) - len(combined_deeptools_region_df)
                ),
                "min_region_length_bp": int(reference_row["compute_matrix_bin_bp"]),
                "compute_matrix_bin_bp": int(reference_row["compute_matrix_bin_bp"]),
                "flank_length": int(reference_row["flank_length"]),
                "region_body_length": int(reference_row["region_body_length"]),
                "average_type_bins": str(reference_row["average_type_bins"]),
            }
        )

    return sort_manifest_df(pd.DataFrame(grouped_rows))


def prepare_region_task_safe(task: dict) -> dict:
    try:
        return {"ok": True, "row": build_region_manifest_row(task)}
    except Exception as exc:
        config_row = task["config_row"]
        return {
            "ok": False,
            "failure": {
                "sample": config_row["sample"],
                "tool": task["tool"],
                "stage": "region_preparation",
                "error": str(exc),
            },
        }


def run_compute_matrix_task(task: dict) -> dict:
    region_row = task["region_row"]
    sample = str(region_row["sample"])
    tool = str(region_row["tool"])
    sample_output_dir = Path(task["deeptools_dir"]) / sample / tool
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    if int(region_row["visualized_regions"]) <= 0:
        raise RuntimeError(
            f"{sample} {tool} had zero regions at least "
            f"{region_row['compute_matrix_bin_bp']} bp after filtering."
        )

    matrix_path = sample_output_dir / f"{sample}.{tool}.methylation.matrix.gz"
    matrix_values_path = sample_output_dir / f"{sample}.{tool}.methylation.matrix.tsv"
    sorted_regions_path = sample_output_dir / f"{sample}.{tool}.sorted_regions.bed"
    bigwig_paths = split_manifest_values(region_row.get("bigwig_paths", region_row["bigwig_path"]))
    if not bigwig_paths:
        raise RuntimeError(f"Missing bigWig inputs for {sample} {tool}.")

    compute_matrix_command = [
        task["compute_matrix_command"],
        "scale-regions",
        "-p",
        str(get_deeptools_processor_count()),
        "-S",
        *bigwig_paths,
        "-R",
        str(region_row["deeptools_region_path"]),
        "-b",
        str(region_row["flank_length"]),
        "-a",
        str(region_row["flank_length"]),
        "--binSize",
        str(region_row["compute_matrix_bin_bp"]),
        "--regionBodyLength",
        str(region_row["region_body_length"]),
        "--averageTypeBins",
        str(region_row["average_type_bins"]),
        "--sortRegions",
        "keep",
        "-o",
        str(matrix_path),
        "--outFileSortedRegions",
        str(sorted_regions_path),
        "--outFileNameMatrix",
        str(matrix_values_path),
    ]
    compute_matrix_cache_used = maybe_run_command(
        compute_matrix_command,
        [matrix_path, matrix_values_path, sorted_regions_path],
        force=bool(task["force_rerun_matrices"]),
    )

    return {
        **region_row,
        "matrix_path": str(matrix_path),
        "matrix_values_path": str(matrix_values_path),
        "sorted_regions_path": str(sorted_regions_path),
        "compute_matrix_cache_used": bool(compute_matrix_cache_used),
    }


def run_compute_matrix_task_safe(task: dict) -> dict:
    try:
        return {"ok": True, "row": run_compute_matrix_task(task)}
    except Exception as exc:
        region_row = task["region_row"]
        return {
            "ok": False,
            "failure": {
                "sample": region_row["sample"],
                "tool": region_row["tool"],
                "stage": "compute_matrix",
                "error": str(exc),
            },
        }


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Build cached methylation deepTools matrices for all selected tools and samples."
    )
    parser.add_argument(
        "--configs-path",
        type=Path,
        default=DEFAULT_CONFIGS_PATH,
        help="Path to the sample config manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where bigWigs, manifests, and matrices will be written.",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=None,
        help="Optional subset of samples to run. Defaults to all samples in configs.txt.",
    )
    parser.add_argument(
        "--tools",
        nargs="+",
        default=list(REGION_CALLING_TOOL_ORDER),
        help="Tools to include. Defaults to all region-calling tools.",
    )
    parser.add_argument(
        "--compute-matrix-bin-size",
        type=int,
        default=None,
        help=(
            "Optional shared computeMatrix --binSize override for every tool. "
            "When omitted, WGBS and HM450K use their platform-specific defaults."
        ),
    )
    parser.add_argument(
        "--wgbs-compute-matrix-bin-size",
        type=int,
        default=DEFAULT_WGBS_COMPUTE_MATRIX_BIN_SIZE,
        help="computeMatrix --binSize for WGBS-backed tools.",
    )
    parser.add_argument(
        "--array-compute-matrix-bin-size",
        type=int,
        default=DEFAULT_ARRAY_COMPUTE_MATRIX_BIN_SIZE,
        help="computeMatrix --binSize for HM450K-backed tools.",
    )
    parser.add_argument(
        "--region-body-length",
        type=int,
        default=DEFAULT_REGION_BODY_LENGTH,
        help="computeMatrix --regionBodyLength value.",
    )
    parser.add_argument(
        "--wgbs-flank-length",
        type=int,
        default=DEFAULT_WGBS_FLANK_LENGTH,
        help="Flank length for WGBS-backed tools.",
    )
    parser.add_argument(
        "--array-flank-length",
        type=int,
        default=DEFAULT_ARRAY_FLANK_LENGTH,
        help="Flank length for HM450K-backed tools.",
    )
    parser.add_argument(
        "--average-type-bins",
        choices=AVERAGE_TYPE_BINS_CHOICES,
        default=DEFAULT_AVERAGE_TYPE_BINS,
        help="computeMatrix --averageTypeBins value.",
    )
    parser.add_argument(
        "--force-rebuild-bigwigs",
        action="store_true",
        help="Rebuild cached signal bigWigs even if they already exist.",
    )
    parser.add_argument(
        "--force-rerun-matrices",
        action="store_true",
        help="Regenerate computeMatrix outputs even if they already exist.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker count for region and matrix stages. Defaults to available CPUs.",
    )
    parser.add_argument(
        "--combine-samples-per-tool",
        action="store_true",
        help=(
            "When more than one sample is selected, build one combined computeMatrix "
            "input per tool across the selected samples."
        ),
    )
    return parser


def run(
    configs_path=DEFAULT_CONFIGS_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
    samples=None,
    tools=None,
    compute_matrix_bin_size=None,
    wgbs_compute_matrix_bin_size=DEFAULT_WGBS_COMPUTE_MATRIX_BIN_SIZE,
    array_compute_matrix_bin_size=DEFAULT_ARRAY_COMPUTE_MATRIX_BIN_SIZE,
    region_body_length=DEFAULT_REGION_BODY_LENGTH,
    wgbs_flank_length=DEFAULT_WGBS_FLANK_LENGTH,
    array_flank_length=DEFAULT_ARRAY_FLANK_LENGTH,
    average_type_bins=DEFAULT_AVERAGE_TYPE_BINS,
    force_rebuild_bigwigs=False,
    force_rerun_matrices=False,
    workers=None,
    combine_samples_per_tool=False,
):
    compute_matrix_command = ensure_compute_matrix_command()
    if compute_matrix_bin_size is not None:
        wgbs_compute_matrix_bin_size = int(compute_matrix_bin_size)
        array_compute_matrix_bin_size = int(compute_matrix_bin_size)
    selected_tools = validate_selected_tools(
        list(tools) if tools is not None else list(REGION_CALLING_TOOL_ORDER)
    )
    sample_configs_df = load_sample_configs(configs_path, samples)
    ensure_wgbstools_if_needed(sample_configs_df)
    required_tracks = required_signal_tracks(selected_tools)
    configure_pybedtools()

    output_dir = Path(output_dir).expanduser().resolve()
    bigwig_dir = output_dir / "bigwigs"
    prepared_region_dir = output_dir / "prepared_regions"
    deeptools_dir = output_dir / "deeptools"
    shared_prep_dir = output_dir / "shared_prep"
    tables_dir = output_dir / "tables"
    for directory in [output_dir, bigwig_dir, prepared_region_dir, deeptools_dir, shared_prep_dir, tables_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    for stale_manifest_path in [
        tables_dir / "signal_manifest.tsv",
        tables_dir / "region_manifest.tsv",
        tables_dir / "deeptools_region_manifest.tsv",
        tables_dir / "deeptools_matrix_outputs.tsv",
        tables_dir / "failures.tsv",
    ]:
        if stale_manifest_path.exists():
            stale_manifest_path.unlink()

    available_cpus = _available_cpu_count()
    worker_count = available_cpus if workers is None else max(int(workers), 1)

    print("Methylation deepTools matrix configuration:")
    print(f"  Configs path:           {Path(configs_path).resolve()}")
    print(f"  Output dir:             {output_dir}")
    print(f"  Samples:                {', '.join(sample_configs_df['sample'].tolist())}")
    print(f"  Tools:                  {', '.join(selected_tools)}")
    print(f"  Signal tracks:          {', '.join(required_tracks)}")
    print(f"  WGBS computeMatrix bin: {int(wgbs_compute_matrix_bin_size)}")
    print(f"  Array computeMatrix bin:{int(array_compute_matrix_bin_size)}")
    print(f"  Region body length:     {int(region_body_length)}")
    print(f"  WGBS flank length:      {int(wgbs_flank_length)}")
    print(f"  Array flank length:     {int(array_flank_length)}")
    print(f"  averageTypeBins:        {average_type_bins}")
    print(f"  Available CPUs:         {available_cpus}")
    print(f"  Worker count:           {worker_count}")
    print(f"  Force rebuild bigWigs:  {bool(force_rebuild_bigwigs)}")
    print(f"  Force rerun matrices:   {bool(force_rerun_matrices)}")
    print(f"  Combine samples/tool:   {bool(combine_samples_per_tool)}")

    signal_tasks = [
        {
            "config_row": config_row,
            "required_tracks": required_tracks,
            "bigwig_dir": str(bigwig_dir),
            "shared_prep_dir": str(shared_prep_dir),
            "force_rebuild_bigwigs": bool(force_rebuild_bigwigs),
        }
        for config_row in sample_configs_df.to_dict("records")
    ]
    signal_results = run_parallel(
        signal_tasks,
        build_signal_tracks_task,
        min(worker_count, len(signal_tasks)),
        "signal-track export",
    )
    signal_rows = []
    failure_rows = []
    for result in signal_results:
        signal_rows.extend(result["signal_rows"])
        failure_rows.extend(result["failure_rows"])

    signal_manifest_df = pd.DataFrame(signal_rows)
    if not signal_manifest_df.empty:
        signal_manifest_df = signal_manifest_df.sort_values(
            ["config_order", "sample", "signal_track"]
        ).reset_index(drop=True)
    signal_manifest_path = tables_dir / "signal_manifest.tsv"
    if not signal_manifest_df.empty:
        signal_manifest_df.to_csv(signal_manifest_path, sep="\t", index=False)

    signal_lookup = {
        (row["sample"], row["signal_track"]): row
        for row in signal_manifest_df.to_dict("records")
    }
    region_tasks = []
    for config_row in sample_configs_df.to_dict("records"):
        for tool_order, tool in enumerate(selected_tools):
            signal_track = signal_track_for_tool(tool)
            signal_row = signal_lookup.get((config_row["sample"], signal_track))
            if signal_row is None:
                failure_rows.append(
                    {
                        "sample": config_row["sample"],
                        "tool": tool,
                        "stage": "region_preparation",
                        "error": f"Missing {signal_track} signal track for {config_row['sample']} {tool}.",
                    }
                )
                continue
            region_tasks.append(
                {
                    "config_row": config_row,
                    "tool": tool,
                    "tool_order": tool_order,
                    "signal_row": signal_row,
                    "prepared_region_dir": str(prepared_region_dir),
                    "deeptools_dir": str(deeptools_dir),
                    "compute_matrix_bin_size": compute_matrix_bin_size_for_tool(
                        tool,
                        wgbs_compute_matrix_bin_size=wgbs_compute_matrix_bin_size,
                        array_compute_matrix_bin_size=array_compute_matrix_bin_size,
                    ),
                    "flank_length": flank_length_for_tool(
                        tool,
                        wgbs_flank_length=wgbs_flank_length,
                        array_flank_length=array_flank_length,
                    ),
                    "region_body_length": int(region_body_length),
                    "average_type_bins": str(average_type_bins),
                }
            )

    region_results = run_parallel(
        region_tasks,
        prepare_region_task_safe,
        min(worker_count, len(region_tasks) or 1),
        "region preparation",
    )
    region_rows = []
    for result in region_results:
        if result["ok"]:
            region_rows.append(result["row"])
        else:
            failure_rows.append(result["failure"])
    region_manifest_df = sort_manifest_df(pd.DataFrame(region_rows))
    if region_manifest_df.empty:
        failures_df = pd.DataFrame(failure_rows)
        failures_path = tables_dir / "failures.tsv"
        if not failures_df.empty:
            failures_df.to_csv(failures_path, sep="\t", index=False)
        raise RuntimeError(
            "No region sets were prepared. Check failures.tsv for details."
            if failures_path.exists()
            else "No region sets were prepared."
        )

    region_manifest_path = tables_dir / "region_manifest.tsv"
    region_manifest_df.to_csv(region_manifest_path, sep="\t", index=False)

    deeptools_region_manifest_df = build_deeptools_region_manifest_df(
        region_manifest_df=region_manifest_df,
        deeptools_dir=deeptools_dir,
        combine_samples_per_tool=bool(combine_samples_per_tool),
    )
    deeptools_region_manifest_path = tables_dir / "deeptools_region_manifest.tsv"
    deeptools_region_manifest_df.to_csv(
        deeptools_region_manifest_path, sep="\t", index=False
    )

    matrix_tasks = [
        {
            "region_row": region_row,
            "deeptools_dir": str(deeptools_dir),
            "compute_matrix_command": compute_matrix_command,
            "force_rerun_matrices": bool(force_rerun_matrices),
        }
        for region_row in deeptools_region_manifest_df.to_dict("records")
    ]
    matrix_rows = []
    for result in run_parallel(
        matrix_tasks,
        run_compute_matrix_task_safe,
        min(worker_count, len(matrix_tasks) or 1),
        "computeMatrix execution",
    ):
        if result["ok"]:
            matrix_rows.append(result["row"])
        else:
            failure_rows.append(result["failure"])
    deeptools_matrix_outputs_df = sort_manifest_df(pd.DataFrame(matrix_rows))
    deeptools_matrix_outputs_path = tables_dir / "deeptools_matrix_outputs.tsv"
    if not deeptools_matrix_outputs_df.empty:
        deeptools_matrix_outputs_df.to_csv(
            deeptools_matrix_outputs_path, sep="\t", index=False
        )

    failures_df = pd.DataFrame(failure_rows)
    failures_path = tables_dir / "failures.tsv"
    if not failures_df.empty:
        failures_df.to_csv(failures_path, sep="\t", index=False)

    print("\nMethylation deepTools matrix build complete.")
    print(f"  Signal manifest:           {signal_manifest_path}")
    print(f"  Region manifest:           {region_manifest_path}")
    print(f"  deepTools region manifest: {deeptools_region_manifest_path}")
    print(f"  Matrix outputs:            {deeptools_matrix_outputs_path}")
    if failures_path.exists():
        print(f"  Failures:                  {failures_path}")


def main(argv=None):
    args = _build_parser().parse_args(argv)
    run(
        configs_path=args.configs_path,
        output_dir=args.output_dir,
        samples=args.samples,
        tools=args.tools,
        compute_matrix_bin_size=args.compute_matrix_bin_size,
        wgbs_compute_matrix_bin_size=args.wgbs_compute_matrix_bin_size,
        array_compute_matrix_bin_size=args.array_compute_matrix_bin_size,
        region_body_length=args.region_body_length,
        wgbs_flank_length=args.wgbs_flank_length,
        array_flank_length=args.array_flank_length,
        average_type_bins=args.average_type_bins,
        force_rebuild_bigwigs=args.force_rebuild_bigwigs,
        force_rerun_matrices=args.force_rerun_matrices,
        workers=args.workers,
        combine_samples_per_tool=args.combine_samples_per_tool,
    )


if __name__ == "__main__":
    main()
