from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:
    import plotly.express as px
except ImportError:  # pragma: no cover - environment-specific dependency
    px = None

WORKSPACE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORKSPACE_DIR.parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
REGION_CALLING_UTILS_DIR = ANALYSIS_DIR / "01_region_calling_analysis" / "utils"
SYNTHETIC_ANALYSIS_DIR = ANALYSIS_DIR / "02_synthetic_analysis"
FUNCTIONAL_ANALYSIS_DIR = ANALYSIS_DIR / "03_chromatin_analysis"
LAD_ANALYSIS_DIR = ANALYSIS_DIR / "04_lad_analysis"
DATA_DIR = PROJECT_ROOT / "data" / "methylation_data"
CHROMATIN_DATA_DIR = PROJECT_ROOT / "data" / "chromatin_data"
DEVELOPMENT_RESULTS_DIR = PROJECT_ROOT / "results" / "00_methylseg_development"
DEFAULT_WORKSPACE_RESULTS_DIR = DEVELOPMENT_RESULTS_DIR / "workspace"
DEFAULT_SWEEP_ROOT = DEVELOPMENT_RESULTS_DIR
DEFAULT_SAMPLE_NAMES = ["TE5.wgbs", "ESO26.wgbs"]
DEFAULT_SYNTHETIC_SAMPLE_IDS = [
    "synthetic_WGBS_colon_primary_normal_1_hg38",
    "synthetic_WGBS_colon_primary_normal_2_hg38",
    "synthetic_WGBS_colon_primary_normal_3_hg38",
]
CANONICAL_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
DEFAULT_DEEPTOOLS_BIN_SIZE = 1000
DEFAULT_REGION_BODY_LENGTH = 1_000_000
DEFAULT_FLANK_LENGTH = 500_000
TRACKED_CONFIG_KEYS = [
    "clean_min_cpgs",
    "clean_min_region_length",
    "clean_merge_gap_bp",
    "wgbs_window_specs",
    "hm450_window_specs",
    "wgbs_hmm_type",
    "wgbs_hmm_params",
    "hm450_hmm_type",
    "hm450_hmm_params",
    "min_coverage",
    "n_states",
    "int_low_cutoff",
    "int_high_cutoff",
    "high_cutoff",
    "random_state",
]
SWEEP_MANIFEST_COLUMNS = [
    "config_name",
    "status",
    "out_root",
    "config_hash",
    "samples",
    "max_workers",
    "run_chromatin",
    "run_lad",
    "run_synthetic_prep",
    "run_synthetic_aggregate",
    "clean_min_cpgs",
    "clean_min_region_length",
    "clean_merge_gap_bp",
    "wgbs_window_specs",
    "hm450_window_specs",
    "wgbs_hmm_type",
    "wgbs_hmm_params",
    "hm450_hmm_type",
    "hm450_hmm_params",
    "error",
]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
for local_import_dir in [
    REGION_CALLING_UTILS_DIR,
    SYNTHETIC_ANALYSIS_DIR,
    FUNCTIONAL_ANALYSIS_DIR,
    LAD_ANALYSIS_DIR,
]:
    if str(local_import_dir) not in sys.path:
        sys.path.append(str(local_import_dir))

CORE_METHYLSEG_CLASS = None

DEFAULT_SAMPLE_SPECS = {
    "TE5.wgbs": {
        "sample": "TE5.wgbs",
        "meth_file": DATA_DIR / "TE5.wgbs.bed.gz",
        "genome": "hg38",
    },
    "ESO26.wgbs": {
        "sample": "ESO26.wgbs",
        "meth_file": DATA_DIR / "ESO26.wgbs.bed.gz",
        "genome": "hg38",
    },
}

METHYLSEG_TOOL_CONFIGS = [
    {
        "tool": "methylseg",
        "tool_label": "MethylSeg WGBS",
        "tool_family": "methylseg",
        "parser_family": "methylseg",
        "platform": "wgbs",
        "region_type": "PMD",
        "rank_order": 0,
        "deeptools_order": 0,
        "path_parts": [
            "methylseg",
            "{sample}",
            "out",
            "wgbs",
            "summary_files",
            "segments_cleaned_PMD.bed",
        ],
    },
    {
        "tool": "methylseg_hm450k",
        "tool_label": "MethylSeg HM450K",
        "tool_family": "methylseg",
        "parser_family": "methylseg",
        "platform": "hm450k",
        "region_type": "PMD",
        "rank_order": 1,
        "deeptools_order": 1,
        "path_parts": [
            "methylseg",
            "{sample}",
            "out",
            "hm450k",
            "summary_files",
            "segments_cleaned_PMD.bed",
        ],
    },
]

METHYLSEG_TOOL_CONFIGS_BY_NAME = {
    config["tool"]: config for config in METHYLSEG_TOOL_CONFIGS
}


def _sample_to_sample_id(sample: str) -> str:
    return str(sample).replace(".wgbs", "")


def _import_methylseg_components():
    try:
        from analysis.shared_utils.methylseg.methylseg.helper_classes import (
            MethylStateAssignmentMethod,
        )
        from analysis.shared_utils.methylseg.methylseg.methylseg_pathway import (
            MethylSegPathway as CoreMethylSegPathway,
        )
        from methyl_tool_comparator import (
            MethylSegPathway as ComparatorMethylSegPathway,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        raise ModuleNotFoundError(
            "MethylSeg imports failed. Run this notebook in the project environment "
            "that includes the segmentation dependencies such as 'cthmm'."
        ) from exc
    return (
        CoreMethylSegPathway,
        MethylStateAssignmentMethod,
        ComparatorMethylSegPathway,
    )


def _import_chromatin_components():
    try:
        from chromatin_analysis_utils import (
            compute_tool_metrics_task,
            prepare_sample_background_task,
            prepare_deeptools_regions_task,
            run_deeptools_for_sample_task,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        raise ModuleNotFoundError(
            "Chromatin fallback imports failed. Run this notebook in an environment "
            "that includes chromatin dependencies such as 'pybedtools' and 'pyBigWig'."
        ) from exc
    return {
        "compute_tool_metrics_task": compute_tool_metrics_task,
        "prepare_sample_background_task": prepare_sample_background_task,
        "prepare_deeptools_regions_task": prepare_deeptools_regions_task,
        "run_deeptools_for_sample_task": run_deeptools_for_sample_task,
    }


def _import_lad_components():
    try:
        from slurm_code.run_lad import (
            DEFAULT_HG19_TO_HG38_CHAIN,
            DEFAULT_LAD_INTERVAL_TRACK_PATH,
            DEFAULT_LAMINB1_SIGNAL_TRACK_PATH,
            DEFAULT_LIFTOVER_SCRIPT_PATH,
            prepare_lad_reference,
        )
        from lad_analysis_utils import (
            clean_and_merge_intervals,
            compute_end_window_overlap_metrics,
            load_tool_regions,
            read_bed_intervals,
            resolve_sample_genome,
            sample_to_sample_id,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        raise ModuleNotFoundError(
            "LAD/summarization imports failed. Run this notebook in an environment "
            "that includes LAD dependencies such as 'pybedtools' and 'pyBigWig'."
        ) from exc
    return {
        "DEFAULT_HG19_TO_HG38_CHAIN": DEFAULT_HG19_TO_HG38_CHAIN,
        "DEFAULT_LAD_INTERVAL_TRACK_PATH": DEFAULT_LAD_INTERVAL_TRACK_PATH,
        "DEFAULT_LAMINB1_SIGNAL_TRACK_PATH": DEFAULT_LAMINB1_SIGNAL_TRACK_PATH,
        "DEFAULT_LIFTOVER_SCRIPT_PATH": DEFAULT_LIFTOVER_SCRIPT_PATH,
        "prepare_lad_reference": prepare_lad_reference,
        "clean_and_merge_intervals": clean_and_merge_intervals,
        "compute_end_window_overlap_metrics": compute_end_window_overlap_metrics,
        "load_tool_regions": load_tool_regions,
        "read_bed_intervals": read_bed_intervals,
        "resolve_sample_genome": resolve_sample_genome,
        "sample_to_sample_id": sample_to_sample_id,
    }


def _compute_interval_overlap_bp(
    region_df: pd.DataFrame,
    truth_df: pd.DataFrame,
) -> int:
    lad = _import_lad_components()
    clean_and_merge_intervals = lad["clean_and_merge_intervals"]
    region_df = clean_and_merge_intervals(region_df)
    truth_df = clean_and_merge_intervals(truth_df)
    if region_df.empty or truth_df.empty:
        return 0

    total_overlap_bp = 0
    truth_lookup = {
        chrom: chrom_df.reset_index(drop=True)
        for chrom, chrom_df in truth_df.groupby("chrom", sort=False)
    }
    for chrom, chrom_region_df in region_df.groupby("chrom", sort=False):
        chrom_truth_df = truth_lookup.get(chrom)
        if chrom_truth_df is None or chrom_truth_df.empty:
            continue
        truth_idx = 0
        truth_starts = chrom_truth_df["start"].to_numpy(dtype=np.int64)
        truth_ends = chrom_truth_df["end"].to_numpy(dtype=np.int64)
        for row in chrom_region_df.itertuples(index=False):
            region_start = int(row.start)
            region_end = int(row.end)
            while (
                truth_idx < len(truth_ends)
                and int(truth_ends[truth_idx]) <= region_start
            ):
                truth_idx += 1
            probe_idx = truth_idx
            while (
                probe_idx < len(truth_starts)
                and int(truth_starts[probe_idx]) < region_end
            ):
                overlap_start = max(region_start, int(truth_starts[probe_idx]))
                overlap_end = min(region_end, int(truth_ends[probe_idx]))
                if overlap_end > overlap_start:
                    total_overlap_bp += overlap_end - overlap_start
                probe_idx += 1
    return int(total_overlap_bp)


def _sanitize_for_yaml(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _sanitize_for_yaml(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_yaml(v) for v in value]
    return value


def _make_bar_figure(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    color: str,
    title: str,
    output_stem: Path,
):
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    if px is not None:
        fig = px.bar(df, x=x, y=y, color=color, barmode="group", title=title)
        fig.write_html(output_stem.with_suffix(".html"))
        return fig

    fig, ax = plt.subplots(figsize=(10, 6))
    samples = list(dict.fromkeys(df[x].tolist()))
    groups = list(dict.fromkeys(df[color].tolist()))
    x_positions = np.arange(len(samples), dtype=float)
    width = 0.8 / max(len(groups), 1)

    for group_idx, group_name in enumerate(groups):
        group_df = df.loc[df[color] == group_name].copy()
        group_values = []
        for sample in samples:
            sample_match = group_df.loc[group_df[x] == sample, y]
            group_values.append(
                float(sample_match.iloc[0]) if not sample_match.empty else np.nan
            )
        ax.bar(
            x_positions + group_idx * width - (len(groups) - 1) * width / 2,
            group_values,
            width=width,
            label=str(group_name),
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(samples, rotation=0)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    return fig


def _results_root(out_root: str | Path) -> Path:
    return Path(out_root).expanduser().resolve()


def _normalize_config_value_for_hash(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {
            str(key): _normalize_config_value_for_hash(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_config_value_for_hash(item) for item in value]
    return value


def _effective_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _normalize_config_value_for_hash(config.get(key))
        for key in TRACKED_CONFIG_KEYS
    }


def _config_hash(config: dict[str, Any]) -> str:
    payload = yaml.safe_dump(
        _effective_config_payload(config),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_metadata_path(run_root: str | Path) -> Path:
    return Path(run_root) / "run_metadata.yaml"


def _load_run_metadata(run_root: str | Path) -> dict[str, Any]:
    metadata_path = _run_metadata_path(run_root)
    if not metadata_path.exists():
        return {}
    with open(metadata_path) as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _write_run_metadata(run_root: str | Path, config: dict[str, Any]) -> Path:
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config_hash": _config_hash(config),
        "tracked_config": _effective_config_payload(config),
    }
    metadata_path = _run_metadata_path(run_root)
    with open(metadata_path, "w") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)
    return metadata_path


def _clear_directory_if_exists(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_symlink() or path.is_file():
        path.unlink()
        return True
    shutil.rmtree(path)
    return True


def _cleanup_main_outputs_for_config_change(out_root: str | Path) -> list[str]:
    out_root = _results_root(out_root)
    removed_paths = []
    methylseg_root = out_root / "methylseg"
    if methylseg_root.exists():
        for sample_dir in sorted(
            path for path in methylseg_root.iterdir() if path.is_dir()
        ):
            out_dir = sample_dir / "out"
            if _clear_directory_if_exists(out_dir):
                removed_paths.append(str(out_dir))
    functional_analysis_dir = out_root / "functional_analysis"
    if _clear_directory_if_exists(functional_analysis_dir):
        removed_paths.append(str(functional_analysis_dir))
    return removed_paths


def _cleanup_synthetic_outputs_for_config_change(out_root: str | Path) -> list[str]:
    syn_root = synthetic_root(out_root)
    removed_paths = []
    tool_results_root = syn_root / "tool_results" / "methylseg"
    if tool_results_root.exists():
        for sample_dir in sorted(
            path for path in tool_results_root.iterdir() if path.is_dir()
        ):
            out_dir = sample_dir / "out"
            if _clear_directory_if_exists(out_dir):
                removed_paths.append(str(out_dir))
    synthetic_analysis_dir = syn_root / "analysis"
    if _clear_directory_if_exists(synthetic_analysis_dir):
        removed_paths.append(str(synthetic_analysis_dir))
    return removed_paths


def _clear_outputs_if_requested(
    *,
    enabled: bool,
    run_root: str | Path,
    cleanup_fn,
    label: str,
) -> list[str]:
    if not enabled:
        return []
    removed_paths = cleanup_fn(run_root)
    print(
        f"[methylsegtests] Cleared {len(removed_paths)} cached output path(s) for {label} before rerun."
    )
    return removed_paths


def _invalidate_outputs_if_config_changed(
    *,
    run_root: str | Path,
    cleanup_fn,
    config: dict[str, Any],
    label: str,
) -> list[str]:
    run_root = Path(run_root)
    previous_metadata = _load_run_metadata(run_root)
    previous_hash = previous_metadata.get("config_hash")
    current_hash = _config_hash(config)
    if previous_hash is None:
        return []
    if previous_hash == current_hash:
        return []
    removed_paths = cleanup_fn(run_root)
    print(
        f"[methylsegtests] Config changed for {label}; removed {len(removed_paths)} stale output path(s) while preserving prep files."
    )
    return removed_paths


def _resolve_max_workers(max_workers: int | None, task_count: int) -> int:
    if task_count <= 0:
        return 0
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    return max(1, min(int(max_workers), int(task_count)))


def _merge_config_dicts(
    base_config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(base_config)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _normalize_sweep_config(
    base_config: dict[str, Any] | None,
    config: dict[str, Any],
    *,
    sweep_root: str | Path,
) -> dict[str, Any]:
    merged = _merge_config_dicts(default_config(), base_config)
    merged = _merge_config_dicts(merged, config)
    config_name = str(merged.get("config_name", "")).strip()
    if not config_name:
        raise ValueError("Each sweep config must define a non-empty 'config_name'.")
    merged["config_name"] = config_name
    merged["out_root"] = (
        Path(sweep_root).expanduser().resolve() / config_name / "results"
    )
    return merged


def _resolve_sweep_max_workers(
    configs: list[dict[str, Any]],
    max_config_workers: int | None,
) -> int:
    task_count = len(configs)
    if task_count <= 0:
        return 0
    if max_config_workers is not None:
        return _resolve_max_workers(max_config_workers, task_count)
    cpu_count = os.cpu_count() or 1
    per_config_workers = max(int(config.get("max_workers") or 1) for config in configs)
    return max(1, min(task_count, max(1, cpu_count // max(per_config_workers, 1))))


def _read_tsv_if_exists(path: str | Path) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if isinstance(path, float) and pd.isna(path):
        return pd.DataFrame()
    if isinstance(path, str) and not path.strip():
        return pd.DataFrame()
    path = Path(path)
    if not path.exists() or path.is_dir():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _format_bp_label(bp: int) -> str:
    bp = int(bp)
    if bp >= 1_000_000:
        value = bp / 1_000_000
        text = f"{value:.2f}".rstrip("0").rstrip(".")
        return f"{text}mb"
    if bp >= 1_000:
        value = bp / 1_000
        text = f"{value:.2f}".rstrip("0").rstrip(".")
        return f"{text}kb"
    return f"{bp}bp"


def _sample_log_uniform_int(
    rng: np.random.Generator,
    low: int,
    high: int,
) -> int:
    if low <= 0 or high <= 0:
        raise ValueError("Log-uniform sampling requires positive bounds.")
    value = float(np.exp(rng.uniform(np.log(low), np.log(high))))
    return int(round(min(max(value, low), high)))


def _sample_window_specs(
    rng: np.random.Generator,
    *,
    min_windows: int = 1,
    max_windows: int = 4,
    min_bp: int = 5_000,
    max_bp: int = 2_000_000,
) -> list[tuple[int, str]]:
    n_windows = int(rng.integers(min_windows, max_windows + 1))
    window_sizes = {
        _sample_log_uniform_int(rng, min_bp, max_bp) for _ in range(n_windows)
    }
    while len(window_sizes) < n_windows:
        window_sizes.add(_sample_log_uniform_int(rng, min_bp, max_bp))
    return [(int(bp), _format_bp_label(int(bp))) for bp in sorted(window_sizes)]


def default_config() -> dict[str, Any]:
    return {
        "samples": list(DEFAULT_SAMPLE_NAMES),
        "out_root": DEFAULT_WORKSPACE_RESULTS_DIR,
        "clean_min_cpgs": 10,
        "clean_min_region_length": 10_000,
        "clean_merge_gap_bp": 200_000,
        "wgbs_window_specs": [(10_000, "10kb")],
        "hm450_window_specs": [(10_000, "10kb")],
        "wgbs_hmm_type": "sticky",
        "wgbs_hmm_params": {},
        "hm450_hmm_type": "ct",
        "hm450_hmm_params": {
            "n_emissions": 4,
            "holding_time_guess": 300_000,
            "algorithm": "forward-backward",
            "max_iter": 25,
            "tol": 1e-2,
        },
        "min_coverage": 10,
        "n_states": 4,
        "int_low_cutoff": 0.2,
        "int_high_cutoff": 0.7,
        "high_cutoff": 0.7,
        "random_state": 42,
        "max_workers": min(os.cpu_count() or 1, 4),
        "run_chromatin": True,
        "run_chromatin_deeptools": True,
        "chromatin_include_heatmaps": False,
        "run_lad": True,
        "run_synthetic_prep": True,
        "run_synthetic_aggregate": True,
        "synthetic_overwrite": False,
        "synthetic_sample_ids": list(DEFAULT_SYNTHETIC_SAMPLE_IDS),
        "clear_out_dirs_before_run": False,
        "force_recreate": False,
        "print_logs": False,
    }


def generate_random_search_config_variants(
    *,
    n_configs: int = 25,
    seed: int = 42,
    config_name_prefix: str = "random",
) -> list[dict[str, Any]]:
    if n_configs <= 0:
        raise ValueError("n_configs must be at least 1.")

    rng = np.random.default_rng(seed)
    variants = []
    for idx in range(1, n_configs + 1):
        window_specs = _sample_window_specs(rng)
        holding_time_guess = int(rng.integers(100, 2_000_001))
        clean_min_cpgs = int(rng.integers(0, 101))
        clean_min_region_length = int(rng.integers(0, 200_001))
        clean_merge_gap_bp = int(rng.integers(0, 300_001))
        variants.append(
            {
                "config_name": f"{config_name_prefix}_{idx:02d}",
                "wgbs_window_specs": list(window_specs),
                "hm450_window_specs": list(window_specs),
                "hm450_hmm_params": {
                    "holding_time_guess": holding_time_guess,
                },
                "clean_min_cpgs": clean_min_cpgs,
                "clean_min_region_length": clean_min_region_length,
                "clean_merge_gap_bp": clean_merge_gap_bp,
            }
        )
    return variants


def sample_manifest_dataframe(samples: list[str] | None = None) -> pd.DataFrame:
    samples = list(samples or DEFAULT_SAMPLE_NAMES)
    rows = []
    for sample in samples:
        if sample not in DEFAULT_SAMPLE_SPECS:
            raise KeyError(f"Unsupported sample in notebook defaults: {sample}")
        spec = DEFAULT_SAMPLE_SPECS[sample]
        rows.append(
            {
                "sample": sample,
                "sample_id": _sample_to_sample_id(sample),
                "genome": spec["genome"],
                "meth_file": str(spec["meth_file"]),
            }
        )
    return pd.DataFrame(rows).sort_values("sample").reset_index(drop=True)


def write_config_snapshot(config: dict[str, Any]) -> Path:
    out_root = _results_root(config["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    config_path = out_root / "run_config.yaml"
    with open(config_path, "w") as handle:
        yaml.safe_dump(_sanitize_for_yaml(config), handle, sort_keys=False)
    return config_path


def methylseg_region_path(
    out_root: str | Path,
    sample: str,
    tool_name: str,
) -> Path:
    config = METHYLSEG_TOOL_CONFIGS_BY_NAME[tool_name]
    path = _results_root(out_root)
    for part in config["path_parts"]:
        path = path / part.format(sample=sample)
    return path


def _build_comparator_runner(
    *,
    sample: str,
    meth_file: str | Path,
    genome: str,
    out_root: str | Path,
    config: dict[str, Any],
) -> Any:
    core_class, state_assignment_method_cls, comparator_class = (
        _import_methylseg_components()
    )
    global CORE_METHYLSEG_CLASS
    CORE_METHYLSEG_CLASS = core_class
    runner = comparator_class(
        sample_id=sample,
        meth_file=meth_file,
        genome=genome,
        out_dir=_results_root(out_root) / "methylseg",
        force_recreate=bool(config.get("force_recreate", False)),
        print_logs=bool(config.get("print_logs", False)),
        wgbs_window_specs=list(config["wgbs_window_specs"]),
        hm450_window_specs=list(config["hm450_window_specs"]),
        min_coverage=int(config["min_coverage"]),
        n_states=int(config["n_states"]),
        int_low_cutoff=float(config["int_low_cutoff"]),
        int_high_cutoff=float(config["int_high_cutoff"]),
        high_cutoff=float(config["high_cutoff"]),
        train_sample=None,
        train_sample_file=None,
        train_sample_info=None,
        random_state=int(config["random_state"]),
        hm450_hmm_type=str(config["hm450_hmm_type"]),
        hm450_hmm_params=dict(config["hm450_hmm_params"]),
        wgbs_hmm_type=str(config["wgbs_hmm_type"]),
        wgbs_hmm_params=dict(config["wgbs_hmm_params"]),
        fit_methyl_seg=True,
        state_assignment_method=state_assignment_method_cls.KMEANS,
    )
    runner.methylseg_clean_min_cpgs = int(config["clean_min_cpgs"])
    runner.methylseg_clean_min_region_length = int(config["clean_min_region_length"])
    runner.methylseg_clean_merge_gap_bp = int(config["clean_merge_gap_bp"])
    return runner


def run_methylseg_for_sample(
    *,
    sample: str,
    meth_file: str | Path,
    genome: str,
    out_root: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    runner = _build_comparator_runner(
        sample=sample,
        meth_file=meth_file,
        genome=genome,
        out_root=out_root,
        config=config,
    )
    runner.run_data_prep()
    runner.run_tool()

    wgbs_bed = methylseg_region_path(out_root, sample, "methylseg")
    hm450_bed = methylseg_region_path(out_root, sample, "methylseg_hm450k")

    if not wgbs_bed.exists():
        raise FileNotFoundError(f"Missing WGBS MethylSeg output: {wgbs_bed}")
    if not hm450_bed.exists():
        raise FileNotFoundError(f"Missing HM450K MethylSeg output: {hm450_bed}")

    return {
        "sample": sample,
        "sample_id": _sample_to_sample_id(sample),
        "genome": genome,
        "meth_file": str(Path(meth_file).resolve()),
        "wgbs_pmd_bed": str(wgbs_bed),
        "hm450k_pmd_bed": str(hm450_bed),
        "prep_config": str(
            _results_root(out_root) / "methylseg" / sample / "prep" / "config.yaml"
        ),
    }


def _run_methylseg_for_sample_worker(task: dict[str, Any]) -> dict[str, Any]:
    sample = task["sample"]
    print(f"[methylsegtests] Starting MethylSeg sample run: {sample}")
    result = run_methylseg_for_sample(
        sample=sample,
        meth_file=task["meth_file"],
        genome=task["genome"],
        out_root=task["out_root"],
        config=task["config"],
    )
    print(f"[methylsegtests] Finished MethylSeg sample run: {sample}")
    return result


def run_methylseg_batch(config: dict[str, Any]) -> pd.DataFrame:
    config = dict(config)
    out_root = _results_root(config["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    write_config_snapshot(config)
    _clear_outputs_if_requested(
        enabled=bool(config.get("clear_out_dirs_before_run", False)),
        run_root=out_root,
        cleanup_fn=_cleanup_main_outputs_for_config_change,
        label="main methylseg results",
    )
    _invalidate_outputs_if_config_changed(
        run_root=out_root,
        cleanup_fn=_cleanup_main_outputs_for_config_change,
        config=config,
        label="main methylseg results",
    )

    tasks = []
    for sample in config["samples"]:
        spec = DEFAULT_SAMPLE_SPECS[sample]
        tasks.append(
            {
                "sample": sample,
                "meth_file": spec["meth_file"],
                "genome": spec["genome"],
                "out_root": out_root,
                "config": config,
            }
        )
    max_workers = _resolve_max_workers(config.get("max_workers"), len(tasks))
    print(
        f"[methylsegtests] Running {len(tasks)} primary sample(s) with up to {max_workers} worker(s)."
    )
    rows = []
    if max_workers <= 1:
        for task in tasks:
            rows.append(_run_methylseg_for_sample_worker(task))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_sample = {
                executor.submit(_run_methylseg_for_sample_worker, task): task["sample"]
                for task in tasks
            }
            for future in as_completed(future_to_sample):
                rows.append(future.result())

    manifest_df = pd.DataFrame(rows).sort_values("sample").reset_index(drop=True)
    manifest_path = out_root / "methylseg_run_manifest.tsv"
    manifest_df.to_csv(manifest_path, sep="\t", index=False)
    _write_run_metadata(out_root, config)
    return manifest_df


def summarize_methylseg_outputs(
    samples: list[str],
    out_root: str | Path,
) -> pd.DataFrame:
    lad = _import_lad_components()
    read_bed_intervals = lad["read_bed_intervals"]
    rows = []
    for sample in samples:
        for tool_name in ["methylseg", "methylseg_hm450k"]:
            tool_path = methylseg_region_path(out_root, sample, tool_name)
            region_df = read_bed_intervals(tool_path)
            rows.append(
                {
                    "sample": sample,
                    "tool": tool_name,
                    "tool_label": METHYLSEG_TOOL_CONFIGS_BY_NAME[tool_name][
                        "tool_label"
                    ],
                    "n_regions": int(len(region_df)),
                    "total_bp": (
                        int(region_df["length"].sum()) if not region_df.empty else 0
                    ),
                    "median_region_bp": (
                        float(region_df["length"].median())
                        if not region_df.empty
                        else np.nan
                    ),
                    "path": str(tool_path),
                }
            )
    summary_df = (
        pd.DataFrame(rows).sort_values(["sample", "tool"]).reset_index(drop=True)
    )
    summary_path = _results_root(out_root) / "methylseg_region_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    return summary_df


def run_chromatin_fallback(
    *,
    samples: list[str],
    out_root: str | Path,
    chromatin_data_dir: str | Path = CHROMATIN_DATA_DIR,
    run_deeptools: bool = True,
    include_heatmaps: bool = False,
) -> dict[str, Any]:
    lad = _import_lad_components()
    chromatin = _import_chromatin_components()
    prepare_sample_background_task = chromatin["prepare_sample_background_task"]
    compute_tool_metrics_task = chromatin["compute_tool_metrics_task"]
    prepare_deeptools_regions_task = chromatin["prepare_deeptools_regions_task"]
    run_deeptools_for_sample_task = chromatin["run_deeptools_for_sample_task"]
    out_root = _results_root(out_root)
    output_dir = out_root / "functional_analysis" / "chromatin_fallback"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    cleaned_region_dir = output_dir / "cleaned_regions"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    deeptools_dir = output_dir / "deeptools"
    for directory in [
        output_dir,
        cleaned_region_dir,
        tables_dir,
        figures_dir,
        deeptools_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    background_rows = []
    metric_rows = []
    manifest_rows = []
    for sample in samples:
        background = prepare_sample_background_task(
            {
                "sample": sample,
                "chromatin_data_dir": str(Path(chromatin_data_dir).resolve()),
                "cleaned_region_dir": str(cleaned_region_dir),
                "canonical_chromosomes": CANONICAL_CHROMOSOMES,
            }
        )
        background_rows.append(background)
        for tool_config in METHYLSEG_TOOL_CONFIGS:
            result = compute_tool_metrics_task(
                {
                    "sample": sample,
                    "tool_config": tool_config,
                    "segmentation_results_path": str(out_root),
                    "cleaned_region_dir": str(cleaned_region_dir),
                    "bw_path": background["bw_path"],
                    "peak_path": background["peak_path"],
                    "merged_peak_path": background["merged_peak_path"],
                    "eligible_genome_bp": background["eligible_genome_bp"],
                    "peak_baseline_fraction": background["peak_baseline_fraction"],
                    "genomewide_mean_h3k36me2_signal": background[
                        "genomewide_mean_h3k36me2_signal"
                    ],
                    "chrom_sizes": background["chrom_sizes"],
                }
            )
            manifest_rows.append(result["manifest_row"])
            metric_rows.append(result["metric_row"])

    background_df = (
        pd.DataFrame(background_rows).sort_values("sample").reset_index(drop=True)
    )
    manifest_df = (
        pd.DataFrame(manifest_rows)
        .sort_values(["sample", "deeptools_order", "tool"])
        .reset_index(drop=True)
    )
    metrics_df = (
        pd.DataFrame(metric_rows)
        .sort_values(["sample", "deeptools_order", "tool"])
        .reset_index(drop=True)
    )

    background_df.to_csv(
        tables_dir / "chromatin_background_metrics.tsv", sep="\t", index=False
    )
    manifest_df.to_csv(
        tables_dir / "chromatin_region_manifest.tsv", sep="\t", index=False
    )
    metrics_df.to_csv(
        tables_dir / "chromatin_methylseg_metrics.tsv", sep="\t", index=False
    )

    deeptools_region_df = pd.DataFrame()
    deeptools_outputs_df = pd.DataFrame()
    if run_deeptools:
        missing_deeptools = [
            command
            for command in ["computeMatrix", "plotProfile"]
            + (["plotHeatmap"] if include_heatmaps else [])
            if shutil.which(command) is None
        ]
        if missing_deeptools:
            raise RuntimeError(
                "deepTools commands are not available on PATH: "
                + ", ".join(missing_deeptools)
            )

        deeptools_region_rows = []
        for row in manifest_df.itertuples(index=False):
            deeptools_region_rows.append(
                prepare_deeptools_regions_task(
                    {
                        "sample": row.sample,
                        "sample_id": row.sample_id,
                        "tool": row.tool,
                        "tool_label": row.tool_label,
                        "tool_family": row.tool_family,
                        "platform": row.platform,
                        "region_type": row.region_type,
                        "deeptools_order": int(row.deeptools_order),
                        "clean_region_path": row.clean_region_path,
                        "deeptools_dir": str(deeptools_dir),
                        "deeptools_bin_size": int(DEFAULT_DEEPTOOLS_BIN_SIZE),
                    }
                )
            )
        deeptools_region_df = (
            pd.DataFrame(deeptools_region_rows)
            .sort_values(["sample", "deeptools_order", "tool"])
            .reset_index(drop=True)
        )
        deeptools_region_df.to_csv(
            tables_dir / "deeptools_region_manifest.tsv",
            sep="\t",
            index=False,
        )

        deeptools_rows = []
        selected_tool_order = [config["tool"] for config in METHYLSEG_TOOL_CONFIGS]
        for sample in samples:
            sample_region_rows = deeptools_region_df.loc[
                deeptools_region_df["sample"].eq(sample)
            ].to_dict(orient="records")
            background_row = background_df.loc[background_df["sample"].eq(sample)].iloc[
                0
            ]
            deeptools_rows.append(
                run_deeptools_for_sample_task(
                    {
                        "sample": sample,
                        "sample_id": background_row["sample_id"],
                        "bw_path": background_row["bw_path"],
                        "region_rows": sample_region_rows,
                        "deeptools_tool_order": selected_tool_order,
                        "deeptools_dir": str(deeptools_dir),
                        "deeptools_bin_size": int(DEFAULT_DEEPTOOLS_BIN_SIZE),
                        "flank_length": int(DEFAULT_FLANK_LENGTH),
                        "region_body_length": int(DEFAULT_REGION_BODY_LENGTH),
                        "deeptools_force": True,
                        "include_heatmaps": bool(include_heatmaps),
                    }
                )
            )
        deeptools_outputs_df = (
            pd.DataFrame(deeptools_rows).sort_values("sample").reset_index(drop=True)
        )
        deeptools_outputs_df.to_csv(
            tables_dir / "deeptools_outputs.tsv",
            sep="\t",
            index=False,
        )

    signal_fig = _make_bar_figure(
        metrics_df,
        x="sample",
        y="signal_ratio",
        color="tool_label",
        title="MethylSeg-only Chromatin Signal Ratio",
        output_stem=figures_dir / "chromatin_signal_ratio",
    )
    peak_fig = _make_bar_figure(
        metrics_df,
        x="sample",
        y="peak_ratio",
        color="tool_label",
        title="MethylSeg-only Chromatin Peak Enrichment Ratio",
        output_stem=figures_dir / "chromatin_peak_ratio",
    )

    return {
        "background_df": background_df,
        "manifest_df": manifest_df,
        "metrics_df": metrics_df,
        "deeptools_region_df": deeptools_region_df,
        "deeptools_outputs_df": deeptools_outputs_df,
        "signal_fig": signal_fig,
        "peak_fig": peak_fig,
        "output_dir": output_dir,
    }


def run_lad_fallback(
    *,
    samples: list[str],
    out_root: str | Path,
    primary_window_bp: int = 150_000,
) -> dict[str, Any]:
    lad = _import_lad_components()
    prepare_lad_reference = lad["prepare_lad_reference"]
    resolve_sample_genome = lad["resolve_sample_genome"]
    sample_to_sample_id = lad["sample_to_sample_id"]
    load_tool_regions = lad["load_tool_regions"]
    clean_and_merge_intervals = lad["clean_and_merge_intervals"]
    compute_end_window_overlap_metrics = lad["compute_end_window_overlap_metrics"]
    out_root = _results_root(out_root)
    output_dir = out_root / "functional_analysis" / "lad_fallback"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    reference_dir = output_dir / "reference_tracks"
    cleaned_region_dir = output_dir / "cleaned_regions"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    for directory in [
        output_dir,
        reference_dir,
        cleaned_region_dir,
        tables_dir,
        figures_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    sample_rows = []
    for sample in samples:
        sample_rows.append(
            {
                "sample": sample,
                "sample_id": sample_to_sample_id(sample),
                "genome": resolve_sample_genome(
                    sample,
                    segmentation_results_path=out_root,
                ),
            }
        )
    sample_df = pd.DataFrame(sample_rows).sort_values("sample").reset_index(drop=True)

    reference_lookup = {}
    for genome in sorted(sample_df["genome"].unique()):
        reference_lookup[genome] = prepare_lad_reference(
            genome=genome,
            reference_dir=reference_dir,
            lad_interval_track_path=lad["DEFAULT_LAD_INTERVAL_TRACK_PATH"],
            laminb1_signal_track_path=lad["DEFAULT_LAMINB1_SIGNAL_TRACK_PATH"],
            liftover_script_path=lad["DEFAULT_LIFTOVER_SCRIPT_PATH"],
            hg19_to_hg38_chain=lad["DEFAULT_HG19_TO_HG38_CHAIN"],
            force=False,
        )

    manifest_rows = []
    metric_rows = []
    for row in sample_df.itertuples(index=False):
        for tool_config in METHYLSEG_TOOL_CONFIGS:
            region_df = load_tool_regions(
                tool_config,
                row.sample,
                segmentation_results_path=out_root,
            )
            clean_region_df = clean_and_merge_intervals(region_df)
            clean_region_path = (
                cleaned_region_dir / f"{row.sample}.{tool_config['tool']}.bed"
            )
            clean_region_df.loc[:, ["chrom", "start", "end"]].to_csv(
                clean_region_path,
                sep="\t",
                header=False,
                index=False,
            )
            manifest_rows.append(
                {
                    "sample": row.sample,
                    "sample_id": row.sample_id,
                    "genome": row.genome,
                    "tool": tool_config["tool"],
                    "tool_label": tool_config["tool_label"],
                    "platform": tool_config["platform"],
                    "region_type": tool_config["region_type"],
                    "rank_order": tool_config["rank_order"],
                    "clean_region_path": str(clean_region_path),
                    "n_regions": int(len(clean_region_df)),
                    "total_bp": (
                        int(clean_region_df["length"].sum())
                        if not clean_region_df.empty
                        else 0
                    ),
                }
            )
            metric_rows.append(
                {
                    "sample": row.sample,
                    "sample_id": row.sample_id,
                    "genome": row.genome,
                    "tool": tool_config["tool"],
                    "tool_label": tool_config["tool_label"],
                    "platform": tool_config["platform"],
                    "region_type": tool_config["region_type"],
                    "rank_order": tool_config["rank_order"],
                    "window_bp": int(primary_window_bp),
                    **compute_end_window_overlap_metrics(
                        clean_region_df,
                        reference_lookup[row.genome]["lad_df"],
                        int(primary_window_bp),
                    ),
                }
            )

    manifest_df = (
        pd.DataFrame(manifest_rows)
        .sort_values(["sample", "rank_order"])
        .reset_index(drop=True)
    )
    metrics_df = (
        pd.DataFrame(metric_rows)
        .sort_values(["sample", "rank_order"])
        .reset_index(drop=True)
    )
    sample_df.to_csv(tables_dir / "lad_sample_manifest.tsv", sep="\t", index=False)
    manifest_df.to_csv(tables_dir / "lad_region_manifest.tsv", sep="\t", index=False)
    metrics_df.to_csv(tables_dir / "lad_metrics.tsv", sep="\t", index=False)

    fig = _make_bar_figure(
        metrics_df,
        x="sample",
        y="pct_regions_with_lad_at_either_end",
        color="tool_label",
        title="MethylSeg-only LAD End-Window Overlap",
        output_stem=figures_dir / "lad_end_window_overlap",
    )

    return {
        "sample_df": sample_df,
        "manifest_df": manifest_df,
        "metrics_df": metrics_df,
        "fig": fig,
        "output_dir": output_dir,
    }


def synthetic_root(out_root: str | Path) -> Path:
    return _results_root(out_root) / "synthetic"


def _existing_synthetic_inputs_ready(
    *,
    out_root: str | Path,
    sample_ids: list[str] | None = None,
) -> tuple[bool, pd.DataFrame | None]:
    syn_root = synthetic_root(out_root)
    manifest_path = syn_root / "synthetic_samples" / "synthetic_sample_manifest.tsv"
    config_dir = syn_root / "tool_results" / "configs"
    config_list_path = syn_root / "tool_results" / "configs.txt"

    if not manifest_path.exists():
        return False, None

    manifest_df = (
        pd.read_csv(manifest_path, sep="\t")
        .sort_values("synthetic_sample_id")
        .reset_index(drop=True)
    )
    if manifest_df.empty:
        return False, None

    selected_ids = list(sample_ids or [])
    if selected_ids:
        manifest_df = manifest_df.loc[
            manifest_df["synthetic_sample_id"].isin(selected_ids)
        ].copy()
        if manifest_df.empty or set(selected_ids) - set(
            manifest_df["synthetic_sample_id"]
        ):
            return False, None

    required_columns = [
        "synthetic_sample_id",
        "normalized_source_file",
        "synthetic_file",
        "truth_bed",
    ]
    missing_columns = [
        col for col in required_columns if col not in manifest_df.columns
    ]
    if missing_columns:
        return False, None

    for row in manifest_df.itertuples(index=False):
        required_paths = [
            Path(row.normalized_source_file),
            Path(row.synthetic_file),
            Path(row.truth_bed),
            config_dir / f"{row.synthetic_sample_id}.yaml",
        ]
        if not all(path.exists() for path in required_paths):
            return False, None

    if not config_list_path.exists():
        return False, None

    return True, manifest_df.reset_index(drop=True)


def prepare_synthetic_inputs(
    *,
    out_root: str | Path,
    overwrite: bool = False,
    sample_ids: list[str] | None = None,
) -> pd.DataFrame:
    if not overwrite:
        ready, manifest_df = _existing_synthetic_inputs_ready(
            out_root=out_root,
            sample_ids=sample_ids,
        )
        if ready and manifest_df is not None:
            print(
                "[methylsegtests] Reusing existing synthetic samples and configs; "
                "no synthetic dataprep rerun was needed."
            )
            return manifest_df

    syn_root = synthetic_root(out_root)
    cmd = [
        sys.executable,
        str(SYNTHETIC_ANALYSIS_DIR / "slurm_code" / "run_synthetic_dataprep.py"),
        "--out-root",
        str(syn_root),
    ]
    if overwrite:
        cmd.append("--overwrite")
    for sample_id in sample_ids or []:
        cmd.extend(["--sample-id", sample_id])
    subprocess.run(cmd, check=True)
    return load_synthetic_manifest(out_root)


def load_synthetic_manifest(out_root: str | Path) -> pd.DataFrame:
    manifest_path = (
        synthetic_root(out_root) / "synthetic_samples" / "synthetic_sample_manifest.tsv"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Synthetic manifest not found. Run prepare_synthetic_inputs first: {manifest_path}"
        )
    return (
        pd.read_csv(manifest_path, sep="\t")
        .sort_values("synthetic_sample_id")
        .reset_index(drop=True)
    )


def run_synthetic_methylseg(
    *,
    out_root: str | Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    manifest_df = load_synthetic_manifest(out_root)
    selected_ids = config.get("synthetic_sample_ids")
    if selected_ids:
        manifest_df = manifest_df.loc[
            manifest_df["synthetic_sample_id"].isin(selected_ids)
        ].copy()
    if manifest_df.empty:
        raise ValueError("No synthetic samples selected for MethylSeg execution.")

    synthetic_tool_root = synthetic_root(out_root) / "tool_results"
    synthetic_tool_root.mkdir(parents=True, exist_ok=True)
    _clear_outputs_if_requested(
        enabled=bool(config.get("clear_out_dirs_before_run", False)),
        run_root=synthetic_tool_root,
        cleanup_fn=lambda _: _cleanup_synthetic_outputs_for_config_change(out_root),
        label="synthetic methylseg results",
    )
    _invalidate_outputs_if_config_changed(
        run_root=synthetic_tool_root,
        cleanup_fn=lambda _: _cleanup_synthetic_outputs_for_config_change(out_root),
        config=config,
        label="synthetic methylseg results",
    )

    tasks = []
    for row in manifest_df.itertuples(index=False):
        tasks.append(
            {
                "sample": row.synthetic_sample_id,
                "meth_file": row.synthetic_file,
                "genome": row.source_genome,
                "out_root": synthetic_tool_root,
                "config": config,
            }
        )
    max_workers = _resolve_max_workers(config.get("max_workers"), len(tasks))
    print(
        f"[methylsegtests] Running {len(tasks)} synthetic sample(s) with up to {max_workers} worker(s)."
    )
    rows = []
    if max_workers <= 1:
        for task in tasks:
            rows.append(_run_methylseg_for_sample_worker(task))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_sample = {
                executor.submit(_run_methylseg_for_sample_worker, task): task["sample"]
                for task in tasks
            }
            for future in as_completed(future_to_sample):
                rows.append(future.result())
    result_df = pd.DataFrame(rows).sort_values("sample").reset_index(drop=True)
    result_df.to_csv(
        synthetic_root(out_root) / "tool_results" / "methylseg_run_manifest.tsv",
        sep="\t",
        index=False,
    )
    _write_run_metadata(synthetic_tool_root, config)
    return result_df


def summarize_synthetic_methylseg(
    *,
    out_root: str | Path,
    sample_ids: list[str] | None = None,
) -> dict[str, Any]:
    lad = _import_lad_components()
    clean_and_merge_intervals = lad["clean_and_merge_intervals"]
    read_bed_intervals = lad["read_bed_intervals"]
    load_tool_regions = lad["load_tool_regions"]
    manifest_df = load_synthetic_manifest(out_root)
    if sample_ids:
        manifest_df = manifest_df.loc[
            manifest_df["synthetic_sample_id"].isin(sample_ids)
        ].copy()
    if manifest_df.empty:
        raise ValueError("No synthetic samples selected for summary.")

    tool_results_root = synthetic_root(out_root) / "tool_results"
    summary_rows = []
    for row in manifest_df.itertuples(index=False):
        truth_df = clean_and_merge_intervals(read_bed_intervals(row.truth_bed))
        total_truth_bp = int(truth_df["length"].sum()) if not truth_df.empty else 0
        for tool_config in METHYLSEG_TOOL_CONFIGS:
            region_df = clean_and_merge_intervals(
                load_tool_regions(
                    tool_config,
                    row.synthetic_sample_id,
                    segmentation_results_path=tool_results_root,
                )
            )
            total_call_bp = int(region_df["length"].sum()) if not region_df.empty else 0
            overlap_bp = _compute_interval_overlap_bp(region_df, truth_df)
            summary_rows.append(
                {
                    "sample": row.synthetic_sample_id,
                    "tool": tool_config["tool"],
                    "tool_label": tool_config["tool_label"],
                    "n_truth_regions": int(len(truth_df)),
                    "n_called_regions": int(len(region_df)),
                    "truth_total_bp": total_truth_bp,
                    "called_total_bp": total_call_bp,
                    "overlap_bp": overlap_bp,
                    "bp_recall": (
                        overlap_bp / total_truth_bp if total_truth_bp else np.nan
                    ),
                    "bp_precision": (
                        overlap_bp / total_call_bp if total_call_bp else np.nan
                    ),
                    "truth_bed": str(row.truth_bed),
                }
            )

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["sample", "tool"])
        .reset_index(drop=True)
    )
    output_dir = synthetic_root(out_root) / "analysis"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(
        tables_dir / "synthetic_methylseg_summary.tsv",
        sep="\t",
        index=False,
    )

    recall_fig = _make_bar_figure(
        summary_df,
        x="sample",
        y="bp_recall",
        color="tool_label",
        title="Synthetic MethylSeg BP Recall",
        output_stem=figures_dir / "synthetic_bp_recall",
    )
    precision_fig = _make_bar_figure(
        summary_df,
        x="sample",
        y="bp_precision",
        color="tool_label",
        title="Synthetic MethylSeg BP Precision",
        output_stem=figures_dir / "synthetic_bp_precision",
    )

    return {
        "summary_df": summary_df,
        "recall_fig": recall_fig,
        "precision_fig": precision_fig,
        "output_dir": output_dir,
    }


def run_full_methylseg_config(config: dict[str, Any]) -> dict[str, Any]:
    config = dict(config)
    out_root = _results_root(config["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    config_path = write_config_snapshot(config)

    primary_manifest_df = run_methylseg_batch(config)
    primary_summary_df = summarize_methylseg_outputs(
        samples=config["samples"],
        out_root=out_root,
    )

    chromatin_results = None
    if config.get("run_chromatin", True):
        chromatin_results = run_chromatin_fallback(
            samples=config["samples"],
            out_root=out_root,
            run_deeptools=bool(config.get("run_chromatin_deeptools", True)),
            include_heatmaps=bool(config.get("chromatin_include_heatmaps", False)),
        )

    lad_results = None
    if config.get("run_lad", True):
        lad_results = run_lad_fallback(
            samples=config["samples"],
            out_root=out_root,
        )

    synthetic_manifest_df = None
    synthetic_methylseg_manifest_df = None
    synthetic_summary = None
    if config.get("run_synthetic_prep", True):
        synthetic_manifest_df = prepare_synthetic_inputs(
            out_root=out_root,
            overwrite=bool(config.get("synthetic_overwrite", False)),
            sample_ids=config.get("synthetic_sample_ids"),
        )

    if config.get("run_synthetic_aggregate", True):
        synthetic_methylseg_manifest_df = run_synthetic_methylseg(
            out_root=out_root,
            config=config,
        )
        synthetic_summary = summarize_synthetic_methylseg(
            out_root=out_root,
            sample_ids=config.get("synthetic_sample_ids"),
        )

    return {
        "config_name": config["config_name"],
        "status": "success",
        "out_root": str(out_root),
        "config_path": str(config_path),
        "config_hash": _config_hash(config),
        "methylseg_manifest_path": str(out_root / "methylseg_run_manifest.tsv"),
        "methylseg_summary_path": str(out_root / "methylseg_region_summary.tsv"),
        "chromatin_metrics_path": str(
            out_root
            / "functional_analysis"
            / "chromatin_fallback"
            / "tables"
            / "chromatin_methylseg_metrics.tsv"
        ),
        "chromatin_deeptools_outputs_path": str(
            out_root
            / "functional_analysis"
            / "chromatin_fallback"
            / "tables"
            / "deeptools_outputs.tsv"
        ),
        "lad_metrics_path": str(
            out_root
            / "functional_analysis"
            / "lad_fallback"
            / "tables"
            / "lad_metrics.tsv"
        ),
        "synthetic_manifest_path": str(
            synthetic_root(out_root)
            / "synthetic_samples"
            / "synthetic_sample_manifest.tsv"
        ),
        "synthetic_methylseg_manifest_path": str(
            synthetic_root(out_root) / "tool_results" / "methylseg_run_manifest.tsv"
        ),
        "synthetic_summary_path": str(
            synthetic_root(out_root)
            / "analysis"
            / "tables"
            / "synthetic_methylseg_summary.tsv"
        ),
        "n_primary_samples": int(len(primary_manifest_df)),
        "n_primary_summary_rows": int(len(primary_summary_df)),
        "n_chromatin_rows": (
            int(len(chromatin_results["metrics_df"])) if chromatin_results else 0
        ),
        "n_lad_rows": int(len(lad_results["metrics_df"])) if lad_results else 0,
        "n_synthetic_samples": (
            int(len(synthetic_manifest_df)) if synthetic_manifest_df is not None else 0
        ),
        "n_synthetic_runs": (
            int(len(synthetic_methylseg_manifest_df))
            if synthetic_methylseg_manifest_df is not None
            else 0
        ),
        "n_synthetic_summary_rows": (
            int(len(synthetic_summary["summary_df"]))
            if synthetic_summary is not None
            else 0
        ),
        "error": "",
        "traceback": "",
    }


def _run_full_methylseg_config_worker(config: dict[str, Any]) -> dict[str, Any]:
    config_name = config["config_name"]
    print(f"[methylsegtests] Starting config run: {config_name}")
    try:
        result = run_full_methylseg_config(config)
        print(f"[methylsegtests] Finished config run: {config_name}")
        return result
    except Exception as exc:  # pragma: no cover - multiprocessing failure path
        print(f"[methylsegtests] Failed config run: {config_name}: {exc}")
        return {
            "config_name": config_name,
            "status": "failed",
            "out_root": str(_results_root(config["out_root"])),
            "config_path": str(_results_root(config["out_root"]) / "run_config.yaml"),
            "config_hash": _config_hash(config),
            "methylseg_manifest_path": "",
            "methylseg_summary_path": "",
            "chromatin_metrics_path": "",
            "chromatin_deeptools_outputs_path": "",
            "lad_metrics_path": "",
            "synthetic_manifest_path": "",
            "synthetic_methylseg_manifest_path": "",
            "synthetic_summary_path": "",
            "n_primary_samples": 0,
            "n_primary_summary_rows": 0,
            "n_chromatin_rows": 0,
            "n_lad_rows": 0,
            "n_synthetic_samples": 0,
            "n_synthetic_runs": 0,
            "n_synthetic_summary_rows": 0,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _build_sweep_manifest_row(
    config: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    row = {
        "config_name": config["config_name"],
        "status": result["status"],
        "out_root": str(_results_root(config["out_root"])),
        "config_hash": _config_hash(config),
        "samples": ",".join(config["samples"]),
        "max_workers": int(config.get("max_workers") or 1),
        "run_chromatin": bool(config.get("run_chromatin", True)),
        "run_lad": bool(config.get("run_lad", True)),
        "run_synthetic_prep": bool(config.get("run_synthetic_prep", True)),
        "run_synthetic_aggregate": bool(config.get("run_synthetic_aggregate", True)),
        "clean_min_cpgs": int(config["clean_min_cpgs"]),
        "clean_min_region_length": int(config["clean_min_region_length"]),
        "clean_merge_gap_bp": int(config["clean_merge_gap_bp"]),
        "wgbs_window_specs": yaml.safe_dump(config["wgbs_window_specs"]).strip(),
        "hm450_window_specs": yaml.safe_dump(config["hm450_window_specs"]).strip(),
        "wgbs_hmm_type": str(config["wgbs_hmm_type"]),
        "wgbs_hmm_params": yaml.safe_dump(config["wgbs_hmm_params"]).strip(),
        "hm450_hmm_type": str(config["hm450_hmm_type"]),
        "hm450_hmm_params": yaml.safe_dump(config["hm450_hmm_params"]).strip(),
        "error": result.get("error", ""),
    }
    return row


def run_methylseg_config_sweep(
    configs: list[dict[str, Any]],
    *,
    base_config: dict[str, Any] | None = None,
    sweep_root: str | Path | None = None,
    max_config_workers: int | None = None,
) -> dict[str, Any]:
    if not configs:
        raise ValueError("At least one configuration is required for a sweep.")

    sweep_root = Path(sweep_root or DEFAULT_SWEEP_ROOT).expanduser().resolve()
    normalized_configs = [
        _normalize_sweep_config(base_config, config, sweep_root=sweep_root)
        for config in configs
    ]
    config_names = [config["config_name"] for config in normalized_configs]
    duplicate_names = sorted(
        {name for name in config_names if config_names.count(name) > 1}
    )
    if duplicate_names:
        raise ValueError(
            "Duplicate sweep config_name values are not allowed: "
            + ", ".join(duplicate_names)
        )

    sweep_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    result_rows = []
    worker_count = _resolve_sweep_max_workers(normalized_configs, max_config_workers)
    print(
        f"[methylsegtests] Running {len(normalized_configs)} config(s) with up to {worker_count} config worker(s)."
    )

    if worker_count <= 1:
        for config in normalized_configs:
            result = _run_full_methylseg_config_worker(config)
            result_rows.append(result)
            manifest_rows.append(_build_sweep_manifest_row(config, result))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_config = {
                executor.submit(_run_full_methylseg_config_worker, config): config
                for config in normalized_configs
            }
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                result = future.result()
                result_rows.append(result)
                manifest_rows.append(_build_sweep_manifest_row(config, result))

    manifest_df = (
        pd.DataFrame(manifest_rows, columns=SWEEP_MANIFEST_COLUMNS)
        .sort_values("config_name")
        .reset_index(drop=True)
    )
    results_df = (
        pd.DataFrame(result_rows).sort_values("config_name").reset_index(drop=True)
    )
    manifest_path = sweep_root / "sweep_manifest.tsv"
    results_path = sweep_root / "sweep_results.tsv"
    manifest_df.to_csv(manifest_path, sep="\t", index=False)
    results_df.to_csv(results_path, sep="\t", index=False)
    summary_payload = {
        "sweep_root": str(sweep_root),
        "max_config_workers": int(worker_count),
        "config_names": config_names,
        "manifest_path": str(manifest_path),
        "results_path": str(results_path),
    }
    with open(sweep_root / "sweep_summary.yaml", "w") as handle:
        yaml.safe_dump(summary_payload, handle, sort_keys=False)
    return {
        "sweep_root": sweep_root,
        "max_config_workers": worker_count,
        "manifest_df": manifest_df,
        "results_df": results_df,
        "normalized_configs": normalized_configs,
        "manifest_path": manifest_path,
        "results_path": results_path,
    }


def load_sweep_result_tables(
    *,
    sweep_results_df: pd.DataFrame,
) -> dict[str, dict[str, pd.DataFrame]]:
    tables: dict[str, dict[str, pd.DataFrame]] = {}
    for row in sweep_results_df.itertuples(index=False):
        tables[row.config_name] = {
            "methylseg_manifest_df": _read_tsv_if_exists(row.methylseg_manifest_path),
            "methylseg_summary_df": _read_tsv_if_exists(row.methylseg_summary_path),
            "chromatin_metrics_df": _read_tsv_if_exists(row.chromatin_metrics_path),
            "chromatin_deeptools_outputs_df": _read_tsv_if_exists(
                row.chromatin_deeptools_outputs_path
            ),
            "lad_metrics_df": _read_tsv_if_exists(row.lad_metrics_path),
            "synthetic_manifest_df": _read_tsv_if_exists(row.synthetic_manifest_path),
            "synthetic_methylseg_manifest_df": _read_tsv_if_exists(
                row.synthetic_methylseg_manifest_path
            ),
            "synthetic_summary_df": _read_tsv_if_exists(row.synthetic_summary_path),
        }
    return tables


__all__ = [
    "CHROMATIN_DATA_DIR",
    "CORE_METHYLSEG_CLASS",
    "DATA_DIR",
    "DEFAULT_SAMPLE_NAMES",
    "DEFAULT_SWEEP_ROOT",
    "DEFAULT_SYNTHETIC_SAMPLE_IDS",
    "DEFAULT_WORKSPACE_RESULTS_DIR",
    "METHYLSEG_TOOL_CONFIGS",
    "TRACKED_CONFIG_KEYS",
    "default_config",
    "generate_random_search_config_variants",
    "load_sweep_result_tables",
    "load_synthetic_manifest",
    "methylseg_region_path",
    "prepare_synthetic_inputs",
    "run_chromatin_fallback",
    "run_full_methylseg_config",
    "run_lad_fallback",
    "run_methylseg_batch",
    "run_methylseg_config_sweep",
    "run_synthetic_methylseg",
    "sample_manifest_dataframe",
    "summarize_methylseg_outputs",
    "summarize_synthetic_methylseg",
    "write_config_snapshot",
]
