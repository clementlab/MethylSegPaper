import argparse
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_path in [str(SCRIPT_DIR), str(PROJECT_ROOT)]:
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

matplotlib_cache_dir = Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"
matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(matplotlib_cache_dir)
env_bin = str(Path(sys.executable).resolve().parent)
path_entries = (
    os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
)
if env_bin not in path_entries:
    os.environ["PATH"] = os.pathsep.join([env_bin] + path_entries)

from chromatin_analysis_utils import (
    build_source_id,
    build_tool_region_path,
    configure_pybedtools,
    prepare_deeptools_regions_task,
    prepare_region_task,
    resolve_bigwig_path,
    run_deeptools_for_source_task,
    sample_to_sample_id,
)
from repo_paths import (
    CHROMATIN_DATA_DIR,
    CHROMATIN_RESULTS_DIR,
    REGION_CALLING_RESULTS_DIR,
)

DEFAULT_SAMPLE_NAMES = ["ESO26.wgbs", "TE5.wgbs"]
DEFAULT_SEGMENTATION_RESULTS_PATH = REGION_CALLING_RESULTS_DIR
DEFAULT_CHROMATIN_DATA_DIR = CHROMATIN_DATA_DIR
DEFAULT_OUTPUT_DIR = CHROMATIN_RESULTS_DIR

TOOL_REGISTRY = [
    {
        "tool": "methylseg",
        "tool_label": "MethylSeg WGBS",
        "parser_family": "methylseg",
        "tool_family": "methylseg",
        "platform": "wgbs",
        "region_type": "PMD",
        "deeptools_order": 0,
        "default_region_variant": "cleaned",
        "region_paths": {
            "raw": [
                "methylseg",
                "{sample}",
                "out",
                "wgbs",
                "summary_files",
                "segments_raw_PMD.bed",
            ],
            "cleaned": [
                "methylseg",
                "{sample}",
                "out",
                "wgbs",
                "summary_files",
                "segments_cleaned_PMD.bed",
            ],
        },
    },
    {
        "tool": "methylseg_hm450k",
        "tool_label": "MethylSeg HM450K",
        "parser_family": "methylseg",
        "tool_family": "methylseg",
        "platform": "hm450k",
        "region_type": "PMD",
        "deeptools_order": 1,
        "default_region_variant": "cleaned",
        "region_paths": {
            "raw": [
                "methylseg",
                "{sample}",
                "out",
                "hm450k",
                "summary_files",
                "segments_raw_PMD.bed",
            ],
            "cleaned": [
                "methylseg",
                "{sample}",
                "out",
                "hm450k",
                "summary_files",
                "segments_cleaned_PMD.bed",
            ],
        },
    },
    {
        "tool": "methylseekr",
        "tool_label": "MethylSeekR",
        "parser_family": "methylseekr",
        "tool_family": "methylseekr",
        "platform": "wgbs",
        "region_type": "PMD",
        "deeptools_order": 2,
        "default_region_variant": "source-default",
        "region_paths": {
            "source-default": [
                "methylseekr",
                "{sample}",
                "out",
                "methylseekr_PMDs.bed",
            ],
        },
    },
    {
        "tool": "dnmtools",
        "tool_label": "DNMTools",
        "parser_family": "dnmtools",
        "tool_family": "dnmtools",
        "platform": "wgbs",
        "region_type": "PMD",
        "deeptools_order": 3,
        "default_region_variant": "source-default",
        "region_paths": {
            "source-default": ["dnmtools", "{sample}", "out", "dnmtools_PMDs.bed"],
        },
    },
    {
        "tool": "dnmtools_array",
        "tool_label": "DNMTools Array",
        "parser_family": "dnmtools",
        "tool_family": "dnmtools",
        "platform": "hm450k",
        "region_type": "PMD",
        "deeptools_order": 4,
        "default_region_variant": "source-default",
        "region_paths": {
            "source-default": [
                "dnmtools",
                "{sample}",
                "out",
                "arraymode.dnmtools_PMDs.bed",
            ],
        },
    },
    {
        "tool": "dnmtools_pmr",
        "tool_label": "DNMTools PMR",
        "parser_family": "dnmtools",
        "tool_family": "dnmtools",
        "platform": "wgbs",
        "region_type": "PMR",
        "deeptools_order": 5,
        "default_region_variant": "source-default",
        "region_paths": {
            "source-default": ["dnmtools", "{sample}", "out", "pmr.dnmtools_PMDs.bed"],
        },
    },
    {
        "tool": "mmseekr",
        "tool_label": "MMSeekR",
        "parser_family": "mmseekr",
        "tool_family": "mmseekr",
        "platform": "wgbs",
        "region_type": "PMD",
        "deeptools_order": 6,
        "default_region_variant": "source-default",
        "region_paths": {
            "source-default": [
                "mmseekr",
                "{sample}",
                "out",
                "{sample}.multiModel.PMDs.bed",
            ],
        },
    },
    {
        "tool": "methyl_lasso",
        "tool_label": "MethylLasso",
        "parser_family": "methyl_lasso",
        "tool_family": "methyl_lasso",
        "platform": "wgbs",
        "region_type": "PMD",
        "deeptools_order": 7,
        "default_region_variant": "source-default",
        "region_paths": {
            "source-default": ["methyl_lasso", "{sample}", "out", "{sample}_pmd.tsv"],
        },
    },
]

TOOL_CONFIG_BY_NAME = {config["tool"]: config for config in TOOL_REGISTRY}
REQUIRED_TOOLS = [config["tool"] for config in TOOL_REGISTRY]
REQUIRED_TOOL_SET = set(REQUIRED_TOOLS)
DEFAULT_DEEPTOOLS_TOOLS = [
    config["tool"]
    for config in sorted(TOOL_REGISTRY, key=lambda config: config["deeptools_order"])
]
CANONICAL_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
REGION_BODY_LENGTH = 500_000
WGBS_FLANK_LENGTH = 200_000
ARRAY_FLANK_LENGTH = 500_000
WGBS_DEEPTOOLS_BIN_SIZE = 25_000
ARRAY_DEEPTOOLS_BIN_SIZE = 50_000

SOURCE_LAYOUT = [
    ("methylseg", "cleaned"),
    ("methylseekr", "source-default"),
    ("dnmtools", "source-default"),
    ("dnmtools_pmr", "source-default"),
    ("mmseekr", "source-default"),
    ("methyl_lasso", "source-default"),
    ("methylseg_hm450k", "cleaned"),
    ("dnmtools_array", "source-default"),
    ("methylseg", "raw"),
    ("methylseg_hm450k", "raw"),
]


def _build_required_region_specs(selected_tools=None):
    selected_tool_set = set(selected_tools) if selected_tools is not None else None
    required_specs = []
    for tool, region_variant in SOURCE_LAYOUT:
        if selected_tool_set is not None and tool not in selected_tool_set:
            continue
        tool_config = TOOL_CONFIG_BY_NAME[tool]
        source_order = len(required_specs)
        if tool_config["platform"] == "hm450k":
            flank_length = ARRAY_FLANK_LENGTH
            deeptools_bin_size = ARRAY_DEEPTOOLS_BIN_SIZE
        else:
            flank_length = WGBS_FLANK_LENGTH
            deeptools_bin_size = WGBS_DEEPTOOLS_BIN_SIZE
        required_specs.append(
            {
                "tool": tool,
                "region_variant": region_variant,
                "source_id": build_source_id(tool, region_variant),
                "source_order": source_order,
                "tool_config": tool_config,
                "deeptools_bin_size": deeptools_bin_size,
                "flank_length": flank_length,
                "region_body_length": REGION_BODY_LENGTH,
            }
        )
    return required_specs


def _available_cpu_count():
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(int(slurm_cpus), 1)
        except ValueError:
            pass
    return max(os.cpu_count() or 1, 1)


def _print_dataframe(title, dataframe, columns=None):
    print(f"\n=== {title} ===")
    if dataframe.empty:
        print("<empty>")
        return
    display_df = dataframe if columns is None else dataframe.loc[:, columns]
    print(display_df.to_string(index=False))


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
        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=mp.get_context("fork")
        ) as executor:
            return list(executor.map(worker, tasks))
    except Exception as exc:
        print(
            f"Falling back to sequential execution for {stage_name} "
            f"because multiprocessing failed: {exc}"
        )
        return [worker(task) for task in tasks]


def _assert_source_membership(dataframe, expected_specs, dataframe_name):
    if dataframe.empty:
        raise AssertionError(f"{dataframe_name} is empty.")
    expected_source_ids = [spec["source_id"] for spec in expected_specs]
    for sample, sample_df in dataframe.groupby("sample", sort=True):
        observed_source_ids = sample_df.sort_values("source_order")[
            "source_id"
        ].tolist()
        if observed_source_ids != expected_source_ids:
            raise AssertionError(
                f"Unexpected source membership for {dataframe_name} in {sample}. "
                f"Expected={expected_source_ids}; Observed={observed_source_ids}"
            )


def _assert_source_membership_by_sample(
    dataframe, expected_source_ids_by_sample, dataframe_name
):
    if dataframe.empty:
        raise AssertionError(f"{dataframe_name} is empty.")
    expected_samples = sorted(expected_source_ids_by_sample)
    observed_samples = sorted(dataframe["sample"].astype(str).unique().tolist())
    if observed_samples != expected_samples:
        raise AssertionError(
            f"Unexpected samples for {dataframe_name}. "
            f"Expected={expected_samples}; Observed={observed_samples}"
        )
    for sample, expected_source_ids in expected_source_ids_by_sample.items():
        sample_df = dataframe.loc[dataframe["sample"].eq(sample)].copy()
        observed_source_ids = sample_df.sort_values("source_order")[
            "source_id"
        ].tolist()
        if observed_source_ids != expected_source_ids:
            raise AssertionError(
                f"Unexpected source membership for {dataframe_name} in {sample}. "
                f"Expected={expected_source_ids}; Observed={observed_source_ids}"
            )


def _assert_source_paths(dataframe, dataframe_name):
    for row in dataframe.itertuples():
        source_path = str(row.source_region_path)
        if row.region_variant == "raw" and "segments_raw_PMD.bed" not in source_path:
            raise AssertionError(
                f"Expected raw MethylSeg path for {dataframe_name} {row.sample} {row.source_id}, got {source_path}"
            )
        if row.region_variant == "cleaned" and row.tool.startswith("methylseg"):
            if "segments_cleaned_PMD.bed" not in source_path:
                raise AssertionError(
                    f"Expected cleaned MethylSeg path for {dataframe_name} {row.sample} {row.source_id}, got {source_path}"
                )


def assert_expected_source_regions(dataframe, expected_specs):
    _assert_source_membership(dataframe, expected_specs, "chromatin_region_manifest")
    _assert_source_paths(dataframe, "chromatin_region_manifest")


def assert_expected_deeptools_region_rows(dataframe, expected_specs):
    _assert_source_membership(dataframe, expected_specs, "deeptools_region_manifest")
    _assert_source_paths(dataframe, "deeptools_region_manifest")
    expected_bin_sizes = {
        spec["source_id"]: int(spec["deeptools_bin_size"]) for spec in expected_specs
    }
    expected_flanks = {
        spec["source_id"]: int(spec["flank_length"]) for spec in expected_specs
    }
    expected_bodies = {
        spec["source_id"]: int(spec["region_body_length"]) for spec in expected_specs
    }
    for row in dataframe.itertuples():
        if int(row.min_region_length_bp) != expected_bin_sizes[row.source_id]:
            raise AssertionError(
                f"Unexpected deepTools bin size for deeptools_region_manifest {row.sample} {row.source_id}."
            )
        if int(row.flank_length) != expected_flanks[row.source_id]:
            raise AssertionError(
                f"Unexpected flank length for deeptools_region_manifest {row.sample} {row.source_id}."
            )
        if int(row.region_body_length) != expected_bodies[row.source_id]:
            raise AssertionError(
                f"Unexpected region body length for deeptools_region_manifest {row.sample} {row.source_id}."
            )


def _build_expected_output_source_ids_by_sample(deeptools_region_df):
    eligible_df = deeptools_region_df.loc[
        deeptools_region_df["visualized_regions"] > 0
    ].copy()
    if eligible_df.empty:
        return {}
    expected_source_ids_by_sample = {}
    for sample, sample_df in eligible_df.groupby("sample", sort=True):
        expected_source_ids_by_sample[str(sample)] = sample_df.sort_values(
            "source_order"
        )["source_id"].tolist()
    return expected_source_ids_by_sample


def validate_deeptools_layout(expected_specs):
    invalid_messages = []
    for spec in expected_specs:
        deeptools_bin_size = int(spec["deeptools_bin_size"])
        for field_name in ["flank_length", "region_body_length"]:
            value = int(spec[field_name])
            if value % deeptools_bin_size != 0:
                invalid_messages.append(
                    f"{spec['source_id']} has {field_name}={value}, which is not a multiple of "
                    f"deeptools_bin_size={deeptools_bin_size}."
                )
    if invalid_messages:
        raise ValueError(
            "deepTools layout is incompatible with computeMatrix scale-regions:\n"
            + "\n".join(invalid_messages)
        )


def assert_expected_output_rows(dataframe, deeptools_region_df, expected_specs):
    expected_source_ids_by_sample = _build_expected_output_source_ids_by_sample(
        deeptools_region_df
    )
    if not expected_source_ids_by_sample:
        raise AssertionError(
            "deeptools_outputs is empty because no sample/source pairs retained regions for deepTools."
        )
    _assert_source_membership_by_sample(
        dataframe, expected_source_ids_by_sample, "deeptools_outputs"
    )
    _assert_source_paths(dataframe, "deeptools_outputs")
    expected_bin_sizes = {
        spec["source_id"]: int(spec["deeptools_bin_size"]) for spec in expected_specs
    }
    expected_flanks = {
        spec["source_id"]: int(spec["flank_length"]) for spec in expected_specs
    }
    expected_bodies = {
        spec["source_id"]: int(spec["region_body_length"]) for spec in expected_specs
    }
    for row in dataframe.itertuples():
        if int(row.deeptools_bin_size) != expected_bin_sizes[row.source_id]:
            raise AssertionError(
                f"Unexpected deepTools bin size for deeptools_outputs {row.sample} {row.source_id}."
            )
        if int(row.flank_length) != expected_flanks[row.source_id]:
            raise AssertionError(
                f"Unexpected flank length for deeptools_outputs {row.sample} {row.source_id}."
            )
        if int(row.region_body_length) != expected_bodies[row.source_id]:
            raise AssertionError(
                f"Unexpected region body length for deeptools_outputs {row.sample} {row.source_id}."
            )


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Run chromatin deepTools analysis for comparator region outputs."
    )
    parser.add_argument(
        "--segmentation-results-path",
        type=Path,
        default=DEFAULT_SEGMENTATION_RESULTS_PATH,
        help="Directory containing the comparator pathway outputs.",
    )
    parser.add_argument(
        "--chromatin-data-dir",
        type=Path,
        default=DEFAULT_CHROMATIN_DATA_DIR,
        help="Directory containing sample-specific H3K36me2 bigWig files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where prepared BEDs, manifests, and deepTools outputs will be written. "
            "Defaults to the repo-local chromatin results directory."
        ),
    )
    parser.add_argument(
        "--skip-deeptools",
        action="store_true",
        help="Skip deepTools matrix, profile, and heatmap generation after preparing BED manifests.",
    )
    parser.add_argument(
        "--include-heatmaps",
        action="store_true",
        help="Generate deepTools heatmaps in addition to the default matrix and profile outputs.",
    )
    parser.add_argument(
        "--deeptools-tools",
        nargs="+",
        choices=REQUIRED_TOOLS,
        default=list(DEFAULT_DEEPTOOLS_TOOLS),
        help="Tools to include in chromatin deepTools output. All requested variants for the selected tools are run.",
    )
    return parser


def validate_required_inputs(
    segmentation_results_path, chromatin_data_dir, samples, source_specs
):
    missing_messages = []
    for sample in samples:
        bw_path = resolve_bigwig_path(chromatin_data_dir, sample)
        if not bw_path.exists():
            missing_messages.append(f"{sample}: missing chromatin bigWig {bw_path}")
        for source_spec in source_specs:
            raw_region_path = build_tool_region_path(
                segmentation_results_path,
                sample,
                source_spec["tool_config"],
                region_variant=source_spec["region_variant"],
            )
            if not raw_region_path.exists():
                missing_messages.append(
                    f"{sample}: missing {source_spec['tool']} {source_spec['region_variant']} region file {raw_region_path}"
                )

    if missing_messages:
        raise FileNotFoundError(
            "Chromatin analysis prerequisites are missing:\n"
            + "\n".join(missing_messages)
        )


def run(
    segmentation_results_path,
    chromatin_data_dir,
    output_dir,
    skip_deeptools=False,
    include_heatmaps=False,
    deeptools_tools=None,
):
    segmentation_results_path = Path(segmentation_results_path).resolve()
    chromatin_data_dir = Path(chromatin_data_dir).resolve()
    output_dir = Path(
        output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR
    ).resolve()

    prepared_region_dir = output_dir / "cleaned_regions"
    tables_dir = output_dir / "tables"
    deeptools_dir = output_dir / "deeptools"
    for directory in [output_dir, prepared_region_dir, tables_dir, deeptools_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    selected_deeptools_tools = list(
        dict.fromkeys(deeptools_tools or DEFAULT_DEEPTOOLS_TOOLS)
    )
    if not selected_deeptools_tools:
        raise ValueError("At least one tool must be selected for deepTools output.")
    invalid_deeptools_tools = sorted(set(selected_deeptools_tools) - REQUIRED_TOOL_SET)
    if invalid_deeptools_tools:
        raise ValueError(
            "Unsupported chromatin deepTools tools: "
            + ", ".join(invalid_deeptools_tools)
        )

    required_region_specs = _build_required_region_specs(
        selected_tools=selected_deeptools_tools
    )
    validate_deeptools_layout(required_region_specs)

    available_cpus = _available_cpu_count()
    tool_workers = min(
        len(DEFAULT_SAMPLE_NAMES) * len(required_region_specs), available_cpus
    )
    source_workers = min(
        len(DEFAULT_SAMPLE_NAMES) * len(required_region_specs), available_cpus
    )
    run_deeptools = not skip_deeptools

    print("Chromatin analysis configuration:")
    print(f"  Segmentation results: {segmentation_results_path}")
    print(f"  Chromatin data dir:   {chromatin_data_dir}")
    print(f"  Output dir:           {output_dir}")
    print(f"  Samples:              {', '.join(DEFAULT_SAMPLE_NAMES)}")
    print(f"  Available CPUs:       {available_cpus}")
    print(f"  Region workers:       {tool_workers}")
    print(f"  Source workers:       {source_workers}")
    print(f"  Run deepTools:        {run_deeptools}")
    print(f"  Include heatmaps:     {include_heatmaps}")
    print(f"  WGBS bin size:        {WGBS_DEEPTOOLS_BIN_SIZE}")
    print(f"  Array bin size:       {ARRAY_DEEPTOOLS_BIN_SIZE}")
    print(f"  deepTools tools:      {', '.join(selected_deeptools_tools)}")

    validate_required_inputs(
        segmentation_results_path=segmentation_results_path,
        chromatin_data_dir=chromatin_data_dir,
        samples=DEFAULT_SAMPLE_NAMES,
        source_specs=required_region_specs,
    )

    bedtools_bin = configure_pybedtools()
    if bedtools_bin is None:
        print(
            "bedtools executable was not found on PATH; pybedtools operations may fail."
        )
    else:
        print(f"Using bedtools from {bedtools_bin}")

    region_tasks = []
    for sample in DEFAULT_SAMPLE_NAMES:
        for source_spec in required_region_specs:
            region_tasks.append(
                {
                    "sample": sample,
                    "tool_config": source_spec["tool_config"],
                    "region_variant": source_spec["region_variant"],
                    "source_order": int(source_spec["source_order"]),
                    "segmentation_results_path": str(segmentation_results_path),
                    "chromatin_data_dir": str(chromatin_data_dir),
                    "prepared_region_dir": str(prepared_region_dir),
                    "canonical_chromosomes": CANONICAL_CHROMOSOMES,
                }
            )

    region_rows = run_parallel(
        region_tasks,
        prepare_region_task,
        tool_workers,
        "chromatin region preparation",
    )
    manifest_df = (
        pd.DataFrame(region_rows)
        .sort_values(["sample", "source_order"])
        .reset_index(drop=True)
    )
    if manifest_df.empty:
        raise RuntimeError("No region sets were prepared for chromatin analysis.")
    assert_expected_source_regions(manifest_df, required_region_specs)
    if (manifest_df["n_regions"] <= 0).any():
        raise AssertionError(
            "Some tool/sample region variants produced zero retained regions after preparation."
        )

    region_manifest_path = tables_dir / "chromatin_region_manifest.tsv"
    manifest_df.to_csv(region_manifest_path, sep="\t", index=False)
    _print_dataframe(
        "Chromatin region manifest",
        manifest_df,
        [
            "sample",
            "source_id",
            "tool_variant_label",
            "platform",
            "region_type",
            "region_variant",
            "n_regions",
            "total_bp",
            "prepared_region_path",
        ],
    )

    manifest_lookup = manifest_df.set_index(["sample", "source_id"])
    deeptools_region_tasks = []
    for sample in DEFAULT_SAMPLE_NAMES:
        for source_spec in required_region_specs:
            row = manifest_lookup.loc[(sample, source_spec["source_id"])]
            deeptools_region_tasks.append(
                {
                    "sample": sample,
                    "sample_id": row["sample_id"],
                    "source_order": int(source_spec["source_order"]),
                    "tool": source_spec["tool"],
                    "source_id": source_spec["source_id"],
                    "tool_label": row["tool_label"],
                    "tool_variant_label": row["tool_variant_label"],
                    "tool_family": row["tool_family"],
                    "platform": row["platform"],
                    "region_type": row["region_type"],
                    "region_variant": row["region_variant"],
                    "source_region_path": row["source_region_path"],
                    "prepared_region_path": row["prepared_region_path"],
                    "deeptools_dir": str(deeptools_dir),
                    "deeptools_bin_size": int(source_spec["deeptools_bin_size"]),
                    "flank_length": int(source_spec["flank_length"]),
                    "region_body_length": int(source_spec["region_body_length"]),
                }
            )

    deeptools_region_rows = run_parallel(
        deeptools_region_tasks,
        prepare_deeptools_regions_task,
        tool_workers,
        "deepTools region BED preparation",
    )
    deeptools_region_df = (
        pd.DataFrame(deeptools_region_rows)
        .sort_values(["sample", "source_order"])
        .reset_index(drop=True)
    )
    deeptools_region_manifest_path = tables_dir / "deeptools_region_manifest.tsv"
    deeptools_region_df.to_csv(deeptools_region_manifest_path, sep="\t", index=False)
    assert_expected_deeptools_region_rows(deeptools_region_df, required_region_specs)
    _print_dataframe(
        "deepTools region manifest",
        deeptools_region_df,
        [
            "sample",
            "source_id",
            "tool_variant_label",
            "visualized_regions",
            "excluded_short_regions",
            "deeptools_region_path",
        ],
    )

    deeptools_outputs_path = tables_dir / "deeptools_outputs.tsv"
    if not run_deeptools:
        print("Skipping deepTools execution because --skip-deeptools was provided.")
        print("\nChromatin analysis complete.")
        print(f"  Region manifest:           {region_manifest_path}")
        print(f"  deepTools region manifest: {deeptools_region_manifest_path}")
        return

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

    skipped_source_df = deeptools_region_df.loc[
        deeptools_region_df["visualized_regions"] <= 0
    ].copy()
    if not skipped_source_df.empty:
        _print_dataframe(
            "Skipping deepTools for source rows with zero retained regions",
            skipped_source_df,
            [
                "sample",
                "source_id",
                "tool_variant_label",
                "total_regions",
                "excluded_short_regions",
                "min_region_length_bp",
            ],
        )

    eligible_deeptools_region_df = deeptools_region_df.loc[
        deeptools_region_df["visualized_regions"] > 0
    ].copy()
    if eligible_deeptools_region_df.empty:
        raise RuntimeError(
            "No sample/source pairs retained regions for deepTools after platform-specific minimum-length filtering."
        )

    deeptools_region_lookup = eligible_deeptools_region_df.set_index(
        ["sample", "source_id"]
    )
    deeptools_source_tasks = []
    for sample in DEFAULT_SAMPLE_NAMES:
        bw_path = resolve_bigwig_path(chromatin_data_dir, sample)
        sample_id = sample_to_sample_id(sample)
        for source_spec in required_region_specs:
            source_key = (sample, source_spec["source_id"])
            if source_key not in deeptools_region_lookup.index:
                continue
            row = deeptools_region_lookup.loc[source_key]
            tool_variant_label = str(row["tool_variant_label"])
            deeptools_source_tasks.append(
                {
                    "sample": sample,
                    "sample_id": sample_id,
                    "source_order": int(source_spec["source_order"]),
                    "tool": source_spec["tool"],
                    "source_id": source_spec["source_id"],
                    "tool_label": row["tool_label"],
                    "tool_variant_label": tool_variant_label,
                    "tool_family": row["tool_family"],
                    "platform": row["platform"],
                    "region_type": row["region_type"],
                    "region_variant": row["region_variant"],
                    "source_region_path": row["source_region_path"],
                    "prepared_region_path": row["prepared_region_path"],
                    "deeptools_region_path": row["deeptools_region_path"],
                    "visualized_regions": int(row["visualized_regions"]),
                    "bw_path": str(bw_path),
                    "deeptools_dir": str(deeptools_dir),
                    "deeptools_bin_size": int(source_spec["deeptools_bin_size"]),
                    "flank_length": int(source_spec["flank_length"]),
                    "region_body_length": int(source_spec["region_body_length"]),
                    "include_heatmaps": bool(include_heatmaps),
                    "profile_title": f"{sample} H3K36me2 profile - {tool_variant_label}",
                    "heatmap_title": f"{sample} H3K36me2 heatmap - {tool_variant_label}",
                }
            )

    deeptools_rows = run_parallel(
        deeptools_source_tasks,
        run_deeptools_for_source_task,
        source_workers,
        (
            "deepTools source matrix and plot generation"
            if include_heatmaps
            else "deepTools source matrix and profile generation"
        ),
    )
    deeptools_outputs_df = (
        pd.DataFrame(deeptools_rows)
        .sort_values(["sample", "source_order"])
        .reset_index(drop=True)
    )
    assert_expected_output_rows(
        deeptools_outputs_df, deeptools_region_df, required_region_specs
    )
    deeptools_outputs_df.to_csv(deeptools_outputs_path, sep="\t", index=False)
    _print_dataframe(
        "deepTools outputs",
        deeptools_outputs_df,
        [
            "sample",
            "source_id",
            "matrix_path",
            "sorted_regions_path",
            "profile_path",
            "heatmap_path",
        ],
    )

    print("\nChromatin analysis complete.")
    print(f"  Region manifest:           {region_manifest_path}")
    print(f"  deepTools region manifest: {deeptools_region_manifest_path}")
    print(f"  deepTools outputs:         {deeptools_outputs_path}")


def main(argv=None):
    args = _build_parser().parse_args(argv)
    run(
        segmentation_results_path=args.segmentation_results_path,
        chromatin_data_dir=args.chromatin_data_dir,
        output_dir=args.output_dir,
        skip_deeptools=args.skip_deeptools,
        include_heatmaps=args.include_heatmaps,
        deeptools_tools=args.deeptools_tools,
    )


if __name__ == "__main__":
    main()
