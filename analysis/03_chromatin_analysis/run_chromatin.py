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
path_entries = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
if env_bin not in path_entries:
    os.environ["PATH"] = os.pathsep.join([env_bin] + path_entries)

from chromatin_analysis_utils import (
    build_tool_region_path,
    configure_pybedtools,
    prepare_clean_region_task,
    prepare_deeptools_regions_task,
    resolve_bigwig_path,
    run_deeptools_for_sample_task,
    sample_to_sample_id,
)
from repo_paths import CHROMATIN_DATA_DIR, CHROMATIN_RESULTS_DIR, REGION_CALLING_RESULTS_DIR


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
        "parser_family": "methylseg",
        "tool_family": "methylseg",
        "platform": "hm450k",
        "region_type": "PMD",
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
    {
        "tool": "methylseekr",
        "tool_label": "MethylSeekR",
        "parser_family": "methylseekr",
        "tool_family": "methylseekr",
        "platform": "wgbs",
        "region_type": "PMD",
        "deeptools_order": 2,
        "path_parts": ["methylseekr", "{sample}", "out", "methylseekr_PMDs.bed"],
    },
    {
        "tool": "dnmtools",
        "tool_label": "DNMTools",
        "parser_family": "dnmtools",
        "tool_family": "dnmtools",
        "platform": "wgbs",
        "region_type": "PMD",
        "deeptools_order": 3,
        "path_parts": ["dnmtools", "{sample}", "out", "dnmtools_PMDs.bed"],
    },
    {
        "tool": "dnmtools_array",
        "tool_label": "DNMTools Array",
        "parser_family": "dnmtools",
        "tool_family": "dnmtools",
        "platform": "hm450k",
        "region_type": "PMD",
        "deeptools_order": 4,
        "path_parts": ["dnmtools", "{sample}", "out", "arraymode.dnmtools_PMDs.bed"],
    },
    {
        "tool": "dnmtools_pmr",
        "tool_label": "DNMTools PMR",
        "parser_family": "dnmtools",
        "tool_family": "dnmtools",
        "platform": "wgbs",
        "region_type": "PMR",
        "deeptools_order": 5,
        "path_parts": ["dnmtools", "{sample}", "out", "pmr.dnmtools_PMDs.bed"],
    },
    {
        "tool": "mmseekr",
        "tool_label": "MMSeekR",
        "parser_family": "mmseekr",
        "tool_family": "mmseekr",
        "platform": "wgbs",
        "region_type": "PMD",
        "deeptools_order": 6,
        "path_parts": ["mmseekr", "{sample}", "out", "{sample}.multiModel.PMDs.bed"],
    },
    {
        "tool": "methyl_lasso",
        "tool_label": "MethylLasso",
        "parser_family": "methyl_lasso",
        "tool_family": "methyl_lasso",
        "platform": "wgbs",
        "region_type": "PMD",
        "deeptools_order": 7,
        "path_parts": ["methyl_lasso", "{sample}", "out", "{sample}_pmd.tsv"],
    },
]

TOOL_CONFIG_BY_NAME = {config["tool"]: config for config in TOOL_REGISTRY}
REQUIRED_TOOLS = [config["tool"] for config in TOOL_REGISTRY]
REQUIRED_TOOL_SET = set(REQUIRED_TOOLS)
DEFAULT_DEEPTOOLS_TOOLS = [
    config["tool"] for config in sorted(TOOL_REGISTRY, key=lambda config: config["deeptools_order"])
]
CANONICAL_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
REGION_BODY_LENGTH = 1_000_000
FLANK_LENGTH = 500_000
DEEPTOOLS_BIN_SIZE = 1000


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

    print(f"Running {stage_name} with {max_workers} workers across {len(tasks)} task(s).")
    try:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("fork")) as executor:
            return list(executor.map(worker, tasks))
    except Exception as exc:
        print(
            f"Falling back to sequential execution for {stage_name} "
            f"because multiprocessing failed: {exc}"
        )
        return [worker(task) for task in tasks]


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
            "Directory where cleaned BEDs, manifests, and deepTools outputs will be written. "
            "Defaults to the repo-local chromatin results directory."
        ),
    )
    parser.add_argument(
        "--skip-deeptools",
        action="store_true",
        help="Skip deepTools matrix and profile generation.",
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
        help="Tools to include in chromatin deepTools plots. Defaults to all tools.",
    )
    return parser


def validate_required_inputs(segmentation_results_path, chromatin_data_dir, samples):
    missing_messages = []
    for sample in samples:
        bw_path = resolve_bigwig_path(chromatin_data_dir, sample)
        if not bw_path.exists():
            missing_messages.append(f"{sample}: missing chromatin bigWig {bw_path}")
        for tool_config in TOOL_REGISTRY:
            raw_region_path = build_tool_region_path(segmentation_results_path, sample, tool_config)
            if not raw_region_path.exists():
                missing_messages.append(
                    f"{sample}: missing {tool_config['tool']} region file {raw_region_path}"
                )

    if missing_messages:
        raise FileNotFoundError(
            "Chromatin analysis prerequisites are missing:\n" + "\n".join(missing_messages)
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
    output_dir = Path(output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR).resolve()

    cleaned_region_dir = output_dir / "cleaned_regions"
    tables_dir = output_dir / "tables"
    deeptools_dir = output_dir / "deeptools"
    for directory in [output_dir, cleaned_region_dir, tables_dir, deeptools_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    selected_deeptools_tools = list(dict.fromkeys(deeptools_tools or DEFAULT_DEEPTOOLS_TOOLS))
    if not selected_deeptools_tools:
        raise ValueError("At least one tool must be selected for deepTools output.")

    invalid_deeptools_tools = sorted(set(selected_deeptools_tools) - REQUIRED_TOOL_SET)
    if invalid_deeptools_tools:
        raise ValueError("Unsupported chromatin deepTools tools: " + ", ".join(invalid_deeptools_tools))

    available_cpus = _available_cpu_count()
    tool_workers = min(len(DEFAULT_SAMPLE_NAMES) * len(REQUIRED_TOOLS), available_cpus)
    sample_workers = min(len(DEFAULT_SAMPLE_NAMES), available_cpus)
    run_deeptools = not skip_deeptools

    print("Chromatin analysis configuration:")
    print(f"  Segmentation results: {segmentation_results_path}")
    print(f"  Chromatin data dir:   {chromatin_data_dir}")
    print(f"  Output dir:           {output_dir}")
    print(f"  Samples:              {', '.join(DEFAULT_SAMPLE_NAMES)}")
    print(f"  Available CPUs:       {available_cpus}")
    print(f"  Tool workers:         {tool_workers}")
    print(f"  Sample workers:       {sample_workers}")
    print(f"  Run deepTools:        {run_deeptools}")
    print(f"  Include heatmaps:     {include_heatmaps}")
    print(f"  deepTools tools:      {', '.join(selected_deeptools_tools)}")

    validate_required_inputs(
        segmentation_results_path=segmentation_results_path,
        chromatin_data_dir=chromatin_data_dir,
        samples=DEFAULT_SAMPLE_NAMES,
    )

    bedtools_bin = configure_pybedtools()
    if bedtools_bin is None:
        print("bedtools executable was not found on PATH; pybedtools operations may fail.")
    else:
        print(f"Using bedtools from {bedtools_bin}")

    clean_region_tasks = []
    for sample in DEFAULT_SAMPLE_NAMES:
        for tool_config in TOOL_REGISTRY:
            clean_region_tasks.append(
                {
                    "sample": sample,
                    "tool_config": tool_config,
                    "segmentation_results_path": str(segmentation_results_path),
                    "chromatin_data_dir": str(chromatin_data_dir),
                    "cleaned_region_dir": str(cleaned_region_dir),
                    "canonical_chromosomes": CANONICAL_CHROMOSOMES,
                }
            )

    region_rows = run_parallel(
        clean_region_tasks,
        prepare_clean_region_task,
        tool_workers,
        "cleaned region preparation",
    )
    manifest_df = pd.DataFrame(region_rows).sort_values(["sample", "deeptools_order", "tool"]).reset_index(drop=True)
    if manifest_df.empty:
        raise RuntimeError("No region sets were prepared for chromatin analysis.")
    assert_expected_tools(manifest_df, "chromatin_region_manifest")
    if (manifest_df["n_regions"] <= 0).any():
        raise AssertionError("Some tool/sample pairs produced zero retained regions after cleaning.")

    region_manifest_path = tables_dir / "chromatin_region_manifest.tsv"
    manifest_df.to_csv(region_manifest_path, sep="\t", index=False)
    _print_dataframe(
        "Chromatin region manifest",
        manifest_df,
        ["sample", "tool_label", "platform", "region_type", "n_regions", "total_bp", "clean_region_path"],
    )

    deeptools_outputs_path = tables_dir / "deeptools_outputs.tsv"
    if not run_deeptools:
        print("Skipping deepTools execution because --skip-deeptools was provided.")
        print("\nChromatin analysis complete.")
        print(f"  Region manifest:   {region_manifest_path}")
        return

    missing_deeptools = [
        command
        for command in ["computeMatrix", "plotProfile"] + (["plotHeatmap"] if include_heatmaps else [])
        if shutil.which(command) is None
    ]
    if missing_deeptools:
        raise RuntimeError("deepTools commands are not available on PATH: " + ", ".join(missing_deeptools))

    manifest_lookup = manifest_df.set_index(["sample", "tool"])
    deeptools_region_tasks = []
    for sample in DEFAULT_SAMPLE_NAMES:
        for tool in selected_deeptools_tools:
            row = manifest_lookup.loc[(sample, tool)]
            deeptools_region_tasks.append(
                {
                    "sample": sample,
                    "sample_id": row["sample_id"],
                    "tool": tool,
                    "tool_label": row["tool_label"],
                    "tool_family": row["tool_family"],
                    "platform": row["platform"],
                    "region_type": row["region_type"],
                    "deeptools_order": int(row["deeptools_order"]),
                    "clean_region_path": row["clean_region_path"],
                    "deeptools_dir": str(deeptools_dir),
                    "deeptools_bin_size": int(DEEPTOOLS_BIN_SIZE),
                }
            )

    deeptools_region_rows = run_parallel(
        deeptools_region_tasks,
        prepare_deeptools_regions_task,
        tool_workers,
        "deepTools region BED preparation",
    )
    deeptools_region_df = pd.DataFrame(deeptools_region_rows).sort_values(
        ["sample", "deeptools_order", "tool"]
    ).reset_index(drop=True)
    deeptools_region_manifest_path = tables_dir / "deeptools_region_manifest.tsv"
    deeptools_region_df.to_csv(deeptools_region_manifest_path, sep="\t", index=False)
    _print_dataframe(
        "deepTools region manifest",
        deeptools_region_df,
        [
            "sample",
            "tool_label",
            "visualized_regions",
            "excluded_short_regions",
            "deeptools_region_path",
        ],
    )

    deeptools_sample_tasks = []
    for sample in DEFAULT_SAMPLE_NAMES:
        bw_path = resolve_bigwig_path(chromatin_data_dir, sample)
        sample_region_rows = deeptools_region_df.loc[deeptools_region_df["sample"].eq(sample)].to_dict("records")
        deeptools_sample_tasks.append(
            {
                "sample": sample,
                "sample_id": sample_to_sample_id(sample),
                "bw_path": str(bw_path),
                "region_rows": sample_region_rows,
                "deeptools_tool_order": selected_deeptools_tools,
                "deeptools_dir": str(deeptools_dir),
                "deeptools_bin_size": int(DEEPTOOLS_BIN_SIZE),
                "flank_length": int(FLANK_LENGTH),
                "region_body_length": int(REGION_BODY_LENGTH),
                "include_heatmaps": bool(include_heatmaps),
            }
        )

    deeptools_rows = run_parallel(
        deeptools_sample_tasks,
        run_deeptools_for_sample_task,
        sample_workers,
        "deepTools matrix and plot generation" if include_heatmaps else "deepTools matrix and profile generation",
    )
    deeptools_outputs_df = pd.DataFrame(deeptools_rows).sort_values("sample").reset_index(drop=True)
    deeptools_outputs_df.to_csv(deeptools_outputs_path, sep="\t", index=False)
    _print_dataframe(
        "deepTools outputs",
        deeptools_outputs_df,
        ["sample", "matrix_path", "sorted_regions_path", "profile_path", "heatmap_path"],
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
