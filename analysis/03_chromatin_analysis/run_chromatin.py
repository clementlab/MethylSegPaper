import argparse
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chromatin_analysis_utils import (
    compute_tool_metrics_task,
    configure_pybedtools,
    prepare_deeptools_regions_task,
    prepare_sample_background_task,
    run_deeptools_for_sample_task,
    sample_to_sample_id,
)
from repo_paths import (
    CHROMATIN_DATA_DIR,
    CHROMATIN_RESULTS_DIR,
    REGION_CALLING_RESULTS_DIR,
)

sns.set_theme(style="whitegrid")

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
DEEPTOOLS_TOOL_ORDER = [
    config["tool"] for config in sorted(TOOL_REGISTRY, key=lambda config: config["deeptools_order"])
]
DEFAULT_DEEPTOOLS_TOOLS = list(DEEPTOOLS_TOOL_ORDER)
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


def run_parallel(tasks, worker, max_workers, stage_name, use_multiprocessing=True):
    tasks = list(tasks)
    if not tasks:
        return []

    max_workers = min(max_workers, len(tasks))
    if not use_multiprocessing or max_workers <= 1:
        print(f"Running {stage_name} sequentially across {len(tasks)} task(s).")
        return [worker(task) for task in tasks]

    print(f"Running {stage_name} with {max_workers} workers across {len(tasks)} task(s).")
    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("fork"),
        ) as executor:
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
        description="Run chromatin analysis for comparator outputs in batch mode."
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
        help="Directory containing sample-specific H3K36me2 bigWig and peak files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where tables, figures, and deepTools outputs will be written. "
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
        help=(
            "Tools to include in chromatin deepTools profile plots. "
            "Defaults to all tools."
        ),
    )
    return parser


def validate_required_inputs(segmentation_results_path, chromatin_data_dir, samples):
    segmentation_results_path = Path(segmentation_results_path)
    chromatin_data_dir = Path(chromatin_data_dir)
    missing_messages = []

    for sample in samples:
        bw_path, peak_path = resolve_chromatin_paths(chromatin_data_dir, sample)
        if not bw_path.exists():
            missing_messages.append(f"{sample}: missing chromatin bigWig {bw_path}")
        if not peak_path.exists():
            missing_messages.append(f"{sample}: missing chromatin peak file {peak_path}")

        for tool_config in TOOL_REGISTRY:
            raw_region_path = build_tool_region_path(
                segmentation_results_path, sample, tool_config
            )
            if not raw_region_path.exists():
                missing_messages.append(
                    f"{sample}: missing {tool_config['tool']} region file {raw_region_path}"
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
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir).resolve()

    cleaned_region_dir = output_dir / "cleaned_regions"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    deeptools_dir = output_dir / "deeptools"

    for directory in [output_dir, cleaned_region_dir, tables_dir, figures_dir, deeptools_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    available_cpus = _available_cpu_count()
    sample_workers = min(len(DEFAULT_SAMPLE_NAMES), available_cpus)
    run_deeptools = not skip_deeptools
    selected_deeptools_tools = list(dict.fromkeys(deeptools_tools or DEFAULT_DEEPTOOLS_TOOLS))
    if not selected_deeptools_tools:
        raise ValueError("At least one tool must be selected for chromatin deepTools plots.")
    invalid_deeptools_tools = sorted(set(selected_deeptools_tools) - REQUIRED_TOOL_SET)
    if invalid_deeptools_tools:
        raise ValueError(
            "Unsupported chromatin deepTools tools: " + ", ".join(invalid_deeptools_tools)
        )
    tool_workers = min(len(DEFAULT_SAMPLE_NAMES) * len(selected_deeptools_tools), available_cpus)
    deeptools_force = True

    print("Chromatin analysis configuration:")
    print(f"  Segmentation results: {segmentation_results_path}")
    print(f"  Chromatin data dir:   {chromatin_data_dir}")
    print(f"  Output dir:           {output_dir}")
    print(f"  Samples:              {', '.join(DEFAULT_SAMPLE_NAMES)}")
    print(f"  Available CPUs:       {available_cpus}")
    print(f"  Sample workers:       {sample_workers}")
    print(f"  Tool workers:         {tool_workers}")
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

    sample_background_tasks = [
        {
            "sample": sample,
            "chromatin_data_dir": str(chromatin_data_dir),
            "cleaned_region_dir": str(cleaned_region_dir),
            "canonical_chromosomes": CANONICAL_CHROMOSOMES,
        }
        for sample in DEFAULT_SAMPLE_NAMES
    ]
    sample_background_rows = run_parallel(
        sample_background_tasks,
        prepare_sample_background_task,
        sample_workers,
        "sample background preparation",
    )
    sample_background_rows = sorted(sample_background_rows, key=lambda row: row["sample"])
    sample_background = {row["sample"]: row for row in sample_background_rows}

    background_df = (
        pd.DataFrame(
            [
                {
                    "sample": row["sample"],
                    "sample_id": row["sample_id"],
                    "bw_path": row["bw_path"],
                    "peak_path": row["peak_path"],
                    "eligible_genome_bp": row["eligible_genome_bp"],
                    "peak_total_bp": row["peak_total_bp"],
                    "peak_baseline_fraction": row["peak_baseline_fraction"],
                    "genomewide_mean_h3k36me2_signal": row["genomewide_mean_h3k36me2_signal"],
                    "merged_peak_path": row["merged_peak_path"],
                }
                for row in sample_background_rows
            ]
        )
        .sort_values("sample")
        .reset_index(drop=True)
    )
    background_metrics_path = tables_dir / "chromatin_background_metrics.tsv"
    background_df.to_csv(background_metrics_path, sep="\t", index=False)
    _print_dataframe("Background metrics", background_df)

    tool_tasks = []
    for sample in DEFAULT_SAMPLE_NAMES:
        background = sample_background[sample]
        for tool in REQUIRED_TOOLS:
            tool_tasks.append(
                {
                    "sample": sample,
                    "tool_config": TOOL_CONFIG_BY_NAME[tool],
                    "segmentation_results_path": str(segmentation_results_path),
                    "cleaned_region_dir": str(cleaned_region_dir),
                    "bw_path": background["bw_path"],
                    "peak_path": background["peak_path"],
                    "merged_peak_path": background["merged_peak_path"],
                    "eligible_genome_bp": background["eligible_genome_bp"],
                    "peak_baseline_fraction": background["peak_baseline_fraction"],
                    "genomewide_mean_h3k36me2_signal": background["genomewide_mean_h3k36me2_signal"],
                    "chrom_sizes": background["chrom_sizes"],
                }
            )

    tool_results = run_parallel(
        tool_tasks,
        compute_tool_metrics_task,
        tool_workers,
        "tool region cleaning and metric calculation",
    )

    manifest_rows = [result["manifest_row"] for result in tool_results]
    manifest_df = pd.DataFrame(manifest_rows)
    if manifest_df.empty:
        raise RuntimeError(
            "No region sets were found for the requested samples. "
            f"samples={DEFAULT_SAMPLE_NAMES}; chromatin_data_dir={chromatin_data_dir}"
        )

    manifest_df = manifest_df.sort_values(["sample", "deeptools_order", "tool"]).reset_index(drop=True)
    if set(DEFAULT_SAMPLE_NAMES) != set(manifest_df["sample"].unique()):
        raise AssertionError(
            f"Expected samples {DEFAULT_SAMPLE_NAMES}, found {sorted(manifest_df['sample'].unique())}"
        )
    if (manifest_df.groupby(["sample", "tool"]).size() > 1).any():
        raise AssertionError("Expected exactly one cleaned BED per sample/tool combination.")
    assert_expected_tools(manifest_df, "manifest_df")
    if (manifest_df["n_regions"] <= 0).any():
        raise AssertionError("Some tool/sample pairs produced zero retained regions after cleaning.")

    region_manifest_path = tables_dir / "chromatin_region_manifest.tsv"
    manifest_df.to_csv(region_manifest_path, sep="\t", index=False)
    _print_dataframe(
        "Region manifest",
        manifest_df,
        [
            "sample",
            "tool_label",
            "platform",
            "region_type",
            "n_regions",
            "total_bp",
            "median_region_bp",
        ],
    )

    tool_metric_rows = [result["metric_row"] for result in tool_results]
    metrics_df = pd.DataFrame(tool_metric_rows)
    required_metric_columns = [
        "weighted_mean_h3k36me2_signal",
        "signal_ratio",
        "peak_overlap_fraction",
        "peak_ratio",
    ]
    finite_mask = np.isfinite(metrics_df[required_metric_columns]).all(axis=1)
    if not finite_mask.all():
        raise AssertionError(
            "Some tool/sample rows contain non-finite metric values:\n"
            + metrics_df.loc[
                ~finite_mask,
                ["sample", "tool_label"] + required_metric_columns,
            ].to_string(index=False)
        )

    metrics_df["signal_rank"] = metrics_df.groupby("sample")["signal_ratio"].rank(
        method="dense",
        ascending=True,
    )
    metrics_df["peak_rank"] = metrics_df.groupby("sample")["peak_ratio"].rank(
        method="dense",
        ascending=True,
    )
    metrics_df["combined_depletion_rank"] = metrics_df[["signal_rank", "peak_rank"]].mean(axis=1)
    metrics_df = metrics_df.sort_values(
        ["sample", "combined_depletion_rank", "deeptools_order", "tool"]
    ).reset_index(drop=True)
    assert_expected_tools(metrics_df, "metrics_df")

    tool_metrics_path = tables_dir / "chromatin_tool_metrics.tsv"
    metrics_df.to_csv(tool_metrics_path, sep="\t", index=False)

    for sample, sample_metrics in metrics_df.groupby("sample", sort=True):
        _print_dataframe(
            f"Tool metrics for {sample}",
            sample_metrics,
            [
                "tool_label",
                "platform",
                "region_type",
                "n_regions",
                "total_bp",
                "weighted_mean_h3k36me2_signal",
                "signal_ratio",
                "peak_overlap_fraction",
                "peak_ratio",
                "combined_depletion_rank",
            ],
        )

    combined_summary_df = (
        metrics_df.groupby(
            ["tool", "tool_label", "tool_family", "platform", "region_type", "deeptools_order"],
            as_index=False,
        )
        .agg(
            samples_scored=("sample", "nunique"),
            mean_combined_depletion_rank=("combined_depletion_rank", "mean"),
            mean_signal_ratio=("signal_ratio", "mean"),
            mean_peak_ratio=("peak_ratio", "mean"),
        )
        .sort_values(["mean_combined_depletion_rank", "deeptools_order", "tool"])
        .reset_index(drop=True)
    )

    sample_rank_pivot = (
        metrics_df.pivot(index="tool", columns="sample", values="combined_depletion_rank").reset_index()
    )
    combined_summary_df = combined_summary_df.merge(sample_rank_pivot, on="tool", how="left")

    combined_summary_path = tables_dir / "chromatin_combined_summary.tsv"
    combined_summary_df.to_csv(combined_summary_path, sep="\t", index=False)
    _print_dataframe(
        "Combined ranking",
        combined_summary_df,
        [
            "tool_label",
            "platform",
            "region_type",
            "samples_scored",
            "mean_combined_depletion_rank",
            "mean_signal_ratio",
            "mean_peak_ratio",
        ]
        + DEFAULT_SAMPLE_NAMES,
    )

    plot_rows = []
    for sample, sample_metrics in metrics_df.groupby("sample", sort=True):
        ordered = sample_metrics.sort_values(
            ["combined_depletion_rank", "deeptools_order", "tool"]
        ).reset_index(drop=True)
        plot_order = ordered["tool_label"].tolist()

        fig = plt.figure(figsize=(16, 10), constrained_layout=True)
        gs = fig.add_gridspec(2, 4)
        signal_ax = fig.add_subplot(gs[0, 0:2])
        peak_ax = fig.add_subplot(gs[0, 2:4])
        raw_signal_ax = fig.add_subplot(gs[1, 1:3])

        sns.barplot(
            data=ordered,
            x="signal_ratio",
            y="tool_label",
            hue="tool_label",
            order=plot_order,
            hue_order=plot_order,
            dodge=False,
            palette=sns.color_palette("crest", n_colors=len(ordered)),
            legend=False,
            ax=signal_ax,
        )
        signal_ax.axvline(1.0, linestyle="--", color="black", linewidth=1)
        signal_ax.set_title(f"{sample} signal ratio")
        signal_ax.set_xlabel("Weighted mean H3K36me2 / genomewide mean")
        signal_ax.set_ylabel("")

        sns.barplot(
            data=ordered,
            x="peak_ratio",
            y="tool_label",
            hue="tool_label",
            order=plot_order,
            hue_order=plot_order,
            dodge=False,
            palette=sns.color_palette("flare", n_colors=len(ordered)),
            legend=False,
            ax=peak_ax,
        )
        peak_ax.axvline(1.0, linestyle="--", color="black", linewidth=1)
        peak_ax.set_title(f"{sample} peak ratio")
        peak_ax.set_xlabel("Peak overlap fraction / genomic peak baseline")
        peak_ax.set_ylabel("")

        sns.barplot(
            data=ordered,
            x="weighted_mean_h3k36me2_signal",
            y="tool_label",
            hue="tool_label",
            order=plot_order,
            hue_order=plot_order,
            dodge=False,
            palette=sns.color_palette("mako", n_colors=len(ordered)),
            legend=False,
            ax=raw_signal_ax,
        )
        raw_signal_ax.axvline(
            ordered["genomewide_mean_h3k36me2_signal"].iloc[0],
            linestyle="--",
            color="black",
            linewidth=1,
        )
        raw_signal_ax.set_title(f"{sample} weighted mean signal")
        raw_signal_ax.set_xlabel("Weighted mean H3K36me2 input signal")
        raw_signal_ax.set_ylabel("")

        figure_path = figures_dir / f"{sample}.chromatin_enrichment_summary.png"
        fig.savefig(figure_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        plot_rows.append({"sample": sample, "summary_plot": str(figure_path)})

    combined_order = combined_summary_df.sort_values(
        ["mean_combined_depletion_rank", "deeptools_order", "tool"]
    ).reset_index(drop=True)
    combined_plot_order = combined_order["tool_label"].tolist()
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    sns.barplot(
        data=combined_order,
        x="mean_combined_depletion_rank",
        y="tool_label",
        hue="tool_label",
        order=combined_plot_order,
        hue_order=combined_plot_order,
        dodge=False,
        palette=sns.color_palette("mako", n_colors=len(combined_order)),
        legend=False,
        ax=ax,
    )
    ax.set_title("Combined chromatin depletion rank")
    ax.set_xlabel("Mean combined depletion rank across matched samples")
    ax.set_ylabel("")
    combined_rank_path = figures_dir / "combined_depletion_rank.png"
    fig.savefig(combined_rank_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    plot_df = pd.DataFrame(
        plot_rows + [{"sample": "combined", "summary_plot": str(combined_rank_path)}]
    )
    plot_outputs_path = tables_dir / "chromatin_plot_outputs.tsv"
    plot_df.to_csv(plot_outputs_path, sep="\t", index=False)
    _print_dataframe("Plot outputs", plot_df)

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
                "deepTools commands are not available on PATH: " + ", ".join(missing_deeptools)
            )

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
        deeptools_region_df = (
            pd.DataFrame(deeptools_region_rows)
            .sort_values(["sample", "deeptools_order", "tool"])
            .reset_index(drop=True)
        )
        deeptools_region_manifest_path = tables_dir / "deeptools_region_manifest.tsv"
        deeptools_region_df.to_csv(deeptools_region_manifest_path, sep="\t", index=False)
        _print_dataframe(
            "deepTools region manifest",
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

        deeptools_sample_tasks = []
        for sample in DEFAULT_SAMPLE_NAMES:
            background = sample_background[sample]
            sample_region_rows = deeptools_region_df.loc[
                deeptools_region_df["sample"].eq(sample)
            ].to_dict("records")
            deeptools_sample_tasks.append(
                {
                    "sample": sample,
                    "sample_id": sample_to_sample_id(sample),
                    "bw_path": background["bw_path"],
                    "region_rows": sample_region_rows,
                    "deeptools_tool_order": selected_deeptools_tools,
                    "deeptools_dir": str(deeptools_dir),
                    "deeptools_bin_size": int(DEEPTOOLS_BIN_SIZE),
                    "flank_length": int(FLANK_LENGTH),
                    "region_body_length": int(REGION_BODY_LENGTH),
                    "deeptools_force": bool(deeptools_force),
                    "include_heatmaps": bool(include_heatmaps),
                }
            )

        deeptools_rows = run_parallel(
            deeptools_sample_tasks,
            run_deeptools_for_sample_task,
            sample_workers,
            "deepTools profile generation"
            if not include_heatmaps
            else "deepTools profile and heatmap generation",
        )
        deeptools_outputs_df = (
            pd.DataFrame(deeptools_rows).sort_values("sample").reset_index(drop=True)
        )
        deeptools_outputs_path = tables_dir / "deeptools_outputs.tsv"
        deeptools_outputs_df.to_csv(deeptools_outputs_path, sep="\t", index=False)
        _print_dataframe("deepTools outputs", deeptools_outputs_df)
    else:
        print("Skipping deepTools execution because --skip-deeptools was provided.")

    print("\nChromatin analysis complete.")
    print(f"  Background metrics: {background_metrics_path}")
    print(f"  Region manifest:    {region_manifest_path}")
    print(f"  Tool metrics:       {tool_metrics_path}")
    print(f"  Combined summary:   {combined_summary_path}")
    print(f"  Plot outputs:       {plot_outputs_path}")
    if run_deeptools:
        print(f"  deepTools outputs:  {tables_dir / 'deeptools_outputs.tsv'}")


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
