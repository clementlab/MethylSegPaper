#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px

THIS_DIR = Path(__file__).resolve().parent
SYNTHETIC_ROOT = THIS_DIR.parent
for import_path in (THIS_DIR, SYNTHETIC_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import pipeline_config as cfg
import synthetic_analysis_helpers as sah


LOWER_IS_BETTER_PATTERNS = (
    "_mabe",
    "_distance_from_1",
    "avg_false_pmd_beta",
    "mean_false_pmds_called",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate synthetic recovery metrics from saved outputs."
    )
    parser.add_argument("--out-root", type=Path, default=cfg.DEFAULT_OUT_ROOT)
    parser.add_argument("--boundary-tolerance-bp", type=int, default=10_000)
    parser.add_argument("--n-sample-procs", type=int, default=1)
    parser.add_argument("--n-tool-procs", type=int, default=1)
    parser.add_argument("--from-shards", action="store_true")
    return parser.parse_args()


def metric_title(metric: str) -> str:
    title = metric.replace("_", " ").title()
    return title.replace("Bp", "BP").replace("Pmd", "PMD").replace("Mabe", "MABE")


def plot_file_stem(metric: str, suffix: str) -> str:
    safe_metric = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric)
    return f"{safe_metric}.{suffix}"


def lower_is_better(metric: str) -> bool:
    return any(pattern in metric for pattern in LOWER_IS_BETTER_PATTERNS)


def ordered_summary_for_metric(per_tool_summary_df: pd.DataFrame, metric_col: str):
    return sah.sort_tools_methylseg_first(
        per_tool_summary_df,
        metric_col=metric_col,
        ascending=lower_is_better(metric_col),
    )


def write_html_figure(fig, plot_dir: Path, filename: str, figures: dict, key: str):
    fig.update_layout(template="plotly_white")
    fig.write_html(plot_dir / filename)
    figures[key] = fig


def add_metric_bar_plot(
    per_tool_summary_df: pd.DataFrame,
    metric_col: str,
    plot_dir: Path,
    figures: dict,
    title: str | None = None,
    key: str | None = None,
):
    if metric_col not in per_tool_summary_df.columns:
        return
    plot_df = ordered_summary_for_metric(per_tool_summary_df, metric_col)
    fig = px.bar(
        plot_df,
        x="tool",
        y=metric_col,
        title=title or f"Mean {metric_title(metric_col.removesuffix('_mean'))} by tool",
        hover_data=["n_called_samples"]
        if "n_called_samples" in plot_df.columns
        else None,
    )
    write_html_figure(
        fig,
        plot_dir,
        plot_file_stem(metric_col, "bar.html"),
        figures,
        key or metric_col,
    )


def add_pareto_scatter(
    per_sample_df: pd.DataFrame,
    plot_dir: Path,
    figures: dict,
    recall_col: str = "bp_recall",
    precision_col: str = "bp_precision",
):
    if (
        recall_col not in per_sample_df.columns
        or precision_col not in per_sample_df.columns
    ):
        return
    plot_df = per_sample_df.dropna(subset=[recall_col, precision_col]).copy()
    if plot_df.empty:
        return
    plot_df = sah.sort_tools_methylseg_first(plot_df)
    fig = px.scatter(
        plot_df,
        x=recall_col,
        y=precision_col,
        color="tool",
        hover_data=["synthetic_sample_id", "n_truth_regions", "n_predicted_regions"],
        title="BP recall vs BP precision by sample and tool",
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    write_html_figure(
        fig, plot_dir, "bp_recall_vs_precision.pareto.html", figures, "pareto"
    )


def add_false_beta_violin(false_beta_df: pd.DataFrame, plot_dir: Path, figures: dict):
    if false_beta_df.empty:
        return
    tool_order = sah.sort_tools_methylseg_first(
        false_beta_df.groupby("tool", as_index=False)["mean_beta"].median(),
        metric_col="mean_beta",
        ascending=True,
    )["tool"].tolist()
    fig = px.violin(
        false_beta_df,
        x="tool",
        y="mean_beta",
        box=True,
        points="all",
        category_orders={"tool": tool_order},
        title="False PMD beta distribution by tool",
    )
    write_html_figure(
        fig, plot_dir, "false_pmd_beta.violin.html", figures, "false_pmd_beta"
    )


def write_metric_plots(
    metrics_result: dict,
    plot_dir: Path,
    display_inline: bool = False,
) -> dict:
    plot_dir.mkdir(parents=True, exist_ok=True)
    per_sample_df = metrics_result["per_sample_df"].copy()
    per_tool_summary_df = metrics_result["per_tool_summary_df"].copy()
    false_beta_df = metrics_result["false_beta_df"].copy()
    figures = {}

    for metric in sah.metric_column_names():
        metric_col = f"{metric}_mean"
        add_metric_bar_plot(per_tool_summary_df, metric_col, plot_dir, figures)

    add_metric_bar_plot(
        per_tool_summary_df,
        "fragmentation_distance_from_1_mean",
        plot_dir,
        figures,
        title="Mean fragmentation distance from 1 by tool",
        key="fragmentation_distance_from_1",
    )
    add_metric_bar_plot(
        per_tool_summary_df,
        "absorption_distance_from_1_mean",
        plot_dir,
        figures,
        title="Mean absorption distance from 1 by tool",
        key="absorption_distance_from_1",
    )
    add_metric_bar_plot(
        per_tool_summary_df,
        "region_any_recall_mean",
        plot_dir,
        figures,
        title="Mean region-any recall by tool",
        key="region_any_recall",
    )
    add_pareto_scatter(per_sample_df, plot_dir, figures)
    add_false_beta_violin(false_beta_df, plot_dir, figures)

    if display_inline:
        for fig in figures.values():
            fig.show()
    return figures


def main() -> int:
    args = parse_args()
    path_map = cfg.ensure_base_dirs(args.out_root)
    manifest_path = cfg.require_valid_injected_manifest(args.out_root)
    if args.from_shards:
        metrics_result = sah.finalize_metrics_from_shards(
            injected_manifest=manifest_path,
            metrics_output_dir=path_map["metrics"],
        )
    else:
        metrics_result = sah.run_metrics(
            injected_manifest=manifest_path,
            tool_results_dir=path_map["recovery_tool_results"],
            metrics_output_dir=path_map["metrics"],
            boundary_tolerance_bp=args.boundary_tolerance_bp,
            n_sample_procs=args.n_sample_procs,
            n_tool_procs=args.n_tool_procs,
        )
    write_metric_plots(metrics_result, path_map["plots"])

    sample_status_df = metrics_result["sample_status_df"]
    run_summary = pd.DataFrame(
        [
            {
                "injected_manifest": str(manifest_path),
                "tool_results_dir": str(path_map["recovery_tool_results"]),
                "metrics_dir": str(path_map["metrics"]),
                "plots_dir": str(path_map["plots"]),
                "boundary_tolerance_bp": int(args.boundary_tolerance_bp),
                "n_sample_procs": int(args.n_sample_procs),
                "n_tool_procs": int(args.n_tool_procs),
                "from_shards": bool(args.from_shards),
                "n_metric_rows": int(len(metrics_result["per_sample_df"])),
                "n_expected_samples": int(len(sample_status_df)),
                "n_called_samples": int(
                    sample_status_df["sample_status"].eq("called").sum()
                ),
            }
        ]
    )
    run_summary.to_csv(
        path_map["metrics"] / "aggregation_run_summary.tsv", sep="\t", index=False
    )
    print(f"Wrote metrics to: {path_map['metrics']}")
    print(f"Wrote plots to: {path_map['plots']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
