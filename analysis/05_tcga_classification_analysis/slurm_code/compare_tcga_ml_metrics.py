#!/usr/bin/env python3
"""Compare grouped TCGA metrics with the archived sample-level analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = ("balanced_accuracy", "average_precision", "macro_f1", "mcc", "roc_auc")


def collect_results(root: str | Path, analysis_label: str) -> pd.DataFrame:
    root = Path(root)
    frames = []
    for results_path in sorted(root.glob("TCGA-*/n_features_*/classification_results.tsv")):
        frame = pd.read_csv(results_path, sep="\t")
        if "cohort_id" not in frame.columns:
            frame["cohort_id"] = results_path.parents[1].name
        frame["analysis"] = analysis_label
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No classification results were found under {root}.")
    return pd.concat(frames, ignore_index=True)


def compare_metrics(grouped_root: str | Path, sample_level_root: str | Path) -> pd.DataFrame:
    grouped = collect_results(grouped_root, "patient_grouped")
    sample_level = collect_results(sample_level_root, "sample_level")
    combined = pd.concat([grouped, sample_level], ignore_index=True)
    missing_metrics = sorted(set(METRICS) - set(combined.columns))
    if missing_metrics:
        raise ValueError(f"Classification results are missing metrics: {missing_metrics}")

    summary = (
        combined.groupby(
            ["analysis", "cohort_id", "feature_set", "n_features"],
            as_index=False,
        )
        .agg(
            n_evaluations=("split", "size"),
            **{f"mean_{metric}": (metric, "mean") for metric in METRICS},
        )
    )
    grouped_summary = summary.loc[summary["analysis"].eq("patient_grouped")].drop(
        columns="analysis"
    )
    sample_summary = summary.loc[summary["analysis"].eq("sample_level")].drop(
        columns="analysis"
    )
    key_columns = ["cohort_id", "feature_set", "n_features"]
    comparison = grouped_summary.merge(
        sample_summary,
        on=key_columns,
        how="outer",
        suffixes=("_patient_grouped", "_sample_level"),
        validate="one_to_one",
    )
    for metric in METRICS:
        comparison[f"delta_{metric}"] = (
            comparison[f"mean_{metric}_patient_grouped"]
            - comparison[f"mean_{metric}_sample_level"]
        )
    return comparison.sort_values(key_columns).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped-root", type=Path, required=True)
    parser.add_argument("--sample-level-root", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    comparison = compare_metrics(args.grouped_root, args.sample_level_root)
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_tsv, sep="\t", index=False)

    delta_columns = [f"delta_{metric}" for metric in METRICS]
    report = {
        "grouped_root": str(args.grouped_root),
        "sample_level_root": str(args.sample_level_root),
        "comparison_table": str(args.output_tsv),
        "n_comparisons": int(len(comparison)),
        "mean_metric_deltas": {
            metric: float(comparison[f"delta_{metric}"].mean())
            for metric in METRICS
        },
        "n_missing_grouped_summaries": int(
            comparison["n_evaluations_patient_grouped"].isna().sum()
        ),
        "n_missing_sample_level_summaries": int(
            comparison["n_evaluations_sample_level"].isna().sum()
        ),
        "n_evaluation_count_changes": int(
            comparison["n_evaluations_patient_grouped"]
            .ne(comparison["n_evaluations_sample_level"])
            .sum()
        ),
        "delta_columns": delta_columns,
    }
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
