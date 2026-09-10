#!/usr/bin/env python3
"""Validate grouped TCGA CV splits and completed classification outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_DIR = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT, ANALYSIS_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from repo_paths import REFERENCE_DATA_DIR, TCGA_CLASSIFICATION_RESULTS_DIR  # noqa: E402
from utils.tcga_ml_pipeline import (  # noqa: E402
    TCGA_PATIENT_ID_RULE,
    build_per_cancer_cohort_manifest,
    build_repeated_stratified_group_splits,
    extract_tcga_patient_id,
    make_patient_groups,
    make_sample_types,
)


EXPECTED_FEATURE_COUNTS = (1, 3, 5, 10, 50, 100)
EXPECTED_STOCHASTIC_FEATURE_SETS = (
    "Random PMD",
    "CGI",
    "Random short",
    "Random long",
)
DEFAULT_SAMPLE_MANIFEST = REFERENCE_DATA_DIR / "runAll.sh.samples"
DEFAULT_COHORT_MANIFEST = (
    TCGA_CLASSIFICATION_RESULTS_DIR / "manifests" / "tcga_ml_cohorts.tsv"
)
DEFAULT_OUT_ROOT = TCGA_CLASSIFICATION_RESULTS_DIR / "ml_outputs" / "per_cancer"
DEFAULT_REPORT = (
    TCGA_CLASSIFICATION_RESULTS_DIR
    / "ml_outputs"
    / "grouped_classification_validation.json"
)


def validate_real_manifest_splits(
    sample_manifest_path: str | Path,
    *,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_seed: int = 42,
) -> dict:
    samples = pd.read_csv(sample_manifest_path, sep="\t").copy()
    if "sample" not in samples.columns and "sample_id" in samples.columns:
        samples = samples.rename(columns={"sample_id": "sample"})
    samples = samples.loc[
        samples["sample"].astype(str).str.startswith("TCGA-")
        & samples["project_id"].astype(str).str.startswith("TCGA-")
    ].copy()
    for required_value_column in ("project_descriptor", "methylation_file"):
        if required_value_column in samples.columns:
            samples = samples.loc[samples[required_value_column].notna()].copy()
    cohort_manifest = build_per_cancer_cohort_manifest(
        samples,
        min_splits=n_splits,
    )

    cohort_records = []
    for cohort_id in cohort_manifest["cohort_id"].astype(str):
        cohort_samples = (
            samples.loc[samples["project_id"].astype(str).eq(cohort_id)]
            .sort_values("sample")
            .reset_index(drop=True)
        )
        sample_types = make_sample_types(cohort_samples)
        sample_groups = make_patient_groups(cohort_samples)
        sample_ids = np.asarray(list(sample_types), dtype=object)
        y = np.asarray(
            [1 if sample_types[sample_id] == "Tumor" else 0 for sample_id in sample_ids],
            dtype=int,
        )
        jobs = build_repeated_stratified_group_splits(
            sample_ids,
            y,
            sample_groups,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_seed=random_seed,
        )
        expected_jobs = int(n_splits) * int(n_repeats)
        if len(jobs) != expected_jobs:
            raise ValueError(
                f"{cohort_id} produced {len(jobs)} splits; expected {expected_jobs}."
            )
        cohort_records.append(
            {
                "cohort_id": cohort_id,
                "n_samples": int(len(sample_ids)),
                "n_patients": int(len(set(sample_groups.values()))),
                "n_splits": int(len(jobs)),
                "repeat_seeds": [int(random_seed) + repeat for repeat in range(n_repeats)],
            }
        )

    return {
        "status": "passed",
        "cv_strategy": "repeated_stratified_group_k_fold",
        "patient_id_rule": TCGA_PATIENT_ID_RULE,
        "n_cohorts": int(len(cohort_records)),
        "n_valid_splits": int(sum(record["n_splits"] for record in cohort_records)),
        "cohorts": cohort_records,
    }


def _expected_task_paths(out_root: Path, cohort_ids: list[str]) -> set[Path]:
    return {
        out_root / cohort_id / f"n_features_{feature_count}"
        for cohort_id in cohort_ids
        for feature_count in EXPECTED_FEATURE_COUNTS
    }


def validate_completed_outputs(
    out_root: str | Path,
    cohort_manifest_path: str | Path,
    *,
    expected_sklearn_version: str = "1.8.0",
) -> dict:
    out_root = Path(out_root)
    cohort_manifest = pd.read_csv(cohort_manifest_path, sep="\t")
    cohort_ids = sorted(cohort_manifest["cohort_id"].astype(str).tolist())
    if len(cohort_ids) != 14:
        raise ValueError(f"Expected 14 eligible cohorts; found {len(cohort_ids)}.")

    expected_tasks = _expected_task_paths(out_root, cohort_ids)
    actual_tasks = {
        path
        for path in out_root.glob("TCGA-*/n_features_*")
        if path.is_dir()
    }
    missing_tasks = sorted(str(path) for path in expected_tasks - actual_tasks)
    extra_tasks = sorted(str(path) for path in actual_tasks - expected_tasks)
    if missing_tasks or extra_tasks:
        raise ValueError(
            "Classification task layout is incomplete or mixed. "
            f"Missing={missing_tasks[:10]}; extra={extra_tasks[:10]}."
        )

    total_result_rows = 0
    cohort_fold_assignments: dict[str, pd.DataFrame] = {}
    task_records = []
    for task_dir in sorted(expected_tasks):
        required_paths = {
            "results": task_dir / "classification_results.tsv",
            "regions": task_dir / "sampled_feature_regions.tsv",
            "predictions": task_dir / "fold_predictions.tsv",
            "config": task_dir / "run_config.json",
        }
        missing_files = [str(path) for path in required_paths.values() if not path.exists()]
        if missing_files:
            raise ValueError(f"Incomplete task {task_dir}: missing {missing_files}.")

        config = json.loads(required_paths["config"].read_text(encoding="utf-8"))
        if config.get("cv_strategy") != "repeated_stratified_group_k_fold":
            raise ValueError(f"Non-grouped run_config found in {task_dir}.")
        if config.get("patient_id_rule") != TCGA_PATIENT_ID_RULE:
            raise ValueError(f"Unexpected participant rule in {task_dir}.")
        actual_sklearn = config.get("software_versions", {}).get("scikit-learn")
        if actual_sklearn != expected_sklearn_version:
            raise ValueError(
                f"{task_dir} used scikit-learn {actual_sklearn}; "
                f"expected {expected_sklearn_version}."
            )

        results = pd.read_csv(required_paths["results"], sep="\t")
        result_counts = results.groupby("feature_set", sort=False).size().to_dict()
        expected_counts = {"PMD": 50, "Always cancer": 50}
        expected_counts.update(
            {feature_set: 500 for feature_set in EXPECTED_STOCHASTIC_FEATURE_SETS}
        )
        if result_counts != expected_counts:
            raise ValueError(
                f"Unexpected evaluation counts in {task_dir}: "
                f"observed={result_counts}, expected={expected_counts}."
            )
        if len(results) != 2_100:
            raise ValueError(f"Expected 2,100 result rows in {task_dir}; found {len(results)}.")
        total_result_rows += len(results)

        predictions = pd.read_csv(
            required_paths["predictions"],
            sep="\t",
            usecols=["repeat", "fold", "sample_id", "patient_id"],
        ).drop_duplicates()
        if predictions[["sample_id", "patient_id"]].isna().any().any():
            raise ValueError(f"Missing sample or patient identifiers in {task_dir}.")
        expected_patient_ids = predictions["sample_id"].map(extract_tcga_patient_id)
        if not predictions["patient_id"].astype(str).eq(expected_patient_ids).all():
            raise ValueError(f"Incorrect patient IDs in {task_dir}.")
        patient_fold_counts = predictions.groupby(["repeat", "patient_id"])["fold"].nunique()
        if patient_fold_counts.ne(1).any():
            raise ValueError(f"Patient leakage across test folds in {task_dir}.")
        sample_fold_counts = predictions.groupby(["repeat", "sample_id"])["fold"].nunique()
        if sample_fold_counts.ne(1).any():
            raise ValueError(f"Sample leakage across test folds in {task_dir}.")

        assignment = predictions.sort_values(
            ["repeat", "sample_id"]
        ).reset_index(drop=True)
        cohort_id = str(config["cohort_id"])
        prior_assignment = cohort_fold_assignments.get(cohort_id)
        if prior_assignment is None:
            cohort_fold_assignments[cohort_id] = assignment
        elif not prior_assignment.equals(assignment):
            raise ValueError(
                f"Fold assignments differ across feature-count tasks for {cohort_id}."
            )

        task_records.append(
            {
                "task_dir": str(task_dir),
                "cohort_id": cohort_id,
                "feature_count": int(config["feature_count"]),
                "n_result_rows": int(len(results)),
                "n_unique_test_assignments": int(len(predictions)),
            }
        )

    if len(task_records) != 84:
        raise ValueError(f"Expected 84 completed tasks; found {len(task_records)}.")
    if total_result_rows != 176_400:
        raise ValueError(
            f"Expected 176,400 total result rows; found {total_result_rows}."
        )

    return {
        "status": "passed",
        "n_cohorts": len(cohort_ids),
        "feature_counts": list(EXPECTED_FEATURE_COUNTS),
        "n_completed_tasks": len(task_records),
        "n_classification_result_rows": int(total_result_rows),
        "expected_sklearn_version": expected_sklearn_version,
        "tasks": task_records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", type=Path, default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--cohort-manifest", type=Path, default=DEFAULT_COHORT_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--splits-only", action="store_true")
    parser.add_argument("--expected-sklearn-version", default="1.8.0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = {
        "split_validation": validate_real_manifest_splits(args.sample_manifest),
    }
    if not args.splits_only:
        report["output_validation"] = validate_completed_outputs(
            args.out_root,
            args.cohort_manifest,
            expected_sklearn_version=args.expected_sklearn_version,
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
