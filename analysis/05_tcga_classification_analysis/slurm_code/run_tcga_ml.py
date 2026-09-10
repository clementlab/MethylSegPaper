import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TCGA_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT, TCGA_ANALYSIS_DIR, Path(__file__).resolve().parent):
    import_str = str(import_path)
    if import_str not in sys.path:
        sys.path.insert(0, import_str)

import pandas as pd

from repo_paths import (  # noqa: E402
    FIGURES_OUT_DIR,
    REFERENCE_DATA_DIR,
    TCGA_CLASSIFICATION_RESULTS_DIR,
)
from utils.tcga_ml_pipeline import (  # noqa: E402
    build_per_cancer_cohort_manifest,
    parse_feature_counts,
    write_json,
    write_table,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = TCGA_CLASSIFICATION_RESULTS_DIR
SEGMENTATION_ROOT = RESULTS_ROOT / "segmentation"
MANIFEST_DIR = RESULTS_ROOT / "manifests"
SAMPLE_MANIFEST = MANIFEST_DIR / "tcga_samples.tsv"
COHORT_MANIFEST = MANIFEST_DIR / "tcga_ml_cohorts.tsv"
ML_OUT_ROOT = RESULTS_ROOT / "ml_outputs" / "per_cancer"
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "tcga_ml_config.yaml"

# Easy-to-edit launcher defaults.
DEFAULT_FEATURE_COUNTS = (1, 3, 5, 10, 50, 100)
DEFAULT_N_SPLITS = 5
DEFAULT_N_REPEATS = 10
DEFAULT_N_FEATURE_DRAWS = 10
DEFAULT_MAX_SPLIT_WORKERS = 50
DEFAULT_RANDOM_SEED = 42
DEFAULT_MIN_CPGS_FOR_RANDOM_REGIONS = 1
DEFAULT_MIN_TRAINING_OBSERVED_CPGS_PER_REGION = 1
DEFAULT_ARRAY_TASK_COUNT = 100

SEGMENTATION_SLURM = SCRIPT_DIR / "methylation_tcga_segmentation.slurm"
ML_SLURM = SCRIPT_DIR / "methylation_tcga_ml.slurm"
POSTPROCESS_SLURM = SCRIPT_DIR / "methylation_tcga_postprocess.slurm"
CLEANUP_SLURM = SCRIPT_DIR / "methylation_tcga_cleanup.slurm"
PIPELINE_SCRIPT = SCRIPT_DIR / "run_tcga_ml_pipeline.py"
SEGMENTATION_SCRIPT = SCRIPT_DIR / "run_tcga_segmentation_array.py"
VALIDATION_SCRIPT = SCRIPT_DIR / "validate_tcga_ml.py"
COMPARISON_SCRIPT = SCRIPT_DIR / "compare_tcga_ml_metrics.py"
FIGURE_NOTEBOOK = PROJECT_ROOT / "figures" / "08_tcga_figures.ipynb"
TCGA_FIGURE_OUT_DIR = FIGURES_OUT_DIR / "06_tcga_figures"
SAMPLES_INFO_PATH = REFERENCE_DATA_DIR / "runAll.sh.samples"


def read_launcher_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except ModuleNotFoundError:
        data = json.loads(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")
    return data


def parse_string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        return tuple(piece.strip() for piece in value.split(",") if piece.strip())
    return tuple(str(piece) for piece in value if str(piece).strip())


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    value_str = str(value).strip().lower()
    if value_str in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_str in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Could not parse boolean value: {value}")


def backup_path_for(source_path: Path, timestamp: str) -> Path:
    return source_path.parent / f"{source_path.name}_backup_{timestamp}"


def archive_path_for(source_path: Path, timestamp: str) -> Path:
    return source_path.parent / f"{source_path.name}_archive_{timestamp}"


def stage_backup(
    source_path: Path,
    timestamp: str,
    staged_backups: list[Path],
    *,
    dry_run: bool,
) -> None:
    if not source_path.exists():
        return
    backup_path = backup_path_for(source_path, timestamp)
    if not dry_run:
        source_path.rename(backup_path)
    staged_backups.append(backup_path)


def count_existing_segmentations(segmentation_root: Path) -> int:
    summary_pattern = "*/out/hm450k/summary_files/segments_cleaned_PMD.bed"
    return sum(1 for _ in (segmentation_root / "methylseg").glob(summary_pattern))


def load_tcga_samples_info(samples_info_path: Path = SAMPLES_INFO_PATH) -> pd.DataFrame:
    return pd.read_csv(samples_info_path, sep="\t").copy()


def expected_summary_file(segmentation_root: Path, sample_id: str) -> Path:
    return (
        segmentation_root
        / "methylseg"
        / str(sample_id)
        / "out"
        / "hm450k"
        / "summary_files"
        / "segments_cleaned_PMD.bed"
    )


def build_sample_manifest(segmentation_root: Path) -> pd.DataFrame:
    samples_info = load_tcga_samples_info()
    if "sample" not in samples_info.columns:
        raise ValueError("Expected a sample column containing TCGA barcodes.")
    manifest_df = samples_info.loc[
        samples_info["sample"].astype(str).str.startswith("TCGA-")
        & samples_info["project_id"].astype(str).str.startswith("TCGA-")
        & samples_info["project_descriptor"].notna()
        & samples_info["methylation_file"].notna()
    ].copy()
    manifest_df["sample_id"] = manifest_df["sample"].astype(str)
    duplicate_mask = manifest_df["sample_id"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_ids = sorted(
            manifest_df.loc[duplicate_mask, "sample_id"].unique().tolist()
        )
        raise ValueError(
            "Duplicate TCGA sample barcodes are not allowed in the sample manifest: "
            f"{duplicate_ids[:10]}"
        )
    manifest_df = manifest_df.reset_index(drop=True)
    manifest_df = manifest_df.sort_values(
        ["project_id", "sample_type", "sample_id"]
    ).reset_index(drop=True)
    manifest_df["methylation_file"] = manifest_df["methylation_file"].astype(str)
    manifest_df["methylseg_hm450k_bed"] = manifest_df["sample_id"].map(
        lambda sample_id: str(expected_summary_file(segmentation_root, str(sample_id)))
    )
    return manifest_df[
        [
            "sample_id",
            "project_id",
            "project_descriptor",
            "sample_type",
            "methylation_file",
            "methylseg_hm450k_bed",
        ]
    ].copy()


def build_and_write_manifests(
    *,
    sample_manifest: Path,
    cohort_manifest: Path,
    segmentation_root: Path,
    n_splits: int,
    include_cohorts: tuple[str, ...],
    exclude_cohorts: tuple[str, ...],
) -> tuple[Path, Path, int]:
    sample_manifest_df = build_sample_manifest(segmentation_root)
    write_table(sample_manifest_df, sample_manifest)

    cohort_manifest_df = build_per_cancer_cohort_manifest(
        sample_manifest_df.rename(columns={"sample_id": "sample"}),
        min_splits=n_splits,
        include_cohorts=include_cohorts,
        exclude_cohorts=exclude_cohorts,
    )
    write_table(cohort_manifest_df, cohort_manifest)
    return sample_manifest, cohort_manifest, len(cohort_manifest_df)


def sbatch_command(*args: str | Path) -> list[str]:
    return ["sbatch", *[str(arg) for arg in args]]


def submit_or_echo(command: list[str], *, dry_run: bool) -> str:
    printable = " ".join(str(piece) for piece in command)
    print(printable)
    if dry_run:
        return "dry-run"
    completed = subprocess.run(
        [str(piece) for piece in command],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit the per-cancer TCGA feature-classification workflow."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--sample-manifest", type=Path, default=SAMPLE_MANIFEST)
    parser.add_argument("--cohort-manifest", type=Path, default=COHORT_MANIFEST)
    parser.add_argument("--segmentation-root", type=Path, default=SEGMENTATION_ROOT)
    parser.add_argument("--out-root", type=Path, default=ML_OUT_ROOT)
    parser.add_argument("--feature-counts", type=str, default=None)
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--n-repeats", type=int, default=None)
    parser.add_argument("--n-feature-draws", type=int, default=None)
    parser.add_argument("--max-split-workers", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--min-cpgs-for-random-regions", type=int, default=None)
    parser.add_argument(
        "--min-training-observed-cpgs-per-region",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--exclude-top-normal-shared-pmds",
        dest="exclude_top_normal_shared_pmds",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--keep-top-normal-shared-pmds",
        dest="exclude_top_normal_shared_pmds",
        action="store_false",
        default=None,
    )
    parser.add_argument("--array-task-count", type=int, default=None)
    parser.add_argument("--include-cohorts", type=str, default=None)
    parser.add_argument("--exclude-cohorts", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    config = read_launcher_config(args.config)
    feature_counts = parse_feature_counts(
        args.feature_counts
        if args.feature_counts is not None
        else config.get("feature_counts", DEFAULT_FEATURE_COUNTS)
    )
    settings = {
        "feature_counts": feature_counts,
        "n_splits": int(args.n_splits or config.get("n_splits", DEFAULT_N_SPLITS)),
        "n_repeats": int(args.n_repeats or config.get("n_repeats", DEFAULT_N_REPEATS)),
        "n_feature_draws": int(
            args.n_feature_draws
            or config.get("n_feature_draws", DEFAULT_N_FEATURE_DRAWS)
        ),
        "max_split_workers": int(
            args.max_split_workers
            or config.get("max_split_workers", DEFAULT_MAX_SPLIT_WORKERS)
        ),
        "random_seed": int(
            args.random_seed or config.get("random_seed", DEFAULT_RANDOM_SEED)
        ),
        "min_cpgs_for_random_regions": int(
            args.min_cpgs_for_random_regions
            if args.min_cpgs_for_random_regions is not None
            else config.get(
                "min_cpgs_for_random_regions",
                DEFAULT_MIN_CPGS_FOR_RANDOM_REGIONS,
            )
        ),
        "min_training_observed_cpgs_per_region": int(
            args.min_training_observed_cpgs_per_region
            if args.min_training_observed_cpgs_per_region is not None
            else config.get(
                "min_training_observed_cpgs_per_region",
                DEFAULT_MIN_TRAINING_OBSERVED_CPGS_PER_REGION,
            )
        ),
        "exclude_top_normal_shared_pmds": parse_bool(
            args.exclude_top_normal_shared_pmds
            if args.exclude_top_normal_shared_pmds is not None
            else config.get("exclude_top_normal_shared_pmds", False)
        ),
        "array_task_count": int(
            args.array_task_count
            or config.get("array_task_count", DEFAULT_ARRAY_TASK_COUNT)
        ),
        "include_cohorts": parse_string_list(
            args.include_cohorts
            if args.include_cohorts is not None
            else config.get("cohort_include", ())
        ),
        "exclude_cohorts": parse_string_list(
            args.exclude_cohorts
            if args.exclude_cohorts is not None
            else config.get("cohort_exclude", ())
        ),
    }
    return settings


def main() -> None:
    args = build_parser().parse_args()
    settings = resolve_settings(args)

    args.sample_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_root.parent.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    sample_manifest, cohort_manifest, n_cohorts = build_and_write_manifests(
        sample_manifest=args.sample_manifest,
        cohort_manifest=args.cohort_manifest,
        segmentation_root=args.segmentation_root,
        n_splits=settings["n_splits"],
        include_cohorts=settings["include_cohorts"],
        exclude_cohorts=settings["exclude_cohorts"],
    )

    with sample_manifest.open("r", encoding="utf-8") as handle:
        expected_segment_count = sum(1 for _ in handle) - 1
    segment_count = count_existing_segmentations(args.segmentation_root)
    timestamp = subprocess.run(
        ["date", "+%Y%m%d_%H%M%S"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    staged_backups: list[Path] = []
    retained_archives: list[Path] = []
    retained_figure_archives: list[Path] = []
    segmentation_job_id = ""
    cleanup_job_id = ""

    if args.force:
        stage_backup(
            args.segmentation_root / "methylseg",
            timestamp,
            staged_backups,
            dry_run=args.dry_run,
        )
        stage_backup(
            args.segmentation_root / "comparison",
            timestamp,
            staged_backups,
            dry_run=args.dry_run,
        )
        stage_backup(
            args.segmentation_root / "array_task_summaries",
            timestamp,
            staged_backups,
            dry_run=args.dry_run,
        )
        stage_backup(args.out_root, timestamp, staged_backups, dry_run=args.dry_run)
        segment_count = 0
    elif args.out_root.exists():
        archive_path = archive_path_for(args.out_root, timestamp)
        if not args.dry_run:
            args.out_root.rename(archive_path)
        retained_archives.append(archive_path)
        if TCGA_FIGURE_OUT_DIR.exists():
            figure_archive_path = archive_path_for(TCGA_FIGURE_OUT_DIR, timestamp)
            if not args.dry_run:
                TCGA_FIGURE_OUT_DIR.rename(figure_archive_path)
            retained_figure_archives.append(figure_archive_path)

    if segment_count < expected_segment_count:
        summary_dir = args.segmentation_root / "array_task_summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        segmentation_job_id = submit_or_echo(
            sbatch_command(
                "--parsable",
                f"--chdir={RESULTS_ROOT}",
                f"--job-name=tcga_pmd_{timestamp}",
                (
                    "--export=ALL,"
                    f"ARRAY_TASK_COUNT={settings['array_task_count']},"
                    f"RESULTS_ROOT={RESULTS_ROOT},"
                    f"SEGMENTATION_ROOT={args.segmentation_root},"
                    f"MANIFEST={sample_manifest},"
                    f"SUMMARY_DIR={summary_dir},"
                    f"FORCE_RECREATE_PMDS={1 if args.force else 0},"
                    f"PIPELINE_SCRIPT={SEGMENTATION_SCRIPT}"
                ),
                f"--array=1-{settings['array_task_count']}",
                SEGMENTATION_SLURM,
            ),
            dry_run=args.dry_run,
        )

    if staged_backups:
        cleanup_targets = "|".join(str(path) for path in staged_backups)
        cleanup_job_id = submit_or_echo(
            sbatch_command(
                "--parsable",
                f"--chdir={RESULTS_ROOT}",
                f"--job-name=tcga_cleanup_{timestamp}",
                (
                    "--export=ALL,"
                    f"RESULTS_ROOT={RESULTS_ROOT},"
                    f"CLEANUP_TARGETS={cleanup_targets}"
                ),
                CLEANUP_SLURM,
            ),
            dry_run=args.dry_run,
        )

    ml_job_ids: dict[int, str] = {}
    for feature_count in settings["feature_counts"]:
        ml_command = sbatch_command(
            "--parsable",
            f"--chdir={RESULTS_ROOT}",
            f"--job-name=tcga_ml_n{int(feature_count)}_{timestamp}",
            f"--cpus-per-task={settings['max_split_workers']}",
            (f"--output=logs/methylation_tcga_ml_n{int(feature_count)}.%A_%a.out"),
            (f"--error=logs/methylation_tcga_ml_n{int(feature_count)}.%A_%a.err"),
        )
        if segmentation_job_id and segmentation_job_id != "dry-run":
            ml_command.append(f"--dependency=afterok:{segmentation_job_id}")
        ml_command.extend(
            [
                (
                    "--export=ALL,"
                    f"RESULTS_ROOT={RESULTS_ROOT},"
                    f"COHORT_MANIFEST={cohort_manifest},"
                    f"OUT_ROOT={args.out_root},"
                    f"PIPELINE_SCRIPT={PIPELINE_SCRIPT},"
                    f"FEATURE_COUNT={int(feature_count)},"
                    f"N_SPLITS={settings['n_splits']},"
                    f"N_REPEATS={settings['n_repeats']},"
                    f"N_FEATURE_DRAWS={settings['n_feature_draws']},"
                    f"MAX_SPLIT_WORKERS={settings['max_split_workers']},"
                    f"RANDOM_SEED={settings['random_seed']},"
                    f"MIN_CPGS_FOR_RANDOM_REGIONS={settings['min_cpgs_for_random_regions']},"
                    "MIN_TRAINING_OBSERVED_CPGS_PER_REGION="
                    f"{settings['min_training_observed_cpgs_per_region']},"
                    f"EXCLUDE_TOP_NORMAL_SHARED_PMDS={1 if settings['exclude_top_normal_shared_pmds'] else 0}"
                ),
                f"--array=1-{n_cohorts}",
                ML_SLURM,
            ]
        )
        ml_job_ids[int(feature_count)] = submit_or_echo(
            ml_command, dry_run=args.dry_run
        )

    postprocess_job_id = ""
    if retained_archives:
        postprocess_command = sbatch_command(
            "--parsable",
            f"--chdir={RESULTS_ROOT}",
            f"--job-name=tcga_grouped_postprocess_{timestamp}",
        )
        completed_ml_job_ids = [
            job_id for job_id in ml_job_ids.values() if job_id != "dry-run"
        ]
        if completed_ml_job_ids:
            postprocess_command.append(
                f"--dependency=afterok:{':'.join(completed_ml_job_ids)}"
            )
        postprocess_command.extend(
            [
                (
                    "--export=ALL,"
                    f"PROJECT_ROOT={PROJECT_ROOT},"
                    f"RESULTS_ROOT={RESULTS_ROOT},"
                    f"COHORT_MANIFEST={cohort_manifest},"
                    f"GROUPED_OUT_ROOT={args.out_root},"
                    f"SAMPLE_LEVEL_ARCHIVE={retained_archives[0]},"
                    f"VALIDATION_SCRIPT={VALIDATION_SCRIPT},"
                    f"COMPARISON_SCRIPT={COMPARISON_SCRIPT},"
                    f"FIGURE_NOTEBOOK={FIGURE_NOTEBOOK}"
                ),
                POSTPROCESS_SLURM,
            ]
        )
        postprocess_job_id = submit_or_echo(
            postprocess_command,
            dry_run=args.dry_run,
        )

    launcher_record = {
        "sample_manifest": str(sample_manifest),
        "cohort_manifest": str(cohort_manifest),
        "n_cohorts": int(n_cohorts),
        "expected_segment_count": int(expected_segment_count),
        "existing_segment_count": int(segment_count),
        "settings": {
            **settings,
            "feature_counts": [int(value) for value in settings["feature_counts"]],
            "include_cohorts": list(settings["include_cohorts"]),
            "exclude_cohorts": list(settings["exclude_cohorts"]),
        },
        "segmentation_job_id": segmentation_job_id,
        "cleanup_job_id": cleanup_job_id,
        "retained_ml_archives": [str(path) for path in retained_archives],
        "retained_figure_archives": [
            str(path) for path in retained_figure_archives
        ],
        "ml_job_ids": {str(key): value for key, value in ml_job_ids.items()},
        "postprocess_job_id": postprocess_job_id,
        "dry_run": bool(args.dry_run),
    }
    write_json(
        launcher_record, RESULTS_ROOT / "ml_outputs" / "launcher_submission.json"
    )

    print(f"Sample manifest: {sample_manifest}")
    print(f"Cohort manifest: {cohort_manifest}")
    print(f"Eligible cohorts: {n_cohorts}")
    print(f"Feature counts: {list(settings['feature_counts'])}")
    if cleanup_job_id:
        print(f"Cleanup job: {cleanup_job_id}")
    for archive_path in retained_archives:
        print(f"Retained previous ML archive: {archive_path}")
    for archive_path in retained_figure_archives:
        print(f"Retained previous TCGA figure archive: {archive_path}")
    if segmentation_job_id:
        print(f"Segmentation job: {segmentation_job_id}")
    for feature_count, job_id in ml_job_ids.items():
        print(f"ML array job for n={feature_count}: {job_id}")
    if postprocess_job_id:
        print(f"Validation/comparison/figure job: {postprocess_job_id}")


if __name__ == "__main__":
    main()
