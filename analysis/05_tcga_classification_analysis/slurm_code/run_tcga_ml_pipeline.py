import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

matplotlib_cache_dir = Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"
matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(matplotlib_cache_dir)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TCGA_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT, TCGA_ANALYSIS_DIR):
    import_str = str(import_path)
    if import_str not in sys.path:
        sys.path.insert(0, import_str)

from repo_paths import TCGA_CLASSIFICATION_RESULTS_DIR  # noqa: E402
from utils.tcga_ml_pipeline import run_cohort_feature_classification  # noqa: E402

DEFAULT_OUT_ROOT = TCGA_CLASSIFICATION_RESULTS_DIR / "ml_outputs" / "per_cancer"
DEFAULT_COHORT_MANIFEST = (
    TCGA_CLASSIFICATION_RESULTS_DIR / "manifests" / "tcga_ml_cohorts.tsv"
)


def load_cohort_manifest(manifest_path: str | Path) -> Path:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Cohort manifest does not exist: {manifest_path}")
    return manifest_path


def resolve_cohort_id(
    *,
    cohort_id: str | None,
    cohort_manifest: str | Path | None,
    array_index: int | None,
) -> str:
    if cohort_id:
        return str(cohort_id)
    if cohort_manifest is None or array_index is None:
        raise ValueError(
            "Provide either --cohort-id directly or both --cohort-manifest and --array-index."
        )

    manifest_path = load_cohort_manifest(cohort_manifest)
    cohort_df = pd.read_csv(manifest_path, sep="\t").copy()
    if "cohort_id" not in cohort_df.columns:
        raise ValueError(f"Cohort manifest is missing a cohort_id column: {manifest_path}")
    if array_index < 1 or array_index > len(cohort_df):
        raise ValueError(
            f"--array-index must be between 1 and {len(cohort_df)} inclusive; got {array_index}."
        )
    return str(cohort_df.iloc[array_index - 1]["cohort_id"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one per-cancer TCGA feature-classification task."
    )
    parser.add_argument(
        "--cohort-id",
        type=str,
        default=None,
        help="TCGA cohort ID such as TCGA-BRCA.",
    )
    parser.add_argument(
        "--cohort-manifest",
        type=Path,
        default=None,
        help="Optional cohort manifest used with --array-index.",
    )
    parser.add_argument(
        "--array-index",
        type=int,
        default=None,
        help="1-based array index into the cohort manifest.",
    )
    parser.add_argument(
        "--feature-count",
        type=int,
        required=True,
        help="Number of regions to evaluate for this task.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--n-feature-draws", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-split-workers", type=int, default=50)
    parser.add_argument("--min-cpgs-for-random-regions", type=int, default=1)
    parser.add_argument(
        "--min-training-observed-cpgs-per-region",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--exclude-top-normal-shared-pmds",
        action="store_true",
        help=(
            "Remove tumor training PMDs that overlap the top n recurrent normal "
            "PMDs in each fold before selecting recurrent tumor PMDs."
        ),
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        cohort_id = resolve_cohort_id(
            cohort_id=args.cohort_id,
            cohort_manifest=args.cohort_manifest,
            array_index=args.array_index,
        )
        outputs = run_cohort_feature_classification(
            cohort_id=cohort_id,
            feature_count=args.feature_count,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            n_feature_draws=args.n_feature_draws,
            random_seed=args.random_seed,
            max_split_workers=args.max_split_workers,
            out_root=args.out_root,
            exclude_top_normal_shared_pmds=args.exclude_top_normal_shared_pmds,
            min_cpgs_for_random_regions=args.min_cpgs_for_random_regions,
            min_training_observed_cpgs_per_region=(
                args.min_training_observed_cpgs_per_region
            ),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise

    print("Wrote TCGA ML task outputs:")
    for name, path in sorted(outputs.items()):
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
