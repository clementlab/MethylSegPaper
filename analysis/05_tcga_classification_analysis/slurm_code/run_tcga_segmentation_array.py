import argparse
import math
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TCGA_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT, TCGA_ANALYSIS_DIR):
    import_str = str(import_path)
    if import_str not in sys.path:
        sys.path.insert(0, import_str)

from utils.tcga_segmentation_workflow import (
    DEFAULT_GENOME,
    load_tcga_samples_info,
    run_segmentation_for_sample,
)
from repo_paths import TCGA_CLASSIFICATION_RESULTS_DIR

DEFAULT_SCRATCH_ROOT = TCGA_CLASSIFICATION_RESULTS_DIR
DEFAULT_SEGMENTATION_ROOT = DEFAULT_SCRATCH_ROOT / "segmentation"
DEFAULT_MANIFEST_PATH = DEFAULT_SCRATCH_ROOT / "manifests" / "tcga_samples.tsv"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def expected_summary_file(segmentation_root: str | Path, sample_id: str) -> Path:
    return (
        Path(segmentation_root)
        / "methylseg"
        / str(sample_id)
        / "out"
        / "hm450k"
        / "summary_files"
        / "segments_cleaned_PMD.bed"
    )


def build_tcga_manifest(segmentation_root: str | Path) -> pd.DataFrame:
    samples_info = load_tcga_samples_info()
    manifest_df = samples_info.loc[
        samples_info["sample_id"].astype(str).str.startswith("TCGA-")
        & samples_info["project_id"].astype(str).str.startswith("TCGA-")
        & samples_info["project_descriptor"].notna()
        & samples_info["methylation_file"].notna()
    ].copy()
    manifest_df = manifest_df.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)
    manifest_df = manifest_df.sort_values(["project_id", "sample_type", "sample_id"]).reset_index(drop=True)
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


def write_manifest(manifest_df: pd.DataFrame, manifest_path: str | Path) -> Path:
    manifest_path = Path(manifest_path)
    ensure_dir(manifest_path.parent)
    manifest_df.to_csv(manifest_path, sep="\t", index=False)
    return manifest_path


def load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    manifest_df = pd.read_csv(manifest_path, sep="\t").copy()
    required_cols = {
        "sample_id",
        "project_id",
        "sample_type",
        "methylation_file",
        "methylseg_hm450k_bed",
    }
    missing_cols = sorted(required_cols - set(manifest_df.columns))
    if missing_cols:
        raise ValueError(f"Manifest is missing required columns: {missing_cols}")
    return manifest_df


def get_manifest_chunk(
    manifest_df: pd.DataFrame,
    *,
    array_index: int,
    array_count: int,
) -> pd.DataFrame:
    if array_count < 1:
        raise ValueError("array_count must be >= 1")
    if array_index < 1 or array_index > array_count:
        raise ValueError(
            f"array_index must be between 1 and array_count inclusive; got {array_index} of {array_count}."
        )

    total = len(manifest_df)
    start = math.floor((array_index - 1) * total / array_count)
    end = math.floor(array_index * total / array_count)
    return manifest_df.iloc[start:end].copy().reset_index(drop=True)


def run_array_chunk(
    manifest_df: pd.DataFrame,
    *,
    segmentation_root: str | Path,
    array_index: int,
    array_count: int,
    genome: str,
    skip_existing: bool,
    force_recreate: bool,
) -> pd.DataFrame:
    chunk_df = get_manifest_chunk(
        manifest_df,
        array_index=array_index,
        array_count=array_count,
    )
    records = []
    for row in tqdm(
        chunk_df.itertuples(index=False),
        total=len(chunk_df),
        desc=f"Segmenting task {array_index}/{array_count}",
    ):
        summary_file = expected_summary_file(segmentation_root, str(row.sample_id))
        status = "completed"
        error_message = ""

        try:
            if skip_existing and summary_file.exists():
                status = "skipped_existing"
            else:
                run_segmentation_for_sample(
                    sample_id=str(row.sample_id),
                    meth_file=row.methylation_file,
                    segmentation_root=segmentation_root,
                    genome=genome,
                    force_recreate=force_recreate,
                    print_logs=True,
                    run_on_dnmtools_array=False,
                    clean_individual_chr_outputs=True,
                )
                if not summary_file.exists():
                    status = "missing_output"
        except Exception as exc:  # pragma: no cover - runtime safety for batch jobs
            status = "failed"
            error_message = str(exc)

        records.append(
            {
                "array_index": int(array_index),
                "array_count": int(array_count),
                "sample_id": str(row.sample_id),
                "project_id": str(row.project_id),
                "sample_type": str(row.sample_type),
                "methylation_file": str(row.methylation_file),
                "summary_file": str(summary_file),
                "status": status,
                "summary_exists": bool(summary_file.exists()),
                "error": error_message,
            }
        )

    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment TCGA HM450K samples in fixed-size Slurm array chunks."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Manifest path shared across array and analysis jobs.",
    )
    parser.add_argument(
        "--segmentation-root",
        type=Path,
        default=DEFAULT_SEGMENTATION_ROOT,
        help="Segmentation root that will contain methylseg/<sample_id>/... outputs.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write the TCGA manifest and exit without running segmentation.",
    )
    parser.add_argument(
        "--array-index",
        type=int,
        help="1-based Slurm array index for this task.",
    )
    parser.add_argument(
        "--array-count",
        type=int,
        help="Total number of array tasks in the batch.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=DEFAULT_SEGMENTATION_ROOT / "array_task_summaries",
        help="Directory for per-array task TSV summaries.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose segments_cleaned_PMD.bed already exists.",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Recreate PMD outputs even if they already exist.",
    )
    parser.add_argument(
        "--genome",
        type=str,
        default=DEFAULT_GENOME,
        help="Genome name passed to the segmentation workflow.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.write_manifest:
        manifest_df = build_tcga_manifest(args.segmentation_root)
        manifest_path = write_manifest(manifest_df, args.manifest)
        print(f"Wrote manifest with {len(manifest_df)} TCGA samples: {manifest_path}")
        return

    if args.array_index is None or args.array_count is None:
        raise ValueError("--array-index and --array-count are required unless --write-manifest is used.")

    manifest_df = load_manifest(args.manifest)
    result_df = run_array_chunk(
        manifest_df,
        segmentation_root=args.segmentation_root,
        array_index=args.array_index,
        array_count=args.array_count,
        genome=args.genome,
        skip_existing=args.skip_existing and not args.force_recreate,
        force_recreate=args.force_recreate,
    )

    ensure_dir(args.summary_dir)
    summary_path = args.summary_dir / f"task_{args.array_index:04d}.tsv"
    result_df.to_csv(summary_path, sep="\t", index=False)
    print(f"Wrote task summary: {summary_path}")
    if not result_df.empty:
        print(result_df["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
