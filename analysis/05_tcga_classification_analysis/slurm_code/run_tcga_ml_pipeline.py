import argparse
import os
import sys
import tempfile
from pathlib import Path

matplotlib_cache_dir = Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"
matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(matplotlib_cache_dir)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TCGA_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT, TCGA_ANALYSIS_DIR):
    import_str = str(import_path)
    if import_str not in sys.path:
        sys.path.insert(0, import_str)

from utils.tcga_ml_pipeline import (  # noqa: E402
    DEFAULT_CGI_BED,
    DEFAULT_GENOME_FILE,
    DEFAULT_OUT_DIR,
    DEFAULT_SEGMENTATION_ROOT,
    PipelineConfig,
    parse_n_values,
    recalculate_metrics_from_dir,
    run_pipeline,
)


def parse_case_sets(value: str) -> tuple[str, ...]:
    allowed = {"brca", "per_cancer", "pan_cancer", "multiclass"}
    case_sets = tuple(piece.strip() for piece in value.split(",") if piece.strip())
    unknown = sorted(set(case_sets) - allowed)
    if unknown:
        raise ValueError(f"Unknown case sets: {unknown}. Allowed: {sorted(allowed)}")
    if not case_sets:
        raise ValueError("At least one case set is required.")
    return case_sets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or recalculate the TCGA PMD ML classification pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run CV training and prediction.")
    run_parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    run_parser.add_argument(
        "--segmentation-root", type=Path, default=DEFAULT_SEGMENTATION_ROOT
    )
    run_parser.add_argument(
        "--case-sets",
        type=str,
        default="brca,per_cancer,pan_cancer,multiclass",
        help="Comma-separated subset of brca,per_cancer,pan_cancer,multiclass.",
    )
    run_parser.add_argument(
        "--n-values",
        type=str,
        default="",
        help="Comma-separated region counts. Defaults to 1,3,5,10,50,100,500,1000.",
    )
    run_parser.add_argument("--cv-splits", type=int, default=5)
    run_parser.add_argument("--n-estimators", type=int, default=500)
    run_parser.add_argument("--random-state", type=int, default=42)
    run_parser.add_argument("--cgi-bed", type=Path, default=DEFAULT_CGI_BED)
    run_parser.add_argument("--genome-file", type=Path, default=DEFAULT_GENOME_FILE)
    run_parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Optional debug cap applied within each cohort label.",
    )
    run_parser.add_argument(
        "--no-save-feature-matrices",
        action="store_true",
        help="Skip writing per-fold feature matrices.",
    )
    run_parser.add_argument(
        "--allow-missing-feature-sets",
        action="store_true",
        help="Diagnostic mode: continue when required feature sets fail.",
    )
    run_parser.add_argument(
        "--no-call-missing-pmds",
        action="store_true",
        help="Do not run MethylSeg to create missing PMD files before preflight.",
    )
    run_parser.add_argument(
        "--high-confidence-pmd-min-fraction",
        type=float,
        default=0.50,
        help=(
            "Minimum training-sample PMD support fraction used to define "
            "high-confidence PMDs that Random regions must avoid."
        ),
    )
    run_parser.add_argument(
        "--random-region-min-length-bp",
        type=int,
        default=150000,
        help="Minimum Random-region length in base pairs.",
    )
    run_parser.add_argument(
        "--random-region-max-length-bp",
        type=int,
        default=20000000,
        help="Maximum Random-region length in base pairs.",
    )

    recalc_parser = subparsers.add_parser(
        "recalculate-metrics",
        help="Regenerate metrics and plots from saved prediction artifacts.",
    )
    recalc_parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "run":
            config = PipelineConfig(
                out_dir=args.out_dir,
                segmentation_root=args.segmentation_root,
                case_sets=parse_case_sets(args.case_sets),
                n_values=parse_n_values(args.n_values),
                cv_splits=args.cv_splits,
                n_estimators=args.n_estimators,
                random_state=args.random_state,
                cgi_bed=args.cgi_bed,
                genome_file=args.genome_file,
                max_samples_per_class=args.max_samples_per_class,
                save_feature_matrices=not args.no_save_feature_matrices,
                allow_missing_feature_sets=args.allow_missing_feature_sets,
                call_missing_pmds=not args.no_call_missing_pmds,
                high_confidence_pmd_min_fraction=args.high_confidence_pmd_min_fraction,
                random_region_min_length_bp=args.random_region_min_length_bp,
                random_region_max_length_bp=args.random_region_max_length_bp,
            )
            outputs = run_pipeline(config)
        else:
            outputs = recalculate_metrics_from_dir(args.out_dir)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Wrote TCGA ML outputs:")
    for name, path in sorted(outputs.items()):
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
