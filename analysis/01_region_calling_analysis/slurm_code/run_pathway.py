import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UTILS_DIR = PROJECT_ROOT / "analysis" / "01_region_calling_analysis" / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methyl_tool_comparator import MethylToolComparator
from repo_paths import REGION_CALLING_RESULTS_DIR


def run(config_file, out_dir, force_recreate, n_jobs):
    comparator = MethylToolComparator(
        config_file,
        out_dir,
        force_recreate=force_recreate,
        n_jobs=n_jobs,
    )
    comparator.run()
    print("Script Run")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run script with a specified model")

    parser.add_argument(
        "--config",
        type=str,
        help="Config file for the pathway",
        default="config.yaml",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REGION_CALLING_RESULTS_DIR),
        help="Base output directory to write tool results into.",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force re-running tools even if outputs exist.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=10,
        help="Number of tool processes to run in parallel within a sample.",
    )

    args = parser.parse_args()

    # Call the function with the parsed arguments
    run(args.config, args.out_dir, args.force_recreate, args.n_jobs)
