import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UTILS_DIR = PROJECT_ROOT / "analysis" / "01_region_calling_analysis" / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methyl_tool_comparator import MethylToolAggregateComparator
from repo_paths import REGION_CALLING_RESULTS_DIR


def _load_config_paths(configs_file: Path) -> list[Path]:
    with open(configs_file) as fh:
        return [Path(line.strip()) for line in fh if line.strip()]


def _sample_id_from_config(config_file: Path) -> str:
    with open(config_file) as fh:
        config = yaml.safe_load(fh) or {}

    sample_id = config.get("sample")
    if not sample_id:
        raise ValueError(f"Missing 'sample' in config: {config_file}")
    return str(sample_id)


def run(configs_file: str, out_dir: str):
    configs_path = Path(configs_file).resolve()
    out_path = Path(out_dir).resolve()

    config_paths = _load_config_paths(configs_path)
    if not config_paths:
        raise ValueError(f"No config files found in {configs_path}")

    comparison_out_dirs = []
    for config_path in config_paths:
        sample_id = _sample_id_from_config(config_path)
        comparison_out_dirs.append(out_path / "comparison" / sample_id)

    aggregator = MethylToolAggregateComparator(comparison_out_dirs)
    outputs = aggregator.generate_aggregate_summary_plots()

    print("Aggregate summary generation complete.")
    for name, path in sorted(outputs.items()):
        print(f"{name}: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate cross-sample aggregate methylation comparator summaries."
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs.txt"),
        help="Path to the newline-delimited config list used for the comparator array job.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REGION_CALLING_RESULTS_DIR),
        help="Base output directory containing comparison/<sample_id> results.",
    )

    args = parser.parse_args()
    run(args.configs, args.out_dir)
