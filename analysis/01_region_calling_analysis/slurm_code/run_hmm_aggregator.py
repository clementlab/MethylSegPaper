import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repo_paths import REGION_CALLING_RESULTS_DIR


DEFAULT_HMM_RESULTS_DIR = REGION_CALLING_RESULTS_DIR / "hmm_tests"
REQUIRED_COLUMNS = [
    "sample_id",
    "hmm_type",
    "runtime_sec",
    "config_path",
    "output_dir",
]


def _load_config_paths(configs_file: Path) -> list[Path]:
    with open(configs_file) as handle:
        return [Path(line.strip()) for line in handle if line.strip()]


def _sample_id_from_config(config_file: Path) -> str:
    with open(config_file) as handle:
        config = yaml.safe_load(handle) or {}

    sample_id = config.get("sample")
    if not sample_id:
        raise ValueError(f"Missing 'sample' in config: {config_file}")
    return str(sample_id)


def run(configs_file: str, out_dir: str) -> Path:
    configs_path = Path(configs_file).resolve()
    out_path = Path(out_dir).resolve()

    config_paths = _load_config_paths(configs_path)
    if not config_paths:
        raise ValueError(f"No config files found in {configs_path}")

    frames = []
    missing = []
    for config_path in config_paths:
        sample_id = _sample_id_from_config(config_path)
        sample_stats_path = out_path / sample_id / "run_stats" / "hmm_run_times.tsv"
        if not sample_stats_path.exists():
            missing.append(str(sample_stats_path))
            continue

        sample_df = pd.read_csv(sample_stats_path, sep="\t")
        sample_df["sample_id"] = sample_df["sample_id"].astype(str)
        frames.append(sample_df)

    if missing:
        raise FileNotFoundError(
            "Missing HMM timing files:\n" + "\n".join(sorted(missing))
        )
    if not frames:
        raise FileNotFoundError(
            f"No HMM timing files were collected from {out_path}"
        )

    combined_df = pd.concat(frames, ignore_index=True)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in combined_df.columns]
    if missing_columns:
        raise ValueError(
            f"Combined HMM timing table is missing columns {missing_columns}"
        )

    combined_df["runtime_sec"] = pd.to_numeric(
        combined_df["runtime_sec"], errors="coerce"
    )
    combined_df = combined_df.sort_values(["sample_id", "hmm_type"]).reset_index(
        drop=True
    )

    aggregate_dir = out_path / "aggregate_summaries"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = aggregate_dir / "all_hmm_run_times.csv"
    combined_df.to_csv(aggregate_path, index=False)
    print(f"Wrote aggregate HMM run times to {aggregate_path}")
    return aggregate_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate CT and Sticky HMM timing summaries across samples."
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs.txt"),
        help="Path to the newline-delimited config list used for the HMM array job.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_HMM_RESULTS_DIR),
        help="Directory containing hmm_tests/<sample>/run_stats outputs.",
    )

    args = parser.parse_args()
    run(args.configs, args.out_dir)
