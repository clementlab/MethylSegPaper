import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UTILS_DIR = PROJECT_ROOT / "analysis" / "01_region_calling_analysis" / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from methylseg import HMMType, MethylDataPrep, MethylSegPathway
from methyl_tool_comparator import SharedPrepManager
from repo_paths import REGION_CALLING_RESULTS_DIR


DEFAULT_HMM_RESULTS_DIR = REGION_CALLING_RESULTS_DIR / "hmm_tests"
REQUIRED_CONFIG_KEYS = ("sample", "meth_file", "genome")


def _load_config(config_file: Path) -> dict:
    with open(config_file) as handle:
        config = yaml.safe_load(handle) or {}

    missing = [key for key in REQUIRED_CONFIG_KEYS if not config.get(key)]
    if missing:
        raise ValueError(f"Missing required config keys {missing} in {config_file}")
    return config


def _prepare_sample_info(config: dict, out_dir: Path, force_recreate: bool):
    shared_prep_manager = SharedPrepManager(
        config["sample"],
        config["meth_file"],
        config["genome"],
        out_dir=out_dir,
        force_recreate=force_recreate,
        skip_450k=True,
    )
    shared_outputs = shared_prep_manager.prepare()
    sample_info, _ = MethylDataPrep(
        meth_file=shared_outputs.wgbs_tsv,
        sample_id=config["sample"],
        resolution="wgbs",
        remove_low_coverage_like_cpgs=True,
    ).prepare()
    return sample_info


def _run_hmm(
    *,
    sample_info,
    hmm_type: HMMType,
    output_dir: Path,
    force_recreate: bool,
) -> tuple[list[str], float]:
    pathway = MethylSegPathway(
        train_sample_info=sample_info,
        hmm_type=hmm_type,
        out_dir=output_dir,
    )
    start = time.perf_counter()
    summary_files = pathway.run_pathway(
        force_resegment=force_recreate,
        force_optimize_rules=force_recreate,
    )
    runtime_sec = time.perf_counter() - start
    return summary_files, runtime_sec


def run(config_file: str, out_dir: str, force_recreate: bool = False) -> Path:
    config_path = Path(config_file).resolve()
    out_path = Path(out_dir).resolve()
    config = _load_config(config_path)

    sample_id = str(config["sample"])
    sample_dir = out_path / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_info = _prepare_sample_info(config, out_path, force_recreate)

    run_specs = (
        ("ct", HMMType.CT, sample_dir / "ct"),
        ("sticky", HMMType.STICKY, sample_dir / "sticky"),
    )
    records = []
    for hmm_name, hmm_type, hmm_out_dir in run_specs:
        print(
            f"Running {hmm_name} HMM for sample {sample_id} "
            f"using {config_path}"
        )
        summary_files, runtime_sec = _run_hmm(
            sample_info=sample_info,
            hmm_type=hmm_type,
            output_dir=hmm_out_dir,
            force_recreate=force_recreate,
        )
        records.append(
            {
                "sample_id": sample_id,
                "hmm_type": hmm_name,
                "runtime_sec": runtime_sec,
                "config_path": str(config_path),
                "output_dir": str(hmm_out_dir.resolve()),
            }
        )
        print(
            f"Completed {hmm_name} HMM for {sample_id} in "
            f"{runtime_sec:.2f} seconds ({len(summary_files)} summary files)"
        )

    run_stats_dir = sample_dir / "run_stats"
    run_stats_dir.mkdir(parents=True, exist_ok=True)
    run_stats_path = run_stats_dir / "hmm_run_times.tsv"
    pd.DataFrame.from_records(records).to_csv(run_stats_path, sep="\t", index=False)
    print(f"Wrote HMM run times to {run_stats_path}")
    return run_stats_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CT and Sticky MethylSeg HMM tests for one sample config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML config file for the sample.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_HMM_RESULTS_DIR),
        help="Directory where HMM test outputs should be written.",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force rerunning shared prep and HMM segmentation outputs.",
    )

    args = parser.parse_args()
    run(args.config, args.out_dir, force_recreate=args.force_recreate)
