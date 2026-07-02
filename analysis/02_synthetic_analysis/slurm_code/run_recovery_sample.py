#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
SYNTHETIC_ROOT = THIS_DIR.parent
for import_path in (THIS_DIR, SYNTHETIC_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import pipeline_config as cfg
import synthetic_analysis_helpers as sah


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic PMD recovery tools for one injected sample.")
    parser.add_argument("--out-root", type=Path, default=cfg.DEFAULT_OUT_ROOT)
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--force-recreate", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    return parser.parse_args()


def select_manifest_row(manifest_df: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    manifest_df = manifest_df.sort_values("synthetic_sample_id").reset_index(drop=True)
    if args.sample_id:
        matches = manifest_df.loc[manifest_df["synthetic_sample_id"] == args.sample_id]
        if matches.empty:
            raise ValueError(f"Sample ID not found in injected manifest: {args.sample_id}")
        return matches.iloc[0]
    array_index = args.array_index
    if array_index is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
        array_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if array_index is None:
        raise ValueError("Provide --sample-id, --array-index, or SLURM_ARRAY_TASK_ID.")
    if array_index < 1 or array_index > len(manifest_df):
        raise ValueError(f"array_index must be between 1 and {len(manifest_df)}; got {array_index}.")
    return manifest_df.iloc[array_index - 1]


def main() -> int:
    args = parse_args()
    path_map = cfg.ensure_base_dirs(args.out_root)
    manifest_path = cfg.require_valid_injected_manifest(args.out_root)
    manifest_df = pd.read_csv(manifest_path, sep="\t")
    row = select_manifest_row(manifest_df, args)
    sample_id = row["synthetic_sample_id"]

    config_dir = path_map["recovery_tool_results"] / "configs"
    config_df = sah.write_recovery_configs(pd.DataFrame([row]), config_dir)
    config_path = config_df.iloc[0]["config_path"]
    comparator_class = sah.get_methyl_tool_comparator_class(cfg.COMPARATOR_DIR)
    comparator = comparator_class(
        config_file=str(config_path),
        out_dir=str(path_map["recovery_tool_results"]),
        force_recreate=args.force_recreate,
        n_jobs=args.n_jobs,
    )
    comparator.run()

    summary_dir = path_map["recovery_tool_results"] / "run_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "synthetic_sample_id": sample_id,
                "config_path": str(config_path),
                "tool_results_dir": str(path_map["recovery_tool_results"]),
                "force_recreate": bool(args.force_recreate),
                "n_jobs": int(args.n_jobs),
                "status": "completed",
            }
        ]
    ).to_csv(summary_dir / f"{sample_id}.tsv", sep="\t", index=False)
    print(f"Completed synthetic recovery for {sample_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
