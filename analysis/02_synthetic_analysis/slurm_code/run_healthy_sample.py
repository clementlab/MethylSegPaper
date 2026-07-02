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
    parser = argparse.ArgumentParser(description="Run healthy PMD tool calls for one source sample.")
    parser.add_argument("--out-root", type=Path, default=cfg.DEFAULT_OUT_ROOT)
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--force-recreate", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    return parser.parse_args()


def select_record(args: argparse.Namespace) -> dict:
    if args.sample_id:
        return cfg.get_record_by_sample_id(args.sample_id)
    if args.array_index is not None:
        return cfg.get_record_by_array_index(args.array_index)
    slurm_index = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_index:
        return cfg.get_record_by_array_index(int(slurm_index))
    raise ValueError("Provide --sample-id, --array-index, or SLURM_ARRAY_TASK_ID.")


def main() -> int:
    args = parse_args()
    record = select_record(args)
    sample_id = record["synthetic_sample_id"]
    path_map = cfg.ensure_base_dirs(args.out_root)
    healthy_root = path_map["healthy_pmds"]
    config_dir = healthy_root / "configs"
    summary_dir = healthy_root / "run_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    config_pairs = sah.write_comparator_configs([record], config_dir)
    _, config_path = config_pairs[0]
    comparator_class = sah.get_methyl_tool_comparator_class(cfg.COMPARATOR_DIR)
    comparator = comparator_class(
        config_file=str(config_path),
        out_dir=str(healthy_root),
        force_recreate=args.force_recreate,
        n_jobs=args.n_jobs,
    )
    comparator.run()

    pd.DataFrame(
        [
            {
                "synthetic_sample_id": sample_id,
                "config_path": str(config_path),
                "healthy_pmd_root": str(healthy_root),
                "force_recreate": bool(args.force_recreate),
                "n_jobs": int(args.n_jobs),
                "status": "completed",
            }
        ]
    ).to_csv(summary_dir / f"{sample_id}.tsv", sep="\t", index=False)
    print(f"Completed healthy PMD calls for {sample_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
