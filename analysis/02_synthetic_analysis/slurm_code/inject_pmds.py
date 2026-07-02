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
    parser = argparse.ArgumentParser(description="Create or reuse injected synthetic PMD samples.")
    parser.add_argument("--out-root", type=Path, default=cfg.DEFAULT_OUT_ROOT)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--force-recreate", action="store_true")
    parser.add_argument("--n-procs", type=int, default=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    return parser.parse_args()


def write_recovery_inputs(manifest_df, path_map: dict[str, Path]) -> None:
    config_dir = path_map["recovery_tool_results"] / "configs"
    config_df = sah.write_recovery_configs(manifest_df, config_dir)
    cfg.write_config_list(config_df, path_map["recovery_tool_results"] / "configs.txt")
    cfg.write_sample_list(
        config_df["synthetic_sample_id"].astype(str).tolist(),
        path_map["recovery_tool_results"] / "samples.txt",
    )


def main() -> int:
    args = parse_args()
    path_map = cfg.ensure_base_dirs(args.out_root)
    path_map["injected"].mkdir(parents=True, exist_ok=True)
    path_map["recovery_tool_results"].mkdir(parents=True, exist_ok=True)

    valid_existing, validation_messages = cfg.validate_injected_manifest(args.out_root)
    if valid_existing and not args.force_recreate:
        manifest_path = cfg.injected_manifest_path(args.out_root)
        manifest_df = pd.read_csv(manifest_path, sep="\t")
        if args.sample_id:
            selected = set(args.sample_id)
            manifest_df = manifest_df.loc[
                manifest_df["synthetic_sample_id"].isin(selected)
            ].copy()
        write_recovery_inputs(manifest_df, path_map)
        print(f"Reused validated injected synthetic manifest: {manifest_path}")
        return 0

    if validation_messages and cfg.injected_manifest_path(args.out_root).exists() and not args.force_recreate:
        raise FileNotFoundError(
            "Existing injected synthetic samples failed validation. "
            "Use --force-recreate to rebuild them.\n" + "\n".join(validation_messages)
        )

    background_manifest = path_map["backgrounds"] / "synthetic_sample_manifest.tsv"
    if not background_manifest.exists():
        raise FileNotFoundError(f"Missing healthy-background manifest: {background_manifest}")

    manifest_df, manifest_path = sah.build_injected_manifest(
        background_manifest=background_manifest,
        output_dir=path_map["injected"],
        config=sah.load_default_pmd_config(),
        selected_sample_ids=args.sample_id or None,
        overwrite=args.force_recreate,
        n_procs=args.n_procs,
    )
    validation_df = sah.validate_injected_manifest(manifest_path)
    validation_df.to_csv(path_map["injected"] / "injected_validation.tsv", sep="\t", index=False)
    write_recovery_inputs(manifest_df, path_map)
    print(f"Wrote injected synthetic manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
