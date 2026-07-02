#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SYNTHETIC_ROOT = THIS_DIR.parent
for import_path in (THIS_DIR, SYNTHETIC_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import pipeline_config as cfg
import synthetic_analysis_helpers as sah


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build healthy-background references after healthy PMD calls.")
    parser.add_argument("--out-root", type=Path, default=cfg.DEFAULT_OUT_ROOT)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--min-beta", type=float, default=0.2)
    parser.add_argument("--max-beta", type=float, default=0.8)
    parser.add_argument("--n-procs", type=int, default=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    return parser.parse_args()


def write_detail_tables(sample_results: dict, output_dir: Path) -> None:
    detail_root = output_dir / "region_summaries"
    detail_root.mkdir(parents=True, exist_ok=True)
    for sample_id, result in sample_results.items():
        result["scored_pmds_df"].to_csv(
            detail_root / f"{sample_id}.candidate_pmd_scores.tsv", sep="\t", index=False
        )
        result["pmds_to_remove_df"].to_csv(
            detail_root / f"{sample_id}.removed_pmds.tsv", sep="\t", index=False
        )
        result["kept_pmds_df"].to_csv(
            detail_root / f"{sample_id}.kept_pmds.tsv", sep="\t", index=False
        )
        result["tool_summary_df"].to_csv(
            detail_root / f"{sample_id}.tool_region_counts.tsv", sep="\t", index=False
        )


def main() -> int:
    args = parse_args()
    path_map = cfg.ensure_base_dirs(args.out_root)
    records = cfg.included_records(args.sample_id or None)
    cfg.write_source_manifest(args.out_root, args.sample_id or None)

    sample_results, manifest_df, manifest_path = sah.build_healthy_background_references(
        records=records,
        healthy_pmd_root=path_map["healthy_pmds"],
        output_dir=path_map["backgrounds"],
        min_beta=args.min_beta,
        max_beta=args.max_beta,
        n_procs=args.n_procs,
    )
    write_detail_tables(sample_results, path_map["backgrounds"])
    validation_df = sah.validate_background_manifest(
        manifest_path, healthy_pmd_root=path_map["healthy_pmds"]
    )
    validation_df.to_csv(
        path_map["backgrounds"] / "background_validation.tsv", sep="\t", index=False
    )

    print(f"Wrote healthy-background manifest: {manifest_path}")
    print(f"Background samples: {len(manifest_df):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
