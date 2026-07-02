#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
OUT_ROOT="$REPO_ROOT/results/02_synthetic_analysis"
ARRAY_SPEC=""
CONDA_ROOT=/uufs/chpc.utah.edu/common/home/clementm-group1/conda/mambaforge
CONDA_PYTHON=$CONDA_ROOT/envs/jt_wgbs_analysis/bin/python
if [ ! -x "$CONDA_PYTHON" ]; then
  CONDA_PYTHON=$CONDA_ROOT/env/jt_wgbs_analysis/bin/python
fi

usage() {
  cat <<EOF
Usage: ./rerun_metrics_only.sh [--out-root PATH] [--array SPEC]

Submit only the synthetic metric array and final aggregation jobs.

Defaults:
  --out-root \$REPO_ROOT/results/02_synthetic_analysis
  --array    all samples from injected_pmd_samples/synthetic_sample_manifest.tsv

Examples:
  ./rerun_metrics_only.sh
  ./rerun_metrics_only.sh --array 3,7,11
  ./rerun_metrics_only.sh --array 1-12
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --out-root)
      OUT_ROOT=$2
      shift 2
      ;;
    --array)
      ARRAY_SPEC=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

OUT_ROOT=$("$CONDA_PYTHON" - "$OUT_ROOT" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).expanduser().resolve())
PY
)

mkdir -p "$OUT_ROOT/logs"
cd "$OUT_ROOT"

MANIFEST_PATH="$OUT_ROOT/injected_pmd_samples/synthetic_sample_manifest.tsv"
if [ ! -f "$MANIFEST_PATH" ]; then
  echo "Missing injected manifest: $MANIFEST_PATH" >&2
  exit 1
fi

if [ -z "$ARRAY_SPEC" ]; then
  N_RECOVERY=$("$CONDA_PYTHON" - "$MANIFEST_PATH" <<'PY'
import sys
import pandas as pd
print(len(pd.read_csv(sys.argv[1], sep="\t")))
PY
)
  if [ -z "$N_RECOVERY" ] || [ "$N_RECOVERY" -lt 1 ]; then
    echo "Injected manifest appears empty: $MANIFEST_PATH" >&2
    exit 1
  fi
  ARRAY_SPEC="1-$N_RECOVERY"
fi

metric_job_id=$(sbatch --parsable \
  --chdir="$OUT_ROOT" \
  --export=ALL,SYNTHETIC_OUT_ROOT="$OUT_ROOT",SYNTHETIC_SLURM_CODE_DIR="$SCRIPT_DIR" \
  --array="$ARRAY_SPEC" \
  "$SCRIPT_DIR/synthetic_metric_array.slurm")

finalize_job_id=$(sbatch --parsable \
  --chdir="$OUT_ROOT" \
  --dependency=afterok:${metric_job_id} \
  --export=ALL,SYNTHETIC_OUT_ROOT="$OUT_ROOT",SYNTHETIC_SLURM_CODE_DIR="$SCRIPT_DIR" \
  "$SCRIPT_DIR/synthetic_finalize_metrics.slurm")

echo "Output root: $OUT_ROOT"
echo "Metric array spec: $ARRAY_SPEC"
echo "Submitted metric array job: $metric_job_id"
echo "Submitted finalize job: $finalize_job_id"
