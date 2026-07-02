#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
OUT_ROOT="$REPO_ROOT/results/02_synthetic_analysis"
FORCE_RECREATE=0
CONDA_ROOT=/uufs/chpc.utah.edu/common/home/clementm-group1/conda/mambaforge
CONDA_PYTHON=$CONDA_ROOT/envs/jt_wgbs_analysis/bin/python
if [ ! -x "$CONDA_PYTHON" ]; then
  CONDA_PYTHON=$CONDA_ROOT/env/jt_wgbs_analysis/bin/python
fi

usage() {
  cat <<EOF
Usage: ./run_slurm.sh [--out-root PATH] [--force_recreate]

Submits the synthetic PMD benchmark pipeline.

Defaults:
  --out-root \$REPO_ROOT/results/02_synthetic_analysis

By default, validated injected synthetic samples are reused. Use
--force_recreate to archive them and rebuild synthetic samples from scratch.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --out-root)
      OUT_ROOT=$2
      shift 2
      ;;
    --force_recreate|--force-recreate)
      FORCE_RECREATE=1
      shift
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

HEALTHY_SLURM="$SCRIPT_DIR/synthetic_healthy_array.slurm"
BACKGROUND_SLURM="$SCRIPT_DIR/synthetic_build_backgrounds.slurm"
INJECT_SLURM="$SCRIPT_DIR/synthetic_inject_pmds.slurm"
RECOVERY_SLURM="$SCRIPT_DIR/synthetic_recovery_array.slurm"
METRIC_ARRAY_SLURM="$SCRIPT_DIR/synthetic_metric_array.slurm"
FINALIZE_METRICS_SLURM="$SCRIPT_DIR/synthetic_finalize_metrics.slurm"

mkdir -p "$OUT_ROOT/archives"
cd "$OUT_ROOT"

VALIDATION_STATE=$("$CONDA_PYTHON" - "$SCRIPT_DIR" "$OUT_ROOT" <<'PY'
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
out_root = Path(sys.argv[2])
sys.path.insert(0, str(script_dir))
import pipeline_config as cfg

manifest_path = cfg.injected_manifest_path(out_root)
injected_dir = cfg.paths(out_root)["injected"]
valid, messages = cfg.validate_injected_manifest(out_root)
if valid:
    print("valid")
elif manifest_path.exists() or injected_dir.exists():
    print("invalid")
    for message in messages:
        print(message)
else:
    print("missing")
PY
)

VALIDATION_HEAD=$(printf '%s\n' "$VALIDATION_STATE" | head -n 1)
REUSE_SYNTHETIC=0
if [ "$FORCE_RECREATE" = "0" ]; then
  if [ "$VALIDATION_HEAD" = "valid" ]; then
    REUSE_SYNTHETIC=1
  elif [ "$VALIDATION_HEAD" = "invalid" ]; then
    echo "Existing injected synthetic samples failed validation:" >&2
    printf '%s\n' "$VALIDATION_STATE" | tail -n +2 >&2
    echo "Use --force_recreate to archive and rebuild injected synthetic samples." >&2
    exit 1
  fi
fi

ts=$(date +%Y%m%d_%H%M%S)
archive_path() {
  local path=$1
  if [ -e "$path" ]; then
    mkdir -p archives
    mv "$path" "archives/${path}.${ts}"
    echo "Archived $path -> archives/${path}.${ts}"
  fi
}

archive_path logs
archive_path healthy_pmds
archive_path healthy_background_references
archive_path synthetic_recovery
archive_path manifests
if [ "$FORCE_RECREATE" = "1" ]; then
  archive_path injected_pmd_samples
fi
mkdir -p logs manifests

N_SOURCE=$("$CONDA_PYTHON" - "$SCRIPT_DIR" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import pipeline_config as cfg
print(len(cfg.included_records()))
PY
)

if [ "$REUSE_SYNTHETIC" = "1" ]; then
  N_RECOVERY=$("$CONDA_PYTHON" - "$SCRIPT_DIR" "$OUT_ROOT" <<'PY'
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, sys.argv[1])
import pipeline_config as cfg
manifest = pd.read_csv(cfg.injected_manifest_path(Path(sys.argv[2])), sep="\t")
print(len(manifest))
PY
)
else
  N_RECOVERY=$N_SOURCE
fi

if [ -z "$N_SOURCE" ] || [ "$N_SOURCE" -lt 1 ]; then
  echo "No source samples are configured." >&2
  exit 1
fi
if [ -z "$N_RECOVERY" ] || [ "$N_RECOVERY" -lt 1 ]; then
  echo "No recovery samples are configured." >&2
  exit 1
fi

export_args=ALL,SYNTHETIC_OUT_ROOT="$OUT_ROOT",FORCE_RECREATE="$FORCE_RECREATE",SYNTHETIC_SLURM_CODE_DIR="$SCRIPT_DIR"

if [ "$REUSE_SYNTHETIC" = "1" ]; then
  echo "Reusing validated injected synthetic samples in $OUT_ROOT/injected_pmd_samples"
  inject_job_id=$(sbatch --parsable \
    --chdir="$OUT_ROOT" \
    --job-name=synthetic_reuse_${ts} \
    --export="$export_args" \
    "$INJECT_SLURM")
else
  healthy_job_id=$(sbatch --parsable \
    --chdir="$OUT_ROOT" \
    --job-name=synthetic_healthy_${ts} \
    --export="$export_args" \
    --array=1-"$N_SOURCE" "$HEALTHY_SLURM")

  background_job_id=$(sbatch --parsable \
    --chdir="$OUT_ROOT" \
    --job-name=synthetic_backgrounds_${ts} \
    --dependency=afterok:${healthy_job_id} \
    --export="$export_args" \
    "$BACKGROUND_SLURM")

  inject_job_id=$(sbatch --parsable \
    --chdir="$OUT_ROOT" \
    --job-name=synthetic_inject_${ts} \
    --dependency=afterok:${background_job_id} \
    --export="$export_args" \
    "$INJECT_SLURM")

  echo "Submitted healthy array job: $healthy_job_id"
  echo "Submitted background job: $background_job_id"
fi

recovery_job_id=$(sbatch --parsable \
  --chdir="$OUT_ROOT" \
  --job-name=synthetic_recovery_${ts} \
  --dependency=afterok:${inject_job_id} \
  --export="$export_args" \
  --array=1-"$N_RECOVERY" "$RECOVERY_SLURM")

metric_array_job_id=$(sbatch --parsable \
  --chdir="$OUT_ROOT" \
  --job-name=synthetic_metric_array_${ts} \
  --dependency=afterok:${recovery_job_id} \
  --export="$export_args" \
  --array=1-"$N_RECOVERY" "$METRIC_ARRAY_SLURM")

metrics_job_id=$(sbatch --parsable \
  --chdir="$OUT_ROOT" \
  --job-name=synthetic_finalize_metrics_${ts} \
  --dependency=afterok:${metric_array_job_id} \
  --export="$export_args" \
  "$FINALIZE_METRICS_SLURM")

echo "Output root: $OUT_ROOT"
echo "Source samples: $N_SOURCE"
echo "Recovery samples: $N_RECOVERY"
echo "Force recreate: $FORCE_RECREATE"
echo "Submitted inject/reuse job: $inject_job_id"
echo "Submitted recovery array job: $recovery_job_id"
echo "Submitted metric array job: $metric_array_job_id"
echo "Submitted metrics job: $metrics_job_id"
