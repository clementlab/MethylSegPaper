#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
RESULTS_ROOT="$REPO_ROOT/results/01_region_calling_analysis"
HMM_RESULTS_DIR="$RESULTS_ROOT/hmm_tests"
ARRAY_SLURM="$SCRIPT_DIR/methylation_hmm_test.slurm"
AGGREGATE_SLURM="$SCRIPT_DIR/methylation_hmm_aggregator.slurm"
CONFIG_SOURCE="$SCRIPT_DIR/configs.txt"
RUNTIME_CONFIGS="$HMM_RESULTS_DIR/configs_runtime.txt"

mkdir -p "$HMM_RESULTS_DIR"

ts=$(date +%s)

if [ -d "$HMM_RESULTS_DIR" ] && [ -n "$(ls -A "$HMM_RESULTS_DIR")" ]; then
  mv "$HMM_RESULTS_DIR" "${HMM_RESULTS_DIR}_backup_$ts"
  mkdir -p "$HMM_RESULTS_DIR"
fi

mkdir -p "$HMM_RESULTS_DIR/logs"
cd "$HMM_RESULTS_DIR"

python - "$CONFIG_SOURCE" "$SCRIPT_DIR/configs" "$RUNTIME_CONFIGS" <<'PY'
from pathlib import Path
import sys

source_manifest = Path(sys.argv[1])
config_dir = Path(sys.argv[2]).resolve()
runtime_manifest = Path(sys.argv[3])

resolved_paths = []
missing = []
for raw_line in source_manifest.read_text().splitlines():
    line = raw_line.strip()
    if not line:
        continue
    original_path = Path(line)
    candidates = [
        config_dir / original_path.name,
        original_path,
        source_manifest.parent / original_path,
    ]
    resolved = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
    if resolved is None:
        missing.append(line)
        continue
    resolved_paths.append(str(resolved))

if missing:
    raise SystemExit(
        "Missing config files referenced by configs.txt:\n" + "\n".join(missing)
    )
if not resolved_paths:
    raise SystemExit("No config files were resolved for the HMM array job.")

runtime_manifest.write_text("\n".join(resolved_paths) + "\n")
PY

N=$(wc -l < "$RUNTIME_CONFIGS")
if [ "$N" -lt 1 ]; then
  echo "No config files were found for the HMM array job." >&2
  exit 1
fi

array_job_id=$(sbatch --parsable \
  --chdir="$HMM_RESULTS_DIR" \
  --job-name=hmm_array_${ts} \
  --export=ALL,CONFIGS_FILE="$RUNTIME_CONFIGS",HMM_RESULTS_DIR="$HMM_RESULTS_DIR",HMM_PIPELINE_SCRIPT="$SCRIPT_DIR/run_hmm_test.py" \
  --array=1-"$N" "$ARRAY_SLURM")

aggregate_job_id=$(sbatch --parsable \
  --chdir="$HMM_RESULTS_DIR" \
  --job-name=hmm_aggregate_${ts} \
  --dependency=afterok:${array_job_id} \
  --export=ALL,CONFIGS_FILE="$RUNTIME_CONFIGS",HMM_RESULTS_DIR="$HMM_RESULTS_DIR",HMM_AGGREGATOR_SCRIPT="$SCRIPT_DIR/run_hmm_aggregator.py" \
  "$AGGREGATE_SLURM")

echo "Submitted HMM array job: $array_job_id"
echo "Submitted HMM aggregate job: $aggregate_job_id"
