#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
SCRDIR="$REPO_ROOT/results/01_region_calling_analysis"
ARRAY_SLURM="$SCRIPT_DIR/methylation_comparator.slurm"
AGGREGATE_SLURM="$SCRIPT_DIR/methylation_aggregator.slurm"
CONFIG_SOURCE="$SCRIPT_DIR/configs.txt"
RUNTIME_CONFIGS="$SCRDIR/configs_runtime.txt"

mkdir -p "$SCRDIR"

ts=$(date +%s)

for run_dir in comparison dnmtools methyl_lasso methylseekr methylseg mmseekr aggregate_summaries logs; do
  if [ -d "$SCRDIR/$run_dir" ]; then
    mv "$SCRDIR/$run_dir" "$SCRDIR/${run_dir}_backup_$ts"
  fi
done

mkdir -p "$SCRDIR/logs"
cd "$SCRDIR"

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
        (source_manifest.parent / original_path),
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
    raise SystemExit("No config files were resolved for the Slurm array job.")

runtime_manifest.write_text("\n".join(resolved_paths) + "\n")
PY

N=$(wc -l < "$RUNTIME_CONFIGS")
if [ "$N" -lt 1 ]; then
  echo "No config files were found for the comparator array job." >&2
  exit 1
fi

array_job_id=$(sbatch --parsable \
  --chdir="$SCRDIR" \
  --job-name=comp_array_${ts} \
  --export=ALL,CONFIGS_FILE="$RUNTIME_CONFIGS",RESULTS_DIR="$SCRDIR",PIPELINE_SCRIPT="$SCRIPT_DIR/run_pathway.py" \
  --array=1-$N "$ARRAY_SLURM")

aggregate_job_id=$(sbatch --parsable \
  --chdir="$SCRDIR" \
  --export=ALL,CONFIGS_FILE="$RUNTIME_CONFIGS",RESULTS_DIR="$SCRDIR",AGGREGATOR_SCRIPT="$SCRIPT_DIR/run_aggregator.py" \
  --job-name=aggregate_${ts} \
  --dependency=afterok:${array_job_id} "$AGGREGATE_SLURM")

echo "Submitted comparator array job: $array_job_id"

echo "Submitted aggregate job: $aggregate_job_id"
