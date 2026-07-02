#!/bin/bash
#SBATCH --job-name=methylation_chromatin
#SBATCH --partition=notchpeak-guest
#SBATCH --account=owner-guest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/methylation_chromatin.%j.out
#SBATCH --error=logs/methylation_chromatin.%j.err
#SBATCH --requeue

set -euo pipefail

SCRIPT_DIR="${CHROMATIN_SLURM_CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
OUT_ROOT="${CHROMATIN_OUT_ROOT:-$REPO_ROOT/results/03_chromatin_analysis}"
SEGMENTATION_RESULTS_PATH="${SEGMENTATION_RESULTS_PATH:-$REPO_ROOT/results/01_region_calling_analysis}"
CHROMATIN_DATA_DIR="${CHROMATIN_DATA_DIR:-$REPO_ROOT/data/chromatin_data}"
PIPELINE="${CHROMATIN_PIPELINE_SCRIPT:-$SCRIPT_DIR/run_chromatin.py}"

mkdir -p "$OUT_ROOT/logs"
cd "$OUT_ROOT"

source /uufs/chpc.utah.edu/common/home/clementm-group1/conda/mambaforge/bin/activate
conda activate jt_wgbs_analysis

ARGS=(
  --segmentation-results-path "$SEGMENTATION_RESULTS_PATH"
  --chromatin-data-dir "$CHROMATIN_DATA_DIR"
  --output-dir "$OUT_ROOT"
)

if [ "${SKIP_DEEPTOOLS:-0}" = "1" ]; then
  ARGS+=(--skip-deeptools)
fi

if [ "${INCLUDE_HEATMAPS:-0}" = "1" ]; then
  ARGS+=(--include-heatmaps)
fi

if [ -n "${DEEPTOOLS_TOOLS:-}" ]; then
  # shellcheck disable=SC2206
  TOOL_ARRAY=(${DEEPTOOLS_TOOLS})
  if [ "${#TOOL_ARRAY[@]}" -gt 0 ]; then
    ARGS+=(--deeptools-tools "${TOOL_ARRAY[@]}")
  fi
fi

python "$PIPELINE" "${ARGS[@]}"

echo "Chromatin analysis job done at $(date)"
