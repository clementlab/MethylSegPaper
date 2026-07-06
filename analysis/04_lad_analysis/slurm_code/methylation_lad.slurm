#!/bin/bash
#SBATCH --job-name=methylation_lad
#SBATCH --partition=notchpeak-guest
#SBATCH --account=owner-guest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/methylation_lad.%j.out
#SBATCH --error=logs/methylation_lad.%j.err
#SBATCH --requeue

set -euo pipefail

SCRIPT_DIR="${LAD_SLURM_CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
OUT_ROOT="${LAD_OUT_ROOT:-$REPO_ROOT/results/04_lad_analysis}"
SEGMENTATION_RESULTS_PATH="${SEGMENTATION_RESULTS_PATH:-$REPO_ROOT/results/01_region_calling_analysis}"
PIPELINE="${LAD_PIPELINE_SCRIPT:-$SCRIPT_DIR/01_run_lad.py}"

mkdir -p "$OUT_ROOT/logs"
cd "$OUT_ROOT"

source /uufs/chpc.utah.edu/common/home/clementm-group1/conda/mambaforge/bin/activate
conda activate jt_wgbs_analysis

ARGS=(
  --segmentation-results-path "$SEGMENTATION_RESULTS_PATH"
  --output-dir "$OUT_ROOT"
)

if [ -n "${LAD_SAMPLES:-}" ]; then
  # shellcheck disable=SC2206
  SAMPLE_ARRAY=(${LAD_SAMPLES})
  if [ "${#SAMPLE_ARRAY[@]}" -gt 0 ]; then
    ARGS+=(--samples "${SAMPLE_ARRAY[@]}")
  fi
fi

if [ "${RUN_DEEPTOOLS:-0}" = "1" ]; then
  ARGS+=(--run-deeptools)
else
  ARGS+=(--skip-deeptools)
fi

if [ "${INCLUDE_HEATMAPS:-0}" = "1" ]; then
  ARGS+=(--include-heatmaps)
fi

if [ "${FORCE_REFERENCE_REBUILD:-0}" = "1" ]; then
  ARGS+=(--force-reference-rebuild)
fi

if [ "${FORCE_DEEPTOOLS:-0}" = "1" ]; then
  ARGS+=(--force-deeptools)
fi

if [ -n "${PRIMARY_WINDOW_BP:-}" ]; then
  ARGS+=(--primary-window-bp "$PRIMARY_WINDOW_BP")
fi

if [ -n "${PROFILE_BIN_BP:-}" ]; then
  ARGS+=(--profile-bin-bp "$PROFILE_BIN_BP")
fi

if [ -n "${PROFILE_REGION_BODY_BP:-}" ]; then
  ARGS+=(--profile-region-body-bp "$PROFILE_REGION_BODY_BP")
fi

python "$PIPELINE" "${ARGS[@]}"

echo "LAD analysis job done at $(date)"
