#!/bin/bash
#SBATCH --partition=notchpeak-guest
#SBATCH --account=owner-guest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=1-1
#SBATCH --output=logs/methylation_tcga_segmentation.%A_%a.out
#SBATCH --error=logs/methylation_tcga_segmentation.%A_%a.err
#SBATCH --requeue

set -euo pipefail

RESULTS_ROOT=${RESULTS_ROOT:?RESULTS_ROOT is required}
SEGMENTATION_ROOT=${SEGMENTATION_ROOT:?SEGMENTATION_ROOT is required}
MANIFEST=${MANIFEST:?MANIFEST is required}
SUMMARY_DIR=${SUMMARY_DIR:?SUMMARY_DIR is required}
PIPELINE_SCRIPT=${PIPELINE_SCRIPT:?PIPELINE_SCRIPT is required}

mkdir -p "$RESULTS_ROOT/logs" "$RESULTS_ROOT/manifests" "$SEGMENTATION_ROOT" "$SUMMARY_DIR"
cd "$RESULTS_ROOT"

source /uufs/chpc.utah.edu/common/home/clementm-group1/conda/mambaforge/etc/profile.d/conda.sh
conda activate jt_wgbs_analysis

if [ -z "${ARRAY_TASK_COUNT:-}" ]; then
  echo "ARRAY_TASK_COUNT environment variable is required." >&2
  exit 1
fi

cmd=(
  python "$PIPELINE_SCRIPT"
  --manifest "$MANIFEST"
  --segmentation-root "$SEGMENTATION_ROOT"
  --summary-dir "$SUMMARY_DIR"
  --array-index "${SLURM_ARRAY_TASK_ID}"
  --array-count "${ARRAY_TASK_COUNT}"
)

if [ "${FORCE_RECREATE_PMDS:-0}" = "1" ]; then
  cmd+=(--force-recreate)
else
  cmd+=(--skip-existing)
fi

"${cmd[@]}"

echo "Done at $(date)"
