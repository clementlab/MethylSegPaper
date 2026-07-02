#!/bin/bash
#SBATCH --job-name=tcga_ml
#SBATCH --partition=notchpeak-guest
#SBATCH --account=owner-guest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/methylation_tcga_ml.%j.out
#SBATCH --error=logs/methylation_tcga_ml.%j.err
#SBATCH --requeue

set -euo pipefail

RESULTS_ROOT=${RESULTS_ROOT:?RESULTS_ROOT is required}
SEGMENTATION_ROOT=${SEGMENTATION_ROOT:?SEGMENTATION_ROOT is required}
ML_OUT_DIR=${ML_OUT_DIR:?ML_OUT_DIR is required}
PIPELINE_SCRIPT=${PIPELINE_SCRIPT:?PIPELINE_SCRIPT is required}

mkdir -p "$RESULTS_ROOT/logs" "$ML_OUT_DIR"
cd "$RESULTS_ROOT"

source /uufs/chpc.utah.edu/common/home/clementm-group1/conda/mambaforge/etc/profile.d/conda.sh
conda activate jt_wgbs_analysis

cmd=(
  python "$PIPELINE_SCRIPT" run
  --out-dir "$ML_OUT_DIR"
  --segmentation-root "$SEGMENTATION_ROOT"
  --case-sets "${CASE_SETS:-brca,per_cancer,pan_cancer,multiclass}"
  --cv-splits "${CV_SPLITS:-5}"
  --n-estimators "${N_ESTIMATORS:-500}"
  --random-state "${RANDOM_STATE:-42}"
  --high-confidence-pmd-min-fraction "${HIGH_CONFIDENCE_PMD_MIN_FRACTION:-0.5}"
  --random-region-min-length-bp "${RANDOM_REGION_MIN_LENGTH_BP:-150000}"
  --random-region-max-length-bp "${RANDOM_REGION_MAX_LENGTH_BP:-20000000}"
)

if [ -n "${N_VALUES:-}" ]; then
  cmd+=(--n-values "$N_VALUES")
fi

if [ -n "${MAX_SAMPLES_PER_CLASS:-}" ]; then
  cmd+=(--max-samples-per-class "$MAX_SAMPLES_PER_CLASS")
fi

if [ "${SAVE_FEATURE_MATRICES:-1}" != "1" ]; then
  cmd+=(--no-save-feature-matrices)
fi

if [ "${ALLOW_MISSING_FEATURE_SETS:-0}" = "1" ]; then
  cmd+=(--allow-missing-feature-sets)
fi

if [ "${CALL_MISSING_PMDS:-0}" != "1" ]; then
  cmd+=(--no-call-missing-pmds)
fi

"${cmd[@]}"

echo "Done at $(date)"
