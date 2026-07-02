#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
RESULTS_ROOT="$REPO_ROOT/results/05_tcga_classification_analysis"
SEGMENTATION_ROOT="$RESULTS_ROOT/segmentation"
ML_OUT_DIR="$RESULTS_ROOT/ml_outputs"
ML_SLURM="$SCRIPT_DIR/methylation_tcga_ml.slurm"
ML_PIPELINE="$SCRIPT_DIR/run_tcga_ml_pipeline.py"

CASE_SETS=${CASE_SETS:-brca,per_cancer,pan_cancer,multiclass}
N_VALUES=${N_VALUES:-}
CV_SPLITS=${CV_SPLITS:-5}
N_ESTIMATORS=${N_ESTIMATORS:-500}
RANDOM_STATE=${RANDOM_STATE:-42}
MAX_SAMPLES_PER_CLASS=${MAX_SAMPLES_PER_CLASS:-}
SAVE_FEATURE_MATRICES=${SAVE_FEATURE_MATRICES:-1}
ALLOW_MISSING_FEATURE_SETS=${ALLOW_MISSING_FEATURE_SETS:-0}
CALL_MISSING_PMDS=${CALL_MISSING_PMDS:-0}
HIGH_CONFIDENCE_PMD_MIN_FRACTION=${HIGH_CONFIDENCE_PMD_MIN_FRACTION:-0.5}
RANDOM_REGION_MIN_LENGTH_BP=${RANDOM_REGION_MIN_LENGTH_BP:-150000}
RANDOM_REGION_MAX_LENGTH_BP=${RANDOM_REGION_MAX_LENGTH_BP:-20000000}

if [ ! -d "$SEGMENTATION_ROOT/methylseg" ]; then
  echo "Missing segmentation root: $SEGMENTATION_ROOT/methylseg" >&2
  echo "Run ./run_PMD_detection.sh first." >&2
  exit 1
fi

segment_count=$(find "$SEGMENTATION_ROOT/methylseg" -path '*/summary_files/segments_cleaned_PMD.bed' | wc -l)
if [ "$segment_count" -lt 1 ]; then
  echo "No cached MethylSeg PMD outputs were found under $SEGMENTATION_ROOT." >&2
  echo "Run ./run_PMD_detection.sh first." >&2
  exit 1
fi

mkdir -p "$RESULTS_ROOT/logs"
cd "$RESULTS_ROOT"

ts=$(date +%Y%m%d_%H%M%S)
if [ -d "$ML_OUT_DIR" ]; then
  mv "$ML_OUT_DIR" "${ML_OUT_DIR}_backup_$ts"
fi

ml_job_id=$(sbatch --parsable \
  --chdir="$RESULTS_ROOT" \
  --job-name=tcga_ml_${ts} \
  --export=ALL,RESULTS_ROOT="${RESULTS_ROOT}",SEGMENTATION_ROOT="${SEGMENTATION_ROOT}",ML_OUT_DIR="${ML_OUT_DIR}",PIPELINE_SCRIPT="${ML_PIPELINE}",CASE_SETS="${CASE_SETS}",N_VALUES="${N_VALUES}",CV_SPLITS="${CV_SPLITS}",N_ESTIMATORS="${N_ESTIMATORS}",RANDOM_STATE="${RANDOM_STATE}",MAX_SAMPLES_PER_CLASS="${MAX_SAMPLES_PER_CLASS}",SAVE_FEATURE_MATRICES="${SAVE_FEATURE_MATRICES}",ALLOW_MISSING_FEATURE_SETS="${ALLOW_MISSING_FEATURE_SETS}",CALL_MISSING_PMDS="${CALL_MISSING_PMDS}",HIGH_CONFIDENCE_PMD_MIN_FRACTION="${HIGH_CONFIDENCE_PMD_MIN_FRACTION}",RANDOM_REGION_MIN_LENGTH_BP="${RANDOM_REGION_MIN_LENGTH_BP}",RANDOM_REGION_MAX_LENGTH_BP="${RANDOM_REGION_MAX_LENGTH_BP}" \
  "$ML_SLURM")

echo "Segmentation root: $SEGMENTATION_ROOT"
echo "ML output dir: $ML_OUT_DIR"
echo "Case sets: $CASE_SETS"
echo "N values: ${N_VALUES:-default}"
echo "CV splits: $CV_SPLITS"
echo "N estimators: $N_ESTIMATORS"
echo "CALL_MISSING_PMDS: $CALL_MISSING_PMDS"
echo "Submitted ML job: $ml_job_id"
