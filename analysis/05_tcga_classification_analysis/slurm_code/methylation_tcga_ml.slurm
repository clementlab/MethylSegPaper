#!/bin/bash
#SBATCH --job-name=tcga_ml
#SBATCH --partition=notchpeak-guest
#SBATCH --account=owner-guest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=50
#SBATCH --mem=600G
#SBATCH --time=48:00:00
#SBATCH --output=logs/methylation_tcga_ml.%A_%a.out
#SBATCH --error=logs/methylation_tcga_ml.%A_%a.err
#SBATCH --requeue

set -euo pipefail

RESULTS_ROOT=${RESULTS_ROOT:?RESULTS_ROOT is required}
COHORT_MANIFEST=${COHORT_MANIFEST:?COHORT_MANIFEST is required}
OUT_ROOT=${OUT_ROOT:?OUT_ROOT is required}
PIPELINE_SCRIPT=${PIPELINE_SCRIPT:?PIPELINE_SCRIPT is required}
FEATURE_COUNT=${FEATURE_COUNT:?FEATURE_COUNT is required}
N_SPLITS=${N_SPLITS:?N_SPLITS is required}
N_REPEATS=${N_REPEATS:?N_REPEATS is required}
N_FEATURE_DRAWS=${N_FEATURE_DRAWS:?N_FEATURE_DRAWS is required}
MAX_SPLIT_WORKERS=${MAX_SPLIT_WORKERS:?MAX_SPLIT_WORKERS is required}
RANDOM_SEED=${RANDOM_SEED:?RANDOM_SEED is required}
MIN_CPGS_FOR_RANDOM_REGIONS=${MIN_CPGS_FOR_RANDOM_REGIONS:?MIN_CPGS_FOR_RANDOM_REGIONS is required}
MIN_TRAINING_OBSERVED_CPGS_PER_REGION=${MIN_TRAINING_OBSERVED_CPGS_PER_REGION:?MIN_TRAINING_OBSERVED_CPGS_PER_REGION is required}
EXCLUDE_TOP_NORMAL_SHARED_PMDS=${EXCLUDE_TOP_NORMAL_SHARED_PMDS:-0}

mkdir -p "$RESULTS_ROOT/logs" "$OUT_ROOT"
cd "$RESULTS_ROOT"

source /uufs/chpc.utah.edu/common/home/clementm-group1/conda/mambaforge/etc/profile.d/conda.sh
conda activate jt_wgbs_analysis

extra_args=()
if [[ "$EXCLUDE_TOP_NORMAL_SHARED_PMDS" == "1" ]]; then
  extra_args+=(--exclude-top-normal-shared-pmds)
fi

python "$PIPELINE_SCRIPT" \
  --cohort-manifest "$COHORT_MANIFEST" \
  --array-index "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}" \
  --feature-count "$FEATURE_COUNT" \
  --n-splits "$N_SPLITS" \
  --n-repeats "$N_REPEATS" \
  --n-feature-draws "$N_FEATURE_DRAWS" \
  --max-split-workers "$MAX_SPLIT_WORKERS" \
  --random-seed "$RANDOM_SEED" \
  --min-cpgs-for-random-regions "$MIN_CPGS_FOR_RANDOM_REGIONS" \
  --min-training-observed-cpgs-per-region "$MIN_TRAINING_OBSERVED_CPGS_PER_REGION" \
  --out-root "$OUT_ROOT" \
  "${extra_args[@]}"

echo "Done at $(date)"
