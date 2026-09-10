#!/bin/bash
#SBATCH --job-name=tcga_grouped_postprocess
#SBATCH --partition=notchpeak-guest
#SBATCH --account=owner-guest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/methylation_tcga_postprocess.%j.out
#SBATCH --error=logs/methylation_tcga_postprocess.%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT is required}
RESULTS_ROOT=${RESULTS_ROOT:?RESULTS_ROOT is required}
COHORT_MANIFEST=${COHORT_MANIFEST:?COHORT_MANIFEST is required}
GROUPED_OUT_ROOT=${GROUPED_OUT_ROOT:?GROUPED_OUT_ROOT is required}
SAMPLE_LEVEL_ARCHIVE=${SAMPLE_LEVEL_ARCHIVE:?SAMPLE_LEVEL_ARCHIVE is required}
VALIDATION_SCRIPT=${VALIDATION_SCRIPT:?VALIDATION_SCRIPT is required}
COMPARISON_SCRIPT=${COMPARISON_SCRIPT:?COMPARISON_SCRIPT is required}
FIGURE_NOTEBOOK=${FIGURE_NOTEBOOK:?FIGURE_NOTEBOOK is required}

mkdir -p "$RESULTS_ROOT/logs"
cd "$PROJECT_ROOT"

source /uufs/chpc.utah.edu/common/home/clementm-group1/conda/mambaforge/etc/profile.d/conda.sh
conda activate jt_wgbs_analysis

python "$VALIDATION_SCRIPT" \
  --cohort-manifest "$COHORT_MANIFEST" \
  --out-root "$GROUPED_OUT_ROOT" \
  --report "$RESULTS_ROOT/ml_outputs/grouped_classification_validation.json" \
  --expected-sklearn-version 1.8.0

python "$COMPARISON_SCRIPT" \
  --grouped-root "$GROUPED_OUT_ROOT" \
  --sample-level-root "$SAMPLE_LEVEL_ARCHIVE" \
  --output-tsv "$RESULTS_ROOT/ml_outputs/grouped_vs_sample_level_metrics.tsv" \
  --output-json "$RESULTS_ROOT/ml_outputs/grouped_vs_sample_level_summary.json"

render_dir=$(mktemp -d /tmp/tcga_grouped_figures.XXXXXX)
jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=-1 \
  --output rendered_tcga_figures.ipynb \
  --output-dir "$render_dir" \
  "$FIGURE_NOTEBOOK"

echo "Validated grouped outputs, wrote the comparison report, and regenerated TCGA figures at $(date)."
