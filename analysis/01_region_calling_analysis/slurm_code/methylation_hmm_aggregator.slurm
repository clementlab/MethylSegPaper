#!/bin/bash
#SBATCH --job-name=methylation_hmm_aggregate
#SBATCH --partition=notchpeak-guest
#SBATCH --account=owner-guest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/methylation_hmm_aggregate.%j.out
#SBATCH --error=logs/methylation_hmm_aggregate.%j.err
#SBATCH --requeue

set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
CONFIGS_FILE="${CONFIGS_FILE:-$SCRIPT_DIR/configs.txt}"
AGGREGATOR="${HMM_AGGREGATOR_SCRIPT:-$SCRIPT_DIR/run_hmm_aggregator.py}"
HMM_RESULTS_DIR="${HMM_RESULTS_DIR:-$REPO_ROOT/results/01_region_calling_analysis/hmm_tests}"

SCRDIR=/scratch/general/vast/$USER/slurm_jobs/
mkdir -p "$SCRDIR"
cd "$SCRDIR"

source /uufs/chpc.utah.edu/common/home/clementm-group1/conda/mambaforge/bin/activate
conda activate jt_wgbs_analysis

python "$AGGREGATOR" --configs "$CONFIGS_FILE" --out-dir "$HMM_RESULTS_DIR"

echo "Aggregate summary job done at $(date)"
