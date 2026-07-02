#!/bin/bash
#SBATCH --job-name=methylation_aggregate
#SBATCH --partition=notchpeak-guest
#SBATCH --account=owner-guest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/methylation_aggregate.%j.out
#SBATCH --error=logs/methylation_aggregate.%j.err
#SBATCH --requeue

set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
CONFIGS="${CONFIGS_FILE:-$SCRIPT_DIR/configs.txt}"
AGGREGATOR="${AGGREGATOR_SCRIPT:-$SCRIPT_DIR/run_aggregator.py}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results/01_region_calling_analysis}"

SCRDIR=/scratch/general/vast/$USER/slurm_jobs/
mkdir -p "$SCRDIR"
cd "$SCRDIR"

source /uufs/chpc.utah.edu/common/home/clementm-group1/conda/mambaforge/bin/activate
conda activate jt_wgbs_analysis

python "$AGGREGATOR" --configs "$CONFIGS" --out-dir "$RESULTS_DIR"

echo "Aggregate summary job done at $(date)"
