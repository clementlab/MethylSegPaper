#!/bin/bash
#SBATCH --partition=notchpeak-guest
#SBATCH --account=owner-guest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=370G
#SBATCH --time=12:00:00
#SBATCH --array=1-1
#SBATCH --output=logs/methylation.%A_%a.out
#SBATCH --error=logs/methylation.%A_%a.err
#SBATCH --requeue

set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
CONFIGS_FILE="${CONFIGS_FILE:-$SCRIPT_DIR/configs.txt}"
PIPELINE="${PIPELINE_SCRIPT:-$SCRIPT_DIR/run_pathway.py}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results/01_region_calling_analysis}"

# ---- Scratch ----
SCRDIR=/scratch/general/vast/$USER/slurm_jobs/
mkdir -p "$SCRDIR"
cd "$SCRDIR"

# ---- Conda ----
source /uufs/chpc.utah.edu/common/home/clementm-group1/conda/mambaforge/bin/activate
conda activate jt_wgbs_analysis

# ---- Select config for this task ----
CONFIG=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CONFIGS_FILE")
if [ -z "$CONFIG" ]; then
  echo "No config found for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID in $CONFIGS_FILE" >&2
  exit 1
fi

echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Config: $CONFIG"
echo "CPUs: $SLURM_CPUS_PER_TASK"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python "$PIPELINE" \
  --config "$CONFIG" \
  --out-dir "$RESULTS_DIR" \
  --n-jobs 6

echo "Done at $(date)"
