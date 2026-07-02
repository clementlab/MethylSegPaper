#!/bin/bash
#SBATCH --job-name=synthetic_metric_sample
#SBATCH --partition=notchpeak-guest
#SBATCH --account=owner-guest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=08:00:00
#SBATCH --array=1-1
#SBATCH --output=logs/synthetic_metric_sample.%A_%a.out
#SBATCH --error=logs/synthetic_metric_sample.%A_%a.err
#SBATCH --requeue

set -euo pipefail

SCRIPT_DIR=${SYNTHETIC_SLURM_CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
OUT_ROOT=${SYNTHETIC_OUT_ROOT:-/scratch/general/vast/$USER/slurm_jobs/synthetic_analysis}
PIPELINE="$SCRIPT_DIR/run_metric_sample.py"

mkdir -p "$OUT_ROOT/logs"
cd "$OUT_ROOT"

source /uufs/chpc.utah.edu/common/home/clementm-group1/conda/mambaforge/etc/profile.d/conda.sh
conda activate jt_wgbs_analysis

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python "$PIPELINE" \
  --out-root "$OUT_ROOT" \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --boundary-tolerance-bp "${BOUNDARY_TOLERANCE_BP:-10000}" \
  --n-tool-procs "${SLURM_CPUS_PER_TASK:-1}"

echo "Synthetic metric sample job done at $(date)"
