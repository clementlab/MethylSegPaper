#!/bin/bash
#SBATCH --job-name=synthetic_metrics
#SBATCH --partition=notchpeak-guest
#SBATCH --account=owner-guest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/synthetic_metrics.%j.out
#SBATCH --error=logs/synthetic_metrics.%j.err
#SBATCH --requeue

set -euo pipefail

SCRIPT_DIR=${SYNTHETIC_SLURM_CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
OUT_ROOT=${SYNTHETIC_OUT_ROOT:-/scratch/general/vast/$USER/slurm_jobs/synthetic_analysis}
PIPELINE="$SCRIPT_DIR/aggregate_metrics.py"

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
  --boundary-tolerance-bp "${BOUNDARY_TOLERANCE_BP:-10000}"

echo "Synthetic metrics job done at $(date)"
