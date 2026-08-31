#!/bin/bash
#SBATCH --partition=notchpeak-guest
#SBATCH --account=owner-guest
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/methylation_tcga_cleanup.%j.out
#SBATCH --error=logs/methylation_tcga_cleanup.%j.err
#SBATCH --requeue

set -euo pipefail

RESULTS_ROOT=${RESULTS_ROOT:?RESULTS_ROOT is required}
CLEANUP_TARGETS=${CLEANUP_TARGETS:?CLEANUP_TARGETS is required}

mkdir -p "$RESULTS_ROOT/logs"
cd "$RESULTS_ROOT"

IFS='|' read -r -a cleanup_paths <<< "$CLEANUP_TARGETS"

for cleanup_path in "${cleanup_paths[@]}"; do
  if [ -z "$cleanup_path" ]; then
    continue
  fi
  if [ -e "$cleanup_path" ]; then
    echo "Deleting backup: $cleanup_path"
    rm -rf -- "$cleanup_path"
  else
    echo "Skipping missing backup: $cleanup_path"
  fi
done

echo "Cleanup done at $(date)"
