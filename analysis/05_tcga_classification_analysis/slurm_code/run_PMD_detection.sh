#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
RESULTS_ROOT="$REPO_ROOT/results/05_tcga_classification_analysis"
MANIFEST="$RESULTS_ROOT/manifests/tcga_samples.tsv"
SEGMENTATION_ROOT="$RESULTS_ROOT/segmentation"
SUMMARY_DIR="$SEGMENTATION_ROOT/array_task_summaries"
SEGMENTATION_SLURM="$SCRIPT_DIR/methylation_tcga_segmentation.slurm"
CLEANUP_SLURM="$SCRIPT_DIR/methylation_tcga_cleanup.slurm"
SEGMENTATION_PY="$SCRIPT_DIR/run_tcga_segmentation_array.py"
ARRAY_TASK_COUNT=${ARRAY_TASK_COUNT:-100}
FORCE_RECREATE_PMDS=0

backup_path_for() {
  local source_path="$1"
  local ts="$2"
  local parent_dir
  local base_name
  parent_dir=$(dirname "$source_path")
  base_name=$(basename "$source_path")
  printf '%s/%s_backup_%s' "$parent_dir" "$base_name" "$ts"
}

stage_backup() {
  local source_path="$1"
  local ts="$2"
  local backup_path
  if [ ! -e "$source_path" ]; then
    return 1
  fi
  backup_path=$(backup_path_for "$source_path" "$ts")
  mv "$source_path" "$backup_path"
  STAGED_BACKUPS+=("$backup_path")
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --force)
      FORCE_RECREATE_PMDS=1
      shift
      ;;
    --array-task-count)
      if [ "$#" -lt 2 ]; then
        echo "--array-task-count requires a value" >&2
        exit 1
      fi
      ARRAY_TASK_COUNT="$2"
      shift 2
      ;;
    -h|--help)
      cat <<EOF
Usage: ./run_PMD_detection.sh [--force] [--array-task-count N]

Submits only the TCGA PMD-detection Slurm array job.

Options:
  --force              Delete existing MethylSeg PMD outputs and remake them.
  --array-task-count N Submit N array tasks. Default: ARRAY_TASK_COUNT or 100.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [ "$ARRAY_TASK_COUNT" -lt 1 ]; then
  echo "ARRAY_TASK_COUNT must be >= 1" >&2
  exit 1
fi

mkdir -p "$RESULTS_ROOT/logs" "$RESULTS_ROOT/manifests" "$SEGMENTATION_ROOT"
cd "$RESULTS_ROOT"

source /uufs/chpc.utah.edu/common/home/clementm-group1/conda/mambaforge/etc/profile.d/conda.sh
conda activate jt_wgbs_analysis

python "$SEGMENTATION_PY" \
  --write-manifest \
  --manifest "$MANIFEST" \
  --segmentation-root "$SEGMENTATION_ROOT"

ts=$(date +%Y%m%d_%H%M%S)
cleanup_job_id=""
STAGED_BACKUPS=()

if [ "$FORCE_RECREATE_PMDS" = "1" ]; then
  echo "Force enabled: moving existing MethylSeg PMD outputs to timestamped backups under $SEGMENTATION_ROOT"
  stage_backup "$SEGMENTATION_ROOT/methylseg" "$ts" || true
  stage_backup "$SEGMENTATION_ROOT/comparison" "$ts" || true
  stage_backup "$SUMMARY_DIR" "$ts" || true
fi

if [ -d "$SUMMARY_DIR" ]; then
  stage_backup "$SUMMARY_DIR" "$ts" || true
fi
mkdir -p "$SUMMARY_DIR"

if [ "${#STAGED_BACKUPS[@]}" -gt 0 ]; then
  cleanup_targets=$(printf '%s|' "${STAGED_BACKUPS[@]}")
  cleanup_targets=${cleanup_targets%|}
  cleanup_job_id=$(sbatch --parsable \
    --chdir="$RESULTS_ROOT" \
    --job-name=tcga_cleanup_${ts} \
    --export=ALL,RESULTS_ROOT="${RESULTS_ROOT}",CLEANUP_TARGETS="${cleanup_targets}" \
    "$CLEANUP_SLURM")
fi

array_job_id=$(sbatch --parsable \
  --chdir="$RESULTS_ROOT" \
  --job-name=tcga_pmd_${ts} \
  --export=ALL,ARRAY_TASK_COUNT="${ARRAY_TASK_COUNT}",RESULTS_ROOT="${RESULTS_ROOT}",SEGMENTATION_ROOT="${SEGMENTATION_ROOT}",MANIFEST="${MANIFEST}",SUMMARY_DIR="${SUMMARY_DIR}",FORCE_RECREATE_PMDS="${FORCE_RECREATE_PMDS}",PIPELINE_SCRIPT="${SEGMENTATION_PY}" \
  --array=1-"${ARRAY_TASK_COUNT}" "$SEGMENTATION_SLURM")

echo "Manifest: $MANIFEST"
echo "Segmentation root: $SEGMENTATION_ROOT"
echo "Array task summaries: $SUMMARY_DIR"
echo "Array tasks: $ARRAY_TASK_COUNT"
echo "Force recreate PMDs: $FORCE_RECREATE_PMDS"
if [ -n "$cleanup_job_id" ]; then
  echo "Submitted cleanup job: $cleanup_job_id"
fi
echo "Submitted PMD detection array job: $array_job_id"
