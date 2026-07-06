#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
OUT_ROOT="$REPO_ROOT/results/03_chromatin_analysis"
SEGMENTATION_RESULTS_PATH="$REPO_ROOT/results/01_region_calling_analysis"
CHROMATIN_DATA_DIR="$REPO_ROOT/data/chromatin_data"
SKIP_DEEPTOOLS=0
INCLUDE_HEATMAPS=0
DEEPTOOLS_TOOLS=""
DEPENDENT_JOB_ID=""

CONDA_ROOT=/uufs/chpc.utah.edu/common/home/clementm-group1/conda/mambaforge
CONDA_PYTHON=$CONDA_ROOT/envs/jt_wgbs_analysis/bin/python
if [ ! -x "$CONDA_PYTHON" ]; then
  CONDA_PYTHON=$CONDA_ROOT/env/jt_wgbs_analysis/bin/python
fi

usage() {
  cat <<EOF
Usage: ./run_slurm.sh [options]

Submit the chromatin analysis workflow.

Defaults:
  --out-root                  \$REPO_ROOT/results/03_chromatin_analysis
  --segmentation-results-path \$REPO_ROOT/results/01_region_calling_analysis
  --chromatin-data-dir        \$REPO_ROOT/data/chromatin_data

Options:
  --dependent_job_id <slurm_job_id>
  --skip-deeptools
  --include-heatmaps
  --deeptools-tools "methylseg methylseg_hm450k methylseekr dnmtools dnmtools_array dnmtools_pmr mmseekr methyl_lasso"

If --deeptools-tools is omitted, the default is to render profiles for all tools.
If --dependent_job_id is provided, region-output preflight checks are deferred to the runtime job.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --out-root)
      OUT_ROOT=$2
      shift 2
      ;;
    --segmentation-results-path)
      SEGMENTATION_RESULTS_PATH=$2
      shift 2
      ;;
    --chromatin-data-dir)
      CHROMATIN_DATA_DIR=$2
      shift 2
      ;;
    --dependent_job_id)
      DEPENDENT_JOB_ID=$2
      shift 2
      ;;
    --skip-deeptools)
      SKIP_DEEPTOOLS=1
      shift
      ;;
    --include-heatmaps)
      INCLUDE_HEATMAPS=1
      shift
      ;;
    --deeptools-tools)
      DEEPTOOLS_TOOLS=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ -n "${DEPENDENT_JOB_ID}" ] && [ -z "${DEPENDENT_JOB_ID// }" ]; then
  echo "The --dependent_job_id value must not be empty." >&2
  exit 1
fi

OUT_ROOT=$("$CONDA_PYTHON" - "$OUT_ROOT" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).expanduser().resolve())
PY
)

SEGMENTATION_RESULTS_PATH=$("$CONDA_PYTHON" - "$SEGMENTATION_RESULTS_PATH" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).expanduser().resolve())
PY
)

CHROMATIN_DATA_DIR=$("$CONDA_PYTHON" - "$CHROMATIN_DATA_DIR" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).expanduser().resolve())
PY
)

mkdir -p "$OUT_ROOT"
cd "$OUT_ROOT"

ts=$(date +%Y%m%d_%H%M%S)
for run_dir in cleaned_regions tables figures deeptools logs; do
  if [ -e "$OUT_ROOT/$run_dir" ]; then
    mv "$OUT_ROOT/$run_dir" "$OUT_ROOT/${run_dir}_backup_$ts"
  fi
done
mkdir -p "$OUT_ROOT/logs"

"$CONDA_PYTHON" - "$SEGMENTATION_RESULTS_PATH" "$CHROMATIN_DATA_DIR" "$DEPENDENT_JOB_ID" <<'PY'
import sys
from pathlib import Path

segmentation_root = Path(sys.argv[1])
chromatin_dir = Path(sys.argv[2])
dependent_job_id = sys.argv[3].strip()
samples = ["ESO26.wgbs", "TE5.wgbs"]
tool_paths = {
    "methylseg": ["methylseg", "{sample}", "out", "wgbs", "summary_files", "segments_cleaned_PMD.bed"],
    "methylseg_hm450k": ["methylseg", "{sample}", "out", "hm450k", "summary_files", "segments_cleaned_PMD.bed"],
    "methylseekr": ["methylseekr", "{sample}", "out", "methylseekr_PMDs.bed"],
    "dnmtools": ["dnmtools", "{sample}", "out", "dnmtools_PMDs.bed"],
    "dnmtools_array": ["dnmtools", "{sample}", "out", "arraymode.dnmtools_PMDs.bed"],
    "dnmtools_pmr": ["dnmtools", "{sample}", "out", "pmr.dnmtools_PMDs.bed"],
    "mmseekr": ["mmseekr", "{sample}", "out", "{sample}.multiModel.PMDs.bed"],
    "methyl_lasso": ["methyl_lasso", "{sample}", "out", "{sample}_pmd.tsv"],
}

missing = []
for sample in samples:
    sample_id = sample.replace(".wgbs", "")
    bw_path = chromatin_dir / f"{sample_id}.h3k36me2.bw"
    if not bw_path.exists():
        missing.append(f"{sample}: missing chromatin file {bw_path}")
    if dependent_job_id:
        continue
    for tool_name, parts in tool_paths.items():
        path = segmentation_root
        for part in parts:
            path = path / part.format(sample=sample)
        if not path.exists():
            missing.append(f"{sample}: missing {tool_name} output {path}")

if missing:
    raise SystemExit(
        "Chromatin analysis prerequisites are missing:\n" + "\n".join(missing)
    )
PY

dependency_args=()
if [ -n "$DEPENDENT_JOB_ID" ]; then
  dependency_args+=(--dependency="afterok:${DEPENDENT_JOB_ID}")
fi

job_id=$(sbatch --parsable \
  --chdir="$OUT_ROOT" \
  --job-name="chromatin_${ts}" \
  "${dependency_args[@]}" \
  --export=ALL,CHROMATIN_OUT_ROOT="$OUT_ROOT",SEGMENTATION_RESULTS_PATH="$SEGMENTATION_RESULTS_PATH",CHROMATIN_DATA_DIR="$CHROMATIN_DATA_DIR",CHROMATIN_SLURM_CODE_DIR="$SCRIPT_DIR",CHROMATIN_PIPELINE_SCRIPT="$REPO_ROOT/analysis/03_chromatin_analysis/run_chromatin.py",SKIP_DEEPTOOLS="$SKIP_DEEPTOOLS",INCLUDE_HEATMAPS="$INCLUDE_HEATMAPS",DEEPTOOLS_TOOLS="$DEEPTOOLS_TOOLS" \
  "$SCRIPT_DIR/methylation_chromatin.slurm")

echo "Output root: $OUT_ROOT"
echo "Segmentation results: $SEGMENTATION_RESULTS_PATH"
echo "Chromatin data dir: $CHROMATIN_DATA_DIR"
echo "Skip deepTools: $SKIP_DEEPTOOLS"
echo "Include heatmaps: $INCLUDE_HEATMAPS"
if [ -n "$DEPENDENT_JOB_ID" ]; then
  echo "Dependent job id: $DEPENDENT_JOB_ID"
fi
if [ -n "$DEEPTOOLS_TOOLS" ]; then
  echo "deepTools tools: $DEEPTOOLS_TOOLS"
fi
echo "Submitted chromatin job: $job_id"
