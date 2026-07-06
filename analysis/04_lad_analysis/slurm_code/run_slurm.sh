#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
OUT_ROOT="$REPO_ROOT/results/04_lad_analysis"
SEGMENTATION_RESULTS_PATH="$REPO_ROOT/results/01_region_calling_analysis"
RUN_DEEPTOOLS=0
INCLUDE_HEATMAPS=0
FORCE_REFERENCE_REBUILD=0
FORCE_DEEPTOOLS=0
PRIMARY_WINDOW_BP=""
PROFILE_BIN_BP=""
PROFILE_REGION_BODY_BP=""
SAMPLES=()

CONDA_ROOT=/uufs/chpc.utah.edu/common/home/clementm-group1/conda/mambaforge
CONDA_PYTHON=$CONDA_ROOT/envs/jt_wgbs_analysis/bin/python
if [ ! -x "$CONDA_PYTHON" ]; then
  CONDA_PYTHON=$CONDA_ROOT/env/jt_wgbs_analysis/bin/python
fi

usage() {
  cat <<EOF
Usage: ./run_slurm.sh [options]

Submit the LAD analysis workflow.

Defaults:
  --out-root                  \$REPO_ROOT/results/04_lad_analysis
  --segmentation-results-path \$REPO_ROOT/results/01_region_calling_analysis
  --skip-deeptools

Options:
  --samples sample1 sample2 ...
  --run-deeptools
  --include-heatmaps
  --force-reference-rebuild
  --force-deeptools
  --primary-window-bp N
  --profile-bin-bp N
  --profile-region-body-bp N
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
    --samples)
      shift
      while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
        SAMPLES+=("$1")
        shift
      done
      ;;
    --skip-deeptools)
      RUN_DEEPTOOLS=0
      shift
      ;;
    --run-deeptools)
      RUN_DEEPTOOLS=1
      shift
      ;;
    --include-heatmaps)
      INCLUDE_HEATMAPS=1
      shift
      ;;
    --force-reference-rebuild)
      FORCE_REFERENCE_REBUILD=1
      shift
      ;;
    --force-deeptools)
      FORCE_DEEPTOOLS=1
      shift
      ;;
    --primary-window-bp)
      PRIMARY_WINDOW_BP=$2
      shift 2
      ;;
    --profile-bin-bp)
      PROFILE_BIN_BP=$2
      shift 2
      ;;
    --profile-region-body-bp)
      PROFILE_REGION_BODY_BP=$2
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

mkdir -p "$OUT_ROOT"
cd "$OUT_ROOT"

ts=$(date +%Y%m%d_%H%M%S)
for run_dir in cleaned_regions tables figures deeptools reference_tracks logs; do
  if [ -e "$OUT_ROOT/$run_dir" ]; then
    mv "$OUT_ROOT/$run_dir" "$OUT_ROOT/${run_dir}_backup_$ts"
  fi
done
mkdir -p "$OUT_ROOT/logs"

if [ "${#SAMPLES[@]}" -gt 0 ]; then
  SAMPLE_ARGS_JSON=$("$CONDA_PYTHON" - "${SAMPLES[@]}" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1:]))
PY
)
else
  SAMPLE_ARGS_JSON='[]'
fi

"$CONDA_PYTHON" - "$SEGMENTATION_RESULTS_PATH" "$SAMPLE_ARGS_JSON" <<'PY'
import json
import sys
from pathlib import Path

segmentation_root = Path(sys.argv[1])
selected_samples = json.loads(sys.argv[2])
repo_root = Path("/uufs/chpc.utah.edu/common/home/u0914269/clement/projects/20260624_methylseg")
analysis_root = repo_root / "analysis" / "04_lad_analysis"
for import_path in (repo_root, analysis_root):
    import_str = str(import_path)
    if import_str not in sys.path:
        sys.path.insert(0, import_str)

from importlib.util import module_from_spec, spec_from_file_location

target_path = analysis_root / "01_run_lad.py"
spec = spec_from_file_location("methylseg_lad_runner", target_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"Could not load LAD runner from {target_path}")
module = module_from_spec(spec)
spec.loader.exec_module(module)

from lad_analysis_utils import build_tool_region_path

DEFAULT_HG19_TO_HG38_CHAIN = module.DEFAULT_HG19_TO_HG38_CHAIN
DEFAULT_LAD_INTERVAL_TRACK_PATH = module.DEFAULT_LAD_INTERVAL_TRACK_PATH
DEFAULT_LAMINB1_SIGNAL_TRACK_PATH = module.DEFAULT_LAMINB1_SIGNAL_TRACK_PATH
DEFAULT_LIFTOVER_SCRIPT_PATH = module.DEFAULT_LIFTOVER_SCRIPT_PATH
DEFAULT_SELECTED_SAMPLES = module.DEFAULT_SELECTED_SAMPLES
REQUIRED_TOOLS = module.REQUIRED_TOOLS
TOOL_CONFIG_BY_NAME = module.TOOL_CONFIG_BY_NAME

samples = selected_samples or list(DEFAULT_SELECTED_SAMPLES)
missing = []

for path in [
    DEFAULT_LAD_INTERVAL_TRACK_PATH,
    DEFAULT_LAMINB1_SIGNAL_TRACK_PATH,
    DEFAULT_LIFTOVER_SCRIPT_PATH,
    DEFAULT_HG19_TO_HG38_CHAIN,
]:
    if not Path(path).exists():
        missing.append(f"Missing LAD reference file {path}")

for sample in samples:
    for tool in REQUIRED_TOOLS:
        tool_config = TOOL_CONFIG_BY_NAME[tool]
        path = build_tool_region_path(segmentation_root, sample, tool_config)
        if not path.exists():
            missing.append(f"{sample}: missing {tool} output {path}")

if missing:
    raise SystemExit("LAD analysis prerequisites are missing:\n" + "\n".join(missing))
PY

LAD_SAMPLES=""
if [ "${#SAMPLES[@]}" -gt 0 ]; then
  LAD_SAMPLES="${SAMPLES[*]}"
fi

job_id=$(sbatch --parsable \
  --chdir="$OUT_ROOT" \
  --job-name="lad_${ts}" \
  --export=ALL,LAD_OUT_ROOT="$OUT_ROOT",SEGMENTATION_RESULTS_PATH="$SEGMENTATION_RESULTS_PATH",LAD_SLURM_CODE_DIR="$SCRIPT_DIR",LAD_PIPELINE_SCRIPT="$SCRIPT_DIR/01_run_lad.py",LAD_SAMPLES="$LAD_SAMPLES",RUN_DEEPTOOLS="$RUN_DEEPTOOLS",INCLUDE_HEATMAPS="$INCLUDE_HEATMAPS",FORCE_REFERENCE_REBUILD="$FORCE_REFERENCE_REBUILD",FORCE_DEEPTOOLS="$FORCE_DEEPTOOLS",PRIMARY_WINDOW_BP="$PRIMARY_WINDOW_BP",PROFILE_BIN_BP="$PROFILE_BIN_BP",PROFILE_REGION_BODY_BP="$PROFILE_REGION_BODY_BP" \
  "$SCRIPT_DIR/methylation_lad.slurm")

echo "Output root: $OUT_ROOT"
echo "Segmentation results: $SEGMENTATION_RESULTS_PATH"
if [ "${#SAMPLES[@]}" -gt 0 ]; then
  echo "Samples: ${SAMPLES[*]}"
else
  echo "Samples: default 01_run_lad.py sample set"
fi
echo "Run deepTools: $RUN_DEEPTOOLS"
echo "Include heatmaps: $INCLUDE_HEATMAPS"
echo "Force reference rebuild: $FORCE_REFERENCE_REBUILD"
echo "Force deepTools: $FORCE_DEEPTOOLS"
if [ -n "$PRIMARY_WINDOW_BP" ]; then
  echo "Primary window bp: $PRIMARY_WINDOW_BP"
fi
if [ -n "$PROFILE_BIN_BP" ]; then
  echo "Profile bin bp: $PROFILE_BIN_BP"
fi
if [ -n "$PROFILE_REGION_BODY_BP" ]; then
  echo "Profile region body bp: $PROFILE_REGION_BODY_BP"
fi
echo "Submitted LAD job: $job_id"
