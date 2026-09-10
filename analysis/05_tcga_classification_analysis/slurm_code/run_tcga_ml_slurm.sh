#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONDA_ROOT=/uufs/chpc.utah.edu/common/home/clementm-group1/conda/mambaforge
CONDA_PYTHON=$CONDA_ROOT/envs/jt_wgbs_analysis/bin/python
if [ ! -x "$CONDA_PYTHON" ]; then
  CONDA_PYTHON=$CONDA_ROOT/env/jt_wgbs_analysis/bin/python
fi
if [ ! -x "$CONDA_PYTHON" ]; then
  echo "Could not find the jt_wgbs_analysis Python interpreter." >&2
  exit 1
fi

exec "$CONDA_PYTHON" "$SCRIPT_DIR/run_tcga_ml.py" "$@"
