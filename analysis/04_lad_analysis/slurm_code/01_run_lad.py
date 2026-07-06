from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TARGET_PATH = PROJECT_ROOT / "analysis" / "04_lad_analysis" / "01_run_lad.py"


def _load_target_module():
    spec = spec_from_file_location("methylseg_run_lad", TARGET_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load LAD runner from {TARGET_PATH}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    _load_target_module().main(sys.argv[1:])
