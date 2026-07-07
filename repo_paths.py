from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_CODE_DIR = PROJECT_ROOT / "figures"
FIGURES_OUT_DIR = FIGURES_CODE_DIR / "out"
RESULTS_DIR = PROJECT_ROOT / "results"

METHYLATION_DATA_DIR = DATA_DIR / "methylation_data"
CHROMATIN_DATA_DIR = DATA_DIR / "chromatin_data"
REFERENCE_DATA_DIR = DATA_DIR / "reference_data"

REGION_CALLING_ANALYSIS_DIR = ANALYSIS_DIR / "01_region_calling_analysis"
SYNTHETIC_ANALYSIS_DIR = ANALYSIS_DIR / "02_synthetic_analysis"
CHROMATIN_ANALYSIS_DIR = ANALYSIS_DIR / "03_chromatin_analysis"
LAD_ANALYSIS_DIR = ANALYSIS_DIR / "04_lad_analysis"
TCGA_CLASSIFICATION_ANALYSIS_DIR = ANALYSIS_DIR / "05_tcga_classification_analysis"

REGION_CALLING_RESULTS_DIR = RESULTS_DIR / "01_region_calling_analysis"
SYNTHETIC_RESULTS_DIR = RESULTS_DIR / "02_synthetic_analysis"
CHROMATIN_RESULTS_DIR = RESULTS_DIR / "03_chromatin_analysis"
LAD_RESULTS_DIR = RESULTS_DIR / "04_lad_analysis"
TCGA_CLASSIFICATION_RESULTS_DIR = RESULTS_DIR / "05_tcga_classification_analysis"
FIGURE_RESULTS_DIR = FIGURES_OUT_DIR

ANALYSIS_RESULTS_DIRS = {
    "01_region_calling_analysis": REGION_CALLING_RESULTS_DIR,
    "02_synthetic_analysis": SYNTHETIC_RESULTS_DIR,
    "03_chromatin_analysis": CHROMATIN_RESULTS_DIR,
    "04_lad_analysis": LAD_RESULTS_DIR,
    "05_tcga_classification_analysis": TCGA_CLASSIFICATION_RESULTS_DIR,
}


def ensure_results_dirs() -> dict[str, Path]:
    for path in ANALYSIS_RESULTS_DIRS.values():
        path.mkdir(parents=True, exist_ok=True)
    return ANALYSIS_RESULTS_DIRS.copy()
