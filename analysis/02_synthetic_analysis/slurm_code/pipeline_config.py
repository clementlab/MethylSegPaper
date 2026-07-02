from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
while not (REPO_ROOT / "repo_paths.py").exists():
    if REPO_ROOT.parent == REPO_ROOT:
        raise ModuleNotFoundError("Could not locate repo_paths.py from pipeline_config.py")
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_paths import (
    METHYLATION_DATA_DIR,
    PROJECT_ROOT,
    REGION_CALLING_ANALYSIS_DIR,
    SYNTHETIC_RESULTS_DIR,
)


SLURM_CODE_DIR = Path(__file__).resolve().parent
SYNTHETIC_ROOT = SLURM_CODE_DIR.parent
DATA_DIR = METHYLATION_DATA_DIR
COMPARATOR_DIR = REGION_CALLING_ANALYSIS_DIR / "utils"

DEFAULT_OUT_ROOT = SYNTHETIC_RESULTS_DIR
DEFAULT_PARTITION = "notchpeak-guest"
DEFAULT_ACCOUNT = "owner-guest"

SOURCE_SAMPLE_MANIFEST = [
    {
        "source_file": DATA_DIR / "WGBS_colon-primary-normal_1_meth.bed.gz",
        "source_genome": "hg38",
        "source_kind": "wgbs_bed_gz",
        "include": True,
        "synthetic_sample_id": "synthetic_WGBS_colon_primary_normal_1_hg38",
    },
    {
        "source_file": DATA_DIR / "WGBS_colon-primary-normal_2_meth.bed.gz",
        "source_genome": "hg38",
        "source_kind": "wgbs_bed_gz",
        "include": True,
        "synthetic_sample_id": "synthetic_WGBS_colon_primary_normal_2_hg38",
    },
    {
        "source_file": DATA_DIR / "WGBS_colon-primary-normal_3_meth.bed.gz",
        "source_genome": "hg38",
        "source_kind": "wgbs_bed_gz",
        "include": True,
        "synthetic_sample_id": "synthetic_WGBS_colon_primary_normal_3_hg38",
    },
    {
        "source_file": DATA_DIR / "GSM5652176_Adipocytes-Z000000T7.beta",
        "source_genome": "hg19",
        "source_kind": "beta",
        "include": True,
        "synthetic_sample_id": "synthetic_GSM5652176_Adipocytes_Z000000T7_hg19",
    },
    {
        "source_file": DATA_DIR / "GSM5652250_Pancreas-Beta-Z00000452.beta",
        "source_genome": "hg19",
        "source_kind": "beta",
        "include": True,
        "synthetic_sample_id": "synthetic_GSM5652250_Pancreas_Beta_Z00000452_hg19",
    },
    {
        "source_file": DATA_DIR / "GSM5652251_Pancreas-Beta-Z00000455.beta",
        "source_genome": "hg19",
        "source_kind": "beta",
        "include": True,
        "synthetic_sample_id": "synthetic_GSM5652251_Pancreas_Beta_Z00000455_hg19",
    },
    {
        "source_file": DATA_DIR / "GSM5652348_Breast-Luminal-Epithelial-Z000000VJ.beta",
        "source_genome": "hg19",
        "source_kind": "beta",
        "include": True,
        "synthetic_sample_id": "synthetic_GSM5652348_Breast_Luminal_Epithelial_Z000000VJ_hg19",
    },
    {
        "source_file": DATA_DIR / "GSM5652349_Breast-Luminal-Epithelial-Z000000VN.beta",
        "source_genome": "hg19",
        "source_kind": "beta",
        "include": True,
        "synthetic_sample_id": "synthetic_GSM5652349_Breast_Luminal_Epithelial_Z000000VN_hg19",
    },
    {
        "source_file": DATA_DIR / "GSM5652350_Breast-Basal-Epithelial-Z000000V6.beta",
        "source_genome": "hg19",
        "source_kind": "beta",
        "include": True,
        "synthetic_sample_id": "synthetic_GSM5652350_Breast_Basal_Epithelial_Z000000V6_hg19",
    },
    {
        "source_file": DATA_DIR / "GSM5652351_Breast-Basal-Epithelial-Z000000VG.beta",
        "source_genome": "hg19",
        "source_kind": "beta",
        "include": True,
        "synthetic_sample_id": "synthetic_GSM5652351_Breast_Basal_Epithelial_Z000000VG_hg19",
    },
    {
        "source_file": DATA_DIR / "GSM5652352_Breast-Basal-Epithelial-Z000000VL.beta",
        "source_genome": "hg19",
        "source_kind": "beta",
        "include": True,
        "synthetic_sample_id": "synthetic_GSM5652352_Breast_Basal_Epithelial_Z000000VL_hg19",
    },
    {
        "source_file": DATA_DIR / "GSM5652353_Breast-Basal-Epithelial-Z0000043E.beta",
        "source_genome": "hg19",
        "source_kind": "beta",
        "include": True,
        "synthetic_sample_id": "synthetic_GSM5652353_Breast_Basal_Epithelial_Z0000043E_hg19",
    },
]


def included_records(sample_ids: list[str] | None = None) -> list[dict]:
    records = [record.copy() for record in SOURCE_SAMPLE_MANIFEST if record.get("include", True)]
    if sample_ids:
        selected = set(sample_ids)
        records = [record for record in records if record["synthetic_sample_id"] in selected]
        missing = sorted(selected - {record["synthetic_sample_id"] for record in records})
        if missing:
            raise ValueError(f"Unknown synthetic sample IDs: {missing}")
    return records


def get_record_by_array_index(array_index: int) -> dict:
    records = included_records()
    if array_index < 1 or array_index > len(records):
        raise ValueError(f"array_index must be between 1 and {len(records)}; got {array_index}.")
    return records[array_index - 1]


def get_record_by_sample_id(sample_id: str) -> dict:
    matches = [record for record in included_records() if record["synthetic_sample_id"] == sample_id]
    if not matches:
        raise ValueError(f"Unknown synthetic sample ID: {sample_id}")
    return matches[0]


def out_root(path_like: str | Path | None = None) -> Path:
    return Path(path_like or DEFAULT_OUT_ROOT).expanduser().resolve()


def paths(root: str | Path | None = None) -> dict[str, Path]:
    root = out_root(root)
    recovery_root = root / "synthetic_recovery"
    return {
        "root": root,
        "logs": root / "logs",
        "archives": root / "archives",
        "manifests": root / "manifests",
        "healthy_pmds": root / "healthy_pmds",
        "backgrounds": root / "healthy_background_references",
        "injected": root / "injected_pmd_samples",
        "recovery": recovery_root,
        "recovery_tool_results": recovery_root / "tool_results",
        "metrics": recovery_root / "metrics",
        "metrics_shards": recovery_root / "metrics" / "shards",
        "plots": recovery_root / "plots",
    }


def ensure_base_dirs(root: str | Path | None = None) -> dict[str, Path]:
    path_map = paths(root)
    for key in ["root", "logs", "archives", "manifests"]:
        path_map[key].mkdir(parents=True, exist_ok=True)
    return path_map


def source_manifest_df(sample_ids: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for record in included_records(sample_ids):
        row = record.copy()
        row["source_file"] = str(row["source_file"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("synthetic_sample_id").reset_index(drop=True)


def write_source_manifest(root: str | Path | None = None, sample_ids: list[str] | None = None) -> Path:
    path_map = ensure_base_dirs(root)
    manifest_path = path_map["manifests"] / "source_samples.tsv"
    source_manifest_df(sample_ids).to_csv(manifest_path, sep="\t", index=False)
    return manifest_path


def resolve_manifest_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (SYNTHETIC_ROOT / path).resolve()


def injected_manifest_path(root: str | Path | None = None) -> Path:
    return paths(root)["injected"] / "synthetic_sample_manifest.tsv"


def validate_injected_manifest(root: str | Path | None = None) -> tuple[bool, list[str]]:
    manifest_path = injected_manifest_path(root)
    if not manifest_path.exists():
        return False, [f"Missing injected manifest: {manifest_path}"]
    try:
        manifest_df = pd.read_csv(manifest_path, sep="\t")
    except Exception as exc:
        return False, [f"Could not read injected manifest: {exc}"]
    required_cols = {"synthetic_sample_id", "synthetic_file", "truth_bed", "source_genome"}
    missing_cols = sorted(required_cols - set(manifest_df.columns))
    if missing_cols:
        return False, [f"Injected manifest is missing columns: {missing_cols}"]
    if manifest_df.empty:
        return False, ["Injected manifest is empty."]
    messages = []
    for row in manifest_df.itertuples(index=False):
        for col in ("synthetic_file", "truth_bed"):
            file_path = resolve_manifest_path(getattr(row, col))
            if not file_path.exists():
                messages.append(f"{row.synthetic_sample_id}: missing {col} {file_path}")
    return len(messages) == 0, messages


def require_valid_injected_manifest(root: str | Path | None = None) -> Path:
    valid, messages = validate_injected_manifest(root)
    if not valid:
        raise FileNotFoundError("\n".join(messages))
    return injected_manifest_path(root)


def write_config_list(config_df: pd.DataFrame, config_list_path: str | Path) -> Path:
    config_list_path = Path(config_list_path)
    config_list_path.parent.mkdir(parents=True, exist_ok=True)
    config_paths = config_df["config_path"].astype(str).tolist()
    config_list_path.write_text("\n".join(config_paths) + ("\n" if config_paths else ""))
    return config_list_path


def write_sample_list(sample_ids: list[str], sample_list_path: str | Path) -> Path:
    sample_list_path = Path(sample_list_path)
    sample_list_path.parent.mkdir(parents=True, exist_ok=True)
    sample_list_path.write_text("\n".join(sample_ids) + ("\n" if sample_ids else ""))
    return sample_list_path
