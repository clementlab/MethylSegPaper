from __future__ import annotations

import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TCGA_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(TCGA_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(TCGA_ANALYSIS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.tcga_segmentation_workflow import (  # noqa: E402
    load_meth_ref,
    load_tcga_sample_beta_dataframe,
    load_tcga_samples_info,
    run_segmentation_for_sample,
)
from repo_paths import REFERENCE_DATA_DIR, TCGA_CLASSIFICATION_RESULTS_DIR

AUTOSOMES = [f"chr{i}" for i in range(1, 23)]
COORD_COLS = ["CpG_chrm", "CpG_beg", "CpG_end"]
METRIC_COLS = [
    "mcc",
    "balanced_accuracy",
    "macro_f1",
    "specificity",
    "negative_precision",
]
BINARY_FEATURE_ORDER = ["PMD", "Random", "CGI", "Always cancer"]
MULTICLASS_FEATURE_ORDER = ["PMD", "Random", "CGI", "Majority class", "Random class"]
FEATURE_COLORS = {
    "PMD": "#4C78A8",
    "Random": "#A0A0A0",
    "CGI": "#59A14F",
    "Always cancer": "#E45756",
    "Majority class": "#E45756",
    "Random class": "#F28E2B",
}

DEFAULT_GENOME_FILE = REFERENCE_DATA_DIR / "hg38.genome"
DEFAULT_CGI_BED = REFERENCE_DATA_DIR / "cgi.bed"
DEFAULT_SEGMENTATION_ROOT = TCGA_CLASSIFICATION_RESULTS_DIR / "segmentation"
DEFAULT_OUT_DIR = TCGA_CLASSIFICATION_RESULTS_DIR / "ml_outputs"
DEFAULT_N_VALUES = (1, 3, 5, 10, 50, 100, 500, 1000)


@dataclass(frozen=True)
class PipelineConfig:
    out_dir: Path
    segmentation_root: Path
    case_sets: tuple[str, ...] = ("brca", "per_cancer", "pan_cancer", "multiclass")
    n_values: tuple[int, ...] = ()
    cv_splits: int = 5
    n_estimators: int = 500
    random_state: int = 42
    cgi_bed: Path = DEFAULT_CGI_BED
    genome_file: Path = DEFAULT_GENOME_FILE
    max_samples_per_class: int | None = None
    save_feature_matrices: bool = True
    allow_missing_feature_sets: bool = False
    call_missing_pmds: bool = True
    high_confidence_pmd_min_fraction: float = 0.50
    random_region_min_length_bp: int = 150_000
    random_region_max_length_bp: int = 20_000_000


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def write_table(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, sep="\t", index=index)
    return path


def concat_nonempty(frames: Iterable[pd.DataFrame | None]) -> pd.DataFrame:
    usable_frames = [df for df in frames if df is not None and not df.empty]
    if not usable_frames:
        return pd.DataFrame()
    return pd.concat(usable_frames, ignore_index=True)


def write_json(data: dict, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def expected_feature_order(task_type: str) -> list[str]:
    return BINARY_FEATURE_ORDER if task_type == "binary" else MULTICLASS_FEATURE_ORDER


def add_feature_order(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "feature_set" not in df.columns:
        return df
    out = df.copy()
    ranks = {}
    for task_type in out.get("task_type", pd.Series(["binary"])).dropna().unique():
        for idx, feature_set in enumerate(expected_feature_order(str(task_type))):
            ranks[(str(task_type), feature_set)] = idx
    out["_feature_order"] = out.apply(
        lambda row: ranks.get(
            (str(row.get("task_type", "binary")), row["feature_set"]), 999
        ),
        axis=1,
    )
    sort_cols = [
        col
        for col in [
            "case_set",
            "cohort_id",
            "n_regions_requested",
            "fold",
            "_feature_order",
            "feature_set",
        ]
        if col in out.columns
    ]
    out = (
        out.sort_values(sort_cols)
        .drop(columns=["_feature_order"])
        .reset_index(drop=True)
    )
    return out


def parse_n_values(value: str | Sequence[int] | None) -> tuple[int, ...]:
    if value is None or value == "":
        return DEFAULT_N_VALUES
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        parsed = [int(piece) for piece in pieces]
    else:
        parsed = [int(piece) for piece in value]
    parsed = sorted({n for n in parsed if n > 0})
    if not parsed:
        raise ValueError("At least one positive n value is required.")
    return tuple(parsed)


def logspace_n_values(start: int = 1, stop: int = 1000, num: int = 20) -> list[int]:
    values = np.rint(np.logspace(np.log10(start), np.log10(stop), num)).astype(int)
    return sorted(set(int(v) for v in values if v > 0))


def normalize_region_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["chr", "start", "end"])
    out = df.copy()
    rename = {"chrom": "chr", "CpG_chrm": "chr", "CpG_beg": "start", "CpG_end": "end"}
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    missing = {"chr", "start", "end"} - set(out.columns)
    if missing:
        raise ValueError(f"Region table is missing required columns: {sorted(missing)}")
    out = out.copy()
    out["chr"] = out["chr"].astype(str)
    out["start"] = pd.to_numeric(out["start"], errors="coerce")
    out["end"] = pd.to_numeric(out["end"], errors="coerce")
    out = out.dropna(subset=["chr", "start", "end"]).copy()
    out["start"] = out["start"].astype(np.int64)
    out["end"] = out["end"].astype(np.int64)
    out = out.loc[out["end"] > out["start"]].copy()
    out["length"] = out["end"] - out["start"]
    return out.sort_values(["chr", "start", "end"]).reset_index(drop=True)


def make_region_ids(regions_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    regions_df = normalize_region_df(regions_df)
    regions_df = regions_df.reset_index(drop=True).copy()
    regions_df["region_id"] = [
        f"{prefix}_{i:06d}_{row.chr}_{int(row.start)}_{int(row.end)}"
        for i, row in enumerate(regions_df.itertuples(index=False), start=1)
    ]
    return regions_df


def read_genome_sizes(genome_file: str | Path) -> dict[str, int]:
    genome_df = pd.read_csv(genome_file, sep="\t", header=None, usecols=[0, 1])
    genome_df.columns = ["chr", "size"]
    genome_df["chr"] = genome_df["chr"].astype(str)
    genome_df["size"] = (
        pd.to_numeric(genome_df["size"], errors="coerce").fillna(0).astype(int)
    )
    return dict(zip(genome_df["chr"], genome_df["size"]))


def _merge_regions_with_metadata(
    regions_df: pd.DataFrame,
    metadata_cols: Sequence[str],
) -> pd.DataFrame:
    regions_df = normalize_region_df(regions_df)
    if regions_df.empty:
        return pd.DataFrame(columns=["chr", "start", "end", *metadata_cols, "length"])
    merged_rows = []
    for chrom, chrom_df in regions_df.sort_values(["chr", "start", "end"]).groupby(
        "chr", sort=True
    ):
        current_start = None
        current_end = None
        metadata_values = {col: set() for col in metadata_cols}
        for row in chrom_df.itertuples(index=False):
            row_start = int(row.start)
            row_end = int(row.end)
            if current_start is None or row_start > current_end:
                if current_start is not None:
                    merged_rows.append(
                        {
                            "chr": chrom,
                            "start": current_start,
                            "end": current_end,
                            **{
                                col: ",".join(
                                    sorted(str(v) for v in values if pd.notna(v))
                                )
                                for col, values in metadata_values.items()
                            },
                        }
                    )
                current_start = row_start
                current_end = row_end
                metadata_values = {col: set() for col in metadata_cols}
            else:
                current_end = max(current_end, row_end)
            for col in metadata_cols:
                value = getattr(row, col)
                for piece in str(value).split(","):
                    if piece and piece != "nan":
                        metadata_values[col].add(piece)
        if current_start is not None:
            merged_rows.append(
                {
                    "chr": chrom,
                    "start": current_start,
                    "end": current_end,
                    **{
                        col: ",".join(sorted(str(v) for v in values if pd.notna(v)))
                        for col, values in metadata_values.items()
                    },
                }
            )
    return normalize_region_df(pd.DataFrame(merged_rows))


def _regions_with_any_overlap(
    regions_df: pd.DataFrame, query_df: pd.DataFrame
) -> pd.DataFrame:
    regions = normalize_region_df(regions_df)
    query = normalize_region_df(query_df)
    if regions.empty or query.empty:
        return regions.iloc[0:0].copy()
    supported_rows = []
    query_by_chrom = {
        chrom: chrom_df.sort_values("start").reset_index(drop=True)
        for chrom, chrom_df in query.groupby("chr", sort=False)
    }
    for row in regions.itertuples(index=False):
        chrom_query = query_by_chrom.get(str(row.chr))
        if chrom_query is None or chrom_query.empty:
            continue
        starts = chrom_query["start"].to_numpy()
        ends = chrom_query["end"].to_numpy()
        idx = np.searchsorted(starts, int(row.end), side="left")
        if idx > 0 and np.any(ends[:idx] > int(row.start)):
            supported_rows.append(row._asdict())
    return normalize_region_df(pd.DataFrame(supported_rows))


def eligible_tcga_samples(all_samples: pd.DataFrame | None = None) -> pd.DataFrame:
    samples = load_tcga_samples_info() if all_samples is None else all_samples.copy()
    out = samples.loc[
        samples["sample_id"].astype(str).str.startswith("TCGA-")
        & samples["project_id"].astype(str).str.startswith("TCGA-")
        & samples["methylation_file"].notna()
        & samples["sample_type"].isin(["Primary Tumor", "Solid Tissue Normal"])
    ].copy()
    out = out.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)
    out["sample_id"] = out["sample_id"].astype(str)
    out["project_id"] = out["project_id"].astype(str)
    out["sample_type"] = out["sample_type"].astype(str)
    return out


def summarize_tumor_projects(all_samples: pd.DataFrame, cv_splits: int) -> pd.DataFrame:
    eligible = eligible_tcga_samples(all_samples)
    summary = (
        eligible.groupby(["project_id", "sample_type"]).size().unstack(fill_value=0)
    )
    for col in ["Primary Tumor", "Solid Tissue Normal"]:
        if col not in summary.columns:
            summary[col] = 0
    summary = summary[["Primary Tumor", "Solid Tissue Normal"]].reset_index()
    summary = summary.rename(
        columns={"Primary Tumor": "n_tumors", "Solid Tissue Normal": "n_normals"}
    )
    summary["has_normal"] = summary["n_normals"] > 0
    summary["binary_cv_feasible"] = (
        summary["has_normal"]
        & (summary["n_tumors"] >= cv_splits)
        & (summary["n_normals"] >= cv_splits)
    )
    summary["multiclass_plus_normal_eligible"] = summary["has_normal"] & (
        summary["n_tumors"] >= cv_splits
    )
    return summary.sort_values(
        ["binary_cv_feasible", "n_normals", "n_tumors", "project_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def build_labels(samples_df: pd.DataFrame, task_type: str) -> pd.Series:
    samples = samples_df.set_index("sample_id", drop=False)
    if task_type == "binary":
        labels = np.where(
            samples["sample_type"].astype(str).eq("Solid Tissue Normal"),
            "Solid Tissue Normal",
            "Cancer",
        )
    elif task_type == "multiclass":
        labels = np.where(
            samples["sample_type"].astype(str).eq("Solid Tissue Normal"),
            "Normal",
            samples["project_id"].astype(str),
        )
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    return pd.Series(labels, index=samples.index, name="label")


def apply_sample_cap(
    samples_df: pd.DataFrame,
    task_type: str,
    max_samples_per_class: int | None,
    random_state: int,
) -> pd.DataFrame:
    if max_samples_per_class is None:
        return samples_df.copy().reset_index(drop=True)
    labels = build_labels(samples_df, task_type=task_type)
    capped_parts = []
    rng = np.random.default_rng(random_state)
    for label in sorted(labels.unique()):
        label_ids = labels.loc[labels.eq(label)].index.to_numpy()
        if len(label_ids) > max_samples_per_class:
            label_ids = rng.choice(label_ids, size=max_samples_per_class, replace=False)
        capped_parts.append(samples_df.loc[samples_df["sample_id"].isin(label_ids)])
    return pd.concat(capped_parts, ignore_index=True).drop_duplicates("sample_id")


def build_case_cohorts(
    all_samples: pd.DataFrame,
    case_sets: Sequence[str],
    cv_splits: int,
    random_state: int,
    max_samples_per_class: int | None = None,
) -> tuple[list[dict], pd.DataFrame]:
    eligible = eligible_tcga_samples(all_samples)
    project_summary = summarize_tumor_projects(eligible, cv_splits=cv_splits)
    cohorts: list[dict] = []

    if "brca" in case_sets:
        brca = eligible.loc[eligible["project_id"].eq("TCGA-BRCA")].copy()
        brca = apply_sample_cap(brca, "binary", max_samples_per_class, random_state)
        cohorts.append(
            {
                "case_set": "brca",
                "cohort_id": "TCGA-BRCA",
                "cohort_label": "TCGA-BRCA tumor vs normal",
                "task_type": "binary",
                "pmd_strategy": "cancer_specific",
                "samples_df": brca,
            }
        )

    if "per_cancer" in case_sets:
        project_ids = project_summary.loc[
            project_summary["binary_cv_feasible"], "project_id"
        ].astype(str)
        for project_id in project_ids:
            cohort = eligible.loc[eligible["project_id"].eq(project_id)].copy()
            cohort = apply_sample_cap(
                cohort, "binary", max_samples_per_class, random_state
            )
            cohorts.append(
                {
                    "case_set": "per_cancer",
                    "cohort_id": project_id,
                    "cohort_label": f"{project_id} tumor vs normal",
                    "task_type": "binary",
                    "pmd_strategy": "cancer_specific",
                    "samples_df": cohort,
                }
            )

    if "pan_cancer" in case_sets:
        pan = eligible.loc[
            eligible["project_id"].isin(
                project_summary.loc[project_summary["has_normal"], "project_id"]
            )
        ].copy()
        pan = apply_sample_cap(pan, "binary", max_samples_per_class, random_state)
        cohorts.append(
            {
                "case_set": "pan_cancer",
                "cohort_id": "PAN_CANCER",
                "cohort_label": "All cancers vs normal",
                "task_type": "binary",
                "pmd_strategy": "shared",
                "samples_df": pan,
            }
        )

    if "multiclass" in case_sets:
        project_ids = project_summary.loc[
            project_summary["multiclass_plus_normal_eligible"], "project_id"
        ].astype(str)
        multi = eligible.loc[eligible["project_id"].isin(project_ids)].copy()
        multi = apply_sample_cap(
            multi, "multiclass", max_samples_per_class, random_state
        )
        cohorts.append(
            {
                "case_set": "multiclass",
                "cohort_id": "TUMOR_TYPES_PLUS_NORMAL",
                "cohort_label": "Cancer multiclassification",
                "task_type": "multiclass",
                "pmd_strategy": "cancer_specific",
                "samples_df": multi,
            }
        )

    validated = []
    warnings = []
    for cohort in cohorts:
        labels = build_labels(cohort["samples_df"], cohort["task_type"])
        label_counts = labels.value_counts()
        too_small = label_counts[label_counts < cv_splits]
        if too_small.empty and label_counts.shape[0] >= 2:
            validated.append(cohort)
        else:
            warnings.append(
                {
                    "case_set": cohort["case_set"],
                    "cohort_id": cohort["cohort_id"],
                    "warning": (
                        f"Skipping cohort because labels cannot support {cv_splits}-fold "
                        f"CV: {label_counts.to_dict()}"
                    ),
                }
            )
    warning_df = pd.DataFrame(warnings)
    return validated, warning_df


def prepare_probe_df(meth_ref_df: pd.DataFrame | None = None) -> pd.DataFrame:
    meth_ref_df = load_meth_ref() if meth_ref_df is None else meth_ref_df.copy()
    probe_df = meth_ref_df.loc[:, ["CpG_chrm", "CpG_beg", "CpG_end", "key"]].copy()
    probe_df = probe_df.dropna(subset=["CpG_chrm", "CpG_beg", "CpG_end", "key"])
    probe_df["CpG_chrm"] = probe_df["CpG_chrm"].astype(str)
    if not probe_df["CpG_chrm"].str.startswith("chr").all():
        probe_df["CpG_chrm"] = "chr" + probe_df["CpG_chrm"].str.replace(
            "^chr", "", regex=True
        )
    probe_df["CpG_beg"] = pd.to_numeric(probe_df["CpG_beg"], errors="coerce")
    probe_df["CpG_end"] = pd.to_numeric(probe_df["CpG_end"], errors="coerce")
    probe_df = probe_df.dropna(subset=["CpG_beg", "CpG_end"]).copy()
    probe_df["CpG_beg"] = probe_df["CpG_beg"].astype(np.int64)
    probe_df["CpG_end"] = probe_df["CpG_end"].astype(np.int64)
    probe_df = probe_df.loc[
        probe_df["CpG_chrm"].isin(AUTOSOMES)
        & (probe_df["CpG_end"] > probe_df["CpG_beg"])
    ].copy()
    return probe_df.sort_values(["CpG_chrm", "CpG_beg", "CpG_end", "key"]).reset_index(
        drop=True
    )


def build_wide_methylation_table(
    samples_df: pd.DataFrame,
    meth_ref_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    meth_ref_df = load_meth_ref() if meth_ref_df is None else meth_ref_df.copy()
    coord_df = None
    sample_series = {}
    warnings = []
    for row in samples_df.itertuples(index=False):
        sample_id = str(row.sample_id)
        try:
            sample_df = load_tcga_sample_beta_dataframe(
                sample_id,
                row.methylation_file,
                drop_nas=False,
                meth_ref_df=meth_ref_df,
            )
        except Exception as exc:
            warnings.append(
                {
                    "sample_id": sample_id,
                    "warning": f"Failed to load methylation: {exc}",
                }
            )
            continue
        sample_df = sample_df.drop_duplicates(subset=["probe"]).set_index("probe")
        if coord_df is None:
            coord_df = sample_df.loc[:, COORD_COLS].copy()
        sample_series[sample_id] = sample_df["beta"].rename(sample_id)
    if coord_df is None or not sample_series:
        raise ValueError("No methylation samples could be loaded.")
    sample_beta_df = pd.concat(sample_series.values(), axis=1)
    meth_data_df = pd.concat([coord_df, sample_beta_df], axis=1).reset_index(drop=True)
    return meth_data_df, pd.DataFrame(warnings)


def read_bed(path: str | Path, names: Sequence[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=["chr", "start", "end"])
    try:
        df = pd.read_csv(path, sep="\t", header=None, comment="#")
    except EmptyDataError:
        return pd.DataFrame(columns=["chr", "start", "end"])
    if names is None:
        names = ["chr", "start", "end"] + [
            f"field_{i}" for i in range(4, df.shape[1] + 1)
        ]
    df = df.iloc[:, : len(names)].copy()
    df.columns = list(names[: df.shape[1]])
    return normalize_region_df(df)


def resolve_pmd_summary_path(segmentation_root: str | Path, sample_id: str) -> Path:
    summary_dir = (
        Path(segmentation_root)
        / "methylseg"
        / str(sample_id)
        / "out"
        / "hm450k"
        / "summary_files"
    )
    candidates = [
        summary_dir / "segments_cleaned_PMD.bed",
        summary_dir / "segments_PMD.bed",
    ]
    for path in candidates:
        if path.exists():
            return path
    pmr_candidates = sorted(summary_dir.glob("*PMR*.bed"))
    if pmr_candidates:
        raise ValueError(
            f"PMD file is missing for {sample_id}; refusing to use PMR file "
            f"{pmr_candidates[0]}"
        )
    raise FileNotFoundError(f"No PMD BED file found for {sample_id} in {summary_dir}")


def load_pmd_regions_for_samples(
    sample_ids: Iterable[str], segmentation_root: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    warnings = []
    for sample_id in sample_ids:
        sample_id = str(sample_id)
        try:
            path = resolve_pmd_summary_path(segmentation_root, sample_id)
            pmd_df = read_bed(path)
        except Exception as exc:
            warnings.append(
                {"sample_id": sample_id, "warning": f"Could not load PMDs: {exc}"}
            )
            continue
        if pmd_df.empty:
            continue
        pmd_df["sample_id"] = sample_id
        frames.append(pmd_df)
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out = out.loc[out["chr"].isin(AUTOSOMES)].reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=["chr", "start", "end", "length", "sample_id"])
    return out, pd.DataFrame(warnings)


def validate_pmd_preflight(
    cohorts: Sequence[dict],
    segmentation_root: str | Path,
) -> pd.DataFrame:
    rows = []
    for cohort in cohorts:
        tumor_samples = cohort["samples_df"].loc[
            cohort["samples_df"]["sample_type"].eq("Primary Tumor")
        ]
        for sample_id in tumor_samples["sample_id"].astype(str):
            row = {
                "case_set": cohort["case_set"],
                "cohort_id": cohort["cohort_id"],
                "cohort_label": cohort["cohort_label"],
                "task_type": cohort["task_type"],
                "feature_set": "PMD",
                "fold": np.nan,
                "n_regions_requested": np.nan,
                "sample_id": sample_id,
                "status": "available",
                "n_regions_selected": np.nan,
                "failure_reason": "",
            }
            try:
                row["pmd_path"] = str(
                    resolve_pmd_summary_path(segmentation_root, sample_id)
                )
            except Exception as exc:
                row["status"] = "missing"
                row["failure_reason"] = str(exc)
                row["pmd_path"] = ""
            rows.append(row)

    status_df = pd.DataFrame(rows)
    return status_df


def _pmd_candidate_samples(cohorts: Sequence[dict]) -> pd.DataFrame:
    frames = []
    for cohort in cohorts:
        samples = cohort["samples_df"].copy()
        samples = samples.loc[samples["sample_type"].eq("Primary Tumor")].copy()
        if samples.empty:
            continue
        samples["case_set"] = cohort["case_set"]
        samples["cohort_id"] = cohort["cohort_id"]
        samples["cohort_label"] = cohort["cohort_label"]
        samples["task_type"] = cohort["task_type"]
        frames.append(samples)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("sample_id")


def ensure_missing_pmds(
    cohorts: Sequence[dict],
    segmentation_root: str | Path,
    *,
    call_missing_pmds: bool,
    genome: str = "hg38",
) -> pd.DataFrame:
    candidates = _pmd_candidate_samples(cohorts)
    rows = []
    for row in candidates.itertuples(index=False):
        sample_id = str(row.sample_id)
        status_row = {
            "case_set": row.case_set,
            "cohort_id": row.cohort_id,
            "cohort_label": row.cohort_label,
            "task_type": row.task_type,
            "sample_id": sample_id,
            "methylation_file": str(row.methylation_file),
            "status": "existing",
            "pmd_path": "",
            "message": "",
        }
        try:
            status_row["pmd_path"] = str(
                resolve_pmd_summary_path(segmentation_root, sample_id)
            )
            rows.append(status_row)
            continue
        except Exception as missing_exc:
            if not call_missing_pmds:
                status_row["status"] = "missing"
                status_row["message"] = str(missing_exc)
                rows.append(status_row)
                continue

        try:
            result = run_segmentation_for_sample(
                sample_id=sample_id,
                meth_file=row.methylation_file,
                genome=genome,
                segmentation_root=segmentation_root,
                force_recreate=False,
                print_logs=True,
                run_on_dnmtools_array=False,
                clean_individual_chr_outputs=True,
            )
            pmd_path = resolve_pmd_summary_path(segmentation_root, sample_id)
            status_row["status"] = "generated"
            status_row["pmd_path"] = str(pmd_path)
            status_row["message"] = (
                f"Generated PMD file via MethylSeg: {result.methylseg_hm450k_bed}"
            )
        except Exception as exc:
            status_row["status"] = "generation_failed"
            status_row["message"] = str(exc)
        rows.append(status_row)
    return pd.DataFrame(rows)


def merge_pmds_with_support(pmd_df: pd.DataFrame) -> pd.DataFrame:
    pmd_df = normalize_region_df(pmd_df)
    if pmd_df.empty:
        return pd.DataFrame(
            columns=["chr", "start", "end", "sample_ids", "n_samples", "length"]
        )
    if "sample_id" not in pmd_df.columns:
        raise ValueError("PMD table must include sample_id for support ranking.")
    pmd_df = pmd_df.rename(columns={"sample_id": "sample_ids"})
    merged = _merge_regions_with_metadata(pmd_df, ["sample_ids"])
    merged["n_samples"] = merged["sample_ids"].str.split(",").apply(len)
    return merged


def add_support_fraction(
    pmd_support_df: pd.DataFrame, n_training_tumor_samples: int
) -> pd.DataFrame:
    if n_training_tumor_samples <= 0:
        raise ValueError("n_training_tumor_samples must be positive.")
    out = pmd_support_df.copy()
    if out.empty:
        out["support_fraction"] = pd.Series(dtype=float)
        return out
    out["support_fraction"] = out["n_samples"].astype(float) / float(
        n_training_tumor_samples
    )
    return out


def prepare_fold_pmd_tables(
    train_tumor_samples_df: pd.DataFrame,
    segmentation_root: str | Path,
    genome_file: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_pmds, warnings = load_pmd_regions_for_samples(
        train_tumor_samples_df["sample_id"].astype(str), segmentation_root
    )
    if all_pmds.empty:
        empty = pd.DataFrame(
            columns=[
                "chr",
                "start",
                "end",
                "sample_ids",
                "n_samples",
                "length",
                "support_fraction",
            ]
        )
        return all_pmds, empty, warnings

    all_merged = merge_pmds_with_support(all_pmds)
    all_merged = add_support_fraction(all_merged, len(train_tumor_samples_df))
    return all_pmds, all_merged, warnings


def select_pmd_regions(
    train_tumor_samples_df: pd.DataFrame,
    segmentation_root: str | Path,
    genome_file: str | Path,
    n_regions: int,
    strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_pmds, all_merged, warnings = prepare_fold_pmd_tables(
        train_tumor_samples_df, segmentation_root, genome_file
    )
    selected, all_merged, selection_warnings = select_pmd_regions_from_fold_tables(
        train_tumor_samples_df,
        all_pmds,
        all_merged,
        genome_file,
        n_regions,
        strategy,
    )
    return selected, all_merged, concat_nonempty([warnings, selection_warnings])


def select_pmd_regions_from_fold_tables(
    train_tumor_samples_df: pd.DataFrame,
    all_pmds: pd.DataFrame,
    all_merged: pd.DataFrame,
    genome_file: str | Path,
    n_regions: int,
    strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if all_pmds.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_merged = all_merged.loc[
        (all_merged["length"] >= 1000) & (all_merged["length"] <= 100000000)
    ].copy()
    if all_merged.empty:
        return pd.DataFrame(), all_merged, pd.DataFrame()

    if strategy == "shared":
        selected = all_merged.sort_values(
            ["n_samples", "length"], ascending=[False, False]
        ).head(n_regions)
    elif strategy == "cancer_specific":
        selected_parts = []
        for project_id, project_samples in train_tumor_samples_df.groupby(
            "project_id", sort=True
        ):
            project_pmds = all_pmds.loc[
                all_pmds["sample_id"].isin(project_samples["sample_id"].astype(str))
            ].copy()
            merged = merge_pmds_with_support(project_pmds)
            merged = merged.loc[
                (merged["length"] >= 1000) & (merged["length"] <= 100000000)
            ].copy()
            merged = merged.sort_values(
                ["n_samples", "length"], ascending=[False, False]
            ).head(n_regions)
            if not merged.empty:
                merged["source_project_id"] = str(project_id)
                selected_parts.append(merged)
        selected = (
            pd.concat(selected_parts, ignore_index=True)
            if selected_parts
            else pd.DataFrame(columns=all_merged.columns)
        )
        if not selected.empty:
            selected = _merge_regions_with_metadata(
                selected.loc[
                    :, ["chr", "start", "end", "source_project_id", "sample_ids"]
                ],
                ["source_project_id", "sample_ids"],
            )
            selected["n_samples"] = (
                selected["sample_ids"]
                .astype(str)
                .str.split(",")
                .apply(lambda values: len(set(values)))
            )
    else:
        raise ValueError(f"Unknown PMD strategy: {strategy}")

    selected = make_region_ids(selected.head(max(n_regions, len(selected))), "pmd")
    selected["feature_set"] = "PMD"
    return selected, all_merged, pd.DataFrame()


def probe_supported_regions(
    regions_df: pd.DataFrame, probe_df: pd.DataFrame
) -> pd.DataFrame:
    regions = normalize_region_df(regions_df)
    if regions.empty:
        return regions
    probe = probe_df.rename(
        columns={"CpG_chrm": "chr", "CpG_beg": "start", "CpG_end": "end"}
    ).loc[:, ["chr", "start", "end"]]
    return _regions_with_any_overlap(regions, probe)


def build_non_pmd_regions(
    pmd_union_df: pd.DataFrame, probe_df: pd.DataFrame, genome_file: str | Path
) -> pd.DataFrame:
    pmd_union_df = _merge_regions_with_metadata(normalize_region_df(pmd_union_df), [])
    genome_sizes = read_genome_sizes(genome_file)
    complement_rows = []
    for chrom in AUTOSOMES:
        chrom_size = genome_sizes.get(chrom)
        if not chrom_size:
            continue
        chrom_pmd = pmd_union_df.loc[pmd_union_df["chr"].eq(chrom)].sort_values("start")
        cursor = 0
        for row in chrom_pmd.itertuples(index=False):
            if int(row.start) > cursor:
                complement_rows.append(
                    {"chr": chrom, "start": cursor, "end": int(row.start)}
                )
            cursor = max(cursor, int(row.end))
        if cursor < chrom_size:
            complement_rows.append({"chr": chrom, "start": cursor, "end": chrom_size})
    complement = normalize_region_df(pd.DataFrame(complement_rows))
    probe = probe_df.rename(
        columns={"CpG_chrm": "chr", "CpG_beg": "start", "CpG_end": "end"}
    ).loc[:, ["chr", "start", "end"]]
    non_pmd = _regions_with_any_overlap(complement, probe)
    return non_pmd.loc[non_pmd["chr"].isin(AUTOSOMES)].reset_index(drop=True)


def build_candidate_control_space(
    pmd_support_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    genome_file: str | Path,
    high_confidence_pmd_min_fraction: float,
) -> pd.DataFrame:
    recurrent_pmds = select_high_confidence_pmds(
        pmd_support_df, high_confidence_pmd_min_fraction
    )
    return build_non_pmd_regions(recurrent_pmds, probe_df, genome_file)


def select_high_confidence_pmds(
    pmd_support_df: pd.DataFrame,
    high_confidence_pmd_min_fraction: float,
) -> pd.DataFrame:
    if pmd_support_df.empty:
        return pd.DataFrame(columns=["chr", "start", "end", "length"])
    if "support_fraction" not in pmd_support_df.columns:
        raise ValueError("PMD support table must include support_fraction.")
    if (
        high_confidence_pmd_min_fraction < 0
        or high_confidence_pmd_min_fraction > 1
    ):
        raise ValueError("high_confidence_pmd_min_fraction must be between 0 and 1.")
    return pmd_support_df.loc[
        pmd_support_df["support_fraction"] >= float(high_confidence_pmd_min_fraction)
    ].copy()


def region_has_cpgs(
    meth_data_df: pd.DataFrame, chrom: str, start: int, end: int
) -> bool:
    subset = meth_data_df.loc[
        meth_data_df["CpG_chrm"].astype(str).eq(str(chrom))
        & (meth_data_df["CpG_beg"] < int(end))
        & (meth_data_df["CpG_end"] > int(start))
    ]
    if subset.empty:
        return False
    beta_cols = [col for col in subset.columns if col not in COORD_COLS]
    return not subset.loc[:, beta_cols].isna().all(axis=0).any()


def subtract_interval_from_pool(
    pool_df: pd.DataFrame, chrom: str, used_start: int, used_end: int
) -> pd.DataFrame:
    pool = normalize_region_df(pool_df)
    kept = []
    for row in pool.itertuples(index=False):
        if (
            str(row.chr) != str(chrom)
            or int(row.end) <= used_start
            or int(row.start) >= used_end
        ):
            kept.append({"chr": row.chr, "start": int(row.start), "end": int(row.end)})
            continue
        if int(row.start) < used_start:
            kept.append(
                {"chr": row.chr, "start": int(row.start), "end": int(used_start)}
            )
        if int(row.end) > used_end:
            kept.append({"chr": row.chr, "start": int(used_end), "end": int(row.end)})
    return normalize_region_df(pd.DataFrame(kept))


def sample_random_non_pmd_regions(
    selected_pmd_df: pd.DataFrame,
    non_pmd_df: pd.DataFrame,
    seed: int,
    random_region_min_length_bp: int,
    random_region_max_length_bp: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    pool = normalize_region_df(non_pmd_df)
    selected = normalize_region_df(selected_pmd_df)
    if selected.empty:
        return make_region_ids(pd.DataFrame(), "random"), pd.DataFrame()
    min_len = int(random_region_min_length_bp)
    max_len = int(random_region_max_length_bp)
    if min_len <= 0 or max_len <= 0:
        raise ValueError("Random region min/max lengths must be positive integers.")
    if min_len > max_len:
        raise ValueError(
            "random_region_min_length_bp cannot exceed random_region_max_length_bp."
        )
    random_rows = []
    skipped = []
    for pmd in selected.itertuples(index=False):
        candidates = pool.loc[
            pool["chr"].astype(str).eq(str(pmd.chr))
            & (pool["length"] >= min_len)
        ].copy()
        chosen = None
        if not candidates.empty:
            candidate_rows = []
            weights = []
            for row in candidates.itertuples(index=False):
                upper_len = min(max_len, int(row.length))
                if upper_len < min_len:
                    continue
                feasible_positions = 0
                for length in range(min_len, upper_len + 1):
                    feasible_positions += int(row.length) - length + 1
                if feasible_positions <= 0:
                    continue
                candidate_rows.append(row)
                weights.append(feasible_positions)
            if candidate_rows:
                chosen_row = rng.choices(candidate_rows, weights=weights, k=1)[0]
                upper_len = min(max_len, int(chosen_row.length))
                chosen_len = rng.randint(min_len, upper_len)
                if chosen_len == int(chosen_row.length):
                    chosen_start = int(chosen_row.start)
                    chosen_end = int(chosen_row.end)
                    match_type = "chromosome_matched_interval_at_sampled_length"
                elif rng.random() < 0.5:
                    chosen_start = int(chosen_row.start)
                    chosen_end = chosen_start + chosen_len
                    match_type = "chromosome_matched_left_edge_subinterval"
                else:
                    chosen_end = int(chosen_row.end)
                    chosen_start = chosen_end - chosen_len
                    match_type = "chromosome_matched_right_edge_subinterval"
                chosen = {
                    "chr": chosen_row.chr,
                    "start": chosen_start,
                    "end": chosen_end,
                    "requested_length": int(pmd.end - pmd.start),
                    "sampled_length": chosen_len,
                    "match_type": match_type,
                    "source_region_id": getattr(pmd, "region_id", ""),
                }
        if chosen is None:
            skipped.append(
                {
                    "source_region_id": getattr(pmd, "region_id", ""),
                    "requested_length": int(pmd.end - pmd.start),
                    "reason": (
                        "no same-chromosome control interval with enough remaining "
                        f"space for configured random region bounds {min_len}-{max_len} bp"
                    ),
                }
            )
        else:
            random_rows.append(chosen)
            pool = subtract_interval_from_pool(
                pool,
                chosen["chr"],
                int(chosen["start"]),
                int(chosen["end"]),
            )
    random_df = make_region_ids(pd.DataFrame(random_rows), "random")
    if not random_df.empty:
        random_df["feature_set"] = "Random"
    return random_df, pd.DataFrame(skipped)


def load_cgi_regions(cgi_bed: str | Path, probe_df: pd.DataFrame) -> pd.DataFrame:
    cgi = read_bed(cgi_bed)
    cgi = cgi.loc[cgi["chr"].isin(AUTOSOMES)].reset_index(drop=True)
    cgi = probe_supported_regions(cgi, probe_df)
    cgi = make_region_ids(cgi, "cgi")
    cgi["feature_set"] = "CGI"
    return cgi


def sample_cgi_regions(
    cgi_df: pd.DataFrame, n_regions: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cgi_df.empty:
        return cgi_df.copy(), pd.DataFrame(
            [{"warning": "No CGI regions with probe support were available."}]
        )
    n_take = min(int(n_regions), len(cgi_df))
    selected = cgi_df.sample(n=n_take, random_state=seed).reset_index(drop=True)
    warnings = []
    if n_take < n_regions:
        warnings.append(
            {
                "warning": f"Only {n_take} CGI regions were available for requested n={n_regions}."
            }
        )
    return selected, pd.DataFrame(warnings)


def extract_feature_matrix(
    regions_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    meth_data_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    regions = (
        make_region_ids(regions_df, "region")
        if "region_id" not in regions_df.columns
        else regions_df.copy()
    )
    sample_ids = samples_df["sample_id"].astype(str).tolist()
    features = {}
    dropped_rows = []
    for region in regions.itertuples(index=False):
        subset = meth_data_df.loc[
            meth_data_df["CpG_chrm"].astype(str).eq(str(region.chr))
            & (meth_data_df["CpG_beg"] < int(region.end))
            & (meth_data_df["CpG_end"] > int(region.start))
        ]
        feature_name = str(region.region_id)
        if subset.empty:
            dropped_rows.append(
                {"region_id": feature_name, "reason": "no_overlapping_cpgs"}
            )
            continue
        values = subset.loc[:, sample_ids].mean(axis=0, skipna=True)
        if values.isna().all():
            dropped_rows.append(
                {"region_id": feature_name, "reason": "all_missing_beta"}
            )
            continue
        features[feature_name] = values
    X = pd.DataFrame(features, index=sample_ids)
    X.index.name = "sample_id"
    return X, pd.DataFrame(dropped_rows)


def prepare_train_test_data(
    X: pd.DataFrame,
    labels: pd.Series,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    X_train = X.loc[list(train_ids)].copy()
    X_test = X.loc[list(test_ids)].copy()
    y_train = labels.loc[list(train_ids)]
    y_test = labels.loc[list(test_ids)]
    dropped = []
    all_missing_cols = X_train.columns[X_train.isna().all(axis=0)].tolist()
    if all_missing_cols:
        dropped = [
            {"region_id": col, "reason": "all_missing_training"}
            for col in all_missing_cols
        ]
        X_train = X_train.drop(columns=all_missing_cols)
        X_test = X_test.drop(columns=all_missing_cols)
    if X_train.shape[1] == 0:
        raise ValueError(
            "No usable feature columns remain after training-set filtering."
        )
    imputer = SimpleImputer(strategy="mean")
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test), index=X_test.index, columns=X_test.columns
    )
    return X_train, X_test, y_train, y_test, pd.DataFrame(dropped)


def prediction_tables_for_model(
    X: pd.DataFrame,
    labels: pd.Series,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    metadata: dict,
    n_estimators: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train, X_test, y_train, y_test, dropped = prepare_train_test_data(
        X, labels, train_ids, test_ids
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    y_pred = pd.Series(clf.predict(X_test), index=y_test.index)
    predictions = []
    for sample_id in y_test.index:
        row = {
            **metadata,
            "sample_id": sample_id,
            "true_label": str(y_test.loc[sample_id]),
            "predicted_label": str(y_pred.loc[sample_id]),
            "n_features_used": int(X_train.shape[1]),
        }
        predictions.append(row)
    pred_df = pd.DataFrame(predictions)
    score_df = pd.DataFrame(
        clf.predict_proba(X_test),
        index=X_test.index,
        columns=[str(c) for c in clf.classes_],
    )
    score_records = []
    for sample_id, score_row in score_df.iterrows():
        for class_label, score in score_row.items():
            score_records.append(
                {
                    **metadata,
                    "sample_id": sample_id,
                    "class_label": class_label,
                    "score": float(score),
                }
            )
    return pred_df, pd.DataFrame(score_records), dropped


def baseline_prediction_tables(
    labels: pd.Series,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    task_type: str,
    metadata: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_train = labels.loc[list(train_ids)]
    y_test = labels.loc[list(test_ids)]
    if task_type == "binary":
        predicted = "Cancer"
        classes = ["Solid Tissue Normal", "Cancer"]
    else:
        predicted = str(y_train.astype(str).value_counts().idxmax())
        classes = sorted(labels.astype(str).unique())
    pred_df = pd.DataFrame(
        [
            {
                **metadata,
                "sample_id": sample_id,
                "true_label": str(y_test.loc[sample_id]),
                "predicted_label": predicted,
                "n_features_used": 0,
            }
            for sample_id in y_test.index
        ]
    )
    score_records = []
    for sample_id in y_test.index:
        for class_label in classes:
            score_records.append(
                {
                    **metadata,
                    "sample_id": sample_id,
                    "class_label": class_label,
                    "score": 1.0 if class_label == predicted else 0.0,
                }
            )
    return pred_df, pd.DataFrame(score_records)


def random_class_prediction_tables(
    labels: pd.Series,
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    metadata: dict,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_train = labels.loc[list(train_ids)].astype(str)
    y_test = labels.loc[list(test_ids)].astype(str)
    class_probs = y_train.value_counts(normalize=True).sort_index()
    classes = class_probs.index.astype(str).tolist()
    probs = class_probs.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(classes, size=len(y_test), replace=True, p=probs)
    pred_df = pd.DataFrame(
        [
            {
                **metadata,
                "sample_id": sample_id,
                "true_label": str(y_test.loc[sample_id]),
                "predicted_label": str(predicted_label),
                "n_features_used": 0,
            }
            for sample_id, predicted_label in zip(y_test.index, sampled)
        ]
    )
    score_records = []
    for sample_id in y_test.index:
        for class_label, score in zip(classes, probs):
            score_records.append(
                {
                    **metadata,
                    "sample_id": sample_id,
                    "class_label": class_label,
                    "score": float(score),
                }
            )
    return pred_df, pd.DataFrame(score_records)


def compute_group_metrics(group: pd.DataFrame) -> dict:
    y_true = group["true_label"].astype(str)
    y_pred = group["predicted_label"].astype(str)
    labels = sorted(set(y_true).union(set(y_pred)))
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    if group["task_type"].iloc[0] == "binary":
        cm = confusion_matrix(y_true, y_pred, labels=["Solid Tissue Normal", "Cancer"])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
            negative_precision = tn / (tn + fn) if (tn + fn) else 0.0
        else:
            specificity = 0.0
            negative_precision = 0.0
    else:
        mcm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
        specificities = []
        negative_precisions = []
        for cm in mcm:
            tn, fp, fn, tp = cm.ravel()
            specificities.append(tn / (tn + fp) if (tn + fp) else 0.0)
            negative_precisions.append(tn / (tn + fn) if (tn + fn) else 0.0)
        specificity = float(np.mean(specificities)) if specificities else 0.0
        negative_precision = (
            float(np.mean(negative_precisions)) if negative_precisions else 0.0
        )
    return {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(macro_f1),
        "specificity": float(specificity),
        "negative_precision": float(negative_precision),
    }


def recalculate_metric_tables(
    predictions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_cols = [
        "case_set",
        "cohort_id",
        "cohort_label",
        "task_type",
        "feature_set",
        "n_regions_requested",
        "fold",
    ]
    metric_rows = []
    for keys, group in predictions_df.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, keys))
        row.update(compute_group_metrics(group))
        row["n_test_samples"] = int(group["sample_id"].nunique())
        row["n_features_used"] = int(group["n_features_used"].max())
        metric_rows.append(row)
    fold_metrics = add_feature_order(pd.DataFrame(metric_rows))

    summary_rows = []
    summary_cols = [c for c in group_cols if c != "fold"]
    for keys, group in fold_metrics.groupby(summary_cols, dropna=False, sort=False):
        row = dict(zip(summary_cols, keys))
        row["n_folds_completed"] = int(group["fold"].nunique())
        row["mean_n_features_used"] = float(group["n_features_used"].mean())
        for metric in METRIC_COLS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std()) if len(group) > 1 else 0.0
            row[f"{metric}_median"] = float(group[metric].median())
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        summary_rows.append(row)
    summary = add_feature_order(pd.DataFrame(summary_rows))

    class_rows = []
    for keys, group in predictions_df.groupby(group_cols, dropna=False, sort=False):
        base = dict(zip(group_cols, keys))
        labels = sorted(
            set(group["true_label"].astype(str)).union(
                group["predicted_label"].astype(str)
            )
        )
        precision, recall, f1, support = precision_recall_fscore_support(
            group["true_label"].astype(str),
            group["predicted_label"].astype(str),
            labels=labels,
            zero_division=0,
        )
        for label, p, r, f, s in zip(labels, precision, recall, f1, support):
            class_rows.append(
                {
                    **base,
                    "class_label": label,
                    "precision": float(p),
                    "recall": float(r),
                    "f1": float(f),
                    "support": int(s),
                }
            )
    per_class = add_feature_order(pd.DataFrame(class_rows))

    confusion_rows = []
    for keys, group in predictions_df.groupby(group_cols, dropna=False, sort=False):
        base = dict(zip(group_cols, keys))
        labels = sorted(
            set(group["true_label"].astype(str)).union(
                group["predicted_label"].astype(str)
            )
        )
        cm = confusion_matrix(
            group["true_label"], group["predicted_label"], labels=labels
        )
        for i, true_label in enumerate(labels):
            for j, predicted_label in enumerate(labels):
                confusion_rows.append(
                    {
                        **base,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": int(cm[i, j]),
                    }
                )
    confusion = add_feature_order(pd.DataFrame(confusion_rows))
    return fold_metrics, summary, per_class, confusion


def build_elbow_summary(
    summary_df: pd.DataFrame, metric: str = "mcc_mean"
) -> pd.DataFrame:
    rows = []
    group_cols = ["case_set", "cohort_id", "cohort_label", "feature_set"]
    if summary_df.empty or metric not in summary_df.columns:
        return pd.DataFrame()
    for keys, group in summary_df.groupby(group_cols, sort=False):
        group = group.dropna(subset=[metric]).copy()
        if group.empty:
            continue
        best = group[metric].max()
        threshold = 0.95 * best
        eligible = group.loc[group[metric] >= threshold].sort_values(
            "n_regions_requested"
        )
        row = dict(zip(group_cols, keys))
        row["metric"] = metric
        row["best_value"] = float(best)
        row["threshold_95pct_best"] = float(threshold)
        row["elbow_n_regions"] = int(eligible.iloc[0]["n_regions_requested"])
        rows.append(row)
    return pd.DataFrame(rows)


def save_feature_matrix(X: pd.DataFrame, path_base: Path) -> Path:
    ensure_dir(path_base.parent)
    try:
        path = path_base.with_suffix(".parquet")
        X.to_parquet(path, compression="snappy")
        return path
    except Exception:
        path = path_base.with_suffix(".tsv.gz")
        X.to_csv(path, sep="\t", compression="gzip")
        return path


def save_selected_regions(
    regions_df: pd.DataFrame, out_dir: Path, metadata: dict
) -> Path:
    region_path = (
        out_dir
        / "selected_regions"
        / safe_name(metadata["case_set"])
        / safe_name(metadata["cohort_id"])
        / safe_name(metadata["feature_set"])
        / f"n{int(metadata['n_regions_requested']):04d}_fold{int(metadata['fold']):02d}.tsv"
    )
    regions = regions_df.copy()
    for key, value in metadata.items():
        regions[key] = value
    return write_table(regions, region_path)


def save_random_failure_debug_artifacts(
    *,
    out_dir: Path,
    metadata: dict,
    selected_pmd_df: pd.DataFrame,
    high_confidence_pmd_df: pd.DataFrame,
    control_space_df: pd.DataFrame,
    random_regions_df: pd.DataFrame,
    random_audit_df: pd.DataFrame,
) -> Path:
    debug_dir = ensure_dir(
        out_dir
        / "random_failure_debug"
        / safe_name(metadata["case_set"])
        / safe_name(metadata["cohort_id"])
        / f"n{int(metadata['n_regions_requested']):04d}_fold{int(metadata['fold']):02d}"
    )
    selected = selected_pmd_df.copy()
    high_conf = high_confidence_pmd_df.copy()
    control_space = control_space_df.copy()
    random_regions = random_regions_df.copy()
    random_audit = random_audit_df.copy()

    if "source_region_id" in random_audit.columns and "region_id" in selected.columns:
        source_map = selected.loc[:, ["region_id", "chr", "start", "end"]].rename(
            columns={
                "region_id": "source_region_id",
                "chr": "failed_chr",
                "start": "failed_start",
                "end": "failed_end",
            }
        )
        random_audit = random_audit.merge(source_map, on="source_region_id", how="left")
    else:
        random_audit["failed_chr"] = pd.Series(dtype=str)
        random_audit["failed_start"] = pd.Series(dtype=float)
        random_audit["failed_end"] = pd.Series(dtype=float)

    failed_chroms = (
        random_audit["failed_chr"].dropna().astype(str).drop_duplicates().tolist()
        if not random_audit.empty
        else []
    )
    summary_rows = []
    for chrom in failed_chroms:
        chrom_name = safe_name(chrom)
        failed_requested = random_audit.loc[random_audit["failed_chr"].astype(str).eq(chrom)].copy()
        selected_chrom = selected.loc[selected["chr"].astype(str).eq(chrom)].copy()
        high_conf_chrom = high_conf.loc[high_conf["chr"].astype(str).eq(chrom)].copy()
        control_space_chrom = control_space.loc[
            control_space["chr"].astype(str).eq(chrom)
        ].copy()
        sampled_random_chrom = random_regions.loc[
            random_regions["chr"].astype(str).eq(chrom)
        ].copy()

        failed_path = write_table(
            failed_requested,
            debug_dir / f"{chrom_name}__failed_requested_pmds.tsv",
        )
        selected_path = write_table(
            selected_chrom,
            debug_dir / f"{chrom_name}__selected_pmds.tsv",
        )
        high_conf_path = write_table(
            high_conf_chrom,
            debug_dir / f"{chrom_name}__high_confidence_pmds.tsv",
        )
        control_space_path = write_table(
            control_space_chrom,
            debug_dir / f"{chrom_name}__remaining_random_space.tsv",
        )
        sampled_random_path = write_table(
            sampled_random_chrom,
            debug_dir / f"{chrom_name}__sampled_random_regions.tsv",
        )
        summary_rows.append(
            {
                **metadata,
                "chromosome": chrom,
                "n_failed_requested_pmds": int(len(failed_requested)),
                "n_selected_pmds_on_chrom": int(len(selected_chrom)),
                "n_high_confidence_pmds_on_chrom": int(len(high_conf_chrom)),
                "n_remaining_random_space_intervals": int(len(control_space_chrom)),
                "remaining_random_space_bp": int(control_space_chrom.get("length", pd.Series(dtype=int)).sum()),
                "n_sampled_random_regions_on_chrom": int(len(sampled_random_chrom)),
                "failed_requested_pmds_path": str(failed_path),
                "selected_pmds_path": str(selected_path),
                "high_confidence_pmds_path": str(high_conf_path),
                "remaining_random_space_path": str(control_space_path),
                "sampled_random_regions_path": str(sampled_random_path),
            }
        )

    if summary_rows:
        write_table(pd.DataFrame(summary_rows), debug_dir / "random_failure_summary.tsv")
    else:
        write_table(random_audit, debug_dir / "random_failure_summary.tsv")
    return debug_dir


def make_feature_status(
    metadata: dict,
    feature_set: str,
    status: str,
    n_regions_selected: int = 0,
    failure_reason: str = "",
) -> dict:
    return {
        **metadata,
        "feature_set": feature_set,
        "status": status,
        "n_regions_selected": int(n_regions_selected),
        "failure_reason": failure_reason,
    }


def raise_or_record_missing_feature(
    *,
    out_dir: Path,
    feature_status_rows: list[dict],
    metadata: dict,
    feature_set: str,
    failure_reason: str,
    allow_missing_feature_sets: bool,
) -> None:
    feature_status_rows.append(
        make_feature_status(
            metadata,
            feature_set,
            "failed",
            n_regions_selected=0,
            failure_reason=failure_reason,
        )
    )
    if not allow_missing_feature_sets:
        write_table(
            add_feature_order(pd.DataFrame(feature_status_rows)),
            out_dir / "feature_set_status.tsv",
        )
        raise ValueError(
            f"{feature_set} failed for {metadata['case_set']} / {metadata['cohort_id']} "
            f"fold={metadata['fold']} n={metadata['n_regions_requested']}: {failure_reason}. "
            "Use --allow-missing-feature-sets only for diagnostics."
        )


def run_pipeline(config: PipelineConfig) -> dict[str, Path]:
    out_dir = ensure_dir(config.out_dir)
    if (
        config.high_confidence_pmd_min_fraction < 0
        or config.high_confidence_pmd_min_fraction > 1
    ):
        raise ValueError("high_confidence_pmd_min_fraction must be between 0 and 1.")
    if config.random_region_min_length_bp <= 0 or config.random_region_max_length_bp <= 0:
        raise ValueError("Random region min/max lengths must be positive integers.")
    if config.random_region_min_length_bp > config.random_region_max_length_bp:
        raise ValueError(
            "random_region_min_length_bp cannot exceed random_region_max_length_bp."
        )
    write_json(
        {
            **asdict(config),
            "out_dir": str(config.out_dir),
            "segmentation_root": str(config.segmentation_root),
            "cgi_bed": str(config.cgi_bed),
            "genome_file": str(config.genome_file),
            "n_values": list(config.n_values or DEFAULT_N_VALUES),
        },
        out_dir / "run_config.json",
    )

    all_samples = load_tcga_samples_info()
    cohorts, cohort_warnings = build_case_cohorts(
        all_samples,
        case_sets=config.case_sets,
        cv_splits=config.cv_splits,
        random_state=config.random_state,
        max_samples_per_class=config.max_samples_per_class,
    )
    if not cohorts:
        raise ValueError("No cohorts were available for the requested case sets.")
    n_values = tuple(config.n_values or DEFAULT_N_VALUES)

    pmd_call_status = ensure_missing_pmds(
        cohorts,
        config.segmentation_root,
        call_missing_pmds=config.call_missing_pmds,
    )
    write_table(pmd_call_status, out_dir / "pmd_call_status.tsv")

    preflight_status = validate_pmd_preflight(cohorts, config.segmentation_root)
    write_table(preflight_status, out_dir / "pmd_preflight_status.tsv")
    missing_preflight = preflight_status.loc[preflight_status["status"].ne("available")]
    if not missing_preflight.empty and not config.allow_missing_feature_sets:
        write_table(
            add_feature_order(preflight_status),
            out_dir / "feature_set_status.tsv",
        )
        example = missing_preflight.head(8).loc[
            :, ["case_set", "cohort_id", "sample_id", "failure_reason"]
        ]
        raise ValueError(
            "PMD preflight failed. PMD and Random are required by default, "
            "but some candidate tumor samples do not have PMD files under "
            f"{config.segmentation_root}. Examples: {example.to_dict(orient='records')}. "
            "Use --allow-missing-feature-sets only for diagnostics."
        )

    all_cohort_samples = pd.concat(
        [cohort["samples_df"] for cohort in cohorts], ignore_index=True
    ).drop_duplicates("sample_id")
    write_table(all_cohort_samples, out_dir / "sample_metadata.tsv")
    cohort_summary = []
    for cohort in cohorts:
        labels = build_labels(cohort["samples_df"], cohort["task_type"])
        for label, n_samples in labels.value_counts().items():
            cohort_summary.append(
                {
                    "case_set": cohort["case_set"],
                    "cohort_id": cohort["cohort_id"],
                    "cohort_label": cohort["cohort_label"],
                    "task_type": cohort["task_type"],
                    "label": label,
                    "n_samples": int(n_samples),
                }
            )
    write_table(pd.DataFrame(cohort_summary), out_dir / "cohort_summary.tsv")

    meth_ref_df = load_meth_ref()
    probe_df = prepare_probe_df(meth_ref_df)
    cgi_df = load_cgi_regions(config.cgi_bed, probe_df)
    meth_data_df, meth_warnings = build_wide_methylation_table(
        all_cohort_samples, meth_ref_df
    )

    prediction_frames = []
    score_frames = []
    split_rows = []
    warning_frames = [cohort_warnings, meth_warnings]
    audit_frames = []
    feature_status_rows: list[dict] = []

    for cohort in cohorts:
        samples_df = cohort["samples_df"].copy().reset_index(drop=True)
        labels = build_labels(samples_df, cohort["task_type"])
        splitter = StratifiedKFold(
            n_splits=config.cv_splits, shuffle=True, random_state=config.random_state
        )
        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(samples_df, labels), start=1
        ):
            train_df = samples_df.iloc[train_idx].reset_index(drop=True)
            test_df = samples_df.iloc[test_idx].reset_index(drop=True)
            train_ids = train_df["sample_id"].astype(str).tolist()
            test_ids = test_df["sample_id"].astype(str).tolist()
            for sample_id in train_ids:
                split_rows.append(
                    {
                        "case_set": cohort["case_set"],
                        "cohort_id": cohort["cohort_id"],
                        "fold": fold,
                        "split": "train",
                        "sample_id": sample_id,
                    }
                )
            for sample_id in test_ids:
                split_rows.append(
                    {
                        "case_set": cohort["case_set"],
                        "cohort_id": cohort["cohort_id"],
                        "fold": fold,
                        "split": "test",
                        "sample_id": sample_id,
                    }
                )

            train_tumors = train_df.loc[
                train_df["sample_type"].eq("Primary Tumor")
            ].copy()
            fold_all_pmds, fold_pmd_support, pmd_warnings = prepare_fold_pmd_tables(
                train_tumors,
                config.segmentation_root,
                config.genome_file,
            )
            if not pmd_warnings.empty:
                warning_frames.append(
                    pmd_warnings.assign(
                        case_set=cohort["case_set"],
                        cohort_id=cohort["cohort_id"],
                        cohort_label=cohort["cohort_label"],
                        task_type=cohort["task_type"],
                        fold=int(fold),
                    )
                )
            control_space = build_candidate_control_space(
                fold_pmd_support,
                probe_df,
                config.genome_file,
                config.high_confidence_pmd_min_fraction,
            )
            high_confidence_pmds = select_high_confidence_pmds(
                fold_pmd_support,
                config.high_confidence_pmd_min_fraction,
            )
            for n_regions in n_values:
                common_meta = {
                    "case_set": cohort["case_set"],
                    "cohort_id": cohort["cohort_id"],
                    "cohort_label": cohort["cohort_label"],
                    "task_type": cohort["task_type"],
                    "n_regions_requested": int(n_regions),
                    "fold": int(fold),
                }
                selected_pmd, fold_pmd_support, selection_warnings = (
                    select_pmd_regions_from_fold_tables(
                        train_tumors,
                        fold_all_pmds,
                        fold_pmd_support,
                        config.genome_file,
                        int(n_regions),
                        cohort["pmd_strategy"],
                    )
                )
                if not selection_warnings.empty:
                    warning_frames.append(selection_warnings.assign(**common_meta))

                feature_region_map = {}
                if not selected_pmd.empty:
                    feature_region_map["PMD"] = selected_pmd
                    random_regions, random_audit = sample_random_non_pmd_regions(
                        selected_pmd,
                        non_pmd_df=control_space,
                        seed=config.random_state + fold + int(n_regions),
                        random_region_min_length_bp=config.random_region_min_length_bp,
                        random_region_max_length_bp=config.random_region_max_length_bp,
                    )
                    if not random_audit.empty:
                        audit_frames.append(
                            random_audit.assign(**common_meta, feature_set="Random")
                        )
                    if not random_regions.empty and len(random_regions) >= len(
                        selected_pmd
                    ):
                        feature_region_map["Random"] = random_regions
                    else:
                        debug_dir = save_random_failure_debug_artifacts(
                            out_dir=out_dir,
                            metadata=common_meta,
                            selected_pmd_df=selected_pmd,
                            high_confidence_pmd_df=high_confidence_pmds,
                            control_space_df=control_space,
                            random_regions_df=random_regions,
                            random_audit_df=random_audit,
                        )
                        failed_chroms = []
                        if (
                            not random_audit.empty
                            and "source_region_id" in random_audit.columns
                            and "region_id" in selected_pmd.columns
                        ):
                            failed_lookup = selected_pmd.loc[
                                selected_pmd["region_id"].isin(
                                    random_audit["source_region_id"].astype(str)
                                ),
                                "chr",
                            ]
                            failed_chroms = (
                                failed_lookup.astype(str).drop_duplicates().tolist()
                            )
                        failure_reason = (
                            f"Generated {len(random_regions)} random non-PMD regions "
                            f"for {len(selected_pmd)} selected PMD regions. "
                            f"Failed chromosome(s): {failed_chroms or ['unknown']}. "
                            f"Debug artifacts: {debug_dir}"
                        )
                        raise_or_record_missing_feature(
                            out_dir=out_dir,
                            feature_status_rows=feature_status_rows,
                            metadata=common_meta,
                            feature_set="Random",
                            failure_reason=failure_reason,
                            allow_missing_feature_sets=config.allow_missing_feature_sets,
                        )
                else:
                    failure_reason = "No PMD regions were selected for this fold and n."
                    raise_or_record_missing_feature(
                        out_dir=out_dir,
                        feature_status_rows=feature_status_rows,
                        metadata=common_meta,
                        feature_set="PMD",
                        failure_reason=failure_reason,
                        allow_missing_feature_sets=config.allow_missing_feature_sets,
                    )
                    warning_frames.append(
                        pd.DataFrame(
                            [
                                {
                                    **common_meta,
                                    "warning": failure_reason,
                                }
                            ]
                        )
                    )

                cgi_regions, cgi_warnings = sample_cgi_regions(
                    cgi_df,
                    int(n_regions),
                    seed=config.random_state + fold + int(n_regions),
                )
                if not cgi_warnings.empty:
                    warning_frames.append(
                        cgi_warnings.assign(**common_meta, feature_set="CGI")
                    )
                if not cgi_regions.empty:
                    feature_region_map["CGI"] = cgi_regions

                for feature_set, regions_df in feature_region_map.items():
                    metadata = {**common_meta, "feature_set": feature_set}
                    save_selected_regions(regions_df, out_dir, metadata)
                    try:
                        X, dropped = extract_feature_matrix(
                            regions_df, samples_df, meth_data_df
                        )
                        if not dropped.empty:
                            audit_frames.append(dropped.assign(**metadata))
                        if config.save_feature_matrices:
                            feature_base = (
                                out_dir
                                / "feature_matrices"
                                / safe_name(cohort["case_set"])
                                / safe_name(cohort["cohort_id"])
                                / safe_name(feature_set)
                                / f"n{int(n_regions):04d}_fold{fold:02d}"
                            )
                            save_feature_matrix(X, feature_base)
                        pred_df, score_df, train_dropped = prediction_tables_for_model(
                            X,
                            labels,
                            train_ids,
                            test_ids,
                            metadata=metadata,
                            n_estimators=config.n_estimators,
                            random_state=config.random_state,
                        )
                        if not train_dropped.empty:
                            audit_frames.append(train_dropped.assign(**metadata))
                        prediction_frames.append(pred_df)
                        score_frames.append(score_df)
                        feature_status_rows.append(
                            make_feature_status(
                                metadata,
                                feature_set,
                                "completed",
                                n_regions_selected=len(regions_df),
                            )
                        )
                    except Exception as exc:
                        raise_or_record_missing_feature(
                            out_dir=out_dir,
                            feature_status_rows=feature_status_rows,
                            metadata=metadata,
                            feature_set=feature_set,
                            failure_reason=str(exc),
                            allow_missing_feature_sets=config.allow_missing_feature_sets,
                        )
                        warning_frames.append(
                            pd.DataFrame(
                                [
                                    {
                                        **metadata,
                                        "warning": f"{feature_set} failed: {exc}",
                                    }
                                ]
                            )
                        )

                baseline_name = (
                    "Always cancer"
                    if cohort["task_type"] == "binary"
                    else "Majority class"
                )
                baseline_meta = {**common_meta, "feature_set": baseline_name}
                pred_df, score_df = baseline_prediction_tables(
                    labels,
                    train_ids,
                    test_ids,
                    task_type=cohort["task_type"],
                    metadata=baseline_meta,
                )
                prediction_frames.append(pred_df)
                score_frames.append(score_df)
                feature_status_rows.append(
                    make_feature_status(
                        baseline_meta,
                        baseline_name,
                        "completed",
                        n_regions_selected=0,
                    )
                )
                if cohort["task_type"] == "multiclass":
                    random_meta = {**common_meta, "feature_set": "Random class"}
                    pred_df, score_df = random_class_prediction_tables(
                        labels,
                        train_ids,
                        test_ids,
                        metadata=random_meta,
                        seed=config.random_state + 100000 * fold + int(n_regions),
                    )
                    prediction_frames.append(pred_df)
                    score_frames.append(score_df)
                    feature_status_rows.append(
                        make_feature_status(
                            random_meta,
                            "Random class",
                            "completed",
                            n_regions_selected=0,
                        )
                    )

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    scores_df = pd.concat(score_frames, ignore_index=True)
    splits_df = pd.DataFrame(split_rows)
    warnings_df = concat_nonempty(warning_frames)
    audits_df = concat_nonempty(audit_frames)

    outputs = write_metric_outputs(out_dir, predictions_df, scores_df, splits_df)
    write_table(warnings_df, out_dir / "warnings.tsv")
    write_table(audits_df, out_dir / "feature_audits.tsv")
    outputs["feature_set_status"] = write_table(
        add_feature_order(pd.DataFrame(feature_status_rows)),
        out_dir / "feature_set_status.tsv",
    )
    outputs["pmd_preflight_status"] = out_dir / "pmd_preflight_status.tsv"
    outputs["pmd_call_status"] = out_dir / "pmd_call_status.tsv"
    return outputs


def write_metric_outputs(
    out_dir: str | Path,
    predictions_df: pd.DataFrame,
    scores_df: pd.DataFrame | None = None,
    splits_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    out_dir = ensure_dir(out_dir)
    outputs: dict[str, Path] = {}
    outputs["predictions"] = write_table(predictions_df, out_dir / "cv_predictions.tsv")
    if scores_df is not None:
        outputs["scores"] = write_table(scores_df, out_dir / "cv_prediction_scores.tsv")
    if splits_df is not None:
        outputs["splits"] = write_table(splits_df, out_dir / "cv_splits.tsv")

    fold_metrics, summary, per_class, confusion = recalculate_metric_tables(
        predictions_df
    )
    elbow = build_elbow_summary(summary)
    outputs["fold_metrics"] = write_table(fold_metrics, out_dir / "fold_metrics.tsv")
    outputs["summary_metrics"] = write_table(summary, out_dir / "summary_metrics.tsv")
    outputs["per_class_metrics"] = write_table(
        per_class, out_dir / "per_class_metrics.tsv"
    )
    outputs["confusion_matrix"] = write_table(
        confusion, out_dir / "confusion_matrix.tsv"
    )
    outputs["elbow_summary"] = write_table(elbow, out_dir / "elbow_summary.tsv")
    plot_metric_outputs(fold_metrics, summary, confusion, out_dir / "plots")
    return outputs


def recalculate_metrics_from_dir(out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    predictions_path = out_dir / "cv_predictions.tsv"
    scores_path = out_dir / "cv_prediction_scores.tsv"
    splits_path = out_dir / "cv_splits.tsv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing saved predictions: {predictions_path}")
    predictions_df = pd.read_csv(predictions_path, sep="\t")
    scores_df = pd.read_csv(scores_path, sep="\t") if scores_path.exists() else None
    splits_df = pd.read_csv(splits_path, sep="\t") if splits_path.exists() else None
    return write_metric_outputs(out_dir, predictions_df, scores_df, splits_df)


def plot_metric_outputs(
    fold_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    confusion: pd.DataFrame,
    plot_dir: str | Path,
) -> None:
    plot_dir = ensure_dir(plot_dir)
    for subdir in [
        "average_metric_bars",
        "cv_score_boxplots",
        "confusion_matrices",
        "metric_lines",
    ]:
        ensure_dir(plot_dir / subdir)
    if summary.empty and fold_metrics.empty:
        return

    cohort_keys = []
    if not summary.empty:
        cohort_keys.extend(
            summary.loc[:, ["case_set", "cohort_id"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
    if not fold_metrics.empty:
        cohort_keys.extend(
            fold_metrics.loc[:, ["case_set", "cohort_id"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
    cohort_keys = list(dict.fromkeys(cohort_keys))

    for case_set, cohort_id in cohort_keys:
        cohort_summary = summary.loc[
            summary["case_set"].eq(case_set) & summary["cohort_id"].eq(cohort_id)
        ].copy()
        cohort_folds = fold_metrics.loc[
            fold_metrics["case_set"].eq(case_set)
            & fold_metrics["cohort_id"].eq(cohort_id)
        ].copy()
        task_type = (
            cohort_summary["task_type"].dropna().iloc[0]
            if not cohort_summary.empty
            else cohort_folds["task_type"].dropna().iloc[0]
        )
        cohort_label = (
            cohort_summary["cohort_label"].dropna().iloc[0]
            if not cohort_summary.empty
            else cohort_folds["cohort_label"].dropna().iloc[0]
        )
        feature_order = expected_feature_order(str(task_type))
        n_values_for_cohort = sorted(
            set(
                cohort_summary.get("n_regions_requested", pd.Series(dtype=float))
                .dropna()
                .astype(int)
                .tolist()
            )
            | set(
                cohort_folds.get("n_regions_requested", pd.Series(dtype=float))
                .dropna()
                .astype(int)
                .tolist()
            )
        )

        for n_value in n_values_for_cohort:
            n_dir = f"n_{int(n_value):04d}"
            n_folds = (
                cohort_folds.loc[
                    cohort_folds["n_regions_requested"].astype(int).eq(int(n_value))
                ].copy()
                if not cohort_folds.empty
                else pd.DataFrame()
            )

            if not n_folds.empty:
                ensure_dir(plot_dir / "average_metric_bars" / n_dir)
                ensure_dir(plot_dir / "cv_score_boxplots" / n_dir)
                fig, axes = plt.subplots(
                    1,
                    len(METRIC_COLS),
                    figsize=(4.2 * len(METRIC_COLS), 4.5),
                    sharey=True,
                )
                if len(METRIC_COLS) == 1:
                    axes = [axes]
                x = np.arange(len(feature_order))
                for ax, metric in zip(axes, METRIC_COLS):
                    values = [
                        n_folds.loc[
                            n_folds["feature_set"].eq(feature_set), metric
                        ].mean()
                        for feature_set in feature_order
                    ]
                    colors = [
                        FEATURE_COLORS.get(feature_set, "#777777")
                        for feature_set in feature_order
                    ]
                    ax.bar(x, values, color=colors, alpha=0.9)
                    ax.set_xticks(x)
                    ax.set_xticklabels(feature_order, rotation=25, ha="right")
                    ax.set_ylim(0, 1.02)
                    ax.set_title(metric.replace("_", " ").title())
                    ax.grid(axis="y", alpha=0.25)
                    ax.set_axisbelow(True)
                axes[0].set_ylabel("Mean CV score")
                handles = [
                    plt.Line2D(
                        [0], [0], color=FEATURE_COLORS.get(feature_set, "#777777"), lw=6
                    )
                    for feature_set in feature_order
                ]
                fig.legend(
                    handles,
                    feature_order,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.92),
                    ncol=len(feature_order),
                    frameon=False,
                )
                fig.suptitle(
                    f"Average CV metrics by model: {cohort_label} (n={int(n_value)})",
                    y=0.99,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.82])
                fig.savefig(
                    plot_dir
                    / "average_metric_bars"
                    / n_dir
                    / f"{safe_name(case_set)}__{safe_name(cohort_id)}__n{int(n_value):04d}__average_metrics.png",
                    dpi=160,
                    bbox_inches="tight",
                    pad_inches=0.25,
                )
                plt.close(fig)

                fig, axes = plt.subplots(
                    1,
                    len(METRIC_COLS),
                    figsize=(4.2 * len(METRIC_COLS), 4.5),
                    sharey=True,
                )
                if len(METRIC_COLS) == 1:
                    axes = [axes]
                for ax, metric in zip(axes, METRIC_COLS):
                    data = [
                        n_folds.loc[n_folds["feature_set"].eq(feature_set), metric]
                        .dropna()
                        .to_numpy()
                        for feature_set in feature_order
                    ]
                    positions = np.arange(1, len(feature_order) + 1)
                    nonempty_positions = [
                        pos for pos, values in zip(positions, data) if len(values)
                    ]
                    nonempty_data = [values for values in data if len(values)]
                    if nonempty_data:
                        box = ax.boxplot(
                            nonempty_data,
                            positions=nonempty_positions,
                            widths=0.55,
                            patch_artist=True,
                            boxprops={"edgecolor": "dimgray", "linewidth": 1.3},
                            medianprops={"color": "black", "linewidth": 1.8},
                            whiskerprops={"color": "dimgray", "linewidth": 1.2},
                            capprops={"color": "dimgray", "linewidth": 1.2},
                            flierprops={
                                "marker": "o",
                                "markerfacecolor": "gray",
                                "markeredgecolor": "dimgray",
                                "alpha": 0.55,
                                "markersize": 3,
                            },
                        )
                        for patch, pos in zip(box["boxes"], nonempty_positions):
                            patch.set_facecolor(
                                FEATURE_COLORS.get(feature_order[pos - 1], "#D0D0D0")
                            )
                            patch.set_alpha(0.75)
                    ax.set_xticks(positions)
                    ax.set_xticklabels(feature_order, rotation=25, ha="right")
                    ax.set_ylim(0, 1.02)
                    ax.set_title(metric.replace("_", " ").title())
                    ax.grid(axis="y", alpha=0.25)
                    ax.set_axisbelow(True)
                axes[0].set_ylabel("Fold CV score")
                fig.suptitle(
                    f"Fold-level CV scores by model: {cohort_label} (n={int(n_value)})",
                    y=0.98,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.88])
                fig.savefig(
                    plot_dir
                    / "cv_score_boxplots"
                    / n_dir
                    / f"{safe_name(case_set)}__{safe_name(cohort_id)}__n{int(n_value):04d}__cv_scores.png",
                    dpi=160,
                    bbox_inches="tight",
                    pad_inches=0.25,
                )
                plt.close(fig)

        if not cohort_summary.empty:
            n_values = sorted(cohort_summary["n_regions_requested"].dropna().unique())
            for metric in METRIC_COLS:
                metric_col = f"{metric}_mean"
                if metric_col not in cohort_summary.columns:
                    continue
                fig, ax = plt.subplots(figsize=(7, 4.5))
                for feature_set in feature_order:
                    feature_df = cohort_summary.loc[
                        cohort_summary["feature_set"].eq(feature_set)
                    ].sort_values("n_regions_requested")
                    if feature_df.empty:
                        y_values = [np.nan] * len(n_values)
                        x_values = n_values
                    else:
                        mapped = feature_df.set_index("n_regions_requested")[metric_col]
                        x_values = n_values
                        y_values = [mapped.get(n_value, np.nan) for n_value in n_values]
                    ax.plot(
                        x_values,
                        y_values,
                        marker="o",
                        linewidth=1.6,
                        label=feature_set,
                        color=FEATURE_COLORS.get(feature_set),
                    )
                ax.set_xscale("log")
                ax.set_ylim(0, 1.02)
                ax.set_xlabel("Number of regions")
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.set_title(
                    f"{metric.replace('_', ' ').title()} by region count: {cohort_label}",
                    pad=14,
                )
                ax.legend(fontsize=8, frameon=False)
                ax.grid(axis="y", alpha=0.25)
                fig.tight_layout()
                fig.savefig(
                    plot_dir
                    / "metric_lines"
                    / f"{safe_name(case_set)}__{safe_name(cohort_id)}__{metric}_line.png",
                    dpi=160,
                    bbox_inches="tight",
                    pad_inches=0.25,
                )
                plt.close(fig)

        if confusion.empty:
            continue
        cohort_confusion = confusion.loc[
            confusion["case_set"].eq(case_set) & confusion["cohort_id"].eq(cohort_id)
        ].copy()
        for n_value in sorted(
            cohort_confusion["n_regions_requested"].dropna().astype(int).unique()
        ):
            n_dir = f"n_{int(n_value):04d}"
            ensure_dir(plot_dir / "confusion_matrices" / n_dir)
            n_confusion = cohort_confusion.loc[
                cohort_confusion["n_regions_requested"].astype(int).eq(int(n_value))
            ].copy()
            for feature_set in feature_order:
                feature_confusion = n_confusion.loc[
                    n_confusion["feature_set"].eq(feature_set)
                ].copy()
                if feature_confusion.empty:
                    continue
                pivot = feature_confusion.pivot_table(
                    index="true_label",
                    columns="predicted_label",
                    values="count",
                    aggfunc="sum",
                    fill_value=0,
                )
                pivot = pivot.sort_index().sort_index(axis=1)
                normalized = pivot.div(
                    pivot.sum(axis=1).replace(0, np.nan), axis=0
                ).fillna(0)
                fig, ax = plt.subplots(
                    figsize=(
                        1.2 * len(normalized.columns) + 2,
                        1.0 * len(normalized.index) + 2,
                    )
                )
                im = ax.imshow(normalized.to_numpy(), cmap="Blues", vmin=0, vmax=1)
                ax.set_xticks(range(len(normalized.columns)))
                ax.set_xticklabels(normalized.columns, rotation=90)
                ax.set_yticks(range(len(normalized.index)))
                ax.set_yticklabels(normalized.index)
                ax.set_xlabel("Predicted label")
                ax.set_ylabel("True label")
                ax.set_title(
                    f"Normalized confusion matrix: {feature_set}\n{cohort_label} (n={int(n_value)})",
                    pad=14,
                )
                for i in range(normalized.shape[0]):
                    for j in range(normalized.shape[1]):
                        value = normalized.iat[i, j]
                        ax.text(
                            j,
                            i,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            color="black" if value < 0.65 else "white",
                            fontsize=9,
                        )
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                fig.savefig(
                    plot_dir
                    / "confusion_matrices"
                    / n_dir
                    / f"{safe_name(case_set)}__{safe_name(cohort_id)}__n{int(n_value):04d}__{safe_name(feature_set)}__confusion.png",
                    dpi=160,
                    bbox_inches="tight",
                    pad_inches=0.25,
                )
                plt.close(fig)
