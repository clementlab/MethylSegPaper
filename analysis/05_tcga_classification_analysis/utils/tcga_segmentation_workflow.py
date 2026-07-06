from __future__ import annotations

import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TCGA_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
SHARED_METHYLSEG_DIR = PROJECT_ROOT / "analysis" / "shared_utils" / "methylseg"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repo_paths import (
    REFERENCE_DATA_DIR,
    REGION_CALLING_ANALYSIS_DIR,
    TCGA_CLASSIFICATION_RESULTS_DIR,
)

OUT_DIR = TCGA_CLASSIFICATION_RESULTS_DIR
SAMPLES_INFO_PATH = REFERENCE_DATA_DIR / "runAll.sh.samples"
METH_REF_PATH = REFERENCE_DATA_DIR / "parse450K.pl.order.lookup"

TRAIN_SAMPLE_ID = "TCGA-BD-A3EP-01A"
DEFAULT_GENOME = "hg38"
DEFAULT_RANDOM_STATE = 42

SEGMENTATION_OUT_DIR = OUT_DIR / "01_segmentation"
COLLECT_OUT_DIR = OUT_DIR / "02"
ANALYSIS_OUT_DIR = OUT_DIR / "03"
IGV_EXPORT_OUT_DIR = OUT_DIR / "04_igv_exports"

TCGA_IGV_REGION_TRACKS = {
    "dnmtools_array": "dnmtools_array_bed",
    "methylseg_hm450k": "methylseg_hm450k_bed",
}
TCGA_LAD_REFERENCE_BED = (
    ANALYSIS_OUT_DIR / "lad_reference" / "laminB1Lads.hg38.cleaned.bed"
)
TCGA_CONSENSUS_REGIONS_PATH = ANALYSIS_OUT_DIR / "consensus_regions.tsv"

COMPARATOR_DIR = REGION_CALLING_ANALYSIS_DIR / "utils"
LAD_SLURM_DIR = TCGA_ANALYSIS_DIR.parent / "04_lad_analysis"

for import_dir in (SHARED_METHYLSEG_DIR, PROJECT_ROOT):
    import_str = str(import_dir)
    if import_str not in sys.path:
        sys.path.insert(0, import_str)

DEFAULT_LAD_INTERVAL_TRACK_PATH = REFERENCE_DATA_DIR / "LAD_intervals.bed"
DEFAULT_LAMINB1_SIGNAL_TRACK_PATH = REFERENCE_DATA_DIR / "laminB1_signal.bedGraph"
DEFAULT_LIFTOVER_SCRIPT_PATH = REFERENCE_DATA_DIR / "liftover_bed.r"
DEFAULT_HG19_TO_HG38_CHAIN = REFERENCE_DATA_DIR / "hg19ToHg38.over.chain"

SEGMENTATION_IMPORT_ERROR = None
METHYLSEG_IMPORT_ERROR = None
MethylDataPrep = None
MethylSegToolPathway = None
SampleInfo = None
MethylationStates = None
MethylStateAssignmentMethod = None
try:
    from methylseg import (
        SampleInfo,
        MethylDataPrep,
        MethylationStates,
        MethylSegPathway as MethylSegToolPathway,
        MethylStateAssignmentMethod,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    METHYLSEG_IMPORT_ERROR = exc

for import_dir in (COMPARATOR_DIR, LAD_SLURM_DIR):
    import_str = str(import_dir)
    if import_str not in sys.path:
        sys.path.append(import_str)

try:
    from methyl_tool_comparator import (
        DNMToolsPathway,
        MethylSegPathway,
        RunStats,
        SharedPrepManager,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    DNMToolsPathway = None  # type: ignore[assignment]
    MethylSegPathway = object  # type: ignore[assignment]
    RunStats = None  # type: ignore[assignment]
    SharedPrepManager = None  # type: ignore[assignment]
    SEGMENTATION_IMPORT_ERROR = exc


def _raise_if_segmentation_deps_missing() -> None:
    if METHYLSEG_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "MethylSeg dependencies are unavailable. "
            "Install the environment that provides analysis.shared_utils.methyl_seg dependencies."
        ) from METHYLSEG_IMPORT_ERROR
    if SEGMENTATION_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "Segmentation dependencies are unavailable. "
            "Install the environment that provides methyl_tool_comparator dependencies."
        ) from SEGMENTATION_IMPORT_ERROR


CANONICAL_SEGMENT_COLUMNS = [
    "sample_id",
    "tool",
    "chrom",
    "start",
    "end",
    "length_bp",
]


@dataclass(frozen=True)
class SegmentRunResult:
    sample_id: str
    dnmtools_array_bed: Path
    methylseg_hm450k_bed: Path
    dnmtools_dir: Path
    methylseg_dir: Path
    prep_dir: Path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_tcga_samples_info(
    samples_info_path: str | Path = SAMPLES_INFO_PATH,
) -> pd.DataFrame:
    samples_info = pd.read_csv(samples_info_path, sep="\t").copy()
    if "sample" in samples_info.columns:
        samples_info["sample_barcode"] = samples_info["sample"].astype(str)
        if "sample_id" in samples_info.columns:
            samples_info = samples_info.rename(
                columns={"sample_id": "source_sample_uuid"}
            )
        samples_info["sample_id"] = samples_info["sample_barcode"]
    return samples_info


def load_meth_ref(meth_ref_path: str | Path = METH_REF_PATH) -> pd.DataFrame:
    return pd.read_csv(meth_ref_path, sep="\t").copy()


def load_tcga_sample_beta_dataframe(
    sample_id: str,
    meth_file: str | Path,
    drop_nas: bool = True,
    *,
    meth_ref_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    meth_file = Path(meth_file)
    if meth_file.suffix.lower() in {".csv", ".tsv", ".txt"}:
        meth_data_df = pd.read_csv(meth_file, sep=None, engine="python").copy()
        required_cols = {"CpG_chrm", "CpG_beg", "CpG_end", "beta"}
        missing_cols = sorted(required_cols - set(meth_data_df.columns))
        if missing_cols:
            raise ValueError(
                f"Tabular TCGA methylation file for {sample_id} is missing columns: {missing_cols}"
            )
        out_df = meth_data_df.loc[:, ["CpG_chrm", "CpG_beg", "CpG_end"]].copy()
        out_df["beta"] = pd.to_numeric(meth_data_df["beta"], errors="coerce")
        if "probe" in meth_data_df.columns:
            out_df["probe"] = meth_data_df["probe"].astype(str)
        else:
            out_df["probe"] = (
                out_df["CpG_chrm"].astype(str)
                + ":"
                + out_df["CpG_beg"].astype(str)
                + "-"
                + out_df["CpG_end"].astype(str)
            )
    else:
        meth_ref_df = load_meth_ref() if meth_ref_df is None else meth_ref_df.copy()
        meth_data = np.load(meth_file)
        if meth_data.ndim == 2 and meth_data.shape[1] == 1:
            meth_data = meth_data[:, 0]
        if meth_data.ndim != 1:
            raise ValueError(
                f"Unexpected TCGA methylation shape for {sample_id}: {meth_data.shape}"
            )
        if len(meth_data) != len(meth_ref_df):
            raise ValueError(
                f"Length mismatch for {sample_id}: meth_ref has {len(meth_ref_df)} rows "
                f"but methylation array has {len(meth_data)} values."
            )

        meth_data = np.where(meth_data == 255, np.nan, meth_data)
        meth_data_df = meth_ref_df.copy()
        meth_data_df["beta"] = meth_data.astype(float) / 100.0

        out_df = meth_data_df.loc[:, ["CpG_chrm", "CpG_beg", "CpG_end"]].copy()
        out_df["beta"] = pd.to_numeric(meth_data_df["beta"], errors="coerce")
        out_df["probe"] = meth_data_df["key"].astype(str)
    out_df["CpG_beg"] = pd.to_numeric(out_df["CpG_beg"], errors="coerce")
    out_df["CpG_end"] = pd.to_numeric(out_df["CpG_end"], errors="coerce")
    if drop_nas:
        out_df = out_df.dropna(subset=["CpG_beg", "CpG_end", "beta"]).copy()
        out_df["CpG_beg"] = out_df["CpG_beg"].astype(np.int64)
        out_df["CpG_end"] = out_df["CpG_end"].astype(np.int64)
    else:
        out_df["CpG_beg"] = out_df["CpG_beg"].astype("Int64")
        out_df["CpG_end"] = out_df["CpG_end"].astype("Int64")
    out_df["CpG_chrm"] = out_df["CpG_chrm"].astype(str)
    if not out_df["CpG_chrm"].str.startswith("chr").all():
        out_df["CpG_chrm"] = "chr" + out_df["CpG_chrm"].str.replace(
            "^chr", "", regex=True
        )
    out_df = out_df.sort_values(["CpG_chrm", "CpG_beg", "CpG_end"]).reset_index(
        drop=True
    )
    return out_df


def write_tcga_hm450k_beta_file(
    sample_id: str,
    meth_file: str | Path,
    out_path: str | Path,
    *,
    meth_ref_df: pd.DataFrame | None = None,
) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    beta_df = load_tcga_sample_beta_dataframe(
        sample_id=sample_id,
        meth_file=meth_file,
        meth_ref_df=meth_ref_df,
    )
    beta_df.to_csv(out_path, sep="\t", header=False, index=False)
    return out_path


def write_tcga_dnmtools_array_input(
    sample_id: str,
    meth_file: str | Path,
    out_path: str | Path,
    *,
    meth_ref_df: pd.DataFrame | None = None,
) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    beta_df = load_tcga_sample_beta_dataframe(
        sample_id=sample_id,
        meth_file=meth_file,
        meth_ref_df=meth_ref_df,
    )
    dnm_df = pd.DataFrame(
        {
            "chr": beta_df["CpG_chrm"].astype(str),
            "pos": beta_df["CpG_beg"].astype(np.int64),
            "strand": "+",
            "context": "CpG",
            "meth_level": beta_df["beta"].astype(float),
            "coverage": 1,
        }
    )
    dnm_df.to_csv(out_path, sep="\t", header=False, index=False)
    return out_path


def select_tcga_samples(
    n_samples: int,
    samples_info_path: str | Path = SAMPLES_INFO_PATH,
    random_state: int = DEFAULT_RANDOM_STATE,
    include_train_sample: bool = True,
    train_sample_id: str = TRAIN_SAMPLE_ID,
    include_non_tumor_samples: bool = False,
    min_group_size: int | None = None,
    group_col: str = "project_id",
) -> pd.DataFrame:
    samples_info = load_tcga_samples_info(samples_info_path)
    include_all_solid_tissue_normals = include_non_tumor_samples and n_samples > 800

    tumor_samples = samples_info.loc[
        samples_info["sample_id"].astype(str).str.startswith("TCGA-")
        & samples_info["project_id"].astype(str).str.startswith("TCGA-")
        & samples_info["project_descriptor"].notna()
        & samples_info["methylation_file"].notna()
    ].copy()

    solid_tissue_normal_samples = tumor_samples.loc[
        tumor_samples["sample_type"].astype(str).eq("Solid Tissue Normal")
    ].copy()

    if not include_non_tumor_samples:
        tumor_samples = tumor_samples.loc[
            tumor_samples["sample_type"].astype(str).eq("Primary Tumor")
        ].copy()

    tumor_samples = tumor_samples.drop_duplicates(subset=["sample_id"]).reset_index(
        drop=True
    )
    selection_group_col = group_col

    if include_non_tumor_samples and min_group_size is not None:
        selection_group_col = "__selection_group"
        tumor_samples[selection_group_col] = (
            tumor_samples[group_col].astype(str)
            + " | "
            + tumor_samples["sample_type"].astype(str)
        )

    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if n_samples > len(tumor_samples):
        raise ValueError(
            f"Requested n_samples={n_samples}, but only {len(tumor_samples)} tumor samples are available."
        )

    rng = np.random.default_rng(random_state)

    if min_group_size is None:
        selected = tumor_samples.sample(n=n_samples, random_state=random_state).copy()
    else:
        if group_col not in tumor_samples.columns:
            raise KeyError(f"{group_col!r} not found in samples_info")

        grouped = {
            str(group): df.copy()
            for group, df in tumor_samples.groupby(
                tumor_samples[selection_group_col].astype(str)
            )
        }

        # keep only groups that can satisfy the minimum
        grouped = {g: df for g, df in grouped.items() if len(df) >= min_group_size}
        if len(grouped) == 0:
            raise ValueError(
                f"No groups have at least min_group_size={min_group_size} samples."
            )

        groups = sorted(grouped.keys())
        min_required = len(groups) * min_group_size

        if n_samples < min_required:
            raise ValueError(
                f"n_samples={n_samples} is too small to give {min_group_size} samples "
                f"to each of the {len(groups)} groups that meet the minimum "
                f"(need at least {min_required})."
            )

        # reserve the minimum for each group
        parts = []
        for g in groups:
            parts.append(grouped[g].sample(n=min_group_size, random_state=random_state))

        selected = pd.concat(parts, ignore_index=True)

        # distribute remaining samples across the same groups, capped by availability
        remaining = n_samples - len(selected)
        if remaining > 0:
            available = tumor_samples.loc[
                tumor_samples["sample_id"]
                .astype(str)
                .isin(selected["sample_id"].astype(str))
                == False
            ].copy()

            # keep adding samples, preferring groups that still have room
            while remaining > 0 and len(available) > 0:
                candidate_groups = [
                    g
                    for g in groups
                    if len(grouped[g])
                    > (selected[selection_group_col].astype(str).eq(g).sum())
                ]
                if not candidate_groups:
                    break

                g = candidate_groups[rng.integers(0, len(candidate_groups))]
                group_remaining = grouped[g].loc[
                    ~grouped[g]["sample_id"]
                    .astype(str)
                    .isin(selected["sample_id"].astype(str))
                ]
                if len(group_remaining) == 0:
                    continue

                extra = group_remaining.sample(n=1, random_state=random_state)
                selected = pd.concat([selected, extra], ignore_index=True)
                remaining -= 1

    if include_train_sample and train_sample_id in set(
        tumor_samples["sample_id"].astype(str)
    ):
        train_row = tumor_samples.loc[
            tumor_samples["sample_id"].astype(str).eq(str(train_sample_id))
        ].iloc[0]

        if not selected["sample_id"].astype(str).eq(str(train_sample_id)).any():
            selected.iloc[0] = train_row

    if include_all_solid_tissue_normals and len(solid_tissue_normal_samples) > 0:
        selected = pd.concat(
            [selected, solid_tissue_normal_samples],
            ignore_index=True,
        )
        selected = selected.drop_duplicates(
            subset=["sample_id"], keep="first"
        ).reset_index(drop=True)

    selected = selected.sort_values(["project_descriptor", "sample_id"]).reset_index(
        drop=True
    )
    if (
        selection_group_col == "__selection_group"
        and selection_group_col in selected.columns
    ):
        selected = selected.drop(columns=[selection_group_col])
    selected["selection_random_state"] = int(random_state)
    selected["selection_size_requested"] = int(n_samples)
    selected["selection_size_actual"] = int(len(selected))
    selected["is_train_anchor"] = (
        selected["sample_id"].astype(str).eq(str(train_sample_id))
    )
    return selected


def write_selected_samples_manifest(
    selected_samples: pd.DataFrame,
    out_path: str | Path = OUT_DIR / "00" / "selected_samples_info.tsv",
) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    selected_samples.to_csv(out_path, sep="\t", index=False)
    return out_path


class TCGADNMToolsArrayPathway(
    DNMToolsPathway if DNMToolsPathway is not None else object
):
    def __init__(self, *args, **kwargs):
        _raise_if_segmentation_deps_missing()
        super().__init__(*args, **kwargs)

    def run_tool(self) -> dict:
        hm450k_file = Path(self.out_dir) / self.sample_id / "prep" / "450k.tsv"
        output_dir = Path(self.out_dir) / self.sample_id / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        array_output_file = output_dir / "arraymode.dnmtools_PMDs.bed"

        pmd_array_cmd = [
            "dnmtools",
            "pmd",
            "-i",
            "1000",
            "-a",
            "-d",
            "100000",
            "-b",
            "20000",
            "-o",
            str(array_output_file),
            str(hm450k_file),
        ]

        array_run_stats = {}
        if not array_output_file.exists() or self.force_recreate:
            array_run_stats = self._run(pmd_array_cmd)

        self.run_stats.append(
            RunStats(tool_name="DNMTools_Array_PMD", **array_run_stats)
        )

        prep_dir = self._prep_dir()
        self._write_prep_config(
            {
                "tool_name": "DNMTools_Array",
                "sample_id": self.sample_id,
                "genome": self.genome,
                "input_paths": {
                    "wgbs_tsv": prep_dir / "wgbs.tsv",
                    "wgbs_450k_tsv": prep_dir / "wgbs_450k.tsv",
                    "dnmtools_450k_tsv": prep_dir / "450k.tsv",
                },
                "prep_filters": {
                    "hm450k": [
                        "coverage > 0",
                        "drop rows with non-finite meth_level before writing 450k.tsv",
                    ],
                },
                "tool_commands": {
                    "array_pmd": pmd_array_cmd,
                },
            }
        )
        return {"pmd_array_bed": str(array_output_file)}

    def run_data_prep(self) -> dict:
        prep_dir = self._prep_dir()
        meth_ref_df = load_meth_ref()
        hm450k_beta = write_tcga_hm450k_beta_file(
            sample_id=self.sample_id,
            meth_file=self.meth_file,
            out_path=prep_dir / "450k.beta",
            meth_ref_df=meth_ref_df,
        )
        dnmtools_450k_tsv = write_tcga_dnmtools_array_input(
            sample_id=self.sample_id,
            meth_file=self.meth_file,
            out_path=prep_dir / "450k.tsv",
            meth_ref_df=meth_ref_df,
        )
        return {
            "hm450k_beta": hm450k_beta,
            "dnmtools_450k_tsv": dnmtools_450k_tsv,
        }


class TCGAMethylSegHM450KPathway(MethylSegPathway):
    def __init__(self, *args, **kwargs):
        _raise_if_segmentation_deps_missing()
        super().__init__(*args, **kwargs)

    def run_tool(self) -> dict:
        _raise_if_segmentation_deps_missing()
        prep_dir = self._prep_dir()
        hm450_meth_data = prep_dir / "450k.beta"
        hm450_sample_info, hm450_removed_cpgs = MethylDataPrep(
            meth_file=hm450_meth_data,
            sample_id=f"{self.sample_id}_hm450k",
            resolution="450k",
            remove_low_coverage_like_cpgs=True,
        ).prepare()
        self.hm450_removed_cpgs = self._annotate_removed_cpg_reason(
            hm450_removed_cpgs,
            "low coverage like",
        )
        self._write_removed_cpgs("hm450k", self.hm450_removed_cpgs)

        methyl_seg_out_dir = Path(self.out_dir) / self.sample_id / "out" / "hm450k"
        methyl_seg_out_dir.mkdir(parents=True, exist_ok=True)

        hm450k_train_sample = self.train_sample_info or hm450_sample_info
        hm450_methylseg = MethylSegToolPathway(
            train_sample_info=hm450k_train_sample,
            window_specs=self.hm450_window_specs,
            n_states=self.n_states,
            int_low_cutoff=self.int_low_cutoff,
            int_high_cutoff=self.int_high_cutoff,
            high_cutoff=self.high_cutoff,
            out_dir=methyl_seg_out_dir,
            random_state=self.random_state,
            hmm_type=self.hm450_hmm_type,
            hmm_params=self.hm450_hmm_params,
            min_region_length=self.methylseg_clean_min_region_length,
            min_region_cpgs=self.methylseg_clean_min_cpgs,
            merge_gap_bp=self.methylseg_clean_merge_gap_bp,
            state_assignment_method=self.state_assignment_method.value,
        )

        _, summary_files = self._run_methylseg_on_each_chromosome(
            hm450_methylseg,
            hm450_sample_info,
            tool_suffix="HM450K",
        )
        self._write_prep_config(
            {
                "tool_name": "MethylSeg_HM450K",
                "sample_id": self.sample_id,
                "genome": self.genome,
                "input_paths": {
                    "wgbs_tsv": prep_dir / "wgbs.tsv",
                    "hm450k_beta": prep_dir / "450k.beta",
                    "hm450k_meth_ref": prep_dir / "450k_meth_ref.tsv",
                },
                "tool_runtime_settings": {
                    "fit_methyl_seg": self.fit_methyl_seg,
                    "train_sample": self.train_sample,
                    "random_state": self.random_state,
                },
                "platforms": {
                    "hm450k": {
                        "window_specs": self.hm450_window_specs,
                        "hmm_type": self.hm450_hmm_type,
                        "hmm_params": self.hm450_hmm_params,
                        "min_region_length": self.methylseg_clean_min_region_length,
                        "min_region_cpgs": self.methylseg_clean_min_cpgs,
                        "merge_gap_bp": self.methylseg_clean_merge_gap_bp,
                    }
                },
            }
        )
        return {
            "summary_files": summary_files,
            "hm450k_pmd_bed": str(
                methyl_seg_out_dir / "summary_files" / "segments_cleaned_PMD.bed"
            ),
        }

    def run_data_prep(self) -> dict:
        prep_dir = self._prep_dir()
        meth_ref_df = load_meth_ref()
        hm450k_beta = write_tcga_hm450k_beta_file(
            sample_id=self.sample_id,
            meth_file=self.meth_file,
            out_path=prep_dir / "450k.beta",
            meth_ref_df=meth_ref_df,
        )
        hm450k_meth_ref = prep_dir / "450k_meth_ref.tsv"
        meth_ref_df.loc[:, ["CpG_chrm", "CpG_beg", "CpG_end"]].to_csv(
            hm450k_meth_ref,
            sep="\t",
            index=False,
        )
        return {
            "hm450k_beta": hm450k_beta,
            "hm450k_meth_ref": hm450k_meth_ref,
        }


def build_shared_prep_manager(
    sample_id: str,
    meth_file: str | Path,
    *,
    genome: str = DEFAULT_GENOME,
    segmentation_root: str | Path = SEGMENTATION_OUT_DIR,
    force_recreate: bool = False,
) -> SharedPrepManager:
    _raise_if_segmentation_deps_missing()
    return SharedPrepManager(
        sample_id=sample_id,
        meth_file=meth_file,
        genome=genome,
        out_dir=Path(segmentation_root) / "comparison",
        force_recreate=force_recreate,
    )


def run_segmentation_for_sample(
    sample_id: str,
    meth_file: str | Path,
    *,
    genome: str = DEFAULT_GENOME,
    segmentation_root: str | Path = SEGMENTATION_OUT_DIR,
    force_recreate: bool = False,
    print_logs: bool = True,
    run_on_dnmtools_array: bool = True,
    shared_prep: SharedPrepManager | None = None,
    clean_individual_chr_outputs: bool = False,
) -> SegmentRunResult:
    _raise_if_segmentation_deps_missing()
    segmentation_root = ensure_dir(segmentation_root)
    dnmtools_dir = segmentation_root / "dnmtools"
    methylseg_dir = segmentation_root / "methylseg"
    # prep_dir = ensure_dir(segmentation_root / "prep" / sample_id)

    if run_on_dnmtools_array:
        dnmtools = TCGADNMToolsArrayPathway(
            sample_id,
            meth_file,
            genome,
            dnmtools_dir,
            force_recreate=force_recreate,
            print_logs=print_logs,
            shared_prep=shared_prep,
        )
        dnmtools.run()

    methylseg = TCGAMethylSegHM450KPathway(
        sample_id,
        meth_file,
        genome,
        methylseg_dir,
        force_recreate=force_recreate,
        print_logs=print_logs,
        shared_prep=shared_prep,
        train_sample_info=None,
        train_sample=None,
        train_sample_file=None,
    )
    methylseg.run()

    if clean_individual_chr_outputs:
        methylseg_out_dir = methylseg_dir / sample_id / "out" / "hm450k"
        clean_regions_dir = methylseg_out_dir / "clean_regions"
        if clean_regions_dir.exists():
            for file in clean_regions_dir.glob("segments_cleaned_chr*.bed"):
                file.unlink()

    return SegmentRunResult(
        sample_id=sample_id,
        dnmtools_array_bed=(
            dnmtools_dir / sample_id / "out" / "arraymode.dnmtools_PMDs.bed"
            if run_on_dnmtools_array
            else Path("")
        ),
        methylseg_hm450k_bed=methylseg_dir
        / sample_id
        / "out"
        / "hm450k"
        / "summary_files"
        / "segments_cleaned_PMD.bed",
        dnmtools_dir=dnmtools_dir / sample_id if run_on_dnmtools_array else Path(""),
        methylseg_dir=methylseg_dir / sample_id,
        prep_dir=None,
    )


def run_segmentation_for_manifest(
    manifest_df: pd.DataFrame,
    *,
    segmentation_root: str | Path = SEGMENTATION_OUT_DIR,
    genome: str = DEFAULT_GENOME,
    force_recreate: bool = False,
    print_logs: bool = True,
    train_sample_file: str | Path | None = None,
) -> pd.DataFrame:
    records = []
    for row in manifest_df.itertuples(index=False):
        result = run_segmentation_for_sample(
            sample_id=row.sample_id,
            meth_file=row.methylation_file,
            genome=genome,
            segmentation_root=segmentation_root,
            force_recreate=force_recreate,
            print_logs=print_logs,
        )
        records.append(
            {
                "sample_id": result.sample_id,
                "genome": genome,
                "meth_file": str(row.methylation_file),
                "dnmtools_array_bed": str(result.dnmtools_array_bed),
                "methylseg_hm450k_bed": str(result.methylseg_hm450k_bed),
                "dnmtools_dir": str(result.dnmtools_dir),
                "methylseg_dir": str(result.methylseg_dir),
                "prep_dir": str(result.prep_dir),
                "dnmtools_exists": result.dnmtools_array_bed.exists(),
                "methylseg_exists": result.methylseg_hm450k_bed.exists(),
            }
        )
    return pd.DataFrame(records)


def write_segmentation_run_manifest(
    run_manifest_df: pd.DataFrame,
    out_path: str | Path = OUT_DIR
    / "01_segmentation"
    / "segmentation_run_manifest.tsv",
) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    run_manifest_df.to_csv(out_path, sep="\t", index=False)
    return out_path


def _normalize_bed_df(df: pd.DataFrame, sample_id: str, tool: str) -> pd.DataFrame:
    normalized = df.iloc[:, :3].copy()
    normalized.columns = ["chrom", "start", "end"]
    normalized["chrom"] = normalized["chrom"].astype(str)
    normalized["start"] = pd.to_numeric(normalized["start"], errors="coerce").astype(
        "Int64"
    )
    normalized["end"] = pd.to_numeric(normalized["end"], errors="coerce").astype(
        "Int64"
    )
    normalized = normalized.dropna(subset=["start", "end"]).copy()
    normalized["start"] = normalized["start"].astype(np.int64)
    normalized["end"] = normalized["end"].astype(np.int64)
    normalized = normalized.loc[normalized["end"] > normalized["start"]].copy()
    normalized["sample_id"] = sample_id
    normalized["tool"] = tool
    normalized["length_bp"] = normalized["end"] - normalized["start"]
    return normalized.loc[:, CANONICAL_SEGMENT_COLUMNS]


def load_segments_from_bed(
    bed_path: str | Path,
    *,
    sample_id: str,
    tool: str,
) -> pd.DataFrame:
    bed_path = Path(bed_path)
    if not bed_path.exists():
        raise FileNotFoundError(f"Missing BED file: {bed_path}")
    if bed_path.stat().st_size == 0:
        return pd.DataFrame(columns=CANONICAL_SEGMENT_COLUMNS)
    try:
        df = pd.read_csv(bed_path, sep="\t", header=None)
    except EmptyDataError:
        return pd.DataFrame(columns=CANONICAL_SEGMENT_COLUMNS)
    return _normalize_bed_df(df, sample_id=sample_id, tool=tool)


def collect_all_segments(run_manifest_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for row in run_manifest_df.itertuples(index=False):
        frames.append(
            load_segments_from_bed(
                row.dnmtools_array_bed,
                sample_id=row.sample_id,
                tool="dnmtools_array",
            )
        )
        frames.append(
            load_segments_from_bed(
                row.methylseg_hm450k_bed,
                sample_id=row.sample_id,
                tool="methylseg_hm450k",
            )
        )
    if not frames:
        return pd.DataFrame(columns=CANONICAL_SEGMENT_COLUMNS)
    all_segments = pd.concat(frames, ignore_index=True)
    return all_segments.sort_values(
        ["sample_id", "tool", "chrom", "start", "end"]
    ).reset_index(drop=True)


def write_all_segments(
    all_segments_df: pd.DataFrame,
    out_path: str | Path = COLLECT_OUT_DIR / "all_segments.tsv",
) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    all_segments_df.to_csv(out_path, sep="\t", index=False)
    return out_path


def summarize_segments(all_segments_df: pd.DataFrame) -> pd.DataFrame:
    if all_segments_df.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "tool",
                "n_segments",
                "total_bp",
                "mean_length_bp",
                "median_length_bp",
                "min_length_bp",
                "max_length_bp",
            ]
        )
    summary = (
        all_segments_df.groupby(["sample_id", "tool"], sort=False)
        .agg(
            n_segments=("length_bp", "size"),
            total_bp=("length_bp", "sum"),
            mean_length_bp=("length_bp", "mean"),
            median_length_bp=("length_bp", "median"),
            min_length_bp=("length_bp", "min"),
            max_length_bp=("length_bp", "max"),
        )
        .reset_index()
    )
    return summary


def _count_overlaps(source_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if source_df.empty:
        return pd.DataFrame(
            columns=[
                "chrom",
                "start",
                "end",
                "length_bp",
                "n_target_overlaps",
                "overlap_bp",
            ]
        )
    for source_row in source_df.itertuples(index=False):
        target_chr = target_df.loc[target_df["chrom"].eq(source_row.chrom)]
        overlap_bp = 0
        n_target_overlaps = 0
        for target_row in target_chr.itertuples(index=False):
            overlap = max(
                0,
                min(source_row.end, target_row.end)
                - max(source_row.start, target_row.start),
            )
            if overlap > 0:
                n_target_overlaps += 1
                overlap_bp += overlap
        rows.append(
            {
                "chrom": source_row.chrom,
                "start": source_row.start,
                "end": source_row.end,
                "length_bp": source_row.length_bp,
                "n_target_overlaps": n_target_overlaps,
                "overlap_bp": overlap_bp,
            }
        )
    return pd.DataFrame(rows)


def compute_fragmentation_absorption(all_segments_df: pd.DataFrame) -> pd.DataFrame:
    directions = [
        ("methylseg_hm450k", "dnmtools_array", "methylseg_to_dnmtools"),
        ("dnmtools_array", "methylseg_hm450k", "dnmtools_to_methylseg"),
    ]
    records = []
    for sample_id, sample_df in all_segments_df.groupby("sample_id", sort=False):
        for source_tool, target_tool, direction in directions:
            source_df = sample_df.loc[sample_df["tool"].eq(source_tool)].copy()
            target_df = sample_df.loc[sample_df["tool"].eq(target_tool)].copy()
            overlap_df = _count_overlaps(source_df, target_df)
            if overlap_df.empty:
                records.append(
                    {
                        "sample_id": sample_id,
                        "direction": direction,
                        "source_tool": source_tool,
                        "target_tool": target_tool,
                        "n_source_segments": 0,
                        "mean_target_overlaps": np.nan,
                        "median_target_overlaps": np.nan,
                        "fraction_split_gt1": np.nan,
                        "fraction_any_overlap": np.nan,
                        "mean_overlap_bp": np.nan,
                        "mean_overlap_fraction": np.nan,
                    }
                )
                continue

            overlap_fraction = np.where(
                overlap_df["length_bp"] > 0,
                overlap_df["overlap_bp"] / overlap_df["length_bp"],
                np.nan,
            )
            records.append(
                {
                    "sample_id": sample_id,
                    "direction": direction,
                    "source_tool": source_tool,
                    "target_tool": target_tool,
                    "n_source_segments": int(len(overlap_df)),
                    "mean_target_overlaps": float(
                        overlap_df["n_target_overlaps"].mean()
                    ),
                    "median_target_overlaps": float(
                        overlap_df["n_target_overlaps"].median()
                    ),
                    "fraction_split_gt1": float(
                        (overlap_df["n_target_overlaps"] > 1).mean()
                    ),
                    "fraction_any_overlap": float(
                        (overlap_df["n_target_overlaps"] > 0).mean()
                    ),
                    "mean_overlap_bp": float(overlap_df["overlap_bp"].mean()),
                    "mean_overlap_fraction": float(np.nanmean(overlap_fraction)),
                }
            )
    return pd.DataFrame(records)


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    intervals = sorted(
        (int(start), int(end)) for start, end in intervals if int(end) > int(start)
    )
    if not intervals:
        return []
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _consensus_for_chrom(tool_df: pd.DataFrame, chrom: str) -> pd.DataFrame:
    chrom_df = tool_df.loc[tool_df["chrom"].eq(chrom)].copy()
    if chrom_df.empty:
        return pd.DataFrame(columns=["chrom", "start", "end", "n_supporting_samples"])
    breakpoints = sorted(
        set(chrom_df["start"].astype(int)).union(set(chrom_df["end"].astype(int)))
    )
    rows = []
    for start, end in zip(breakpoints[:-1], breakpoints[1:]):
        if end <= start:
            continue
        support = chrom_df.loc[
            (chrom_df["start"] < end) & (chrom_df["end"] > start),
            "sample_id",
        ].nunique()
        if support > 0:
            rows.append(
                {
                    "chrom": chrom,
                    "start": int(start),
                    "end": int(end),
                    "n_supporting_samples": int(support),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["chrom", "start", "end", "n_supporting_samples"])
    rows_df = pd.DataFrame(rows)
    merged_rows = []
    for support, support_df in rows_df.groupby("n_supporting_samples", sort=False):
        merged = _merge_intervals(
            support_df[["start", "end"]].itertuples(index=False, name=None)
        )
        for start, end in merged:
            merged_rows.append(
                {
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "n_supporting_samples": int(support),
                }
            )
    return pd.DataFrame(merged_rows)


def build_consensus_regions(all_segments_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tool, tool_df in all_segments_df.groupby("tool", sort=False):
        for chrom in sorted(tool_df["chrom"].unique()):
            chrom_consensus = _consensus_for_chrom(tool_df, chrom)
            if chrom_consensus.empty:
                continue
            chrom_consensus["tool"] = tool
            rows.append(chrom_consensus)
    if not rows:
        return pd.DataFrame(
            columns=["tool", "chrom", "start", "end", "n_supporting_samples"]
        )
    consensus_df = pd.concat(rows, ignore_index=True)
    return consensus_df.loc[
        :, ["tool", "chrom", "start", "end", "n_supporting_samples"]
    ]


def _interval_overlap_bp(
    region_df: pd.DataFrame, other_df: pd.DataFrame
) -> tuple[int, int]:
    total_bp = (
        int(region_df["length_bp"].sum())
        if "length_bp" in region_df.columns
        else int((region_df["end"] - region_df["start"]).sum())
    )
    overlap_bp = 0
    for region in region_df.itertuples(index=False):
        overlaps = other_df.loc[other_df["chrom"].eq(region.chrom)]
        for other in overlaps.itertuples(index=False):
            overlap_bp += max(
                0, min(region.end, other.end) - max(region.start, other.start)
            )
    return total_bp, overlap_bp


def compute_consensus_consistency(
    all_segments_df: pd.DataFrame,
    consensus_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if consensus_df is None:
        consensus_df = build_consensus_regions(all_segments_df)
    n_samples = max(1, all_segments_df["sample_id"].nunique())
    high_support_cutoff = max(2, int(np.ceil(n_samples / 2.0)))
    records = []
    for (sample_id, tool), sample_tool_df in all_segments_df.groupby(
        ["sample_id", "tool"], sort=False
    ):
        tool_consensus = consensus_df.loc[consensus_df["tool"].eq(tool)].copy()
        high_support_df = tool_consensus.loc[
            tool_consensus["n_supporting_samples"] >= high_support_cutoff
        ].copy()
        total_bp, overlap_bp = _interval_overlap_bp(sample_tool_df, tool_consensus)
        _, high_support_overlap_bp = _interval_overlap_bp(
            sample_tool_df, high_support_df
        )
        records.append(
            {
                "sample_id": sample_id,
                "tool": tool,
                "total_bp": total_bp,
                "consensus_overlap_bp": overlap_bp,
                "high_support_cutoff": high_support_cutoff,
                "high_support_overlap_bp": high_support_overlap_bp,
                "fraction_bp_in_consensus": (
                    (overlap_bp / total_bp) if total_bp else np.nan
                ),
                "fraction_bp_in_high_support_consensus": (
                    (high_support_overlap_bp / total_bp) if total_bp else np.nan
                ),
            }
        )
    return pd.DataFrame(records)


def _read_cached_chrom_sizes(chrom_sizes_path: Path) -> dict[str, int]:
    if not chrom_sizes_path.exists():
        return {}
    chrom_sizes = {}
    with chrom_sizes_path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                continue
            chrom, size = fields[:2]
            try:
                chrom_sizes[chrom] = int(size)
            except ValueError:
                continue
    return chrom_sizes


def _load_cached_tcga_lad_reference(genome: str, out_dir: Path) -> dict | None:
    lad_clean_path = out_dir / f"laminB1Lads.{genome}.cleaned.bed"
    if not lad_clean_path.exists():
        return None

    chrom_sizes_path = out_dir / f"{genome}.chrom.sizes"
    return {
        "genome": genome,
        "chrom_sizes": _read_cached_chrom_sizes(chrom_sizes_path),
        "chrom_sizes_path": chrom_sizes_path,
        "lad_df": load_igv_interval_file(lad_clean_path),
        "lad_clean_path": lad_clean_path,
        "occupancy_bedgraph_path": out_dir / f"laminB1Lads.{genome}.occupancy.bedGraph",
        "occupancy_bw_path": out_dir / f"laminB1Lads.{genome}.occupancy.bw",
        "signal_clean_path": out_dir / f"laminB1.{genome}.cleaned.bedGraph",
        "signal_bw_path": out_dir / f"laminB1.{genome}.signal.bw",
    }


def prepare_tcga_lad_reference(
    *,
    genome: str = DEFAULT_GENOME,
    out_dir: str | Path = ANALYSIS_OUT_DIR / "lad_reference",
    force_reference_rebuild: bool = False,
) -> dict:
    if str(LAD_SLURM_DIR) not in sys.path:
        sys.path.append(str(LAD_SLURM_DIR))
    from importlib.util import module_from_spec, spec_from_file_location

    target_path = LAD_SLURM_DIR / "01_run_lad.py"
    spec = spec_from_file_location("methylseg_lad_runner", target_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load LAD runner from {target_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    prepare_lad_reference = module.prepare_lad_reference

    out_dir = ensure_dir(out_dir)
    if not force_reference_rebuild:
        cached_reference = _load_cached_tcga_lad_reference(str(genome), out_dir)
        if cached_reference is not None:
            return cached_reference

    return prepare_lad_reference(
        genome=genome,
        reference_dir=out_dir,
        lad_interval_track_path=DEFAULT_LAD_INTERVAL_TRACK_PATH,
        laminb1_signal_track_path=DEFAULT_LAMINB1_SIGNAL_TRACK_PATH,
        liftover_script_path=DEFAULT_LIFTOVER_SCRIPT_PATH,
        hg19_to_hg38_chain=DEFAULT_HG19_TO_HG38_CHAIN,
        force=force_reference_rebuild,
    )


def compute_lad_overlap(
    all_segments_df: pd.DataFrame,
    lad_df: pd.DataFrame,
) -> pd.DataFrame:
    if lad_df.empty:
        raise ValueError("lad_df is empty")
    lad_df = lad_df.copy()
    if "chrom" in lad_df.columns and "chr" not in lad_df.columns:
        lad_df = lad_df.rename(columns={"chrom": "chr"})
    records = []
    for (sample_id, tool), sample_tool_df in all_segments_df.groupby(
        ["sample_id", "tool"], sort=False
    ):
        n_segments = len(sample_tool_df)
        total_bp = int(sample_tool_df["length_bp"].sum())
        overlapping_segments = 0
        overlap_bp = 0
        for region in sample_tool_df.itertuples(index=False):
            region_overlap = 0
            lad_chr = lad_df.loc[lad_df["chr"].astype(str).eq(region.chrom)]
            for lad_row in lad_chr.itertuples(index=False):
                region_overlap += max(
                    0,
                    min(region.end, int(lad_row.end))
                    - max(region.start, int(lad_row.start)),
                )
            if region_overlap > 0:
                overlapping_segments += 1
                overlap_bp += region_overlap
        records.append(
            {
                "sample_id": sample_id,
                "tool": tool,
                "n_segments": n_segments,
                "segments_with_lad_overlap": overlapping_segments,
                "fraction_segments_with_lad_overlap": (
                    overlapping_segments / n_segments if n_segments else np.nan
                ),
                "total_bp": total_bp,
                "lad_overlap_bp": overlap_bp,
                "fraction_bp_in_lad": (overlap_bp / total_bp) if total_bp else np.nan,
            }
        )
    return pd.DataFrame(records)


def save_analysis_table(
    df: pd.DataFrame, filename: str, *, out_dir: str | Path = ANALYSIS_OUT_DIR
) -> Path:
    out_dir = ensure_dir(out_dir)
    out_path = Path(out_dir) / filename
    df.to_csv(out_path, sep="\t", index=False)
    return out_path


def load_manifest(
    manifest_path: str | Path = OUT_DIR / "00" / "selected_samples_info.tsv",
) -> pd.DataFrame:
    return pd.read_csv(manifest_path, sep="\t")


def load_run_manifest(
    run_manifest_path: str | Path = OUT_DIR
    / "01_segmentation"
    / "segmentation_run_manifest.tsv",
) -> pd.DataFrame:
    return pd.read_csv(run_manifest_path, sep="\t")


def ensure_igv_interval_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["chrom", "start", "end"])

    out_df = df.copy().rename(
        columns={
            "chr": "chrom",
            "CpG_chrm": "chrom",
            "CpG_beg": "start",
            "CpG_end": "end",
        }
    )
    if not {"chrom", "start", "end"}.issubset(out_df.columns):
        out_df = out_df.iloc[:, :3].copy()
        out_df.columns = ["chrom", "start", "end"]
    else:
        out_df = out_df.loc[:, ["chrom", "start", "end"]].copy()

    out_df["chrom"] = out_df["chrom"].astype(str)
    out_df["start"] = pd.to_numeric(out_df["start"], errors="coerce").astype("Int64")
    out_df["end"] = pd.to_numeric(out_df["end"], errors="coerce").astype("Int64")
    out_df = out_df.dropna(subset=["chrom", "start", "end"]).copy()
    out_df["start"] = out_df["start"].astype(np.int64)
    out_df["end"] = out_df["end"].astype(np.int64)
    out_df = out_df.loc[out_df["end"] > out_df["start"]].copy()
    return out_df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)


def load_igv_interval_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["chrom", "start", "end"])
    try:
        raw_df = pd.read_csv(path, sep="\t", header=None, comment="#")
    except EmptyDataError:
        return pd.DataFrame(columns=["chrom", "start", "end"])
    if raw_df.empty:
        return pd.DataFrame(columns=["chrom", "start", "end"])

    first_values = [
        str(value).strip().lower()
        for value in raw_df.iloc[0, : min(3, raw_df.shape[1])]
    ]
    if first_values == ["chr", "start", "end"] or first_values == [
        "chrom",
        "start",
        "end",
    ]:
        raw_df = raw_df.iloc[1:].reset_index(drop=True)
    if raw_df.shape[1] < 3:
        raise ValueError(
            f"Expected at least 3 columns in {path}, found {raw_df.shape[1]}"
        )

    raw_df = raw_df.rename(columns={0: "chrom", 1: "start", 2: "end"})
    return ensure_igv_interval_df(raw_df)


def load_tcga_beta_bedgraph(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["chrom", "start", "end", "beta"])
    try:
        raw_df = pd.read_csv(path, sep="\t", header=None, comment="#")
    except EmptyDataError:
        return pd.DataFrame(columns=["chrom", "start", "end", "beta"])
    if raw_df.empty:
        return pd.DataFrame(columns=["chrom", "start", "end", "beta"])
    if raw_df.shape[1] < 4:
        raise ValueError(
            f"Expected at least 4 columns in {path}, found {raw_df.shape[1]}"
        )

    beta_df = raw_df.iloc[:, :4].copy()
    beta_df.columns = ["chrom", "start", "end", "beta"]
    beta_df["chrom"] = beta_df["chrom"].astype(str)
    beta_df["start"] = pd.to_numeric(beta_df["start"], errors="coerce").astype("Int64")
    beta_df["end"] = pd.to_numeric(beta_df["end"], errors="coerce").astype("Int64")
    beta_df["beta"] = pd.to_numeric(beta_df["beta"], errors="coerce")
    beta_df = beta_df.dropna(subset=["chrom", "start", "end", "beta"]).copy()
    beta_df["start"] = beta_df["start"].astype(np.int64)
    beta_df["end"] = beta_df["end"].astype(np.int64)
    beta_df = beta_df.loc[beta_df["end"] > beta_df["start"]].copy()
    return beta_df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)


def write_igv_bed(
    interval_df: pd.DataFrame, out_path: str | Path, track_name: str
) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    bed_df = ensure_igv_interval_df(interval_df)
    bed_df["name"] = [f"{track_name}_{idx + 1}" for idx in range(len(bed_df))]
    bed_df.loc[:, ["chrom", "start", "end", "name"]].to_csv(
        out_path, sep="\t", header=False, index=False
    )
    return out_path


def write_igv_bedgraph(beta_df: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    beta_df.loc[:, ["chrom", "start", "end", "beta"]].to_csv(
        out_path, sep="\t", header=False, index=False
    )
    return out_path


def _has_manifest_value(value: object) -> bool:
    return value is not None and str(value) not in {"", "nan", "None"}


def _row_value(row: dict | pd.Series, column: str) -> object:
    if isinstance(row, pd.Series):
        return row.get(column)
    return row.get(column)


def _candidate_tcga_beta_paths(row: dict | pd.Series) -> list[Path]:
    candidates = []
    for dir_column in ("dnmtools_dir", "methylseg_dir"):
        dir_value = _row_value(row, dir_column)
        if _has_manifest_value(dir_value):
            candidates.append(Path(str(dir_value)) / "prep" / "450k.beta")
    return candidates


def load_tcga_beta_bedgraph_for_manifest_row(row: dict | pd.Series) -> pd.DataFrame:
    sample_id = str(_row_value(row, "sample_id"))
    for beta_path in _candidate_tcga_beta_paths(row):
        if beta_path.exists():
            return load_tcga_beta_bedgraph(beta_path)

    meth_file = _row_value(row, "meth_file")
    if not _has_manifest_value(meth_file):
        raise FileNotFoundError(
            f"No 450k.beta file or meth_file path was available for {sample_id}."
        )

    beta_df = load_tcga_sample_beta_dataframe(sample_id, str(meth_file)).rename(
        columns={
            "CpG_chrm": "chrom",
            "CpG_beg": "start",
            "CpG_end": "end",
        }
    )
    return load_tcga_beta_bedgraph_from_dataframe(beta_df)


def load_tcga_beta_bedgraph_from_dataframe(beta_df: pd.DataFrame) -> pd.DataFrame:
    beta_df = beta_df.loc[:, ["chrom", "start", "end", "beta"]].copy()
    beta_df["chrom"] = beta_df["chrom"].astype(str)
    beta_df["start"] = pd.to_numeric(beta_df["start"], errors="coerce").astype("Int64")
    beta_df["end"] = pd.to_numeric(beta_df["end"], errors="coerce").astype("Int64")
    beta_df["beta"] = pd.to_numeric(beta_df["beta"], errors="coerce")
    beta_df = beta_df.dropna(subset=["chrom", "start", "end", "beta"]).copy()
    beta_df["start"] = beta_df["start"].astype(np.int64)
    beta_df["end"] = beta_df["end"].astype(np.int64)
    beta_df = beta_df.loc[beta_df["end"] > beta_df["start"]].copy()
    return beta_df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)


def load_tcga_lad_reference_for_igv(
    *,
    lad_bed_path: str | Path = TCGA_LAD_REFERENCE_BED,
    force_reference_rebuild: bool = False,
) -> pd.DataFrame:
    lad_bed_path = Path(lad_bed_path)
    if lad_bed_path.exists() and not force_reference_rebuild:
        return load_igv_interval_file(lad_bed_path)
    lad_reference = prepare_tcga_lad_reference(
        force_reference_rebuild=force_reference_rebuild
    )
    return ensure_igv_interval_df(lad_reference["lad_df"])


def load_tcga_consensus_regions(
    consensus_path: str | Path = TCGA_CONSENSUS_REGIONS_PATH,
) -> pd.DataFrame:
    consensus_path = Path(consensus_path)
    if not consensus_path.exists() or consensus_path.stat().st_size == 0:
        return pd.DataFrame(
            columns=["tool", "chrom", "start", "end", "n_supporting_samples"]
        )
    consensus_df = pd.read_csv(consensus_path, sep="\t")
    return consensus_df


def _resolve_igv_sample_rows(
    run_manifest_df: pd.DataFrame,
    sample_ids: Sequence[str] | None,
) -> pd.DataFrame:
    if sample_ids is None:
        return run_manifest_df.copy()

    sample_id_set = {str(sample_id) for sample_id in sample_ids}
    sample_rows = run_manifest_df.loc[
        run_manifest_df["sample_id"].astype(str).isin(sample_id_set)
    ].copy()
    missing = sorted(
        sample_id_set.difference(set(sample_rows["sample_id"].astype(str)))
    )
    if missing:
        raise ValueError(f"Sample IDs not found in run manifest: {', '.join(missing)}")
    return sample_rows


def _make_igv_track_record(
    sample_id: str,
    track_name: str,
    track_type: str,
    n_regions: int,
    path: str | Path,
) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "track_name": track_name,
        "track_type": track_type,
        "n_regions": str(n_regions),
        "path": str(path),
    }


def export_tcga_igv_sample(
    row: dict | pd.Series,
    out_root: str | Path = IGV_EXPORT_OUT_DIR,
    *,
    clean: bool = False,
) -> dict[str, str]:
    sample_id = str(_row_value(row, "sample_id"))
    sample_dir = Path(out_root) / sample_id
    tracks_dir = sample_dir / "tracks"

    if clean and sample_dir.exists():
        shutil.rmtree(sample_dir)
    ensure_dir(tracks_dir)

    track_records = []
    for track_name, path_column in TCGA_IGV_REGION_TRACKS.items():
        bed_path_value = _row_value(row, path_column)
        if not _has_manifest_value(bed_path_value):
            continue
        bed_path = Path(str(bed_path_value))
        if not bed_path.exists():
            print(f"Skipping missing {track_name} track for {sample_id}: {bed_path}")
            continue
        regions_df = load_igv_interval_file(bed_path)
        out_path = tracks_dir / f"{track_name}.bed"
        write_igv_bed(regions_df, out_path, track_name)
        track_records.append(
            _make_igv_track_record(
                sample_id, track_name, "bed", len(regions_df), out_path
            )
        )

    beta_df = load_tcga_beta_bedgraph_for_manifest_row(row)
    beta_path = tracks_dir / "hm450k_beta.bedGraph"
    write_igv_bedgraph(beta_df, beta_path)
    track_records.append(
        _make_igv_track_record(
            sample_id, "hm450k_beta", "bedGraph", len(beta_df), beta_path
        )
    )

    track_manifest_path = sample_dir / "track_manifest.tsv"
    pd.DataFrame(track_records).to_csv(track_manifest_path, sep="\t", index=False)
    return {
        "sample_id": sample_id,
        "sample_dir": str(sample_dir),
        "track_manifest": str(track_manifest_path),
    }


def export_tcga_shared_igv_tracks(
    out_root: str | Path = IGV_EXPORT_OUT_DIR,
    *,
    consensus_df: pd.DataFrame | None = None,
    lad_df: pd.DataFrame | None = None,
    high_support_cutoff: int | None = None,
    force_reference_rebuild: bool = False,
) -> pd.DataFrame:
    shared_tracks_dir = ensure_dir(Path(out_root) / "shared_tracks")
    shared_records = []

    if lad_df is None:
        lad_df = load_tcga_lad_reference_for_igv(
            force_reference_rebuild=force_reference_rebuild
        )
    lad_df = ensure_igv_interval_df(lad_df)
    lad_path = shared_tracks_dir / "laminB1_lad_regions.bed"
    write_igv_bed(lad_df, lad_path, "laminB1_lad")
    shared_records.append(
        _make_igv_track_record(
            "shared", "laminB1_lad_regions", "bed", len(lad_df), lad_path
        )
    )

    if consensus_df is None:
        consensus_df = load_tcga_consensus_regions()
    if consensus_df is not None and not consensus_df.empty:
        consensus_df = consensus_df.copy()
        for tool, tool_consensus_df in consensus_df.groupby("tool", sort=False):
            track_name = f"{tool}_consensus"
            out_path = shared_tracks_dir / f"{track_name}.bed"
            write_igv_bed(tool_consensus_df, out_path, track_name)
            shared_records.append(
                _make_igv_track_record(
                    "shared", track_name, "bed", len(tool_consensus_df), out_path
                )
            )

            if (
                high_support_cutoff is not None
                and "n_supporting_samples" in tool_consensus_df
            ):
                high_support_df = tool_consensus_df.loc[
                    pd.to_numeric(
                        tool_consensus_df["n_supporting_samples"], errors="coerce"
                    )
                    >= high_support_cutoff
                ].copy()
                high_support_track_name = (
                    f"{tool}_consensus_support_ge_{high_support_cutoff}"
                )
                high_support_path = shared_tracks_dir / f"{high_support_track_name}.bed"
                write_igv_bed(
                    high_support_df, high_support_path, high_support_track_name
                )
                shared_records.append(
                    _make_igv_track_record(
                        "shared",
                        high_support_track_name,
                        "bed",
                        len(high_support_df),
                        high_support_path,
                    )
                )

    shared_manifest_df = pd.DataFrame(shared_records)
    shared_manifest_df.to_csv(
        Path(out_root) / "shared_track_manifest.tsv", sep="\t", index=False
    )
    return shared_manifest_df


def export_tcga_igv_tracks(
    run_manifest_df: pd.DataFrame,
    out_root: str | Path = IGV_EXPORT_OUT_DIR,
    *,
    sample_ids: Sequence[str] | None = None,
    consensus_df: pd.DataFrame | None = None,
    lad_df: pd.DataFrame | None = None,
    clean: bool = False,
    force_reference_rebuild: bool = False,
) -> dict[str, object]:
    out_root = ensure_dir(out_root)
    sample_rows = _resolve_igv_sample_rows(run_manifest_df, sample_ids)
    summary_records = []
    for row in sample_rows.to_dict(orient="records"):
        print(f"Exporting IGV tracks for {row['sample_id']}")
        summary_records.append(export_tcga_igv_sample(row, out_root, clean=clean))

    high_support_cutoff = max(
        2, int(np.ceil(run_manifest_df["sample_id"].nunique() / 2.0))
    )
    shared_track_df = export_tcga_shared_igv_tracks(
        out_root,
        consensus_df=consensus_df,
        lad_df=lad_df,
        high_support_cutoff=high_support_cutoff,
        force_reference_rebuild=force_reference_rebuild,
    )

    summary_df = pd.DataFrame(summary_records)
    summary_path = Path(out_root) / "igv_export_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)
    return {
        "summary_path": summary_path,
        "summary_df": summary_df,
        "shared_track_manifest_path": Path(out_root) / "shared_track_manifest.tsv",
        "shared_track_df": shared_track_df,
        "out_root": Path(out_root),
    }
