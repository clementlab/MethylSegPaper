import argparse
import os
import sys
import tempfile
from pathlib import Path

matplotlib_cache_dir = Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"
matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(matplotlib_cache_dir)
env_bin = str(Path(sys.executable).resolve().parent)
path_entries = (
    os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
)
if env_bin not in path_entries:
    os.environ["PATH"] = os.pathsep.join([env_bin] + path_entries)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybedtools
import umap.umap_ as umap
from scipy.stats import fisher_exact
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TCGA_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT, TCGA_ANALYSIS_DIR):
    import_str = str(import_path)
    if import_str not in sys.path:
        sys.path.append(import_str)

from utils.tcga_segmentation_workflow import (
    load_meth_ref,
)
from repo_paths import TCGA_CLASSIFICATION_RESULTS_DIR

DEFAULT_SCRATCH_ROOT = TCGA_CLASSIFICATION_RESULTS_DIR
DEFAULT_MANIFEST_PATH = DEFAULT_SCRATCH_ROOT / "manifests" / "tcga_samples.tsv"
DEFAULT_SEGMENTATION_ROOT = DEFAULT_SCRATCH_ROOT / "segmentation"
DEFAULT_OUT_DIR = DEFAULT_SCRATCH_ROOT / "analysis"

COORD_COLS = ["CpG_chrm", "CpG_beg", "CpG_end"]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    manifest_df = pd.read_csv(manifest_path, sep="\t").copy()
    required_cols = {
        "sample_id",
        "project_id",
        "sample_type",
        "methylation_file",
        "methylseg_hm450k_bed",
    }
    missing_cols = sorted(required_cols - set(manifest_df.columns))
    if missing_cols:
        raise ValueError(f"Manifest is missing required columns: {missing_cols}")
    return manifest_df


def load_pmd_bed_df(bed_path: str | Path) -> pd.DataFrame:
    bed_path = Path(bed_path)
    if not bed_path.exists():
        raise FileNotFoundError(f"Missing PMD BED file: {bed_path}")
    if bed_path.stat().st_size == 0:
        return pd.DataFrame(columns=["chr", "start", "end"])
    df = pd.read_csv(bed_path, sep="\t", header=None, usecols=[0, 1, 2]).copy()
    df.columns = ["chr", "start", "end"]
    df["chr"] = df["chr"].astype(str)
    df["start"] = pd.to_numeric(df["start"], errors="coerce").astype("Int64")
    df["end"] = pd.to_numeric(df["end"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["start", "end"]).copy()
    df["start"] = df["start"].astype(np.int64)
    df["end"] = df["end"].astype(np.int64)
    df = df.loc[df["end"] > df["start"]].copy()
    return df.sort_values(["chr", "start", "end"]).reset_index(drop=True)


def build_candidate_pmds_and_presence(
    manifest_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pybedtools.BedTool]]:
    sample_bedtools: dict[str, pybedtools.BedTool] = {}
    interval_frames = []
    for row in manifest_df.itertuples(index=False):
        pmd_df = load_pmd_bed_df(row.methylseg_hm450k_bed)
        if pmd_df.empty:
            sample_bedtools[str(row.sample_id)] = pybedtools.BedTool(
                "", from_string=True
            )
            continue
        sample_bedtools[str(row.sample_id)] = pybedtools.BedTool.from_dataframe(pmd_df)
        interval_frames.append(pmd_df)

    if interval_frames:
        candidate_pmds = (
            pd.concat(interval_frames, ignore_index=True)
            .drop_duplicates()
            .sort_values(["chr", "start", "end"])
            .reset_index(drop=True)
        )
    else:
        candidate_pmds = pd.DataFrame(columns=["chr", "start", "end"])

    candidate_pmds["region_id"] = (
        candidate_pmds["chr"].astype(str)
        + ":"
        + candidate_pmds["start"].astype(str)
        + "-"
        + candidate_pmds["end"].astype(str)
    )

    candidate_pmds_bed = pybedtools.BedTool.from_dataframe(
        candidate_pmds[["chr", "start", "end"]]
    )
    presence_df = candidate_pmds.copy()
    for sample_id, sample_bed in tqdm(
        sample_bedtools.items(),
        total=len(sample_bedtools),
        desc="Building PMD presence matrix",
    ):
        if candidate_pmds.empty:
            presence_df[sample_id] = pd.Series(dtype=int)
            continue
        overlapping_regions = candidate_pmds_bed.intersect(
            sample_bed, u=True
        ).to_dataframe(
            disable_auto_names=True,
            names=["chr", "start", "end"],
        )
        overlapping_region_ids = (
            overlapping_regions["chr"].astype(str)
            + ":"
            + overlapping_regions["start"].astype(str)
            + "-"
            + overlapping_regions["end"].astype(str)
        )
        presence_df[sample_id] = (
            presence_df["region_id"].isin(set(overlapping_region_ids)).astype(int)
        )

    return presence_df, sample_bedtools


def fisher_region_vs_others(
    row: pd.Series, group_a_ids: list[str], group_b_ids: list[str]
) -> pd.Series:
    a = int(row[group_a_ids].sum())
    b = len(group_a_ids) - a
    c = int(row[group_b_ids].sum())
    d = len(group_b_ids) - c

    if len(group_a_ids) == 0 or len(group_b_ids) == 0:
        return pd.Series(
            {"odds_ratio": 1.0, "pval": 1.0, "a": a, "b": b, "c": c, "d": d}
        )

    odds_ratio, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
    return pd.Series(
        {"odds_ratio": odds_ratio, "pval": pval, "a": a, "b": b, "c": c, "d": d}
    )


def add_qvalues(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        df["qval"] = pd.Series(dtype=float)
        return df
    df["qval"] = multipletests(df["pval"], method="fdr_bh")[1]
    return df


def merge_significant_regions(
    significant_df: pd.DataFrame, label_col: str
) -> pd.DataFrame:
    merged_columns = [
        "chr",
        "start",
        "end",
        "region_id",
        label_col,
        "n_member_pmds",
        "member_region_ids",
        "min_qval",
        "max_odds_ratio",
    ]
    if significant_df.empty:
        return pd.DataFrame(columns=merged_columns)

    merged_parts = []
    for label_value, label_df in significant_df.groupby(label_col, sort=True):
        bed_df = label_df[["chr", "start", "end", "region_id", "qval"]].copy()
        bed_df["member_region_ids"] = bed_df["region_id"].astype(str)
        merged_df = (
            pybedtools.BedTool.from_dataframe(
                bed_df[["chr", "start", "end", "member_region_ids", "qval"]]
            )
            .sort()
            .merge(d=0, c=[4, 5], o=["collapse", "min"])
            .to_dataframe(
                disable_auto_names=True,
                names=["chr", "start", "end", "member_region_ids", "min_qval"],
            )
        )
        merged_df[label_col] = label_value
        merged_df["region_id"] = (
            merged_df["chr"].astype(str)
            + ":"
            + merged_df["start"].astype(str)
            + "-"
            + merged_df["end"].astype(str)
        )
        merged_df["member_region_ids"] = (
            merged_df["member_region_ids"].astype(str).str.split(",")
        )
        merged_df["n_member_pmds"] = merged_df["member_region_ids"].str.len()
        merged_df["max_odds_ratio"] = merged_df["member_region_ids"].apply(
            lambda member_ids: label_df.loc[
                label_df["region_id"].isin(member_ids), "odds_ratio"
            ].max()
        )
        merged_parts.append(merged_df[merged_columns])

    return pd.concat(merged_parts, ignore_index=True)


def merge_shared_regions(shared_df: pd.DataFrame) -> pd.DataFrame:
    merged_columns = [
        "chr",
        "start",
        "end",
        "region_id",
        "n_member_pmds",
        "member_region_ids",
        "min_qval",
        "max_odds_ratio",
        "n_significant_cancers",
        "significant_cancers",
    ]
    if shared_df.empty:
        return pd.DataFrame(columns=merged_columns)

    working_df = shared_df.copy()
    working_df["shared_label"] = "shared_cancer"
    merged_df = merge_significant_regions(working_df, label_col="shared_label")
    merged_df["n_significant_cancers"] = merged_df["member_region_ids"].apply(
        lambda member_ids: max(
            working_df.loc[
                working_df["region_id"].isin(member_ids), "n_significant_cancers"
            ].max(),
            0,
        )
    )
    merged_df["significant_cancers"] = merged_df["member_region_ids"].apply(
        lambda member_ids: sorted(
            {
                cancer
                for cancers in working_df.loc[
                    working_df["region_id"].isin(member_ids),
                    "significant_cancers",
                ]
                for cancer in cancers
            }
        )
    )
    merged_df = merged_df.drop(columns=["shared_label"])
    return merged_df[merged_columns]


def build_wide_methylation_table(manifest_df: pd.DataFrame) -> pd.DataFrame:
    sample_inputs = (
        manifest_df[["sample_id", "methylation_file"]]
        .drop_duplicates(subset=["sample_id"])
        .reset_index(drop=True)
    )
    meth_ref_df = load_meth_ref().copy()

    coord_df = meth_ref_df.loc[:, COORD_COLS].copy()
    coord_df["CpG_chrm"] = coord_df["CpG_chrm"].astype(str)
    if not coord_df["CpG_chrm"].str.startswith("chr").all():
        coord_df["CpG_chrm"] = "chr" + coord_df["CpG_chrm"].str.replace(
            "^chr", "", regex=True
        )
    coord_df["CpG_beg"] = pd.to_numeric(coord_df["CpG_beg"], errors="coerce")
    coord_df["CpG_end"] = pd.to_numeric(coord_df["CpG_end"], errors="coerce")

    valid_probe_mask = coord_df["CpG_beg"].notna() & coord_df["CpG_end"].notna()
    coord_df = coord_df.loc[valid_probe_mask].copy()
    coord_df["CpG_beg"] = coord_df["CpG_beg"].astype(np.int64)
    coord_df["CpG_end"] = coord_df["CpG_end"].astype(np.int64)
    coord_df = coord_df.sort_values(COORD_COLS).reset_index(drop=True)

    beta_columns = {}
    for sample_id, methylation_file in tqdm(
        sample_inputs.itertuples(index=False, name=None),
        total=len(sample_inputs),
        desc="Preloading methylation data",
    ):
        meth_data = np.load(methylation_file)
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
        beta_columns[str(sample_id)] = (
            np.where(meth_data == 255, np.nan, meth_data).astype(np.float32)[
                valid_probe_mask
            ]
            / 100.0
        )

    beta_df = pd.DataFrame(beta_columns)
    return pd.concat(
        [coord_df.reset_index(drop=True), beta_df.reset_index(drop=True)], axis=1
    )


def compute_avg_methylation_matrix(
    pmd_df: pd.DataFrame,
    meth_data_df: pd.DataFrame,
    *,
    sample_ids: list[str],
) -> pd.DataFrame:
    pmd_df = pmd_df[["chr", "start", "end", "region_id"]].copy().reset_index(drop=True)
    sample_ids = [
        sample_id for sample_id in sample_ids if sample_id in meth_data_df.columns
    ]

    pmd_bed = pybedtools.BedTool.from_dataframe(
        pmd_df[["chr", "start", "end", "region_id"]]
    )
    cpg_coord_df = meth_data_df[COORD_COLS].copy().reset_index(names="cpg_row_id")
    cpg_bed_df = cpg_coord_df.rename(
        columns={"CpG_chrm": "chr", "CpG_beg": "start", "CpG_end": "end"}
    )
    cpg_bed = pybedtools.BedTool.from_dataframe(
        cpg_bed_df[["chr", "start", "end", "cpg_row_id"]]
    )

    overlaps = pmd_bed.intersect(cpg_bed, wa=True, wb=True).to_dataframe(
        disable_auto_names=True,
        names=[
            "pmd_chr",
            "pmd_start",
            "pmd_end",
            "region_id",
            "cpg_chr",
            "cpg_start",
            "cpg_end",
            "cpg_row_id",
        ],
    )
    if overlaps.empty:
        empty_matrix = pd.DataFrame(
            index=sample_ids, columns=pmd_df["region_id"], dtype=float
        )
        empty_matrix.index.name = "sample_id"
        empty_matrix.columns.name = "region_id"
        return empty_matrix

    overlaps["cpg_row_id"] = pd.to_numeric(
        overlaps["cpg_row_id"], errors="raise"
    ).astype(int)
    meth_values_df = (
        meth_data_df.loc[:, sample_ids].copy().reset_index(names="cpg_row_id")
    )
    overlaps_with_beta = overlaps[["region_id", "cpg_row_id"]].merge(
        meth_values_df,
        on="cpg_row_id",
        how="left",
    )
    region_means_df = overlaps_with_beta.groupby("region_id", sort=False)[
        sample_ids
    ].mean()
    region_means_df = region_means_df.reindex(pmd_df["region_id"])
    avg_methylation_df = region_means_df.T
    avg_methylation_df.index.name = "sample_id"
    avg_methylation_df.columns.name = "region_id"
    return avg_methylation_df


def run_sample_umap(
    methylation_matrix_df: pd.DataFrame,
    sample_metadata_df: pd.DataFrame,
    *,
    title: str,
    plot_path: str | Path,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> pd.DataFrame:
    umap_input_df = methylation_matrix_df.copy()
    umap_input_df = umap_input_df.loc[umap_input_df.notna().any(axis=1)].copy()
    if umap_input_df.empty:
        return pd.DataFrame(
            columns=["sample_id", "UMAP1", "UMAP2", "project_id", "sample_type"]
        )

    sample_metadata = (
        sample_metadata_df[["sample_id", "project_id", "sample_type"]]
        .drop_duplicates(subset=["sample_id"])
        .set_index("sample_id")
        .loc[umap_input_df.index]
        .copy()
    )
    if umap_input_df.shape[0] < 2 or umap_input_df.shape[1] == 0:
        umap_results = sample_metadata.reset_index().copy()
        umap_results["UMAP1"] = np.nan
        umap_results["UMAP2"] = np.nan
        plt.figure(figsize=(10, 8))
        plt.title(title)
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        return umap_results[
            ["sample_id", "UMAP1", "UMAP2", "project_id", "sample_type"]
        ]

    X = SimpleImputer(strategy="mean").fit_transform(umap_input_df)
    X = StandardScaler().fit_transform(X)

    effective_neighbors = min(int(n_neighbors), len(umap_input_df) - 1)
    umap_model = umap.UMAP(
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedding = umap_model.fit_transform(X)
    umap_results = pd.DataFrame(
        {
            "sample_id": umap_input_df.index,
            "UMAP1": embedding[:, 0],
            "UMAP2": embedding[:, 1],
        }
    ).merge(sample_metadata.reset_index(), on="sample_id", how="left")

    plt.figure(figsize=(10, 8))
    for (cancer_type, sample_type), group in umap_results.groupby(
        ["project_id", "sample_type"], sort=True
    ):
        marker = "o" if sample_type == "Primary Tumor" else "X"
        plt.scatter(
            group["UMAP1"],
            group["UMAP2"],
            label=f"{cancer_type} | {sample_type}",
            s=60,
            alpha=0.8,
            marker=marker,
        )
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    return umap_results


def write_dataframe(
    df: pd.DataFrame, out_path: str | Path, *, include_index: bool = False
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    df.to_csv(out_path, sep="\t", index=include_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute TCGA PMD distinct/shared analyses and sample-level UMAP outputs."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--segmentation-root", type=Path, default=DEFAULT_SEGMENTATION_ROOT
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    manifest_df = load_manifest(args.manifest)
    for row in manifest_df.itertuples(index=False):
        pmd_path = Path(row.methylseg_hm450k_bed)
        if not pmd_path.exists():
            raise FileNotFoundError(
                f"Expected PMD summary file is missing for {row.sample_id}: {pmd_path}"
            )

    presence_df, _ = build_candidate_pmds_and_presence(manifest_df)
    sample_ids = manifest_df["sample_id"].astype(str).tolist()
    tumor_samples = manifest_df.loc[
        manifest_df["sample_type"].astype(str) != "Solid Tissue Normal"
    ].copy()
    normal_samples = manifest_df.loc[
        manifest_df["sample_type"].astype(str) == "Solid Tissue Normal"
    ].copy()
    tumor_sample_ids = [
        sample_id
        for sample_id in tumor_samples["sample_id"].astype(str)
        if sample_id in sample_ids
    ]
    normal_sample_ids = [
        sample_id
        for sample_id in normal_samples["sample_id"].astype(str)
        if sample_id in sample_ids
    ]
    cancer_to_tumor_sample_ids = {
        str(cancer): group["sample_id"].astype(str).tolist()
        for cancer, group in tumor_samples.groupby("project_id", sort=True)
    }
    all_cancer_types = sorted(cancer_to_tumor_sample_ids.keys())

    tumor_vs_normal_presence_df = presence_df[
        ["chr", "start", "end", "region_id"] + tumor_sample_ids + normal_sample_ids
    ].copy()
    tumor_vs_normal_stats_df = tumor_vs_normal_presence_df.apply(
        lambda row: fisher_region_vs_others(row, tumor_sample_ids, normal_sample_ids),
        axis=1,
    )
    tumor_vs_normal_fisher_df = pd.concat(
        [
            tumor_vs_normal_presence_df[
                ["chr", "start", "end", "region_id"]
            ].reset_index(drop=True),
            pd.Series(
                ["tumor_vs_normal"] * len(tumor_vs_normal_presence_df),
                name="comparison",
            ),
            tumor_vs_normal_stats_df.reset_index(drop=True),
        ],
        axis=1,
    )
    tumor_vs_normal_fisher_df = add_qvalues(tumor_vs_normal_fisher_df)
    tumor_vs_normal_pmds = tumor_vs_normal_fisher_df[
        (tumor_vs_normal_fisher_df["qval"] < 0.05)
        & (tumor_vs_normal_fisher_df["odds_ratio"] > 1)
    ].copy()

    tumor_vs_normal_region_ids = tumor_vs_normal_pmds["region_id"].tolist()
    tumor_vs_normal_presence_subset_df = presence_df.loc[
        presence_df["region_id"].isin(tumor_vs_normal_region_ids),
        ["chr", "start", "end", "region_id"] + tumor_sample_ids + normal_sample_ids,
    ].copy()

    cancer_vs_normal_fisher_results = []
    for cancer in all_cancer_types:
        cancer_sample_ids = cancer_to_tumor_sample_ids[cancer]
        stats_df = tumor_vs_normal_presence_subset_df.apply(
            lambda row: fisher_region_vs_others(
                row, cancer_sample_ids, normal_sample_ids
            ),
            axis=1,
        )
        cancer_vs_normal_fisher_results.append(
            pd.concat(
                [
                    tumor_vs_normal_presence_subset_df[
                        ["chr", "start", "end", "region_id"]
                    ].reset_index(drop=True),
                    pd.Series(
                        [cancer] * len(tumor_vs_normal_presence_subset_df),
                        name="cancer",
                    ),
                    stats_df.reset_index(drop=True),
                ],
                axis=1,
            )
        )
    cancer_vs_normal_fisher_df = pd.concat(
        cancer_vs_normal_fisher_results, ignore_index=True
    )
    cancer_vs_normal_fisher_df = add_qvalues(cancer_vs_normal_fisher_df)

    tumor_only_presence_subset_df = tumor_vs_normal_presence_subset_df[
        ["chr", "start", "end", "region_id"] + tumor_sample_ids
    ].copy()
    cancer_type_distinct_fisher_results = []
    for cancer in all_cancer_types:
        cancer_sample_ids = cancer_to_tumor_sample_ids[cancer]
        other_tumor_sample_ids = [
            sample_id
            for sample_id in tumor_sample_ids
            if sample_id not in cancer_sample_ids
        ]
        stats_df = tumor_only_presence_subset_df.apply(
            lambda row: fisher_region_vs_others(
                row, cancer_sample_ids, other_tumor_sample_ids
            ),
            axis=1,
        )
        cancer_type_distinct_fisher_results.append(
            pd.concat(
                [
                    tumor_only_presence_subset_df[
                        ["chr", "start", "end", "region_id"]
                    ].reset_index(drop=True),
                    pd.Series(
                        [cancer] * len(tumor_only_presence_subset_df), name="cancer"
                    ),
                    stats_df.reset_index(drop=True),
                ],
                axis=1,
            )
        )
    cancer_type_distinct_fisher_df = pd.concat(
        cancer_type_distinct_fisher_results, ignore_index=True
    )
    cancer_type_distinct_fisher_df = add_qvalues(cancer_type_distinct_fisher_df)

    cancer_vs_normal_significant_df = cancer_vs_normal_fisher_df[
        (cancer_vs_normal_fisher_df["qval"] < 0.05)
        & (cancer_vs_normal_fisher_df["odds_ratio"] > 1)
    ].copy()
    cancer_type_distinct_pmds = cancer_type_distinct_fisher_df.merge(
        cancer_vs_normal_significant_df[["region_id", "cancer"]],
        on=["region_id", "cancer"],
        how="inner",
    )
    cancer_type_distinct_pmds = cancer_type_distinct_pmds[
        (cancer_type_distinct_pmds["qval"] < 0.05)
        & (cancer_type_distinct_pmds["odds_ratio"] > 1)
    ].copy()
    cancer_type_distinct_summary = cancer_type_distinct_pmds.sort_values(
        ["cancer", "qval", "odds_ratio"], ascending=[True, True, False]
    ).reset_index(drop=True)
    cancer_type_distinct_merged_pmds = merge_significant_regions(
        cancer_type_distinct_summary,
        label_col="cancer",
    )
    if cancer_type_distinct_merged_pmds.empty:
        cancer_type_distinct_merged_cross_cancer_pmds = pd.DataFrame(
            columns=["chr", "start", "end", "member_region_ids", "region_id"]
        )
    else:
        cancer_type_distinct_merged_cross_cancer_pmds = (
            pybedtools.BedTool.from_dataframe(
                cancer_type_distinct_merged_pmds[["chr", "start", "end", "region_id"]]
            )
            .sort()
            .merge(d=1_000, c=4, o="collapse")
            .to_dataframe(
                disable_auto_names=True,
                names=["chr", "start", "end", "member_region_ids"],
            )
        )
        cancer_type_distinct_merged_cross_cancer_pmds["region_id"] = (
            cancer_type_distinct_merged_cross_cancer_pmds["chr"].astype(str)
            + ":"
            + cancer_type_distinct_merged_cross_cancer_pmds["start"].astype(str)
            + "-"
            + cancer_type_distinct_merged_cross_cancer_pmds["end"].astype(str)
        )
        cancer_type_distinct_merged_cross_cancer_pmds["member_region_ids"] = (
            cancer_type_distinct_merged_cross_cancer_pmds["member_region_ids"]
            .astype(str)
            .str.split(",")
        )

    shared_cancer_summary = (
        cancer_vs_normal_significant_df.groupby("region_id", sort=False)
        .agg(
            chr=("chr", "first"),
            start=("start", "first"),
            end=("end", "first"),
            n_significant_cancers=("cancer", "nunique"),
            significant_cancers=("cancer", lambda x: sorted(pd.unique(x))),
            qval=("qval", "min"),
            odds_ratio=("odds_ratio", "max"),
        )
        .reset_index()
    )
    shared_cancer_pmds = shared_cancer_summary[
        shared_cancer_summary["n_significant_cancers"] >= 3
    ].copy()
    shared_cancer_merged_pmds = merge_shared_regions(shared_cancer_pmds)

    meth_data_df = build_wide_methylation_table(manifest_df)
    sample_id_order = manifest_df["sample_id"].astype(str).tolist()
    cancer_type_distinct_avg_methylation_df = compute_avg_methylation_matrix(
        cancer_type_distinct_merged_cross_cancer_pmds,
        meth_data_df,
        sample_ids=sample_id_order,
    )
    shared_cancer_avg_methylation_df = compute_avg_methylation_matrix(
        shared_cancer_merged_pmds,
        meth_data_df,
        sample_ids=sample_id_order,
    )

    cancer_type_distinct_umap_results = run_sample_umap(
        cancer_type_distinct_avg_methylation_df,
        manifest_df,
        title="UMAP of Average Methylation in Cancer-Type-Distinct PMDs",
        plot_path=args.out_dir / "cancer_type_distinct_umap.png",
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
    )
    shared_cancer_umap_results = run_sample_umap(
        shared_cancer_avg_methylation_df,
        manifest_df,
        title="UMAP of Average Methylation in Shared Cancer PMDs",
        plot_path=args.out_dir / "shared_cancer_umap.png",
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
    )

    write_dataframe(
        tumor_vs_normal_fisher_df, args.out_dir / "tumor_vs_normal_fisher_df.tsv"
    )
    write_dataframe(tumor_vs_normal_pmds, args.out_dir / "tumor_vs_normal_pmds.tsv")
    write_dataframe(
        cancer_vs_normal_fisher_df, args.out_dir / "cancer_vs_normal_fisher_df.tsv"
    )
    write_dataframe(
        cancer_type_distinct_pmds, args.out_dir / "cancer_type_distinct_pmds.tsv"
    )
    write_dataframe(
        cancer_type_distinct_merged_pmds,
        args.out_dir / "cancer_type_distinct_merged_pmds.tsv",
    )
    write_dataframe(
        cancer_type_distinct_merged_cross_cancer_pmds,
        args.out_dir / "cancer_type_distinct_merged_cross_cancer_pmds.tsv",
    )
    write_dataframe(shared_cancer_pmds, args.out_dir / "shared_cancer_pmds.tsv")
    write_dataframe(
        shared_cancer_merged_pmds, args.out_dir / "shared_cancer_merged_pmds.tsv"
    )
    write_dataframe(
        cancer_type_distinct_avg_methylation_df,
        args.out_dir / "cancer_type_distinct_avg_methylation.tsv",
        include_index=True,
    )
    write_dataframe(
        shared_cancer_avg_methylation_df,
        args.out_dir / "shared_cancer_avg_methylation.tsv",
        include_index=True,
    )
    write_dataframe(
        cancer_type_distinct_umap_results,
        args.out_dir / "cancer_type_distinct_umap.tsv",
    )
    write_dataframe(
        shared_cancer_umap_results,
        args.out_dir / "shared_cancer_umap.tsv",
    )
    print(f"Wrote TCGA cancer analysis outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
