from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm.auto import tqdm

from repo_paths import REFERENCE_DATA_DIR, TCGA_CLASSIFICATION_RESULTS_DIR

TCGA_SAMPLES = REFERENCE_DATA_DIR / "runAll.sh.samples"
CGI_BED = REFERENCE_DATA_DIR / "cgi.bed"
PMD_PATH = TCGA_CLASSIFICATION_RESULTS_DIR / "segmentation" / "methylseg"
HG38_PATH = REFERENCE_DATA_DIR / "hg38.chrom.sizes"
CENTROMERE_PATH = REFERENCE_DATA_DIR / "centromere.bed"
GAP_PATH = REFERENCE_DATA_DIR / "gap.bed"
METH_REF = REFERENCE_DATA_DIR / "parse450K.pl.order.lookup"
RANDOM_SEED = 42
MAX_SPLIT_WORKERS = 50
_RANDOM_REGION_REFERENCE_CACHE = None
_FEATURE_CLASSIFICATION_WORKER_STATE = {}
CANNONICAL_CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
CLASSIFICATION_METRICS = ['balanced_accuracy', 'average_precision', 'macro_f1', 'mcc', 'roc_auc']
FEATURE_SET_ORDER = ['PMD', 'Random PMD', 'CGI', 'Random long', 'Random short', 'Always cancer']
FEATURE_SET_PALETTE = {
    'PMD': '#0b5394',
    'Random PMD': '#3d85c6',
    'CGI': '#38761d',
    'Random long': '#9c6ade',
    'Random short': '#e69138',
    'Always cancer': '#b7b7b7',
}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data: dict, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_table(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, sep='	', index=index)
    return path


def parse_feature_counts(value: str | Sequence[int] | None) -> tuple[int, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(',') if piece.strip()]
        parsed = [int(piece) for piece in pieces]
    else:
        parsed = [int(piece) for piece in value]
    parsed = sorted({count for count in parsed if int(count) > 0})
    if not parsed:
        raise ValueError('At least one positive feature count is required.')
    return tuple(parsed)


def cohort_output_dir(out_root: str | Path, cohort_id: str, feature_count: int) -> Path:
    return Path(out_root) / str(cohort_id) / f'n_features_{int(feature_count)}'


def make_sample_types(samples_info: pd.DataFrame) -> dict[str, str]:
    return {
        str(sample): ('Normal' if str(sample_type) == 'Solid Tissue Normal' else 'Tumor')
        for sample, sample_type in zip(samples_info['sample'], samples_info['sample_type'])
    }


def summarize_cohort_samples(samples_info: pd.DataFrame) -> dict[str, int]:
    sample_types = make_sample_types(samples_info)
    labels = pd.Series(sample_types, dtype='object')
    return {
        'n_samples': int(len(labels)),
        'n_tumor_samples': int(labels.eq('Tumor').sum()),
        'n_normal_samples': int(labels.eq('Normal').sum()),
    }


def build_per_cancer_cohort_manifest(
    samples_info: pd.DataFrame,
    *,
    min_splits: int,
    include_cohorts: Sequence[str] | None = None,
    exclude_cohorts: Sequence[str] | None = None,
) -> pd.DataFrame:
    cohort_df = samples_info.copy()
    cohort_df['cohort_id'] = cohort_df['project_id'].astype(str)
    cohort_df['binary_label'] = np.where(
        cohort_df['sample_type'].astype(str).eq('Solid Tissue Normal'),
        'Normal',
        'Tumor',
    )
    counts = (
        cohort_df.groupby(['cohort_id', 'binary_label'], as_index=False)
        .size()
        .pivot(index='cohort_id', columns='binary_label', values='size')
        .fillna(0)
        .reset_index()
    )
    if 'Tumor' not in counts.columns:
        counts['Tumor'] = 0
    if 'Normal' not in counts.columns:
        counts['Normal'] = 0
    counts['n_samples'] = counts['Tumor'].astype(int) + counts['Normal'].astype(int)
    counts['n_tumor_samples'] = counts['Tumor'].astype(int)
    counts['n_normal_samples'] = counts['Normal'].astype(int)
    counts = counts.rename(columns={'cohort_id': 'cohort_id'})
    counts = counts[['cohort_id', 'n_samples', 'n_tumor_samples', 'n_normal_samples']].copy()
    counts['is_eligible'] = (
        counts['n_tumor_samples'].ge(int(min_splits))
        & counts['n_normal_samples'].ge(int(min_splits))
    )

    if include_cohorts:
        include = {str(value) for value in include_cohorts}
        counts = counts.loc[counts['cohort_id'].isin(include)].copy()
    if exclude_cohorts:
        exclude = {str(value) for value in exclude_cohorts}
        counts = counts.loc[~counts['cohort_id'].isin(exclude)].copy()

    counts = counts.loc[counts['is_eligible']].drop(columns=['is_eligible']).copy()
    counts = counts.sort_values('cohort_id').reset_index(drop=True)
    if counts.empty:
        raise ValueError('No eligible per-cancer cohorts were found for the requested settings.')
    return counts


def save_feature_classification_outputs(
    results: pd.DataFrame,
    sampled_feature_regions: pd.DataFrame,
    *,
    out_dir: str | Path,
    run_config: dict,
    fold_predictions: pd.DataFrame | None = None,
) -> dict[str, Path]:
    out_dir = ensure_dir(out_dir)
    outputs = {
        'classification_results': write_table(results, out_dir / 'classification_results.tsv'),
        'sampled_feature_regions': write_table(sampled_feature_regions, out_dir / 'sampled_feature_regions.tsv'),
        'run_config': write_json(run_config, out_dir / 'run_config.json'),
    }
    if fold_predictions is not None:
        outputs['fold_predictions'] = write_table(fold_predictions, out_dir / 'fold_predictions.tsv')
    return outputs


def load_saved_feature_classification_task(
    task_dir: str | Path,
    *,
    include_fold_predictions: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    task_dir = Path(task_dir)
    results_path = task_dir / 'classification_results.tsv'
    sampled_path = task_dir / 'sampled_feature_regions.tsv'
    fold_predictions_path = task_dir / 'fold_predictions.tsv'
    config_path = task_dir / 'run_config.json'
    if not results_path.exists():
        raise FileNotFoundError(f'Missing classification results: {results_path}')
    if not sampled_path.exists():
        raise FileNotFoundError(f'Missing sampled feature regions: {sampled_path}')
    if not config_path.exists():
        raise FileNotFoundError(f'Missing run config: {config_path}')
    results = pd.read_csv(results_path, sep='	')
    sampled_regions = pd.read_csv(sampled_path, sep='	')
    run_config = json.loads(config_path.read_text())
    if not include_fold_predictions:
        return results, sampled_regions, run_config
    if not fold_predictions_path.exists():
        raise FileNotFoundError(f'Missing fold predictions: {fold_predictions_path}')
    fold_predictions = pd.read_csv(fold_predictions_path, sep='	')
    return results, sampled_regions, fold_predictions, run_config


def collect_saved_feature_classification_outputs(
    out_root: str | Path,
    cohort_id: str,
    *,
    include_fold_predictions: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort_root = Path(out_root) / str(cohort_id)
    task_dirs = sorted(
        path for path in cohort_root.glob('n_features_*')
        if path.is_dir() and (path / 'classification_results.tsv').exists()
    )
    if not task_dirs:
        raise FileNotFoundError(f'No saved feature-classification outputs were found under {cohort_root}.')

    results_frames = []
    sampled_frames = []
    fold_prediction_frames = []
    task_records = []
    for task_dir in task_dirs:
        if include_fold_predictions:
            results_df, sampled_df, fold_predictions_df, run_config = load_saved_feature_classification_task(
                task_dir,
                include_fold_predictions=True,
            )
            fold_prediction_frames.append(fold_predictions_df)
        else:
            results_df, sampled_df, run_config = load_saved_feature_classification_task(task_dir)
        results_frames.append(results_df)
        sampled_frames.append(sampled_df)
        task_record = {
            'task_dir': str(task_dir),
            'feature_count': int(run_config['feature_count']),
            'cohort_id': str(run_config['cohort_id']),
            'n_results_rows': int(len(results_df)),
            'n_region_rows': int(len(sampled_df)),
        }
        if include_fold_predictions:
            task_record['n_prediction_rows'] = int(len(fold_predictions_df))
        task_records.append(task_record)

    results = pd.concat(results_frames, ignore_index=True).sort_values(
        ['n_features', 'feature_set', 'split', 'feature_draw']
    ).reset_index(drop=True)
    sampled_regions = pd.concat(sampled_frames, ignore_index=True).sort_values(
        ['n_features', 'feature_set', 'split', 'feature_draw', 'region_index']
    ).reset_index(drop=True)
    task_summary = pd.DataFrame(task_records).sort_values(['feature_count']).reset_index(drop=True)
    if not include_fold_predictions:
        return results, sampled_regions, task_summary
    fold_predictions = pd.concat(fold_prediction_frames, ignore_index=True).sort_values(
        ['n_features', 'split', 'sample_id', 'feature_draw', 'feature_set']
    ).reset_index(drop=True)
    return results, sampled_regions, fold_predictions, task_summary


def run_cohort_feature_classification(
    *,
    cohort_id: str,
    feature_count: int,
    n_splits: int,
    n_repeats: int,
    n_feature_draws: int,
    random_seed: int,
    max_split_workers: int | None,
    out_root: str | Path,
    exclude_top_normal_shared_pmds: bool = False,
    min_cpgs_for_random_regions: int = 1,
) -> dict[str, Path]:
    samples_info = load_tcga_samples(str(cohort_id)).reset_index(drop=True)
    if samples_info.empty:
        raise ValueError(f'No TCGA samples were found for cohort {cohort_id}.')

    sample_summary = summarize_cohort_samples(samples_info)
    if sample_summary['n_tumor_samples'] < int(n_splits) or sample_summary['n_normal_samples'] < int(n_splits):
        raise ValueError(
            f'Cohort {cohort_id} does not have enough tumor and normal samples for n_splits={int(n_splits)}.'
        )

    meth_data = load_methylation_data(samples_info)
    sample_types = make_sample_types(samples_info)
    pmds_per_sample = load_pmds_per_sample(samples_info)
    cgis = load_cgis()
    results, sampled_feature_regions, fold_predictions = run_feature_classification(
        meth_data=meth_data,
        sample_types=sample_types,
        pmds_per_sample=pmds_per_sample,
        cgis=cgis,
        feature_counts=[int(feature_count)],
        n_splits=int(n_splits),
        n_repeats=int(n_repeats),
        n_feature_draws=int(n_feature_draws),
        random_seed=int(random_seed),
        max_split_workers=max_split_workers,
        return_fold_predictions=True,
        exclude_top_normal_shared_pmds=bool(exclude_top_normal_shared_pmds),
        min_cpgs_for_random_regions=int(min_cpgs_for_random_regions),
    )

    for frame in (results, sampled_feature_regions, fold_predictions):
        frame['cohort_id'] = str(cohort_id)
        for key, value in sample_summary.items():
            frame[key] = int(value)

    run_config = {
        'cohort_id': str(cohort_id),
        'feature_count': int(feature_count),
        'n_splits': int(n_splits),
        'n_repeats': int(n_repeats),
        'n_feature_draws': int(n_feature_draws),
        'random_seed': int(random_seed),
        'max_split_workers': int(MAX_SPLIT_WORKERS if max_split_workers is None else max_split_workers),
        'exclude_top_normal_shared_pmds': bool(exclude_top_normal_shared_pmds),
        'min_cpgs_for_random_regions': max(1, int(min_cpgs_for_random_regions)),
        'out_root': str(Path(out_root)),
        'sample_summary': sample_summary,
    }
    return save_feature_classification_outputs(
        results,
        sampled_feature_regions,
        out_dir=cohort_output_dir(out_root, cohort_id, feature_count),
        run_config=run_config,
        fold_predictions=fold_predictions,
    )
def load_tcga_samples(cancer_type = None):
    """
    Load TCGA samples for a given cancer type.

    Parameters:
    cancer_type (str): The type of cancer to load samples for. If None, load all samples.

    Returns:
    pd.DataFrame: A DataFrame containing the TCGA samples.
    """

    samples_info_path = pd.read_csv(TCGA_SAMPLES, sep='\t')

    if cancer_type:
        tcga_samples = samples_info_path[samples_info_path['project_id'] == cancer_type]
    else:
        tcga_samples = samples_info_path

    return tcga_samples

def load_methylation_data(samples_info):
    """
    Load methylation data for each sample.

    Parameters
    ----------
    samples_info : pd.DataFrame
        Must contain:
        - sample
        - methylation_file

    Returns
    -------
    pd.DataFrame
        CpG coordinates followed by one methylation column per sample.
    """

    meth_ref = pd.read_csv(METH_REF, sep="\t")

    required_ref_columns = ["CpG_chrm", "CpG_beg", "CpG_end"]
    missing_ref_columns = [
        column
        for column in required_ref_columns
        if column not in meth_ref.columns
    ]

    if missing_ref_columns:
        raise ValueError(
            f"Methylation reference is missing columns: {missing_ref_columns}"
        )

    meth_vals = {}

    for _, row in samples_info.iterrows():
        sample_id = row["sample"]
        meth_file = row["methylation_file"]

        meth = np.load(meth_file).astype(float)

        # 255 represents a missing methylation value.
        meth[meth == 255] = np.nan

        # Ensure each methylation array matches the CpG reference.
        if len(meth) != len(meth_ref):
            raise ValueError(
                f"Length mismatch for sample {sample_id}: "
                f"{len(meth)} methylation values but "
                f"{len(meth_ref)} CpGs in METH_REF."
            )

        meth_vals[sample_id] = meth

    meth_data_df = pd.DataFrame(meth_vals)

    coordinate_df = (
        meth_ref[required_ref_columns]
        .reset_index(drop=True)
        .copy()
    )

    meth_data_df = meth_data_df.reset_index(drop=True)

    result = pd.concat(
        [coordinate_df, meth_data_df],
        axis=1,
    )

    return result


def build_measured_cpg_anchor_table(meth_data: pd.DataFrame) -> pd.DataFrame:
    """
    Build a reusable HM450K CpG anchor table from the measured methylation data.

    Parameters
    ----------
    meth_data : pd.DataFrame
        CpG coordinate table returned by load_methylation_data().

    Returns
    -------
    pd.DataFrame
        Canonical-chromosome CpG anchors with one anchor position per measured CpG.
    """

    required_columns = ['CpG_chrm', 'CpG_beg', 'CpG_end']
    missing_columns = [
        column for column in required_columns
        if column not in meth_data.columns
    ]
    if missing_columns:
        raise ValueError(
            f'meth_data is missing CpG coordinate columns required for anchor sampling: {missing_columns}'
        )

    anchors = meth_data[required_columns].copy()
    anchors['chrom'] = anchors['CpG_chrm'].astype(str)
    anchors['start'] = pd.to_numeric(anchors['CpG_beg'], errors='coerce')
    anchors['end'] = pd.to_numeric(anchors['CpG_end'], errors='coerce')
    anchors = anchors.dropna(subset=['start', 'end']).copy()
    anchors['start'] = anchors['start'].astype(int)
    anchors['end'] = anchors['end'].astype(int)
    anchors = anchors.loc[anchors['end'] > anchors['start']].copy()
    anchors = anchors.loc[anchors['chrom'].isin(CANNONICAL_CHROMOSOMES)].copy()
    anchors['anchor_pos'] = ((anchors['start'] + anchors['end']) // 2).astype(int)
    anchors['length'] = anchors['end'] - anchors['start']
    anchors = anchors.drop_duplicates(subset=['chrom', 'anchor_pos']).reset_index(drop=True)
    return anchors[['chrom', 'start', 'end', 'anchor_pos', 'length']].copy()


def _build_cpg_position_lookup(cpg_anchor_df: pd.DataFrame) -> dict[str, np.ndarray]:
    anchors = cpg_anchor_df.copy()
    if anchors.empty:
        return {}

    positions_by_chrom = {}
    for chrom, chrom_df in anchors.groupby('chrom', sort=False):
        positions = np.sort(chrom_df['anchor_pos'].to_numpy(dtype=int))
        if positions.size:
            positions_by_chrom[str(chrom)] = positions
    return positions_by_chrom


def _count_cpg_positions_in_region(
    chrom: str,
    start: int,
    end: int,
    cpg_positions_by_chrom: dict[str, np.ndarray] | None,
) -> int:
    if cpg_positions_by_chrom is None:
        return 0

    positions = cpg_positions_by_chrom.get(str(chrom))
    if positions is None or positions.size == 0:
        return 0

    start = int(start)
    end = int(end)
    left = np.searchsorted(positions, start, side='left')
    right = np.searchsorted(positions, end, side='left')
    return int(right - left)


def _count_cpg_positions_per_interval(
    interval_df: pd.DataFrame,
    cpg_positions_by_chrom: dict[str, np.ndarray] | None,
) -> np.ndarray:
    if interval_df.empty:
        return np.zeros(0, dtype=int)
    if cpg_positions_by_chrom is None:
        return np.zeros(len(interval_df), dtype=int)

    intervals = interval_df.reset_index(drop=True)
    counts = np.zeros(len(intervals), dtype=int)
    for chrom, chrom_idx in intervals.groupby('chrom', sort=False).groups.items():
        positions = cpg_positions_by_chrom.get(str(chrom))
        if positions is None or positions.size == 0:
            continue

        interval_index = np.asarray(list(chrom_idx), dtype=int)
        starts = intervals.loc[interval_index, 'start'].to_numpy(dtype=int)
        ends = intervals.loc[interval_index, 'end'].to_numpy(dtype=int)
        left = np.searchsorted(positions, starts, side='left')
        right = np.searchsorted(positions, ends, side='left')
        counts[interval_index] = right - left
    return counts


def _filter_interval_pool_for_requested_length(
    candidate_pool: pd.DataFrame,
    requested_length: int,
    cpg_positions_by_chrom: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    filtered = candidate_pool.loc[
        candidate_pool['length'] >= int(requested_length)
    ].reset_index(drop=True)
    if filtered.empty or cpg_positions_by_chrom is None:
        return filtered

    filtered = filtered.copy()
    filtered['anchor_count'] = _count_cpg_positions_per_interval(
        filtered,
        cpg_positions_by_chrom,
    )
    return filtered.loc[filtered['anchor_count'] > 0].reset_index(drop=True)


def _sample_anchor_position_from_interval(
    chosen_interval: pd.Series,
    cpg_positions_by_chrom: dict[str, np.ndarray],
    rng: np.random.RandomState,
) -> int | None:
    positions = cpg_positions_by_chrom.get(str(chosen_interval['chrom']))
    if positions is None or positions.size == 0:
        return None

    start = int(chosen_interval['start'])
    end = int(chosen_interval['end'])
    left = np.searchsorted(positions, start, side='left')
    right = np.searchsorted(positions, end, side='left')
    if right <= left:
        return None
    return int(positions[int(rng.randint(left, right))])


def _place_anchor_random_region(
    chosen_interval: pd.Series,
    anchor_pos: int,
    requested_length: int,
    rng: np.random.RandomState,
    cpg_positions_by_chrom: dict[str, np.ndarray],
    min_cpgs_for_random_regions: int,
    max_start_attempts: int = 64,
) -> dict[str, int | str] | None:
    interval_start = int(chosen_interval['start'])
    interval_end = int(chosen_interval['end'])
    requested_length = int(requested_length)
    anchor_pos = int(anchor_pos)
    min_cpgs_for_random_regions = max(1, int(min_cpgs_for_random_regions))

    min_start = max(interval_start, anchor_pos - requested_length + 1)
    max_start = min(interval_end - requested_length, anchor_pos)
    if max_start < min_start:
        return None

    if min_cpgs_for_random_regions <= 1:
        chosen_start = min_start if max_start == min_start else int(
            rng.randint(min_start, max_start + 1)
        )
        return {
            'chrom': str(chosen_interval['chrom']),
            'start': chosen_start,
            'end': chosen_start + requested_length,
            'length': requested_length,
        }

    candidate_starts = []
    for _ in range(max(1, int(max_start_attempts))):
        if max_start == min_start:
            candidate_starts.append(min_start)
        else:
            candidate_starts.append(int(rng.randint(min_start, max_start + 1)))

    candidate_starts.extend(
        [
            min_start,
            max_start,
            min(max_start, max(min_start, anchor_pos - requested_length // 2)),
        ]
    )

    seen_starts = set()
    for start in candidate_starts:
        start = int(start)
        if start in seen_starts:
            continue
        seen_starts.add(start)
        end = start + requested_length
        cpg_count = _count_cpg_positions_in_region(
            str(chosen_interval['chrom']),
            start,
            end,
            cpg_positions_by_chrom,
        )
        if cpg_count >= min_cpgs_for_random_regions:
            return {
                'chrom': str(chosen_interval['chrom']),
                'start': start,
                'end': end,
                'length': requested_length,
            }

    return None

def load_pmds_per_sample(samples_info):
    """
    Load PMDs (Partially Methylated Domains) for each sample.

    Parameters:
    samples_info (pd.DataFrame): A DataFrame containing sample information.

    Returns:
    dict: A dictionary where keys are sample IDs and values are DataFrames of PMDs.
    """

    pmds_per_sample = {}

    for _, row in samples_info.iterrows():
        sample_id = row['sample']
        pmds_file_path = PMD_PATH / sample_id / 'out' / 'hm450k' / 'summary_files' / 'segments_cleaned_PMD.bed'
        pmds_data = pd.read_csv(
            pmds_file_path,
            sep='\t',
            header=None,
            names=['chrom', 'start', 'end', 'type'],
        )
        pmds_per_sample[sample_id] = pmds_data

    return pmds_per_sample

def _normalize_regions(region_df):
    if region_df is None or region_df.empty:
        return pd.DataFrame(columns=['chrom', 'start', 'end', 'length'])

    if 'chrom' in region_df.columns:
        chrom_col = 'chrom'
    elif 'chr' in region_df.columns:
        chrom_col = 'chr'
    elif "CpG_chrm" in region_df.columns:
        chrom_col = "CpG_chrm"
    else:
        raise ValueError("Region dataframe must contain either a 'chrom' or 'chr' column.")

    normalized = region_df.rename(columns={chrom_col: 'chrom'})[['chrom', 'start', 'end']].copy()
    normalized['chrom'] = normalized['chrom'].astype(str)
    normalized['start'] = pd.to_numeric(normalized['start'], errors='coerce')
    normalized['end'] = pd.to_numeric(normalized['end'], errors='coerce')
    normalized = normalized.dropna(subset=['start', 'end']).copy()
    normalized['start'] = normalized['start'].astype(int)
    normalized['end'] = normalized['end'].astype(int)
    normalized['length'] = normalized['end'] - normalized['start']
    return normalized.reset_index(drop=True)

def _merge_intervals(interval_df):
    intervals = _normalize_regions(interval_df)
    if intervals.empty:
        return intervals

    intervals = intervals.sort_values(['chrom', 'start', 'end']).reset_index(drop=True)
    merged_rows = []

    for chrom, chrom_df in intervals.groupby('chrom', sort=False):
        current_start = int(chrom_df.iloc[0]['start'])
        current_end = int(chrom_df.iloc[0]['end'])

        for row in chrom_df.iloc[1:].itertuples(index=False):
            start = int(row.start)
            end = int(row.end)
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged_rows.append({'chrom': chrom, 'start': current_start, 'end': current_end})
                current_start = start
                current_end = end

        merged_rows.append({'chrom': chrom, 'start': current_start, 'end': current_end})

    merged_df = pd.DataFrame(merged_rows, columns=['chrom', 'start', 'end'])
    merged_df['length'] = merged_df['end'] - merged_df['start']
    return merged_df.reset_index(drop=True)

def _load_random_region_reference_data():
    global _RANDOM_REGION_REFERENCE_CACHE

    if _RANDOM_REGION_REFERENCE_CACHE is None:
        centromeres = pd.read_csv(
            CENTROMERE_PATH,
            sep='\t',
            header=None,
            usecols=[0, 1, 2],
            names=['chrom', 'start', 'end'],
        )
        gaps = pd.read_csv(
            GAP_PATH,
            sep='\t',
            usecols=['chrom', 'chromStart', 'chromEnd', 'type'],
        )
        telomeres = gaps.loc[
            gaps['type'].astype(str).eq('telomere'),
            ['chrom', 'chromStart', 'chromEnd'],
        ].rename(columns={'chromStart': 'start', 'chromEnd': 'end'})
        forbidden_regions = _merge_intervals(pd.concat([centromeres, telomeres], ignore_index=True))

        chromosome_sizes = pd.read_csv(
            HG38_PATH,
            sep='\t',
            header=None,
            names=['chrom', 'size'],
        )
        chromosome_sizes['chrom'] = chromosome_sizes['chrom'].astype(str)
        chromosome_sizes['size'] = pd.to_numeric(chromosome_sizes['size'], errors='coerce')
        chromosome_sizes = chromosome_sizes.dropna(subset=['size']).copy()
        chromosome_sizes['size'] = chromosome_sizes['size'].astype(int)

        _RANDOM_REGION_REFERENCE_CACHE = (forbidden_regions, chromosome_sizes)

    return _RANDOM_REGION_REFERENCE_CACHE

def _build_allowed_region_pool(excluded_regions=None, min_length_bp=1):
    forbidden_regions, chromosome_sizes = _load_random_region_reference_data()
    excluded = _normalize_regions(excluded_regions)
    min_length_bp = max(1, int(min_length_bp))
    if not excluded.empty:
        forbidden_regions = _merge_intervals(
            pd.concat(
                [
                    forbidden_regions[['chrom', 'start', 'end']],
                    excluded[['chrom', 'start', 'end']],
                ],
                ignore_index=True,
            )
        )
    forbidden_by_chrom = {
        chrom: chrom_df.reset_index(drop=True)
        for chrom, chrom_df in forbidden_regions.groupby('chrom', sort=False)
    }

    allowed_rows = []
    for row in chromosome_sizes.itertuples(index=False):
        chrom = str(row.chrom)
        chrom_size = int(row.size)
        cursor = 0
        chrom_forbidden = forbidden_by_chrom.get(chrom)

        if chrom_forbidden is not None and not chrom_forbidden.empty:
            for interval in chrom_forbidden.itertuples(index=False):
                start = max(0, int(interval.start))
                end = min(chrom_size, int(interval.end))
                if start > cursor:
                    allowed_rows.append({'chrom': chrom, 'start': cursor, 'end': start})
                cursor = max(cursor, end)

        if cursor < chrom_size:
            allowed_rows.append({'chrom': chrom, 'start': cursor, 'end': chrom_size})

    allowed_pool = _normalize_regions(pd.DataFrame(allowed_rows, columns=['chrom', 'start', 'end']))
    return allowed_pool.loc[allowed_pool['length'] >= min_length_bp].reset_index(drop=True)

def _subtract_interval_from_pool(pool_df, chrom, used_start, used_end):
    if pool_df.empty:
        return pool_df.copy()

    kept_rows = []
    for row in pool_df.itertuples(index=False):
        if row.chrom != chrom or int(row.end) <= used_start or int(row.start) >= used_end:
            kept_rows.append({'chrom': row.chrom, 'start': int(row.start), 'end': int(row.end)})
            continue
        if int(row.start) < used_start:
            kept_rows.append({'chrom': row.chrom, 'start': int(row.start), 'end': int(used_start)})
        if int(row.end) > used_end:
            kept_rows.append({'chrom': row.chrom, 'start': int(used_end), 'end': int(row.end)})

    return _normalize_regions(pd.DataFrame(kept_rows, columns=['chrom', 'start', 'end']))

def collect_all_pmds(pmds_dict, fuzzy_merge_distance=1_000):
    """
    Merge PMDs across samples, allowing nearby regions to collapse together.

    Returns columns:
    chrom, start, end, length, sample_count, samples
    """
    parts = []
    for sample_id, df in pmds_dict.items():
        if df is None or df.empty:
            continue

        part = df[['chrom', 'start', 'end']].copy()
        part['sample_id'] = sample_id
        parts.append(part)

    if not parts:
        return pd.DataFrame(columns=['chrom', 'start', 'end', 'length', 'sample_count', 'samples'])

    all_pmds = (
        pd.concat(parts, ignore_index=True)
        .sort_values(['chrom', 'start', 'end'])
        .reset_index(drop=True)
    )

    merged_rows = []

    for chrom, chrom_df in all_pmds.groupby('chrom', sort=False):
        current_start = None
        current_end = None
        current_samples = set()

        for row in chrom_df.itertuples(index=False):
            start = int(row.start)
            end = int(row.end)
            sample_id = row.sample_id

            if current_start is None:
                current_start = start
                current_end = end
                current_samples = {sample_id}
                continue

            if start <= current_end + fuzzy_merge_distance:
                current_end = max(current_end, end)
                current_samples.add(sample_id)
            else:
                merged_rows.append({
                    'chrom': chrom,
                    'start': current_start,
                    'end': current_end,
                    'length': current_end - current_start,
                    'sample_count': len(current_samples),
                    'samples': sorted(current_samples),
                })
                current_start = start
                current_end = end
                current_samples = {sample_id}

        merged_rows.append({
            'chrom': chrom,
            'start': current_start,
            'end': current_end,
            'length': current_end - current_start,
            'sample_count': len(current_samples),
            'samples': sorted(current_samples),
        })

    return pd.DataFrame(merged_rows)


def exclude_overlapping_regions(region_df, excluded_regions, proximity_bp=0):
    """
    Remove regions that overlap or lie close to excluded regions.

    Parameters
    ----------
    region_df : pd.DataFrame
        Candidate regions to retain.
    excluded_regions : pd.DataFrame
        Regions that should be removed from the candidates.
    proximity_bp : int, default 0
        Maximum gap size still treated as overlapping.

    Returns
    -------
    pd.DataFrame
        region_df rows that do not overlap excluded_regions.
    """

    regions = region_df.copy().reset_index(drop=True)
    if regions.empty:
        return regions

    excluded = _normalize_regions(excluded_regions)
    if excluded.empty:
        return regions

    proximity_bp = max(0, int(proximity_bp))
    excluded_by_chrom = {
        str(chrom): chrom_df[['start', 'end']].to_numpy(dtype=int)
        for chrom, chrom_df in excluded.groupby('chrom', sort=False)
    }

    keep_mask = np.ones(len(regions), dtype=bool)
    for row_idx, row in enumerate(regions.itertuples(index=False)):
        excluded_intervals = excluded_by_chrom.get(str(row.chrom))
        if excluded_intervals is None or excluded_intervals.size == 0:
            continue

        start = int(row.start)
        end = int(row.end)
        is_near_or_overlapping = (
            (excluded_intervals[:, 0] <= end + proximity_bp)
            & (excluded_intervals[:, 1] >= start - proximity_bp)
        )
        if np.any(is_near_or_overlapping):
            keep_mask[row_idx] = False

    return regions.loc[keep_mask].reset_index(drop=True)

def load_cgis(filter_to_cannonical=True):
    """
    Load CpG islands (CGIs) data.

    Returns:
    pd.DataFrame: A DataFrame containing the CGI data.
    """

    cgi_data = pd.read_csv(
        CGI_BED,
        sep='\t',
        header=None,
        usecols=[0, 1, 2, 3],
        names=['chrom', 'start', 'end', 'name'],
    )
    cgi_data['length'] = cgi_data['end'] - cgi_data['start']

    if filter_to_cannonical:
        cgi_data = cgi_data[cgi_data['chrom'].isin(CANNONICAL_CHROMOSOMES)].reset_index(drop=True)

    return cgi_data

def pick_recurrent_pmds(all_pmds):
    recurrent_pmds = all_pmds.copy()

    if recurrent_pmds.empty:
        return recurrent_pmds.reset_index(drop=True)

    if "length" not in recurrent_pmds.columns:
        recurrent_pmds["length"] = (
            recurrent_pmds["end"] - recurrent_pmds["start"]
        )

    return (
        recurrent_pmds
        .sort_values(
            ["sample_count", "length"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

def fit_gamma_length_distribution(reference_regions: pd.DataFrame):
    """
    Fit a gamma distribution to observed region lengths.

    Returns a dictionary that can be passed into pick_random_methylation_regions.
    """

    reference = _normalize_regions(reference_regions)
    if reference.empty:
        raise ValueError('reference_regions must contain at least one valid interval.')

    lengths = reference['length'].to_numpy(dtype=float)
    if np.any(lengths <= 0):
        raise ValueError('reference_regions must contain positive interval lengths.')

    shape, loc, scale = stats.gamma.fit(lengths, floc=0)

    return {
        'distribution': 'gamma',
        'shape': float(shape),
        'scale': float(scale),
        'loc': float(loc),
        'min_length': int(np.floor(lengths.min())),
        'max_length': int(np.ceil(lengths.max())),
        'mean_length': float(lengths.mean()),
        'variance': float(lengths.var(ddof=1)) if len(lengths) > 1 else 0.0,
        'n_observations': int(len(lengths)),
    }

def _sample_gamma_lengths(n_regions, length_distribution, rng):
    if length_distribution.get('distribution') != 'gamma':
        raise ValueError('length_distribution must come from fit_gamma_length_distribution().')

    sampled = rng.gamma(
        shape=float(length_distribution['shape']),
        scale=float(length_distribution['scale']),
        size=int(n_regions),
    ) + float(length_distribution.get('loc', 0.0))
    sampled = np.rint(sampled).astype(int)
    sampled = np.clip(
        sampled,
        int(length_distribution['min_length']),
        int(length_distribution['max_length']),
    )
    sampled[sampled < 1] = 1
    return sampled

def _gamma_min_realistic_length(length_distribution):
    if length_distribution.get('distribution') != 'gamma':
        raise ValueError('length_distribution must come from fit_gamma_length_distribution().')

    min_length = stats.gamma.ppf(
        0.25,
        a=float(length_distribution['shape']),
        loc=float(length_distribution.get('loc', 0.0)),
        scale=float(length_distribution['scale']),
    )
    if np.isnan(min_length):
        raise ValueError('Could not compute a realistic minimum length from the gamma distribution.')
    return max(1, int(np.rint(min_length)))

def _choose_pool_interval(candidate_pool, rng, requested_length=None):
    if candidate_pool.empty:
        return None

    if requested_length is None:
        weights = candidate_pool['length'].to_numpy(dtype=float)
    else:
        weights = (candidate_pool['length'] - int(requested_length) + 1).to_numpy(dtype=float)

    weights = weights / weights.sum()
    chosen_idx = int(rng.choice(candidate_pool.index.to_numpy(), p=weights))
    return candidate_pool.loc[chosen_idx]

def _place_random_region(chosen_interval, rng, requested_length=None):
    if chosen_interval is None:
        return None

    interval_length = int(chosen_interval['length'])
    realized_length = interval_length if requested_length is None else int(requested_length)
    max_offset = interval_length - realized_length
    offset = 0 if max_offset == 0 else int(rng.randint(0, max_offset + 1))
    start = int(chosen_interval['start']) + offset
    end = start + realized_length

    return {
        'chrom': str(chosen_interval['chrom']),
        'start': start,
        'end': end,
        'length': realized_length,
    }

def pick_random_methylation_regions(
    n_regions,
    reference_regions: pd.DataFrame,
    length_distribution,
    random_seed=RANDOM_SEED,
    excluded_regions=None,
    cpg_positions_by_chrom: dict[str, np.ndarray] | None = None,
    min_cpgs_for_random_regions: int = 1,
    sample_from_cpg_anchors: bool = False,
):
    """
    Sample random genomic regions one at a time from the remaining allowed pool.

    Regions are drawn from hg38, exclude centromeres and telomeres, can also exclude
    training-fold PMDs, and do not overlap each other within a sampling call. The
    sampler prefers same-chromosome full-length fits, then any-chromosome full-length
    fits, and finally falls back to whole realistic intervals when no full-length fit exists.
    When sample_from_cpg_anchors=True, Random short/long regions are instead anchored
    on measured HM450K CpGs within intervals that can support the sampled length.
    """

    n_regions = int(n_regions)
    if n_regions <= 0:
        return pd.DataFrame(columns=['chrom', 'start', 'end', 'length'])

    reference = _normalize_regions(reference_regions)
    if reference.empty:
        raise ValueError('reference_regions must contain at least one valid interval.')

    rng = np.random.RandomState(random_seed)
    chrom_weights = reference['chrom'].value_counts(normalize=True).sort_index()
    chrom_options = chrom_weights.index.to_numpy(dtype=object)
    chrom_probabilities = chrom_weights.to_numpy(dtype=float)
    min_length_bp = _gamma_min_realistic_length(length_distribution)
    min_cpgs_for_random_regions = max(1, int(min_cpgs_for_random_regions))

    if sample_from_cpg_anchors and cpg_positions_by_chrom is None:
        raise ValueError('CpG-anchored random sampling requires cpg_positions_by_chrom.')

    allowed_pool = _build_allowed_region_pool(
        excluded_regions=excluded_regions,
        min_length_bp=min_length_bp,
    )
    if allowed_pool.empty:
        raise ValueError(
            'No realistic intervals remained in the allowed pool after excluding forbidden regions '
            f'and applying min_length_bp={min_length_bp}.'
        )
    chromosome_sizes = set(allowed_pool['chrom'])
    missing_chroms = sorted(set(chrom_options) - chromosome_sizes)
    if missing_chroms:
        raise ValueError(f'No chromosome size information was available for: {missing_chroms}')

    random_rows = []
    failed_regions = []

    for region_index in range(n_regions):
        placed_region = None
        failure_context = None

        for attempt_number in range(256):
            requested_length = int(_sample_gamma_lengths(1, length_distribution, rng)[0])
            chrom = str(rng.choice(chrom_options, p=chrom_probabilities))

            if sample_from_cpg_anchors:
                same_chrom_pool = _filter_interval_pool_for_requested_length(
                    allowed_pool.loc[allowed_pool['chrom'] == chrom].reset_index(drop=True),
                    requested_length=requested_length,
                    cpg_positions_by_chrom=cpg_positions_by_chrom,
                )
                any_chrom_pool = _filter_interval_pool_for_requested_length(
                    allowed_pool,
                    requested_length=requested_length,
                    cpg_positions_by_chrom=cpg_positions_by_chrom,
                )

                for candidate_pool in (same_chrom_pool, any_chrom_pool):
                    if candidate_pool.empty:
                        continue
                    chosen_interval = _choose_pool_interval(
                        candidate_pool,
                        rng,
                        requested_length=requested_length,
                    )
                    if chosen_interval is None:
                        continue
                    anchor_pos = _sample_anchor_position_from_interval(
                        chosen_interval,
                        cpg_positions_by_chrom,
                        rng,
                    )
                    if anchor_pos is None:
                        continue
                    placed_region = _place_anchor_random_region(
                        chosen_interval,
                        anchor_pos=anchor_pos,
                        requested_length=requested_length,
                        rng=rng,
                        cpg_positions_by_chrom=cpg_positions_by_chrom,
                        min_cpgs_for_random_regions=min_cpgs_for_random_regions,
                    )
                    if placed_region is not None:
                        break

                if placed_region is not None:
                    break
                failure_context = {
                    'region_index': region_index,
                    'attempt_number': attempt_number,
                    'chrom': chrom,
                    'requested_length': requested_length,
                    'min_cpgs_for_random_regions': min_cpgs_for_random_regions,
                }
                continue

            same_chrom_pool = allowed_pool.loc[
                (allowed_pool['chrom'] == chrom)
                & (allowed_pool['length'] >= requested_length)
            ].reset_index(drop=True)
            any_chrom_pool = allowed_pool.loc[
                allowed_pool['length'] >= requested_length
            ].reset_index(drop=True)

            if not same_chrom_pool.empty:
                chosen_interval = _choose_pool_interval(
                    same_chrom_pool,
                    rng,
                    requested_length=requested_length,
                )
                placed_region = _place_random_region(
                    chosen_interval,
                    rng,
                    requested_length=requested_length,
                )
            elif not any_chrom_pool.empty:
                chosen_interval = _choose_pool_interval(
                    any_chrom_pool,
                    rng,
                    requested_length=requested_length,
                )
                placed_region = _place_random_region(
                    chosen_interval,
                    rng,
                    requested_length=requested_length,
                )
            else:
                fallback_pool = allowed_pool.loc[
                    allowed_pool['length'] >= min_length_bp
                ].reset_index(drop=True)
                if fallback_pool.empty:
                    failure_context = {
                        'region_index': region_index,
                        'chrom': chrom,
                        'requested_length': requested_length,
                        'min_length_bp': min_length_bp,
                    }
                    break

                chosen_interval = _choose_pool_interval(fallback_pool, rng)
                placed_region = _place_random_region(chosen_interval, rng)
                if placed_region is not None:
                    break

            if placed_region is not None:
                break

        if placed_region is None:
            failed_regions.append(failure_context or {'region_index': region_index})
            break

        random_rows.append(placed_region)
        allowed_pool = _subtract_interval_from_pool(
            allowed_pool,
            placed_region['chrom'],
            placed_region['start'],
            placed_region['end'],
        )
        allowed_pool = allowed_pool.loc[allowed_pool['length'] >= min_length_bp].reset_index(drop=True)

    if failed_regions:
        failure_df = pd.DataFrame(failed_regions)
        raise ValueError(
            'Unable to sample enough non-overlapping random regions because no realistic intervals remained in the pool. '
            f'Remaining failures: {failure_df.to_dict(orient="records")}'
        )

    random_regions = pd.DataFrame(random_rows, columns=['chrom', 'start', 'end', 'length'])
    return random_regions.sort_values(['chrom', 'start', 'end']).reset_index(drop=True)

def _sample_region_rows(region_df, n_regions, seed):
    regions = region_df.copy()
    if regions.empty:
        return regions
    replace = len(regions) < int(n_regions)
    sampled = regions.sample(n=int(n_regions), replace=replace, random_state=seed).reset_index(drop=True)
    if {'start', 'end'}.issubset(sampled.columns) and 'length' not in sampled.columns:
        sampled['length'] = sampled['end'] - sampled['start']
    return sampled

def pick_features(
    n_features,
    training_pmds,
    cgis,
    offset=1000,
    random_seed=42,
    cpg_positions_by_chrom: dict[str, np.ndarray] | None = None,
    min_cpgs_for_random_regions: int = 1,
):
    """
    Build the five region sets used in the classification benchmark.

    Parameters
    ----------
    n_features : int
        Number of regions to include in each feature set.
    training_pmds : pd.DataFrame
        Fuzzy-merged PMDs from tumor samples in the current training fold.
    cgis : pd.DataFrame
        CpG island regions.
    offset : int, default 1000
        Offset added to the base seed so each draw is reproducible but distinct.
    random_seed : int, default 42
        Base random seed.

    Returns
    -------
    dict[str, pd.DataFrame]
        Named region sets keyed by feature family.
    """

    rng = np.random.RandomState(random_seed + offset)

    recurrent_pmds = (
        pick_recurrent_pmds(training_pmds)
        .head(n_features)
        .reset_index(drop=True)
    )

    random_pmds = _sample_region_rows(
        training_pmds,
        n_features,
        seed=int(rng.randint(0, 2**31 - 1)),
    )

    random_cgis = _sample_region_rows(
        cgis,
        n_features,
        seed=int(rng.randint(0, 2**31 - 1)),
    )

    pmd_length_distribution = fit_gamma_length_distribution(training_pmds)
    cgi_length_distribution = fit_gamma_length_distribution(cgis)

    random_long_regions = pick_random_methylation_regions(
        n_regions=n_features,
        reference_regions=training_pmds,
        length_distribution=pmd_length_distribution,
        random_seed=int(rng.randint(0, 2**31 - 1)),
        excluded_regions=training_pmds,
        cpg_positions_by_chrom=cpg_positions_by_chrom,
        min_cpgs_for_random_regions=min_cpgs_for_random_regions,
        sample_from_cpg_anchors=True,
    )

    random_short_regions = pick_random_methylation_regions(
        n_regions=n_features,
        reference_regions=cgis,
        length_distribution=cgi_length_distribution,
        random_seed=int(rng.randint(0, 2**31 - 1)),
        excluded_regions=training_pmds,
        cpg_positions_by_chrom=cpg_positions_by_chrom,
        min_cpgs_for_random_regions=min_cpgs_for_random_regions,
        sample_from_cpg_anchors=True,
    )

    return {
        "PMD": recurrent_pmds,
        "Random PMD": random_pmds,
        "CGI": random_cgis,
        "Random long": random_long_regions,
        "Random short": random_short_regions,
    }

def calculate_region_methylation(sample_ids, regions, methylation_data):
    """
    Compute mean methylation for each region in each sample.

    Parameters
    ----------
    sample_ids : sequence of str
        Samples to include as columns in the output.
    regions : pd.DataFrame
        Region table with chromosome, start, and end columns.
    methylation_data : pd.DataFrame
        CpG coordinate table followed by one methylation column per sample.

    Returns
    -------
    pd.DataFrame
        Region-by-sample matrix of mean methylation values.
    """

    sample_ids = list(sample_ids)
    regions = _normalize_regions(regions)
    if regions.empty or methylation_data.empty:
        return pd.DataFrame(columns=sample_ids)

    region_means = []
    for region in regions.itertuples(index=False):
        region_cpgs = methylation_data.loc[
            methylation_data['CpG_chrm'].astype(str).eq(str(region.chrom))
            & (methylation_data['CpG_beg'] < int(region.end))
            & (methylation_data['CpG_end'] > int(region.start)),
            sample_ids,
        ]

        if region_cpgs.empty:
            mean_values = pd.Series(np.nan, index=sample_ids, dtype=float)
        else:
            mean_values = region_cpgs.mean(axis=0)

        region_means.append(mean_values)

    return pd.DataFrame(region_means).reset_index(drop=True)

def get_training_pmds(train_sample_ids, sample_types, pmds_per_sample, label='Tumor'):
    """
    Merge PMDs from one label class in the current training fold.

    Parameters
    ----------
    train_sample_ids : sequence of str
        Sample IDs assigned to the training split.
    sample_types : dict[str, str]
        Mapping from sample ID to label such as 'Tumor' or 'Normal'.
    pmds_per_sample : dict[str, pd.DataFrame]
        PMD calls for each sample.
    label : str, default 'Tumor'
        Class label to collect PMDs from within the training split.

    Returns
    -------
    pd.DataFrame
        Fuzzy-merged PMD table for the selected training samples.
    """

    training_pmd_dict = {
        sample_id: pmds_per_sample[sample_id]
        for sample_id in train_sample_ids
        if sample_types.get(sample_id) == str(label) and sample_id in pmds_per_sample
    }
    return collect_all_pmds(training_pmd_dict)

def build_region_feature_matrix(sample_ids, regions, methylation_data):
    """
    Convert a region list into a sample-by-feature methylation matrix.

    Parameters
    ----------
    sample_ids : sequence of str
        Samples that should become matrix rows.
    regions : pd.DataFrame
        Region table to summarize.
    methylation_data : pd.DataFrame
        CpG coordinate table followed by one methylation column per sample.

    Returns
    -------
    pd.DataFrame
        Sample-by-region feature matrix.
    """

    sample_ids = list(sample_ids)
    region_matrix = calculate_region_methylation(sample_ids, regions, methylation_data)
    if region_matrix.empty:
        return pd.DataFrame(index=sample_ids)

    feature_matrix = region_matrix.T
    feature_matrix.index = sample_ids
    feature_matrix.columns = [f'region_{idx}' for idx in range(feature_matrix.shape[1])]
    return feature_matrix

def score_always_cancer(y_test):
    """
    Score a baseline that predicts tumor for every held-out sample.

    Parameters
    ----------
    y_test : array-like of int
        True labels for the held-out samples.

    Returns
    -------
    dict
        Metric dictionary containing balanced accuracy, average precision,
        macro F1, MCC, and ROC AUC.
    """

    y_test = np.asarray(y_test)
    predicted_class = np.ones_like(y_test)
    predicted_probability = np.ones_like(y_test, dtype=float)

    metrics = _score_predictions(
        y_test=y_test,
        predicted_class=predicted_class,
        predicted_probability=predicted_probability,
    )
    return metrics


def _score_predictions(y_test, predicted_class, predicted_probability):
    y_test = np.asarray(y_test)
    predicted_class = np.asarray(predicted_class)
    predicted_probability = np.asarray(predicted_probability, dtype=float)
    metrics = {
        'balanced_accuracy': balanced_accuracy_score(y_test, predicted_class),
        'average_precision': average_precision_score(y_test, predicted_probability),
        'macro_f1': f1_score(y_test, predicted_class, average='macro', zero_division=0),
        'mcc': matthews_corrcoef(y_test, predicted_class),
    }
    metrics['roc_auc'] = (
        roc_auc_score(y_test, predicted_probability)
        if np.unique(y_test).size == 2
        else np.nan
    )
    return metrics

def score_feature_set(
    train_sample_ids,
    test_sample_ids,
    y_train,
    y_test,
    regions,
    methylation_data,
    random_state,
    rf_n_jobs=1,
    return_predictions=False,
):
    """
    Fit one random-forest model for a single feature set and score it.

    Returns
    -------
    dict | None
        Metric dictionary containing balanced accuracy, average precision,
        macro F1, MCC, and ROC AUC, or None when the feature matrix is unusable.
    """

    X_train = build_region_feature_matrix(train_sample_ids, regions, methylation_data)
    X_test = build_region_feature_matrix(test_sample_ids, regions, methylation_data)
    if X_train.empty or X_test.empty:
        return None

    valid_columns = ~X_train.isna().all(axis=0)
    X_train = X_train.loc[:, valid_columns]
    X_test = X_test.loc[:, valid_columns]
    if X_train.shape[1] == 0:
        return None

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=500,
        class_weight='balanced',
        n_jobs=int(rf_n_jobs),
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    predicted_class = model.predict(X_test)
    predicted_probability = model.predict_proba(X_test)[:, 1]

    metrics = _score_predictions(
        y_test=y_test,
        predicted_class=predicted_class,
        predicted_probability=predicted_probability,
    )
    if return_predictions:
        return metrics, predicted_class, predicted_probability
    return metrics


def _append_fold_prediction_rows(
    fold_prediction_rows,
    test_sample_ids,
    y_test,
    predicted_class,
    predicted_probability,
    *,
    split,
    repeat,
    fold,
    feature_draw,
    feature_set,
    n_features,
):
    fold_key = f'repeat_{int(repeat):02d}_fold_{int(fold):02d}_split_{int(split):04d}'
    prediction_rows = pd.DataFrame({
        'fold_key': fold_key,
        'split': int(split),
        'repeat': int(repeat),
        'fold': int(fold),
        'sample_id': pd.Index(test_sample_ids, dtype='object'),
        'sample_label': np.where(np.asarray(y_test, dtype=int) == 1, 'Tumor', 'Normal'),
        'y_true': np.asarray(y_test, dtype=int),
        'feature_draw': int(feature_draw),
        'feature_set': str(feature_set),
        'n_features': int(n_features),
        'y_pred': np.asarray(predicted_class, dtype=int),
        'y_score': np.asarray(predicted_probability, dtype=float),
    })
    prediction_rows['correct'] = prediction_rows['y_true'].eq(prediction_rows['y_pred'])
    fold_prediction_rows.extend(
        prediction_rows[
            [
                'fold_key',
                'split',
                'repeat',
                'fold',
                'sample_id',
                'sample_label',
                'y_true',
                'feature_draw',
                'feature_set',
                'n_features',
                'y_pred',
                'y_score',
                'correct',
            ]
        ].to_dict(orient='records')
    )

def _append_feature_region_rows(
    feature_region_rows,
    regions,
    split,
    repeat,
    fold,
    feature_draw,
    feature_set,
    n_features,
):
    """
    Save the exact regions used for one evaluated feature set.
    """

    if regions is None:
        return

    normalized_regions = _normalize_regions(regions)
    if normalized_regions.empty:
        return

    region_records = normalized_regions.copy().reset_index(drop=True)
    region_records['split'] = int(split)
    region_records['repeat'] = int(repeat)
    region_records['fold'] = int(fold)
    region_records['feature_draw'] = int(feature_draw)
    region_records['feature_set'] = str(feature_set)
    region_records['n_features'] = int(n_features)
    region_records['region_index'] = np.arange(len(region_records), dtype=int)

    metadata_columns = [
        'split',
        'repeat',
        'fold',
        'feature_draw',
        'feature_set',
        'n_features',
        'region_index',
    ]
    preferred_region_columns = ['chrom', 'start', 'end', 'length', 'sample_count', 'samples', 'name']
    region_columns = [column for column in preferred_region_columns if column in region_records.columns]

    feature_region_rows.extend(
        region_records[metadata_columns + region_columns].to_dict(orient='records')
    )

def _init_feature_classification_worker(
    meth_data,
    sample_types,
    pmds_per_sample,
    cgis,
    feature_counts,
    n_feature_draws,
    random_seed,
    exclude_top_normal_shared_pmds,
    min_cpgs_for_random_regions,
):
    global _FEATURE_CLASSIFICATION_WORKER_STATE
    cpg_anchor_df = build_measured_cpg_anchor_table(meth_data)
    _FEATURE_CLASSIFICATION_WORKER_STATE = {
        'meth_data': meth_data,
        'sample_types': sample_types,
        'pmds_per_sample': pmds_per_sample,
        'cgis': cgis,
        'cpg_positions_by_chrom': _build_cpg_position_lookup(cpg_anchor_df),
        'feature_counts': tuple(int(value) for value in feature_counts),
        'n_feature_draws': int(n_feature_draws),
        'random_seed': int(random_seed),
        'exclude_top_normal_shared_pmds': bool(exclude_top_normal_shared_pmds),
        'min_cpgs_for_random_regions': max(1, int(min_cpgs_for_random_regions)),
    }

def _run_feature_classification_split(split_job):
    """
    Evaluate one cross-validation split for the feature classification benchmark.
    """

    state = _FEATURE_CLASSIFICATION_WORKER_STATE
    split_number = int(split_job['split_number'])
    repeat = int(split_job['repeat'])
    fold = int(split_job['fold'])
    train_idx = np.asarray(split_job['train_idx'], dtype=int)
    test_idx = np.asarray(split_job['test_idx'], dtype=int)

    sample_types = state['sample_types']
    sample_ids = np.asarray(list(sample_types.keys()))
    y = np.asarray([1 if sample_types[sample_id] == 'Tumor' else 0 for sample_id in sample_ids])
    train_sample_ids = sample_ids[train_idx].tolist()
    test_sample_ids = sample_ids[test_idx].tolist()
    y_train = y[train_idx]
    y_test = y[test_idx]

    training_tumor_pmds = get_training_pmds(
        train_sample_ids,
        sample_types,
        state['pmds_per_sample'],
        label='Tumor',
    )
    if training_tumor_pmds.empty:
        return [], [], []

    training_normal_pmds = get_training_pmds(
        train_sample_ids,
        sample_types,
        state['pmds_per_sample'],
        label='Normal',
    )
    recurrent_normal_regions = pick_recurrent_pmds(training_normal_pmds)

    results = []
    feature_region_rows = []
    fold_prediction_rows = []
    always_cancer_metrics = score_always_cancer(y_test)
    for n_features in state['feature_counts']:
        recurrent_training_pmds = training_tumor_pmds
        if state['exclude_top_normal_shared_pmds'] and not recurrent_normal_regions.empty:
            excluded_normal_regions = recurrent_normal_regions.head(n_features).reset_index(drop=True)
            recurrent_training_pmds = exclude_overlapping_regions(
                training_tumor_pmds,
                excluded_normal_regions,
                proximity_bp=1_000,
            )
        recurrent_regions = pick_recurrent_pmds(recurrent_training_pmds)

        always_cancer_predicted_class = np.ones_like(y_test, dtype=int)
        always_cancer_predicted_probability = np.ones_like(y_test, dtype=float)
        results.append({
            'split': split_number,
            'repeat': repeat,
            'fold': fold,
            'feature_draw': 0,
            'feature_set': 'Always cancer',
            'n_features': n_features,
            **always_cancer_metrics,
        })
        _append_fold_prediction_rows(
            fold_prediction_rows=fold_prediction_rows,
            test_sample_ids=test_sample_ids,
            y_test=y_test,
            predicted_class=always_cancer_predicted_class,
            predicted_probability=always_cancer_predicted_probability,
            split=split_number,
            repeat=repeat,
            fold=fold,
            feature_draw=0,
            feature_set='Always cancer',
            n_features=n_features,
        )

        selected_recurrent_regions = recurrent_regions.head(n_features).reset_index(drop=True)
        if len(selected_recurrent_regions) == n_features:
            recurrent_seed = state['random_seed'] + split_number * 100_000 + n_features * 100
            scoring_payload = score_feature_set(
                train_sample_ids=train_sample_ids,
                test_sample_ids=test_sample_ids,
                y_train=y_train,
                y_test=y_test,
                regions=selected_recurrent_regions,
                methylation_data=state['meth_data'],
                random_state=recurrent_seed,
                rf_n_jobs=1,
                return_predictions=True,
            )
            if scoring_payload is not None:
                metrics, predicted_class, predicted_probability = scoring_payload
                results.append({
                    'split': split_number,
                    'repeat': repeat,
                    'fold': fold,
                    'feature_draw': 0,
                    'feature_set': 'PMD',
                    'n_features': n_features,
                    **metrics,
                })
                _append_feature_region_rows(
                    feature_region_rows=feature_region_rows,
                    regions=selected_recurrent_regions,
                    split=split_number,
                    repeat=repeat,
                    fold=fold,
                    feature_draw=0,
                    feature_set='PMD',
                    n_features=n_features,
                )
                _append_fold_prediction_rows(
                    fold_prediction_rows=fold_prediction_rows,
                    test_sample_ids=test_sample_ids,
                    y_test=y_test,
                    predicted_class=predicted_class,
                    predicted_probability=predicted_probability,
                    split=split_number,
                    repeat=repeat,
                    fold=fold,
                    feature_draw=0,
                    feature_set='PMD',
                    n_features=n_features,
                )

        for draw in range(state['n_feature_draws']):
            selection_seed = state['random_seed'] + split_number * 100_000 + n_features * 100 + draw
            feature_sets = pick_features(
                n_features=n_features,
                training_pmds=training_tumor_pmds,
                cgis=state['cgis'],
                offset=selection_seed,
                random_seed=state['random_seed'],
                cpg_positions_by_chrom=state['cpg_positions_by_chrom'],
                min_cpgs_for_random_regions=state['min_cpgs_for_random_regions'],
            )

            for feature_set_name, regions in feature_sets.items():
                if feature_set_name == 'PMD' or len(regions) < n_features:
                    continue

                scoring_payload = score_feature_set(
                    train_sample_ids=train_sample_ids,
                    test_sample_ids=test_sample_ids,
                    y_train=y_train,
                    y_test=y_test,
                    regions=regions,
                    methylation_data=state['meth_data'],
                    random_state=selection_seed,
                    rf_n_jobs=1,
                    return_predictions=True,
                )
                if scoring_payload is None:
                    continue
                metrics, predicted_class, predicted_probability = scoring_payload

                results.append({
                    'split': split_number,
                    'repeat': repeat,
                    'fold': fold,
                    'feature_draw': draw,
                    'feature_set': feature_set_name,
                    'n_features': n_features,
                    **metrics,
                })
                _append_feature_region_rows(
                    feature_region_rows=feature_region_rows,
                    regions=regions,
                    split=split_number,
                    repeat=repeat,
                    fold=fold,
                    feature_draw=draw,
                    feature_set=feature_set_name,
                    n_features=n_features,
                )
                _append_fold_prediction_rows(
                    fold_prediction_rows=fold_prediction_rows,
                    test_sample_ids=test_sample_ids,
                    y_test=y_test,
                    predicted_class=predicted_class,
                    predicted_probability=predicted_probability,
                    split=split_number,
                    repeat=repeat,
                    fold=fold,
                    feature_draw=draw,
                    feature_set=feature_set_name,
                    n_features=n_features,
                )

    return results, feature_region_rows, fold_prediction_rows

def _feature_classification_process_worker(
    split_jobs,
    result_queue,
    meth_data,
    sample_types,
    pmds_per_sample,
    cgis,
    feature_counts,
    n_feature_draws,
    random_seed,
    exclude_top_normal_shared_pmds,
    min_cpgs_for_random_regions,
):
    """
    Run a chunk of CV splits inside one forked worker process.
    """

    _init_feature_classification_worker(
        meth_data=meth_data,
        sample_types=sample_types,
        pmds_per_sample=pmds_per_sample,
        cgis=cgis,
        feature_counts=feature_counts,
        n_feature_draws=n_feature_draws,
        random_seed=random_seed,
        exclude_top_normal_shared_pmds=exclude_top_normal_shared_pmds,
        min_cpgs_for_random_regions=min_cpgs_for_random_regions,
    )

    for split_job in split_jobs:
        result_queue.put(_run_feature_classification_split(split_job))

    result_queue.put(None)

def run_feature_classification(
    meth_data,
    sample_types,
    pmds_per_sample,
    cgis,
    feature_counts,
    n_splits=5,
    n_repeats=5,
    n_feature_draws=10,
    random_seed=42,
    max_split_workers=None,
    return_fold_predictions=False,
    exclude_top_normal_shared_pmds=False,
    min_cpgs_for_random_regions=1,
):
    """
    Benchmark region-derived feature sets with repeated stratified CV.

    Parameters
    ----------
    meth_data : pd.DataFrame
        CpG coordinate table followed by one methylation column per sample.
    sample_types : dict[str, str]
        Mapping from sample ID to class label. 'Tumor' is encoded as 1 and
        all other labels are encoded as 0.
    pmds_per_sample : dict[str, pd.DataFrame]
        Per-sample PMD calls.
    cgis : pd.DataFrame
        CpG island region table.
    feature_counts : sequence of int
        Numbers of regions to evaluate for each feature family.
    n_splits : int, default 5
        Stratified folds per repeat.
    n_repeats : int, default 5
        Number of repeated CV rounds.
    n_feature_draws : int, default 10
        Independent redraws of the stochastic feature sets for each fold and
        feature count. Deterministic recurrent PMDs and the always-cancer
        baseline are evaluated once per split.
    random_seed : int, default 42
        Base seed used for CV splitting and feature selection.
    max_split_workers : int | None, default None
        Number of multiprocessing workers used across CV splits. When None,
        use the notebook-level MAX_SPLIT_WORKERS setting.
    exclude_top_normal_shared_pmds : bool, default False
        When True, remove tumor training PMDs that overlap the top n shared
        normal PMDs in the same fold before ranking recurrent tumor PMDs for
        each feature count n.
    min_cpgs_for_random_regions : int, default 1
        Minimum measured HM450K CpGs that must fall inside each anchored
        Random short / Random long region.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Classification results plus the exact regions used for each evaluated
        genomic feature set, and optionally one prediction row per held-out
        sample / feature-set evaluation.
    """

    sample_ids = np.asarray(list(sample_types.keys()))
    y = np.asarray([1 if sample_types[sample_id] == 'Tumor' else 0 for sample_id in sample_ids])

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_seed,
    )

    split_jobs = []
    for split_number, (train_idx, test_idx) in enumerate(cv.split(sample_ids, y)):
        split_jobs.append({
            'split_number': int(split_number),
            'repeat': int(split_number // n_splits),
            'fold': int(split_number % n_splits),
            'train_idx': np.asarray(train_idx, dtype=int),
            'test_idx': np.asarray(test_idx, dtype=int),
        })

    if max_split_workers is None:
        max_split_workers = MAX_SPLIT_WORKERS
    max_split_workers = max(1, int(max_split_workers))

    results = []
    feature_region_rows = []
    fold_prediction_rows = []
    if max_split_workers == 1:
        _init_feature_classification_worker(
            meth_data=meth_data,
            sample_types=sample_types,
            pmds_per_sample=pmds_per_sample,
            cgis=cgis,
            feature_counts=feature_counts,
            n_feature_draws=n_feature_draws,
            random_seed=random_seed,
            exclude_top_normal_shared_pmds=exclude_top_normal_shared_pmds,
            min_cpgs_for_random_regions=min_cpgs_for_random_regions,
        )
        split_iterator = tqdm(split_jobs, total=len(split_jobs), desc='CV splits')
        for split_job in split_iterator:
            split_results, split_feature_regions, split_fold_predictions = _run_feature_classification_split(split_job)
            results.extend(split_results)
            feature_region_rows.extend(split_feature_regions)
            fold_prediction_rows.extend(split_fold_predictions)
    else:
        worker_count = min(max_split_workers, len(split_jobs))
        ctx = mp.get_context('fork')
        result_queue = ctx.Queue()
        split_job_chunks = [
            split_jobs[worker_index::worker_count]
            for worker_index in range(worker_count)
        ]
        processes = []
        for split_job_chunk in split_job_chunks:
            process = ctx.Process(
                target=_feature_classification_process_worker,
                args=(
                    split_job_chunk,
                    result_queue,
                    meth_data,
                    sample_types,
                    pmds_per_sample,
                    cgis,
                    feature_counts,
                    n_feature_draws,
                    random_seed,
                    exclude_top_normal_shared_pmds,
                    min_cpgs_for_random_regions,
                ),
            )
            process.start()
            processes.append(process)

        finished_workers = 0
        progress_bar = tqdm(total=len(split_jobs), desc='CV splits')
        try:
            while finished_workers < worker_count:
                payload = result_queue.get()
                if payload is None:
                    finished_workers += 1
                    continue

                split_results, split_feature_regions, split_fold_predictions = payload
                results.extend(split_results)
                feature_region_rows.extend(split_feature_regions)
                fold_prediction_rows.extend(split_fold_predictions)
                progress_bar.update(1)
        finally:
            progress_bar.close()
            for process in processes:
                process.join()

    results_df = pd.DataFrame(results)
    feature_region_columns = [
        'split',
        'repeat',
        'fold',
        'feature_draw',
        'feature_set',
        'n_features',
        'region_index',
        'chrom',
        'start',
        'end',
        'length',
        'sample_count',
        'samples',
        'name',
    ]
    feature_regions_df = pd.DataFrame(feature_region_rows)
    if feature_regions_df.empty:
        feature_regions_df = pd.DataFrame(columns=feature_region_columns)
    else:
        available_columns = [column for column in feature_region_columns if column in feature_regions_df.columns]
        feature_regions_df = feature_regions_df[available_columns].copy()

    fold_prediction_columns = [
        'fold_key',
        'split',
        'repeat',
        'fold',
        'sample_id',
        'sample_label',
        'y_true',
        'feature_draw',
        'feature_set',
        'n_features',
        'y_pred',
        'y_score',
        'correct',
    ]
    fold_predictions_df = pd.DataFrame(fold_prediction_rows)
    if fold_predictions_df.empty:
        fold_predictions_df = pd.DataFrame(columns=fold_prediction_columns)
    else:
        fold_predictions_df = (
            fold_predictions_df[fold_prediction_columns]
            .sort_values(['n_features', 'split', 'sample_id', 'feature_draw', 'feature_set'])
            .reset_index(drop=True)
        )

    if return_fold_predictions:
        return results_df, feature_regions_df, fold_predictions_df
    return results_df, feature_regions_df

def summarize_classification_results(results, metrics=None):
    """
    Aggregate classification scores by feature set and feature count.

    Parameters
    ----------
    results : pd.DataFrame
        Output from run_feature_classification.
    metrics : sequence of str, optional
        Metrics to summarize. Defaults to the standard classifier metrics.

    Returns
    -------
    pd.DataFrame
        Long-format summary with mean, standard deviation, and standard
        error for each metric.
    """

    if metrics is None:
        metrics = ['balanced_accuracy', 'average_precision', 'macro_f1', 'mcc', 'roc_auc']

    if results.empty:
        return pd.DataFrame(
            columns=['feature_set', 'n_features', 'metric', 'mean', 'std', 'n_runs', 'sem']
        )

    available_metrics = [metric for metric in metrics if metric in results.columns]
    long_results = results[
        ['feature_set', 'n_features', *available_metrics]
    ].melt(
        id_vars=['feature_set', 'n_features'],
        value_vars=available_metrics,
        var_name='metric',
        value_name='score',
    )

    summary = (
        long_results
        .groupby(['feature_set', 'n_features', 'metric'], as_index=False)['score']
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .rename(columns={'count': 'n_runs'})
    )
    summary['sem'] = summary['std'].fillna(0.0) / np.sqrt(summary['n_runs'].clip(lower=1))
    return summary

def _default_feature_set_labels():
    return {
        'PMD': 'Recurrent PMDs',
        'Random PMD': 'Random PMDs',
        'CGI': 'Random CGIs',
        'Random long': 'Random long',
        'Random short': 'Random short',
        'Always cancer': 'Always cancer',
    }

def plot_classification_results(
    results,
    metrics=None,
    feature_set_order=None,
    palette=None,
    feature_set_labels=None,
    x_log_scale=False,
    subplot_title_suffix=None,
    figure_title=None,
    title_size=18,
    label_size=14,
    tick_size=11,
    figure_title_size=16,
    legend_font_size=None,
):
    """
    Plot classifier performance as a function of feature-set size.

    Parameters
    ----------
    results : pd.DataFrame
        Output from run_feature_classification.
    metrics : sequence of str, optional
        Metrics to display. Defaults to the standard classifier metrics.
    feature_set_order : sequence of str, optional
        Plot order for feature families.
    palette : dict[str, str], optional
        Colors for each feature family.
    feature_set_labels : dict[str, str], optional
        Display labels for each feature family.
    x_log_scale : bool, default False
        Whether to use a logarithmic x-axis for the feature-count values.
    subplot_title_suffix : str | None, optional
        Optional suffix appended to each metric subplot title.
    figure_title : str | None, optional
        Optional overall figure title. Defaults to the standard classifier title.

    Returns
    -------
    pd.DataFrame
        Summary table used for plotting.
    """

    if metrics is None:
        metrics = ['balanced_accuracy', 'average_precision', 'macro_f1', 'mcc', 'roc_auc']

    if feature_set_order is None:
        feature_set_order = ['PMD', 'Random PMD', 'CGI', 'Random long', 'Random short', 'Always cancer']

    if palette is None:
        palette = {
            'PMD': '#0b5394',
            'Random PMD': '#3d85c6',
            'CGI': '#38761d',
            'Random long': '#9c6ade',
            'Random short': '#e69138',
            'Always cancer': '#b7b7b7',
        }

    if feature_set_labels is None:
        feature_set_labels = _default_feature_set_labels()

    summary = summarize_classification_results(results, metrics=metrics)
    if summary.empty:
        raise ValueError('results is empty. Run run_feature_classification() before plotting.')

    metric_titles = {
        'balanced_accuracy': 'Balanced accuracy',
        'average_precision': 'Average precision',
        'macro_f1': 'Macro F1 score',
        'mcc': 'Matthews correlation coefficient',
        'roc_auc': 'ROC AUC',
    }

    plotted_metrics = [metric for metric in metrics if metric in summary['metric'].unique()]
    n_cols = 2
    n_rows = int(np.ceil(len(plotted_metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, metric in zip(axes, plotted_metrics):
        metric_summary = summary.loc[summary['metric'] == metric].copy()

        for feature_set in feature_set_order:
            feature_df = metric_summary.loc[
                metric_summary['feature_set'] == feature_set
            ].sort_values('n_features')
            if feature_df.empty:
                continue

            x = feature_df['n_features'].to_numpy(dtype=float)
            y = feature_df['mean'].to_numpy(dtype=float)
            sem = feature_df['sem'].fillna(0.0).to_numpy(dtype=float)
            color = palette.get(feature_set)

            ax.plot(
                x,
                y,
                marker='o',
                linewidth=2,
                color=color,
                label=feature_set_labels.get(feature_set, feature_set),
            )
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.15)

        metric_title = metric_titles.get(metric, metric.replace('_', ' ').title())
        if subplot_title_suffix:
            metric_title = f'{metric_title} - {subplot_title_suffix}'
        ax.set_title(metric_title, fontsize=title_size)
        ax.set_xlabel('Number of regions', fontsize=label_size)
        ax.set_ylabel('Score', fontsize=label_size)
        ax.tick_params(axis='x', labelbottom=True, labelsize=tick_size)
        ax.tick_params(axis='y', labelsize=tick_size)
        if x_log_scale:
            ax.set_xscale('log')
        if metric == 'mcc':
            ax.set_ylim(-1.05, 1.05)
        else:
            ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)

    for ax in axes[len(plotted_metrics):]:
        ax.axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    if figure_title is None:
        figure_title = 'Classification performance across feature-set sizes'
    fig.legend(
        unique_labels.values(),
        unique_labels.keys(),
        loc='upper center',
        bbox_to_anchor=(0.5, 0.955),
        ncol=min(len(unique_labels), 3),
        frameon=False,
        prop={'size': legend_font_size or tick_size},
    )
    fig.suptitle(figure_title, y=0.99, fontsize=figure_title_size)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()

    return summary

def plot_classification_box_results(
    results,
    n_features=5,
    metrics=None,
    feature_set_order=None,
    palette=None,
    feature_set_labels=None,
    subplot_title_suffix=None,
    figure_title=None,
    title_size=18,
    label_size=14,
    tick_size=11,
    figure_title_size=16,
):
    """
    Plot per-run score distributions for a single feature-set size with box-and-whisker plots.

    Parameters
    ----------
    results : pd.DataFrame
        Output from run_feature_classification.
    n_features : int, default 5
        Feature-set size to display.
    metrics : sequence of str, optional
        Metrics to display. Defaults to the standard classifier metrics.
    feature_set_order : sequence of str, optional
        Plot order for feature families.
    palette : dict[str, str], optional
        Colors for each feature family.
    feature_set_labels : dict[str, str], optional
        Display labels for each feature family.
    subplot_title_suffix : str | None, optional
        Optional suffix appended to each metric subplot title.
    figure_title : str | None, optional
        Optional overall figure title. Defaults to the standard classifier title.

    Returns
    -------
    pd.DataFrame
        Long-format table used for plotting.
    """

    if metrics is None:
        metrics = ['balanced_accuracy', 'average_precision', 'macro_f1', 'mcc', 'roc_auc']

    if feature_set_order is None:
        feature_set_order = ['PMD', 'Random PMD', 'CGI', 'Random long', 'Random short', 'Always cancer']

    if palette is None:
        palette = {
            'PMD': '#0b5394',
            'Random PMD': '#3d85c6',
            'CGI': '#38761d',
            'Random long': '#9c6ade',
            'Random short': '#e69138',
            'Always cancer': '#b7b7b7',
        }

    if feature_set_labels is None:
        feature_set_labels = _default_feature_set_labels()

    if results.empty:
        raise ValueError('results is empty. Run run_feature_classification() before plotting.')

    available_metrics = [metric for metric in metrics if metric in results.columns]
    box_data = results.loc[
        results['n_features'] == int(n_features),
        ['feature_set', 'n_features', *available_metrics],
    ].copy()
    if box_data.empty:
        raise ValueError(f'No results were found for n_features={int(n_features)}.')

    box_long = box_data.melt(
        id_vars=['feature_set', 'n_features'],
        value_vars=available_metrics,
        var_name='metric',
        value_name='score',
    )
    box_long['feature_set'] = pd.Categorical(
        box_long['feature_set'],
        categories=feature_set_order,
        ordered=True,
    )
    display_feature_set_order = [
        feature_set_labels.get(feature_set, feature_set)
        for feature_set in feature_set_order
    ]
    display_palette = {
        feature_set_labels.get(feature_set, feature_set): color
        for feature_set, color in palette.items()
    }
    box_long['display_feature_set'] = box_long['feature_set'].map(
        lambda feature_set: feature_set_labels.get(feature_set, feature_set)
    )
    box_long['display_feature_set'] = pd.Categorical(
        box_long['display_feature_set'],
        categories=display_feature_set_order,
        ordered=True,
    )
    box_long = box_long.sort_values(['metric', 'feature_set']).reset_index(drop=True)

    metric_titles = {
        'balanced_accuracy': 'Balanced accuracy',
        'average_precision': 'Average precision',
        'macro_f1': 'Macro F1 score',
        'mcc': 'Matthews correlation coefficient',
        'roc_auc': 'ROC AUC',
    }

    plotted_metrics = [metric for metric in metrics if metric in box_long['metric'].unique()]
    n_cols = 2
    n_rows = int(np.ceil(len(plotted_metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows), sharey=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, metric in zip(axes, plotted_metrics):
        metric_df = box_long.loc[box_long['metric'] == metric].copy()
        sns.boxplot(
            data=metric_df,
            x='display_feature_set',
            y='score',
            hue='display_feature_set',
            order=display_feature_set_order,
            palette=display_palette,
            dodge=False,
            saturation=1,
            fliersize=0,
            linewidth=1,
            ax=ax,
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        sns.stripplot(
            data=metric_df,
            x='display_feature_set',
            y='score',
            order=display_feature_set_order,
            color='black',
            alpha=0.45,
            size=3,
            jitter=0.15,
            ax=ax,
        )
        metric_title = metric_titles.get(metric, metric.replace('_', ' ').title())
        if subplot_title_suffix:
            metric_title = f'{metric_title} - {subplot_title_suffix}'
        ax.set_title(metric_title, fontsize=title_size)
        ax.set_xlabel('Feature set', fontsize=label_size)
        ax.set_ylabel('Score', fontsize=label_size)
        ax.tick_params(axis='x', rotation=25, labelsize=tick_size)
        ax.tick_params(axis='y', labelsize=tick_size)
        if metric == 'mcc':
            ax.set_ylim(-1.05, 1.05)
        else:
            ax.set_ylim(-0.05, 1.05)
        ax.grid(True, axis='y', alpha=0.25)

    for ax in axes[len(plotted_metrics):]:
        ax.axis('off')

    if figure_title is None:
        figure_title = f'Per-run classification scores for {int(n_features)} regions'
    fig.suptitle(figure_title, y=1.02, fontsize=figure_title_size)
    fig.tight_layout()
    plt.show()

    return box_long

def plot_sampled_feature_length_distributions(
    sampled_feature_regions,
    n_features=None,
    feature_set_order=None,
    palette=None,
    feature_set_labels=None,
    bins=30,
):
    """
    Plot the region-length distributions of the feature sets actually used by the classifier.

    Parameters
    ----------
    sampled_feature_regions : pd.DataFrame
        Region table returned by run_feature_classification.
    n_features : int | None, default None
        Feature-set size to display. When None, combine sampled regions across
        all available feature-set sizes.
    feature_set_order : sequence of str, optional
        Plot order for feature families with genomic intervals.
    palette : dict[str, str], optional
        Colors for each feature family.
    feature_set_labels : dict[str, str], optional
        Display labels for each feature family.
    bins : int, default 30
        Number of histogram bins.

    Returns
    -------
    pd.DataFrame
        Long-format table used for plotting.
    """

    if feature_set_order is None:
        feature_set_order = ['PMD', 'Random PMD', 'CGI', 'Random long', 'Random short']

    if palette is None:
        palette = {
            'PMD': '#0b5394',
            'Random PMD': '#3d85c6',
            'CGI': '#38761d',
            'Random long': '#9c6ade',
            'Random short': '#e69138',
        }

    if feature_set_labels is None:
        feature_set_labels = _default_feature_set_labels()

    if sampled_feature_regions.empty:
        raise ValueError(
            'sampled_feature_regions is empty. Run run_feature_classification() before plotting.'
        )

    if n_features is None:
        plot_data = sampled_feature_regions.loc[
            :,
            ['feature_set', 'n_features', 'length'],
        ].copy()
        title_suffix = 'all feature-set sizes'
    else:
        plot_data = sampled_feature_regions.loc[
            sampled_feature_regions['n_features'] == int(n_features),
            ['feature_set', 'n_features', 'length'],
        ].copy()
        title_suffix = f'{int(n_features)}-feature sets'

    if plot_data.empty:
        if n_features is None:
            raise ValueError('No sampled feature regions were found.')
        raise ValueError(f'No sampled feature regions were found for n_features={int(n_features)}.')

    plot_data = plot_data.loc[
        plot_data['feature_set'].isin(feature_set_order)
    ].copy()
    display_feature_set_order = [
        feature_set_labels.get(feature_set, feature_set)
        for feature_set in feature_set_order
    ]
    display_palette = {
        feature_set_labels.get(feature_set, feature_set): color
        for feature_set, color in palette.items()
    }
    plot_data['feature_set'] = pd.Categorical(
        plot_data['feature_set'],
        categories=feature_set_order,
        ordered=True,
    )
    plot_data['display_feature_set'] = plot_data['feature_set'].map(
        lambda feature_set: feature_set_labels.get(feature_set, feature_set)
    )
    plot_data['display_feature_set'] = pd.Categorical(
        plot_data['display_feature_set'],
        categories=display_feature_set_order,
        ordered=True,
    )
    plot_data = plot_data.sort_values(['feature_set', 'length']).reset_index(drop=True)

    plotted_feature_sets = [
        feature_set
        for feature_set in feature_set_order
        if feature_set in set(plot_data['feature_set'].astype(str))
    ]
    n_cols = 2
    n_rows = int(np.ceil(len(plotted_feature_sets) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 3.8 * n_rows),
        sharex=False,
        sharey=False,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, feature_set in zip(axes, plotted_feature_sets):
        display_feature_set = feature_set_labels.get(feature_set, feature_set)
        feature_df = plot_data.loc[
            plot_data['feature_set'] == feature_set,
            ['length'],
        ].copy()
        sns.histplot(
            data=feature_df,
            x='length',
            bins=int(bins),
            color=display_palette.get(display_feature_set),
            element='bars',
            alpha=0.55,
            stat='density',
            ax=ax,
        )
        ax.set_xscale('log')
        ax.set_title(display_feature_set)
        ax.set_xlabel('Region length (bp)')
        ax.set_ylabel('Density')
        ax.grid(True, axis='y', alpha=0.25)

    for ax in axes[len(plotted_feature_sets):]:
        ax.axis('off')

    fig.suptitle(f'Sampled region lengths for {title_suffix}', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return plot_data


__all__ = [
    'MAX_SPLIT_WORKERS',
    'CLASSIFICATION_METRICS',
    'FEATURE_SET_ORDER',
    'FEATURE_SET_PALETTE',
    'TCGA_SAMPLES',
    'PMD_PATH',
    'cohort_output_dir',
    'parse_feature_counts',
    'make_sample_types',
    'summarize_cohort_samples',
    'build_per_cancer_cohort_manifest',
    'save_feature_classification_outputs',
    'load_saved_feature_classification_task',
    'collect_saved_feature_classification_outputs',
    'run_cohort_feature_classification',
    'load_tcga_samples',
    'load_methylation_data',
    'build_measured_cpg_anchor_table',
    'load_pmds_per_sample',
    'collect_all_pmds',
    'load_cgis',
    'pick_recurrent_pmds',
    'fit_gamma_length_distribution',
    'pick_random_methylation_regions',
    'pick_features',
    'run_feature_classification',
    'summarize_classification_results',
    'plot_classification_results',
    'plot_classification_box_results',
    'plot_sampled_feature_length_distributions',
]
