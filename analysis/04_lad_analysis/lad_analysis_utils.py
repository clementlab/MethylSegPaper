from __future__ import annotations

import shlex
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from repo_paths import REGION_CALLING_RESULTS_DIR

DEFAULT_SEGMENTATION_RESULTS_PATH = REGION_CALLING_RESULTS_DIR
UCSC_DOWNLOAD_BASE = "https://hgdownload.soe.ucsc.edu/goldenPath"

BED_COLUMNS = ["chrom", "start", "end"]
INTERVAL_COLUMNS = BED_COLUMNS + ["length"]
BEDGRAPH_COLUMNS = BED_COLUMNS + ["value"]
END_WINDOW_COLUMNS = ["region_id", "chrom", "start", "end", "window_label", "length"]
CANONICAL_CHROMS = tuple([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"])
CHROM_RANK = {chrom: idx for idx, chrom in enumerate(CANONICAL_CHROMS)}


def empty_interval_df() -> pd.DataFrame:
    return pd.DataFrame(columns=INTERVAL_COLUMNS)


def empty_signal_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BEDGRAPH_COLUMNS)


def empty_end_window_df() -> pd.DataFrame:
    return pd.DataFrame(columns=END_WINDOW_COLUMNS)


def sample_to_sample_id(sample: str) -> str:
    return sample.replace(".wgbs", "")


def threshold_label(threshold_bp: int) -> str:
    threshold_bp = int(threshold_bp)
    if threshold_bp == 0:
        return "0bp"
    if threshold_bp % 1000 == 0:
        return f"{threshold_bp // 1000}kb"
    return f"{threshold_bp}bp"


def build_tool_region_path(
    segmentation_results_path: str | Path,
    sample: str,
    tool_config: dict,
) -> Path:
    path = Path(segmentation_results_path)
    for part in tool_config["path_parts"]:
        path = path / part.format(sample=sample)
    return path


def resolve_sample_genome(
    sample: str,
    reference_tool: str = "methylseg",
    segmentation_results_path: str | Path = DEFAULT_SEGMENTATION_RESULTS_PATH,
) -> str:
    segmentation_results_path = Path(segmentation_results_path)
    candidate_tools = [reference_tool, "methylseekr", "dnmtools", "mmseekr", "methyl_lasso"]
    seen_tools: set[str] = set()

    for tool_name in candidate_tools:
        if tool_name in seen_tools:
            continue
        seen_tools.add(tool_name)
        config_path = segmentation_results_path / tool_name / sample / "prep" / "config.yaml"
        if not config_path.exists():
            continue

        with config_path.open() as handle:
            config = yaml.safe_load(handle)

        genome = str(config.get("genome", "")).strip()
        if genome in {"hg19", "hg38"}:
            return genome
        raise ValueError(f"Unsupported genome {genome!r} for {sample} in {config_path}")

    # The current benchmark cohort has a stable genome split even when
    # scratch-space prep configs are unavailable.
    if sample.startswith("WGBS_colon"):
        return "hg38"
    return "hg19"


def ensure_interval_df(interval_df: pd.DataFrame | None) -> pd.DataFrame:
    if interval_df is None or interval_df.empty:
        return empty_interval_df()

    clean_df = interval_df.copy()
    if "chrom" not in clean_df.columns and "chr" in clean_df.columns:
        clean_df = clean_df.rename(columns={"chr": "chrom"})

    missing_columns = [column for column in BED_COLUMNS if column not in clean_df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required BED columns {missing_columns} in {list(clean_df.columns)}"
        )

    clean_df = clean_df.loc[:, BED_COLUMNS].copy()
    clean_df["chrom"] = clean_df["chrom"].astype(str)
    clean_df["start"] = pd.to_numeric(clean_df["start"], errors="coerce")
    clean_df["end"] = pd.to_numeric(clean_df["end"], errors="coerce")
    clean_df = clean_df.dropna(subset=["chrom", "start", "end"]).copy()
    if clean_df.empty:
        return empty_interval_df()

    clean_df["start"] = clean_df["start"].astype(int)
    clean_df["end"] = clean_df["end"].astype(int)
    clean_df["start"] = clean_df["start"].clip(lower=0)
    clean_df = clean_df.loc[clean_df["end"] > clean_df["start"]].copy()
    if clean_df.empty:
        return empty_interval_df()

    clean_df["length"] = clean_df["end"] - clean_df["start"]
    return clean_df.loc[:, INTERVAL_COLUMNS].reset_index(drop=True)


def ensure_signal_df(signal_df: pd.DataFrame | None) -> pd.DataFrame:
    if signal_df is None or signal_df.empty:
        return empty_signal_df()

    clean_df = signal_df.copy()
    if "chrom" not in clean_df.columns and "chr" in clean_df.columns:
        clean_df = clean_df.rename(columns={"chr": "chrom"})

    missing_columns = [column for column in BEDGRAPH_COLUMNS if column not in clean_df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required signal columns {missing_columns} in {list(clean_df.columns)}"
        )

    clean_df = clean_df.loc[:, BEDGRAPH_COLUMNS].copy()
    clean_df["chrom"] = clean_df["chrom"].astype(str)
    clean_df["start"] = pd.to_numeric(clean_df["start"], errors="coerce")
    clean_df["end"] = pd.to_numeric(clean_df["end"], errors="coerce")
    clean_df["value"] = pd.to_numeric(clean_df["value"], errors="coerce")
    clean_df = clean_df.dropna(subset=["chrom", "start", "end", "value"]).copy()
    if clean_df.empty:
        return empty_signal_df()

    clean_df["start"] = clean_df["start"].astype(int)
    clean_df["end"] = clean_df["end"].astype(int)
    clean_df["value"] = clean_df["value"].astype(float)
    clean_df["start"] = clean_df["start"].clip(lower=0)
    clean_df = clean_df.loc[clean_df["end"] > clean_df["start"]].copy()
    if clean_df.empty:
        return empty_signal_df()

    return clean_df.loc[:, BEDGRAPH_COLUMNS].reset_index(drop=True)


def _load_regions_from_path(tool_config: dict, raw_path: str | Path) -> pd.DataFrame:
    parser_family = tool_config["parser_family"]
    raw_path = Path(raw_path)

    if parser_family == "methylseekr":
        region_df = pd.read_csv(raw_path, sep="\t")
    elif parser_family == "dnmtools":
        region_df = pd.read_csv(
            raw_path,
            sep="\t",
            header=None,
            names=["chr", "start", "end", "label", "score", "strand"],
        )
    elif parser_family == "methyl_lasso":
        region_df = pd.read_csv(
            raw_path,
            sep="\t",
            header=0,
            names=["chr", "start", "end", "num.cpgs", "meth", "std", "category"],
        )
        if "category" in region_df.columns:
            region_df = region_df.loc[region_df["category"].eq("PMD")].copy()
    elif parser_family == "mmseekr":
        region_df = pd.read_csv(
            raw_path,
            sep="\t",
            header=None,
            names=["chr", "start", "end"],
        )
    elif parser_family == "methylseg":
        region_df = pd.read_csv(
            raw_path,
            sep="\t",
            header=None,
            names=["chr", "start", "end", "label"],
        )
    else:
        raise ValueError(
            f"Unsupported parser family for {tool_config['tool']}: {parser_family}"
        )

    return ensure_interval_df(region_df)


def load_tool_regions(
    tool_config: dict,
    sample: str,
    segmentation_results_path: str | Path = DEFAULT_SEGMENTATION_RESULTS_PATH,
) -> pd.DataFrame:
    raw_path = build_tool_region_path(segmentation_results_path, sample, tool_config)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing region file for {sample} {tool_config['tool']}: {raw_path}"
        )
    return _load_regions_from_path(tool_config, raw_path)


def read_bed_intervals(bed_path: str | Path) -> pd.DataFrame:
    bed_path = Path(bed_path)
    if not bed_path.exists() or bed_path.stat().st_size == 0:
        return empty_interval_df()

    try:
        interval_df = pd.read_csv(
            bed_path,
            sep="\t",
            header=None,
            usecols=[0, 1, 2],
            names=BED_COLUMNS,
            comment="#",
        )
    except pd.errors.EmptyDataError:
        return empty_interval_df()

    return ensure_interval_df(interval_df)


def read_bedgraph_signal(signal_path: str | Path) -> pd.DataFrame:
    signal_path = Path(signal_path)
    if not signal_path.exists() or signal_path.stat().st_size == 0:
        return empty_signal_df()

    try:
        signal_df = pd.read_csv(
            signal_path,
            sep="\t",
            header=None,
            usecols=[0, 1, 2, 3],
            names=BEDGRAPH_COLUMNS,
            comment="#",
        )
    except pd.errors.EmptyDataError:
        return empty_signal_df()

    return ensure_signal_df(signal_df)


def clean_and_merge_intervals(
    interval_df: pd.DataFrame,
    canonical_chroms: tuple[str, ...] = CANONICAL_CHROMS,
) -> pd.DataFrame:
    clean_df = ensure_interval_df(interval_df)
    if clean_df.empty:
        return clean_df

    clean_df = clean_df.loc[clean_df["chrom"].isin(canonical_chroms)].copy()
    if clean_df.empty:
        return empty_interval_df()

    clean_df["_chrom_rank"] = clean_df["chrom"].map(CHROM_RANK)
    clean_df = clean_df.sort_values(
        ["_chrom_rank", "chrom", "start", "end"],
        kind="mergesort",
    ).reset_index(drop=True)

    merged_rows: list[dict[str, int | str]] = []
    for chrom, chrom_df in clean_df.groupby("chrom", sort=False):
        starts = chrom_df["start"].to_numpy(dtype=np.int64)
        ends = chrom_df["end"].to_numpy(dtype=np.int64)
        current_start = int(starts[0])
        current_end = int(ends[0])

        for start, end in zip(starts[1:], ends[1:]):
            start = int(start)
            end = int(end)
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged_rows.append(
                    {"chrom": chrom, "start": current_start, "end": current_end}
                )
                current_start = start
                current_end = end

        merged_rows.append({"chrom": chrom, "start": current_start, "end": current_end})

    merged_df = pd.DataFrame(merged_rows, columns=BED_COLUMNS)
    return ensure_interval_df(merged_df)


def _append_signal_row(
    rows: list[dict[str, float | int | str]],
    chrom: str,
    start: int,
    end: int,
    value: float,
) -> None:
    if end <= start or pd.isna(value):
        return

    start = int(start)
    end = int(end)
    value = float(value)
    if rows and rows[-1]["chrom"] == chrom:
        last_row = rows[-1]
        if int(last_row["end"]) == start and np.isclose(float(last_row["value"]), value):
            last_row["end"] = end
            return

    rows.append({"chrom": chrom, "start": start, "end": end, "value": value})


def clean_signal_df(
    signal_df: pd.DataFrame,
    chrom_sizes: dict[str, int] | None = None,
    canonical_chroms: tuple[str, ...] = CANONICAL_CHROMS,
) -> pd.DataFrame:
    clean_df = ensure_signal_df(signal_df)
    if clean_df.empty:
        return clean_df

    clean_df = clean_df.loc[clean_df["chrom"].isin(canonical_chroms)].copy()
    if clean_df.empty:
        return empty_signal_df()

    if chrom_sizes:
        clean_df = clean_df.loc[clean_df["chrom"].isin(chrom_sizes)].copy()
        if clean_df.empty:
            return empty_signal_df()
        clean_df["chrom_size"] = clean_df["chrom"].map(chrom_sizes).astype(int)
        clean_df["end"] = np.minimum(clean_df["end"], clean_df["chrom_size"])
        clean_df = clean_df.loc[clean_df["end"] > clean_df["start"], BEDGRAPH_COLUMNS].copy()
        if clean_df.empty:
            return empty_signal_df()

    clean_df["_chrom_rank"] = clean_df["chrom"].map(CHROM_RANK)
    clean_df = clean_df.sort_values(
        ["_chrom_rank", "chrom", "start", "end"],
        kind="mergesort",
    ).reset_index(drop=True)

    cleaned_rows: list[dict[str, float | int | str]] = []
    for row in clean_df.itertuples(index=False):
        chrom = str(row.chrom)
        start = int(row.start)
        end = int(row.end)
        value = float(row.value)

        if not cleaned_rows or cleaned_rows[-1]["chrom"] != chrom:
            _append_signal_row(cleaned_rows, chrom, start, end, value)
            continue

        last_row = cleaned_rows[-1]
        last_end = int(last_row["end"])
        if start >= last_end:
            _append_signal_row(cleaned_rows, chrom, start, end, value)
            continue

        # Lifted bedGraph fragments can overlap. Resolve them into a valid
        # non-overlapping signal track by splitting at the overlap and taking
        # the mean value across the shared segment.
        cleaned_rows.pop()
        last_start = int(last_row["start"])
        last_value = float(last_row["value"])
        overlap_end = min(last_end, end)

        if last_start < start:
            _append_signal_row(cleaned_rows, chrom, last_start, start, last_value)

        _append_signal_row(
            cleaned_rows,
            chrom,
            start,
            overlap_end,
            np.mean([last_value, value]),
        )

        if last_end > overlap_end:
            _append_signal_row(cleaned_rows, chrom, overlap_end, last_end, last_value)

        if end > overlap_end:
            _append_signal_row(cleaned_rows, chrom, overlap_end, end, value)

    return ensure_signal_df(pd.DataFrame(cleaned_rows, columns=BEDGRAPH_COLUMNS))


def write_bed(interval_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df = ensure_interval_df(interval_df)
    if clean_df.empty:
        output_path.write_text("")
        return output_path

    clean_df.loc[:, BED_COLUMNS].to_csv(output_path, sep="\t", header=False, index=False)
    return output_path


def write_bedgraph(signal_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df = ensure_signal_df(signal_df)
    if clean_df.empty:
        output_path.write_text("")
        return output_path

    clean_df.loc[:, BEDGRAPH_COLUMNS].to_csv(output_path, sep="\t", header=False, index=False)
    return output_path


def read_chrom_sizes(
    chrom_sizes_path: str | Path,
    canonical_only: bool = True,
) -> dict[str, int]:
    chrom_sizes_path = Path(chrom_sizes_path)
    if not chrom_sizes_path.exists():
        raise FileNotFoundError(f"Missing chromosome sizes file: {chrom_sizes_path}")

    sizes: dict[str, int] = {}
    with chrom_sizes_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            chrom, size = line.split("\t")[:2]
            if canonical_only and chrom not in CANONICAL_CHROMS:
                continue
            sizes[chrom] = int(size)

    if canonical_only:
        return {chrom: sizes[chrom] for chrom in CANONICAL_CHROMS if chrom in sizes}
    return sizes


def ensure_chrom_sizes(
    genome: str,
    output_path: str | Path,
    force: bool = False,
) -> tuple[dict[str, int], Path]:
    genome = str(genome).strip()
    if genome not in {"hg19", "hg38"}:
        raise ValueError(f"Unsupported genome for chrom sizes: {genome}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists() or force:
        url = f"{UCSC_DOWNLOAD_BASE}/{genome}/bigZips/{genome}.chrom.sizes"
        with urllib.request.urlopen(url) as response:
            output_path.write_bytes(response.read())

    chrom_sizes = read_chrom_sizes(output_path, canonical_only=True)
    if not chrom_sizes:
        raise RuntimeError(f"No canonical chromosome sizes were parsed from {output_path}")
    return chrom_sizes, output_path


def build_binary_signal_df(interval_df: pd.DataFrame, value: float = 1.0) -> pd.DataFrame:
    clean_df = clean_and_merge_intervals(interval_df)
    if clean_df.empty:
        return empty_signal_df()

    signal_df = clean_df.loc[:, BED_COLUMNS].copy()
    signal_df["value"] = float(value)
    return ensure_signal_df(signal_df)


def write_bigwig(
    signal_df: pd.DataFrame,
    chrom_sizes: dict[str, int],
    output_path: str | Path,
) -> Path:
    import pyBigWig

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df = clean_signal_df(signal_df, chrom_sizes=chrom_sizes)

    with pyBigWig.open(str(output_path), "w") as bw_handle:
        header = [
            (chrom, int(chrom_sizes[chrom]))
            for chrom in CANONICAL_CHROMS
            if chrom in chrom_sizes
        ]
        bw_handle.addHeader(header)
        if clean_df.empty:
            return output_path

        for chrom in [chrom for chrom in CANONICAL_CHROMS if chrom in set(clean_df["chrom"])]:
            chrom_df = clean_df.loc[clean_df["chrom"].eq(chrom), BEDGRAPH_COLUMNS]
            if chrom_df.empty:
                continue
            bw_handle.addEntries(
                chrom_df["chrom"].tolist(),
                chrom_df["start"].astype(int).tolist(),
                ends=chrom_df["end"].astype(int).tolist(),
                values=chrom_df["value"].astype(float).tolist(),
            )

    validate_bigwig(output_path)
    return output_path


def validate_bigwig(bw_path: str | Path) -> Path:
    import pyBigWig

    bw_path = Path(bw_path)
    if not bw_path.exists():
        raise FileNotFoundError(f"Missing bigWig file: {bw_path}")
    if bw_path.stat().st_size == 0:
        raise RuntimeError(f"bigWig file is empty: {bw_path}")

    with pyBigWig.open(str(bw_path)) as bw_handle:
        header = bw_handle.header() or {}
        chroms = bw_handle.chroms() or {}
        if not chroms:
            raise RuntimeError(f"bigWig contains no chromosome header entries: {bw_path}")

        n_bases_covered = int(header.get("nBasesCovered", 0) or 0)
        if n_bases_covered <= 0:
            raise RuntimeError(
                f"bigWig contains no covered bases and is likely invalid: {bw_path}"
            )

        for chrom in CANONICAL_CHROMS:
            chrom_size = chroms.get(chrom)
            if chrom_size is None or chrom_size <= 0:
                continue
            probe_end = min(int(chrom_size), 1)
            values = bw_handle.values(chrom, 0, probe_end)
            if values is None or len(values) != probe_end:
                raise RuntimeError(
                    "bigWig failed a random-access query and is likely truncated or malformed: "
                    f"{bw_path}"
                )
            break
        else:
            raise RuntimeError(
                f"bigWig does not expose any canonical chromosomes for validation: {bw_path}"
            )

    return bw_path


def run_liftover_bed(
    input_bed: str | Path,
    output_bed: str | Path,
    chain_file: str | Path,
    script_path: str | Path,
    force: bool = False,
) -> Path:
    input_bed = Path(input_bed)
    output_bed = Path(output_bed)
    chain_file = Path(chain_file)
    script_path = Path(script_path)

    if not input_bed.exists():
        raise FileNotFoundError(f"Missing BED input for liftover: {input_bed}")
    if not chain_file.exists():
        raise FileNotFoundError(f"Missing chain file for liftover: {chain_file}")
    if not script_path.exists():
        raise FileNotFoundError(f"Missing liftover helper script: {script_path}")

    if output_bed.exists() and not force:
        return output_bed

    output_bed.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "Rscript",
        str(script_path),
        "--input-bed",
        str(input_bed),
        "--output-bed",
        str(output_bed),
        "--chain-file",
        str(chain_file),
    ]
    subprocess.run(command, check=True)
    return output_bed


def build_region_end_windows(interval_df: pd.DataFrame, window_bp: int) -> pd.DataFrame:
    clean_df = clean_and_merge_intervals(interval_df).reset_index(drop=True)
    if clean_df.empty:
        return empty_end_window_df()

    window_bp = int(window_bp)
    if window_bp <= 0:
        raise ValueError(f"window_bp must be positive, received {window_bp}")

    window_rows: list[dict[str, int | str]] = []
    for region_id, row in enumerate(clean_df.itertuples(index=False)):
        start_window_end = min(int(row.end), int(row.start) + window_bp)
        end_window_start = max(int(row.start), int(row.end) - window_bp)

        if end_window_start <= start_window_end:
            window_rows.append(
                {
                    "region_id": region_id,
                    "chrom": row.chrom,
                    "start": int(row.start),
                    "end": int(row.end),
                    "window_label": "merged",
                    "length": int(row.end) - int(row.start),
                }
            )
            continue

        window_rows.append(
            {
                "region_id": region_id,
                "chrom": row.chrom,
                "start": int(row.start),
                "end": int(start_window_end),
                "window_label": "start",
                "length": int(start_window_end) - int(row.start),
            }
        )
        window_rows.append(
            {
                "region_id": region_id,
                "chrom": row.chrom,
                "start": int(end_window_start),
                "end": int(row.end),
                "window_label": "end",
                "length": int(row.end) - int(end_window_start),
            }
        )

    return pd.DataFrame(window_rows, columns=END_WINDOW_COLUMNS)


def _window_overlap_flags_for_chrom(
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    lad_starts: np.ndarray,
    lad_ends: np.ndarray,
) -> np.ndarray:
    flags = np.zeros(len(window_starts), dtype=bool)
    lad_idx = 0

    for window_idx, (window_start, window_end) in enumerate(zip(window_starts, window_ends)):
        window_start = int(window_start)
        window_end = int(window_end)

        while lad_idx < len(lad_ends) and int(lad_ends[lad_idx]) <= window_start:
            lad_idx += 1

        if lad_idx >= len(lad_starts):
            break

        if int(lad_starts[lad_idx]) < window_end and int(lad_ends[lad_idx]) > window_start:
            flags[window_idx] = True

    return flags


def _interval_overlap_pairs_for_chrom(
    query_starts: np.ndarray,
    query_ends: np.ndarray,
    target_starts: np.ndarray,
    target_ends: np.ndarray,
) -> list[tuple[int, int]]:
    overlap_pairs: list[tuple[int, int]] = []
    query_idx = 0
    target_idx = 0

    while query_idx < len(query_starts) and target_idx < len(target_starts):
        query_start = int(query_starts[query_idx])
        query_end = int(query_ends[query_idx])
        target_start = int(target_starts[target_idx])
        target_end = int(target_ends[target_idx])

        if query_end <= target_start:
            query_idx += 1
            continue
        if target_end <= query_start:
            target_idx += 1
            continue

        overlap_pairs.append((query_idx, target_idx))
        if query_end <= target_end:
            query_idx += 1
        else:
            target_idx += 1

    return overlap_pairs


def compute_region_lad_association_metrics(
    region_df: pd.DataFrame,
    lad_df: pd.DataFrame,
    window_bp: int,
) -> dict[str, float | int]:
    clean_region_df = clean_and_merge_intervals(region_df).reset_index(drop=True)
    clean_lad_df = clean_and_merge_intervals(lad_df).reset_index(drop=True)

    n_regions = int(len(clean_region_df))
    n_lad_regions = int(len(clean_lad_df))

    region_end_matches = np.zeros(n_regions, dtype=bool)
    region_same_lad_matches = np.zeros(n_regions, dtype=bool)
    lad_matches = np.zeros(n_lad_regions, dtype=bool)

    if n_regions > 0 and n_lad_regions > 0:
        end_window_df = build_region_end_windows(clean_region_df, window_bp)
        lad_lookup = {
            chrom: chrom_df
            for chrom, chrom_df in clean_lad_df.groupby("chrom", sort=False)
        }

        if not end_window_df.empty:
            for chrom, chrom_window_df in end_window_df.groupby("chrom", sort=False):
                lad_chrom_df = lad_lookup.get(chrom)
                if lad_chrom_df is None or lad_chrom_df.empty:
                    continue

                chrom_flags = _window_overlap_flags_for_chrom(
                    chrom_window_df["start"].to_numpy(dtype=np.int64),
                    chrom_window_df["end"].to_numpy(dtype=np.int64),
                    lad_chrom_df["start"].to_numpy(dtype=np.int64),
                    lad_chrom_df["end"].to_numpy(dtype=np.int64),
                )
                if chrom_flags.any():
                    matched_region_ids = chrom_window_df.loc[
                        chrom_flags, "region_id"
                    ].to_numpy(dtype=int)
                    region_end_matches[matched_region_ids] = True

        region_overlap_counts = np.zeros(n_regions, dtype=np.int64)
        lad_overlap_counts = np.zeros(n_lad_regions, dtype=np.int64)

        for chrom, chrom_region_df in clean_region_df.groupby("chrom", sort=False):
            lad_chrom_df = lad_lookup.get(chrom)
            if lad_chrom_df is None or lad_chrom_df.empty:
                continue

            region_indices = chrom_region_df.index.to_numpy(dtype=np.int64)
            lad_indices = lad_chrom_df.index.to_numpy(dtype=np.int64)
            overlap_pairs = _interval_overlap_pairs_for_chrom(
                chrom_region_df["start"].to_numpy(dtype=np.int64),
                chrom_region_df["end"].to_numpy(dtype=np.int64),
                lad_chrom_df["start"].to_numpy(dtype=np.int64),
                lad_chrom_df["end"].to_numpy(dtype=np.int64),
            )
            for local_region_idx, local_lad_idx in overlap_pairs:
                region_idx = int(region_indices[local_region_idx])
                lad_idx = int(lad_indices[local_lad_idx])
                region_overlap_counts[region_idx] += 1
                lad_overlap_counts[lad_idx] += 1

        if lad_overlap_counts.size:
            lad_matches = lad_overlap_counts > 0

        if region_overlap_counts.size and lad_overlap_counts.size:
            for chrom, chrom_region_df in clean_region_df.groupby("chrom", sort=False):
                lad_chrom_df = lad_lookup.get(chrom)
                if lad_chrom_df is None or lad_chrom_df.empty:
                    continue

                region_indices = chrom_region_df.index.to_numpy(dtype=np.int64)
                lad_indices = lad_chrom_df.index.to_numpy(dtype=np.int64)
                overlap_pairs = _interval_overlap_pairs_for_chrom(
                    chrom_region_df["start"].to_numpy(dtype=np.int64),
                    chrom_region_df["end"].to_numpy(dtype=np.int64),
                    lad_chrom_df["start"].to_numpy(dtype=np.int64),
                    lad_chrom_df["end"].to_numpy(dtype=np.int64),
                )
                for local_region_idx, local_lad_idx in overlap_pairs:
                    region_idx = int(region_indices[local_region_idx])
                    lad_idx = int(lad_indices[local_lad_idx])
                    if lad_overlap_counts[lad_idx] >= 2:
                        region_same_lad_matches[region_idx] = True

    matched_regions = int(region_end_matches.sum())
    unmatched_regions = int(n_regions - matched_regions)
    touched_lads = int(lad_matches.sum())
    untouched_lads = int(n_lad_regions - touched_lads)
    shared_lad_regions = int(region_same_lad_matches.sum())
    non_shared_lad_regions = int(n_regions - shared_lad_regions)

    return {
        "n_regions": n_regions,
        "regions_with_lad_at_either_end": matched_regions,
        "pct_regions_with_lad_at_either_end": (
            100.0 * matched_regions / n_regions if n_regions else np.nan
        ),
        "regions_without_lad_at_either_end": unmatched_regions,
        "n_lad_regions": n_lad_regions,
        "lad_regions_touched_by_regions": touched_lads,
        "pct_lad_regions_touched_by_regions": (
            100.0 * touched_lads / n_lad_regions if n_lad_regions else np.nan
        ),
        "lad_regions_not_touched_by_regions": untouched_lads,
        "regions_touching_same_lad": shared_lad_regions,
        "pct_regions_touching_same_lad": (
            100.0 * shared_lad_regions / n_regions if n_regions else np.nan
        ),
        "regions_not_touching_same_lad": non_shared_lad_regions,
    }


def compute_end_window_overlap_metrics(
    region_df: pd.DataFrame,
    lad_df: pd.DataFrame,
    window_bp: int,
) -> dict[str, float | int]:
    metrics = compute_region_lad_association_metrics(region_df, lad_df, window_bp)
    return {
        "n_regions": int(metrics["n_regions"]),
        "regions_with_lad_at_either_end": int(metrics["regions_with_lad_at_either_end"]),
        "pct_regions_with_lad_at_either_end": metrics["pct_regions_with_lad_at_either_end"],
        "regions_without_lad_at_either_end": int(metrics["regions_without_lad_at_either_end"]),
    }


def filter_regions_for_deeptools(
    interval_df: pd.DataFrame,
    min_length_bp: int,
) -> pd.DataFrame:
    clean_df = ensure_interval_df(interval_df)
    if clean_df.empty:
        return clean_df
    return (
        clean_df.loc[clean_df["length"] >= int(min_length_bp)]
        .reset_index(drop=True)
        .copy()
    )


def run_command(
    command: list[str | Path],
    force: bool = False,
    expected_outputs: list[str | Path] | None = None,
) -> None:
    rendered_command = [str(part) for part in command]
    expected_output_paths = [Path(path) for path in (expected_outputs or [])]
    if expected_output_paths and not force and all(path.exists() for path in expected_output_paths):
        print("Skipping existing output:", ", ".join(str(path) for path in expected_output_paths))
        return

    print("$", " ".join(shlex.quote(part) for part in rendered_command))
    subprocess.run(rendered_command, check=True)


def run_deeptools_profile(
    *,
    sample: str,
    sample_id: str,
    signal_path: str | Path,
    signal_label: str,
    plot_label: str,
    region_rows: list[dict],
    deeptools_tool_order: list[str],
    sample_output_dir: str | Path,
    file_stem: str,
    bin_size_bp: int,
    flank_bp: int,
    region_body_bp: int,
    include_heatmaps: bool = False,
    force: bool = False,
) -> dict[str, str | int]:
    sample_output_dir = Path(sample_output_dir)
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    region_rows_by_tool = {row["tool"]: row for row in region_rows}
    region_paths: list[str] = []
    region_labels: list[str] = []
    visualized_tools = 0

    for tool in deeptools_tool_order:
        row = region_rows_by_tool.get(tool)
        if row is None:
            raise AssertionError(f"Missing deepTools region row for {sample} {tool}")
        if int(row["visualized_regions"]) == 0:
            continue
        region_path = Path(row["deeptools_region_path"])
        if not region_path.exists():
            raise FileNotFoundError(
                f"Missing deepTools region BED for {sample} {tool}: {region_path}"
            )
        region_paths.append(str(region_path))
        region_labels.append(str(row["tool_label"]))
        visualized_tools += 1

    if not region_paths:
        raise RuntimeError(
            f"No regions remained for deepTools plotting in {sample} after minimum-length filtering."
        )

    matrix_path = sample_output_dir / f"{sample}.{file_stem}.matrix.gz"
    sorted_regions_path = sample_output_dir / f"{sample}.{file_stem}.sorted_regions.bed"
    profile_path = sample_output_dir / f"{sample}.{file_stem}.profile.png"
    heatmap_path = sample_output_dir / f"{sample}.{file_stem}.heatmap.png"

    compute_matrix_command = [
        "computeMatrix",
        "scale-regions",
        "-S",
        str(signal_path),
        "-R",
        *region_paths,
        "-b",
        str(int(flank_bp)),
        "-a",
        str(int(flank_bp)),
        "--binSize",
        str(int(bin_size_bp)),
        "--regionBodyLength",
        str(int(region_body_bp)),
        "--missingDataAsZero",
        "--sortRegions",
        "keep",
        "-o",
        str(matrix_path),
        "--outFileSortedRegions",
        str(sorted_regions_path),
    ]
    run_command(
        compute_matrix_command,
        force=force,
        expected_outputs=[matrix_path, sorted_regions_path],
    )

    profile_command = [
        "plotProfile",
        "--numPlotsPerRow",
        "3",
        "-m",
        str(matrix_path),
        "--perGroup",
        "--samplesLabel",
        str(signal_label),
        "--regionsLabel",
        *region_labels,
        "--startLabel",
        "Outside region",
        "--endLabel",
        "Outside region",
        "--plotTitle",
        f"{sample} {plot_label} across tool-derived regions",
        "-out",
        str(profile_path),
    ]
    run_command(
        profile_command,
        force=force,
        expected_outputs=[profile_path],
    )

    if include_heatmaps:
        heatmap_command = [
            "plotHeatmap",
            "-m",
            str(matrix_path),
            "--samplesLabel",
            str(signal_label),
            "--regionsLabel",
            *region_labels,
            "--startLabel",
            "Outside region",
            "--endLabel",
            "Outside region",
            "--plotTitle",
            f"{sample} {plot_label} heatmap across tool-derived regions",
            "--sortRegions",
            "keep",
            "--heatmapWidth",
            "14",
            "--heatmapHeight",
            "12",
            "--whatToShow",
            "heatmap and colorbar",
            "-out",
            str(heatmap_path),
        ]
        run_command(
            heatmap_command,
            force=force,
            expected_outputs=[heatmap_path],
        )

    return {
        "sample": sample,
        "sample_id": sample_id,
        "signal_label": str(signal_label),
        "matrix_path": str(matrix_path),
        "sorted_regions_path": str(sorted_regions_path),
        "profile_path": str(profile_path),
        "heatmap_path": str(heatmap_path) if include_heatmaps else "",
        "include_heatmaps": bool(include_heatmaps),
        "visualized_tools": int(visualized_tools),
        "bin_size_bp": int(bin_size_bp),
        "flank_bp": int(flank_bp),
        "region_body_bp": int(region_body_bp),
    }


def build_boundaries(interval_df: pd.DataFrame) -> dict[str, np.ndarray]:
    clean_df = clean_and_merge_intervals(interval_df)
    if clean_df.empty:
        return {}

    boundaries: dict[str, np.ndarray] = {}
    for chrom, chrom_df in clean_df.groupby("chrom", sort=False):
        chrom_boundaries = np.unique(
            np.concatenate(
                [
                    chrom_df["start"].to_numpy(dtype=np.int64),
                    chrom_df["end"].to_numpy(dtype=np.int64),
                ]
            )
        )
        boundaries[chrom] = np.sort(chrom_boundaries.astype(np.int64, copy=False))
    return boundaries


def _compute_overlap_bp_for_chrom(
    region_starts: np.ndarray,
    region_ends: np.ndarray,
    lad_starts: np.ndarray,
    lad_ends: np.ndarray,
) -> int:
    region_idx = 0
    lad_idx = 0
    overlap_bp = 0

    while region_idx < len(region_starts) and lad_idx < len(lad_starts):
        start = max(int(region_starts[region_idx]), int(lad_starts[lad_idx]))
        end = min(int(region_ends[region_idx]), int(lad_ends[lad_idx]))
        if end > start:
            overlap_bp += end - start

        if int(region_ends[region_idx]) <= int(lad_ends[lad_idx]):
            region_idx += 1
        else:
            lad_idx += 1

    return int(overlap_bp)


def compute_overlap_metrics(region_df: pd.DataFrame, lad_df: pd.DataFrame) -> dict[str, float | int]:
    region_df = clean_and_merge_intervals(region_df)
    lad_df = clean_and_merge_intervals(lad_df)

    region_total_bp = int(region_df["length"].sum()) if not region_df.empty else 0
    lad_total_bp = int(lad_df["length"].sum()) if not lad_df.empty else 0
    overlap_bp = 0

    if not region_df.empty and not lad_df.empty:
        region_lookup = {chrom: df for chrom, df in region_df.groupby("chrom", sort=False)}
        lad_lookup = {chrom: df for chrom, df in lad_df.groupby("chrom", sort=False)}

        for chrom in sorted(set(region_lookup) & set(lad_lookup), key=CHROM_RANK.get):
            region_chrom_df = region_lookup[chrom]
            lad_chrom_df = lad_lookup[chrom]
            overlap_bp += _compute_overlap_bp_for_chrom(
                region_chrom_df["start"].to_numpy(dtype=np.int64),
                region_chrom_df["end"].to_numpy(dtype=np.int64),
                lad_chrom_df["start"].to_numpy(dtype=np.int64),
                lad_chrom_df["end"].to_numpy(dtype=np.int64),
            )

    union_bp = int(region_total_bp + lad_total_bp - overlap_bp)
    return {
        "region_total_bp": region_total_bp,
        "lad_total_bp": lad_total_bp,
        "overlap_bp": int(overlap_bp),
        "union_bp": union_bp,
        "region_bp_in_lad_fraction": (
            overlap_bp / region_total_bp if region_total_bp else np.nan
        ),
        "lad_bp_covered_fraction": overlap_bp / lad_total_bp if lad_total_bp else np.nan,
        "jaccard": overlap_bp / union_bp if union_bp else np.nan,
    }


def _nearest_distances(query_positions: np.ndarray, target_positions: np.ndarray) -> np.ndarray:
    query_positions = np.asarray(query_positions, dtype=np.int64)
    target_positions = np.asarray(target_positions, dtype=np.int64)

    if query_positions.size == 0:
        return np.array([], dtype=float)
    if target_positions.size == 0:
        return np.full(query_positions.shape, np.inf, dtype=float)

    insert_idx = np.searchsorted(target_positions, query_positions, side="left")

    left_dist = np.full(query_positions.shape, np.inf, dtype=float)
    right_dist = np.full(query_positions.shape, np.inf, dtype=float)

    has_left = insert_idx > 0
    has_right = insert_idx < target_positions.size

    if np.any(has_left):
        left_idx = insert_idx[has_left] - 1
        left_dist[has_left] = np.abs(
            query_positions[has_left] - target_positions[left_idx]
        ).astype(float)
    if np.any(has_right):
        right_idx = insert_idx[has_right]
        right_dist[has_right] = np.abs(
            target_positions[right_idx] - query_positions[has_right]
        ).astype(float)

    return np.minimum(left_dist, right_dist)


def _collect_reciprocal_distances(
    source_boundaries: dict[str, np.ndarray],
    target_boundaries: dict[str, np.ndarray],
) -> np.ndarray:
    distance_parts = []
    for chrom in sorted(source_boundaries, key=CHROM_RANK.get):
        source_positions = source_boundaries[chrom]
        target_positions = target_boundaries.get(chrom, np.array([], dtype=np.int64))
        distance_parts.append(_nearest_distances(source_positions, target_positions))

    if not distance_parts:
        return np.array([], dtype=float)
    return np.concatenate(distance_parts).astype(float, copy=False)


def _finite_median(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan
    return float(np.median(finite_values))


def compute_boundary_metrics(
    region_boundaries: dict[str, np.ndarray],
    lad_boundaries: dict[str, np.ndarray],
    thresholds_bp: list[int] | tuple[int, ...],
) -> dict[str, float | int]:
    region_to_lad = _collect_reciprocal_distances(region_boundaries, lad_boundaries)
    lad_to_region = _collect_reciprocal_distances(lad_boundaries, region_boundaries)

    metrics: dict[str, float | int] = {
        "n_region_boundaries": int(sum(len(values) for values in region_boundaries.values())),
        "n_lad_boundaries": int(sum(len(values) for values in lad_boundaries.values())),
        "median_tool_to_lad_distance_bp": _finite_median(region_to_lad),
        "median_lad_to_tool_distance_bp": _finite_median(lad_to_region),
    }

    for threshold_bp in thresholds_bp:
        threshold_bp = int(threshold_bp)
        label = threshold_label(threshold_bp)
        precision = (
            float(np.mean(region_to_lad <= threshold_bp))
            if region_to_lad.size
            else np.nan
        )
        recall = (
            float(np.mean(lad_to_region <= threshold_bp))
            if lad_to_region.size
            else np.nan
        )

        if pd.isna(precision) or pd.isna(recall) or (precision + recall) == 0:
            f1 = np.nan
        else:
            f1 = 2.0 * precision * recall / (precision + recall)

        metrics[f"boundary_precision_{label}"] = precision
        metrics[f"boundary_recall_{label}"] = recall
        metrics[f"boundary_f1_{label}"] = f1

    return metrics
