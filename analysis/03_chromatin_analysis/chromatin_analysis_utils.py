import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
import pybedtools
from pybedtools import BedTool


BED_COLUMNS = ["chrom", "start", "end"]
INTERVAL_COLUMNS = BED_COLUMNS + ["length"]


def configure_pybedtools():
    bedtools_bin = Path(sys.executable).resolve().parent / "bedtools"
    if not bedtools_bin.exists():
        fallback_bedtools = shutil.which("bedtools")
        if fallback_bedtools:
            bedtools_bin = Path(fallback_bedtools)
    if bedtools_bin.exists():
        pybedtools.helpers.set_bedtools_path(str(bedtools_bin.parent))
        return bedtools_bin
    return None


def empty_interval_df():
    return pd.DataFrame(columns=INTERVAL_COLUMNS)


def sample_to_sample_id(sample):
    return sample.replace(".wgbs", "")


def build_tool_region_path(segmentation_results_path, sample, tool_config):
    path = Path(segmentation_results_path)
    for part in tool_config["path_parts"]:
        path = path / part.format(sample=sample)
    return path


def resolve_bigwig_path(chromatin_data_dir, sample):
    sample_id = sample_to_sample_id(sample)
    return Path(chromatin_data_dir) / f"{sample_id}.h3k36me2.bw"


def get_eligible_chrom_sizes(canonical_chromosomes, bw_path):
    with pyBigWig.open(str(bw_path)) as bw_handle:
        chrom_sizes = bw_handle.chroms()
    return {chrom: int(chrom_sizes[chrom]) for chrom in canonical_chromosomes if chrom in chrom_sizes}


def ensure_interval_df(interval_df):
    if interval_df.empty:
        return empty_interval_df()

    clean_df = interval_df.copy()
    if "chrom" not in clean_df.columns and "chr" in clean_df.columns:
        clean_df = clean_df.rename(columns={"chr": "chrom"})

    missing_columns = [column for column in BED_COLUMNS if column not in clean_df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required BED columns {missing_columns} in columns {list(clean_df.columns)}"
        )

    clean_df = clean_df.loc[:, BED_COLUMNS].copy()
    clean_df["start"] = pd.to_numeric(clean_df["start"], errors="coerce")
    clean_df["end"] = pd.to_numeric(clean_df["end"], errors="coerce")
    clean_df = clean_df.dropna().copy()
    if clean_df.empty:
        return empty_interval_df()

    clean_df["chrom"] = clean_df["chrom"].astype(str)
    clean_df["start"] = clean_df["start"].astype(int)
    clean_df["end"] = clean_df["end"].astype(int)
    clean_df = clean_df.loc[clean_df["end"] > clean_df["start"]].copy()
    if clean_df.empty:
        return empty_interval_df()

    clean_df["length"] = clean_df["end"] - clean_df["start"]
    return clean_df.loc[:, INTERVAL_COLUMNS].reset_index(drop=True)


def load_regions(tool_config, raw_path):
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
        region_df = pd.read_csv(raw_path, sep="\t", header=None, names=["chr", "start", "end"])
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


def read_bed_interval_df(bed_path):
    bed_path = Path(bed_path)
    if not bed_path.exists() or bed_path.stat().st_size == 0:
        return empty_interval_df()
    try:
        interval_df = pd.read_csv(bed_path, sep="\t", header=None, names=BED_COLUMNS)
    except pd.errors.EmptyDataError:
        return empty_interval_df()
    return ensure_interval_df(interval_df)


def merge_intervals(interval_df):
    configure_pybedtools()
    if interval_df.empty:
        return empty_interval_df()

    merged_bed = BedTool.from_dataframe(interval_df.loc[:, BED_COLUMNS]).sort().merge()
    try:
        merged_df = merged_bed.to_dataframe(names=BED_COLUMNS)
    except pd.errors.EmptyDataError:
        return empty_interval_df()

    if merged_df.empty:
        return empty_interval_df()
    return ensure_interval_df(merged_df)


def clean_interval_df(interval_df, chrom_sizes):
    if interval_df.empty:
        return empty_interval_df()

    clean_df = ensure_interval_df(interval_df)
    clean_df = clean_df.loc[clean_df["chrom"].isin(chrom_sizes)].copy()
    if clean_df.empty:
        return empty_interval_df()

    clean_df["chrom_size"] = clean_df["chrom"].map(chrom_sizes).astype(int)
    clean_df["start"] = clean_df["start"].clip(lower=0)
    clean_df["end"] = np.minimum(clean_df["end"], clean_df["chrom_size"])
    clean_df = clean_df.loc[clean_df["end"] > clean_df["start"], BED_COLUMNS].copy()
    if clean_df.empty:
        return empty_interval_df()

    return merge_intervals(clean_df)


def write_bed(interval_df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df = ensure_interval_df(interval_df)
    if clean_df.empty:
        output_path.write_text("")
        return
    clean_df.loc[:, BED_COLUMNS].to_csv(output_path, sep="\t", header=False, index=False)


def filter_regions_for_deeptools(interval_df, min_length_bp):
    if interval_df.empty:
        return interval_df.copy()
    return interval_df.loc[interval_df["length"] >= int(min_length_bp)].reset_index(drop=True).copy()


def run_command(command, expected_outputs=None):
    rendered_command = [str(part) for part in command]
    expected_outputs = [Path(path) for path in (expected_outputs or [])]
    print("$", " ".join(shlex.quote(part) for part in rendered_command))
    subprocess.run(rendered_command, check=True)
    for output_path in expected_outputs:
        if not output_path.exists():
            raise FileNotFoundError(f"Expected output was not created: {output_path}")


def get_deeptools_processor_count():
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(int(slurm_cpus) // 2, 1)
        except ValueError:
            pass
    return 1


def prepare_clean_region_task(task):
    configure_pybedtools()
    sample = task["sample"]
    tool_config = task["tool_config"]
    raw_region_path = build_tool_region_path(task["segmentation_results_path"], sample, tool_config)
    if not raw_region_path.exists():
        raise FileNotFoundError(
            f"Missing region file for {sample} {tool_config['tool']}: {raw_region_path}"
        )

    bw_path = resolve_bigwig_path(task["chromatin_data_dir"], sample)
    if not bw_path.exists():
        raise FileNotFoundError(f"Missing chromatin bigWig for {sample}: {bw_path}")

    chrom_sizes = get_eligible_chrom_sizes(task["canonical_chromosomes"], bw_path)
    if not chrom_sizes:
        raise RuntimeError(f"No eligible canonical chromosomes found in {bw_path}")

    clean_region_path = Path(task["cleaned_region_dir"]) / f"{sample}.{tool_config['tool']}.bed"
    raw_region_df = load_regions(tool_config, raw_region_path)
    clean_region_df = clean_interval_df(raw_region_df, chrom_sizes)
    if clean_region_df.empty:
        raise AssertionError(
            f"{sample} {tool_config['tool']} produced zero retained regions after cleaning."
        )
    write_bed(clean_region_df, clean_region_path)

    return {
        "sample": sample,
        "sample_id": sample_to_sample_id(sample),
        "tool": tool_config["tool"],
        "tool_label": tool_config["tool_label"],
        "tool_family": tool_config["tool_family"],
        "platform": tool_config["platform"],
        "region_type": tool_config["region_type"],
        "deeptools_order": int(tool_config["deeptools_order"]),
        "raw_region_path": str(raw_region_path),
        "clean_region_path": str(clean_region_path),
        "bw_path": str(bw_path),
        "n_regions": int(len(clean_region_df)),
        "total_bp": int(clean_region_df["length"].sum()),
        "median_region_bp": float(clean_region_df["length"].median()),
    }


def prepare_deeptools_regions_task(task):
    sample = task["sample"]
    tool = task["tool"]
    clean_region_path = Path(task["clean_region_path"])
    sample_output_dir = Path(task["deeptools_dir"]) / sample
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    region_df = read_bed_interval_df(clean_region_path)
    filtered_region_df = filter_regions_for_deeptools(region_df, task["deeptools_bin_size"])
    filtered_path = sample_output_dir / f"{sample}.{tool}.deeptools_regions.bed"
    write_bed(filtered_region_df, filtered_path)

    total_regions = int(len(region_df))
    visualized_regions = int(len(filtered_region_df))
    return {
        "sample": sample,
        "sample_id": task["sample_id"],
        "tool": tool,
        "tool_label": task["tool_label"],
        "tool_family": task["tool_family"],
        "platform": task["platform"],
        "region_type": task["region_type"],
        "deeptools_order": int(task["deeptools_order"]),
        "clean_region_path": str(clean_region_path),
        "deeptools_region_path": str(filtered_path),
        "total_regions": total_regions,
        "visualized_regions": visualized_regions,
        "excluded_short_regions": total_regions - visualized_regions,
        "min_region_length_bp": int(task["deeptools_bin_size"]),
    }


def run_deeptools_for_sample_task(task):
    sample = task["sample"]
    sample_output_dir = Path(task["deeptools_dir"]) / sample
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    deeptools_processors = get_deeptools_processor_count()

    region_rows_by_tool = {row["tool"]: row for row in task["region_rows"]}
    region_paths = []
    region_labels = []
    for tool in task["deeptools_tool_order"]:
        row = region_rows_by_tool.get(tool)
        if row is None:
            raise AssertionError(f"Missing deepTools region manifest row for {sample} {tool}")
        if int(row["visualized_regions"]) == 0:
            print(
                f"Skipping {sample} {tool} for deepTools because no regions were at least "
                f"{task['deeptools_bin_size']} bp."
            )
            continue
        region_path = Path(row["deeptools_region_path"])
        if not region_path.exists():
            raise FileNotFoundError(
                f"Missing deepTools region BED for {sample} {tool}: {region_path}"
            )
        region_paths.append(str(region_path))
        region_labels.append(row["tool_label"])

    if not region_paths:
        raise RuntimeError(
            f"No regions remained for deepTools in {sample} after filtering out regions shorter "
            f"than {task['deeptools_bin_size']} bp."
        )

    matrix_path = sample_output_dir / f"{sample}.h3k36me2.matrix.gz"
    matrix_values_path = sample_output_dir / f"{sample}.h3k36me2.matrix.tsv"
    sorted_regions_path = sample_output_dir / f"{sample}.h3k36me2.sorted_regions.bed"
    profile_path = sample_output_dir / f"{sample}.h3k36me2.profile.png"
    include_heatmaps = bool(task.get("include_heatmaps", False))
    heatmap_path = sample_output_dir / f"{sample}.h3k36me2.heatmap.png"

    compute_matrix_command = [
        "computeMatrix",
        "scale-regions",
        "-p",
        str(deeptools_processors),
        "-S",
        task["bw_path"],
        "-R",
        *region_paths,
        "-b",
        str(task["flank_length"]),
        "-a",
        str(task["flank_length"]),
        "--binSize",
        str(task["deeptools_bin_size"]),
        "--regionBodyLength",
        str(task["region_body_length"]),
        "--missingDataAsZero",
        "--sortRegions",
        "keep",
        "-o",
        str(matrix_path),
        "--outFileSortedRegions",
        str(sorted_regions_path),
        "--outFileNameMatrix",
        str(matrix_values_path),
    ]
    run_command(
        compute_matrix_command,
        expected_outputs=[matrix_path, matrix_values_path, sorted_regions_path],
    )

    profile_command = [
        "plotProfile",
        "--numPlotsPerRow",
        "3",
        "-m",
        str(matrix_path),
        "--perGroup",
        "--samplesLabel",
        task["sample_id"],
        "--regionsLabel",
        *region_labels,
        "--startLabel",
        "Region start",
        "--endLabel",
        "Region end",
        "--plotTitle",
        f"{sample} H3K36me2 profile across tool-derived regions",
        "-out",
        str(profile_path),
    ]
    run_command(profile_command, expected_outputs=[profile_path])

    if include_heatmaps:
        heatmap_command = [
            "plotHeatmap",
            "-m",
            str(matrix_path),
            "--samplesLabel",
            task["sample_id"],
            "--regionsLabel",
            *region_labels,
            "--startLabel",
            "Region start",
            "--endLabel",
            "Region end",
            "--plotTitle",
            f"{sample} H3K36me2 heatmap across tool-derived regions",
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
        run_command(heatmap_command, expected_outputs=[heatmap_path])

    return {
        "sample": sample,
        "sample_id": task["sample_id"],
        "bw_path": str(task["bw_path"]),
        "matrix_path": str(matrix_path),
        "matrix_values_path": str(matrix_values_path),
        "sorted_regions_path": str(sorted_regions_path),
        "profile_path": str(profile_path),
        "heatmap_path": str(heatmap_path) if include_heatmaps else "",
        "include_heatmaps": include_heatmaps,
        "deeptools_bin_size": int(task["deeptools_bin_size"]),
        "flank_length": int(task["flank_length"]),
        "region_body_length": int(task["region_body_length"]),
        "region_bed_paths": json.dumps(region_paths),
        "region_labels": json.dumps(region_labels),
    }
