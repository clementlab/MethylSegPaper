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


def build_tool_region_path(segmentation_results_path, sample, tool_config, region_variant=None):
    region_variant = region_variant or tool_config.get("default_region_variant", "source-default")
    region_paths = tool_config.get("region_paths")
    if region_paths is None:
        raise ValueError(f"Missing region_paths for {tool_config['tool']}")
    if region_variant not in region_paths:
        raise ValueError(
            f"Unsupported region variant {region_variant!r} for {tool_config['tool']}. "
            f"Supported variants: {sorted(region_paths)}"
        )
    path = Path(segmentation_results_path)
    for part in region_paths[region_variant]:
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


def normalize_interval_df(interval_df, chrom_sizes, merge_overlaps=False):
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

    normalized_df = ensure_interval_df(clean_df)
    if merge_overlaps:
        return merge_intervals(normalized_df)
    return normalized_df


def clean_interval_df(interval_df, chrom_sizes):
    return normalize_interval_df(interval_df, chrom_sizes, merge_overlaps=True)


def normalize_raw_interval_df(interval_df, chrom_sizes):
    return normalize_interval_df(interval_df, chrom_sizes, merge_overlaps=False)


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


def build_source_id(tool, region_variant):
    return f"{tool}__{region_variant}"


def format_tool_variant_label(tool_label, region_variant):
    if region_variant == "raw":
        return f"{tool_label} Raw"
    if region_variant == "cleaned" and tool_label.startswith("MethylSeg"):
        return f"{tool_label} Cleaned"
    return tool_label


def prepare_region_task(task):
    configure_pybedtools()
    sample = task["sample"]
    tool_config = task["tool_config"]
    region_variant = task["region_variant"]
    raw_region_path = build_tool_region_path(
        task["segmentation_results_path"],
        sample,
        tool_config,
        region_variant=region_variant,
    )
    if not raw_region_path.exists():
        raise FileNotFoundError(
            f"Missing region file for {sample} {tool_config['tool']} {region_variant}: {raw_region_path}"
        )

    bw_path = resolve_bigwig_path(task["chromatin_data_dir"], sample)
    if not bw_path.exists():
        raise FileNotFoundError(f"Missing chromatin bigWig for {sample}: {bw_path}")

    chrom_sizes = get_eligible_chrom_sizes(task["canonical_chromosomes"], bw_path)
    if not chrom_sizes:
        raise RuntimeError(f"No eligible canonical chromosomes found in {bw_path}")

    prepared_region_path = (
        Path(task["prepared_region_dir"]) / f"{sample}.{tool_config['tool']}.{region_variant}.bed"
    )
    raw_region_df = load_regions(tool_config, raw_region_path)
    if region_variant in {"cleaned", "source-default"}:
        prepared_region_df = clean_interval_df(raw_region_df, chrom_sizes)
    elif region_variant == "raw":
        prepared_region_df = normalize_raw_interval_df(raw_region_df, chrom_sizes)
    else:
        prepared_region_df = normalize_raw_interval_df(raw_region_df, chrom_sizes)
    if prepared_region_df.empty:
        raise AssertionError(
            f"{sample} {tool_config['tool']} {region_variant} produced zero retained regions after preparation."
        )
    write_bed(prepared_region_df, prepared_region_path)

    return {
        "sample": sample,
        "sample_id": sample_to_sample_id(sample),
        "tool": tool_config["tool"],
        "source_id": build_source_id(tool_config["tool"], region_variant),
        "tool_label": tool_config["tool_label"],
        "tool_variant_label": format_tool_variant_label(tool_config["tool_label"], region_variant),
        "tool_family": tool_config["tool_family"],
        "platform": tool_config["platform"],
        "region_type": tool_config["region_type"],
        "source_order": int(task["source_order"]),
        "region_variant": region_variant,
        "source_region_path": str(raw_region_path),
        "prepared_region_path": str(prepared_region_path),
        "bw_path": str(bw_path),
        "n_regions": int(len(prepared_region_df)),
        "total_bp": int(prepared_region_df["length"].sum()),
        "median_region_bp": float(prepared_region_df["length"].median()),
    }


def prepare_deeptools_regions_task(task):
    sample = task["sample"]
    source_id = task["source_id"]
    prepared_region_path = Path(task["prepared_region_path"])
    sample_output_dir = Path(task["deeptools_dir"]) / sample / source_id
    sample_output_dir.mkdir(parents=True, exist_ok=True)

    region_df = read_bed_interval_df(prepared_region_path)
    filtered_region_df = filter_regions_for_deeptools(region_df, task["deeptools_bin_size"])
    filtered_path = sample_output_dir / f"{sample}.{source_id}.deeptools_regions.bed"
    write_bed(filtered_region_df, filtered_path)

    total_regions = int(len(region_df))
    visualized_regions = int(len(filtered_region_df))
    return {
        "sample": sample,
        "sample_id": task["sample_id"],
        "source_order": int(task["source_order"]),
        "tool": task["tool"],
        "source_id": source_id,
        "tool_label": task["tool_label"],
        "tool_variant_label": task["tool_variant_label"],
        "tool_family": task["tool_family"],
        "platform": task["platform"],
        "region_type": task["region_type"],
        "region_variant": task["region_variant"],
        "source_region_path": task["source_region_path"],
        "prepared_region_path": str(prepared_region_path),
        "deeptools_region_path": str(filtered_path),
        "total_regions": total_regions,
        "visualized_regions": visualized_regions,
        "excluded_short_regions": total_regions - visualized_regions,
        "min_region_length_bp": int(task["deeptools_bin_size"]),
        "flank_length": int(task["flank_length"]),
        "region_body_length": int(task["region_body_length"]),
    }


def run_deeptools_for_source_task(task):
    sample = task["sample"]
    source_id = task["source_id"]
    sample_output_dir = Path(task["deeptools_dir"]) / sample / source_id
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    deeptools_processors = get_deeptools_processor_count()

    if int(task["visualized_regions"]) == 0:
        raise RuntimeError(
            f"{sample} {source_id} had zero regions at least {task['deeptools_bin_size']} bp "
            "after deepTools filtering."
        )

    region_path = Path(task["deeptools_region_path"])
    if not region_path.exists():
        raise FileNotFoundError(
            f"Missing deepTools region BED for {sample} {source_id}: {region_path}"
        )

    matrix_path = sample_output_dir / f"{sample}.{source_id}.h3k36me2.matrix.gz"
    matrix_values_path = sample_output_dir / f"{sample}.{source_id}.h3k36me2.matrix.tsv"
    sorted_regions_path = sample_output_dir / f"{sample}.{source_id}.h3k36me2.sorted_regions.bed"
    profile_path = sample_output_dir / f"{sample}.{source_id}.h3k36me2.profile.png"
    include_heatmaps = bool(task.get("include_heatmaps", False))
    heatmap_path = sample_output_dir / f"{sample}.{source_id}.h3k36me2.heatmap.png"

    compute_matrix_command = [
        "computeMatrix",
        "scale-regions",
        "-p",
        str(deeptools_processors),
        "-S",
        task["bw_path"],
        "-R",
        str(region_path),
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
        "1",
        "-m",
        str(matrix_path),
        "--perGroup",
        "--samplesLabel",
        task["sample_id"],
        "--regionsLabel",
        task["tool_variant_label"],
        "--startLabel",
        "Region start",
        "--endLabel",
        "Region end",
        "--plotTitle",
        task["profile_title"],
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
            task["tool_variant_label"],
            "--startLabel",
            "Region start",
            "--endLabel",
            "Region end",
            "--plotTitle",
            task["heatmap_title"],
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
        "source_order": int(task["source_order"]),
        "tool": task["tool"],
        "source_id": source_id,
        "tool_label": task["tool_label"],
        "tool_variant_label": task["tool_variant_label"],
        "tool_family": task["tool_family"],
        "platform": task["platform"],
        "region_type": task["region_type"],
        "region_variant": task["region_variant"],
        "source_region_path": task["source_region_path"],
        "prepared_region_path": task["prepared_region_path"],
        "deeptools_region_path": task["deeptools_region_path"],
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
        "profile_title": task["profile_title"],
        "heatmap_title": task["heatmap_title"],
        "region_bed_path": str(region_path),
        "region_label": task["tool_variant_label"],
        "region_bed_paths": json.dumps([str(region_path)]),
        "region_labels": json.dumps([task["tool_variant_label"]]),
    }
