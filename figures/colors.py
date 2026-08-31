from __future__ import annotations

from collections.abc import Iterable, Sequence

COLOR_PALETTE = [
    "#0B5394",
    "#71B5D2",
    "#009966",
    "#B81E05",
    "#FF7F50",
    "#F3BB33",
    "#FFD700",
    "#FFBFA0",
    "#FFA1A8",
    "#9AA0A6",
    "#FFFFFF",
    "#202020",
]

PRIMARY = COLOR_PALETTE[0]
SECONDARY = COLOR_PALETTE[1]
GRAY = COLOR_PALETTE[9]
WHITE = COLOR_PALETTE[10]
BLACK = COLOR_PALETTE[11]


# Added during figure color centralization.
METHYLSEG_STATE_COLORS = {
    "LOW": COLOR_PALETTE[0],
    "PMD": COLOR_PALETTE[2],
    "INTERMEDIATE": COLOR_PALETTE[7],
    "HIGH": COLOR_PALETTE[3],
}

# Added during figure color centralization.
METHYLSEG_STATE_COLORS_BY_VALUE = {
    0: METHYLSEG_STATE_COLORS["LOW"],
    1: METHYLSEG_STATE_COLORS["PMD"],
    2: METHYLSEG_STATE_COLORS["INTERMEDIATE"],
    3: METHYLSEG_STATE_COLORS["HIGH"],
}

LOW = METHYLSEG_STATE_COLORS["LOW"]
PMD = METHYLSEG_STATE_COLORS["PMD"]
INTERMEDIATE = METHYLSEG_STATE_COLORS["INTERMEDIATE"]
HIGH = METHYLSEG_STATE_COLORS["HIGH"]


TOOL_HIGHLIGHT_COLORS = {
    "MethylSeg WGBS": COLOR_PALETTE[0],
    "MethylSeg HM450K": COLOR_PALETTE[1],
    "MethylSeg WGBS Raw": COLOR_PALETTE[0],
    "MethylSeg WGBS Cleaned": COLOR_PALETTE[1],
    "MethylSeg HM450K Raw": COLOR_PALETTE[1],
    "MethylSeg HM450K Cleaned": COLOR_PALETTE[1],
    "MethylSeekR": GRAY,
    "DNMTools": GRAY,
    "DNMTools Array": GRAY,
    "DNMTools PMR": GRAY,
    "MMSeekR": GRAY,
    "MethylLasso": GRAY,
    "Comparator": GRAY,
}

# Added during figure color centralization.
TOOL_HIGHLIGHT_COLORS_BY_SLUG = {
    "methylseg": TOOL_HIGHLIGHT_COLORS["MethylSeg WGBS"],
    "methylseg_wgbs": TOOL_HIGHLIGHT_COLORS["MethylSeg WGBS"],
    "methylseg_hm450k": TOOL_HIGHLIGHT_COLORS["MethylSeg HM450K"],
    "methylseekr": TOOL_HIGHLIGHT_COLORS["MethylSeekR"],
    "dnmtools": TOOL_HIGHLIGHT_COLORS["DNMTools"],
    "dnmtools_array": TOOL_HIGHLIGHT_COLORS["DNMTools Array"],
    "dnmtools_pmr": TOOL_HIGHLIGHT_COLORS["DNMTools PMR"],
    "mmseekr": TOOL_HIGHLIGHT_COLORS["MMSeekR"],
    "methyl_lasso": TOOL_HIGHLIGHT_COLORS["MethylLasso"],
    "methylasso": TOOL_HIGHLIGHT_COLORS["MethylLasso"],
}

# Added during figure color centralization.
TOOL_DISTINCT_COLORS = {
    "MethylSeg WGBS": COLOR_PALETTE[0],
    "MethylSeg HM450K": COLOR_PALETTE[1],
    "MethylSeekR": COLOR_PALETTE[2],
    "DNMTools": COLOR_PALETTE[4],
    "DNMTools Array": COLOR_PALETTE[5],
    "DNMTools PMR": COLOR_PALETTE[6],
    "MMSeekR": COLOR_PALETTE[3],
    "MethylLasso": COLOR_PALETTE[7],
}

# Added during figure color centralization.
TOOL_DISTINCT_COLORS_BY_SLUG = {
    "methylseg": TOOL_DISTINCT_COLORS["MethylSeg WGBS"],
    "methylseg_wgbs": TOOL_DISTINCT_COLORS["MethylSeg WGBS"],
    "methylseg_hm450k": TOOL_DISTINCT_COLORS["MethylSeg HM450K"],
    "methylseekr": TOOL_DISTINCT_COLORS["MethylSeekR"],
    "dnmtools": TOOL_DISTINCT_COLORS["DNMTools"],
    "dnmtools_array": TOOL_DISTINCT_COLORS["DNMTools Array"],
    "dnmtools_pmr": TOOL_DISTINCT_COLORS["DNMTools PMR"],
    "mmseekr": TOOL_DISTINCT_COLORS["MMSeekR"],
    "methyl_lasso": TOOL_DISTINCT_COLORS["MethylLasso"],
    "methylasso": TOOL_DISTINCT_COLORS["MethylLasso"],
}

# Added during figure color centralization.
LAD_TOOL_COLORS = {
    "methylseg": TOOL_DISTINCT_COLORS_BY_SLUG["methylseg"],
    "methylseg_hm450k": TOOL_DISTINCT_COLORS_BY_SLUG["methylseg_hm450k"],
    "methylseekr": TOOL_DISTINCT_COLORS_BY_SLUG["methylseekr"],
    "dnmtools": TOOL_DISTINCT_COLORS_BY_SLUG["dnmtools"],
    "dnmtools_array": TOOL_DISTINCT_COLORS_BY_SLUG["dnmtools_array"],
    "dnmtools_pmr": TOOL_DISTINCT_COLORS_BY_SLUG["dnmtools_pmr"],
    "mmseekr": TOOL_DISTINCT_COLORS_BY_SLUG["mmseekr"],
    "methyl_lasso": TOOL_DISTINCT_COLORS_BY_SLUG["methyl_lasso"],
}

METHYLSEG_WGBS = TOOL_HIGHLIGHT_COLORS["MethylSeg WGBS"]
METHYLSEG_HM450K = TOOL_HIGHLIGHT_COLORS["MethylSeg HM450K"]
DNMTOOLS_WGBS = TOOL_HIGHLIGHT_COLORS["DNMTools"]
DNMTOOLS_ARRAY = TOOL_HIGHLIGHT_COLORS["DNMTools Array"]
DNMTOOLS_PMR = TOOL_HIGHLIGHT_COLORS["DNMTools PMR"]
MMSEEKR = TOOL_HIGHLIGHT_COLORS["MMSeekR"]
METHYLASSO = TOOL_HIGHLIGHT_COLORS["MethylLasso"]
METHYLSEEKR = TOOL_HIGHLIGHT_COLORS["MethylSeekR"]


# Added during figure color centralization.
CHROMATIN_PROFILE_COLORS = {
    "MethylSeg WGBS": METHYLSEG_WGBS,
    "MethylSeg HM450K": METHYLSEG_HM450K,
    "MethylSeg WGBS Raw": METHYLSEG_WGBS,
    "MethylSeg WGBS Cleaned": METHYLSEG_WGBS,
    "MethylSeg HM450K Raw": METHYLSEG_HM450K,
    "MethylSeg HM450K Cleaned": METHYLSEG_HM450K,
    "MethylSeekR": METHYLSEEKR,
    "DNMTools": DNMTOOLS_WGBS,
    "DNMTools Array": DNMTOOLS_ARRAY,
    "DNMTools PMR": DNMTOOLS_PMR,
    "MMSeekR": MMSEEKR,
    "MethylLasso": METHYLASSO,
}


# Added during figure color centralization.
REPLICATE_COLORS = {
    "wgbs": [COLOR_PALETTE[1], COLOR_PALETTE[2], COLOR_PALETTE[3]],
    "hm450k": [COLOR_PALETTE[4], COLOR_PALETTE[5], COLOR_PALETTE[8]],
}

WGBS_REPLICATE_1 = REPLICATE_COLORS["wgbs"][0]
WGBS_REPLICATE_2 = REPLICATE_COLORS["wgbs"][1]
WGBS_REPLICATE_3 = REPLICATE_COLORS["wgbs"][2]
HM450K_REPLICATE_1 = REPLICATE_COLORS["hm450k"][0]
HM450K_REPLICATE_2 = REPLICATE_COLORS["hm450k"][1]
HM450K_REPLICATE_3 = REPLICATE_COLORS["hm450k"][2]


# Added during figure color centralization.
REGION_CATEGORY_COLORS = {
    "microarray_unique": SECONDARY,
    "microarray_missed": GRAY,
    "shared": PRIMARY,
}


FEATURE_SET_COLORS = {
    "PMD": PRIMARY,
    "Random PMD": SECONDARY,
    "CGI": COLOR_PALETTE[2],
    "Random long": COLOR_PALETTE[3],
    "Random short": COLOR_PALETTE[4],
    "Always cancer": COLOR_PALETTE[5],
}

RECURRENT_PMDS = FEATURE_SET_COLORS["PMD"]
RANDOM_PMDS = FEATURE_SET_COLORS["Random PMD"]
RANDOM_CGIS = FEATURE_SET_COLORS["CGI"]
RANDOM_LONG_REGIONS = FEATURE_SET_COLORS["Random long"]
RANDOM_SHORT_REGIONS = FEATURE_SET_COLORS["Random short"]

HEATMAP_HIGH = COLOR_PALETTE[3]
HEATMAP_MEDIUM = COLOR_PALETTE[4]
HEATMAP_LOW = "#FFF7BC"

# Added during figure color centralization.
HEATMAP_COLOR_SCALES = {
    "white_to_blue": [[0.0, WHITE], [1.0, PRIMARY]],
    "tcga_macro_f1": [
        "#FFF7BC",
        "#FEE391",
        "#FEC44F",
        "#FE9929",
        "#EC7014",
        "#CC4C02",
        "#990000",
    ],
    "tcga_mcc_diverging": [  # Added during figure color centralization.
        "#8E0152",
        "#C51B7D",
        "#DE77AE",
        "#F1B6DA",
        "#FDE0EF",
        "#F7F7F7",
        "#E6F5D0",
        "#B8E186",
        "#7FBC41",
        "#4D9221",
        "#276419",
    ],
    "confusion_white_to_blue": [  # Added during figure color centralization.
        "#FFFFFF",
        "#EAF3FB",
        "#D1E4F6",
        "#9FC5E8",
        "#6FA8DC",
        "#3D85C6",
        PRIMARY,
    ],
}


# Added during figure color centralization.
ANNOTATION_COLORS = {
    "synthetic_background_fill": COLOR_PALETTE[2],
    "synthetic_injected_fill": COLOR_PALETTE[3],
    "table_header_fill": "#D9D9D9",
    "table_row_fill": WHITE,
    "table_header_alt_fill": "#E9EDF2",  # Added during figure color centralization.
    "tcga_table_header_fill": "#DBEAFE",
    "tcga_table_alt_fill": "#F8FAFC",
    "tcga_table_base_fill": WHITE,
    "tcga_table_edge": "#D0D7DE",
    "feature_subset_default": "#BDBDBD",
    "venn_overlap_fill": "#4F6D7A",  # Added during figure color centralization.
}


NEUTRAL_COLORS = {
    "white": WHITE,
    "near_white": "#F8FAFC",
    "paper_bg": WHITE,
    "plot_bg": "#F2F2F2",
    "light_grid_bg": "#E5ECF6",
    "light_gray": "#D9D9D9",
    "soft_gray": "#DDDDDD",
    "medium_gray": "#9AA0A6",
    "dark_gray": "#5F5F5F",
    "axis_gray": "#4F4F4F",
    "muted_text": "#565656",  # Added during figure color centralization.
    "text_gray": "#666666",
    "deep_text": "#303030",  # Added during figure color centralization.
    "outline_gray": "#404040",
    "black": BLACK,
}

BARCHART_PRIMARY = PRIMARY
BARCHART_SECONDARY = SECONDARY
BARCHART_TERTIARY = GRAY
BOXPLOT_PRIMARY = PRIMARY
BOXPLOT_SECONDARY = SECONDARY
BOXPLOT_TERTIARY = GRAY


# Added during figure color centralization.
SYNTHETIC_TRACK_COLORS = {
    "source": PRIMARY,
    "potential": COLOR_PALETTE[2],
    "cleaned": SECONDARY,
    "injected": COLOR_PALETTE[4],
}

# Added during figure color centralization.
SYNTHETIC_INTERVAL_STYLES = {
    "background": {"fillcolor": COLOR_PALETTE[2], "opacity": 0.16},
    "injected": {"fillcolor": COLOR_PALETTE[3], "opacity": 0.18},
}


# Added during figure color centralization.
CATEGORICAL_SEQUENCES = {
    "tab10": [
        "#1F77B4",
        "#FF7F0E",
        "#2CA02C",
        "#D62728",
        "#9467BD",
        "#8C564B",
        "#E377C2",
        "#7F7F7F",
        "#BCBD22",
        "#17BECF",
    ],
    "tab20": [
        "#1F77B4",
        "#AEC7E8",
        "#FF7F0E",
        "#FFBB78",
        "#2CA02C",
        "#98DF8A",
        "#D62728",
        "#FF9896",
        "#9467BD",
        "#C5B0D5",
        "#8C564B",
        "#C49C94",
        "#E377C2",
        "#F7B6D2",
        "#7F7F7F",
        "#C7C7C7",
        "#BCBD22",
        "#DBDB8D",
        "#17BECF",
        "#9EDAE5",
    ],
    "set2": [
        "#66C2A5",
        "#FC8D62",
        "#8DA0CB",
        "#E78AC3",
        "#A6D854",
        "#FFD92F",
        "#E5C494",
        "#B3B3B3",
    ],
}


def get_categorical_color_map(
    values: Iterable[object],
    *,
    sequence_name: str = "tab20",
) -> dict[str, str]:
    labels = [str(value) for value in values if value is not None]
    ordered_labels: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label not in seen:
            ordered_labels.append(label)
            seen.add(label)

    if not ordered_labels:
        return {}

    sequence = CATEGORICAL_SEQUENCES[sequence_name]
    return {
        label: sequence[idx % len(sequence)] for idx, label in enumerate(ordered_labels)
    }


def ordered_palette(
    labels: Sequence[str],
    mapping: dict[str, str],
    *,
    fallback: str | None = None,
) -> list[str]:
    default_color = NEUTRAL_COLORS["medium_gray"] if fallback is None else fallback
    return [mapping.get(str(label), default_color) for label in labels]
