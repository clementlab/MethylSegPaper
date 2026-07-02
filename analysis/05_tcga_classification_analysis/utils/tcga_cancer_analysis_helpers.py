from __future__ import annotations

import colorsys
import sys
from typing import Iterable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import umap.umap_ as umap
from matplotlib.patches import Patch
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize

try:
    from PyComplexHeatmap import ClusterMapPlotter, HeatmapAnnotation, anno_simple

    HAS_PYCOMPLEXHEATMAP = True
except ImportError:  # pragma: no cover - optional dependency
    ClusterMapPlotter = None
    HeatmapAnnotation = None
    anno_simple = None
    HAS_PYCOMPLEXHEATMAP = False

try:
    import colorcet as cc

    HAS_COLORCET = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_COLORCET = False


RFC_AUC_METRIC_COLUMNS = ["analysis", "source", "metric", "value", "primary_metric"]
RFC_NULL_SUMMARY_COLUMNS = RFC_AUC_METRIC_COLUMNS + ["null_std", "n_null_repeats"]
RFC_COMPARISON_COLUMNS = [
    "analysis",
    "metric",
    "real_value",
    "randomized_value",
    "real_minus_randomized",
    "randomized_std",
    "n_randomized_repeats",
    "primary_metric",
]


def _normalize_label_name(label: object) -> str:
    label_str = str(label).strip().lower()
    replacements = {
        "solid tissue normal": "normal",
        "primary tumor": "tumor",
    }
    return replacements.get(label_str, label_str)


def _prepare_sample_metadata(all_samples: pd.DataFrame, sample_ids: Iterable[str]) -> pd.DataFrame:
    metadata = (
        all_samples[["sample_id", "project_id", "sample_type"]]
        .drop_duplicates(subset=["sample_id"])
        .set_index("sample_id")
        .loc[list(sample_ids)]
        .copy()
    )
    metadata["tumor_normal_status"] = np.where(
        metadata["sample_type"].astype(str) == "Solid Tissue Normal",
        "normal",
        "tumor",
    )
    return metadata


def _prepare_feature_matrix(
    methylation_matrix_df: pd.DataFrame,
    all_samples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    feature_df = methylation_matrix_df.copy()
    feature_df = feature_df.loc[
        feature_df.notna().any(axis=1),
        feature_df.notna().any(axis=0),
    ].copy()
    if feature_df.empty:
        raise ValueError("No non-empty methylation values are available for this analysis.")
    if feature_df.isna().any().any():
        raise ValueError(
            "Feature matrix still contains nulls. Apply train-split null filtering before downstream analyses."
        )

    sample_metadata = _prepare_sample_metadata(all_samples, feature_df.index)
    feature_df = feature_df.loc[sample_metadata.index].copy()
    X = StandardScaler().fit_transform(feature_df.to_numpy(dtype=float))
    return feature_df, sample_metadata, X


def _adjust_lightness(color: object, factor: float) -> tuple[float, float, float]:
    rgb = np.array(mcolors.to_rgb(color), dtype=float)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = np.clip(l * factor, 0.0, 1.0)
    return colorsys.hls_to_rgb(h, l, s)


def _distinct_palette(n: int) -> list[tuple[float, float, float]]:
    if n <= 0:
        return []
    if HAS_COLORCET:
        palette = [mcolors.to_rgb(c) for c in cc.glasbey[:n]]
        if len(palette) == n:
            return palette

    golden_ratio = 0.618033988749895
    h = 0.15
    colors = []
    for i in range(n):
        h = (h + golden_ratio) % 1.0
        s = 0.72 - 0.08 * (i % 2)
        v = 0.92 - 0.06 * ((i // 2) % 2)
        colors.append(colorsys.hsv_to_rgb(h, s, v))
    return colors


def _build_project_color_map(project_ids: Iterable[str]) -> dict[str, tuple[float, float, float]]:
    project_ids = [str(x) for x in sorted(pd.unique(list(project_ids)))]
    return dict(zip(project_ids, _distinct_palette(len(project_ids))))


def _build_project_status_color_map(
    project_ids: Iterable[str],
) -> dict[tuple[str, str], tuple[float, float, float]]:
    base_map = _build_project_color_map(project_ids)
    return {
        (project_id, "tumor"): _adjust_lightness(base_color, 0.90)
        for project_id, base_color in base_map.items()
    } | {
        (project_id, "normal"): _adjust_lightness(base_color, 1.35)
        for project_id, base_color in base_map.items()
    }


def _build_categorical_color_map(values: Iterable[object], palette_name: str = "tab20") -> dict[str, tuple[float, float, float]]:
    value_series = pd.Series(list(values)).astype(str)
    unique_values = sorted(value_series.unique())
    palette = sns.color_palette(palette_name, n_colors=max(len(unique_values), 1))
    return {value: palette[idx] for idx, value in enumerate(unique_values)}


def _plotly_color(value: Iterable[float]) -> str:
    rgb = np.clip(np.round(np.array(value, dtype=float) * 255), 0, 255).astype(int)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def _marker_symbol_for_sample_type(sample_type: object) -> str:
    sample_type = str(sample_type)
    if sample_type == "Primary Tumor":
        return "circle"
    if sample_type == "Solid Tissue Normal":
        return "diamond"
    return "square"


def _build_color_map(sample_metadata: pd.DataFrame, color_col: str) -> dict[object, str]:
    if color_col == "tumor_normal_status":
        return {"tumor": "#B33A3A", "normal": "#2D6A9F"}
    if color_col == "project_id":
        return {
            key: _plotly_color(value)
            for key, value in _build_project_status_color_map(sample_metadata["project_id"]).items()
        }
    return {
        key: _plotly_color(value)
        for key, value in _build_categorical_color_map(sample_metadata[color_col]).items()
    }


def _draw_embedding_plot(
    results_df: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    color_col: str,
    marker_col: str,
    annotate_centroids: bool = False,
) -> None:
    color_map = _build_color_map(sample_metadata, color_col)
    fig = go.Figure()
    legend_seen = set()
    for (color_value, marker_value), group in results_df.groupby(
        [color_col, marker_col], sort=True, dropna=False
    ):
        color_key = (
            (str(color_value), _normalize_label_name(group["tumor_normal_status"].iloc[0]))
            if color_col == "project_id"
            else str(color_value)
        )
        marker_color = color_map.get(color_key, color_map.get(str(color_value), "#4C78A8"))
        legend_name = (
            str(color_value)
            if color_col == "tumor_normal_status"
            else f"{color_value} | {marker_value}"
        )
        showlegend = legend_name not in legend_seen
        legend_seen.add(legend_name)
        fig.add_trace(
            go.Scattergl(
                x=group[x_col],
                y=group[y_col],
                mode="markers",
                name=legend_name,
                showlegend=showlegend,
                marker={
                    "size": 7,
                    "opacity": 0.82,
                    "color": marker_color,
                    "symbol": _marker_symbol_for_sample_type(marker_value),
                    "line": {"width": 0.4, "color": "rgba(0,0,0,0.25)"},
                },
                customdata=np.column_stack(
                    [
                        group["sample_id"].astype(str),
                        group["project_id"].astype(str),
                        group["tumor_normal_status"].astype(str),
                        group[marker_col].astype(str),
                    ]
                ),
                hovertemplate=(
                    "sample: %{customdata[0]}<br>"
                    "cancer: %{customdata[1]}<br>"
                    "status: %{customdata[2]}<br>"
                    f"{marker_col}: %{{customdata[3]}}<br>"
                    f"{x_col}: %{{x:.3f}}<br>"
                    f"{y_col}: %{{y:.3f}}<extra></extra>"
                ),
            )
        )

    if annotate_centroids and not results_df.empty:
        centroid_df = results_df.groupby(color_col, dropna=False)[[x_col, y_col]].mean().reset_index()
        for row in centroid_df.itertuples(index=False):
            fig.add_annotation(
                x=getattr(row, x_col),
                y=getattr(row, y_col),
                text=str(getattr(row, color_col)),
                showarrow=False,
                font={"size": 11, "color": "#222"},
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="rgba(0,0,0,0.15)",
                borderwidth=1,
            )

    fig.update_layout(
        title=title,
        width=980,
        height=720,
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "top", "y": -0.14, "xanchor": "left", "x": 0.0},
        margin={"l": 60, "r": 30, "t": 80, "b": 110},
        xaxis_title=x_col,
        yaxis_title=y_col,
    )
    fig.show()


def run_pmr_umap(
    methylation_matrix_df: pd.DataFrame,
    all_samples: pd.DataFrame,
    title: str,
    *,
    color_col: str = "project_id",
    marker_col: str = "sample_type",
    project_ids_to_show: list[str] | None = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
    annotate_centroids: bool = False,
) -> pd.DataFrame:
    feature_df, sample_metadata, X = _prepare_feature_matrix(methylation_matrix_df, all_samples)

    if project_ids_to_show is not None:
        keep_mask = sample_metadata["project_id"].astype(str).isin({str(x) for x in project_ids_to_show})
        sample_metadata = sample_metadata.loc[keep_mask].copy()
        feature_df = feature_df.loc[sample_metadata.index].copy()
        X = StandardScaler().fit_transform(feature_df.to_numpy(dtype=float))

    effective_neighbors = max(2, min(int(n_neighbors), len(feature_df) - 1))
    embedding = umap.UMAP(
        n_neighbors=effective_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    ).fit_transform(X)

    results_df = (
        pd.DataFrame({"sample_id": feature_df.index, "UMAP1": embedding[:, 0], "UMAP2": embedding[:, 1]})
        .merge(sample_metadata.reset_index(), on="sample_id", how="left")
    )
    _draw_embedding_plot(
        results_df,
        sample_metadata,
        x_col="UMAP1",
        y_col="UMAP2",
        title=title,
        color_col=color_col,
        marker_col=marker_col,
        annotate_centroids=annotate_centroids,
    )
    return results_df


def run_sample_pca(
    methylation_matrix_df: pd.DataFrame,
    all_samples: pd.DataFrame,
    title: str,
    *,
    color_col: str = "project_id",
    marker_col: str = "sample_type",
    random_state: int = 42,
) -> pd.DataFrame:
    feature_df, sample_metadata, X = _prepare_feature_matrix(methylation_matrix_df, all_samples)
    pca_model = PCA(n_components=min(2, X.shape[0], X.shape[1]), random_state=random_state)
    embedding = pca_model.fit_transform(X)
    explained = pca_model.explained_variance_ratio_
    results_df = (
        pd.DataFrame(
            {
                "sample_id": feature_df.index,
                "PC1": embedding[:, 0],
                "PC2": embedding[:, 1] if embedding.shape[1] > 1 else np.zeros(len(feature_df)),
                "pc1_explained_variance_ratio": float(explained[0]) if len(explained) else 0.0,
                "pc2_explained_variance_ratio": float(explained[1]) if len(explained) > 1 else 0.0,
            }
        ).merge(sample_metadata.reset_index(), on="sample_id", how="left")
    )
    _draw_embedding_plot(
        results_df,
        sample_metadata,
        x_col="PC1",
        y_col="PC2",
        title=title,
        color_col=color_col,
        marker_col=marker_col,
    )
    return results_df


def _draw_row_color_legends(
    ax: plt.Axes,
    project_color_map: dict[str, tuple[float, float, float]],
    sample_type_color_map: dict[str, tuple[float, float, float]],
) -> None:
    project_handles = [Patch(facecolor=color, edgecolor="none", label=label) for label, color in project_color_map.items()]
    sample_type_handles = [Patch(facecolor=color, edgecolor="none", label=label) for label, color in sample_type_color_map.items()]
    project_legend = ax.legend(
        handles=project_handles,
        title="Cancer Type",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(project_legend)
    ax.legend(
        handles=sample_type_handles,
        title="Sample Type",
        bbox_to_anchor=(1.02, 0.55),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
        title_fontsize=9,
    )


def run_sample_hierarchical_clustermap(
    methylation_matrix_df: pd.DataFrame,
    all_samples: pd.DataFrame,
    title: str,
    *,
    max_regions: int = 300,
    metric: str = "euclidean",
    method: str = "average",
    z_score_regions: bool = True,
    prefer_pycomplexheatmap: bool = True,
) -> dict[str, object]:
    cluster_input_df = methylation_matrix_df.copy()
    cluster_input_df = cluster_input_df.loc[
        cluster_input_df.notna().any(axis=1),
        cluster_input_df.notna().any(axis=0),
    ].copy()
    if cluster_input_df.empty:
        raise ValueError("No non-empty methylation values are available for hierarchical clustering.")
    if cluster_input_df.isna().any().any():
        raise ValueError("Cluster input still contains nulls after filtering.")

    sample_metadata = _prepare_sample_metadata(all_samples, cluster_input_df.index)
    cluster_input_df = cluster_input_df.loc[sample_metadata.index].copy()
    if cluster_input_df.shape[1] > max_regions:
        keep_cols = (
            cluster_input_df.var(axis=0, skipna=True)
            .sort_values(ascending=False)
            .head(max_regions)
            .index.tolist()
        )
        cluster_input_df = cluster_input_df.loc[:, keep_cols].copy()

    display_df = cluster_input_df.copy()
    if z_score_regions:
        display_df = display_df.subtract(display_df.mean(axis=0), axis=1)
        display_df = display_df.divide(display_df.std(axis=0).replace(0, np.nan), axis=1).fillna(0.0)

    project_color_map = _build_categorical_color_map(sample_metadata["project_id"], palette_name="tab20")
    sample_type_color_map = _build_categorical_color_map(sample_metadata["sample_type"], palette_name="Set2")
    row_colors = pd.DataFrame(
        {
            "Cancer Type": sample_metadata["project_id"].astype(str).map(project_color_map),
            "Sample Type": sample_metadata["sample_type"].astype(str).map(sample_type_color_map),
        },
        index=display_df.index,
    )

    if prefer_pycomplexheatmap and HAS_PYCOMPLEXHEATMAP:
        try:
            row_annotation = HeatmapAnnotation(
                CancerType=anno_simple(sample_metadata["project_id"].astype(str), colors=project_color_map, add_text=False),
                SampleType=anno_simple(sample_metadata["sample_type"].astype(str), colors=sample_type_color_map, add_text=False),
                axis=0,
            )
            plt.figure(figsize=(16, 10))
            ClusterMapPlotter(
                data=display_df,
                left_annotation=row_annotation,
                cmap="vlag",
                row_cluster=True,
                col_cluster=True,
                show_rownames=True,
                show_colnames=False,
            )
            plt.suptitle(title, y=1.02)
            plt.show()
            return {"cluster_input_df": cluster_input_df, "display_df": display_df, "backend": "PyComplexHeatmap"}
        except Exception as exc:  # pragma: no cover - plotting fallback
            print(f"PyComplexHeatmap plotting failed ({exc}); falling back to seaborn clustermap.")

    original_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(max(original_limit, int(10 * max(display_df.shape[0], display_df.shape[1], 1))))
        grid = sns.clustermap(
            display_df,
            cmap="vlag",
            row_cluster=True,
            col_cluster=True,
            method=method,
            metric=metric,
            row_colors=row_colors,
            xticklabels=False,
            yticklabels=True,
            figsize=(16, 10),
            dendrogram_ratio=(0.12, 0.12),
            cbar_pos=(0.02, 0.8, 0.02, 0.15),
        )
    finally:
        sys.setrecursionlimit(original_limit)

    grid.fig.suptitle(title, y=1.02)
    _draw_row_color_legends(grid.ax_heatmap, project_color_map, sample_type_color_map)
    plt.show()
    return {
        "cluster_input_df": cluster_input_df,
        "display_df": display_df,
        "row_colors": row_colors,
        "grid": grid,
        "backend": "seaborn",
    }


def run_random_forest_classifier(
    methylation_matrix_df: pd.DataFrame,
    all_samples: pd.DataFrame,
    label_col: str,
    title: str,
    *,
    train_sample_ids: list[str],
    test_sample_ids: list[str],
    positive_label: str | None = None,
    random_state: int = 42,
) -> dict[str, object]:
    feature_df = methylation_matrix_df.copy()
    feature_df = feature_df.loc[feature_df.notna().any(axis=1), feature_df.notna().any(axis=0)].copy()
    if feature_df.empty:
        raise ValueError("No non-empty methylation values are available for this analysis.")
    if feature_df.isna().any().any():
        raise ValueError("Classifier feature matrix still contains nulls after filtering.")

    sample_metadata = _prepare_sample_metadata(all_samples, feature_df.index)
    modeling_df = sample_metadata.reset_index().copy()
    if positive_label is None:
        modeling_df["label"] = modeling_df[label_col].astype(str)
    else:
        modeling_df["label"] = np.where(
            modeling_df[label_col].astype(str) == str(positive_label),
            "normal",
            "tumor",
        )

    train_df = modeling_df.loc[modeling_df["sample_id"].isin(set(train_sample_ids))].copy()
    test_df = modeling_df.loc[modeling_df["sample_id"].isin(set(test_sample_ids))].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(f"{title} requires non-empty train and test sample sets.")
    if train_df["label"].nunique() < 2:
        raise ValueError(f"{title} requires at least 2 classes in the training split.")

    train_feature_df = feature_df.loc[train_df["sample_id"]].copy()
    test_feature_df = feature_df.loc[test_df["sample_id"]].copy()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feature_df.to_numpy(dtype=float))
    X_test = scaler.transform(test_feature_df.to_numpy(dtype=float))
    y_train = train_df["label"].astype(str).to_numpy()
    y_test = test_df["label"].astype(str).to_numpy()

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    model_labels = [str(label) for label in model.classes_]

    confusion_df = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=model_labels),
        index=pd.Index(model_labels, name="true_label"),
        columns=pd.Index(model_labels, name="predicted_label"),
    )
    report_df = (
        pd.DataFrame(classification_report(y_test, y_pred, labels=model_labels, output_dict=True, zero_division=0))
        .T.reset_index()
        .rename(columns={"index": "label"})
    )
    class_counts_df = pd.concat(
        [
            train_df["label"].value_counts().sort_index().rename("count").rename_axis("label").reset_index().assign(split="train"),
            test_df["label"].value_counts().sort_index().rename("count").rename_axis("label").reset_index().assign(split="test"),
        ],
        ignore_index=True,
    )[["split", "label", "count"]]

    summary_records = [
        {"metric": "n_train_samples", "value": float(len(train_df))},
        {"metric": "n_test_samples", "value": float(len(test_df))},
        {"metric": "n_regions", "value": float(feature_df.shape[1])},
        {"metric": "n_classes", "value": float(len(model_labels))},
        {"metric": "accuracy", "value": float(accuracy_score(y_test, y_pred))},
        {"metric": "macro_f1", "value": float(f1_score(y_test, y_pred, average="macro", zero_division=0))},
        {"metric": "micro_f1", "value": float(f1_score(y_test, y_pred, average="micro", zero_division=0))},
        {"metric": "macro_precision", "value": float(precision_score(y_test, y_pred, average="macro", zero_division=0))},
        {"metric": "micro_precision", "value": float(precision_score(y_test, y_pred, average="micro", zero_division=0))},
    ]

    if len(model_labels) == 2:
        normal_label = next((label for label in model_labels if "normal" in _normalize_label_name(label)), model_labels[-1])
        normal_col_idx = model_labels.index(normal_label)
        y_binary = (pd.Series(y_test).astype(str) == normal_label).astype(int)
        scores = y_proba[:, normal_col_idx]
        if y_binary.nunique() > 1:
            try:
                summary_records.append({"metric": "roc_auc", "value": float(roc_auc_score(y_binary, scores))})
            except ValueError:
                pass
            try:
                summary_records.append({"metric": "average_precision", "value": float(average_precision_score(y_binary, scores))})
            except ValueError:
                pass
    else:
        y_binary = label_binarize(y_test, classes=model_labels)
        try:
            summary_records.append(
                {"metric": "roc_auc_ovr_macro", "value": float(roc_auc_score(y_binary, y_proba, multi_class="ovr", average="macro"))}
            )
        except ValueError:
            pass
        try:
            summary_records.append(
                {"metric": "average_precision_macro", "value": float(average_precision_score(y_binary, y_proba, average="macro"))}
            )
        except ValueError:
            pass

    summary_df = pd.DataFrame(summary_records)
    test_predictions_df = test_df[["sample_id", "label"]].rename(columns={"label": "true_label"}).copy()
    test_predictions_df["predicted_label"] = y_pred
    for idx, label in enumerate(model_labels):
        test_predictions_df[label] = y_proba[:, idx]

    plt.figure(figsize=(6, 6))
    ConfusionMatrixDisplay(confusion_matrix=confusion_df.to_numpy(), display_labels=model_labels).plot(
        ax=plt.gca(), xticks_rotation=90, colorbar=False
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()

    if len(model_labels) == 2 and y_binary.nunique() > 1:
        plt.figure(figsize=(6, 6))
        RocCurveDisplay.from_predictions(y_binary, scores, ax=plt.gca())
        plt.title(f"ROC Curve: {title}")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 6))
        PrecisionRecallDisplay.from_predictions(y_binary, scores, ax=plt.gca())
        plt.title(f"Precision-Recall Curve: {title}")
        plt.tight_layout()
        plt.show()

    return {
        "title": title,
        "class_counts": class_counts_df,
        "summary": summary_df,
        "confusion_matrix": confusion_df,
        "classification_report": report_df,
        "test_predictions": test_predictions_df,
        "labels": model_labels,
        "model": model,
    }


def summarize_rfc_metrics(
    rfc_results: dict[str, object],
    analysis_name: str,
    source_name: str,
    primary_metric: str,
) -> pd.DataFrame:
    summary_df = rfc_results["summary"].copy()
    metric_rows = summary_df.loc[
        summary_df["metric"].isin(
            ["accuracy", "macro_f1", "micro_f1", "macro_precision", "micro_precision"]
        )
    ].copy()
    metric_rows["analysis"] = analysis_name
    metric_rows["source"] = source_name
    metric_rows["primary_metric"] = primary_metric
    return metric_rows[["analysis", "source", "metric", "value", "primary_metric"]].reset_index(drop=True)


def _match_score_column(score_cols: list[str], label: object) -> str | None:
    normalized_label = _normalize_label_name(label)
    for col in score_cols:
        normalized_col = _normalize_label_name(col)
        if normalized_col == normalized_label or normalized_label in normalized_col or normalized_col in normalized_label:
            return col
    return None


def summarize_rfc_auc_metrics(
    rfc_results: dict[str, object],
    analysis_name: str,
    source_name: str,
    primary_metric: str,
    positive_label: str | None = None,
) -> pd.DataFrame:
    summary_df = rfc_results["summary"].copy()
    auc_rows = summary_df.loc[
        summary_df["metric"].astype(str).str.contains("auc|average_precision", case=False, regex=True)
    ].copy()
    if not auc_rows.empty:
        auc_rows["analysis"] = analysis_name
        auc_rows["source"] = source_name
        auc_rows["primary_metric"] = primary_metric
        return auc_rows[RFC_AUC_METRIC_COLUMNS].reset_index(drop=True)

    test_predictions = rfc_results.get("test_predictions")
    if not isinstance(test_predictions, pd.DataFrame) or "true_label" not in test_predictions.columns:
        return pd.DataFrame(columns=RFC_AUC_METRIC_COLUMNS)

    metadata_cols = {"sample_id", "true_label", "predicted_label"}
    score_cols = [col for col in test_predictions.columns if col not in metadata_cols and pd.api.types.is_numeric_dtype(test_predictions[col])]
    y_true = test_predictions["true_label"].astype(str)
    unique_labels = list(pd.unique(y_true))
    records: list[dict[str, object]] = []

    if positive_label is not None and len(unique_labels) == 2:
        selected_label = next(
            (label for label in unique_labels if _normalize_label_name(label) == _normalize_label_name(positive_label)),
            None,
        )
        if selected_label is None:
            selected_label = next(
                (label for label in unique_labels if "normal" in _normalize_label_name(label)),
                unique_labels[-1],
            )
        score_col = _match_score_column(score_cols, selected_label) or (score_cols[0] if len(score_cols) == 1 else None)
        if score_col is not None:
            y_binary = (y_true == str(selected_label)).astype(int)
            if y_binary.nunique() > 1:
                scores = test_predictions[score_col].astype(float)
                records.extend(
                    [
                        {"analysis": analysis_name, "source": source_name, "metric": "roc_auc", "value": roc_auc_score(y_binary, scores), "primary_metric": primary_metric},
                        {"analysis": analysis_name, "source": source_name, "metric": "average_precision", "value": average_precision_score(y_binary, scores), "primary_metric": primary_metric},
                    ]
                )

    if not records:
        class_score_cols = [col for col in score_cols if str(col) in unique_labels]
        if len(class_score_cols) == len(unique_labels) and len(unique_labels) > 2:
            scores = test_predictions[class_score_cols].astype(float).to_numpy()
            y_binary = label_binarize(y_true, classes=[str(col) for col in class_score_cols])
            records.extend(
                [
                    {"analysis": analysis_name, "source": source_name, "metric": "roc_auc_ovr_macro", "value": roc_auc_score(y_binary, scores, multi_class="ovr", average="macro"), "primary_metric": primary_metric},
                    {"analysis": analysis_name, "source": source_name, "metric": "average_precision_macro", "value": average_precision_score(y_binary, scores, average="macro"), "primary_metric": primary_metric},
                ]
            )

    return pd.DataFrame(records, columns=RFC_AUC_METRIC_COLUMNS)


def average_null_metric_tables(metric_tables: list[pd.DataFrame]) -> pd.DataFrame:
    if not metric_tables:
        return pd.DataFrame(columns=RFC_NULL_SUMMARY_COLUMNS)
    combined_df = pd.concat(metric_tables, ignore_index=True)
    averaged_df = (
        combined_df.groupby(["analysis", "source", "metric", "primary_metric"], as_index=False)
        .agg(value=("value", "mean"), null_std=("value", "std"), n_null_repeats=("value", "size"))
    )
    averaged_df["null_std"] = averaged_df["null_std"].fillna(0.0)
    return averaged_df


def build_rfc_comparison_df(real_metrics_df: pd.DataFrame, randomized_metrics_df: pd.DataFrame) -> pd.DataFrame:
    real_df = real_metrics_df.copy().rename(columns={"value": "real_value"})
    randomized_df = randomized_metrics_df.copy().rename(
        columns={
            "value": "randomized_value",
            "null_std": "randomized_std",
            "n_null_repeats": "n_randomized_repeats",
        }
    )
    comparison_df = real_df.merge(
        randomized_df[["analysis", "metric", "randomized_value", "randomized_std", "n_randomized_repeats"]],
        on=["analysis", "metric"],
        how="left",
    )
    comparison_df["real_minus_randomized"] = comparison_df["real_value"] - comparison_df["randomized_value"]
    return comparison_df[RFC_COMPARISON_COLUMNS].sort_values(["analysis", "metric"]).reset_index(drop=True)


def _build_chr_bounds_from_meth_data(meth_data_df: pd.DataFrame) -> dict[str, tuple[int, int]]:
    chr_bounds_df = (
        meth_data_df.groupby("CpG_chrm", sort=False)
        .agg(chr_start_min=("CpG_beg", "min"), chr_end_max=("CpG_end", "max"))
        .reset_index()
    )
    return {
        str(row.CpG_chrm): (int(row.chr_start_min), int(row.chr_end_max))
        for row in chr_bounds_df.itertuples(index=False)
    }


def _build_interval_lookup(region_df: pd.DataFrame) -> dict[str, list[tuple[int, int]]]:
    interval_lookup: dict[str, list[tuple[int, int]]] = {}
    for row in region_df[["chr", "start", "end"]].itertuples(index=False):
        interval_lookup.setdefault(str(row.chr), []).append((int(row.start), int(row.end)))
    for chr_name in interval_lookup:
        interval_lookup[chr_name] = sorted(interval_lookup[chr_name])
    return interval_lookup


def _interval_overlaps_any(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    return any(start < other_end and end > other_start for other_start, other_end in intervals)


def _interval_overlap_bp(start: int, end: int, intervals: list[tuple[int, int]]) -> int:
    return sum(max(0, min(end, other_end) - max(start, other_start)) for other_start, other_end in intervals)


def randomize_regions_globally(
    region_df: pd.DataFrame,
    meth_data_df: pd.DataFrame,
    random_state: int = 42,
    max_tries: int = 5000,
) -> pd.DataFrame:
    if region_df.empty:
        return region_df.copy()

    rng = np.random.default_rng(random_state)
    chr_bounds = _build_chr_bounds_from_meth_data(meth_data_df)
    original_lookup = _build_interval_lookup(region_df)
    randomized_df = region_df.copy().reset_index(drop=True)
    placement_modes = []
    overlap_bps = []

    for idx, row in enumerate(region_df.reset_index(drop=True).itertuples(index=False)):
        chr_name = str(row.chr)
        length = max(1, int(row.end) - int(row.start))
        chr_start_min, chr_end_max = chr_bounds[chr_name]
        max_start = chr_end_max - length
        exact_start = None

        for _ in range(max_tries):
            candidate_start = int(rng.integers(chr_start_min, max_start + 1))
            candidate_end = candidate_start + length
            if not _interval_overlaps_any(candidate_start, candidate_end, original_lookup.get(chr_name, [])):
                exact_start = candidate_start
                placement_modes.append("non_overlapping")
                overlap_bps.append(0)
                break

        if exact_start is None:
            candidate_starts = np.linspace(chr_start_min, max_start, num=min(512, max_start - chr_start_min + 1), dtype=int)
            best_start = None
            best_overlap = None
            for candidate_start in np.unique(candidate_starts):
                candidate_end = candidate_start + length
                overlap_bp = _interval_overlap_bp(candidate_start, candidate_end, original_lookup.get(chr_name, []))
                if best_overlap is None or overlap_bp < best_overlap:
                    best_overlap = overlap_bp
                    best_start = int(candidate_start)
                    if overlap_bp == 0:
                        break
            exact_start = best_start
            placement_modes.append("best_effort_overlap")
            overlap_bps.append(int(best_overlap if best_overlap is not None else 0))

        randomized_df.at[idx, "chr"] = chr_name
        randomized_df.at[idx, "start"] = int(exact_start)
        randomized_df.at[idx, "end"] = int(exact_start + length)

    randomized_df["placement_mode"] = placement_modes
    randomized_df["overlap_bp_with_original"] = overlap_bps
    return randomized_df


def summarize_randomized_region_table(
    original_df: pd.DataFrame,
    randomized_df: pd.DataFrame,
    meth_data_df: pd.DataFrame,
    table_name: str,
) -> pd.DataFrame:
    chr_bounds = _build_chr_bounds_from_meth_data(meth_data_df)
    original_lookup = _build_interval_lookup(original_df)
    original_chr = original_df["chr"].astype(str).reset_index(drop=True)
    randomized_chr = randomized_df["chr"].astype(str).reset_index(drop=True)
    original_lengths = original_df["end"].to_numpy() - original_df["start"].to_numpy()
    randomized_lengths = randomized_df["end"].to_numpy() - randomized_df["start"].to_numpy()

    in_bounds_count = 0
    overlap_count = 0
    total_overlap_bp = 0
    for row in randomized_df.itertuples(index=False):
        chr_name = str(row.chr)
        start = int(row.start)
        end = int(row.end)
        chr_start_min, chr_end_max = chr_bounds[chr_name]
        if start >= chr_start_min and end <= chr_end_max:
            in_bounds_count += 1
        overlap_bp = _interval_overlap_bp(start, end, original_lookup.get(chr_name, []))
        if overlap_bp > 0:
            overlap_count += 1
        total_overlap_bp += overlap_bp

    placement_series = randomized_df.get("placement_mode", pd.Series(["unknown"] * len(randomized_df)))
    return pd.DataFrame(
        {
            "table_name": [table_name],
            "n_original_regions": [len(original_df)],
            "n_randomized_regions": [len(randomized_df)],
            "chromosomes_preserved": [bool(original_chr.equals(randomized_chr))],
            "lengths_preserved": [bool(np.array_equal(original_lengths, randomized_lengths))],
            "randomized_regions_in_bounds": [in_bounds_count],
            "randomized_regions_overlapping_original": [overlap_count],
            "best_effort_overlap_regions": [int((placement_series == "best_effort_overlap").sum())],
            "total_overlap_bp_with_original": [int(total_overlap_bp)],
        }
    )


def summarize_randomized_chromosome_counts(
    original_df: pd.DataFrame,
    randomized_df: pd.DataFrame,
    table_name: str,
) -> pd.DataFrame:
    original_counts = original_df["chr"].astype(str).value_counts().sort_index().rename("n_original_regions")
    randomized_counts = randomized_df["chr"].astype(str).value_counts().sort_index().rename("n_randomized_regions")
    summary_df = (
        pd.concat([original_counts, randomized_counts], axis=1)
        .fillna(0)
        .reset_index()
        .rename(columns={"index": "chr"})
    )
    summary_df["n_original_regions"] = summary_df["n_original_regions"].astype(int)
    summary_df["n_randomized_regions"] = summary_df["n_randomized_regions"].astype(int)
    summary_df["delta_regions"] = summary_df["n_randomized_regions"] - summary_df["n_original_regions"]
    summary_df["table_name"] = table_name
    return summary_df[["table_name", "chr", "n_original_regions", "n_randomized_regions", "delta_regions"]]


def plot_rfc_comparison_heatmaps(
    randomized_rfc_comparison_df: pd.DataFrame,
    randomized_auc_comparison_df: pd.DataFrame,
) -> None:
    summary_metric_order = ["accuracy", "macro_f1", "micro_f1", "macro_precision", "micro_precision"]
    auc_metric_order = ["roc_auc", "average_precision", "roc_auc_ovr_macro", "average_precision_macro"]

    summary_plot_df = randomized_rfc_comparison_df.loc[
        randomized_rfc_comparison_df["metric"].isin(summary_metric_order)
    ].copy()
    summary_plot_df["metric"] = pd.Categorical(
        summary_plot_df["metric"], categories=summary_metric_order, ordered=True
    )
    summary_plot_df = summary_plot_df.sort_values(["analysis", "metric"])

    auc_plot_df = randomized_auc_comparison_df.copy()
    if not auc_plot_df.empty:
        categories = [metric for metric in auc_metric_order if metric in auc_plot_df["metric"].tolist()] or list(pd.unique(auc_plot_df["metric"]))
        auc_plot_df["metric"] = pd.Categorical(auc_plot_df["metric"], categories=categories, ordered=True)
        auc_plot_df = auc_plot_df.sort_values(["analysis", "metric"])

    fig, axes = plt.subplots(1, 2, figsize=(18, 5), gridspec_kw={"width_ratios": [3, 2]})
    sns.heatmap(
        summary_plot_df.pivot(index="analysis", columns="metric", values="real_minus_randomized"),
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        ax=axes[0],
    )
    axes[0].set_title("PMD Minus Randomized RFC Metrics")
    axes[0].set_xlabel("Metric")
    axes[0].set_ylabel("Analysis")

    if not auc_plot_df.empty:
        sns.heatmap(
            auc_plot_df.pivot(index="analysis", columns="metric", values="real_minus_randomized"),
            annot=True,
            fmt=".3f",
            cmap="RdBu_r",
            center=0,
            linewidths=0.5,
            ax=axes[1],
        )
        axes[1].set_title("PMD Minus Randomized RFC AUC")
        axes[1].set_xlabel("AUC Metric")
        axes[1].set_ylabel("")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "AUC metrics unavailable", ha="center", va="center", fontsize=11)

    plt.tight_layout()
    plt.show()


def make_umap_summary_figure(
    cancer_type_distinct_umap_results: pd.DataFrame,
    shared_cancer_umap_results: pd.DataFrame,
) -> None:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Cancer-Type-Distinct PMDs", "Shared-Cancer PMDs"),
        shared_xaxes=False,
        shared_yaxes=False,
    )
    for col_idx, results_df in enumerate(
        [cancer_type_distinct_umap_results, shared_cancer_umap_results], start=1
    ):
        if results_df.empty:
            continue
        color_col = "project_id" if col_idx == 1 else "tumor_normal_status"
        color_series = results_df[color_col].astype(str)
        if color_col == "project_id":
            color_lookup = {
                key: _plotly_color(value)
                for key, value in _build_project_color_map(color_series).items()
            }
        else:
            color_lookup = {"tumor": "#B33A3A", "normal": "#2D6A9F"}
        for label, group in results_df.groupby(color_col, sort=True):
            fig.add_trace(
                go.Scattergl(
                    x=group["UMAP1"],
                    y=group["UMAP2"],
                    mode="markers",
                    name=str(label),
                    marker={"size": 6, "opacity": 0.8, "color": color_lookup.get(str(label), "#4C78A8")},
                    legendgroup=f"{col_idx}-{label}",
                    showlegend=(col_idx == 1) or (color_col == "tumor_normal_status"),
                    hovertemplate="sample: %{customdata}<extra></extra>",
                    customdata=group["sample_id"].astype(str),
                ),
                row=1,
                col=col_idx,
            )
    fig.update_layout(height=650, width=1200, template="plotly_white")
    fig.show()
