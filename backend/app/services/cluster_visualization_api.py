"""
Plotly visualization helpers for content-style and business clustering endpoints.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import io
import base64
from typing import Any, Dict, Iterable, Tuple

# Avoid rare BLAS/OpenMP stalls in small API containers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


BINARY_CONTENT_COLS = [
    "CTA_present",
    "promo_post",
    "mentions_location",
    "religious_theme",
    "patriotic_theme",
    "arabic_dialect_style",
]

CONTENT_REQUIRED_COLS = [
    "likes_count",
    "comments_count",
    "views_count",
    *BINARY_CONTENT_COLS,
    "sector",
    "post_type",
]

CONTENT_HOVER_CANDIDATES = [
    "business_name",
    "post_id",
    "caption_text",
    "post_type",
    "sector",
    "likes_count",
    "comments_count",
    "views_count",
]

DEFAULT_BUSINESS_FEATURE_COLS = [
    "log_followers_count",
    "avg_engagement_rate",
    "avg_view_rate",
    "avg_comment_rate",
    "posting_frequency",
    "avg_caption_length",
    "avg_hashtags_count",
    "avg_emoji_count",
    "percentage_reels",
    "percentage_images",
    "percentage_carousels",
    "percentage_promo_posts",
    "percentage_CTA_posts",
    "percentage_location_posts",
    "percentage_religious_theme",
    "percentage_patriotic_theme",
    "percentage_arabic_dialect_style",
]

_SOURCE_MARKER = {
    "original_kpi_dataset": "o",
    "uploaded_dataset": "^",
}


def _color_for_cluster(cluster_key: str) -> str:
    palette = [
        "#2563eb",
        "#16a34a",
        "#dc2626",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#65a30d",
        "#be123c",
        "#4f46e5",
        "#0f766e",
    ]
    if cluster_key == "-1":
        return "#6b7280"
    idx = abs(hash(cluster_key)) % len(palette)
    return palette[idx]


def _figure_to_base64(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _build_matplotlib_scatter_base64(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    cluster_col: str,
    source_col: str,
    title: str,
    color_fn=None,
    source_color_fn=None,
) -> str:
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.2)
    if color_fn is None:
        color_fn = _color_for_cluster

    for source_name in sorted(df[source_col].astype(str).unique()):
        source_df = df[df[source_col].astype(str) == source_name]
        marker = _SOURCE_MARKER.get(source_name, "o")
        for cluster_key in sorted(source_df[cluster_col].astype(str).unique()):
            part = source_df[source_df[cluster_col].astype(str) == cluster_key]
            if source_color_fn is not None:
                point_color = source_color_fn(source_name, cluster_key)
            else:
                point_color = color_fn(cluster_key)
            ax.scatter(
                part[x_col].to_numpy(dtype=float),
                part[y_col].to_numpy(dtype=float),
                s=26,
                alpha=0.72,
                marker=marker,
                c=point_color,
                label=f"cluster {cluster_key} | {source_name}",
            )

    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > 16:
        # Keep the figure readable for large cluster/source combinations.
        handles = handles[:16]
        labels = labels[:16]
        labels.append("...")
    ax.legend(handles, labels, fontsize=8, loc="best")
    return _figure_to_base64(fig)


def _to_binary_flag(x: Any) -> int:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0
    if isinstance(x, (int, float, np.integer, np.floating)):
        return int(float(x) != 0.0)
    t = str(x).strip().lower()
    if t in {"1", "true", "yes", "y", "on", "present"}:
        return 1
    if t in {"0", "false", "no", "n", "off", "none", "nan", ""}:
        return 0
    return 1


def _safe_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out


def _ensure_content_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = _safe_numeric(out, ["likes_count", "comments_count", "views_count"])

    for col in BINARY_CONTENT_COLS:
        if col not in out.columns:
            out[col] = 0
        out[col] = out[col].apply(_to_binary_flag).astype(int)

    for col in ["sector", "post_type"]:
        if col not in out.columns:
            out[col] = "unknown"
        out[col] = out[col].fillna("unknown").astype(str)

    return out


def _load_business_module(business_module_path: str | Path):
    path = Path(business_module_path)
    spec = importlib.util.spec_from_file_location("business_clustering_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import business module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _load_business_json(cluster_json_path: str | Path) -> Dict[str, Any]:
    with open(cluster_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _business_feature_columns(payload: Dict[str, Any]) -> list[str]:
    cols = payload.get("kmeans_centroids", {}).get("feature_columns")
    if cols:
        return list(cols)
    return DEFAULT_BUSINESS_FEATURE_COLS


def _assign_nearest_business_cluster(
    row: pd.Series,
    centroids_original: list[dict[str, Any]],
    feature_cols: list[str],
) -> int:
    if not centroids_original:
        raise ValueError("business_cluster_coordinates JSON has no original-space centroids.")

    vec = np.array([float(row.get(c, 0.0)) for c in feature_cols], dtype=float)

    def dist2(centroid: dict[str, Any]) -> float:
        cvec = np.array([float(centroid.get(c, 0.0)) for c in feature_cols], dtype=float)
        return float(np.sum((vec - cvec) ** 2))

    best = min(centroids_original, key=dist2)
    return int(best["kmeans_cluster"])


def _tsne_2d(matrix: np.ndarray, mode: str) -> np.ndarray:
    n = int(matrix.shape[0])
    if n < 2:
        raise ValueError("Need at least 2 rows to create a t-SNE visualization.")

    if mode == "content":
        perplexity = max(5, min(30, n // 10))
    elif mode == "business":
        perplexity = max(5, min(30, n // 5))
    else:
        raise ValueError("mode must be either 'content' or 'business'.")

    perplexity = min(perplexity, max(1, n - 1))
    if n <= 3:
        perplexity = 1

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
    )
    return tsne.fit_transform(matrix)


def build_content_cluster_plot(
    original_posts_df: pd.DataFrame,
    uploaded_posts_df: pd.DataFrame,
    content_pipeline_path: str | Path,
    title: str = "Content posts: original KPI dataset vs uploaded dataset",
) -> Tuple[dict[str, Any], dict[str, Any]]:
    if original_posts_df.empty:
        raise ValueError("original_posts_df is empty.")
    if uploaded_posts_df.empty:
        raise ValueError("uploaded_posts_df is empty.")

    pipe = joblib.load(content_pipeline_path)

    original = _ensure_content_schema(original_posts_df)
    uploaded = _ensure_content_schema(uploaded_posts_df)
    original["source"] = "original_kpi_dataset"
    uploaded["source"] = "uploaded_dataset"

    combined = pd.concat([original, uploaded], ignore_index=True, sort=False)
    combined["predicted_content_cluster"] = pipe.predict(combined[CONTENT_REQUIRED_COLS]).astype(str)

    prep = pipe.named_steps.get("prep") if hasattr(pipe, "named_steps") else None
    if prep is None:
        raise ValueError("The content joblib must be an sklearn Pipeline with a 'prep' step.")
    X = prep.transform(combined[CONTENT_REQUIRED_COLS])
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)

    X2 = _tsne_2d(X, mode="content")
    combined["tsne_1"] = X2[:, 0]
    combined["tsne_2"] = X2[:, 1]

    hover_cols = [c for c in CONTENT_HOVER_CANDIDATES if c in combined.columns]
    image_base64 = _build_matplotlib_scatter_base64(
        df=combined,
        x_col="tsne_1",
        y_col="tsne_2",
        cluster_col="predicted_content_cluster",
        source_col="source",
        title=title,
    )

    metadata = {
        "original_rows": int(len(original)),
        "uploaded_rows": int(len(uploaded)),
        "total_rows": int(len(combined)),
        "content_model_classes": [str(x) for x in getattr(pipe, "classes_", [])],
        "note": "t-SNE was fit on original + uploaded together because sklearn TSNE has no transform method.",
    }
    return {"format": "png_base64", "image_base64": image_base64, "hover_columns": hover_cols}, metadata


def build_business_cluster_plot(
    original_posts_df: pd.DataFrame,
    uploaded_posts_df: pd.DataFrame,
    business_module_path: str | Path,
    business_cluster_json_path: str | Path,
    title: str = "Business clusters: original KPI dataset vs uploaded dataset",
) -> Tuple[dict[str, Any], dict[str, Any]]:
    if original_posts_df.empty:
        raise ValueError("original_posts_df is empty.")
    if uploaded_posts_df.empty:
        raise ValueError("uploaded_posts_df is empty.")

    business_module = _load_business_module(business_module_path)
    payload = _load_business_json(business_cluster_json_path)
    feature_cols = _business_feature_columns(payload)
    centroids_original = payload.get("kmeans_centroids", {}).get("original_space", [])
    json_records = payload.get("records", [])
    json_label_by_name = {
        str(r.get("business_name", "")).strip().lower(): int(r["kmeans_cluster"])
        for r in json_records
        if "business_name" in r and "kmeans_cluster" in r
    }

    original_biz = business_module.preprocess_business_dataset(original_posts_df)
    uploaded_biz = business_module.preprocess_business_dataset(uploaded_posts_df)
    original_biz["source"] = "original_kpi_dataset"
    uploaded_biz["source"] = "uploaded_dataset"

    combined = pd.concat([original_biz, uploaded_biz], ignore_index=True, sort=False)
    for col in feature_cols:
        if col not in combined.columns:
            combined[col] = 0.0
        combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0.0)

    labels: list[int] = []
    for _, row in combined.iterrows():
        name_norm = str(row.get("business_name", "")).strip().lower()
        if name_norm in json_label_by_name:
            labels.append(json_label_by_name[name_norm])
        else:
            labels.append(_assign_nearest_business_cluster(row, centroids_original, feature_cols))
    combined["assigned_business_cluster"] = [str(x) for x in labels]

    X_scaled = StandardScaler().fit_transform(combined[feature_cols].to_numpy(dtype=float))
    X2 = _tsne_2d(X_scaled, mode="business")
    combined["tsne_1"] = X2[:, 0]
    combined["tsne_2"] = X2[:, 1]

    hover_cols = [
        c
        for c in [
            "business_name",
            "avg_engagement_rate",
            "avg_view_rate",
            "avg_comment_rate",
            "posting_frequency",
            "percentage_reels",
            "percentage_images",
            "percentage_carousels",
        ]
        if c in combined.columns
    ]

    image_base64 = _build_matplotlib_scatter_base64(
        df=combined,
        x_col="tsne_1",
        y_col="tsne_2",
        cluster_col="assigned_business_cluster",
        source_col="source",
        title=title,
        # Requested: only uploaded dataset triangles are black.
        source_color_fn=lambda source_name, cluster_key: (
            "#000000" if str(source_name) == "uploaded_dataset" else _color_for_cluster(str(cluster_key))
        ),
    )

    metadata = {
        "original_businesses": int(len(original_biz)),
        "uploaded_businesses": int(len(uploaded_biz)),
        "total_businesses": int(len(combined)),
        "business_feature_columns_used": feature_cols,
        "cluster_assignment_method": "JSON exact name lookup, else nearest original-space centroid",
        "note": "Business t-SNE was recomputed on original + uploaded together. Original scaler was not present in the uploaded artifacts.",
    }
    return {"format": "png_base64", "image_base64": image_base64, "hover_columns": hover_cols}, metadata


def build_cluster_visualization_response(
    original_kpi_dataset: pd.DataFrame,
    uploaded_dataset: pd.DataFrame,
    content_pipeline_path: str | Path,
    business_module_path: str | Path,
    business_cluster_json_path: str | Path,
    business_original_kpi_dataset: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    if business_original_kpi_dataset is None:
        business_original_kpi_dataset = original_kpi_dataset

    content_fig, content_meta = build_content_cluster_plot(
        original_posts_df=original_kpi_dataset,
        uploaded_posts_df=uploaded_dataset,
        content_pipeline_path=content_pipeline_path,
    )
    business_fig, business_meta = build_business_cluster_plot(
        original_posts_df=business_original_kpi_dataset,
        uploaded_posts_df=uploaded_dataset,
        business_module_path=business_module_path,
        business_cluster_json_path=business_cluster_json_path,
    )

    return {
        "content_cluster_plot": content_fig,
        "business_cluster_plot": business_fig,
        "metadata": {
            "content": content_meta,
            "business": business_meta,
        },
    }
