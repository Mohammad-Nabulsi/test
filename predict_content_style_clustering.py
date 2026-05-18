"""
Content-Style Cluster Prediction  (Inference Module)
=====================================================
Self-contained inference module that loads a trained SVM pipeline once,
then predicts content-style clusters for any DataFrame.

Usage::

    from predict_content_style_clustering import predict_content_style_clustering,
    df = pd.read_json("my_kpi_dataset.json")
    results = predict_content_style_clustering(df)

    for biz in results:
        print(biz["business_name"], biz["low_performing"], biz["high_performing"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

# All columns the trained SVM pipeline expects (matching its ColumnTransformer)
FEATURE_COLS: List[str] = [
    "likes_count",
    "comments_count",
    "views_count",
    "CTA_present",
    "promo_post",
    "mentions_location",
    "religious_theme",
    "patriotic_theme",
    "arabic_dialect_style",
    "sector",
    "post_type",
]

BINARY_FEATURE_COLS: List[str] = [
    "CTA_present",
    "promo_post",
    "mentions_location",
    "religious_theme",
    "patriotic_theme",
    "arabic_dialect_style",
]

NUMERIC_FEATURE_COLS: List[str] = [
    "likes_count",
    "comments_count",
    "views_count",
]

BEHAVIOR_NUMERIC_COLS: List[str] = [
    "likes_count",
    "comments_count",
    "views_count",
    "CTA_present",
    "promo_post",
    "mentions_location",
    "religious_theme",
    "patriotic_theme",
    "arabic_dialect_style",
]

CATEGORICAL_BEHAVIOR_COLS: List[str] = [
    "sector",
    "post_type",
]

ENGAGEMENT_COL: str = "engagement_rate"
VIEW_COL: str = "view_rate"
BUSINESS_COL: str = "business_name"

# Default model path (relative to project root)
DEFAULT_MODEL_PATH: Path = Path(
    "notebooks/clustering/artifacts/content_style_clustering/models/content_style_svm_best_pipeline.joblib"
)

# =============================================================================
# CLUSTER MEANING TABLE (hardcoded from domain analysis)
# =============================================================================

CLUSTER_INFO: Dict[int, Dict[str, str]] = {
    -1: {
        "meaning": "Outlier / exceptional local-community content",
        "why": (
            "DBSCAN marks -1 as noise/outliers. The examples are mostly "
            "supermarket posts with very high views/likes, religious tone, "
            "Palestinian/local language, and location/community feeling."
        ),
        "recommendation": (
            "Do not treat this as a normal cluster. Analyze it separately as "
            "\u201cspecial viral/community cases.\u201d Use emotional local language, "
            "religious/community themes, and location-based identity carefully "
            "because they seem to create strong reactions."
        ),
    },
    0: {
        "meaning": "Weak generic promotional content",
        "why": (
            "Very low engagement and view rate. Mostly Fashion and Cafes. "
            "Examples have short/empty captions, weak hooks, little dialect, "
            "no strong CTA, and basic promo style."
        ),
        "recommendation": (
            "Improve caption quality: add a clear hook, stronger visuals, "
            "Arabic/local tone, and a simple CTA. "
            "Avoid posting only product photos or empty captions."
        ),
    },
    1: {
        "meaning": "High-performing organic/personality content",
        "why": (
            "Highest engagement rate, highest view rate, and highest "
            "view-engagement rate. Examples include influencers and funny/simple "
        ),
        "recommendation": (
            "Create more human, relatable, funny, or story-based content. "
            "Avoid over-marketing. This cluster shows that personality and "
            "natural dialect can outperform direct promotion."
        ),
    },
    2: {
        "meaning": "Hard-selling promotional content with weak performance",
        "why": (
            "Lowest engagement rate and low view-engagement rate. Mostly Fashion. "
            "Examples have offers, WhatsApp links, openings, CTAs, promotions, "
            "and location mentions."
        ),
        "recommendation": (
            "Reduce aggressive selling. Make the offer clearer, add storytelling, "
            "social proof, before/after, customer reaction, or emotional value "
            "before the CTA."
        ),
    },
    3: {
        "meaning": "Moderate local product/service promotion",
        "why": (
            "Middle performance. Mostly Fashion and Cafes. Captions are promotional "
            "but more natural than cluster 0. Arabic dialect appears, but "
            "CTA/location/religious themes are not very strong."
        ),
        "recommendation": (
            "This cluster can improve by adding stronger hooks and clearer "
            "reasons to engage. Good for regular product/service posts, "
            "but it needs more personality or interaction questions."
        ),
    },
    4: {
        "meaning": "CTA-heavy gym/tutorial/YouTube promotion content",
        "why": (
            "Mostly Fashion/Gym, with many Gym examples. Engagement is better "
            "than cluster 0 and 2, but still below cluster 1. Posts often promote "
            "episodes, YouTube links, coaches, or fitness content."
        ),
        "recommendation": (
            "Keep the educational/story format, but make the value clear before "
            "saying \u201clink in bio.\u201d Use stronger hooks like transformation, "
            "challenge, mistake, tip, or question."
        ),
    },
}

# =============================================================================
# MODEL LOADING  (singleton  loaded once)
# =============================================================================

_model_cache: Any = None
_loaded_model_path: Path | None = None


def _load_model(model_path: Path) -> Any:
    """Load the SVM pipeline once and cache it."""
    global _model_cache, _loaded_model_path
    if _model_cache is not None and _loaded_model_path == model_path:
        return _model_cache
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    _model_cache = joblib.load(model_path)
    _loaded_model_path = model_path
    return _model_cache


# =============================================================================
# INTERNAL  PREPROCESSING
# =============================================================================

def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and sanitise feature columns for the SVM pipeline."""
    new_df = df[FEATURE_COLS].copy()

    for col in BINARY_FEATURE_COLS:
        if col in new_df.columns:
            new_df[col] = new_df[col].astype(int)

    for col in NUMERIC_FEATURE_COLS:
        if col in new_df.columns:
            new_df[col] = pd.to_numeric(new_df[col], errors="coerce")

    new_df = new_df.fillna(0)
    return new_df


# =============================================================================
# INTERNAL  PERFORMANCE ANALYSIS
# =============================================================================

def _compute_cluster_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Group by business + cluster; compute avg engagement & view rates."""
    perf = (
        df.groupby([BUSINESS_COL, "predicted_cluster"], as_index=False)
        .agg(
            avg_engagement_rate=(ENGAGEMENT_COL, "mean"),
            avg_view_rate=(VIEW_COL, "mean"),
            post_count=("predicted_cluster", "size"),
        )
        .sort_values([BUSINESS_COL, "predicted_cluster"])
    )
    perf["avg_engagement_rate"] = perf["avg_engagement_rate"].round(6)
    perf["avg_view_rate"] = perf["avg_view_rate"].round(6)
    return perf


def _safe_mode(series: pd.Series) -> str:
    """Return first mode as string, or 'unknown' when unavailable."""
    mode = series.dropna().mode()
    if mode.empty:
        return "unknown"
    return str(mode.iloc[0])


def _compute_cluster_behavior(df: pd.DataFrame) -> pd.DataFrame:
    """Compute average behavior columns per business + predicted cluster."""
    behavior = (
        df.groupby([BUSINESS_COL, "predicted_cluster"], as_index=False)
        .agg(
            avg_likes_count=("likes_count", "mean"),
            avg_comments_count=("comments_count", "mean"),
            avg_views_count=("views_count", "mean"),
            avg_CTA_present=("CTA_present", "mean"),
            avg_promo_post=("promo_post", "mean"),
            avg_mentions_location=("mentions_location", "mean"),
            avg_religious_theme=("religious_theme", "mean"),
            avg_patriotic_theme=("patriotic_theme", "mean"),
            avg_arabic_dialect_style=("arabic_dialect_style", "mean"),
            dominant_sector=("sector", _safe_mode),
            dominant_post_type=("post_type", _safe_mode),
        )
    )

    for col in [
        "avg_likes_count",
        "avg_comments_count",
        "avg_views_count",
        "avg_CTA_present",
        "avg_promo_post",
        "avg_mentions_location",
        "avg_religious_theme",
        "avg_patriotic_theme",
        "avg_arabic_dialect_style",
    ]:
        behavior[col] = behavior[col].round(6)

    return behavior


def _identify_extremes(perf_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Find low- and high- performing clusters per business using a normalised composite score."""
    results: Dict[str, Dict[str, Any]] = {}

    for business in perf_df[BUSINESS_COL].unique():
        biz = perf_df[perf_df[BUSINESS_COL] == business].copy()
        if biz.empty:
            continue

        max_eng = biz["avg_engagement_rate"].max() or 1.0
        max_view = biz["avg_view_rate"].max() or 1.0

        biz["__score"] = (
            biz["avg_engagement_rate"] / max_eng
            + biz["avg_view_rate"] / max_view
        )

        low_row = biz.loc[biz["__score"].idxmin()]
        high_row = biz.loc[biz["__score"].idxmax()]

        low_c = int(low_row["predicted_cluster"])
        high_c = int(high_row["predicted_cluster"])

        results[business] = {
            "business_name": business,
            "low_performing": {
                "cluster": low_c,
                "meaning": CLUSTER_INFO.get(low_c, {}).get("meaning", f"Unknown ({low_c})"),
                "avg_engagement_rate": float(low_row["avg_engagement_rate"]),
                "avg_view_rate": float(low_row["avg_view_rate"]),
            },
            "high_performing": {
                "cluster": high_c,
                "meaning": CLUSTER_INFO.get(high_c, {}).get("meaning", f"Unknown ({high_c})"),
                "avg_engagement_rate": float(high_row["avg_engagement_rate"]),
                "avg_view_rate": float(high_row["avg_view_rate"]),
            },
            "recommendation": _build_recommendation(
                low_c,
                high_c,
                float(low_row["avg_engagement_rate"]),
                float(low_row["avg_view_rate"]),
                float(high_row["avg_engagement_rate"]),
                float(high_row["avg_view_rate"]),
            ),
            "all_clusters": [
                {
                    "cluster": int(row["predicted_cluster"]),
                    "meaning": CLUSTER_INFO.get(int(row["predicted_cluster"]), {}).get("meaning", f"Unknown ({int(row['predicted_cluster'])})"),
                    "avg_engagement_rate": float(row["avg_engagement_rate"]),
                    "avg_view_rate": float(row["avg_view_rate"]),
                    "post_count": int(row["post_count"]),
                    "avg_likes_count": float(row.get("avg_likes_count", 0.0)),
                    "avg_comments_count": float(row.get("avg_comments_count", 0.0)),
                    "avg_views_count": float(row.get("avg_views_count", 0.0)),
                    "avg_CTA_present": float(row.get("avg_CTA_present", 0.0)),
                    "avg_promo_post": float(row.get("avg_promo_post", 0.0)),
                    "avg_mentions_location": float(row.get("avg_mentions_location", 0.0)),
                    "avg_religious_theme": float(row.get("avg_religious_theme", 0.0)),
                    "avg_patriotic_theme": float(row.get("avg_patriotic_theme", 0.0)),
                    "avg_arabic_dialect_style": float(row.get("avg_arabic_dialect_style", 0.0)),
                    "dominant_sector": str(row.get("dominant_sector", "unknown")),
                    "dominant_post_type": str(row.get("dominant_post_type", "unknown")),
                }
                for _, row in biz.iterrows()
            ],
        }
    return results


def _build_recommendation(
    low_cluster: int,
    high_cluster: int,
    low_eng: float,
    low_view: float,
    high_eng: float,
    high_view: float,
) -> str:
    """Build a recommendation comparing the low cluster to the high cluster."""
    base = CLUSTER_INFO.get(low_cluster, {}).get(
        "recommendation", "No specific recommendation available."
    )
    high_meaning = CLUSTER_INFO.get(high_cluster, {}).get(
        "meaning", f"Cluster {high_cluster}"
    )
    return (
        f"{base} "
        f"{high_meaning}"
    )


# =============================================================================
# PUBLIC  INFERENCE FUNCTION
# =============================================================================

def predict_content_style_clustering(
    df: pd.DataFrame,
    model_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    """
    Predict content-style clusters for every row in *df* and return a
    per-business performance analysis.

    The input DataFrame **must** contain these columns::

        business_name
        engagement_rate
        view_rate
        likes_count    comments_count    views_count
        CTA_present     promo_post         mentions_location
        religious_theme  patriotic_theme    arabic_dialect_style
        sector           post_type

    Parameters
    ----------
    df : pd.DataFrame
        Social-media KPI dataset (one row per post).
    model_path : str or Path, optional
        Path to the .joblib SVM pipeline.  Uses the default model if omitted.

    Returns
    -------
    list[dict]
        One dict per business. Each dict contains::

            {
                "business_name": str,
                "low_performing":  {"cluster": int, "meaning": str, ...},
                "high_performing": {"cluster": int, "meaning": str, ...},
                "recommendation": str,
                "all_clusters": [{"cluster": int, ...}, ...],
            }
    """
    mpath = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH

    # ---- Validate required columns ----
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise KeyError(
            f"Missing feature columns: {missing_features}\n"
            f"Available: {df.columns.tolist()}"
        )
    for col, label in [(ENGAGEMENT_COL, "engagement"), (VIEW_COL, "view")]:
        if col not in df.columns:
            raise KeyError(f"Missing performance column '{col}' ({label} rate).")
    if BUSINESS_COL not in df.columns:
        raise KeyError(f"Missing business column '{BUSINESS_COL}'.")

    # ---- Predict ----
    model = _load_model(mpath)
    X = _prepare_features(df)
    df = df.copy()
    df["predicted_cluster"] = model.predict(X)

    # ---- Analyse ----
    perf = _compute_cluster_performance(df)
    behavior = _compute_cluster_behavior(df)
    merged = perf.merge(
        behavior,
        on=[BUSINESS_COL, "predicted_cluster"],
        how="left",
    )
    return list(_identify_extremes(merged).values())
