from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


FEATURE_COLS: List[str] = [
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

RATE_COLS: List[str] = [
    "avg_engagement_rate",
    "avg_view_rate",
    "avg_comment_rate",
]


def _to_binary_flag(x) -> int:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0
    if isinstance(x, (int, float)):
        return int(x != 0)
    t = str(x).strip().lower()
    if t in {"1", "true", "yes", "y", "on", "present"}:
        return 1
    if t in {"0", "false", "no", "n", "off", "none", "nan", ""}:
        return 0
    return 1


def preprocess_business_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Notebook-style business preprocessing."""
    work = df.copy()

    if "business_name" not in work.columns:
        raise ValueError("Missing required column: business_name")

    # match notebook behavior
    work = work[work["business_name"] != "Family Market PS"].copy()

    numeric_cols = [
        "followers_count",
        "likes_count",
        "comments_count",
        "views_count",
        "caption_length",
        "hashtags_count",
        "emoji_count",
        "discount_percent",
        "engagement_rate",
        "view_rate",
        "comment_rate",
    ]
    for c in numeric_cols:
        if c not in work.columns:
            work[c] = np.nan
        work[c] = pd.to_numeric(work[c], errors="coerce")

    for c in [
        "CTA_present",
        "promo_post",
        "mentions_location",
        "religious_theme",
        "patriotic_theme",
        "arabic_dialect_style",
    ]:
        if c not in work.columns:
            work[c] = 0
        work[c] = work[c].apply(_to_binary_flag).astype(int)

    if "post_type" not in work.columns:
        work["post_type"] = "other"
    work["post_type"] = work["post_type"].astype(str).str.strip().str.lower()
    work["post_type_std"] = np.select(
        [
            work["post_type"].str.contains("reel", na=False),
            work["post_type"].str.contains("carousel", na=False),
            work["post_type"].str.contains("image|photo|post", na=False),
        ],
        ["reel", "carousel", "image"],
        default="other",
    )

    if "sector" not in work.columns:
        work["sector"] = "unknown"

    work["post_date"] = pd.to_datetime(work.get("post_date"), errors="coerce")

    biz_base = (
        work.groupby(["business_name", "sector"], dropna=False)
        .agg(
            followers_count=("followers_count", "max"),
            posts_count=("business_name", "size"),
            avg_engagement_rate=("engagement_rate", "mean"),
            avg_view_rate=("view_rate", "mean"),
            avg_comment_rate=("comment_rate", "mean"),
            avg_caption_length=("caption_length", "mean"),
            avg_hashtags_count=("hashtags_count", "mean"),
            avg_emoji_count=("emoji_count", "mean"),
            percentage_promo_posts=("promo_post", "mean"),
            percentage_CTA_posts=("CTA_present", "mean"),
            percentage_location_posts=("mentions_location", "mean"),
            percentage_religious_theme=("religious_theme", "mean"),
            percentage_patriotic_theme=("patriotic_theme", "mean"),
            percentage_arabic_dialect_style=("arabic_dialect_style", "mean"),
        )
        .reset_index()
    )

    active_days = (
        work.groupby("business_name")["post_date"]
        .agg(lambda s: (s.max() - s.min()).days + 1 if s.notna().any() else np.nan)
        .rename("active_days")
    )

    biz = biz_base.merge(active_days, on="business_name", how="left")
    biz["posting_frequency"] = np.where(
        biz["active_days"] > 0, biz["posts_count"] / biz["active_days"], np.nan
    )

    pt = pd.crosstab(work["business_name"], work["post_type_std"], normalize="index")
    for c in ["reel", "image", "carousel"]:
        if c not in pt.columns:
            pt[c] = 0.0
    pt = pt[["reel", "image", "carousel"]].rename(
        columns={
            "reel": "percentage_reels",
            "image": "percentage_images",
            "carousel": "percentage_carousels",
        }
    ).reset_index()

    biz = biz.merge(pt, on="business_name", how="left")

    for c in biz.columns:
        if biz[c].dtype.kind in "biufc":
            med = biz[c].median()
            biz[c] = biz[c].fillna(0 if pd.isna(med) else med)

    pct_cols = [
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
    for c in pct_cols:
        biz[c] = biz[c] * 100

    biz["log_followers_count"] = np.log1p(
        pd.to_numeric(biz["followers_count"], errors="coerce").fillna(0)
    )

    keep_cols = ["business_name"] + FEATURE_COLS
    for c in keep_cols:
        if c not in biz.columns:
            biz[c] = 0.0

    return biz[keep_cols].copy()


def _load_cluster_json_info(
    cluster_json_path: str,
) -> Tuple[int, Dict[int, Dict[str, float]], dict]:
    with open(cluster_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    n_clusters = int(payload.get("kmeans_params", {}).get("n_clusters", 3))

    avg_rates = payload.get("kmeans_cluster_avg_rates", [])
    if avg_rates:
        cluster_avg_by_id = {
            int(c["kmeans_cluster"]): {
                "avg_engagement_rate": float(c["avg_engagement_rate"]),
                "avg_view_rate": float(c["avg_view_rate"]),
                "avg_comment_rate": float(c["avg_comment_rate"]),
            }
            for c in avg_rates
            if "kmeans_cluster" in c
        }
    else:
        centroids = payload.get("kmeans_centroids", {}).get("original_space", [])
        cluster_avg_by_id = {
            int(c["kmeans_cluster"]): {
                "avg_engagement_rate": float(c["avg_engagement_rate"]),
                "avg_view_rate": float(c["avg_view_rate"]),
                "avg_comment_rate": float(c["avg_comment_rate"]),
            }
            for c in centroids
            if "kmeans_cluster" in c
        }

    if not cluster_avg_by_id:
        raise ValueError("No cluster average rates found in JSON.")

    return n_clusters, cluster_avg_by_id, payload

def assign_business_cluster_pure_inference(
    biz_df: pd.DataFrame,
    business_name: str,
    cluster_json_path: str,
) -> int:
    """
    Pure inference cluster assignment (no KMeans.fit()):
    1) Lookup exact normalized business_name in JSON records.
    2) Fallback to nearest centroid in JSON original_space.
    """
    if biz_df.empty:
        raise ValueError("No rows available for inference.")

    _, _, payload = _load_cluster_json_info(cluster_json_path)
    records = payload.get("records", [])
    centroid_info = payload.get("kmeans_centroids", {})
    centroid_feature_cols = centroid_info.get("feature_columns", FEATURE_COLS)
    centroids_original = centroid_info.get("original_space", [])

    name_norm = str(business_name).strip().lower()

    # 1) Exact lookup from precomputed JSON records
    matched = [
        r
        for r in records
        if str(r.get("business_name", "")).strip().lower() == name_norm
    ]
    if matched and "kmeans_cluster" in matched[0]:
        return int(matched[0]["kmeans_cluster"])

    # 2) Fallback: nearest centroid in original feature space
    row_mask = biz_df["business_name"].astype(str).str.strip().str.lower().eq(name_norm)
    if not row_mask.any():
        raise ValueError(f"business_name '{business_name}' was not found in dataset.")
    if not centroids_original:
        raise ValueError("No JSON centroids found for fallback assignment.")

    row = biz_df.loc[row_mask].iloc[0]
    vec = np.array([float(row.get(c, 0)) for c in centroid_feature_cols], dtype=float)

    best = min(
        centroids_original,
        key=lambda c: sum(
            (float(c.get(f, 0)) - vec[i]) ** 2 for i, f in enumerate(centroid_feature_cols)
        ),
    )
    return int(best["kmeans_cluster"])


def compare_business_pure_inference(
    biz_features: pd.DataFrame,
    business_name: str,
    cluster_json_path: str = r"C:\Users\hanib\data-mining-project\marketing\notebooks\clustering\artifacts\business_clustering\json\business_cluster_coordinates.json",
) -> pd.DataFrame:
    """
    Pure inference: assign cluster and compare business vs cluster averages.
    Accepts an already-preprocessed business-level DataFrame (output of
    preprocess_business_dataset).
    """
    assigned_cluster = assign_business_cluster_pure_inference(
        biz_df=biz_features,
        business_name=business_name,
        cluster_json_path=cluster_json_path,
    )

    _, cluster_avg_by_id, _ = _load_cluster_json_info(cluster_json_path)
    if assigned_cluster not in cluster_avg_by_id:
        raise ValueError(f"Assigned cluster '{assigned_cluster}' not found in JSON averages.")

    name_norm = str(business_name).strip().lower()
    row_match = biz_features["business_name"].astype(str).str.strip().str.lower().eq(name_norm)
    if not row_match.any():
        raise ValueError(f"business_name '{business_name}' was not found in dataset.")
    row = biz_features.loc[row_match].iloc[0]

    out_rows = []
    for col in RATE_COLS:
        business_val = float(row[col])
        cluster_avg = float(cluster_avg_by_id[assigned_cluster][col])
        if np.isclose(business_val, cluster_avg, rtol=1e-9, atol=1e-12):
            comparison = "equal"
        else:
            comparison = "higher" if business_val > cluster_avg else "lower"

        out_rows.append(
            {
                "business_name": str(row["business_name"]),
                "kmeans_cluster": assigned_cluster,
                "feature": col,
                "business_value": business_val,
                "cluster_avg": cluster_avg,
                "comparison": comparison,
            }
        )

    return pd.DataFrame(out_rows)

