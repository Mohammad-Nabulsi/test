"""
business_feature_engineering.py

Reusable feature-engineering script for the business clustering notebook.
It takes a post-level pandas DataFrame and returns the same business-level
features used in 04_business_clustring.ipynb.

Expected input columns, when available:
- business_name, sector, followers_count
- likes_count, comments_count, views_count
- caption_length, hashtags_count, emoji_count
- engagement_rate, view_rate, comment_rate
- CTA_present, promo_post, mentions_location
- religious_theme, patriotic_theme, arabic_dialect_style
- post_type, post_date

If engagement_rate, view_rate, or comment_rate are missing, they are computed
from likes/comments/views/followers when possible.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd


# Final feature columns used in the notebook's clustering matrix.
# Note: percentage_carousels is created in the business table, but the uploaded
# notebook does not include it in feature_cols for clustering.
FEATURE_COLS = [
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
    "percentage_promo_posts",
    "percentage_CTA_posts",
    "percentage_location_posts",
    "percentage_religious_theme",
    "percentage_patriotic_theme",
    "percentage_arabic_dialect_style",
]


BINARY_COLS = [
    "CTA_present",
    "promo_post",
    "mentions_location",
    "religious_theme",
    "patriotic_theme",
    "arabic_dialect_style",
]


NUMERIC_COLS = [
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
    "engagement",
]


PERCENTAGE_COLS = [
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


def to_binary_flag(value) -> int:
    """Convert mixed boolean-like values into 0/1, matching the notebook logic."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0

    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value != 0)

    text = str(value).strip().lower()

    if text in {"1", "true", "yes", "y", "on", "present"}:
        return 1

    if text in {"0", "false", "no", "n", "off", "none", "nan", ""}:
        return 0

    # Same behavior as the notebook: unknown non-empty values count as present.
    return 1


def _ensure_required_columns(work: pd.DataFrame) -> pd.DataFrame:
    """Create missing columns used by the notebook so the function works safely."""
    work = work.copy()

    if "business_name" not in work.columns:
        raise ValueError("Input DataFrame must contain a 'business_name' column.")

    if "sector" not in work.columns:
        work["sector"] = "Unknown"

    if "post_type" not in work.columns:
        work["post_type"] = "other"

    if "post_date" not in work.columns:
        work["post_date"] = pd.NaT

    for col in NUMERIC_COLS:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        else:
            work[col] = np.nan

    for col in BINARY_COLS:
        if col in work.columns:
            work[col] = work[col].apply(to_binary_flag).astype(int)
        else:
            work[col] = 0

    return work


def _create_missing_rates(work: pd.DataFrame) -> pd.DataFrame:
    """
    Create engagement_rate, view_rate, and comment_rate when they are missing.

    The notebook expects these columns to already exist in the KPI dataset.
    This function makes the script reusable for raw post-level DataFrames too.
    """
    work = work.copy()

    if work["engagement"].isna().all():
        work["engagement"] = work["likes_count"].fillna(0) + work["comments_count"].fillna(0)

    valid_followers = work["followers_count"] > 0

    if work["engagement_rate"].isna().all():
        work["engagement_rate"] = np.where(
            valid_followers,
            work["engagement"] / work["followers_count"],
            np.nan,
        )

    if work["view_rate"].isna().all():
        work["view_rate"] = np.where(
            valid_followers,
            work["views_count"] / work["followers_count"],
            np.nan,
        )

    if work["comment_rate"].isna().all():
        work["comment_rate"] = np.where(
            valid_followers,
            work["comments_count"] / work["followers_count"],
            np.nan,
        )

    return work


def prepare_business_features(
    df: pd.DataFrame,
    exclude_businesses: Optional[Iterable[str]] = ("Family Market PS",),
    return_feature_matrix: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Build the same business-level features used inside the business clustering notebook.

    Parameters
    ----------
    df:
        Post-level input DataFrame.
    exclude_businesses:
        Optional iterable of business names to remove before aggregation.
        The notebook removes "Family Market PS".
        Pass None or an empty list if you do not want to remove anything.
    return_feature_matrix:
        If False, return the full business-level feature DataFrame.
        If True, return (business_features, X), where X contains only FEATURE_COLS
        cleaned the same way as the notebook before scaling/clustering.

    Returns
    -------
    pd.DataFrame or tuple[pd.DataFrame, pd.DataFrame]
        The business-level features, and optionally the clustering matrix X.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    work = df.copy()

    # Same exclusion used in the notebook.
    if exclude_businesses:
        exclude_set = set(exclude_businesses)
        work = work[~work["business_name"].isin(exclude_set)].copy()

    work = _ensure_required_columns(work)
    work = _create_missing_rates(work)

    # Normalize post type into reel / carousel / image / other.
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

    # Important: do not convert post_date to numeric first.
    # The notebook later uses pd.to_datetime, so this keeps date strings valid.
    work["post_date"] = pd.to_datetime(work["post_date"], errors="coerce")

    # Business-level aggregation.
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

    # Posting frequency = posts per active day.
    active_days = (
        work.groupby("business_name")["post_date"]
        .agg(lambda s: (s.max() - s.min()).days + 1 if s.notna().any() else np.nan)
        .rename("active_days")
    )

    biz = biz_base.merge(active_days, on="business_name", how="left")
    biz["posting_frequency"] = np.where(
        biz["active_days"] > 0,
        biz["posts_count"] / biz["active_days"],
        np.nan,
    )

    # Post-type percentages.
    post_type_table = pd.crosstab(work["business_name"], work["post_type_std"], normalize="index")

    for col in ["reel", "image", "carousel"]:
        if col not in post_type_table.columns:
            post_type_table[col] = 0

    post_type_table = (
        post_type_table[["reel", "image", "carousel"]]
        .rename(
            columns={
                "reel": "percentage_reels",
                "image": "percentage_images",
                "carousel": "percentage_carousels",
            }
        )
        .reset_index()
    )

    biz = biz.merge(post_type_table, on="business_name", how="left")

    # Fill remaining numeric nulls with median, same as the notebook.
    for col in biz.columns:
        if biz[col].dtype.kind in "biufc":
            median_value = biz[col].median()
            if pd.isna(median_value):
                median_value = 0
            biz[col] = biz[col].fillna(median_value)

    # Convert proportions to percentages for readability and for consistency
    # with the notebook feature values.
    for col in PERCENTAGE_COLS:
        if col in biz.columns:
            biz[col] = biz[col] * 100

    # Final feature used for clustering.
    biz["log_followers_count"] = np.log1p(biz["followers_count"])

    # Keep a predictable column order.
    first_cols = [
        "business_name",
        "sector",
        "followers_count",
        "posts_count",
        "active_days",
    ]
    ordered_cols = first_cols + [c for c in biz.columns if c not in first_cols]
    biz = biz[ordered_cols]

    if not return_feature_matrix:
        return biz

    X = biz[FEATURE_COLS].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    for col in FEATURE_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        median_value = X[col].median()
        if pd.isna(median_value):
            median_value = 0
        X[col] = X[col].fillna(median_value)

    return biz, X


if __name__ == "__main__":
    # Example usage:
    # python business_feature_engineering.py input_posts.json output_business_features.csv
    import argparse

    parser = argparse.ArgumentParser(description="Create business clustering features from a post-level dataset.")
    parser.add_argument("input_path", help="Path to input file: .json, .jsonl, or .csv")
    parser.add_argument("output_path", help="Path to output file: .csv or .json")
    args = parser.parse_args()

    input_path = args.input_path.lower()

    if input_path.endswith(".csv"):
        input_df = pd.read_csv(args.input_path)
    elif input_path.endswith(".jsonl"):
        input_df = pd.read_json(args.input_path, lines=True)
    elif input_path.endswith(".json"):
        try:
            input_df = pd.read_json(args.input_path)
        except ValueError:
            input_df = pd.read_json(args.input_path, lines=True)
    else:
        raise ValueError("Unsupported input file type. Use .csv, .json, or .jsonl")

    business_features = prepare_business_features(input_df)

    output_path = args.output_path.lower()

    if output_path.endswith(".csv"):
        business_features.to_csv(args.output_path, index=False)
    elif output_path.endswith(".json"):
        business_features.to_json(args.output_path, orient="records", force_ascii=False, indent=2)
    else:
        raise ValueError("Unsupported output file type. Use .csv or .json")

    print(f"Saved business features to: {args.output_path}")
