from __future__ import annotations

from typing import Dict

import pandas as pd


def _group_engagement(df: pd.DataFrame, group_col: str) -> list[dict]:
    if group_col not in df.columns:
        return []
    g = (
        df.groupby(group_col, dropna=False)
        .agg(
            posts=("business_name", "size"),
            avg_total_engagement=("total_engagement", "mean"),
            avg_engagement_rate=("engagement_rate_followers", "mean"),
            avg_views_per_follower=("views_per_follower", "mean"),
        )
        .reset_index()
    )
    # Normalize types for JSON
    g[group_col] = g[group_col].astype(str)
    return g.sort_values("avg_engagement_rate", ascending=False).to_dict(orient="records")


def build_eda_summary(kpis_df: pd.DataFrame) -> Dict:
    df = kpis_df.copy()

    # Top businesses by engagement rate
    top_businesses = (
        df.groupby(["business_name", "sector"], dropna=False)
        .agg(
            posts=("business_name", "size"),
            followers=("followers_count", "max"),
            avg_engagement_rate=("engagement_rate_followers", "mean"),
            avg_views_per_follower=("views_per_follower", "mean"),
        )
        .reset_index()
        .sort_values(["avg_engagement_rate", "posts"], ascending=False)
        .head(20)
        .to_dict(orient="records")
    )

    # Top posts by content quality score
    top_posts = (
        df.sort_values("content_quality_score", ascending=False)
        .head(20)[
            [
                "business_name",
                "sector",
                "post_date",
                "post_type",
                "language",
                "CTA_present",
                "promo_post",
                "total_engagement",
                "views_count",
                "engagement_rate_followers",
                "views_per_follower",
                "content_quality_score",
            ]
        ]
        .copy()
    )
    if "post_date" in top_posts.columns:
        top_posts["post_date"] = top_posts["post_date"].astype(str)
    top_posts = top_posts.to_dict(orient="records")

    summary = {
        "engagement_by_sector": _group_engagement(df, "sector"),
        "engagement_by_post_type": _group_engagement(df, "post_type"),
        "engagement_by_day_of_week": _group_engagement(df, "day_of_week"),
        "engagement_by_posting_hour": _group_engagement(df, "posting_hour"),
        "engagement_by_language": _group_engagement(df, "language"),
        "engagement_by_CTA_present": _group_engagement(df, "CTA_present"),
        "engagement_by_promo_post": _group_engagement(df, "promo_post"),
        "engagement_by_arabic_dialect_style": _group_engagement(df, "arabic_dialect_style"),
        "top_businesses_by_engagement_rate_followers": top_businesses,
        "top_posts_by_content_quality_score": top_posts,
        "dataset_overview": {
            "rows": int(len(df)),
            "businesses": int(df["business_name"].nunique()) if "business_name" in df.columns else 0,
            "sectors": int(df["sector"].nunique()) if "sector" in df.columns else 0,
        },
    }
    return summary

