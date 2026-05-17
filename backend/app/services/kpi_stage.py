from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KpiSpec:
    required_inputs: list[str]
    optional_inputs: list[str]
    outputs: list[str]


KPI_SPEC = KpiSpec(
    required_inputs=[
        "business_name",
        "sector",
        "followers_count",
        "post_date",
        "post_type",
        "likes_count",
        "comments_count",
        "views_count",
    ],
    optional_inputs=[
        "posting_hour",
        "caption_length",
        "hashtags_count",
        "emoji_count",
        "discount_percent",
        "CTA_present",
        "promo_post",
        "mentions_location",
        "arabic_dialect_style",
    ],
    outputs=[
        "kpi dataset with engagement/view/comment KPIs",
        "weekly key (week)",
        "summary tables by sector/business/post_type/week",
    ],
)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace({0: np.nan})
    return (numerator / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _safe_qcut(series: pd.Series, q: int, labels: list[str]) -> pd.Series:
    try:
        return pd.qcut(series, q=q, labels=labels, duplicates="drop")
    except Exception:
        ranked = series.rank(method="average", pct=True)
        bins = np.linspace(0, 1, len(labels) + 1)
        return pd.cut(ranked, bins=bins, labels=labels, include_lowest=True)


def _coerce_month_series(month_series: pd.Series, post_date_series: pd.Series) -> pd.Series:
    """
    Handle numeric or textual month values (e.g., "April", "Apr") safely.
    """
    numeric = pd.to_numeric(month_series, errors="coerce")
    if numeric.isna().any():
        text_month = pd.to_datetime(month_series.astype(str), format="%B", errors="coerce").dt.month
        short_text_month = pd.to_datetime(month_series.astype(str), format="%b", errors="coerce").dt.month
        numeric = numeric.fillna(text_month).fillna(short_text_month)
    fallback = pd.to_datetime(post_date_series, errors="coerce").dt.month
    return numeric.fillna(fallback).fillna(1).clip(lower=1, upper=12).astype(int)


def engineer_kpis_from_notebook_logic(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    out["post_date"] = pd.to_datetime(out["post_date"], errors="coerce")

    for col in ["likes_count", "comments_count", "views_count", "followers_count", "posting_hour", "caption_length", "hashtags_count", "emoji_count"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    for bcol in ["CTA_present", "promo_post", "mentions_location", "arabic_dialect_style"]:
        if bcol not in out.columns:
            out[bcol] = False
        out[bcol] = out[bcol].astype(bool)

    out["engagement"] = out["likes_count"] + out["comments_count"]
    out["engagement_rate"] = _safe_divide(out["engagement"], out["followers_count"])
    out["like_rate"] = _safe_divide(out["likes_count"], out["followers_count"])
    out["comment_rate"] = _safe_divide(out["comments_count"], out["followers_count"])
    out["view_rate"] = _safe_divide(out["views_count"], out["followers_count"])
    out["view_engagement_rate"] = _safe_divide(out["engagement"], out["views_count"])

    # Compatibility aliases with the old backend naming.
    out["total_engagement"] = out["engagement"]
    out["engagement_rate_followers"] = out["engagement_rate"]
    out["likes_rate_followers"] = out["like_rate"]
    out["comments_rate_followers"] = out["comment_rate"]
    out["views_per_follower"] = out["view_rate"]
    out["like_view_rate"] = _safe_divide(out["likes_count"], out["views_count"])
    out["comment_view_rate"] = _safe_divide(out["comments_count"], out["views_count"])

    out["week"] = out["post_date"].dt.to_period("W").astype(str)
    month_source = out["month"] if "month" in out.columns else pd.Series([np.nan] * len(out), index=out.index)
    out["month"] = _coerce_month_series(month_source, out["post_date"])

    post_type = out.get("post_type", pd.Series(["unknown"] * len(out))).astype(str).str.lower()
    out["is_video"] = (post_type == "video").astype(int)
    out["is_reel"] = (post_type == "reel").astype(int)

    out["posting_time_bin"] = "night"
    out.loc[(out["posting_hour"] >= 6) & (out["posting_hour"] < 12), "posting_time_bin"] = "morning"
    out.loc[(out["posting_hour"] >= 12) & (out["posting_hour"] < 17), "posting_time_bin"] = "afternoon"
    out.loc[(out["posting_hour"] >= 17) & (out["posting_hour"] < 21), "posting_time_bin"] = "evening"

    out["caption_length_bin"] = pd.cut(
        out["caption_length"], bins=[-1, 60, 140, np.inf], labels=["short", "medium", "long"]
    ).astype(str)
    out["hashtags_count_bin"] = pd.cut(
        out["hashtags_count"], bins=[-1, 0, 4, np.inf], labels=["none", "few", "many"]
    ).astype(str)
    out["emoji_count_bin"] = pd.cut(
        out["emoji_count"], bins=[-1, 0, 3, np.inf], labels=["none", "few", "many"]
    ).astype(str)

    out["engagement_level"] = _safe_qcut(out["engagement_rate"], q=3, labels=["low", "medium", "high"]).astype(str)
    out["business_size_bin"] = _safe_qcut(out["followers_count"], q=3, labels=["small", "medium", "large"]).astype(str)

    out["high_engagement"] = (out["engagement_level"] == "high").astype(int)
    out["high_view_rate"] = (out["view_rate"] >= out["view_rate"].quantile(0.67)).astype(int)
    out["high_comment_rate"] = (out["comment_rate"] >= out["comment_rate"].quantile(0.67)).astype(int)

    for col in [
        "engagement_rate",
        "like_rate",
        "comment_rate",
        "view_rate",
        "view_engagement_rate",
        "engagement_rate_followers",
        "likes_rate_followers",
        "comments_rate_followers",
        "views_per_follower",
    ]:
        out[col] = out[col].replace([np.inf, -np.inf], 0).fillna(0.0)

    summaries: dict[str, list[dict[str, Any]]] = {}
    for key, grp in {
        "sector": ["sector"],
        "business": ["business_name", "sector"],
        "post_type": ["post_type"],
        "week": ["week"],
    }.items():
        for col in grp:
            if col not in out.columns:
                out[col] = "Unknown"
        summary_df = (
            out.groupby(grp, as_index=False)
            .agg(
                posts_count=("business_name", "size"),
                engagement_mean=("engagement", "mean"),
                engagement_rate_mean=("engagement_rate", "mean"),
                view_rate_mean=("view_rate", "mean"),
                comment_rate_mean=("comment_rate", "mean"),
            )
            .sort_values("engagement_rate_mean", ascending=False)
        )
        summaries[key] = summary_df.to_dict(orient="records")

    invalid_rate_counts = {
        c: int(np.isinf(out[c]).sum() + out[c].isna().sum())
        for c in ["engagement_rate", "like_rate", "comment_rate", "view_rate", "view_engagement_rate"]
    }

    meta = {
        "rows": int(len(out)),
        "invalid_rate_counts": invalid_rate_counts,
        "followers_count_zero_rows": int((out["followers_count"] == 0).sum()),
        "views_count_zero_rows": int((out["views_count"] == 0).sum()),
        "summary_tables": summaries,
    }
    return out.reset_index(drop=True), meta
