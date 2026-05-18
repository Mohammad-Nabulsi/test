import numpy as np
import pandas as pd
from .utils import safe_divide


def _safe_qcut(series, q, labels):
    try:
        return pd.qcut(series, q=q, labels=labels, duplicates="drop")
    except Exception:
        ranked = series.rank(method="average", pct=True)
        bins = np.linspace(0, 1, len(labels) + 1)
        return pd.cut(ranked, bins=bins, labels=labels, include_lowest=True)


def engineer_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["engagement"] = out["likes_count"] + out["comments_count"]
    out["engagement_rate"] = safe_divide(out["engagement"], out["followers_count"], default=0.0)
    out["like_rate"] = safe_divide(out["likes_count"], out["followers_count"], default=0.0)
    out["comment_rate"] = safe_divide(out["comments_count"], out["followers_count"], default=0.0)
    out["view_rate"] = safe_divide(out["views_count"], out["followers_count"], default=0.0)
    out["view_engagement_rate"] = safe_divide(out["engagement"], out["views_count"], default=0.0)

    out["week"] = out["post_date"].dt.isocalendar().week.astype(int)
    out["month"] = out["post_date"].dt.month.astype(int)

    out["is_video"] = (out["post_type"].str.lower() == "video").astype(int)
    out["is_reel"] = (out["post_type"].str.lower() == "reel").astype(int)

    hour = pd.to_numeric(out["posting_hour"], errors="coerce").fillna(0)
    out["posting_time_bin"] = "night"
    out.loc[(hour >= 6) & (hour < 12), "posting_time_bin"] = "morning"
    out.loc[(hour >= 12) & (hour < 17), "posting_time_bin"] = "afternoon"
    out.loc[(hour >= 17) & (hour < 21), "posting_time_bin"] = "evening"

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

    for col in ["engagement_rate", "like_rate", "comment_rate", "view_rate", "view_engagement_rate"]:
        out[col] = out[col].replace([np.inf, -np.inf], 0).fillna(0)

    return out


def build_post_feature_sets(df: pd.DataFrame):
    performance = [
        "engagement_rate",
        "view_rate",
        "comment_rate",
        "like_rate",
        "view_engagement_rate",
        "discount_percent",
        "followers_count",
    ]
    content = [
        "post_type",
        "posting_time_bin",
        "caption_length_bin",
        "hashtags_count_bin",
        "emoji_count_bin",
        "language",
        "CTA_present",
        "promo_post",
        "mentions_location",
        "arabic_dialect_style",
    ]
    return {
        "performance-only": [c for c in performance if c in df.columns],
        "content-only": [c for c in content if c in df.columns],
        "combined": [c for c in performance + content if c in df.columns],
    }


def aggregate_business_features(df: pd.DataFrame):
    return (
        df.groupby(["business_name", "sector"], as_index=False)
        .agg(
            followers_count=("followers_count", "median"),
            posts_count=("business_name", "size"),
            engagement_mean=("engagement", "mean"),
            engagement_rate_mean=("engagement_rate", "mean"),
            view_rate_mean=("view_rate", "mean"),
            comment_rate_mean=("comment_rate", "mean"),
            promo_share=("promo_post", "mean"),
            cta_share=("CTA_present", "mean"),
            location_share=("mentions_location", "mean"),
            dialect_share=("arabic_dialect_style", "mean"),
            discount_mean=("discount_percent", "mean"),
            video_share=("is_video", "mean"),
            reel_share=("is_reel", "mean"),
        )
        .reset_index(drop=True)
    )
