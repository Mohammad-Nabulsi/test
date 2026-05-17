from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom2 = denom.replace({0: np.nan})
    return (numer / denom2).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _bin_label(value: float, bins: list[float], labels: list[str]) -> str:
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return labels[i]
    return labels[-1]


def engineer_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["total_engagement"] = out["likes_count"].fillna(0) + out["comments_count"].fillna(0)
    out["engagement_rate_followers"] = _safe_div(out["total_engagement"], out["followers_count"])
    out["likes_rate_followers"] = _safe_div(out["likes_count"], out["followers_count"])
    out["comments_rate_followers"] = _safe_div(out["comments_count"], out["followers_count"])
    out["views_per_follower"] = _safe_div(out["views_count"], out["followers_count"])

    out["view_engagement_rate"] = _safe_div(out["total_engagement"], out["views_count"])
    out["like_view_rate"] = _safe_div(out["likes_count"], out["views_count"])
    out["comment_view_rate"] = _safe_div(out["comments_count"], out["views_count"])

    # Grouping columns
    caption_bins = [0, 60, 120, 220, 100000]
    caption_labels = ["short", "medium", "long", "very_long"]
    out["caption_length_group"] = out["caption_length"].fillna(0).apply(
        lambda x: _bin_label(float(x), caption_bins, caption_labels)
    )

    hashtag_bins = [0, 1, 4, 9, 100000]
    hashtag_labels = ["none", "few", "some", "many"]
    out["hashtag_group"] = out["hashtags_count"].fillna(0).apply(lambda x: _bin_label(float(x), hashtag_bins, hashtag_labels))

    emoji_bins = [0, 1, 3, 6, 100000]
    emoji_labels = ["none", "few", "some", "many"]
    out["emoji_group"] = out["emoji_count"].fillna(0).apply(lambda x: _bin_label(float(x), emoji_bins, emoji_labels))

    discount_bins = [0, 1, 10, 25, 50, 101]
    discount_labels = ["none", "tiny", "small", "medium", "big"]
    out["discount_range"] = out["discount_percent"].fillna(0).apply(
        lambda x: _bin_label(float(x), discount_bins, discount_labels)
    )

    # Labels
    # Use quantiles to stay robust across datasets.
    er = out["engagement_rate_followers"].fillna(0)
    thr_high = float(er.quantile(0.75)) if len(er) >= 20 else float(er.mean() + er.std())
    thr_viral = float(out["views_per_follower"].fillna(0).quantile(0.9)) if len(out) >= 20 else float(out["views_per_follower"].mean() + 2 * out["views_per_follower"].std())
    thr_comments = float(out["comments_rate_followers"].fillna(0).quantile(0.85)) if len(out) >= 20 else float(out["comments_rate_followers"].mean() + 2 * out["comments_rate_followers"].std())

    out["high_engagement"] = er >= max(thr_high, 0.0001)
    out["viral_post"] = out["views_per_follower"].fillna(0) >= max(thr_viral, 0.01)
    out["high_comment_post"] = out["comments_rate_followers"].fillna(0) >= max(thr_comments, 0.00005)

    # Content quality score: intentionally simple, explainable.
    # Encourage: engagement rate, comments rate, non-salesy mix, and avoid zero-view anomalies.
    score = (
        60 * out["engagement_rate_followers"]
        + 25 * out["comments_rate_followers"]
        + 10 * out["views_per_follower"]
        + 5 * out["CTA_present"].astype(int)
        - 3 * out["promo_post"].astype(int)
        + 2 * out["arabic_dialect_style"].astype(int)
    )
    out["content_quality_score"] = score.clip(lower=0).astype(float)

    return out

