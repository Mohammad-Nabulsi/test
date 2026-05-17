"""
similar_business_recommender.py

KNN-based recommender that compares one business with similar successful
businesses and generates practical, business-oriented recommendations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SUCCESS_COMPONENTS = {
    "avg_engagement_rate_followers": {
        "label": "Engagement contribution",
        "weight": 0.42,
    },
    "avg_views_per_follower": {
        "label": "Reach contribution",
        "weight": 0.18,
    },
    "avg_shares_count": {
        "label": "Shares contribution",
        "weight": 0.16,
    },
    "avg_saves_count": {
        "label": "Saves contribution",
        "weight": 0.14,
    },
    "avg_comments_count": {
        "label": "Conversation contribution",
        "weight": 0.10,
    },
}


BASE_KPI_COLUMNS = [
    "engagement_rate_followers",
    "views_per_follower",
    "likes_count",
    "comments_count",
    "shares_count",
    "saves_count",
]


BEHAVIOR_COLUMNS = [
    "posting_hour",
    "hashtags_count",
    "caption_length",
    "emoji_count",
    "CTA_present",
    "promo_post",
    "mentions_location",
    "arabic_dialect_style",
    "religious_theme",
    "patriotic_theme",
]


POST_TYPE_VALUES = ["image", "reel", "video", "carousel"]


FEATURE_IMPORTANCE = {
    "avg_engagement_rate_followers": 1.00,
    "avg_views_per_follower": 0.90,
    "avg_shares_count": 0.78,
    "avg_saves_count": 0.74,
    "avg_comments_count": 0.66,
    "pct_CTA_present": 0.54,
    "pct_mentions_location": 0.48,
    "pct_arabic_dialect_style": 0.44,
    "pct_promo_post": 0.42,
    "top_post_type": 0.40,
    "avg_posting_hour": 0.30,
}


def _safe_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return 0.0
    return float(numeric.mean())


def _safe_max(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return 0.0
    return float(numeric.max())


def _mode_or_unknown(series: pd.Series) -> str:
    clean = series.dropna()
    if clean.empty:
        return "Unknown"
    mode_values = clean.mode()
    if mode_values.empty:
        return str(clean.iloc[0])
    return str(mode_values.iloc[0])


def _boolean_rate(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.astype(bool).mean())


def _available_columns(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    return [column for column in columns if column in df.columns]


def _format_number(value: object, digits: int = 3) -> str:
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _prepare_post_metrics(posts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds missing KPI columns from raw post metrics.
    """

    df = posts_df.copy()

    if "shares_count" not in df.columns:
        df["shares_count"] = 0.0

    if "saves_count" not in df.columns:
        df["saves_count"] = 0.0

    if "comments_count" not in df.columns:
        df["comments_count"] = 0.0

    if "likes_count" not in df.columns:
        df["likes_count"] = 0.0

    if "views_count" not in df.columns:
        df["views_count"] = 0.0

    if "followers_count" not in df.columns:
        df["followers_count"] = 0.0

    followers = pd.to_numeric(df["followers_count"], errors="coerce").replace(0, pd.NA)
    likes = pd.to_numeric(df["likes_count"], errors="coerce").fillna(0)
    comments = pd.to_numeric(df["comments_count"], errors="coerce").fillna(0)
    views = pd.to_numeric(df["views_count"], errors="coerce").fillna(0)

    if "engagement_rate_followers" not in df.columns:
        engagement = likes + (comments * 2) + (views * 0.1)
        df["engagement_rate_followers"] = (engagement / followers).fillna(0)

    if "views_per_follower" not in df.columns:
        df["views_per_follower"] = (views / followers).fillna(0)

    return df


def _build_business_profiles(posts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds one business profile per business from post-level rows.

    Profile building:
    - engagement metrics are averaged per business
    - follower count is kept as the business size proxy
    - behavioral patterns are converted into rates or averages
    - post formats are converted into simple share-of-posts features
    """

    required_columns = {"business_name", "sector"}
    missing = required_columns - set(posts_df.columns)

    if missing:
        raise ValueError(
            "Missing required column(s): "
            + ", ".join(sorted(missing))
        )

    df = _prepare_post_metrics(posts_df)

    aggregations = {
        "posts_count": ("business_name", "size"),
        "followers_count": ("followers_count", _safe_max),
    }

    for column in _available_columns(df, BASE_KPI_COLUMNS):
        aggregations[f"avg_{column}"] = (column, _safe_mean)

    for column in _available_columns(df, BEHAVIOR_COLUMNS):
        if df[column].dtype == bool or column in {
            "CTA_present",
            "promo_post",
            "mentions_location",
            "arabic_dialect_style",
            "religious_theme",
            "patriotic_theme",
        }:
            aggregations[f"pct_{column}"] = (column, _boolean_rate)
        else:
            aggregations[f"avg_{column}"] = (column, _safe_mean)

    if "post_type" in df.columns:
        aggregations["top_post_type"] = ("post_type", _mode_or_unknown)

    profiles = (
        df
        .groupby(["business_name", "sector"], dropna=False)
        .agg(**aggregations)
        .reset_index()
    )

    if "post_type" in df.columns:
        post_type_rates = (
            pd.crosstab(
                [df["business_name"], df["sector"]],
                df["post_type"],
                normalize="index",
            )
            .reset_index()
        )

        post_type_rates.columns = [
            column if column in {"business_name", "sector"} else f"pct_post_type_{column}"
            for column in post_type_rates.columns
        ]

        profiles = profiles.merge(
            post_type_rates,
            on=["business_name", "sector"],
            how="left",
        )

    for value in POST_TYPE_VALUES:
        column = f"pct_post_type_{value}"
        if column not in profiles.columns:
            profiles[column] = 0.0

    profiles = _add_success_score_breakdown(profiles)

    return profiles


def _minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    min_value = float(values.min())
    max_value = float(values.max())

    if max_value == min_value:
        return pd.Series(0.0, index=series.index)

    return (values - min_value) / (max_value - min_value)


def _add_success_score_breakdown(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the weighted success score and stores each contribution.

    Score calculation:
    each performance metric is min-max normalized across businesses, multiplied
    by its business importance weight, then summed into total success_score.
    """

    profiles = profiles_df.copy()
    contribution_columns = []

    for metric, config in SUCCESS_COMPONENTS.items():
        contribution_column = metric.replace("avg_", "").replace("_count", "")
        contribution_column = f"{contribution_column}_contribution"

        if metric in profiles.columns:
            profiles[contribution_column] = (
                _minmax(profiles[metric]) * float(config["weight"])
            ).round(4)
        else:
            profiles[contribution_column] = 0.0

        contribution_columns.append(contribution_column)

    profiles["success_score"] = profiles[contribution_columns].sum(axis=1).round(4)

    return profiles


def _success_breakdown_from_profile(profile: Dict[str, object]) -> Dict[str, float]:
    breakdown = {}

    for metric, config in SUCCESS_COMPONENTS.items():
        contribution_column = metric.replace("avg_", "").replace("_count", "")
        contribution_column = f"{contribution_column}_contribution"
        breakdown[str(config["label"])] = round(float(profile.get(contribution_column, 0) or 0), 4)

    return breakdown


def _knn_feature_columns(profiles_df: pd.DataFrame) -> List[str]:
    """
    Selects numeric engagement and behavior features for the KNN model.
    """

    candidate_columns = [
        "followers_count",
        "posts_count",
        "avg_engagement_rate_followers",
        "avg_views_per_follower",
        "avg_likes_count",
        "avg_comments_count",
        "avg_shares_count",
        "avg_saves_count",
        "avg_posting_hour",
        "avg_hashtags_count",
        "avg_caption_length",
        "avg_emoji_count",
        "pct_CTA_present",
        "pct_promo_post",
        "pct_mentions_location",
        "pct_arabic_dialect_style",
        "pct_religious_theme",
        "pct_patriotic_theme",
        "pct_post_type_image",
        "pct_post_type_reel",
        "pct_post_type_video",
        "pct_post_type_carousel",
    ]

    return [
        column for column in candidate_columns
        if column in profiles_df.columns
        and pd.api.types.is_numeric_dtype(profiles_df[column])
    ]


def _fit_knn_and_find_neighbors(
    target: pd.Series,
    candidates: pd.DataFrame,
    feature_columns: List[str],
    top_n: int,
) -> pd.DataFrame:
    """
    Fits KNN on successful peer vectors and searches for nearest peers.

    Normalization:
    StandardScaler places engagement, reach, followers, and behavior features
    on a comparable scale before KNN distance is calculated.
    """

    if candidates.empty or not feature_columns:
        return pd.DataFrame()

    # Fit the scaler on target + candidates so the query and peer vectors use
    # the same standardized feature space.
    combined = pd.concat(
        [
            target[feature_columns].to_frame().T,
            candidates[feature_columns],
        ],
        ignore_index=True,
    )
    combined = combined.apply(pd.to_numeric, errors="coerce").fillna(0)

    scaler = StandardScaler()
    scaled_combined = scaler.fit_transform(combined)

    target_vector = scaled_combined[:1]
    candidate_vectors = scaled_combined[1:]

    neighbor_count = min(top_n, len(candidates))
    model = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="euclidean",
    )

    # KNN fitting happens only on successful candidate businesses.
    model.fit(candidate_vectors)

    distances, indices = model.kneighbors(target_vector)

    nearest = candidates.iloc[indices[0]].copy().reset_index(drop=True)
    nearest["knn_distance"] = distances[0].round(4)
    nearest["similarity_score"] = (1 / (1 + nearest["knn_distance"])).round(4)

    return nearest.sort_values(
        by=["similarity_score", "success_score"],
        ascending=False,
    ).reset_index(drop=True)


def _explain_peer_advantage(
    target: pd.Series,
    peer: pd.Series,
) -> str:
    reasons = []

    comparisons = [
        ("avg_engagement_rate_followers", "higher engagement rate", 0.10),
        ("avg_views_per_follower", "stronger reach per follower", 0.10),
        ("avg_shares_count", "more shares", 0.10),
        ("avg_saves_count", "more saves", 0.10),
        ("avg_comments_count", "more comments", 0.10),
        ("pct_CTA_present", "clearer CTA usage", 0.15),
        ("pct_mentions_location", "more location usage", 0.15),
    ]

    for column, label, threshold in comparisons:
        if column not in peer.index:
            continue

        target_value = float(target.get(column, 0) or 0)
        peer_value = float(peer.get(column, 0) or 0)

        if peer_value > target_value * (1 + threshold) and peer_value > 0:
            reasons.append(label)

    if not reasons:
        return "Similar profile, but stronger overall success score."

    return ", ".join(reasons[:4])


def find_similar_successful_businesses(
    posts_df: pd.DataFrame,
    business_name: str,
    sector: Optional[str] = None,
    top_n: int = 5,
    min_success_percentile: float = 0.60,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Finds the nearest similar successful businesses using KNN.

    Similarity search:
    - build business-level vectors from engagement and behavior features
    - keep successful peers only
    - normalize vectors with StandardScaler
    - use NearestNeighbors to retrieve the nearest successful businesses
    """

    profiles = _build_business_profiles(posts_df)

    target_filter = profiles["business_name"].astype(str).str.lower() == business_name.lower()

    if sector is not None:
        target_filter &= profiles["sector"].astype(str).str.lower() == sector.lower()

    if not target_filter.any():
        raise ValueError(f"Business was not found: {business_name}")

    target = profiles[target_filter].iloc[0]
    target_sector = str(target["sector"])
    target_score = float(target.get("success_score", 0) or 0)

    same_sector = profiles[
        (profiles["business_name"] != target["business_name"])
        & (profiles["sector"].astype(str) == target_sector)
    ].copy()

    all_other_businesses = profiles[
        profiles["business_name"] != target["business_name"]
    ].copy()

    success_cutoff = float(profiles["success_score"].quantile(min_success_percentile))

    candidates = same_sector[
        (same_sector["success_score"] >= success_cutoff)
        | (same_sector["success_score"] > target_score)
    ].copy()

    if candidates.empty:
        candidates = all_other_businesses[
            (all_other_businesses["success_score"] >= success_cutoff)
            | (all_other_businesses["success_score"] > target_score)
        ].copy()

    if candidates.empty:
        candidates = all_other_businesses.copy()

    if candidates.empty:
        return pd.DataFrame(), {
            "ok": False,
            "message": "No peer businesses are available for comparison.",
            "target_business": target.to_dict(),
        }

    feature_columns = _knn_feature_columns(profiles)
    peers = _fit_knn_and_find_neighbors(
        target=target,
        candidates=candidates,
        feature_columns=feature_columns,
        top_n=top_n,
    )

    if peers.empty:
        return pd.DataFrame(), {
            "ok": False,
            "message": "Could not build KNN peer vectors.",
            "target_business": target.to_dict(),
        }

    peers["score_gap"] = (peers["success_score"] - target_score).round(4)
    peers["outperformance_reason"] = peers.apply(
        lambda row: _explain_peer_advantage(target, row),
        axis=1,
    )

    metadata = {
        "ok": True,
        "target_business": target.to_dict(),
        "peer_count": int(len(peers)),
        "success_cutoff": round(success_cutoff, 4),
        "knn_feature_columns": feature_columns,
        "nearest_peers": peers.to_dict("records"),
        "target_success_breakdown": _success_breakdown_from_profile(target.to_dict()),
        "performance_explanation": _build_performance_explanation(target, peers),
    }

    return peers.reset_index(drop=True), metadata


def _priority_label(score: float) -> str:
    if score >= 0.70:
        return "High"
    if score >= 0.42:
        return "Medium"
    return "Low"


def _calculate_priority_score(
    target_value: float,
    peer_value: float,
    similarity_score: float,
    feature_importance: float,
    reverse_gap: bool = False,
) -> float:
    """
    Calculates recommendation priority from gap, similarity, and importance.
    """

    if reverse_gap:
        denominator = max(abs(peer_value), 0.01)
        gap = max((target_value - peer_value) / denominator, 0)
    else:
        denominator = max(abs(peer_value), 0.01)
        gap = max((peer_value - target_value) / denominator, 0)

    priority_score = (
        min(gap, 1.0) * 0.50
        + similarity_score * 0.30
        + feature_importance * 0.20
    )

    return round(float(priority_score), 4)


def _build_performance_explanation(
    target: pd.Series,
    peers: pd.DataFrame,
) -> List[str]:
    """
    Explains why the selected business is weaker than its nearest peers.
    """

    explanations = []

    checks = [
        (
            "avg_engagement_rate_followers",
            "Lower engagement rate",
            "posts are creating less interaction per follower",
            False,
        ),
        (
            "avg_views_per_follower",
            "Lower views per follower",
            "content is reaching fewer people relative to audience size",
            False,
        ),
        (
            "pct_mentions_location",
            "Less location usage",
            "posts may be missing local discovery signals",
            False,
        ),
        (
            "pct_CTA_present",
            "Less CTA usage",
            "audiences are being asked to act less often",
            False,
        ),
        (
            "avg_shares_count",
            "Lower shares",
            "content is being passed along less often",
            False,
        ),
        (
            "avg_saves_count",
            "Lower saves",
            "content may be less useful or less worth revisiting",
            False,
        ),
    ]

    for column, title, description, reverse_gap in checks:
        if column not in peers.columns:
            continue

        target_value = float(target.get(column, 0) or 0)
        peer_value = float(peers[column].mean())

        if reverse_gap:
            weaker = target_value > peer_value * 1.15
        else:
            weaker = peer_value > target_value * 1.15 and peer_value > 0

        if weaker:
            explanations.append(
                f"{title}: {_format_number(target_value)} vs peer average "
                f"{_format_number(peer_value)}; {description}."
            )

    if not explanations:
        explanations.append(
            "The selected business is close to its nearest peers, but its total "
            "success score is still lower because peers perform better across the "
            "combined weighted metrics."
        )

    return explanations


def generate_similar_business_recommendations(
    posts_df: pd.DataFrame,
    business_name: str,
    sector: Optional[str] = None,
    top_n: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Generates recommendations from the nearest successful businesses found by KNN.
    """

    peers, metadata = find_similar_successful_businesses(
        posts_df=posts_df,
        business_name=business_name,
        sector=sector,
        top_n=top_n,
    )

    if peers.empty:
        return pd.DataFrame(), metadata

    target = metadata["target_business"]
    peer_similarity = float(peers["similarity_score"].mean())
    comparison_names = ", ".join(peers["business_name"].astype(str).head(3))
    recommendations = []

    def add_recommendation(
        recommendation: str,
        reason: str,
        target_value: object,
        peer_value: object,
        evidence_metric: str,
        priority_score: float,
    ) -> None:
        recommendations.append(
            {
                "business_name": target["business_name"],
                "sector": target["sector"],
                "recommendation": recommendation,
                "reason": reason,
                "your_value": target_value,
                "successful_peer_value": peer_value,
                "evidence_metric": evidence_metric,
                "priority_score": priority_score,
                "priority": _priority_label(priority_score),
                "comparison_businesses": comparison_names,
            }
        )

    # Recommendation generation uses only the nearest KNN peers, not global
    # averages, so every suggestion is anchored in businesses that look similar.
    metric_recommendations = [
        (
            "avg_engagement_rate_followers",
            "Improve content hooks and interaction prompts",
            "Nearest successful peers generate stronger engagement per follower. Review their strongest post types, openings, and audience prompts.",
            False,
        ),
        (
            "avg_views_per_follower",
            "Increase reach with more discoverable formats",
            "Nearest successful peers reach more viewers per follower. Test more shareable topics, stronger first-line captions, and formats with broader discovery.",
            False,
        ),
        (
            "avg_shares_count",
            "Create posts people are likely to share",
            "Nearest successful peers receive more shares. Add useful tips, local relevance, or offer-led content that customers would send to others.",
            False,
        ),
        (
            "avg_saves_count",
            "Publish more save-worthy content",
            "Nearest successful peers earn more saves. Add checklists, before-and-after examples, price guidance, or practical advice.",
            False,
        ),
        (
            "pct_CTA_present",
            "Use clearer calls-to-action more often",
            "Nearest successful peers ask the audience to act more consistently. Use simple CTAs such as booking, messaging, commenting, or visiting.",
            False,
        ),
        (
            "pct_mentions_location",
            "Add location signals to more posts",
            "Nearest successful peers mention locations more often, which can improve local relevance and customer recognition.",
            False,
        ),
        (
            "pct_arabic_dialect_style",
            "Use a more conversational local tone",
            "Nearest successful peers use local Arabic dialect more often, making their content feel more familiar and community-focused.",
            False,
        ),
        (
            "pct_promo_post",
            "Reduce sales-heavy posting and add value content",
            "Nearest successful peers rely less on promotional posts. Balance offers with education, proof, stories, and customer-focused posts.",
            True,
        ),
    ]

    for column, recommendation, reason, reverse_gap in metric_recommendations:
        if column not in peers.columns:
            continue

        peer_value = float(peers[column].mean())
        target_value = float(target.get(column, 0) or 0)

        if reverse_gap:
            needs_change = target_value > peer_value + 0.15
        else:
            needs_change = peer_value > target_value * 1.15 and peer_value > 0

        if not needs_change:
            continue

        priority_score = _calculate_priority_score(
            target_value=target_value,
            peer_value=peer_value,
            similarity_score=peer_similarity,
            feature_importance=FEATURE_IMPORTANCE.get(column, 0.35),
            reverse_gap=reverse_gap,
        )

        add_recommendation(
            recommendation=recommendation,
            reason=reason,
            target_value=round(target_value, 4),
            peer_value=round(peer_value, 4),
            evidence_metric=column,
            priority_score=priority_score,
        )

    if "top_post_type" in peers.columns:
        peer_top_post_type = _mode_or_unknown(peers["top_post_type"])
        target_top_post_type = str(target.get("top_post_type", "Unknown"))

        if peer_top_post_type != "Unknown" and peer_top_post_type != target_top_post_type:
            priority_score = round(
                peer_similarity * 0.45
                + FEATURE_IMPORTANCE["top_post_type"] * 0.35
                + 0.20,
                4,
            )
            add_recommendation(
                recommendation=f"Test more {peer_top_post_type} posts",
                reason="Nearest successful peers use a different dominant post format. Run a small test before changing the full content mix.",
                target_value=target_top_post_type,
                peer_value=peer_top_post_type,
                evidence_metric="top_post_type",
                priority_score=priority_score,
            )

    if "avg_posting_hour" in peers.columns:
        peer_hour = round(float(peers["avg_posting_hour"].mean()))
        target_hour = round(float(target.get("avg_posting_hour", 0) or 0))

        if abs(peer_hour - target_hour) >= 3:
            priority_score = round(
                0.35
                + peer_similarity * 0.35
                + FEATURE_IMPORTANCE["avg_posting_hour"] * 0.30,
                4,
            )
            add_recommendation(
                recommendation="Test posting closer to peer timing",
                reason="Nearest successful peers post at meaningfully different hours. Treat this as an experiment, not a permanent rule.",
                target_value=target_hour,
                peer_value=peer_hour,
                evidence_metric="avg_posting_hour",
                priority_score=priority_score,
            )

    recommendations_df = pd.DataFrame(recommendations)

    if recommendations_df.empty:
        recommendations_df = pd.DataFrame(
            [
                {
                    "business_name": target["business_name"],
                    "sector": target["sector"],
                    "recommendation": "Keep monitoring nearest successful peers",
                    "reason": "KNN found similar successful businesses, but no major gap crossed the recommendation threshold. Focus on small tests and consistency.",
                    "your_value": "Similar",
                    "successful_peer_value": "Similar",
                    "evidence_metric": "overall_profile",
                    "priority_score": round(peer_similarity * 0.50, 4),
                    "priority": "Low",
                    "comparison_businesses": comparison_names,
                }
            ]
        )

    recommendations_df = (
        recommendations_df
        .sort_values(by=["priority_score"], ascending=False)
        .reset_index(drop=True)
    )

    metadata["recommendation_count"] = int(len(recommendations_df))

    return recommendations_df, metadata


def display_similar_business_report(
    recommendations_df: pd.DataFrame,
    metadata: Dict[str, object],
) -> None:
    """
    Prints a professional terminal report for notebook or script usage.
    """

    print("\n" + "=" * 92)
    print("SIMILAR BUSINESS RECOMMENDER - KNN PEER BENCHMARK REPORT")
    print("=" * 92)

    if not metadata.get("ok", False):
        print(metadata.get("message", "Could not generate report."))
        return

    target = metadata["target_business"]
    target_score = float(target.get("success_score", 0) or 0)

    print("\nSelected Business")
    print("-" * 92)
    print(f"Business name       : {target['business_name']}")
    print(f"Sector              : {target['sector']}")
    print(f"Followers           : {_format_number(target.get('followers_count', 0), 0)}")
    print(f"Posts analyzed      : {_format_number(target.get('posts_count', 0), 0)}")
    print(f"Total success score : {_format_number(target_score, 4)}")

    print("\nBusiness Success Score Breakdown")
    print("-" * 92)
    for label, value in metadata["target_success_breakdown"].items():
        print(f"* {label}: {_format_number(value, 4)}")
    print(f"* Total Success Score: {_format_number(target_score, 4)}")

    print("\nWhy This Business Is Weaker Than Similar Successful Peers")
    print("-" * 92)
    for explanation in metadata["performance_explanation"]:
        print(f"* {explanation}")

    print("\nNearest Similar Successful Businesses")
    print("-" * 92)
    peers = metadata.get("nearest_peers")

    if peers is None:
        print("Peer details unavailable.")
    else:
        for index, peer in enumerate(peers, start=1):
            print(
                f"{index}. {peer['business_name']} | "
                f"Sector: {peer['sector']} | "
                f"Similarity: {_format_number(peer['similarity_score'], 4)} | "
                f"Success: {_format_number(peer['success_score'], 4)}"
            )
            print(f"   Why it outperforms: {peer['outperformance_reason']}")

    print("\nActionable Recommendations")
    print("-" * 92)

    for index, row in recommendations_df.iterrows():
        print(f"\n{index + 1}. {row['recommendation']}")
        print(f"   Priority          : {row['priority']} ({_format_number(row['priority_score'], 4)})")
        print(f"   Business reason   : {row['reason']}")
        print(f"   Your value        : {row['your_value']}")
        print(f"   Peer benchmark    : {row['successful_peer_value']}")
        print(f"   Evidence metric   : {row['evidence_metric']}")
        print(f"   Compared against  : {row['comparison_businesses']}")


def main() -> None:
    """
    Example run using the sample dataset.
    """

    base_dir = Path(__file__).resolve().parents[2]
    data_path = base_dir / "data" / "sample_synthetic_posts.csv"

    posts_df = pd.read_csv(data_path)
    example_business = str(posts_df["business_name"].dropna().iloc[0])

    recommendations_df, metadata = generate_similar_business_recommendations(
        posts_df=posts_df,
        business_name=example_business,
        top_n=5,
    )

    display_similar_business_report(
        recommendations_df=recommendations_df,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
