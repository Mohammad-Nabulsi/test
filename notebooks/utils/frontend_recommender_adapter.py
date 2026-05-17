"""
frontend_recommender_adapter.py

Purpose
-------
Provides three independent orchestration pipelines for the backend API.

Input
-----
- Processed multi-business dataset for rule generation
- Full multi-business dataset for KNN benchmarking
- Single-business dataset for next-post recommendations

Output
------
- Frontend-safe structured JSON dictionaries

Pipeline Role
-------------
Connects API endpoints to existing recommendation modules without placing
algorithm logic inside the API layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

if __package__:
    from .llm_benchmark_dashboard_formatter import generate_benchmark_dashboard_texts
    from .llm_recommendation_formatter import rewrite_recommendation
    from .generate_rules import (
        generate_all_rules,
        load_saved_rules,
    )
    from .recommendation_system import generate_recommendations, summarize_business_behavior
    from .similar_business_recommender import (
        build_business_feature_profiles,
        build_knn_comparison_dataset,
        extract_target_business_identity,
        generate_similar_business_recommendations,
        load_master_comparison_dataset,
    )
else:
    from llm_benchmark_dashboard_formatter import generate_benchmark_dashboard_texts
    from llm_recommendation_formatter import rewrite_recommendation
    from generate_rules import generate_all_rules, load_saved_rules
    from recommendation_system import generate_recommendations, summarize_business_behavior
    from similar_business_recommender import (
        build_business_feature_profiles,
        build_knn_comparison_dataset,
        extract_target_business_identity,
        generate_similar_business_recommendations,
        load_master_comparison_dataset,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NEXT_POST_REQUIRED_COLUMNS: List[str] = [
    "business_name",
    "sector",
    "followers_count",
    "post_type",
    "caption_length",
    "hashtags_count",
    "emoji_count",
    "likes_count",
    "comments_count",
    "views_count",
    "language",
    "CTA_present",
    "promo_post",
    "mentions_location",
    "religious_theme",
    "patriotic_theme",
    "arabic_dialect_style",
]

SIMILAR_BUSINESS_REQUIRED_COLUMNS: List[str] = [
    "business_name",
    "sector",
    "followers_count",
    "likes_count",
    "comments_count",
    "views_count",
    "post_type",
    "hashtags_count",
    "caption_length",
    "emoji_count",
    "CTA_present",
    "promo_post",
    "mentions_location",
]

OPTIONAL_COLUMNS: List[str] = [
    "day_of_week",
    "posting_hour",
    "month",
    "discount_percent",
]


# Path step: Resolve direct, project-relative, or data/ JSON paths.
def resolve_json_path(json_path: Union[str, Path]) -> Path:
    """Resolve a JSON path to an existing file."""
    requested_path = Path(str(json_path).strip())
    candidates = [
        requested_path,
        PROJECT_ROOT / requested_path,
        PROJECT_ROOT / "data" / requested_path,
    ]

    for candidate in candidates:
        resolved_candidate = candidate.expanduser().resolve()
        if resolved_candidate.exists() and resolved_candidate.is_file():
            return resolved_candidate

    raise FileNotFoundError(f"JSON file was not found: {json_path}")


# Load step: Read any API dataset into a DataFrame.
def load_business_json(json_path: Union[str, Path]) -> pd.DataFrame:
    """Load a JSON dataset with pandas."""
    resolved_path = resolve_json_path(json_path)

    try:
        posts_df = pd.read_json(resolved_path)
    except ValueError as error:
        raise ValueError(f"Invalid JSON dataset: {resolved_path}") from error

    if posts_df.empty:
        raise ValueError("Dataset is empty.")

    return posts_df


# Validation step: Check a dataset has the columns required by one pipeline.
def validate_required_columns(posts_df: pd.DataFrame, required_columns: List[str]) -> None:
    """Validate that a dataset contains required columns."""
    missing_columns = [
        column for column in required_columns if column not in posts_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required input columns: "
            + ", ".join(sorted(missing_columns))
        )


# Cleanup step: Fill optional columns used by rule and summary logic.
def fill_missing_optional_columns(posts_df: pd.DataFrame) -> pd.DataFrame:
    """Add safe defaults for optional fields."""
    normalized_df = posts_df.copy()

    if "day_of_week" not in normalized_df.columns:
        normalized_df["day_of_week"] = "Unknown"

    if "month" not in normalized_df.columns:
        normalized_df["month"] = "Unknown"

    if "posting_hour" not in normalized_df.columns:
        normalized_df["posting_hour"] = pd.NA

    if "discount_percent" not in normalized_df.columns:
        normalized_df["discount_percent"] = 0.0

    return normalized_df


# Summary step: Build behavior summary for next-post responses.
def build_business_behavior_summary(posts_df: pd.DataFrame) -> Dict[str, Any]:
    """Summarize uploaded business behavior."""
    return summarize_business_behavior(posts_df.copy())


# Formatting step: Keep next-post recommendation output frontend-safe.
def clamp_percentage(value: float, minimum: int, maximum: int) -> int:
    """Keep frontend percentage values in a realistic display range."""
    return int(round(max(minimum, min(maximum, value))))


def infer_recommendation_category(item: Dict[str, Any]) -> str:
    """Infer a frontend category without changing recommendation text."""
    evidence_metric = str(item.get("evidence_metric", "")).lower()
    if evidence_metric in {"pct_cta_present"}:
        return "engagement"
    if evidence_metric in {"pct_mentions_location"}:
        return "location"
    if evidence_metric in {"avg_hashtags_count"}:
        return "hashtags"
    if evidence_metric in {"top_post_type", "pct_post_type_reel", "pct_post_type_post"}:
        return "reels"
    if evidence_metric in {"posting_hour", "posting_time", "day_of_week"}:
        return "posting_time"

    signal = " ".join(
        str(item.get(key, ""))
        for key in [
            "title",
            "explanation",
            "recommendation",
            "reason",
            "evidence",
            "feature",
            "evidence_metric",
        ]
    ).lower()

    if any(term in signal for term in ["cta", "call-to-action", "call to action", "calls to action", "order", "message", "book", "action", "دعوة", "دعوة الفعل", "فعل", "تفاعل", "اطلب", "راسل", "احجز", "تعليق", "كومنت"]):
        return "engagement"
    if any(term in signal for term in ["posting_time", "posting_hour", "day_of_week", "hour", "time", "وقت", "توقيت", "ساعة", "الصباح", "المساء", "نشر"]):
        return "posting_time"
    if any(term in signal for term in ["hashtag", "hashtags", "هاشتاغ", "هاشتاغات"]):
        return "hashtags"
    if any(term in signal for term in ["reel", "reels", "video", "post_type", "ريل", "ريلز", "فيديو"]):
        return "reels"
    if any(term in signal for term in ["location", "mentions_location", "map", "موقع", "الموقع", "منطقة", "محلي", "محلية", "رام الله"]):
        return "location"
    if any(term in signal for term in ["caption", "language", "arabic", "dialect", "كابشن", "عربي", "لهجة", "نص", "كتابة"]):
        return "content"
    return "content"


def infer_benchmark_category(item: Dict[str, Any]) -> str:
    """Infer category names used by the similar-business dashboard."""
    category = infer_recommendation_category(item)
    return category


def infer_action_type(category: str, item: Dict[str, Any]) -> str:
    """Infer the frontend action type from category and recommendation signal."""
    signal = " ".join(
        str(item.get(key, ""))
        for key in ["title", "explanation", "evidence", "feature"]
    ).lower()

    if category == "location":
        return "improve_local_visibility"
    if any(term in signal for term in ["cta", "promo", "discount", "conversion", "order", "message"]):
        return "improve_conversion"
    if category in {"hashtags", "posting_time", "reels"}:
        return "increase_reach"
    return "increase_engagement"


def icon_for_category(category: str) -> str:
    """Map frontend categories to icon names."""
    return {
        "content": "text",
        "hashtags": "hashtag",
        "posting_time": "clock",
        "reels": "video",
        "engagement": "sparkles",
        "location": "map-pin",
        "captions": "text",
    }.get(category, "sparkles")


def calculate_expected_impact(priority_score: float) -> int:
    """Estimate impact from recommendation priority for card display."""
    return clamp_percentage(6 + (priority_score * 0.26), 8, 38)


def calculate_confidence_score(priority_score: float, reliability: float = 1.0) -> int:
    """Estimate confidence from priority and dataset reliability."""
    return clamp_percentage(62 + (priority_score * 0.35 * reliability), 70, 95)


def normalize_benchmark_priority(priority_score: float) -> float:
    """Convert benchmark priority values to a 0-100 frontend scale."""
    if priority_score <= 1:
        return priority_score * 100
    return min(priority_score, 100)


def benchmark_importance_bonus(category: str) -> float:
    """Return a deterministic impact bonus by recommendation type."""
    return {
        "engagement": 5.0,
        "location": 4.0,
        "reels": 4.0,
        "hashtags": 3.0,
        "posting_time": 3.0,
        "content": 2.0,
    }.get(category, 2.0)


def calculate_benchmark_expected_impact(
    priority_score: float,
    category: str,
    peer_count: int,
) -> int:
    """Estimate card impact from benchmark strength, category, and peer support."""
    peer_bonus = min(peer_count, 5) * 1.4
    raw_impact = 8 + (priority_score * 0.48) + benchmark_importance_bonus(category) + peer_bonus
    return clamp_percentage(raw_impact, 12, 45)


def calculate_benchmark_confidence_score(
    priority_score: float,
    peer_count: int,
    inspired_by_count: int,
) -> int:
    """Estimate confidence from recommendation strength and peer consistency."""
    peer_consistency = min(peer_count, 5) * 2.2
    inspiration_bonus = min(inspired_by_count, 3) * 1.5
    raw_confidence = 76 + (priority_score * 0.16) + peer_consistency + inspiration_bonus
    return clamp_percentage(raw_confidence, 80, 95)


def format_recommendation_item(
    item: Dict[str, Any],
    recommendation_id: int,
    reliability: float = 1.0,
) -> Dict[str, Any]:
    """Return only display-ready recommendation fields."""
    priority_score = float(item.get("priority_score", 0) or 0)
    category = infer_recommendation_category(item)

    return {
        "id": recommendation_id,
        "type": str(item.get("type", "recommendation")),
        "category": category,
        "title": str(item.get("title", item.get("recommendation", "Recommendation"))),
        "explanation": str(item.get("explanation", item.get("reason", ""))),
        "priority_score": priority_score,
        "expected_impact": calculate_expected_impact(priority_score),
        "confidence_score": calculate_confidence_score(priority_score, reliability),
        "action_type": infer_action_type(category, item),
        "icon": icon_for_category(category),
    }


# Formatting step: Keep benchmark recommendation output simple and non-technical.
def format_benchmark_recommendation_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Return only user-facing benchmark recommendation fields."""
    return {
        "title": str(item.get("title", item.get("recommendation", "Recommendation"))),
        "explanation": str(item.get("explanation", item.get("reason", ""))),
        "priority_score": float(item.get("priority_score", 0) or 0),
    }


def parse_comparison_businesses(value: Any) -> List[str]:
    """Convert comparison business text/list into clean display names."""
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [
        item.strip()
        for item in str(value or "").split(",")
        if item.strip()
    ]


def format_recommendation_card(
    item: Dict[str, Any],
    recommendation_id: int,
    default_peer_names: List[str],
    peer_count: int,
) -> Dict[str, Any]:
    """Convert a benchmark recommendation into a frontend card object."""
    raw_priority = float(item.get("priority_score", 0) or 0)
    priority_score = normalize_benchmark_priority(raw_priority)
    category = infer_benchmark_category(item)
    inspired_by = parse_comparison_businesses(item.get("comparison_businesses"))
    if not inspired_by:
        inspired_by = default_peer_names[:3]

    return {
        "id": recommendation_id,
        "title": str(item.get("title", item.get("recommendation", "Recommendation"))),
        "explanation": str(item.get("explanation", item.get("reason", ""))),
        "expected_impact": calculate_benchmark_expected_impact(
            priority_score=priority_score,
            category=category,
            peer_count=peer_count,
        ),
        "confidence_score": calculate_benchmark_confidence_score(
            priority_score=priority_score,
            peer_count=peer_count,
            inspired_by_count=len(inspired_by),
        ),
        "priority_score": round(priority_score, 1),
        "category": category,
        "icon": icon_for_category(category),
        "inspired_by": inspired_by[:3],
    }


def rewrite_benchmark_recommendations(recommendations: Any) -> List[Dict[str, Any]]:
    """Apply the existing GPT formatter to benchmark recommendation text once."""
    if isinstance(recommendations, pd.DataFrame):
        records = recommendations.to_dict("records")
    else:
        records = list(recommendations or [])

    rewritten_records = []
    for record in records:
        rewritten_record = dict(record)
        comparison_context = (
            f"Recommendation: {record.get('title', record.get('recommendation', ''))}\n"
            f"Reason: {record.get('explanation', record.get('reason', ''))}\n"
            f"Evidence metric: {record.get('evidence_metric', '')}\n"
            f"Target value: {record.get('your_value', '')}\n"
            f"Peer benchmark value: {record.get('successful_peer_value', '')}\n"
            f"Compared businesses: {record.get('comparison_businesses', '')}"
        )
        rewritten = rewrite_recommendation(
            title=str(record.get("title", record.get("recommendation", ""))),
            explanation=comparison_context,
            recommendation_type=str(record.get("type", "benchmark")),
        )
        rewritten_record["title"] = rewritten["title"]
        rewritten_record["explanation"] = rewritten["explanation"]
        rewritten_record["recommendation"] = rewritten["title"]
        rewritten_record["reason"] = rewritten["explanation"]
        rewritten_records.append(rewritten_record)

    return rewritten_records


def format_recommendation_cards(
    recommendations: Any,
    metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Convert benchmark recommendations into frontend-ready cards."""
    if isinstance(recommendations, pd.DataFrame):
        records = recommendations.to_dict("records")
    else:
        records = list(recommendations or [])

    default_peer_names = [
        str(peer.get("business_name"))
        for peer in metadata.get("nearest_peers", []) or []
        if peer.get("business_name")
    ]
    peer_count = len(default_peer_names)

    seen_titles = set()
    cards = []
    for record in records:
        card = format_recommendation_card(
            record,
            recommendation_id=len(cards) + 1,
            default_peer_names=default_peer_names,
            peer_count=peer_count,
        )
        if card["title"] in seen_titles:
            continue
        seen_titles.add(card["title"])
        cards.append(card)

    return cards


# Formatting step: Normalize list/DataFrame recommendation outputs.
def format_recommendation_list(
    recommendations: Any,
    reliability: float = 1.0,
) -> List[Dict[str, Any]]:
    """Convert recommendation outputs to clean dictionaries."""
    if isinstance(recommendations, pd.DataFrame):
        records = recommendations.to_dict("records")
    else:
        records = list(recommendations or [])

    seen_titles = set()
    clean_items = []
    for record in records:
        clean_item = format_recommendation_item(
            record,
            recommendation_id=len(clean_items) + 1,
            reliability=reliability,
        )
        if clean_item["title"] in seen_titles:
            continue
        seen_titles.add(clean_item["title"])
        clean_items.append(clean_item)

    return clean_items


def build_recommendation_response_summary(
    recommendations: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Build frontend dashboard summary values from returned recommendations."""
    total_recommendations = len(recommendations)
    remaining_no_impact = 1.0

    for recommendation in recommendations:
        impact = float(recommendation.get("expected_impact", 0) or 0) / 100
        remaining_no_impact *= 1 - impact

    estimated_total_impact = clamp_percentage(
        (1 - remaining_no_impact) * 100,
        0,
        95,
    )

    return {
        "total_recommendations": total_recommendations,
        "estimated_total_impact": estimated_total_impact,
    }


def build_similar_business_hero_summary(
    recommendation_cards: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the hero summary for the similar-business dashboard."""
    if recommendation_cards:
        top_card = max(
            recommendation_cards,
            key=lambda item: float(item.get("priority_score", 0) or 0),
        )
        top_opportunity = top_card.get("title", "جربي تحسينات صغيرة على المحتوى")
        estimated_best_impact = int(top_card.get("expected_impact", 0) or 0)
    else:
        top_opportunity = "جربي تحسينات صغيرة على المحتوى"
        estimated_best_impact = 0

    return {
        "title": "خطوات مستوحاة من مشاريع قريبة من أسلوبك ✨",
        "total_recommendations": len(recommendation_cards),
        "top_opportunity": top_opportunity,
        "estimated_best_impact": estimated_best_impact,
    }


# Formatting step: Normalize benchmark recommendation outputs.
def format_benchmark_recommendation_list(recommendations: Any) -> List[Dict[str, Any]]:
    """Convert KNN recommendations to frontend-safe dictionaries."""
    if isinstance(recommendations, pd.DataFrame):
        records = recommendations.to_dict("records")
    else:
        records = list(recommendations or [])

    seen_titles = set()
    clean_items = []
    for record in records:
        clean_item = format_benchmark_recommendation_item(record)
        if clean_item["title"] in seen_titles:
            continue
        seen_titles.add(clean_item["title"])
        clean_items.append(clean_item)

    return clean_items


# Formatting step: Keep similar business peer output readable and safe.
def scale_peer_distribution(
    values: List[float],
    high: int = 94,
    low: int = 82,
) -> List[int]:
    """Scale peer values into a frontend-friendly relative percentage range."""
    if not values:
        return []

    max_value = max(values)
    min_value = min(values)
    if max_value == min_value:
        return [
            clamp_percentage(high - (index * 3), low, high)
            for index, _value in enumerate(values)
        ]

    scaled_values = []
    for value in values:
        relative_position = (value - min_value) / (max_value - min_value)
        scaled_values.append(
            clamp_percentage(low + (relative_position * (high - low)), low, high)
        )

    return scaled_values


def scale_peer_success_scores(peers: List[Dict[str, Any]]) -> List[int]:
    """Scale peer success scores into dashboard-friendly values."""
    raw_scores = [float(peer.get("success_score", 0) or 0) for peer in peers]
    return scale_peer_distribution(raw_scores, high=95, low=60)


def build_peer_tags(peer: Dict[str, Any]) -> List[str]:
    """Build short Arabic behavior tags from peer profile values."""
    tags = []

    if str(peer.get("top_post_type", "")).lower() == "reel" or float(peer.get("pct_post_type_reel", 0) or 0) >= 0.5:
        tags.append("ريلز")
    else:
        tags.append("بوستات")

    if float(peer.get("pct_CTA_present", 0) or 0) >= 0.45:
        tags.append("CTA")
    if float(peer.get("pct_mentions_location", 0) or 0) >= 0.35:
        tags.append("تفاعل محلي")
    if float(peer.get("pct_promo_post", 0) or 0) >= 0.35:
        tags.append("محتوى عروض")
    if float(peer.get("avg_hashtags_count", 0) or 0) >= 5:
        tags.append("هاشتاغات")
    if float(peer.get("pct_arabic_dialect_style", 0) or 0) >= 0.35:
        tags.append("كابشن عربي")

    return tags[:4]


def build_peer_quick_insight(peer: Dict[str, Any]) -> str:
    """Create a short peer advantage insight for card previews."""
    insights = []
    if float(peer.get("pct_CTA_present", 0) or 0) >= 0.45:
        insights.append("CTA أوضح")
    if float(peer.get("pct_mentions_location", 0) or 0) >= 0.35:
        insights.append("مينشن للموقع أكثر")
    if float(peer.get("avg_engagement_rate_followers", 0) or 0) > 0:
        insights.append("تفاعل أقوى")
    if float(peer.get("avg_views_per_follower", 0) or 0) > 0:
        insights.append("وصول أفضل")

    if not insights:
        return str(peer.get("outperformance_reason", "أسلوب محتوى قريب ومفيد للمقارنة"))
    return " و".join(insights[:2])


def format_similar_businesses(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract frontend-safe peer details from KNN metadata."""
    peers = metadata.get("nearest_peers", []) or []
    formatted_peers = []
    similarity_percentages = scale_peer_distribution(
        [float(peer.get("similarity_score", 0) or 0) for peer in peers],
        high=94,
        low=84,
    )
    success_scores = scale_peer_success_scores(peers)

    for index, peer in enumerate(peers):
        formatted_peers.append(
            {
                "business_name": peer.get("business_name"),
                "similarity_score": float(peer.get("similarity_score", 0) or 0),
                "similarity_percentage": similarity_percentages[index],
                "success_score": success_scores[index],
                "quick_insight": build_peer_quick_insight(peer),
                "explanation": peer.get("outperformance_reason", ""),
                "tags": build_peer_tags(peer),
            }
        )

    return formatted_peers


def build_insight_highlights(
    similar_businesses: List[Dict[str, Any]],
    recommendation_cards: List[Dict[str, Any]],
) -> List[str]:
    """Build short dashboard insights from repeated peer/card signals."""
    highlights = []

    category_counts: Dict[str, int] = {}
    for card in recommendation_cards:
        category = str(card.get("category", "content"))
        category_counts[category] = category_counts.get(category, 0) + 1

    peer_text = " ".join(
        " ".join(str(tag) for tag in business.get("tags", []))
        + " "
        + str(business.get("quick_insight", ""))
        for business in similar_businesses
    )

    if category_counts.get("CTA", 0) or "CTA" in peer_text:
        highlights.append("CTA مباشر بكل ريل")
    if category_counts.get("location", 0) or "الموقع" in peer_text or "محلي" in peer_text:
        highlights.append("ذكر أوضح للموقع")
    if category_counts.get("reels", 0) or "ريلز" in peer_text:
        highlights.append("ريلز أقرب للجمهور")
    if category_counts.get("hashtags", 0) or "هاشتاغات" in peer_text:
        highlights.append("هاشتاغات محلية مخصصة")
    if category_counts.get("content", 0) or category_counts.get("captions", 0):
        highlights.append("تنويع أكبر بالمحتوى")

    if not highlights and recommendation_cards:
        highlights.append("فرص أسرع للتفاعل")

    return highlights[:4]


# Formatting step: Keep target business metadata stable for API callers.
def format_target_business(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the stable target business identity fields from KNN metadata."""
    target_business = metadata.get("target_business", {}) or {}
    return {
        "business_name": target_business.get("business_name"),
        "sector": target_business.get("sector"),
    }


def safe_float(value: Any) -> float:
    """Convert numeric dashboard values safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def scale_series_to_range(
    values: pd.Series,
    low: float,
    high: float,
) -> pd.Series:
    """Scale a numeric series into a frontend presentation range."""
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    minimum = float(numeric.min())
    maximum = float(numeric.max())

    if maximum == minimum:
        midpoint = low + ((high - low) * 0.62)
        return pd.Series(midpoint, index=values.index)

    return low + ((numeric - minimum) / (maximum - minimum) * (high - low))


def scale_value_against_top(value: float, top_value: float) -> float:
    """Scale one KPI to a 0-10 chart value using sector top as the anchor."""
    if top_value <= 0:
        return 0.0
    return round(max(0.0, min(10.0, (value / top_value) * 10)), 1)


def normalize_kpi_series_for_dashboard(values: pd.Series) -> pd.Series:
    """Normalize a KPI into a balanced 0-10 frontend scale within its sector."""
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)

    if numeric.empty:
        return pd.Series(dtype=float)

    if float(numeric.max()) == float(numeric.min()):
        return pd.Series(6.0, index=values.index)

    minmax = (numeric - numeric.min()) / (numeric.max() - numeric.min())
    percentile = numeric.rank(method="average", pct=True)

    # Percentile keeps tiny-rate metrics visually meaningful, while min/max
    # preserves distance from the top performer.
    blended = (percentile * 0.65) + (minmax * 0.35)
    return (1.8 + (blended * 7.6)).clip(0, 10).round(1)


def adjust_dashboard_kpi_value(
    metric_key: str,
    value: float,
    target_profile: pd.Series,
    sector_profiles: pd.DataFrame,
) -> float:
    """Apply tiny deterministic presentation nudges for closely related KPIs."""
    if metric_key == "engagement" and "avg_comments_count" in sector_profiles.columns:
        comments = normalize_kpi_series_for_dashboard(sector_profiles["avg_comments_count"])
        target_name = str(target_profile.get("business_name", "")).lower()
        target_mask = (
            sector_profiles["business_name"].astype(str).str.lower()
            == target_name
        )
        if target_mask.any():
            comment_signal = float(comments[target_mask].iloc[0]) - float(comments.mean())
            return round(max(0.0, min(10.0, value + (comment_signal * 0.08))), 1)

    if metric_key == "reach" and "pct_post_type_reel" in sector_profiles.columns:
        reels = normalize_kpi_series_for_dashboard(sector_profiles["pct_post_type_reel"])
        target_name = str(target_profile.get("business_name", "")).lower()
        target_mask = (
            sector_profiles["business_name"].astype(str).str.lower()
            == target_name
        )
        if target_mask.any():
            reel_signal = float(reels[target_mask].iloc[0]) - float(reels.mean())
            return round(max(0.0, min(10.0, value + (reel_signal * 0.08))), 1)

    return value


def numeric_profile_column(profiles: pd.DataFrame, column: str) -> pd.Series:
    """Read a numeric profile column, or return zeros if it is unavailable."""
    if column not in profiles.columns:
        return pd.Series(0.0, index=profiles.index)
    return pd.to_numeric(profiles[column], errors="coerce").fillna(0.0)


def add_dashboard_quality_metric(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """Add a lightweight dashboard-only quality metric from existing profile fields."""
    profiles = profiles_df.copy()
    comments = numeric_profile_column(profiles, "avg_comments_count")
    cta = numeric_profile_column(profiles, "pct_CTA_present")
    location = numeric_profile_column(profiles, "pct_mentions_location")
    local_tone = numeric_profile_column(profiles, "pct_arabic_dialect_style")

    normalized_comments = scale_series_to_range(comments, 0, 1)
    profiles["dashboard_quality_score"] = (
        normalized_comments * 0.40
        + cta * 0.25
        + location * 0.20
        + local_tone * 0.15
    )

    return profiles


def build_dashboard_profiles(target_business_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Build business profiles for the uploaded business and its sector peers."""
    target_business = extract_target_business_identity(target_business_df)
    comparison_df = load_master_comparison_dataset()
    posts_df = build_knn_comparison_dataset(
        target_business_df=target_business_df,
        comparison_df=comparison_df,
        target_business=target_business,
    )
    profiles = build_business_feature_profiles(posts_df)
    return add_dashboard_quality_metric(profiles), target_business


def select_target_and_sector_profiles(
    profiles: pd.DataFrame,
    target_business: Dict[str, str],
) -> Tuple[pd.Series, pd.DataFrame]:
    """Select the uploaded business profile and all businesses in its sector."""
    target_filter = (
        profiles["business_name"].astype(str).str.lower()
        == target_business["business_name"].lower()
    )
    target_filter &= (
        profiles["sector"].astype(str).str.lower()
        == target_business["sector"].lower()
    )

    if not target_filter.any():
        raise ValueError("Uploaded business could not be profiled for benchmarking.")

    target_profile = profiles[target_filter].iloc[0]
    sector_profiles = profiles[
        profiles["sector"].astype(str).str.lower()
        == str(target_profile["sector"]).lower()
    ].copy()

    if sector_profiles.empty:
        raise ValueError("No sector businesses were available for benchmarking.")

    return target_profile, sector_profiles


def build_sector_ranking(
    sector_profiles: pd.DataFrame,
    target_business_name: str,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Build frontend ranking rows using normalized presentation scores."""
    ranked = sector_profiles.copy()
    ranked["dashboard_score"] = scale_series_to_range(
        ranked["success_score"],
        low=60,
        high=95,
    ).round(0).astype(int)
    ranked = ranked.sort_values(
        by=["success_score", "business_name"],
        ascending=[False, True],
    ).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    target_name_lower = target_business_name.lower()
    ranking_rows = []
    target_rank = int(len(ranked))
    target_score = 60

    for _, row in ranked.iterrows():
        is_current = str(row["business_name"]).lower() == target_name_lower
        if is_current:
            target_rank = int(row["rank"])
            target_score = int(row["dashboard_score"])

        ranking_rows.append(
            {
                "business_name": row["business_name"],
                "score": int(row["dashboard_score"]),
                "rank": int(row["rank"]),
                "is_current_business": bool(is_current),
            }
        )

    return ranking_rows, target_rank, target_score


def sector_percentile_from_rank(rank: int, total: int) -> int:
    """Estimate how much of the sector the business is outperforming."""
    if total <= 1:
        return 100
    return clamp_percentage(((total - rank) / (total - 1)) * 100, 0, 100)


def dashboard_metric_definitions() -> List[Dict[str, str]]:
    """Return KPI definitions used by the benchmark dashboard."""
    return [
        {
            "metric_key": "engagement",
            "metric_name": "التفاعل",
            "column": "avg_engagement_rate_followers",
        },
        {
            "metric_key": "reach",
            "metric_name": "الوصول",
            "column": "avg_views_per_follower",
        },
        {
            "metric_key": "frequency",
            "metric_name": "الوتيرة",
            "column": "posts_count",
        },
        {
            "metric_key": "quality",
            "metric_name": "الجودة",
            "column": "dashboard_quality_score",
        },
        {
            "metric_key": "hashtags",
            "metric_name": "الهاشتاغات",
            "column": "avg_hashtags_count",
        },
    ]


def build_kpi_comparisons(
    target_profile: pd.Series,
    sector_profiles: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Build frontend-ready KPI comparison rows."""
    comparisons = []
    target_name = str(target_profile.get("business_name", "")).lower()

    for metric in dashboard_metric_definitions():
        column = metric["column"]
        if column not in sector_profiles.columns:
            continue

        normalized_values = normalize_kpi_series_for_dashboard(
            sector_profiles[column]
        )
        target_mask = (
            sector_profiles["business_name"].astype(str).str.lower()
            == target_name
        )
        if target_mask.any():
            business_value = float(normalized_values[target_mask].iloc[0])
        else:
            business_value = float(normalized_values.mean())
        business_value = adjust_dashboard_kpi_value(
            metric_key=metric["metric_key"],
            value=business_value,
            target_profile=target_profile,
            sector_profiles=sector_profiles,
        )

        sector_average = round(float(normalized_values.mean()), 1)
        top_sector_value = round(float(normalized_values.max()), 1)

        comparisons.append(
            {
                "metric_key": metric["metric_key"],
                "metric_name": metric["metric_name"],
                "business_value": business_value,
                "sector_average": sector_average,
                "top_sector_value": top_sector_value,
                "formatted_text": (
                    f"إنتِ {business_value:.1f} · "
                    f"القطاع {sector_average:.1f} · "
                    f"الأعلى {top_sector_value:.1f}"
                ),
            }
        )

    return comparisons


def build_radar_chart(kpi_comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the normalized 0-10 radar chart payload."""
    return {
        "labels": [item["metric_name"] for item in kpi_comparisons],
        "business_values": [item["business_value"] for item in kpi_comparisons],
        "sector_average_values": [item["sector_average"] for item in kpi_comparisons],
    }


def fallback_kpi_insight(item: Dict[str, Any]) -> str:
    """Create a short Arabic fallback explanation for one KPI row."""
    metric_key = item["metric_key"]
    metric_name = item["metric_name"]
    business_value = float(item["business_value"])
    sector_average = float(item["sector_average"])

    strong_messages = {
        "engagement": "التفاعل عندك أقوى من متوسط المشاريع المشابهة، وهاد مؤشر إن الجمهور مستجيب للمحتوى.",
        "reach": "الوصول عندك أفضل من متوسط القطاع، وهاد بساعد المحتوى يوصل لجمهور أوسع.",
        "frequency": "استمرارية النشر واضحة عندك، وهاي نقطة قوة بتحافظ على حضور البيزنس.",
        "quality": "جودة المحتوى عندك أعلى من المتوسط، خصوصًا بالإشارات اللي بتشجع الناس تتفاعل.",
        "hashtags": "الهاشتاغات عندك أقوى من المتوسط، وهاي نقطة ممتازة لزيادة الاكتشاف.",
    }
    weak_messages = {
        "engagement": "التفاعل أقل من أغلب المشاريع المشابهة، فممكن تحتاجوا CTA أوضح أو محتوى يشجع الناس ترد.",
        "reach": "الوصول أقل من متوسط القطاع، وفي فرصة تقوّوه بريلز أو مواضيع أسهل للمشاركة.",
        "frequency": "وتيرة النشر أقل من المشاريع المشابهة، والاستمرارية ممكن ترفع حضوركم.",
        "quality": "جودة المحتوى البصري لسه فيها فرصة، خصوصًا التصوير والإضاءة وتناسق الهوية البصرية.",
        "hashtags": "استخدام الهاشتاغات أقل من المتوسط، فممكن توسعوا الوصول بهاشتاغات محلية أدق.",
    }

    if business_value >= sector_average + 1:
        return strong_messages.get(
            metric_key,
            f"{metric_name} عندك أعلى من متوسط القطاع وهاد بيعطيك نقطة قوة واضحة.",
        )
    if business_value + 1 <= sector_average:
        return weak_messages.get(
            metric_key,
            f"{metric_name} أقل من متوسط القطاع، وفيه مجال لتحسينه بخطوات بسيطة.",
        )
    return f"{metric_name} قريب من متوسط القطاع، والتحسينات الصغيرة ممكن تعمل فرق واضح."


def fallback_sector_insights(kpi_comparisons: List[Dict[str, Any]]) -> List[str]:
    """Build short dashboard insight pills from KPI gaps."""
    insights = []
    strong_insights = {
        "engagement": "التفاعل أقوى من أغلب المشاريع المشابهة",
        "reach": "الوصول عندكم نقطة قوة واضحة",
        "frequency": "استمرارية النشر قوية عندكم",
        "quality": "المحتوى فيه إشارات تفاعل جيدة",
        "hashtags": "الهاشتاغات بتساعد بالاكتشاف",
    }
    weak_insights = {
        "engagement": "التفاعل أقل من حجم الجمهور الحالي",
        "reach": "في فرصة لتوسيع الوصول",
        "frequency": "وتيرة النشر ممكن تكون أقوى",
        "quality": "CTA والموقع ممكن يكونوا أوضح",
        "hashtags": "فرصة أكبر لاستغلال الهاشتاغات المحلية",
    }

    for item in sorted(
        kpi_comparisons,
        key=lambda row: float(row["business_value"]) - float(row["sector_average"]),
        reverse=True,
    ):
        gap = float(item["business_value"]) - float(item["sector_average"])
        if gap >= 1:
            insights.append(strong_insights.get(item["metric_key"], f"{item['metric_name']} نقطة قوة"))

    for item in sorted(
        kpi_comparisons,
        key=lambda row: float(row["business_value"]) - float(row["sector_average"]),
    ):
        gap = float(item["business_value"]) - float(item["sector_average"])
        if gap <= -1:
            insights.append(weak_insights.get(item["metric_key"], f"{item['metric_name']} فيه فرصة تحسين"))

    if not insights:
        insights.append("الأداء قريب من متوسط القطاع")

    return insights[:5]


def build_benchmark_gpt_context(
    target_profile: pd.Series,
    ranking_rows: List[Dict[str, Any]],
    business_summary: Dict[str, Any],
    kpi_comparisons: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a compact context of calculated values for GPT dashboard copy."""
    strengths = [
        {
            "metric_key": item["metric_key"],
            "metric_name": item["metric_name"],
            "business_value": item["business_value"],
            "sector_average": item["sector_average"],
            "gap": round(float(item["business_value"]) - float(item["sector_average"]), 1),
        }
        for item in kpi_comparisons
        if float(item["business_value"]) >= float(item["sector_average"]) + 1
    ]
    opportunities = [
        {
            "metric_key": item["metric_key"],
            "metric_name": item["metric_name"],
            "business_value": item["business_value"],
            "sector_average": item["sector_average"],
            "gap": round(float(item["business_value"]) - float(item["sector_average"]), 1),
        }
        for item in kpi_comparisons
        if float(item["business_value"]) + 1 <= float(item["sector_average"])
    ]

    return {
        "business_summary": business_summary,
        "target_profile": {
            "business_name": target_profile.get("business_name"),
            "sector": target_profile.get("sector"),
            "posts_count": safe_float(target_profile.get("posts_count")),
            "success_score": safe_float(target_profile.get("success_score")),
        },
        "sector_ranking_top": ranking_rows[:5],
        "kpi_comparisons": kpi_comparisons,
        "interpretation_hints": {
            "strengths": strengths,
            "opportunities": opportunities,
        },
    }


# Rule pipeline: Generate and save positive/negative Apriori rules only.
def run_rule_generation_pipeline(json_path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """Run the independent rule-generation API pipeline."""
    return generate_all_rules(json_path=json_path, save_rules=True)


# Similar-business pipeline: Use uploaded data as target and master data as peers.
def run_similar_business_pipeline(
    json_path: Union[str, Path],
) -> Dict[str, Any]:
    """Run the independent similar-business recommendation API pipeline."""
    target_business_df = load_business_json(json_path)
    validate_required_columns(target_business_df, SIMILAR_BUSINESS_REQUIRED_COLUMNS)
    target_business_df = fill_missing_optional_columns(target_business_df)

    recommendations_df, metadata = generate_similar_business_recommendations(
        target_business_df=target_business_df,
    )

    if not metadata.get("ok", False):
        raise ValueError(str(metadata.get("message", "Could not build KNN benchmark.")))

    rewritten_recommendations = rewrite_benchmark_recommendations(recommendations_df)
    similar_businesses = format_similar_businesses(metadata)
    recommendation_cards = format_recommendation_cards(
        rewritten_recommendations,
        metadata=metadata,
    )

    return {
        "status": "success",
        "hero_summary": build_similar_business_hero_summary(recommendation_cards),
        "target_business": format_target_business(metadata),
        "similar_businesses": similar_businesses,
        "benchmark_recommendations": format_benchmark_recommendation_list(
            rewritten_recommendations
        ),
        "recommendation_cards": recommendation_cards,
        "insight_highlights": build_insight_highlights(
            similar_businesses=similar_businesses,
            recommendation_cards=recommendation_cards,
        ),
    }


# Benchmark dashboard pipeline: Analytics-only sector comparison response.
def run_benchmark_dashboard_pipeline(json_path: Union[str, Path]) -> Dict[str, Any]:
    """Run the dedicated benchmark dashboard analytics pipeline."""
    target_business_df = load_business_json(json_path)
    validate_required_columns(target_business_df, SIMILAR_BUSINESS_REQUIRED_COLUMNS)
    target_business_df = fill_missing_optional_columns(target_business_df)

    profiles, target_business = build_dashboard_profiles(target_business_df)
    target_profile, sector_profiles = select_target_and_sector_profiles(
        profiles=profiles,
        target_business=target_business,
    )

    ranking_rows, sector_rank, business_score = build_sector_ranking(
        sector_profiles=sector_profiles,
        target_business_name=target_business["business_name"],
    )
    total_sector_businesses = int(len(sector_profiles))
    sector_percentile = sector_percentile_from_rank(
        rank=sector_rank,
        total=total_sector_businesses,
    )

    business_summary = {
        "business_name": target_profile.get("business_name"),
        "sector": target_profile.get("sector"),
        "sector_rank": sector_rank,
        "sector_percentile": sector_percentile,
        "business_score": business_score,
        "total_sector_businesses": total_sector_businesses,
    }

    kpi_comparisons = build_kpi_comparisons(
        target_profile=target_profile,
        sector_profiles=sector_profiles,
    )
    radar_chart = build_radar_chart(kpi_comparisons)

    fallback_insights = {
        item["metric_key"]: fallback_kpi_insight(item)
        for item in kpi_comparisons
    }
    fallback_sector = fallback_sector_insights(kpi_comparisons)
    strongest_metrics = sorted(
        kpi_comparisons,
        key=lambda item: float(item["business_value"]) - float(item["sector_average"]),
        reverse=True,
    )
    weakest_metrics = sorted(
        kpi_comparisons,
        key=lambda item: float(item["business_value"]) - float(item["sector_average"]),
    )
    strongest_name = strongest_metrics[0]["metric_name"] if strongest_metrics else "الأداء"
    weakest_name = weakest_metrics[0]["metric_name"] if weakest_metrics else "التفاعل"
    if sector_rank <= max(1, total_sector_businesses // 3):
        fallback_summary = (
            f"أداؤكم من الأقوى بقطاع {target_profile.get('sector')}، "
            f"خصوصًا من ناحية {strongest_name}."
        )
    else:
        fallback_summary = (
            f"وجودكم جيد بقطاع {target_profile.get('sector')}، "
            f"وفي فرصة أوضح لتحسين {weakest_name}."
        )

    gpt_texts = generate_benchmark_dashboard_texts(
        context=build_benchmark_gpt_context(
            target_profile=target_profile,
            ranking_rows=ranking_rows,
            business_summary=business_summary,
            kpi_comparisons=kpi_comparisons,
        ),
        fallback_summary=fallback_summary,
        fallback_kpi_insights=fallback_insights,
        fallback_sector_insights=fallback_sector,
    )

    business_summary["summary_text"] = gpt_texts["summary_text"]
    gpt_kpi_insights = gpt_texts.get("kpi_insights", {}) or {}
    for item in kpi_comparisons:
        item["gpt_insight"] = str(
            gpt_kpi_insights.get(
                item["metric_key"],
                fallback_insights[item["metric_key"]],
            )
        )

    return {
        "status": "success",
        "business_summary": business_summary,
        "sector_ranking": ranking_rows,
        "radar_chart": radar_chart,
        "kpi_comparisons": kpi_comparisons,
        "sector_insights": gpt_texts.get("sector_insights", fallback_sector),
    }


# Rule-loading step: Use saved rules, or generate them from processed data if missing.
def get_rules_for_next_post() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load saved Apriori rules, generating them if needed."""
    try:
        return load_saved_rules()
    except FileNotFoundError:
        run_rule_generation_pipeline()
        return load_saved_rules()


# Next-post pipeline: Generate only engagement guidance for one business.
def run_next_post_recommendation_pipeline(json_path: Union[str, Path]) -> Dict[str, Any]:
    """Run the independent next-post recommendation API pipeline."""
    business_df = load_business_json(json_path)
    validate_required_columns(business_df, NEXT_POST_REQUIRED_COLUMNS)
    business_df = fill_missing_optional_columns(business_df)

    positive_rules, negative_rules = get_rules_for_next_post()
    recommendations, metadata = generate_recommendations(
        business_df=business_df,
        positive_rules=positive_rules,
        negative_rules=negative_rules,
    )

    # Rewrite recommendation titles and explanations using GPT.
    for recommendation in recommendations:

        rewritten = rewrite_recommendation(
            title=recommendation.get("title", ""),
            explanation=recommendation.get("explanation", ""),
            recommendation_type=recommendation.get("type", "positive"),
        )

        recommendation["title"] = rewritten["title"]

        recommendation["explanation"] = rewritten["explanation"]

    formatted_recommendations = format_recommendation_list(
        recommendations,
        reliability=float(metadata.get("dataset_reliability", 1.0) or 1.0),
    )

    return {
        "status": "success",
        "summary": build_recommendation_response_summary(formatted_recommendations),
        "business_behavior_summary": metadata.get(
            "business_summary",
            build_business_behavior_summary(business_df),
        ),
        "engagement_recommendations": formatted_recommendations,
    }


# Backward-compatible wrapper: Preserve the old combined pipeline name.
def run_frontend_pipeline(json_path: Union[str, Path]) -> Dict[str, Any]:
    """Run next-post recommendations using the previous adapter function name."""
    return run_next_post_recommendation_pipeline(json_path)


# Schema helper: Expose expected columns to API or frontend callers.
def get_frontend_schema_documentation() -> Dict[str, Any]:
    """Return frontend schema documentation for input validation."""
    return {
        "next_post_required_columns": NEXT_POST_REQUIRED_COLUMNS,
        "similar_business_required_columns": SIMILAR_BUSINESS_REQUIRED_COLUMNS,
        "optional_columns": OPTIONAL_COLUMNS,
    }


# Backward-compatible aliases for older notebook imports.
validate_frontend_columns = validate_required_columns
load_frontend_business_data = load_business_json
run_frontend_recommendation_pipeline = run_frontend_pipeline
_fill_missing_optional_columns = fill_missing_optional_columns
