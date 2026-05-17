from __future__ import annotations

import math
import pickle
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prefer original project logic when available.
_EXTERNAL_PIPELINES_OK = True
try:
    from notebooks.utils.frontend_recommender_adapter import (
        run_benchmark_dashboard_pipeline as _ext_run_benchmark_dashboard_pipeline,
    )
    from notebooks.utils.frontend_recommender_adapter import (
        run_next_post_recommendation_pipeline as _ext_run_next_post_recommendation_pipeline,
    )
    from notebooks.utils.frontend_recommender_adapter import (
        run_rule_generation_pipeline as _ext_run_rule_generation_pipeline,
    )
    from notebooks.utils.frontend_recommender_adapter import (
        run_similar_business_pipeline as _ext_run_similar_business_pipeline,
    )
    from notebooks.utils.generate_rules import generate_all_rules as _ext_generate_all_rules
except Exception:
    _EXTERNAL_PIPELINES_OK = False

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
except Exception:
    apriori = None
    association_rules = None
    TransactionEncoder = None


router = APIRouter(prefix="/api", tags=["recommendation-apis-single"])
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RULE_DATASET = DATA_DIR / "data_processed.json"
POS_RULES = DATA_DIR / "positive_rules.pkl"
NEG_RULES = DATA_DIR / "negative_rules.pkl"
logger = logging.getLogger(__name__)
RULE_CACHE_CONFIG = {
    "version": "association_rules_v4_uncapped_single_api",
    "min_support": 0.06,
    "min_confidence": 0.50,
    "min_lift": 1.05,
    "top_n": None,
    "sort_columns": ("lift", "confidence", "support"),
}


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


def resolve_existing_json_path(json_path: str) -> Path:
    cleaned_path = str(json_path).strip()
    if not cleaned_path:
        raise ValueError("json_path cannot be empty.")

    candidates = [
        Path(cleaned_path),
        PROJECT_ROOT / cleaned_path,
        PROJECT_ROOT / "data" / cleaned_path,
    ]
    for candidate in candidates:
        resolved_candidate = candidate.expanduser().resolve()
        if resolved_candidate.exists() and resolved_candidate.is_file():
            return resolved_candidate
    raise ValueError(f"JSON file was not found: {json_path}")


def _load_df(path_value: str) -> pd.DataFrame:
    path = resolve_existing_json_path(path_value)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        try:
            df = pd.read_json(path)
        except ValueError:
            df = pd.read_json(path, lines=True)
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except Exception:
        return default


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _group_caption(x: Any) -> str:
    v = _safe_float(x)
    if v <= 50:
        return "short"
    if v <= 100:
        return "medium"
    return "long"


def _group_hashtags(x: Any) -> str:
    v = _safe_float(x)
    if v <= 5:
        return "low"
    if v <= 15:
        return "medium"
    return "high"


def _group_emoji(x: Any) -> str:
    v = _safe_float(x)
    if v <= 0:
        return "none"
    if v <= 3:
        return "low"
    return "high"


def _normalize_sector_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    for old, new in {
        "&": "/",
        " and ": "/",
        " / ": "/",
        "\\": "/",
    }.items():
        text = text.replace(old, new)
    return " ".join(text.split())


def _normalize_rule_input(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ["followers_count", "likes_count", "comments_count", "views_count", "caption_length", "hashtags_count", "emoji_count"]:
        if col not in work.columns:
            work[col] = 0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)
    for col in ["post_type", "language", "sector", "day_of_week", "month"]:
        if col not in work.columns:
            work[col] = "Unknown"
        work[col] = work[col].astype(str).fillna("Unknown")
    for col in ["CTA_present", "promo_post", "mentions_location", "religious_theme", "patriotic_theme", "arabic_dialect_style"]:
        if col not in work.columns:
            work[col] = False
        work[col] = work[col].apply(_boolish)
    if "engagement_rate" not in work.columns:
        denom = work["followers_count"].replace(0, pd.NA)
        work["engagement_rate"] = ((work["likes_count"] + work["comments_count"]) / denom) * 100
    work["engagement_rate"] = pd.to_numeric(work["engagement_rate"], errors="coerce").fillna(0)
    work["caption_group"] = work["caption_length"].apply(_group_caption)
    work["hashtags_group"] = work["hashtags_count"].apply(_group_hashtags)
    work["emoji_group"] = work["emoji_count"].apply(_group_emoji)
    return work


def _mine_rules(df: pd.DataFrame, positive: bool) -> pd.DataFrame:
    if apriori is None or association_rules is None or TransactionEncoder is None:
        return pd.DataFrame(columns=["antecedents", "confidence", "lift", "support"])
    work = _normalize_rule_input(df)
    if len(work) < 12:
        return pd.DataFrame(columns=["antecedents", "confidence", "lift", "support"])
    threshold = float(work["engagement_rate"].quantile(0.55 if positive else 0.45))
    target_col = "high_engagement" if positive else "low_engagement"
    work[target_col] = work["engagement_rate"] >= threshold if positive else work["engagement_rate"] <= threshold
    cols = [
        "sector", "day_of_week", "month", "post_type", "language",
        "CTA_present", "promo_post", "mentions_location", "religious_theme", "patriotic_theme",
        "arabic_dialect_style", "caption_group", "hashtags_group", "emoji_group", target_col,
    ]
    tx = [[f"{c}={row[c]}" for c in cols] for _, row in work[cols].iterrows()]
    te = TransactionEncoder()
    matrix = te.fit(tx).transform(tx)
    tx_df = pd.DataFrame(matrix, columns=te.columns_)
    freq = apriori(tx_df, min_support=RULE_CACHE_CONFIG["min_support"], use_colnames=True)
    if freq.empty:
        return pd.DataFrame(columns=["antecedents", "confidence", "lift", "support"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"mlxtend\.frequent_patterns\.association_rules")
        rules = association_rules(freq, metric="confidence", min_threshold=RULE_CACHE_CONFIG["min_confidence"])
    target_item = f"{target_col}=True"
    rules = rules[rules["consequents"].apply(lambda x: target_item in x)].copy()
    rules = rules[pd.to_numeric(rules["lift"], errors="coerce").fillna(0) > RULE_CACHE_CONFIG["min_lift"]].copy()
    if rules.empty:
        return pd.DataFrame(columns=["antecedents", "confidence", "lift", "support"])
    rules = rules[["antecedents", "confidence", "lift", "support"]].sort_values(
        list(RULE_CACHE_CONFIG["sort_columns"]), ascending=False
    )
    top_n = RULE_CACHE_CONFIG.get("top_n")
    if top_n is not None:
        rules = rules.head(int(top_n))
    rules.attrs["cache_config"] = RULE_CACHE_CONFIG.copy()
    rules.attrs["rule_type"] = "positive" if positive else "negative"
    return rules


def _canonical_itemset(value: Any) -> Tuple[str, ...]:
    if isinstance(value, (set, frozenset, list, tuple)):
        return tuple(sorted(str(item) for item in value))
    return (str(value),)


def _dedupe_rules(rules: pd.DataFrame) -> pd.DataFrame:
    if rules.empty or "antecedents" not in rules.columns:
        return rules
    deduped = rules.copy()
    deduped["_antecedent_key"] = deduped["antecedents"].apply(_canonical_itemset)
    before = len(deduped)
    deduped = deduped.drop_duplicates(subset=["_antecedent_key"], keep="first").drop(columns=["_antecedent_key"])
    removed = before - len(deduped)
    if removed > 0:
        logger.info("Removed %s duplicate association rules before caching.", removed)
    deduped.attrs.update(rules.attrs)
    return deduped


def _load_cached_rules() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not POS_RULES.exists() or not NEG_RULES.exists():
        missing = POS_RULES if not POS_RULES.exists() else NEG_RULES
        raise FileNotFoundError(f"Cached rule file is missing: {missing}")

    logger.info("Loading cached positive_rules.pkl...")
    with POS_RULES.open("rb") as positive_file:
        pos = pickle.load(positive_file)

    logger.info("Loading cached negative_rules.pkl...")
    with NEG_RULES.open("rb") as negative_file:
        neg = pickle.load(negative_file)

    if not isinstance(pos, pd.DataFrame) or not isinstance(neg, pd.DataFrame):
        raise ValueError("Cached rule files are invalid.")
    if pos.attrs.get("cache_config") != RULE_CACHE_CONFIG or neg.attrs.get("cache_config") != RULE_CACHE_CONFIG:
        raise ValueError("Cached rule files were generated with an older filter configuration.")
    return pos, neg


def _write_rule_cache(pos: pd.DataFrame, neg: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pos = _dedupe_rules(pos)
    neg = _dedupe_rules(neg)
    pos.attrs["cache_config"] = RULE_CACHE_CONFIG.copy()
    pos.attrs["rule_type"] = "positive"
    neg.attrs["cache_config"] = RULE_CACHE_CONFIG.copy()
    neg.attrs["rule_type"] = "negative"
    with POS_RULES.open("wb") as positive_file:
        pickle.dump(pos, positive_file, protocol=pickle.HIGHEST_PROTOCOL)
    with NEG_RULES.open("wb") as negative_file:
        pickle.dump(neg, negative_file, protocol=pickle.HIGHEST_PROTOCOL)


def _generate_all_rules_from_dataset(json_path: Optional[str], save_rules: bool = True) -> Dict[str, Any]:
    logger.info("Generating rules from dataset...")
    candidates = [json_path] if json_path else [str(DEFAULT_RULE_DATASET)]
    last: Exception | None = None
    df = None
    for c in candidates:
        try:
            if c is None:
                continue
            df = _load_df(c)
            break
        except Exception as exc:
            last = exc
    if df is None:
        raise FileNotFoundError(f"Rule dataset was not found: {last}")
    pos = _dedupe_rules(_mine_rules(df, positive=True))
    neg = _dedupe_rules(_mine_rules(df, positive=False))
    if save_rules:
        _write_rule_cache(pos, neg)
    return {"status": "success", "positive_rules_count": int(len(pos)), "negative_rules_count": int(len(neg))}


def _cached_or_generate_all_rules(json_path: Optional[str], save_rules: bool = True) -> Dict[str, Any]:
    try:
        pos, neg = _load_cached_rules()
        return {"status": "success", "positive_rules_count": int(len(pos)), "negative_rules_count": int(len(neg))}
    except (FileNotFoundError, pickle.PickleError, EOFError, AttributeError, ImportError, IndexError, ValueError):
        return _generate_all_rules_from_dataset(json_path, save_rules=save_rules)


def _fallback_generate_all_rules(json_path: Optional[str], save_rules: bool = True) -> Dict[str, Any]:
    return _generate_all_rules_from_dataset(json_path, save_rules=save_rules)


def _behavior_summary(df: pd.DataFrame) -> Dict[str, Any]:
    work = df.copy()
    for col in ["hashtags_count", "caption_length", "emoji_count", "followers_count", "likes_count", "comments_count", "views_count"]:
        if col not in work.columns:
            work[col] = 0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)
    for col in ["CTA_present", "promo_post", "mentions_location"]:
        if col not in work.columns:
            work[col] = False
        work[col] = work[col].apply(_boolish)
    return {
        "posts_count": int(len(work)),
        "business_name": str(work["business_name"].iloc[0]) if "business_name" in work.columns else "Uploaded Business",
        "sector": str(work["sector"].iloc[0]) if "sector" in work.columns else "Unknown",
        "avg_hashtags_count": float(work["hashtags_count"].mean()),
        "avg_caption_length": float(work["caption_length"].mean()),
        "avg_emoji_count": float(work["emoji_count"].mean()),
        "CTA_present_rate": float(work["CTA_present"].mean()),
        "promo_post_rate": float(work["promo_post"].mean()),
        "mentions_location_rate": float(work["mentions_location"].mean()),
        "pct_post_type_reel": float((work.get("post_type", "post").astype(str).str.lower() == "reel").mean()) if "post_type" in work.columns else 0.0,
    }


def _fallback_next_post_pipeline(json_path: str) -> Dict[str, Any]:
    df = _load_df(json_path)
    missing = [c for c in NEXT_POST_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required input columns: " + ", ".join(sorted(missing)))
    summary = _behavior_summary(df)
    recs: List[Dict[str, Any]] = []
    if summary["avg_hashtags_count"] <= 5:
        recs.append(("زيد الهاشتاغات بشكل أذكى", "الهاشتاغات عندك قليلة، وزيادة بسيطة ومدروسة منها ممكن توسّع الاكتشاف وتجيب وصول أحسن للمحتوى.", 78))
    if summary["avg_caption_length"] <= 50:
        recs.append(("طوّل الكابشن شوي", "الكابشن القصير كثير ممكن يضيّع الفكرة بسرعة، بينما نص أوضح شوي بيساعد الرسالة توصل ويقوّي التفاعل.", 68))
    if summary["CTA_present_rate"] < 0.35:
        recs.append(("حط دعوة واضحة للتفاعل", "وجود CTA واضح مثل اطلب، احجز، ابعت، أو علّق، برفع احتمالية إن الناس تتجاوب بدل ما تمرّ على المنشور وبس.", 82))
    if summary["mentions_location_rate"] < 0.35:
        recs.append(("اذكر الموقع بشكل أوضح", "ربط المحتوى بالمكان بخلي المنشور أقرب للجمهور المحلي وبيعطيه فرصة أفضل بالوصول القريب منك.", 72))
    if summary["promo_post_rate"] > 0.55:
        recs.append(("خفف النبرة البيعية المباشرة", "لما المحتوى يصير بيعي بزيادة، الناس بتتفاعل أقل. خلط القيمة مع البيع عادة بيعطي نتيجة أحسن.", 74))
    if summary["pct_post_type_reel"] < 0.35:
        recs.append(("جرّب ريلز أكثر", "الريلز غالبًا بتعطي فرصة أقوى للوصول والانتشار، خصوصًا إذا كانت سريعة وواضحة وبداية الفيديو تشد.", 80))
    if not recs:
        recs.append(("كمّل على النهج الحالي مع تحسينات خفيفة", "الأداء عندك متوازن، فالأفضل تشتغل على تجارب صغيرة محسوبة بدل تغييرات كبيرة دفعة وحدة.", 50))
    out = []
    for i, (title, explanation, score) in enumerate(recs, start=1):
        out.append({
            "id": i,
            "type": "positive",
            "category": "content",
            "title": title,
            "explanation": explanation,
            "priority_score": float(score),
            "expected_impact": int(max(8, min(38, round(6 + score * 0.26)))),
            "confidence_score": int(max(70, min(95, round(62 + score * 0.25)))),
            "action_type": "increase_engagement",
            "icon": "sparkles",
        })
    return {
        "status": "success",
        "summary": {"total_recommendations": len(out), "estimated_total_impact": int(min(95, max(0, len(out) * 12)))},
        "business_behavior_summary": summary,
        "engagement_recommendations": out,
    }


def _profiles(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ["followers_count", "likes_count", "comments_count", "views_count", "hashtags_count", "caption_length", "emoji_count"]:
        if col not in work.columns:
            work[col] = 0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)
    for col in ["CTA_present", "promo_post", "mentions_location"]:
        if col not in work.columns:
            work[col] = False
        work[col] = work[col].apply(_boolish)
    denom = work["followers_count"].replace(0, pd.NA)
    work["engagement_rate_followers"] = ((work["likes_count"] + work["comments_count"]) / denom).fillna(0)
    work["views_per_follower"] = (work["views_count"] / denom).fillna(0)
    grouped = work.groupby(["business_name", "sector"], as_index=False).agg({
        "engagement_rate_followers": "mean",
        "views_per_follower": "mean",
        "comments_count": "mean",
        "hashtags_count": "mean",
        "caption_length": "mean",
        "emoji_count": "mean",
        "CTA_present": "mean",
        "promo_post": "mean",
        "mentions_location": "mean",
    })
    grouped.rename(columns={
        "comments_count": "avg_comments_count",
        "hashtags_count": "avg_hashtags_count",
        "caption_length": "avg_caption_length",
        "emoji_count": "avg_emoji_count",
        "CTA_present": "pct_CTA_present",
        "promo_post": "pct_promo_post",
        "mentions_location": "pct_mentions_location",
    }, inplace=True)
    grouped["success_score"] = grouped["engagement_rate_followers"] * 0.42 + grouped["views_per_follower"] * 0.18 + grouped["avg_comments_count"] * 0.40
    return grouped


def _scale_series_to_range(series: pd.Series, low: float, high: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if max_value == min_value:
        return pd.Series(float(low), index=series.index)
    scaled = (numeric - min_value) / (max_value - min_value)
    return (scaled * (high - low)) + low


def _build_benchmark_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build benchmark profiles with normalized success scoring so ranking and
    sector comparison stay stable across businesses of different scales.
    """
    work = df.copy()
    for col in [
        "followers_count",
        "likes_count",
        "comments_count",
        "shares_count",
        "saves_count",
        "views_count",
        "hashtags_count",
        "caption_length",
        "emoji_count",
    ]:
        if col not in work.columns:
            work[col] = 0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    for col in [
        "CTA_present",
        "promo_post",
        "mentions_location",
        "religious_theme",
        "patriotic_theme",
        "arabic_dialect_style",
    ]:
        if col not in work.columns:
            work[col] = False
        work[col] = work[col].apply(_boolish)

    if "post_type" in work.columns:
        work["post_type"] = work["post_type"].astype(str).str.strip().str.lower()

    followers = work["followers_count"].replace(0, pd.NA)
    engagement = work["likes_count"] + (work["comments_count"] * 2) + (work["views_count"] * 0.1)
    work["engagement_rate_followers"] = (engagement / followers).fillna(0.0)
    work["views_per_follower"] = (work["views_count"] / followers).fillna(0.0)

    grouped = work.groupby(["business_name", "sector"], as_index=False).agg({
        "engagement_rate_followers": "mean",
        "views_per_follower": "mean",
        "comments_count": "mean",
        "shares_count": "mean",
        "saves_count": "mean",
        "hashtags_count": "mean",
        "caption_length": "mean",
        "emoji_count": "mean",
        "CTA_present": "mean",
        "promo_post": "mean",
        "mentions_location": "mean",
        "arabic_dialect_style": "mean",
    })
    grouped.rename(columns={
        "comments_count": "avg_comments_count",
        "shares_count": "avg_shares_count",
        "saves_count": "avg_saves_count",
        "hashtags_count": "avg_hashtags_count",
        "caption_length": "avg_caption_length",
        "emoji_count": "avg_emoji_count",
        "CTA_present": "pct_CTA_present",
        "promo_post": "pct_promo_post",
        "mentions_location": "pct_mentions_location",
        "arabic_dialect_style": "pct_arabic_dialect_style",
    }, inplace=True)

    for metric, weight in {
        "engagement_rate_followers": 0.42,
        "views_per_follower": 0.18,
        "avg_shares_count": 0.16,
        "avg_saves_count": 0.14,
        "avg_comments_count": 0.10,
    }.items():
        grouped[f"{metric}_norm"] = _scale_series_to_range(grouped[metric], 0.0, 1.0)
        grouped[f"{metric}_contribution"] = grouped[f"{metric}_norm"] * weight

    grouped["success_score"] = grouped[
        [
            "engagement_rate_followers_contribution",
            "views_per_follower_contribution",
            "avg_shares_count_contribution",
            "avg_saves_count_contribution",
            "avg_comments_count_contribution",
        ]
    ].sum(axis=1).round(4)

    grouped["dashboard_quality_score"] = (
        _scale_series_to_range(grouped["avg_comments_count"], 0.0, 1.0) * 0.40
        + pd.to_numeric(grouped["pct_CTA_present"], errors="coerce").fillna(0.0) * 0.25
        + pd.to_numeric(grouped["pct_mentions_location"], errors="coerce").fillna(0.0) * 0.20
        + pd.to_numeric(grouped["pct_arabic_dialect_style"], errors="coerce").fillna(0.0) * 0.15
    ).round(4)

    return grouped


def _scale_values_to_range(values: List[float], low: int, high: int) -> List[int]:
    if not values:
        return []
    max_value = max(values)
    min_value = min(values)
    if max_value == min_value:
        return [int(round(high - (index * 2))) for index, _ in enumerate(values)]

    scaled = []
    for value in values:
        relative_position = (value - min_value) / max(max_value - min_value, 1e-9)
        display_value = low + (relative_position * (high - low))
        scaled.append(int(round(max(low, min(high, display_value)))))
    return scaled


def _load_similar_business_master_df(target_json_path: str) -> pd.DataFrame:
    """
    Prefer a full multi-business dataset for peer comparison.
    Fall back to the requested JSON only if no richer benchmark dataset exists.
    """
    candidates = [
        "data/data_processed.json",
        "data/processed/data_processed.json",
        "data/vanilla_processed.json",
        "data/vanilla_kpi_dataset.json",
        target_json_path,
    ]

    for candidate in candidates:
        try:
            df = _load_df(candidate)
        except Exception:
            continue

        business_count = (
            int(df["business_name"].astype(str).str.strip().nunique())
            if "business_name" in df.columns
            else 0
        )
        logger.info(
            "Similar-business master candidate path=%s unique_businesses=%s rows=%s",
            candidate,
            business_count,
            len(df),
        )
        if business_count > 1:
            return df

    return _load_df(target_json_path)


def _fallback_similar_pipeline(json_path: str) -> Dict[str, Any]:
    target_df = _load_df(json_path)
    missing = [c for c in SIMILAR_BUSINESS_REQUIRED_COLUMNS if c not in target_df.columns]
    if missing:
        raise ValueError("Missing required input columns: " + ", ".join(sorted(missing)))
    master = _load_similar_business_master_df(json_path)
    target_name = str(target_df["business_name"].iloc[0])
    target_sector_input = str(target_df["sector"].iloc[0]) if "sector" in target_df.columns else ""
    normalized_target_sector = _normalize_sector_label(target_sector_input)
    logger.info("Detected business sector=%s normalized_sector=%s", target_sector_input, normalized_target_sector)

    if {"business_name", "sector"}.issubset(master.columns):
        duplicate_target = (
            master["business_name"].astype(str).str.strip().str.lower().eq(target_name.strip().lower())
            & master["sector"].apply(_normalize_sector_label).eq(normalized_target_sector)
        )
        if duplicate_target.any():
            logger.info("Removing target business rows from master dataset before similarity search=%s", int(duplicate_target.sum()))
            master = master[~duplicate_target].copy()

    all_df = pd.concat([master, target_df], ignore_index=True)
    prof = _profiles(all_df)
    target = prof[
        (prof["business_name"].astype(str).str.strip().str.lower() == target_name.strip().lower())
        & (
            prof["sector"].apply(_normalize_sector_label)
            == normalized_target_sector
        )
    ]
    if target.empty:
        raise ValueError("Target business profile not found.")
    target_row = target.iloc[0]
    target_sector = str(target_row["sector"])
    logger.info("Filtering similarity candidates to sector=%s", target_sector)
    logger.info("Total businesses in dataset=%s", len(prof))
    peers = prof[
        (prof["business_name"].astype(str).str.strip().str.lower() != target_name.strip().lower())
        & (
            prof["sector"].apply(_normalize_sector_label)
            == _normalize_sector_label(target_sector)
        )
    ].copy()
    logger.info("Businesses remaining after same-sector filtering=%s", len(peers))
    logger.info("Businesses remaining after removing target business=%s", len(peers))
    logger.info("Similarity candidate count after sector/category filtering=%s", len(peers))
    if not peers.empty:
        similarity_columns = [
            "engagement_rate_followers",
            "views_per_follower",
            "avg_comments_count",
            "avg_hashtags_count",
            "pct_CTA_present",
            "pct_mentions_location",
            "avg_caption_length",
            "pct_promo_post",
        ]
        available_similarity_columns = [column for column in similarity_columns if column in peers.columns and column in target_row.index]
        if available_similarity_columns:
            scale_frame = pd.concat(
                [
                    target_row[available_similarity_columns].to_frame().T,
                    peers[available_similarity_columns],
                ],
                ignore_index=True,
            ).apply(pd.to_numeric, errors="coerce").fillna(0.0)
            mins = scale_frame.min()
            maxs = scale_frame.max()
            denom = (maxs - mins).replace(0, 1.0)
            normalized = (scale_frame - mins) / denom
            target_vector = normalized.iloc[0]
            peer_vectors = normalized.iloc[1:].reset_index(drop=True)
            peers = peers.reset_index(drop=True)
            column_weights = {
                "engagement_rate_followers": 0.24,
                "views_per_follower": 0.20,
                "avg_comments_count": 0.16,
                "avg_hashtags_count": 0.08,
                "pct_CTA_present": 0.10,
                "pct_mentions_location": 0.08,
                "avg_caption_length": 0.06,
                "pct_promo_post": 0.08,
            }
            applied_weights = pd.Series(
                {column: column_weights.get(column, 0.05) for column in available_similarity_columns},
                dtype="float64",
            )
            applied_weights = applied_weights / applied_weights.sum()
            weighted_distance = peer_vectors.sub(target_vector, axis=1).abs().mul(applied_weights, axis=1).sum(axis=1)
            peers["similarity_score_raw"] = (1 - weighted_distance.clip(lower=0, upper=1)).round(4)
            peers["similarity_score"] = peers["similarity_score_raw"]
            logger.info(
                "Similarity scores before filtering=%s",
                [round(float(score), 4) for score in peers["similarity_score"].head(10).tolist()],
            )
            peers = peers.sort_values(["similarity_score", "success_score"], ascending=[False, False]).head(10)
        else:
            logger.info("Similarity scores before filtering=[]")
            peers = peers.sort_values("success_score", ascending=False).head(10).copy()
            peers["similarity_score_raw"] = 0.65
            peers["similarity_score"] = 0.65
    else:
        peers = peers.copy()
    logger.info("Final similar business count=%s", len(peers))
    raw_similarity_scores = [float(r.get("similarity_score_raw", r.get("similarity_score", 0.65)) or 0.65) for _, r in peers.iterrows()]
    similarity_percentages = _scale_values_to_range(raw_similarity_scores, low=84, high=94)
    raw_success_scores = [float(r["success_score"]) for _, r in peers.iterrows()]
    success_display_scores = _scale_values_to_range(raw_success_scores, low=60, high=95)
    inspired_by = [str(x) for x in peers["business_name"].head(3)]
    inspired_text = "، ".join(inspired_by) if inspired_by else "أقرب المنافسين"
    cards = [{
        "id": 1,
        "title": "قوّي إشارات التفاعل بالمحتوى",
        "explanation": f"البيزنسات الأقرب إلك مثل {inspired_text} عندهم نمط محتوى بجيب تفاعل أوضح، فالأولوية تكون لمحتوى يشجّع الناس تعلق وتحفظ وتتفاعل بسرعة.",
        "expected_impact": 24,
        "confidence_score": 86,
        "priority_score": 72.0,
        "category": "engagement",
        "icon": "sparkles",
        "inspired_by": inspired_by,
    }]
    return {
        "status": "success",
        "hero_summary": {
            "title": "توصيات مبنية على بيزنسات قريبة من نمطك",
            "total_recommendations": len(cards),
            "top_opportunity": cards[0]["title"],
            "estimated_best_impact": cards[0]["expected_impact"],
        },
        "target_business": {"business_name": target_name, "sector": str(target_row["sector"]), "success_score": float(target_row["success_score"])},
        "similar_businesses": [
            {
                "rank": i + 1,
                "business_name": str(r["business_name"]),
                "sector": str(r["sector"]),
                "similarity_score": similarity_percentages[i],
                "similarity_raw_score": float(r.get("similarity_score_raw", r.get("similarity_score", 0.65)) or 0.65),
                "success_score": success_display_scores[i],
                "success_raw_score": float(r["success_score"]),
            }
            for i, (_, r) in enumerate(peers.iterrows())
        ],
        "benchmark_recommendations": [{"title": cards[0]["title"], "explanation": cards[0]["explanation"], "priority_score": cards[0]["priority_score"]}],
        "recommendation_cards": cards,
        "insight_highlights": ["هاد التحليل مبني على أقرب بيزنسات إلك من ناحية الأداء والنمط."],
    }


def _ensure_next_post_recommendations_arabic(payload: Dict[str, Any]) -> Dict[str, Any]:
    mapping = {
        "Use a few more hashtags": (
            "زيد الهاشتاغات بشكل أذكى",
            "زيادة بسيطة ومدروسة بالهاشتاغات المناسبة ممكن توسّع الاكتشاف وتخلي المحتوى يوصل لناس أكثر."
        ),
        "Try slightly longer captions": (
            "طوّل الكابشن شوي",
            "لما الفكرة تنشرح بشكل أوضح داخل الكابشن، فرص الفهم والتفاعل بتصير أحسن."
        ),
        "Add clearer calls-to-action": (
            "حط دعوة واضحة للتفاعل",
            "استخدام CTA مباشر مثل اطلب أو احجز أو ابعت رسالة بيعطي المتابع خطوة واضحة بعد ما يشوف المنشور."
        ),
        "Mention location more often": (
            "اذكر الموقع بشكل أوضح",
            "إظهار المكان داخل المحتوى أو الكابشن بيساعدك تقرّب من الجمهور المحلي ويرفع الصلة."
        ),
        "Reduce heavy promotional tone": (
            "خفف النبرة البيعية المباشرة",
            "إذا كان المحتوى بيعي أكثر من اللازم، التفاعل غالبًا يهبط. الموازنة بين القيمة والبيع بتشتغل أحسن."
        ),
        "Try more reels": (
            "جرّب ريلز أكثر",
            "الريلز عادة بتعطي فرصة أكبر للوصول والانتشار، خاصة إذا أول ثانيتين كانوا شدّاد."
        ),
        "Maintain your current strategy": (
            "كمّل على النهج الحالي مع تحسينات خفيفة",
            "الأداء متوازن، فالأفضل تظل تختبر تحسينات صغيرة بدل تغيير كل شيء مرة وحدة."
        ),
    }
    for item in payload.get("engagement_recommendations", []) or []:
        title = str(item.get("title", "")).strip()
        explanation = str(item.get("explanation", "")).strip()
        if title in mapping:
            item["title"], item["explanation"] = mapping[title]
        elif title and all(ord(ch) < 128 for ch in title):
            item["title"] = "توصية لتحسين أداء المحتوى"
            if not explanation or all(ord(ch) < 128 for ch in explanation):
                item["explanation"] = "في فرصة عملية لتحسين طريقة تقديم المحتوى حتى يوصل أفضل ويجيب تفاعل أوضح."
    return payload


def _ensure_similar_business_recommendations_arabic(payload: Dict[str, Any]) -> Dict[str, Any]:
    hero_summary = payload.get("hero_summary", {}) or {}
    if not str(hero_summary.get("title", "")).strip() or all(ord(ch) < 128 for ch in str(hero_summary.get("title", ""))):
        hero_summary["title"] = "توصيات مبنية على بيزنسات قريبة من نمطك"

    for item in payload.get("benchmark_recommendations", []) or []:
        title = str(item.get("title", "")).strip()
        explanation = str(item.get("explanation", "")).strip()
        if not title or all(ord(ch) < 128 for ch in title):
            item["title"] = "قوّي إشارات التفاعل بالمحتوى"
        if not explanation or all(ord(ch) < 128 for ch in explanation):
            item["explanation"] = "البيزنسات المشابهة إلك عم تعطي إشارات إن المحتوى اللي يشجع على التفاعل السريع والتفاعل الحقيقي بطلع بنتيجة أقوى."

    for item in payload.get("recommendation_cards", []) or []:
        title = str(item.get("title", "")).strip()
        explanation = str(item.get("explanation", "")).strip()
        if not title or all(ord(ch) < 128 for ch in title):
            item["title"] = "قوّي إشارات التفاعل بالمحتوى"
        if not explanation or all(ord(ch) < 128 for ch in explanation):
            inspired_by = item.get("inspired_by", []) or []
            inspired_text = "، ".join(str(x) for x in inspired_by[:3]) if inspired_by else "البيزنسات الأقرب إلك"
            item["explanation"] = f"{inspired_text} عندهم نمط محتوى أقوى من ناحية التفاعل، فالأفضل تركّز على محتوى أوضح، بداية أقوى، وسبب مباشر يخلي الناس تتفاعل."

    highlights = payload.get("insight_highlights", []) or []
    cleaned_highlights = []
    for text in highlights:
        text = str(text).strip()
        if not text or all(ord(ch) < 128 for ch in text):
            cleaned_highlights.append("هاد التحليل مبني على مقارنة مع أقرب بيزنسات من ناحية الأداء والنمط.")
        else:
            cleaned_highlights.append(text)
    payload["insight_highlights"] = cleaned_highlights or ["هاد التحليل مبني على مقارنة مع أقرب بيزنسات من ناحية الأداء والنمط."]
    payload["hero_summary"] = hero_summary
    return payload


def _load_benchmark_master_df() -> pd.DataFrame:
    for candidate in [str(DEFAULT_RULE_DATASET), "data/data_processed.json", "data/vanilla_kpi_dataset.json"]:
        try:
            return _load_df(candidate)
        except Exception:
            continue
    raise FileNotFoundError("Benchmark comparison dataset was not found.")


PLACEHOLDER_INSIGHTS = {
    "",
    "Benchmark comparison generated.",
    "Ranking and KPI gaps are computed from business profiles.",
}


def _metric_business_meaning(metric_key: str) -> Tuple[str, str]:
    meanings = {
        "engagement": ("التفاعل", "الجمهور عم يتجاوب مع المحتوى"),
        "reach": ("الوصول", "المنشورات عم توصل وتنشاف بشكل أوسع"),
        "quality": ("جودة التفاعل", "المحتوى قوي لدرجة يجيب ردود أعمق"),
        "hashtags": ("استخدام الهاشتاغات", "استراتيجية الهاشتاغات عم تدعم الاكتشاف"),
    }
    return meanings.get(metric_key, (metric_key.replace("_", " "), "هالمؤشر عم يدعم أداء السوشال"))


def _comparison_status(business_value: float, sector_average: float, top_sector_value: float) -> Tuple[str, str]:
    if top_sector_value > 0 and business_value >= top_sector_value * 0.95:
        return "leading", "قريب جدًا من أعلى أداء بالقطاع"
    if sector_average <= 0:
        return ("above average", "متقدّم على خط الأساس بالقطاع") if business_value > 0 else ("below average", "لسه بحاجة يبني زخم")
    diff_ratio = (business_value - sector_average) / abs(sector_average)
    if diff_ratio >= 0.15:
        return "above average", "أعلى من متوسط القطاع"
    if diff_ratio <= -0.15:
        return "below average", "أقل من متوسط القطاع"
    return "around average", "قريب من متوسط القطاع"


def _format_gap_phrase(business_value: float, sector_average: float) -> str:
    if sector_average <= 0:
        return "عنده فرق واضح بالمقارنة مع القطاع"
    diff_pct = ((business_value - sector_average) / abs(sector_average)) * 100
    if abs(diff_pct) < 5:
        return "قريب جدًا من متوسط القطاع"
    direction = "أعلى من" if diff_pct > 0 else "أقل من"
    return f"أعلى بنسبة {abs(diff_pct):.0f}% من متوسط القطاع" if diff_pct > 0 else f"أقل بنسبة {abs(diff_pct):.0f}% من متوسط القطاع"


def _ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _build_kpi_text(item: Dict[str, Any], business_name: str) -> Tuple[str, str]:
    metric_key = str(item.get("metric_key", "metric"))
    metric_name = str(item.get("metric_name") or metric_key.replace("_", " ").title())
    business_value = _safe_float(item.get("business_value"))
    sector_average = _safe_float(item.get("sector_average"))
    top_sector_value = _safe_float(item.get("top_sector_value"))
    status, position_phrase = _comparison_status(business_value, sector_average, top_sector_value)
    metric_label, meaning = _metric_business_meaning(metric_key)
    gap_phrase = _format_gap_phrase(business_value, sector_average)

    formatted_text = f"{business_name} {gap_phrase} بمؤشر {metric_name}."

    if status == "leading":
        implication = f"هاي نقطة قوة واضحة، لأن {metric_label} عندك عم ينافس الأفضل، فالأولوية تحافظ على نفس النمط اللي جاب النتيجة."
    elif status == "above average":
        implication = f"أداء {metric_label} قوي، وهذا غالبًا يعني إن {meaning} عندك أحسن من كثير من المنافسين."
    elif status == "below average":
        implication = f"هاي فرصة تحسين، لأن ضعف {metric_label} ممكن يخفف النمو لأنه يأثر على كيف {meaning}."
    else:
        implication = f"الوضع مستقر بمؤشر {metric_label}، لكن تحسينات صغيرة بالمحتوى وطريقة النشر ممكن تدفشك لفوق أكثر من وسط القطاع."

    gpt_insight = (
        f"مؤشر {metric_label} عندك {position_phrase}. {implication} "
        f"القيمة الحالية {business_value:g}، مقابل متوسط قطاع {sector_average:g}، وأعلى قيمة بالقطاع {top_sector_value:g}."
    )
    return formatted_text, gpt_insight


def _build_sector_insights(
    business_summary: Dict[str, Any],
    kpi_comparisons: List[Dict[str, Any]],
) -> List[str]:
    business_name = str(business_summary.get("business_name", "البيزنس"))
    sector = str(business_summary.get("sector", "القطاع"))
    rank = int(_safe_float(business_summary.get("sector_rank"), 0))
    total = int(_safe_float(business_summary.get("total_sector_businesses"), len(kpi_comparisons)))
    percentile = int(_safe_float(business_summary.get("sector_percentile"), 0))

    def gap(item: Dict[str, Any]) -> float:
        avg = _safe_float(item.get("sector_average"))
        if avg == 0:
            return _safe_float(item.get("business_value"))
        return (_safe_float(item.get("business_value")) - avg) / abs(avg)

    ordered = sorted(kpi_comparisons, key=gap, reverse=True)
    strongest = ordered[0] if ordered else {}
    weakest = ordered[-1] if ordered else {}
    strongest_name = str(strongest.get("metric_name", "أقوى مؤشر"))
    weakest_name = str(weakest.get("metric_name", "أضعف مؤشر"))

    if total > 0 and rank > 0:
        position = f"{business_name} ترتيبه {rank} من أصل {total} بقطاع {sector}، وهذا بحطه تقريبًا عند المئين {percentile}."
    else:
        position = f"{business_name} عنده حضور تنافسي واضح بقطاع {sector}."

    insights = [
        position,
        f"أقوى مؤشر نسبيًا هو {strongest_name}، وهاي إشارة للنمط اللي بستاهل يتكرر أكثر.",
        f"أوضح فرصة تحسين هي {weakest_name}، ورفعها ممكن يحسن ترتيبك العام بالمقارنة.",
    ]

    if percentile >= 75:
        insights.append("تمركزك التنافسي قوي، فالأفضل تحافظ على الاستمرارية وتجرب تحسينات صغيرة محسوبة.")
    elif percentile >= 45:
        insights.append("أنت منافس بشكل منيح، لكن لسه في مجال تبعد أكثر عن المتوسط بتنفيذ محتوى أذكى وأوضح.")
    else:
        insights.append("أداءك لسه وراء كثير من المنافسين، فالأولوية لازم تروح للمؤشرات الأبعد عن متوسط القطاع.")

    insights.append("استراتيجية المحتوى لازم تربط الفورمات الأقوى مع CTA أوضح، إشارات وصول أفضل، وعادة نشر ثابتة.")
    return insights[:5]


def _ensure_benchmark_text_quality(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Replace empty or placeholder benchmark copy without changing response shape."""
    business_summary = payload.get("business_summary", {}) or {}
    business_name = str(business_summary.get("business_name", "البيزنس"))
    kpi_comparisons = payload.get("kpi_comparisons", []) or []

    replaced_any = False
    for item in kpi_comparisons:
        formatted_text, insight = _build_kpi_text(item, business_name)
        if str(item.get("formatted_text", "")).strip() in PLACEHOLDER_INSIGHTS:
            item["formatted_text"] = formatted_text
            replaced_any = True
        if str(item.get("gpt_insight", "")).strip() in PLACEHOLDER_INSIGHTS:
            item["gpt_insight"] = insight
            replaced_any = True

    sector_insights = [
        str(item).strip()
        for item in (payload.get("sector_insights", []) or [])
        if str(item).strip() and str(item).strip() not in PLACEHOLDER_INSIGHTS
    ]
    if not sector_insights:
        sector_insights = _build_sector_insights(business_summary, kpi_comparisons)
        replaced_any = True
    payload["sector_insights"] = sector_insights

    if not str(business_summary.get("summary_text", "")).strip():
        business_summary["summary_text"] = sector_insights[0] if sector_insights else f"تحليل المقارنة صار جاهز لـ {business_name}."
        replaced_any = True

    if replaced_any:
        logger.info("Benchmark insight fallback usage: generated local analytical explanations.")
    logger.info("Benchmark insight generation success.")
    return payload


def _fallback_benchmark_pipeline(json_path: str) -> Dict[str, Any]:
    target_df = _load_df(json_path)
    master = _load_benchmark_master_df()
    target_name = str(target_df["business_name"].iloc[0])
    target_sector = str(target_df["sector"].iloc[0]) if "sector" in target_df.columns else ""
    if {"business_name", "sector"}.issubset(master.columns):
        duplicate_target = (
            master["business_name"].astype(str).str.lower().eq(target_name.lower())
            & master["sector"].astype(str).str.lower().eq(target_sector.lower())
        )
        master = master[~duplicate_target].copy()
    all_df = pd.concat([master, target_df], ignore_index=True)
    prof = _build_benchmark_profiles(all_df).sort_values("success_score", ascending=False).reset_index(drop=True)
    logger.info("Benchmark total businesses in dataset=%s", len(prof))
    normalized_target_sector = _normalize_sector_label(target_sector)
    trows = prof[
        prof["business_name"].astype(str).str.strip().str.lower().eq(target_name.strip().lower())
        & prof["sector"].apply(_normalize_sector_label).eq(normalized_target_sector)
    ]
    if trows.empty:
        raise ValueError("Target business profile not found.")
    trow = trows.iloc[0]
    sector_prof = prof[
        prof["sector"].apply(_normalize_sector_label)
        == _normalize_sector_label(str(trow["sector"]))
    ].copy()
    if sector_prof.empty:
        sector_prof = prof
    sector_prof = sector_prof.sort_values("success_score", ascending=False).reset_index(drop=True)
    logger.info("Benchmark sector=%s sector_businesses=%s", str(trow["sector"]), len(sector_prof))
    rank = int(
        sector_prof.index[
            sector_prof["business_name"].astype(str).str.strip().str.lower()
            == target_name.strip().lower()
        ][0]
    ) + 1
    total = len(sector_prof)
    metrics = [
        ("engagement", "Engagement", "engagement_rate_followers"),
        ("reach", "Reach", "views_per_follower"),
        ("quality", "Quality", "dashboard_quality_score"),
        ("hashtags", "Hashtags", "avg_hashtags_count"),
    ]
    kpi = []
    for key, name, col in metrics:
        item = {
            "metric_key": key,
            "metric_name": name,
            "business_value": round(_safe_float(trow.get(col)), 2),
            "sector_average": round(_safe_float(sector_prof[col].mean()), 2),
            "top_sector_value": round(_safe_float(sector_prof[col].max()), 2),
        }
        item["formatted_text"], item["gpt_insight"] = _build_kpi_text(item, target_name)
        kpi.append(item)

    sector_percentile = int(round(100 * (1 - ((rank - 1) / max(total, 1)))))
    payload = {
        "status": "success",
        "business_summary": {
            "business_name": target_name,
            "sector": str(trow["sector"]),
            "sector_rank": rank,
            "sector_percentile": sector_percentile,
            "business_score": round(_safe_float(trow["success_score"]), 4),
            "total_sector_businesses": total,
            "summary_text": (
                f"{target_name} ترتيبه {rank} من أصل {total}، وبقع تقريبًا عند المئين {sector_percentile}، "
                f"وأوضح نقطة قوة عنده هي {max(kpi, key=lambda x: _safe_float(x['business_value']) - _safe_float(x['sector_average']))['metric_name']}."
            ),
        },
        "sector_ranking": [{"rank": i + 1, "business_name": str(r["business_name"]), "success_score": round(_safe_float(r["success_score"]), 4)} for i, (_, r) in enumerate(sector_prof.head(15).iterrows())],
        "radar_chart": {"labels": [m[1] for m in metrics], "business_values": [x["business_value"] for x in kpi], "sector_average_values": [x["sector_average"] for x in kpi]},
        "kpi_comparisons": kpi,
        "sector_insights": [],
    }
    payload["sector_insights"] = _build_sector_insights(payload["business_summary"], kpi)
    logger.info("Benchmark insight fallback usage: generated local benchmark dashboard copy.")
    return _ensure_benchmark_text_quality(payload)


class RuleGenerationRequest(BaseModel):
    json_path: Optional[str] = Field(default=None)

    @field_validator("json_path")
    @classmethod
    def validate_json_path(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        if not str(value).strip():
            raise ValueError("json_path cannot be empty.")
        return str(value).strip()


class SimilarBusinessRequest(BaseModel):
    json_path: str = Field(default="vanilla_kpi_dataset.json", min_length=1)

    @field_validator("json_path")
    @classmethod
    def validate_json_path(cls, value: str) -> str:
        return str(resolve_existing_json_path(value))


class BenchmarkDashboardRequest(BaseModel):
    json_path: str = Field(default="vanilla_kpi_dataset.json", min_length=1)

    @field_validator("json_path")
    @classmethod
    def validate_json_path(cls, value: str) -> str:
        return str(resolve_existing_json_path(value))


class NextPostRecommendationRequest(BaseModel):
    json_path: str = Field(default="vanilla_kpi_dataset.json", min_length=1)

    @field_validator("json_path")
    @classmethod
    def validate_json_path(cls, value: str) -> str:
        return str(resolve_existing_json_path(value))


class RecommendationItem(BaseModel):
    id: int
    type: str
    category: str
    title: str
    explanation: str
    priority_score: float
    expected_impact: int
    confidence_score: int
    action_type: str
    icon: str


class BenchmarkRecommendationItem(BaseModel):
    title: str
    explanation: str
    priority_score: float


class RuleGenerationResponse(BaseModel):
    status: str
    positive_rules_count: int
    negative_rules_count: int


class SimilarBusinessResponse(BaseModel):
    status: str
    hero_summary: Dict[str, Any]
    target_business: Dict[str, Any]
    similar_businesses: List[Dict[str, Any]]
    benchmark_recommendations: List[BenchmarkRecommendationItem]
    recommendation_cards: List[Dict[str, Any]]
    insight_highlights: List[str]


class BenchmarkDashboardResponse(BaseModel):
    status: str
    business_summary: Dict[str, Any]
    sector_ranking: List[Dict[str, Any]]
    radar_chart: Dict[str, Any]
    kpi_comparisons: List[Dict[str, Any]]
    sector_insights: List[str]


class NextPostRecommendationResponse(BaseModel):
    status: str
    summary: Dict[str, int]
    business_behavior_summary: Dict[str, Any]
    engagement_recommendations: List[RecommendationItem]


def raise_clean_http_error(error: Exception) -> None:
    if isinstance(error, FileNotFoundError):
        raise HTTPException(status_code=404, detail={"status": "error", "message": str(error)})
    if isinstance(error, ValueError):
        raise HTTPException(status_code=400, detail={"status": "error", "message": str(error)})
    raise HTTPException(status_code=500, detail={"status": "error", "message": "Unexpected backend error.", "details": str(error)})


@router.post("/generate-rules-single", response_model=RuleGenerationResponse)
def generate_rules_single_endpoint(request: RuleGenerationRequest = RuleGenerationRequest()) -> RuleGenerationResponse:
    try:
        payload = _cached_or_generate_all_rules(request.json_path, save_rules=True)
        return RuleGenerationResponse(**payload)
    except PermissionError:
        payload = _cached_or_generate_all_rules(request.json_path, save_rules=False)
        return RuleGenerationResponse(**payload)
    except Exception as error:
        raise_clean_http_error(error)


@router.post("/similar-business-recommendations-single", response_model=SimilarBusinessResponse)
def similar_business_recommendations_single_endpoint(request: SimilarBusinessRequest) -> SimilarBusinessResponse:
    try:
        payload = _ext_run_similar_business_pipeline(json_path=request.json_path) if _EXTERNAL_PIPELINES_OK else _fallback_similar_pipeline(request.json_path)
        payload = _ensure_similar_business_recommendations_arabic(payload)
        return SimilarBusinessResponse(**payload)
    except Exception as error:
        raise_clean_http_error(error)


@router.post("/benchmark-dashboard-single", response_model=BenchmarkDashboardResponse)
def benchmark_dashboard_single_endpoint(request: BenchmarkDashboardRequest) -> BenchmarkDashboardResponse:
    try:
        payload = _ext_run_benchmark_dashboard_pipeline(json_path=request.json_path) if _EXTERNAL_PIPELINES_OK else _fallback_benchmark_pipeline(request.json_path)
        payload = _ensure_benchmark_text_quality(payload)
        return BenchmarkDashboardResponse(**payload)
    except Exception as error:
        if _EXTERNAL_PIPELINES_OK:
            logger.warning("Benchmark GPT/external generation failed; using local fallback if possible: %s", error)
            try:
                payload = _fallback_benchmark_pipeline(request.json_path)
                return BenchmarkDashboardResponse(**payload)
            except Exception:
                pass
        raise_clean_http_error(error)


@router.post("/next-post-recommendations-single", response_model=NextPostRecommendationResponse)
def next_post_recommendations_single_endpoint(request: NextPostRecommendationRequest) -> NextPostRecommendationResponse:
    try:
        payload = _ext_run_next_post_recommendation_pipeline(request.json_path) if _EXTERNAL_PIPELINES_OK else _fallback_next_post_pipeline(request.json_path)
        payload = _ensure_next_post_recommendations_arabic(payload)
        return NextPostRecommendationResponse(**payload)
    except Exception as error:
        raise_clean_http_error(error)
