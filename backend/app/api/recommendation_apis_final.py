from __future__ import annotations

import math
import pickle
import sys
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
POS_RULES = DATA_DIR / "positive_rules.pkl"
NEG_RULES = DATA_DIR / "negative_rules.pkl"


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
    threshold = float(work["engagement_rate"].quantile(0.6 if positive else 0.4))
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
    freq = apriori(tx_df, min_support=0.08, use_colnames=True)
    if freq.empty:
        return pd.DataFrame(columns=["antecedents", "confidence", "lift", "support"])
    rules = association_rules(freq, metric="confidence", min_threshold=0.35)
    target_item = f"{target_col}=True"
    rules = rules[rules["consequents"].apply(lambda x: target_item in x)].copy()
    if rules.empty:
        return pd.DataFrame(columns=["antecedents", "confidence", "lift", "support"])
    return rules[["antecedents", "confidence", "lift", "support"]].sort_values(
        ["confidence", "lift", "support"], ascending=False
    )


def _fallback_generate_all_rules(json_path: Optional[str], save_rules: bool = True) -> Dict[str, Any]:
    candidates = [json_path] if json_path else ["data_processed.json", "processed_data.json", "data/vanilla_kpi_dataset.json"]
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
    pos = _mine_rules(df, positive=True)
    neg = _mine_rules(df, positive=False)
    if save_rules:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        pos.to_pickle(POS_RULES)
        neg.to_pickle(NEG_RULES)
    return {"status": "success", "positive_rules_count": int(len(pos)), "negative_rules_count": int(len(neg))}


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
        recs.append(("Use a few more hashtags", "A small increase in relevant hashtags can improve discovery.", 78))
    if summary["avg_caption_length"] <= 50:
        recs.append(("Try slightly longer captions", "Longer captions can make the message clearer.", 68))
    if summary["CTA_present_rate"] < 0.35:
        recs.append(("Add clearer calls-to-action", "Clear calls-to-action can improve response.", 82))
    if summary["mentions_location_rate"] < 0.35:
        recs.append(("Mention location more often", "Location context can strengthen local reach.", 72))
    if summary["promo_post_rate"] > 0.55:
        recs.append(("Reduce heavy promotional tone", "Too much promotional content can lower engagement.", 74))
    if summary["pct_post_type_reel"] < 0.35:
        recs.append(("Try more reels", "Reels often help with reach and interaction.", 80))
    if not recs:
        recs.append(("Maintain your current strategy", "Your profile is balanced; iterate with small tests.", 50))
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


def _fallback_similar_pipeline(json_path: str) -> Dict[str, Any]:
    target_df = _load_df(json_path)
    missing = [c for c in SIMILAR_BUSINESS_REQUIRED_COLUMNS if c not in target_df.columns]
    if missing:
        raise ValueError("Missing required input columns: " + ", ".join(sorted(missing)))
    master = _load_df("data/vanilla_kpi_dataset.json")
    all_df = pd.concat([master, target_df], ignore_index=True)
    prof = _profiles(all_df)
    target_name = str(target_df["business_name"].iloc[0])
    target = prof[prof["business_name"].astype(str) == target_name]
    if target.empty:
        raise ValueError("Target business profile not found.")
    target_row = target.iloc[0]
    peers = prof[(prof["business_name"].astype(str) != target_name) & (prof["sector"].astype(str) == str(target_row["sector"]))].copy()
    if peers.empty:
        peers = prof[prof["business_name"].astype(str) != target_name].copy()
    peers = peers.sort_values("success_score", ascending=False).head(5)
    cards = [{
        "id": 1,
        "title": "Increase engagement signals",
        "explanation": "Peer businesses show stronger engagement-related patterns.",
        "expected_impact": 24,
        "confidence_score": 86,
        "priority_score": 72.0,
        "category": "engagement",
        "icon": "sparkles",
        "inspired_by": [str(x) for x in peers["business_name"].head(3)],
    }]
    return {
        "status": "success",
        "hero_summary": {
            "title": "Actions inspired by similar businesses",
            "total_recommendations": len(cards),
            "top_opportunity": cards[0]["title"],
            "estimated_best_impact": cards[0]["expected_impact"],
        },
        "target_business": {"business_name": target_name, "sector": str(target_row["sector"]), "success_score": float(target_row["success_score"])},
        "similar_businesses": [
            {"rank": i + 1, "business_name": str(r["business_name"]), "sector": str(r["sector"]), "similarity_score": 0.8 - i * 0.05, "success_score": float(r["success_score"])}
            for i, (_, r) in enumerate(peers.iterrows())
        ],
        "benchmark_recommendations": [{"title": cards[0]["title"], "explanation": cards[0]["explanation"], "priority_score": cards[0]["priority_score"]}],
        "recommendation_cards": cards,
        "insight_highlights": ["Built from nearest peer profiles."],
    }


def _fallback_benchmark_pipeline(json_path: str) -> Dict[str, Any]:
    target_df = _load_df(json_path)
    master = _load_df("data/vanilla_kpi_dataset.json")
    all_df = pd.concat([master, target_df], ignore_index=True)
    prof = _profiles(all_df).sort_values("success_score", ascending=False).reset_index(drop=True)
    target_name = str(target_df["business_name"].iloc[0])
    trows = prof[prof["business_name"].astype(str) == target_name]
    if trows.empty:
        raise ValueError("Target business profile not found.")
    trow = trows.iloc[0]
    rank = int(prof.index[prof["business_name"].astype(str) == target_name][0]) + 1
    total = len(prof)
    metrics = [
        ("engagement", "Engagement", "engagement_rate_followers"),
        ("reach", "Reach", "views_per_follower"),
        ("quality", "Quality", "avg_comments_count"),
        ("hashtags", "Hashtags", "avg_hashtags_count"),
    ]
    kpi = []
    for key, name, col in metrics:
        kpi.append({
            "metric_key": key,
            "metric_name": name,
            "business_value": round(_safe_float(trow.get(col)), 2),
            "sector_average": round(_safe_float(prof[col].mean()), 2),
            "top_sector_value": round(_safe_float(prof[col].max()), 2),
            "formatted_text": "",
            "gpt_insight": "Benchmark comparison generated.",
        })
    return {
        "status": "success",
        "business_summary": {
            "business_name": target_name,
            "sector": str(trow["sector"]),
            "sector_rank": rank,
            "sector_percentile": int(round(100 * (1 - ((rank - 1) / max(total, 1))))),
            "business_score": round(_safe_float(trow["success_score"]), 4),
            "total_sector_businesses": total,
            "summary_text": f"{target_name} is rank {rank} of {total}.",
        },
        "sector_ranking": [{"rank": i + 1, "business_name": str(r["business_name"]), "success_score": round(_safe_float(r["success_score"]), 4)} for i, (_, r) in enumerate(prof.head(10).iterrows())],
        "radar_chart": {"labels": [m[1] for m in metrics], "business_values": [x["business_value"] for x in kpi], "sector_average_values": [x["sector_average"] for x in kpi]},
        "kpi_comparisons": kpi,
        "sector_insights": ["Ranking and KPI gaps are computed from business profiles."],
    }


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
        if _EXTERNAL_PIPELINES_OK:
            payload = _ext_run_rule_generation_pipeline(request.json_path)
            return RuleGenerationResponse(**payload)
        payload = _fallback_generate_all_rules(request.json_path, save_rules=True)
        return RuleGenerationResponse(**payload)
    except PermissionError:
        payload = _fallback_generate_all_rules(request.json_path, save_rules=False)
        return RuleGenerationResponse(**payload)
    except Exception as error:
        raise_clean_http_error(error)


@router.post("/similar-business-recommendations-single", response_model=SimilarBusinessResponse)
def similar_business_recommendations_single_endpoint(request: SimilarBusinessRequest) -> SimilarBusinessResponse:
    try:
        payload = _ext_run_similar_business_pipeline(json_path=request.json_path) if _EXTERNAL_PIPELINES_OK else _fallback_similar_pipeline(request.json_path)
        return SimilarBusinessResponse(**payload)
    except Exception as error:
        raise_clean_http_error(error)


@router.post("/benchmark-dashboard-single", response_model=BenchmarkDashboardResponse)
def benchmark_dashboard_single_endpoint(request: BenchmarkDashboardRequest) -> BenchmarkDashboardResponse:
    try:
        payload = _ext_run_benchmark_dashboard_pipeline(json_path=request.json_path) if _EXTERNAL_PIPELINES_OK else _fallback_benchmark_pipeline(request.json_path)
        return BenchmarkDashboardResponse(**payload)
    except Exception as error:
        raise_clean_http_error(error)


@router.post("/next-post-recommendations-single", response_model=NextPostRecommendationResponse)
def next_post_recommendations_single_endpoint(request: NextPostRecommendationRequest) -> NextPostRecommendationResponse:
    try:
        payload = _ext_run_next_post_recommendation_pipeline(request.json_path) if _EXTERNAL_PIPELINES_OK else _fallback_next_post_pipeline(request.json_path)
        return NextPostRecommendationResponse(**payload)
    except Exception as error:
        raise_clean_http_error(error)
