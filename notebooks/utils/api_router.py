"""
api_router.py

Purpose
-------
Defines the independent FastAPI recommendation and dashboard endpoints.

Input
-----
- Validated request models from schemas.py

Output
------
- Structured JSON responses for rule generation, KNN benchmarking, benchmark
  dashboard analytics, and next-post recommendations

Pipeline Role
-------------
Keeps endpoint logic thin: receive request, call the correct adapter pipeline,
and return clean responses.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

if __package__:
    from .frontend_recommender_adapter import (
        run_benchmark_dashboard_pipeline,
        run_next_post_recommendation_pipeline,
        run_rule_generation_pipeline,
        run_similar_business_pipeline,
    )
    from .schemas import (
        BenchmarkDashboardRequest,
        BenchmarkDashboardResponse,
        NextPostRecommendationRequest,
        NextPostRecommendationResponse,
        RuleGenerationRequest,
        RuleGenerationResponse,
        SimilarBusinessRequest,
        SimilarBusinessResponse,
    )
else:
    from frontend_recommender_adapter import (
        run_benchmark_dashboard_pipeline,
        run_next_post_recommendation_pipeline,
        run_rule_generation_pipeline,
        run_similar_business_pipeline,
    )
    from schemas import (
        BenchmarkDashboardRequest,
        BenchmarkDashboardResponse,
        NextPostRecommendationRequest,
        NextPostRecommendationResponse,
        RuleGenerationRequest,
        RuleGenerationResponse,
        SimilarBusinessRequest,
        SimilarBusinessResponse,
    )


router = APIRouter()


# Error helper: Convert service exceptions into clean API responses.
def raise_clean_http_error(error: Exception) -> None:
    """Raise an HTTPException with a frontend-safe error payload."""
    if isinstance(error, FileNotFoundError):
        raise HTTPException(status_code=404, detail={"status": "error", "message": str(error)})
    if isinstance(error, ValueError):
        raise HTTPException(status_code=400, detail={"status": "error", "message": str(error)})

    raise HTTPException(
        status_code=500,
        detail={
            "status": "error",
            "message": "Unexpected backend error.",
            "details": str(error),
        },
    )


# Endpoint 1: Generate and save Apriori rules from the processed dataset.
@router.post("/generate-rules", response_model=RuleGenerationResponse)
def generate_rules_endpoint(
    request: RuleGenerationRequest = RuleGenerationRequest(),
) -> RuleGenerationResponse:
    """Generate positive and negative Apriori rules only."""
    try:
        response_payload = run_rule_generation_pipeline(request.json_path)
        return RuleGenerationResponse(**response_payload)
    except Exception as error:
        raise_clean_http_error(error)


# Endpoint 2: Run only KNN similar-business benchmarking.
@router.post("/similar-business-recommendations", response_model=SimilarBusinessResponse)
def similar_business_recommendations_endpoint(
    request: SimilarBusinessRequest,
) -> SimilarBusinessResponse:
    """Generate similar-business benchmark recommendations."""
    try:
        response_payload = run_similar_business_pipeline(
            json_path=request.json_path,
        )
        return SimilarBusinessResponse(**response_payload)
    except Exception as error:
        raise_clean_http_error(error)


# Endpoint 3: Run benchmark dashboard analytics.
@router.post("/benchmark-dashboard", response_model=BenchmarkDashboardResponse)
def benchmark_dashboard_endpoint(
    request: BenchmarkDashboardRequest,
) -> BenchmarkDashboardResponse:
    """Generate sector benchmark dashboard analytics."""
    try:
        response_payload = run_benchmark_dashboard_pipeline(
            json_path=request.json_path,
        )
        return BenchmarkDashboardResponse(**response_payload)
    except Exception as error:
        raise_clean_http_error(error)


# Endpoint 4: Run only next-post engagement recommendations.
@router.post("/next-post-recommendations", response_model=NextPostRecommendationResponse)
def next_post_recommendations_endpoint(
    request: NextPostRecommendationRequest,
) -> NextPostRecommendationResponse:
    """Generate next-post recommendations for a single business dataset."""
    try:
        response_payload = run_next_post_recommendation_pipeline(request.json_path)
        return NextPostRecommendationResponse(**response_payload)
    except Exception as error:
        raise_clean_http_error(error)
