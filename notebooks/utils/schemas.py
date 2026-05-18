"""
schemas.py

Purpose
-------
Defines Pydantic models for the recommendation and dashboard backend APIs.

Input
-----
- Request JSON for rule generation, KNN benchmarking, benchmark analytics, and
  next-post guidance

Output
------
- Structured response models documented in Swagger

Pipeline Role
-------------
Used by api_router.py for request validation and response documentation only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Path helper: Accept direct paths, project-relative paths, or files in data/.
def resolve_existing_json_path(json_path: str) -> Path:
    """Resolve a request path to an existing JSON file."""
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


# Request model: Optional body for regenerating rules from the processed dataset.
class RuleGenerationRequest(BaseModel):
    """Request body for POST /generate-rules."""

    json_path: Optional[str] = Field(default=None)

    @field_validator("json_path")
    @classmethod
    def validate_json_path(cls, value: Optional[str]) -> Optional[str]:
        """Validate an optional custom rule dataset path."""
        if value is None:
            return value
        if not str(value).strip():
            raise ValueError("json_path cannot be empty.")
        return str(value).strip()


# Request model: Uploaded single-business dataset used as the KNN target.
class SimilarBusinessRequest(BaseModel):
    """Request body for POST /similar-business-recommendations."""

    json_path: str = Field(default="vanilla_kpi_dataset.json", min_length=1)

    @field_validator("json_path")
    @classmethod
    def validate_json_path(cls, value: str) -> str:
        """Ensure the full business dataset path exists."""
        return str(resolve_existing_json_path(value))


# Request model: Uploaded single-business dataset used for benchmark analytics.
class BenchmarkDashboardRequest(BaseModel):
    """Request body for POST /benchmark-dashboard."""

    json_path: str = Field(default="vanilla_kpi_dataset.json", min_length=1)

    @field_validator("json_path")
    @classmethod
    def validate_json_path(cls, value: str) -> str:
        """Ensure the benchmark business dataset path exists."""
        return str(resolve_existing_json_path(value))


# Request model: Single-business dataset for next-post recommendations.
class NextPostRecommendationRequest(BaseModel):
    """Request body for POST /next-post-recommendations."""

    json_path: str = Field(default="vanilla_kpi_dataset.json", min_length=1)

    @field_validator("json_path")
    @classmethod
    def validate_json_path(cls, value: str) -> str:
        """Ensure the single-business dataset path exists."""
        return str(resolve_existing_json_path(value))


# Item model: Keep recommendation responses simple and frontend-safe.
class RecommendationItem(BaseModel):
    """Single recommendation item returned to the frontend."""

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


# Item model: Benchmark recommendations omit internal recommendation type.
class BenchmarkRecommendationItem(BaseModel):
    """Single benchmark recommendation returned to the frontend."""

    title: str
    explanation: str
    priority_score: float


# Response model: Metadata returned by rule generation only.
class RuleGenerationResponse(BaseModel):
    """Response for generated Apriori rule counts."""

    status: str
    positive_rules_count: int
    negative_rules_count: int


# Response model: KNN peer details and benchmark recommendations.
class SimilarBusinessResponse(BaseModel):
    """Response for similar-business benchmarking."""

    status: str
    hero_summary: Dict[str, Any]
    target_business: Dict[str, Any]
    similar_businesses: List[Dict[str, Any]]
    benchmark_recommendations: List[BenchmarkRecommendationItem]
    recommendation_cards: List[Dict[str, Any]]
    insight_highlights: List[str]


# Response model: Analytics payload for the benchmark dashboard.
class BenchmarkDashboardResponse(BaseModel):
    """Response for benchmark and sector comparison analytics."""

    status: str
    business_summary: Dict[str, Any]
    sector_ranking: List[Dict[str, Any]]
    radar_chart: Dict[str, Any]
    kpi_comparisons: List[Dict[str, Any]]
    sector_insights: List[str]


# Response model: Next-post recommendation output only.
class NextPostRecommendationResponse(BaseModel):
    """Response for next-post engagement recommendations."""

    status: str
    summary: Dict[str, int]
    business_behavior_summary: Dict[str, Any]
    engagement_recommendations: List[RecommendationItem]
