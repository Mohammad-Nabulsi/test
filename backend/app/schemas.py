from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ValidationIssue(BaseModel):
    type: str
    message: str
    column: Optional[str] = None
    count: Optional[int] = None
    examples: Optional[List[Any]] = None


class ValidationReport(BaseModel):
    ok: bool
    dataset_rows: int = 0
    dataset_columns: int = 0
    missing_required_columns: List[str] = Field(default_factory=list)
    issues: List[ValidationIssue] = Field(default_factory=list)


class UploadResponse(BaseModel):
    dataset_id: str
    validation_report: ValidationReport


class PipelineStepStatus(BaseModel):
    step: str
    ok: bool
    message: str
    output_files: List[str] = Field(default_factory=list)


class PipelineSummary(BaseModel):
    dataset_id: str
    ok: bool
    message: str
    steps: List[PipelineStepStatus] = Field(default_factory=list)
    outputs_dir: str


class DashboardResponse(BaseModel):
    dataset_id: str
    data: Dict[str, Any]


class ClusterPerformance_low(BaseModel):
    cluster: int
    meaning: str = ""
    avg_engagement_rate: float
    avg_view_rate: float
    low_cluster_post_count: int


class ClusterPerformance_high(BaseModel):
    cluster: int
    meaning: str = ""
    avg_engagement_rate: float
    avg_view_rate: float
    high_cluster_post_count: int


class BusinessClusterAnalysis(BaseModel):
    business_name: str
    low_performing: ClusterPerformance_low
    high_performing: ClusterPerformance_high
    recommendation: str


class ContentStyleClusteringResponse(BaseModel):
    file_path: str
    total_rows: int
    unique_clusters: List[int]
    businesses: List[BusinessClusterAnalysis]


class MetricComparison(BaseModel):
    feature: str
    business_value: float
    cluster_avg: float
    comparison: str


class BusinessClusterComparison(BaseModel):
    business_name: str
    kmeans_cluster: int
    metrics: List[MetricComparison]


class BusinessClusteringResponse(BaseModel):
    file_path: str
    total_businesses: int
    businesses: List[BusinessClusterComparison]


class MeBusinessInfo(BaseModel):
    name: str
    nameEn: str
    handle: str
    category: str
    location: str
    followers: int
    followersGrowth: float
    posts: int
    avatarColor: str


class MeKPIItem(BaseModel):
    key: str
    label: str
    value: Union[float, int, str]
    suffix: Optional[str] = None
    delta: float
    spark: List[Union[float, int]]
    string: bool = False


class MeVisibleInfoResponse(BaseModel):
    business: MeBusinessInfo
    kpis: List[MeKPIItem]

