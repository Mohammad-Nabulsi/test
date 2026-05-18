from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.schemas import (
    BusinessClusterAnalysis,
    ClusterPerformance_low,
    ClusterPerformance_high,
    ContentStyleClusteringResponse,
)


_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from predict_content_style_clustering import predict_content_style_clustering

router = APIRouter(prefix="/content-style-clustering")


@router.get("/analyze", response_model=ContentStyleClusteringResponse)
def analyze_content_style_clusters(
    file_path: str = Query(
        default="data/vanilla_kpi_dataset.json",
        description="Path to the KPI dataset file (.json or .csv).",
    ),
) -> ContentStyleClusteringResponse:
    full_path = _project_root / file_path
    if not full_path.exists():
        raise HTTPException(
            status_code=404, detail=f"File not found: {full_path}"
        )

    suffix = full_path.suffix.lower()
    try:
        if suffix == ".json":
            df = pd.read_json(full_path)
        elif suffix == ".csv":
            df = pd.read_csv(full_path)
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type '{suffix}'. "
                    "Please use a .json or .csv file."
                ),
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse {suffix or 'dataset'} file: {e}",
        )

    model_path = _project_root / "notebooks" / "clustering" / "artifacts" / "content_style_clustering" / "models" / "content_style_svm_best_pipeline.joblib"
    if not model_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found: {model_path}",
        )

    try:
        results: List[dict] = predict_content_style_clustering(df, model_path=model_path)
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required column(s): {e}",
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Clustering analysis failed: {e}"
        )

    unique_clusters = sorted(
        {c["cluster"] for r in results for c in r["all_clusters"]}
    )

    businesses: List[BusinessClusterAnalysis] = []
    for r in results:
        all_clusters = r["all_clusters"]
        low_c = r["low_performing"]["cluster"]
        high_c = r["high_performing"]["cluster"]

        high_cluster_post_count = next(
            (c["post_count"] for c in all_clusters if c["cluster"] == high_c),
            0,
        )
        low_cluster_post_count = next(
            (c["post_count"] for c in all_clusters if c["cluster"] == low_c),
            0,
        )

        def _get(cluster_dict: dict, key: str, default: Any = None) -> Any:
            val = cluster_dict.get(key, default)
            if isinstance(val, float) and default is not None:
                return val
            return val if val is not None else default

        def _build_meaning(cluster_dict: dict, label: str) -> str:
            sector = _get(cluster_dict, "dominant_sector", "Unknown")
            ptype = _get(cluster_dict, "dominant_post_type", "Unknown")
            eng = _get(cluster_dict, "avg_engagement_rate", 0.0)
            view = _get(cluster_dict, "avg_view_rate", 0.0)
            return (
                f"{label} — {sector}, {ptype}, "
                f"eng_rate={float(eng):.4f}, view_rate={float(view):.2f}"
            )

        low_perf = ClusterPerformance_low(
            cluster=low_c,
            avg_engagement_rate=_get(r["low_performing"], "avg_engagement_rate", 0.0),
            avg_view_rate=_get(r["low_performing"], "avg_view_rate", 0.0),
            low_cluster_post_count=low_cluster_post_count,
        )

        high_perf = ClusterPerformance_high(
            cluster=high_c,
            avg_engagement_rate=_get(r["high_performing"], "avg_engagement_rate", 0.0),
            avg_view_rate=_get(r["high_performing"], "avg_view_rate", 0.0),
            high_cluster_post_count=high_cluster_post_count,
        )

        low_cluster_stats = next(
            (c for c in all_clusters if c["cluster"] == low_c),
            r["low_performing"],
        )
        high_cluster_stats = next(
            (c for c in all_clusters if c["cluster"] == high_c),
            r["high_performing"],
        )
   

        businesses.append(
            BusinessClusterAnalysis(
                business_name=r["business_name"],
                low_performing=low_perf,
                high_performing=high_perf,
                recommendation=r["recommendation"],
            )
        )

    return ContentStyleClusteringResponse(
        file_path=file_path,
        total_rows=len(df),
        unique_clusters=unique_clusters,
        businesses=businesses,
    )
