from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.schemas import (
    BusinessClusterComparison,
    BusinessClusteringResponse,
    MetricComparison,
)

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from notebooks.clustering.business_clustring_code import (
    preprocess_business_dataset,
    compare_business_pure_inference,
)

_CLUSTER_JSON_DEFAULT = str(
    _project_root
    / "notebooks"
    / "clustering"
    / "artifacts"
    / "business_clustering"
    / "json"
    / "business_cluster_coordinates.json"
)

router = APIRouter(prefix="/business-clustering")


@router.get("/analyze", response_model=BusinessClusteringResponse)
def analyze_business_clustering(
    file_path: str = Query(
        default="data/processed/kpi_dataset.json",
        description="Path to the KPI dataset file (.json or .csv).",
    ),
    business_name: Optional[str] = Query(
        default=None,
        description="Filter to a specific business. If omitted, all businesses are analyzed.",
    ),
    cluster_json_path: str = Query(
        default=_CLUSTER_JSON_DEFAULT,
        description="Path to the precomputed business cluster coordinates JSON.",
    ),
) -> BusinessClusteringResponse:
    full_path = _project_root / file_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {full_path}")

    cluster_json = Path(cluster_json_path)
    if not cluster_json.exists():
        raise HTTPException(
            status_code=404, detail=f"Cluster JSON not found: {cluster_json}"
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
                detail=f"Unsupported file type '{suffix}'. Use .json or .csv.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to parse file: {e}"
        )

    try:
        biz_features = preprocess_business_dataset(df)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Preprocessing failed: {e}"
        )

    if biz_features.empty:
        raise HTTPException(status_code=400, detail="No businesses found after preprocessing.")

    all_biz_names = sorted(biz_features["business_name"].unique().tolist())
    target_names = (
        [business_name] if business_name else all_biz_names
    )

    results: List[BusinessClusterComparison] = []
    for biz in target_names:
        norm_biz = str(biz).strip()
        try:
            comparisons_df = compare_business_pure_inference(
                biz_features=biz_features,
                business_name=norm_biz,
                cluster_json_path=str(cluster_json),
            )
        except ValueError as e:
            results.append(
                BusinessClusterComparison(
                    business_name=norm_biz,
                    kmeans_cluster=-1,
                    metrics=[],
                    summary=f"Error: {e}",
                )
            )
            continue
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Comparison failed for '{norm_biz}': {e}",
            )

        assigned_cluster = int(comparisons_df["kmeans_cluster"].iloc[0])
        metrics = [
            MetricComparison(
                feature=str(row["feature"]),
                business_value=float(row["business_value"]),
                cluster_avg=float(row["cluster_avg"]),
                comparison=str(row["comparison"]),
            )
            for _, row in comparisons_df.iterrows()
        ]

        results.append(
            BusinessClusterComparison(
                business_name=norm_biz,
                kmeans_cluster=assigned_cluster,
                metrics=metrics,
            )
        )

    if not results:
        raise HTTPException(status_code=404, detail="No matching businesses found.")

    return BusinessClusteringResponse(
        file_path=file_path,
        total_businesses=len(results),
        businesses=results,
    )
