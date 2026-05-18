from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.config import settings
from app.schemas import DashboardResponse
from app.utils.file_utils import safe_read_csv

router = APIRouter(prefix="/recommendations")
logger = logging.getLogger(__name__)


def _outputs_dir(dataset_id: str) -> Path:
    return settings.storage_path() / "outputs" / dataset_id


def _read_csv_records(path: Path, max_rows: int = 5000):
    df = safe_read_csv(path)
    if len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_dict(orient="records")


@router.get("/{dataset_id}", response_model=DashboardResponse)
def get_recommendations(dataset_id: str) -> DashboardResponse:
    logger.info("get-recommendations started dataset_id=%s", dataset_id)
    out = _outputs_dir(dataset_id)
    recs_path = out / "recommendations.csv"
    if not recs_path.exists():
        logger.warning("get-recommendations not found dataset_id=%s path=%s", dataset_id, recs_path)
        raise HTTPException(status_code=404, detail="Recommendations not found. Run pipeline first.")

    response = DashboardResponse(
        dataset_id=dataset_id,
        data={"recommendations": _read_csv_records(recs_path)},
    )
    logger.info("get-recommendations completed dataset_id=%s", dataset_id)
    return response


@router.get("/{dataset_id}/rules", response_model=DashboardResponse)
def get_recommendation_rules(dataset_id: str) -> DashboardResponse:
    logger.info("get-recommendation-rules started dataset_id=%s", dataset_id)
    out = _outputs_dir(dataset_id)
    rules_path = out / "association_rules.csv"
    if not rules_path.exists():
        logger.warning("get-recommendation-rules not found dataset_id=%s path=%s", dataset_id, rules_path)
        raise HTTPException(status_code=404, detail="Association rules not found. Run pipeline first.")

    data = {
        "association_rules": _read_csv_records(rules_path),
    }

    bv_rules_path = out / "business_value_rules.csv"
    if bv_rules_path.exists():
        data["business_value_rules"] = _read_csv_records(bv_rules_path)

    response = DashboardResponse(dataset_id=dataset_id, data=data)
    logger.info(
        "get-recommendation-rules completed dataset_id=%s has_business_value_rules=%s",
        dataset_id,
        "business_value_rules" in data,
    )
    return response
