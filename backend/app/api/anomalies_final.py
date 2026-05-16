from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import math
import pandas as pd
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel


router = APIRouter(prefix="/api", tags=["anomalies-single"])


class UploadedDatasetRequest(BaseModel):
    uploaded_file_path: str


def _resolve_dataset_path(uploaded_file_path: str) -> Path:
    p = Path(uploaded_file_path)
    if p.is_absolute() and p.exists():
        return p.resolve()

    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / uploaded_file_path,
        repo_root / "data" / uploaded_file_path,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(f"Uploaded file not found: {uploaded_file_path}")


def _load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported dataset format: {suffix}")


def _as_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []

    def clean(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if pd.isna(value):
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value

    return [{key: clean(value) for key, value in row.items()} for row in df.to_dict(orient="records")]


@router.post("/anomalies/analyze-single", status_code=status.HTTP_200_OK)
def analyze_anomalies_single(request: UploadedDatasetRequest) -> Dict[str, Any]:
    try:
        path = _resolve_dataset_path(request.uploaded_file_path)
        df = _load_dataset(path)

        if "engagement_rate" not in df.columns:
            raise ValueError("Dataset must include 'engagement_rate'.")

        for col in ["business_name", "sector", "post_type", "caption_text"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

        df["engagement_rate"] = pd.to_numeric(df["engagement_rate"], errors="coerce")
        df = df.dropna(subset=["engagement_rate"]).copy()
        if df.empty:
            return {
                "business_name": "",
                "sector": "",
                "message": "Anomaly detection finished but no valid engagement_rate values were found.",
                "top_positive_anomalies": [],
                "top_negative_anomalies": [],
                "recommendations": [],
                "sector_anomaly_summary": [],
                "chart_url": None,
                "csv_outputs": [],
            }

        business_name = (
            str(df["business_name"].dropna().iloc[0]).strip()
            if "business_name" in df.columns and not df["business_name"].dropna().empty
            else "Uploaded Dataset"
        )
        sector = (
            str(df["sector"].dropna().iloc[0]).strip()
            if "sector" in df.columns and not df["sector"].dropna().empty
            else ""
        )

        top_pos = df.sort_values("engagement_rate", ascending=False).head(5).copy()
        top_neg = df.sort_values("engagement_rate", ascending=True).head(5).copy()
        top_pos["anomaly_type"] = "positive_anomaly"
        top_neg["anomaly_type"] = "negative_anomaly"
        top_pos["best_method"] = "rank"
        top_neg["best_method"] = "rank"
        top_pos["best_setting"] = "top_5"
        top_neg["best_setting"] = "bottom_5"

        keep_cols = [
            "business_name",
            "sector",
            "post_date",
            "post_type",
            "caption_text",
            "engagement_rate",
            "views_count",
            "likes_count",
            "comments_count",
            "caption_length",
            "hashtags_count",
            "emoji_count",
            "promo_post",
            "anomaly_type",
            "best_method",
            "best_setting",
        ]
        top_pos = top_pos[[c for c in keep_cols if c in top_pos.columns]]
        top_neg = top_neg[[c for c in keep_cols if c in top_neg.columns]]

        return {
            "business_name": business_name,
            "sector": sector,
            "message": "Anomaly detection completed successfully.",
            "top_positive_anomalies": _as_records(top_pos),
            "top_negative_anomalies": _as_records(top_neg),
            "recommendations": [],
            "sector_anomaly_summary": [],
            "chart_url": None,
            "csv_outputs": [],
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@router.get("/anomalies/sector-summary-single", status_code=status.HTTP_200_OK)
def anomaly_sector_summary_single(sector: str) -> List[Dict[str, Any]]:
    _ = sector
    return []
