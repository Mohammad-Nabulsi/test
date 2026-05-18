from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.services.shared_utils import (
    GLOBAL_KPI_PATH,
    GLOBAL_OUTPUT_DIR,
    NOTEBOOKS_SCRIPTS,
    load_module,
    load_uploaded_dataset,
    safe_identifier,
    safe_output_directory,
    safe_path_string,
    validate_required_columns,
)


router = APIRouter(prefix="/api", tags=["business-momentum-single"])


class UploadedDatasetRequest(BaseModel):
    uploaded_file_path: str


def _load_business_momentum_module():
    path = NOTEBOOKS_SCRIPTS / "04_business_momentum_weekly_trends.py"
    try:
        return load_module("business_momentum_module", path)
    except Exception:
        return None


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date_col = None
    for candidate in ["post_date", "date", "created_at", "timestamp"]:
        if candidate in out.columns:
            date_col = candidate
            break
    if date_col is None:
        raise ValueError("Dataset must contain one date column: post_date/date/created_at/timestamp.")
    out["post_date"] = pd.to_datetime(out[date_col], errors="coerce")
    if out["post_date"].notna().sum() == 0 and "week" in out.columns:
        out["post_date"] = pd.to_datetime(out["week"], errors="coerce")
    out = out.dropna(subset=["post_date"]).copy()
    out["week"] = out["post_date"].dt.to_period("W").dt.start_time
    return out


def _compute_ui_business_vs_sector_analysis(
    global_df: pd.DataFrame,
    uploaded_df: pd.DataFrame,
    output_dir: Any,
    rolling_window: int,
    growth_threshold: float,
) -> Dict[str, Any]:
    gdf = _normalize_dates(global_df)
    udf = _normalize_dates(uploaded_df)
    if udf.empty:
        raise ValueError("Uploaded business dataset has no valid post_date values.")
    if gdf.empty:
        raise ValueError("Global KPI dataset has no valid post_date values.")

    for frame in (gdf, udf):
        if "engagement_rate" not in frame.columns:
            likes = pd.to_numeric(frame.get("likes_count", 0), errors="coerce").fillna(0)
            comments = pd.to_numeric(frame.get("comments_count", 0), errors="coerce").fillna(0)
            followers = pd.to_numeric(frame.get("followers_count", 0), errors="coerce").replace(0, np.nan)
            frame["engagement_rate"] = ((likes + comments) / followers).replace([np.inf, -np.inf], np.nan).fillna(0)
        else:
            frame["engagement_rate"] = pd.to_numeric(frame["engagement_rate"], errors="coerce").fillna(0)

    business_name_series = udf["business_name"].dropna().astype(str).str.strip()
    sector_series = udf["sector"].dropna().astype(str).str.strip()
    business_name = business_name_series.iloc[0] if not business_name_series.empty else "Uploaded Business"
    sector = sector_series.iloc[0] if not sector_series.empty else "unknown"
    sector_df = gdf[gdf["sector"].astype(str).str.lower() == sector.lower()].copy()
    if sector_df.empty:
        sector_df = gdf.copy()

    sector_weekly = sector_df.groupby("week", as_index=False)["engagement_rate"].mean()
    sector_weekly.rename(columns={"engagement_rate": "sector_engagement_rate"}, inplace=True)

    business_weekly = udf.groupby("week", as_index=False)["engagement_rate"].mean()
    business_weekly.rename(columns={"engagement_rate": "business_engagement_rate"}, inplace=True)
    business_weekly["business_name"] = business_name
    business_weekly["sector"] = sector

    merged = pd.merge(business_weekly, sector_weekly, on="week", how="left").sort_values("week")
    if merged.empty:
        raise ValueError("Not enough overlapping weekly data to compute momentum.")
    merged["rolling_business"] = merged["business_engagement_rate"].rolling(rolling_window, min_periods=1).mean()
    merged["growth"] = merged["rolling_business"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    merged["difference_from_sector"] = merged["rolling_business"] - merged["sector_engagement_rate"].fillna(0)

    latest = merged.tail(1).iloc[0]
    growth = float(latest["growth"])
    if growth > growth_threshold:
        trend = "improving"
    elif growth < -growth_threshold:
        trend = "declining"
    else:
        trend = "stable"

    diff = float(latest["difference_from_sector"])
    if diff > 0:
        comparison_label = "above sector average"
    elif diff < 0:
        comparison_label = "below sector average"
    else:
        comparison_label = "equal to sector average"

    ui_summary = pd.DataFrame(
        [
            {
                "business_name": business_name,
                "sector": sector,
                "trend": trend,
                "comparison_label": comparison_label,
                "latest_business_engagement_rate": float(latest["rolling_business"]),
                "latest_sector_engagement_rate": float(latest["sector_engagement_rate"]) if pd.notna(latest["sector_engagement_rate"]) else 0.0,
                "difference_from_sector": diff,
                "recommendation_message": f"{business_name} is {comparison_label} with {trend} momentum.",
            }
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    sector_weekly.to_csv(output_dir / "sector_weekly_trends.csv", index=False)
    business_weekly.to_csv(output_dir / "uploaded_business_weekly_trends.csv", index=False)
    merged.to_csv(output_dir / "business_vs_sector_momentum.csv", index=False)
    ui_summary.to_csv(output_dir / "ui_business_momentum_summary.csv", index=False)

    return {"ui_business_momentum_summary": ui_summary, "chart_path": None}


def _load_global_df_for_momentum() -> pd.DataFrame:
    candidates = [
        "data/vanilla.json",
        str(GLOBAL_KPI_PATH),
    ]
    last_exc: Exception | None = None
    for c in candidates:
        try:
            df = load_uploaded_dataset(c)
            norm = _normalize_dates(df)
            if not norm.empty:
                return df
        except Exception as exc:
            last_exc = exc
            continue
    raise ValueError(f"No usable global dataset with valid dates. Last error: {last_exc}")


def run_business_momentum_pipeline(uploaded_file_path: str) -> Dict[str, Any]:
    momentum_module = _load_business_momentum_module()
    uploaded_df = load_uploaded_dataset(uploaded_file_path)
    required_columns = ["business_name", "sector"]
    validate_required_columns(uploaded_df, required_columns, "uploaded business dataset")
    if uploaded_df.empty:
        raise ValueError("Uploaded business dataset is empty.")

    business_name_series = uploaded_df["business_name"].dropna().astype(str).str.strip()
    business_name = business_name_series.iloc[0] if not business_name_series.empty else "Uploaded Business"
    try:
        output_dir = safe_output_directory(f"business_momentum/{safe_identifier(business_name)}")
    except PermissionError:
        output_dir = (
            GLOBAL_OUTPUT_DIR.parents[2]
            / "backend"
            / "_runtime_outputs"
            / "business_momentum_single"
            / "business_momentum"
            / safe_identifier(business_name)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
    global_df = _load_global_df_for_momentum()
    run_ui = getattr(momentum_module, "run_ui_business_vs_sector_analysis", None) if momentum_module is not None else None
    if callable(run_ui):
        result = run_ui(global_df, uploaded_df, output_dir, rolling_window=3, growth_threshold=0.10)
    else:
        result = _compute_ui_business_vs_sector_analysis(global_df, uploaded_df, output_dir, rolling_window=3, growth_threshold=0.10)

    latest_summary = result["ui_business_momentum_summary"].iloc[-1]
    chart_path = result.get("chart_path")
    csv_outputs = [
        str(output_dir / "sector_weekly_trends.csv"),
        str(output_dir / "uploaded_business_weekly_trends.csv"),
        str(output_dir / "business_vs_sector_momentum.csv"),
        str(output_dir / "ui_business_momentum_summary.csv"),
    ]

    return {
        "business_name": business_name,
        "sector": str(latest_summary["sector"]),
        "trend": str(latest_summary["trend"]),
        "comparison_label": str(latest_summary["comparison_label"]),
        "latest_business_engagement_rate": float(latest_summary["latest_business_engagement_rate"]),
        "latest_sector_engagement_rate": float(latest_summary["latest_sector_engagement_rate"]),
        "difference_from_sector": float(latest_summary["difference_from_sector"]),
        "message": str(latest_summary["recommendation_message"]),
        "chart_url": safe_path_string(chart_path) if chart_path is not None else "",
        "csv_outputs": csv_outputs,
    }


def run_business_momentum_status(business_name: str) -> Dict[str, Any]:
    result_candidates = [
        GLOBAL_OUTPUT_DIR / "business_momentum" / safe_identifier(business_name) / "business_vs_sector_momentum.csv",
        GLOBAL_OUTPUT_DIR.parents[2]
        / "backend"
        / "_runtime_outputs"
        / "business_momentum_single"
        / "business_momentum"
        / safe_identifier(business_name)
        / "business_vs_sector_momentum.csv",
    ]
    result_csv = next((p for p in result_candidates if p.exists()), None)
    if result_csv is None:
        raise ValueError(f"Business momentum data is not available for: {business_name}")
    comparison = pd.read_csv(result_csv)
    if comparison.empty:
        raise ValueError("Business momentum comparison data is empty.")

    latest = comparison.tail(1).iloc[-1]
    diff = float(latest.get("difference_from_sector", 0.0))
    if diff > 0:
        comparison_label = "above sector average"
    elif diff < 0:
        comparison_label = "below sector average"
    else:
        comparison_label = "equal to sector average"
    growth = float(latest.get("growth", 0.0))
    trend = "improving" if growth > 0.10 else "declining" if growth < -0.10 else "stable"
    message = f"{business_name} has {trend} momentum and is {comparison_label}."

    return {
        "business_name": business_name,
        "sector": "",
        "trend": trend,
        "comparison_label": comparison_label,
        "latest_business_engagement_rate": float(latest.get("rolling_business", 0.0)),
        "latest_sector_engagement_rate": float(latest.get("sector_engagement_rate", 0.0)),
        "difference_from_sector": diff,
        "message": message,
        "chart_url": "",
        "csv_outputs": [str(result_csv)],
    }


def get_sector_momentum() -> List[Dict[str, Any]]:
    df = _load_global_df_for_momentum()
    validate_required_columns(df, ["sector", "post_date"], "global KPI dataset")
    df = _normalize_dates(df)
    if "engagement_rate" not in df.columns:
        likes = pd.to_numeric(df.get("likes_count", 0), errors="coerce").fillna(0)
        comments = pd.to_numeric(df.get("comments_count", 0), errors="coerce").fillna(0)
        followers = pd.to_numeric(df.get("followers_count", 0), errors="coerce").replace(0, np.nan)
        df["engagement_rate"] = ((likes + comments) / followers).replace([np.inf, -np.inf], np.nan).fillna(0)
    out = df.groupby("sector", as_index=False)["engagement_rate"].mean().rename(columns={"engagement_rate": "avg_engagement_rate"})
    out["trend"] = "stable"
    return out.to_dict(orient="records")


@router.post("/business-momentum/analyze-single", status_code=status.HTTP_200_OK)
def business_momentum_analyze_single(request: UploadedDatasetRequest) -> Dict[str, Any]:
    try:
        return run_business_momentum_pipeline(request.uploaded_file_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@router.get("/business-momentum/status-single", status_code=status.HTTP_200_OK)
def business_momentum_status_single(business_name: str) -> Dict[str, Any]:
    try:
        return run_business_momentum_status(business_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@router.get("/business-momentum/sector-summary-single", status_code=status.HTTP_200_OK)
def business_momentum_sector_summary_single() -> List[Dict[str, Any]]:
    try:
        return get_sector_momentum()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
