from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.services.shared_utils import (
    GLOBAL_KPI_PATH,
    NOTEBOOKS_SCRIPTS,
    PROJECT_ROOT,
    load_module,
    load_uploaded_dataset,
    safe_identifier,
    validate_required_columns,
)


router = APIRouter(prefix="/api", tags=["business-momentum-single"])

BUSINESS_MOMENTUM_OUTPUT_ROOT = (
    PROJECT_ROOT / "backend" / "storage" / "outputs" / "business_momentum_single" / "business_momentum"
).resolve()


class UploadedDatasetRequest(BaseModel):
    uploaded_file_path: str


def _business_momentum_output_dir(business_name: str) -> Path:
    out = (BUSINESS_MOMENTUM_OUTPUT_ROOT / safe_identifier(business_name)).resolve()
    root = BUSINESS_MOMENTUM_OUTPUT_ROOT.resolve()
    if root not in out.parents and out != root:
        raise ValueError("Unsafe business momentum output directory path.")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_business_momentum_module():
    path = NOTEBOOKS_SCRIPTS / "04_business_momentum_weekly_trends.py"
    try:
        return load_module("business_momentum_module", path)
    except Exception:
        return None


def _first_clean_value(df: pd.DataFrame, column: str, fallback: str) -> str:
    if column not in df.columns:
        return fallback
    values = df[column].dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return fallback
    mode = values.mode()
    return str(mode.iloc[0] if not mode.empty else values.iloc[0])


def _resolve_uploaded_business_and_sector(uploaded_df: pd.DataFrame, global_df: pd.DataFrame | None = None) -> Tuple[str, str]:
    business_name = _first_clean_value(uploaded_df, "business_name", "Uploaded Business")
    uploaded_sector = _first_clean_value(uploaded_df, "sector", "unknown")
    if global_df is None or "sector" not in global_df.columns:
        return business_name, uploaded_sector

    global_sectors = global_df["sector"].dropna().astype(str).str.strip()
    if uploaded_sector != "unknown" and global_sectors.str.casefold().eq(uploaded_sector.casefold()).any():
        return business_name, uploaded_sector

    if "business_name" in global_df.columns:
        same_business = global_df[
            global_df["business_name"].astype(str).str.strip().str.casefold().eq(business_name.casefold())
        ]
        canonical = same_business["sector"].dropna().astype(str).str.strip()
        canonical = canonical[canonical != ""]
        if not canonical.empty:
            mode = canonical.mode()
            return business_name, str(mode.iloc[0] if not mode.empty else canonical.iloc[0])

    return business_name, uploaded_sector


def _ensure_engagement_rate(frame: pd.DataFrame) -> None:
    if "engagement_rate" in frame.columns:
        frame["engagement_rate"] = pd.to_numeric(frame["engagement_rate"], errors="coerce").fillna(0)
        return

    likes = pd.to_numeric(frame.get("likes_count", 0), errors="coerce").fillna(0)
    comments = pd.to_numeric(frame.get("comments_count", 0), errors="coerce").fillna(0)
    followers = pd.to_numeric(frame.get("followers_count", 0), errors="coerce").replace(0, np.nan)
    frame["engagement_rate"] = ((likes + comments) / followers).replace([np.inf, -np.inf], np.nan).fillna(0)


def _generate_business_vs_sector_chart(
    merged: pd.DataFrame,
    output_dir: Path,
    business_name: str,
    sector: str,
) -> Path:
    if merged.empty:
        raise ValueError("No weekly momentum rows are available to chart.")
    missing = [c for c in ["week", "business_engagement_rate", "sector_engagement_rate"] if c not in merged.columns]
    if missing:
        raise ValueError(f"Momentum chart data is missing columns: {', '.join(missing)}")

    chart_df = merged.copy()
    chart_df["week"] = pd.to_datetime(chart_df["week"], errors="coerce")
    chart_df["business_engagement_rate"] = pd.to_numeric(chart_df["business_engagement_rate"], errors="coerce")
    chart_df["sector_engagement_rate"] = pd.to_numeric(chart_df["sector_engagement_rate"], errors="coerce")
    chart_df = chart_df.dropna(subset=["week"]).sort_values("week")
    if chart_df.empty:
        raise ValueError("Momentum chart data has no valid week values.")

    img_path = output_dir / "business_vs_sector_momentum.png"
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(
        chart_df["week"],
        chart_df["business_engagement_rate"],
        marker="o",
        linewidth=2.4,
        markersize=5,
        label=f"{business_name} Business",
        color="#2563eb",
    )
    ax.plot(
        chart_df["week"],
        chart_df["sector_engagement_rate"],
        marker="o",
        linewidth=2.4,
        markersize=5,
        label=f"{sector} Sector",
        color="#f97316",
    )
    ax.set_title(f"{business_name} vs {sector} - Weekly Engagement Rate", fontsize=15, fontweight="bold", pad=16)
    ax.set_xlabel("Week", fontsize=11)
    ax.set_ylabel("Engagement Rate", fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.55)
    ax.legend(frameon=True, loc="best")
    fig.autofmt_xdate(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(img_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return img_path


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date_col = None
    for candidate in ["post_date", "date", "created_at", "timestamp"]:
        if candidate in out.columns:
            date_col = candidate
            break
    if date_col is None:
        raise ValueError("Dataset must contain one date column: post_date/date/created_at/timestamp.")
    numeric_dates = pd.to_numeric(out[date_col], errors="coerce")
    if numeric_dates.notna().sum() > 0:
        out["post_date"] = pd.NaT
        for unit in ("ms", "s"):
            parsed = pd.to_datetime(numeric_dates, errors="coerce", unit=unit)
            plausible = parsed.dt.year.ge(1990).fillna(False)
            if plausible.sum() > 0:
                out["post_date"] = parsed
                break
    else:
        out["post_date"] = pd.to_datetime(out[date_col], errors="coerce")
    if out["post_date"].notna().sum() == 0 and "week" in out.columns:
        week_start = out["week"].astype(str).str.split("/", n=1).str[0]
        out["post_date"] = pd.to_datetime(week_start, errors="coerce")
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
        _ensure_engagement_rate(frame)

    business_name, sector = _resolve_uploaded_business_and_sector(uploaded_df, global_df)
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

    chart_path = None
    chart_warning = ""
    try:
        chart_path = _generate_business_vs_sector_chart(merged, output_dir, business_name, sector)
    except Exception as exc:
        chart_path = None
        chart_warning = f"Chart generation failed: {type(exc).__name__}: {exc}"

    return {"ui_business_momentum_summary": ui_summary, "chart_path": chart_path, "chart_warning": chart_warning}


def _load_global_df_for_momentum() -> pd.DataFrame:
    candidates = [
        "data/vanilla.json",
        str(GLOBAL_KPI_PATH),
        "data/vanilla_kpi_dataset.json",
        "data/processed/kpi_dataset.json",
        "data/data_processed.json",
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

    global_df = _load_global_df_for_momentum()
    business_name, sector = _resolve_uploaded_business_and_sector(uploaded_df, global_df)
    output_dir = _business_momentum_output_dir(business_name)
    run_ui = getattr(momentum_module, "run_ui_business_vs_sector_analysis", None) if momentum_module is not None else None
    if callable(run_ui):
        result = run_ui(global_df, uploaded_df, output_dir, rolling_window=3, growth_threshold=0.10)
    else:
        result = _compute_ui_business_vs_sector_analysis(global_df, uploaded_df, output_dir, rolling_window=3, growth_threshold=0.10)

    latest_summary = result["ui_business_momentum_summary"].iloc[-1]
    chart_path = result.get("chart_path")
    chart_warning = str(result.get("chart_warning") or "")
    if not chart_path or not Path(chart_path).exists():
        try:
            merged = pd.read_csv(output_dir / "business_vs_sector_momentum.csv")
            chart_path = _generate_business_vs_sector_chart(merged, output_dir, business_name, sector)
            chart_warning = ""
        except Exception as exc:
            chart_path = None
            chart_warning = f"Chart generation failed: {type(exc).__name__}: {exc}"
    csv_outputs = [
        str(output_dir / "sector_weekly_trends.csv"),
        str(output_dir / "uploaded_business_weekly_trends.csv"),
        str(output_dir / "business_vs_sector_momentum.csv"),
        str(output_dir / "ui_business_momentum_summary.csv"),
    ]

    # Construct a browser-accessible chart URL if the image exists
    img_file = output_dir / "business_vs_sector_momentum.png"
    chart_url = f"/business_momentum_outputs/{safe_identifier(business_name)}/business_vs_sector_momentum.png" if img_file.exists() else ""

    response = {
        "business_name": business_name,
        "sector": sector or str(latest_summary["sector"]),
        "trend": str(latest_summary["trend"]),
        "comparison_label": str(latest_summary["comparison_label"]),
        "latest_business_engagement_rate": float(latest_summary["latest_business_engagement_rate"]),
        "latest_sector_engagement_rate": float(latest_summary["latest_sector_engagement_rate"]),
        "difference_from_sector": float(latest_summary["difference_from_sector"]),
        "message": str(latest_summary["recommendation_message"]),
        "chart_url": chart_url,
        "csv_outputs": csv_outputs,
    }
    if chart_warning:
        response["warning"] = chart_warning
    return response


def run_business_momentum_status(business_name: str) -> Dict[str, Any]:
    result_candidates = [
        BUSINESS_MOMENTUM_OUTPUT_ROOT / safe_identifier(business_name) / "business_vs_sector_momentum.csv",
        PROJECT_ROOT
        / "storage"
        / "outputs"
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
    sector = str(latest.get("sector", "") or "")
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
        "sector": sector,
        "trend": trend,
        "comparison_label": comparison_label,
        "latest_business_engagement_rate": float(latest.get("rolling_business", 0.0)),
        "latest_sector_engagement_rate": float(latest.get("sector_engagement_rate", 0.0)),
        "difference_from_sector": diff,
        "message": message,
        "chart_url": f"/business_momentum_outputs/{safe_identifier(business_name)}/business_vs_sector_momentum.png"
        if (BUSINESS_MOMENTUM_OUTPUT_ROOT / safe_identifier(business_name) / "business_vs_sector_momentum.png").exists()
        else "",
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
