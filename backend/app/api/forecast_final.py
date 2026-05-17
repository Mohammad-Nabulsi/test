from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel


router = APIRouter(prefix="/api", tags=["forecast-single"])


class UploadedDatasetRequest(BaseModel):
    uploaded_file_path: str


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _storage_root() -> Path:
    # Self-contained replacement for app.config.settings.storage_path()
    env_value = Path(__import__("os").getenv("STORAGE_DIR", "storage"))
    if env_value.is_absolute():
        return env_value
    return _project_root() / env_value


def _safe_output_name(input_path: str, fallback: str = "forecast_run") -> str:
    stem = Path(str(input_path)).stem
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_").lower()
    return safe or fallback


def _resolve_dataset_path(uploaded_file_path: str) -> Path:
    p = Path(uploaded_file_path)
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        _project_root() / uploaded_file_path,
        _project_root() / "data" / uploaded_file_path,
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


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _weekly_series(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    date_col = _pick_col(df, ["post_date", "date", "timestamp", "created_at"])
    y_col = _pick_col(df, ["engagement_rate", "engagement_rate_followers", "engagement"])
    biz_col = _pick_col(df, ["business_name", "business"])
    if not y_col:
        raise ValueError("Dataset must include an engagement metric.")

    frame = df.copy()
    if date_col:
        if pd.api.types.is_numeric_dtype(frame[date_col]):
            frame[date_col] = pd.to_datetime(frame[date_col], unit="ms", errors="coerce")
        else:
            frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    else:
        frame["__synthetic_date__"] = pd.NaT
        date_col = "__synthetic_date__"

    frame[y_col] = pd.to_numeric(frame[y_col], errors="coerce")
    if frame[date_col].isna().all():
        end = pd.Timestamp.utcnow().normalize()
        frame[date_col] = pd.date_range(end=end, periods=len(frame), freq="W-MON")

    frame = frame.dropna(subset=[date_col, y_col]).copy()
    if frame.empty:
        raise ValueError("No valid rows after parsing dates/metrics.")

    frame["week"] = frame[date_col].dt.to_period("W").dt.start_time
    weekly = frame.groupby("week", as_index=False)[y_col].mean().sort_values("week")
    weekly.rename(columns={"week": "ds", y_col: "y"}, inplace=True)

    business_name = "Uploaded Dataset"
    if biz_col and biz_col in frame.columns and not frame[biz_col].dropna().empty:
        business_name = str(frame[biz_col].dropna().astype(str).iloc[0]).strip() or business_name
    return weekly, business_name


def _ma_forecast(y: pd.Series, horizon: int) -> np.ndarray:
    window = int(min(4, max(2, len(y))))
    level = float(pd.to_numeric(y.tail(window), errors="coerce").mean())
    return np.full(horizon, level, dtype=float)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mae = float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else 0.0
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else 0.0
    return mae, rmse


def _run_forecasting_analysis(df: pd.DataFrame, output_dir: Path, periods: int, output_prefix: str) -> Dict[str, Any]:
    weekly, business_name = _weekly_series(df)
    y = weekly["y"].astype(float).reset_index(drop=True)

    split_idx = max(1, int(round(len(y) * 0.8)))
    train, test = y.iloc[:split_idx], y.iloc[split_idx:]
    if len(test) == 0:
        test = train.tail(min(2, len(train)))

    yhat_test = _ma_forecast(train, len(test))
    mae, rmse = _metrics(test.to_numpy(), yhat_test)

    future_pred = _ma_forecast(y, periods)
    last_date = pd.to_datetime(weekly["ds"].max())
    future_dates = pd.date_range(last_date + pd.Timedelta(days=7), periods=periods, freq="W-MON")

    hist_df = pd.DataFrame({"ds": pd.to_datetime(weekly["ds"]), "yhat": y, "forecast_type": "history"})
    fut_df = pd.DataFrame({"ds": future_dates, "yhat": future_pred, "forecast_type": "future"})
    forecast_df = pd.concat([hist_df, fut_df], ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_prefix}.csv"
    forecast_df.to_csv(csv_path, index=False)

    summary_df = pd.DataFrame([{
        "business_name": business_name,
        "best_model": "moving_average",
        "forecast_horizon_weeks": periods,
        "best_MAE": mae,
        "best_RMSE": rmse,
    }])
    return {"forecast_df": forecast_df, "summary_df": summary_df, "chart_path": ""}


def _format_future_forecast(forecast_df: pd.DataFrame) -> list[dict[str, Any]]:
    future = forecast_df[forecast_df["forecast_type"] == "future"].copy()
    future["date"] = pd.to_datetime(future["ds"]).dt.strftime("%Y-%m-%d")
    future["predicted_engagement_rate"] = future["yhat"].astype(float)
    return future[["date", "predicted_engagement_rate"]].to_dict(orient="records")


def _run_forecast_for_path(input_path: str) -> Dict[str, Any]:
    path = _resolve_dataset_path(input_path)
    df = _load_dataset(path)
    out_dir = _storage_root() / "outputs" / "forecast_single" / _safe_output_name(input_path)
    try:
        result = _run_forecasting_analysis(df, out_dir, periods=8, output_prefix="uploaded_weekly_forecast")
    except PermissionError:
        fallback = Path(__file__).resolve().parents[2] / "_runtime_outputs" / "forecast_single" / _safe_output_name(input_path)
        result = _run_forecasting_analysis(df, fallback, periods=8, output_prefix="uploaded_weekly_forecast")
    summary = result["summary_df"].iloc[0]
    return {
        "business_name": str(summary.get("business_name", "")),
        "best_model": str(summary.get("best_model", "")),
        "forecast_horizon_weeks": int(summary.get("forecast_horizon_weeks", 0)),
        "best_MAE": float(summary.get("best_MAE", 0.0)),
        "best_RMSE": float(summary.get("best_RMSE", 0.0)),
        "message": f"Forecast completed using {summary.get('best_model', '')} for {summary.get('business_name', '')}.",
        "future_forecast": _format_future_forecast(result["forecast_df"]),
        "chart_url": "",
    }


@router.post("/forecast/analyze-single", status_code=status.HTTP_200_OK)
def forecast_analyze_single(request: UploadedDatasetRequest) -> Dict[str, Any]:
    try:
        return _run_forecast_for_path(request.uploaded_file_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@router.get("/forecast/static-single", status_code=status.HTTP_200_OK)
def forecast_static_single(
    uploaded_file_path: str | None = Query(
        default=None,
        description="Optional dataset path. If provided, static-single runs on this file instead of built-in defaults.",
    )
) -> Dict[str, Any]:
    if uploaded_file_path:
        try:
            return _run_forecast_for_path(uploaded_file_path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    candidates = [
        "data/social_media_engagement_dataset.csv",
        "data/vanilla_kpi_dataset.json",
        "vanilla_kpi_dataset.json",
    ]
    last_err: Exception | None = None
    for c in candidates:
        try:
            return _run_forecast_for_path(c)
        except Exception as exc:
            last_err = exc
            continue
    raise HTTPException(status_code=500, detail=f"Static forecast failed: {last_err}")
