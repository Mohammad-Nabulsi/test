from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from app.config import settings


router = APIRouter(prefix="/api", tags=["forecast-single"])


class UploadedDatasetRequest(BaseModel):
    uploaded_file_path: str


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _storage_root() -> Path:
    # Keep behavior aligned with app config storage semantics.
    try:
        return Path(settings.storage_path()).resolve()
    except Exception:
        env_value = Path(__import__("os").getenv("STORAGE_DIR", "storage"))
        if env_value.is_absolute():
            return env_value.resolve()
        return (_project_root() / env_value).resolve()


def _forecast_outputs_root() -> Path:
    return (_storage_root() / "outputs" / "forecast_outputs").resolve()


def _safe_dataset_name(path: Path, original_input: str) -> str:
    candidate = path.stem
    if candidate.lower() == "raw" and path.parent.name:
        candidate = path.parent.name
    if not candidate:
        candidate = Path(str(original_input)).stem
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", candidate).strip("_").lower()
    return safe or "uploaded_dataset"


def _resolve_dataset_path(uploaded_file_path: str) -> Path:
    raw_input = str(uploaded_file_path or "").strip()
    if not raw_input:
        raise FileNotFoundError("Dataset file not found")

    p = Path(raw_input)
    if p.exists() and p.is_file():
        return p.resolve()
    if p.is_absolute():
        raise FileNotFoundError("Dataset file not found")

    project_root = _project_root()
    backend_root = _backend_root()
    candidates = [
        # exact relative paths
        project_root / raw_input,
        backend_root / raw_input,
        # preferred common dataset locations
        project_root / "data" / "processed" / raw_input,
        backend_root / "data" / "processed" / raw_input,
        project_root / "storage" / "raw" / raw_input,
        backend_root / "storage" / "raw" / raw_input,
        project_root / "data" / raw_input,
        backend_root / "data" / raw_input,
        # backend-root-relative
        backend_root / "storage" / raw_input,
        # repo-root-relative
        project_root / "storage" / raw_input,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()

    filename = p.name
    search_roots = [
        project_root / "data" / "processed",
        backend_root / "data" / "processed",
        project_root / "storage" / "raw",
        backend_root / "storage" / "raw",
        project_root / "data",
        backend_root / "data",
        project_root / "storage",
        backend_root / "storage",
    ]
    seen: set[Path] = set()
    for root in search_roots:
        root = root.resolve()
        if root in seen or not root.exists():
            continue
        seen.add(root)
        for match in root.rglob(filename):
            if match.is_file():
                return match.resolve()

    raise FileNotFoundError("Dataset file not found")


def _chart_path_to_url(chart_path: str | Path | None) -> str:
    if not chart_path:
        return ""
    try:
        p = Path(chart_path).resolve()
    except Exception:
        return ""
    if not p.exists():
        return ""
    outputs_root = _forecast_outputs_root()
    try:
        rel = p.relative_to(outputs_root)
    except ValueError:
        return ""
    return f"/forecast_outputs/{rel.as_posix()}"


def _load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return _read_json_dataframe(path)
    raise ValueError(f"Unsupported dataset format: {suffix}")


def _find_records(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, list):
        records = [item for item in value if isinstance(item, dict)]
        return records if records else None
    if isinstance(value, dict):
        for key in ("data", "posts", "records", "items", "rows", "results"):
            records = _find_records(value.get(key))
            if records:
                return records
        if value and all(not isinstance(v, (dict, list)) for v in value.values()):
            return [value]
        for nested in value.values():
            records = _find_records(nested)
            if records:
                return records
    return None


def _read_json_dataframe(path: Path) -> pd.DataFrame:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        records = _find_records(payload)
        if records:
            return pd.json_normalize(records)
    except Exception:
        pass
    try:
        return pd.read_json(path)
    except ValueError:
        return pd.read_json(path, lines=True)


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    def norm(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    cols = {norm(str(c)): c for c in df.columns}
    for cand in candidates:
        key = norm(cand)
        if key in cols:
            return cols[key]
    return None


def _available_columns_message(df: pd.DataFrame) -> str:
    cols = ", ".join(map(str, df.columns.tolist()[:80]))
    if len(df.columns) > 80:
        cols += ", ..."
    return f"Available columns: {cols}"


def _num_from_col(df: pd.DataFrame, candidates: list[str], default: float = 0.0) -> pd.Series:
    col = _pick_col(df, candidates)
    if col and col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def _parse_date_series(values: pd.Series) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce")
    if numeric_values.notna().any():
        best: pd.Series | None = None
        best_count = 0
        for unit in ("ms", "s"):
            parsed = pd.to_datetime(numeric_values, unit=unit, errors="coerce")
            plausible = parsed.dt.year.between(1990, 2100).fillna(False)
            count = int(plausible.sum())
            if count > best_count:
                best = parsed.where(plausible)
                best_count = count
        if best is not None and best_count > 0:
            return best

    return pd.to_datetime(values, errors="coerce")


def _ensure_engagement_rate(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    frame = df.copy()
    y_col = _pick_col(frame, ["engagement_rate", "engagementRate", "__forecast_engagement_rate__"])
    if y_col:
        frame[y_col] = pd.to_numeric(frame[y_col], errors="coerce")
        return frame, y_col

    likes = _num_from_col(frame, ["likes_count", "likes", "like_count"]).fillna(0.0)
    comments = _num_from_col(frame, ["comments_count", "comments", "comment_count"]).fillna(0.0)
    followers = _num_from_col(frame, ["followers_count", "followers", "follower_count"]).replace(0, np.nan)
    if followers.notna().any():
        frame["__forecast_engagement_rate__"] = ((likes + comments) / followers).replace([np.inf, -np.inf], np.nan)
        return frame, "__forecast_engagement_rate__"

    views = _num_from_col(frame, ["views_count", "views", "view_count"]).replace(0, np.nan)
    if views.notna().any():
        frame["__forecast_engagement_rate__"] = ((likes + comments) / views).replace([np.inf, -np.inf], np.nan)
        return frame, "__forecast_engagement_rate__"

    return frame, None


def _weekly_series(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    date_col = _pick_col(df, ["post_date", "date", "created_at", "timestamp"])
    frame, y_col = _ensure_engagement_rate(df)
    biz_col = _pick_col(df, ["business_name", "business"])
    if not date_col:
        raise ValueError(f"Dataset must include a date column (post_date, date, created_at, or timestamp). {_available_columns_message(df)}")
    if not y_col:
        raise ValueError(
            "Dataset must include engagement_rate/engagementRate or enough KPI columns to compute it "
            "(likes_count, comments_count, and followers_count or views_count). "
            f"{_available_columns_message(df)}"
        )

    frame[date_col] = _parse_date_series(frame[date_col])

    parsed_dates = frame[date_col]
    metric_values = pd.to_numeric(frame[y_col], errors="coerce")
    frame[date_col] = parsed_dates
    frame[y_col] = metric_values
    valid_date_count = int(parsed_dates.notna().sum())
    valid_metric_count = int(metric_values.notna().sum())
    frame = frame.dropna(subset=[date_col, y_col]).copy()
    if frame.empty:
        raise ValueError(
            "No valid rows after parsing dates/metrics. "
            f"Selected date column: {date_col}. "
            f"Selected metric column: {y_col}. "
            f"Non-null date rows: {valid_date_count}. "
            f"Non-null metric rows: {valid_metric_count}. "
            f"{_available_columns_message(df)}"
        )

    frame["week"] = frame[date_col].dt.to_period("W").dt.start_time
    weekly = frame.groupby("week", as_index=False)[y_col].mean().sort_values("week")
    weekly.rename(columns={"week": "ds", y_col: "y"}, inplace=True)
    weekly = weekly.set_index("ds").asfreq("W-MON")
    weekly["y"] = weekly["y"].interpolate(limit_direction="both")
    weekly = weekly.reset_index()

    business_name = "Uploaded Dataset"
    if biz_col and biz_col in frame.columns and not frame[biz_col].dropna().empty:
        business_name = str(frame[biz_col].dropna().astype(str).iloc[0]).strip() or business_name
    return weekly, business_name


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    mask = np.abs(y_true) > 1e-9
    if not np.any(mask):
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float | None]:
    mae = float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else 0.0
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else 0.0
    return mae, rmse, _safe_mape(y_true, y_pred)


def _weighted_average(values: np.ndarray) -> float:
    weights = np.arange(1, len(values) + 1, dtype=float)
    return float(np.average(values, weights=weights))


def _linear_trend_forecast(values: np.ndarray, horizon: int) -> np.ndarray:
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, values.astype(float), 1)
    future_x = np.arange(len(values), len(values) + horizon, dtype=float)
    return intercept + slope * future_x


def _exp_smoothing_forecast(values: np.ndarray, horizon: int, alpha: float = 0.45) -> np.ndarray:
    level = float(values[0])
    for value in values[1:]:
        level = alpha * float(value) + (1 - alpha) * level
    return np.full(horizon, level, dtype=float)


def _predict_from_history(model: str, history: list[float], horizon: int, window: int | None = None) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    if len(values) == 0:
        return np.zeros(horizon, dtype=float)
    if model == "naive_last_value":
        return np.full(horizon, float(values[-1]), dtype=float)
    if model == "moving_average":
        w = int(min(window or len(values), len(values)))
        return np.full(horizon, float(np.mean(values[-w:])), dtype=float)
    if model == "weighted_moving_average":
        w = int(min(window or len(values), len(values)))
        return np.full(horizon, _weighted_average(values[-w:]), dtype=float)
    if model == "linear_trend":
        w = int(min(window or len(values), len(values)))
        if w < 3:
            return np.full(horizon, float(values[-1]), dtype=float)
        return _linear_trend_forecast(values[-w:], horizon)
    if model == "exponential_smoothing":
        try:
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing

            fit = SimpleExpSmoothing(values, initialization_method="estimated").fit(optimized=True)
            return np.asarray(fit.forecast(horizon), dtype=float)
        except Exception:
            return _exp_smoothing_forecast(values, horizon)
    raise ValueError(f"Unsupported model: {model}")


def _walk_forward_predictions(train: np.ndarray, test: np.ndarray, model: str, window: int | None = None) -> np.ndarray:
    history = train.astype(float).tolist()
    preds: list[float] = []
    for actual in test:
        pred = float(_predict_from_history(model, history, 1, window)[0])
        preds.append(pred)
        history.append(float(actual))
    return np.asarray(preds, dtype=float)


def _candidate_models(train_size: int) -> list[tuple[str, int | None, str]]:
    windows = [2, 3, 4, 6, 8]
    candidates: list[tuple[str, int | None, str]] = [("naive_last_value", None, "naive_last_value")]
    for window in windows:
        if train_size >= window:
            candidates.append(("moving_average", window, f"moving_average_window_{window}"))
            candidates.append(("weighted_moving_average", window, f"weighted_moving_average_window_{window}"))
    candidates.append(("exponential_smoothing", None, "exponential_smoothing"))
    for window in windows:
        if train_size >= max(3, window):
            candidates.append(("linear_trend", window, f"linear_trend_window_{window}"))
    return candidates


def _run_forecasting_analysis(df: pd.DataFrame, output_dir: Path, periods: int, output_prefix: str) -> Dict[str, Any]:
    weekly, business_name = _weekly_series(df)
    y = weekly["y"].astype(float).reset_index(drop=True)
    if len(y) < 4:
        raise ValueError("At least 4 weekly data points are required for forecasting.")

    test_size = max(1, min(max(2, int(round(len(y) * 0.2))), max(1, len(y) // 3)))
    split_idx = max(2, len(y) - test_size)
    train = y.iloc[:split_idx].to_numpy(dtype=float)
    test = y.iloc[split_idx:].to_numpy(dtype=float)

    tuning_summary: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for model, window, label in _candidate_models(len(train)):
        yhat = np.clip(_walk_forward_predictions(train, test, model, window), 0, None)
        mae, rmse, mape = _metrics(test, yhat)
        row = {
            "model": label,
            "base_model": model,
            "window": window,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
        }
        tuning_summary.append(row)
        if best is None or rmse < best["RMSE"]:
            best = {**row, "predictions": yhat, "model_key": model}

    if best is None:
        raise ValueError("No forecasting models could be evaluated.")

    yhat_test = np.asarray(best["predictions"], dtype=float)
    residuals = test - yhat_test
    residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float(best["RMSE"])
    uncertainty = max(float(best["RMSE"]), float(best["MAE"]), residual_std, 1e-6)
    future_pred = np.clip(_predict_from_history(best["model_key"], y.to_list(), periods, best.get("window")), 0, None)
    last_date = pd.to_datetime(weekly["ds"].max())
    future_dates = pd.date_range(last_date + pd.Timedelta(days=7), periods=periods, freq="W-MON")
    test_dates = pd.to_datetime(weekly["ds"].iloc[split_idx:])

    steps = np.sqrt(np.arange(1, periods + 1, dtype=float))
    lower_bound = np.clip(future_pred - 1.28 * uncertainty * steps, 0, None)
    upper_bound = future_pred + 1.28 * uncertainty * steps

    hist_df = pd.DataFrame({"ds": pd.to_datetime(weekly["ds"]), "yhat": y, "forecast_type": "history"})
    test_df = pd.DataFrame({"ds": test_dates, "yhat": yhat_test, "actual": test, "forecast_type": "test"})
    fut_df = pd.DataFrame({
        "ds": future_dates,
        "yhat": future_pred,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "forecast_type": "future",
    })
    forecast_df = pd.concat([hist_df, fut_df], ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_prefix}.csv"
    forecast_df.to_csv(csv_path, index=False)
    chart_path = output_dir / f"{output_prefix}.png"

    # Chart generation is best-effort: API should still succeed if plotting fails.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        import matplotlib.dates as mdates

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(13, 6.5))
        ax.plot(hist_df["ds"], hist_df["yhat"], label="Actual weekly engagement", linewidth=2.6, color="#2563eb")
        ax.plot(test_df["ds"], test_df["yhat"], label="Test prediction", linewidth=2.3, color="#f97316", linestyle="--", marker="o", markersize=4)
        ax.plot(fut_df["ds"], fut_df["yhat"], label="Future forecast", linewidth=2.8, color="#16a34a", marker="o", markersize=4)
        ax.fill_between(
            pd.to_datetime(fut_df["ds"]),
            fut_df["lower_bound"].astype(float),
            fut_df["upper_bound"].astype(float),
            color="#16a34a",
            alpha=0.16,
            label="Forecast uncertainty band",
        )
        ax.axvspan(future_dates.min(), future_dates.max(), color="#dcfce7", alpha=0.22)
        ax.axvline(last_date, color="#64748b", linestyle=":", linewidth=1.6)
        ax.text(last_date, ax.get_ylim()[1] * 0.96, " forecast starts", color="#475569", fontsize=9, va="top")
        ax.set_title(f"Weekly Engagement Forecast - {business_name}", fontsize=16, fontweight="bold", loc="left", pad=14)
        plt.xlabel("Week")
        plt.ylabel("Engagement Rate")
        ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.1f}%")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.grid(True, alpha=0.22)
        ax.legend(loc="upper left", frameon=True, framealpha=0.95)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ymin = max(0, min(float(hist_df["yhat"].min()), float(lower_bound.min())) * 0.88)
        ymax = max(float(hist_df["yhat"].max()), float(upper_bound.max()), float(test_df["yhat"].max())) * 1.12
        if ymax <= ymin:
            ymax = ymin + 0.01
        ax.set_ylim(ymin, ymax)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(chart_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        chart_path = Path("")

    summary_df = pd.DataFrame([{
        "business_name": business_name,
        "best_model": best["model"],
        "forecast_horizon_weeks": periods,
        "best_MAE": best["MAE"],
        "best_RMSE": best["RMSE"],
        "best_MAPE": best["MAPE"],
    }])
    test_predictions = [
        {
            "date": pd.to_datetime(date).strftime("%Y-%m-%d"),
            "actual_engagement_rate": float(actual),
            "predicted_engagement_rate": float(pred),
            "residual": float(actual - pred),
        }
        for date, actual, pred in zip(test_dates, test, yhat_test)
    ]
    confidence_interval = [
        {
            "date": pd.to_datetime(date).strftime("%Y-%m-%d"),
            "lower_bound": float(low),
            "upper_bound": float(high),
        }
        for date, low, high in zip(future_dates, lower_bound, upper_bound)
    ]
    return {
        "forecast_df": forecast_df,
        "summary_df": summary_df,
        "chart_path": str(chart_path) if str(chart_path) else "",
        "tuning_summary": sorted(tuning_summary, key=lambda row: row["RMSE"]),
        "test_predictions": test_predictions,
        "forecast_confidence_interval": confidence_interval,
        "train_size": int(len(train)),
        "test_size": int(len(test)),
    }


def _format_future_forecast(forecast_df: pd.DataFrame) -> list[dict[str, Any]]:
    future = forecast_df[forecast_df["forecast_type"] == "future"].copy()
    future["date"] = pd.to_datetime(future["ds"]).dt.strftime("%Y-%m-%d")
    future["predicted_engagement_rate"] = future["yhat"].astype(float)
    future["lower_bound"] = future["lower_bound"].astype(float)
    future["upper_bound"] = future["upper_bound"].astype(float)
    return future[["date", "predicted_engagement_rate", "lower_bound", "upper_bound"]].to_dict(orient="records")


def _run_forecast_for_path(input_path: str, output_group: str | None, output_prefix: str) -> Dict[str, Any]:
    path = _resolve_dataset_path(input_path)
    df = _load_dataset(path)
    out_dir = _forecast_outputs_root() / (output_group or _safe_dataset_name(path, input_path))
    result = _run_forecasting_analysis(df, out_dir, periods=8, output_prefix=output_prefix)
    summary = result["summary_df"].iloc[0]
    chart_url = _chart_path_to_url(result.get("chart_path"))
    return {
        "business_name": str(summary.get("business_name", "")),
        "best_model": str(summary.get("best_model", "")),
        "forecast_horizon_weeks": int(summary.get("forecast_horizon_weeks", 0)),
        "best_MAE": float(summary.get("best_MAE", 0.0)),
        "best_RMSE": float(summary.get("best_RMSE", 0.0)),
        "best_MAPE": None if pd.isna(summary.get("best_MAPE")) else float(summary.get("best_MAPE")),
        "message": (
            f"Forecast completed using {summary.get('best_model', '')} for {summary.get('business_name', '')}. "
            f"Trained on {result['train_size']} weeks and tested on {result['test_size']} weeks."
        ),
        "future_forecast": _format_future_forecast(result["forecast_df"]),
        "test_predictions": result["test_predictions"],
        "tuning_summary": result["tuning_summary"],
        "forecast_confidence_interval": result["forecast_confidence_interval"],
        "chart_url": chart_url,
    }


@router.post("/forecast/analyze-single", status_code=status.HTTP_200_OK)
def forecast_analyze_single(request: UploadedDatasetRequest) -> Dict[str, Any]:
    try:
        return _run_forecast_for_path(
            request.uploaded_file_path,
            output_group=None,
            output_prefix="uploaded_weekly_forecast",
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

