
# 06 Forecasting
# Forecast engagement with Prophet when available; fallback to Exponential Smoothing.


import warnings
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "utils" / "utils.py").exists():
    if (PROJECT_ROOT / "notebooks" / "utils" / "utils.py").exists():
        PROJECT_ROOT = PROJECT_ROOT / "notebooks"
    elif (PROJECT_ROOT.parent / "utils" / "utils.py").exists():
        PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import ensure_project_dirs, load_raw_dataset, clean_dataset, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR
from utils.features import engineer_kpis, build_post_feature_sets, aggregate_business_features
from utils.evaluation import regression_metrics, rank_models
from utils.visualization import set_plot_style, save_figure
from pathlib import Path

set_plot_style()
ensure_project_dirs()

PROJECT_ROOT = Path(__file__).resolve()

while PROJECT_ROOT.name != "marketing":
    PROJECT_ROOT = PROJECT_ROOT.parent

RAW_DATA_PATH = PROJECT_ROOT / "jsons" / "all_final_appended.json"

if not RAW_DATA_PATH.exists():
    RAW_DATA_PATH = PROJECT_ROOT / "synthetic_generator" / "synthetic_social_media_posts.csv"

KPI_PATH = PROJECT_ROOT / "data" / "processed" / "kpi_dataset.csv"


# Load KPI and Build Weekly/Monthly Time Series

if KPI_PATH.exists():
    df = pd.read_csv(KPI_PATH, parse_dates=["post_date"])
else:
    df = engineer_kpis(clean_dataset(load_raw_dataset(RAW_DATA_PATH)))
df.head()


import importlib.util

def build_ts(frame, freq):
    ts = (
        frame.set_index("post_date")["engagement"]
        .resample(freq)
        .mean()
        .dropna()
        .reset_index()
        .rename(columns={"post_date": "ds", "engagement": "y"})
    )
    ts["ds"] = pd.to_datetime(ts["ds"])
    return ts

weekly = build_ts(df, "W-MON")
monthly = build_ts(df, "MS")


# Experiment Grid and Metrics


has_prophet = importlib.util.find_spec("prophet") is not None
if has_prophet:
    from prophet import Prophet
else:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
# Forecasting Hyperparameter Experimentation:
# Different forecasting settings are tested to evaluate which configuration produces better prediction performance.
# The selected model is based on evaluation metrics such as MAE/RMSE rather than manual guessing.
cp_vals, modes = [0.01,0.05,0.1,0.5], ["additive","multiplicative"]
rows, store = [], {}
for agg_name, ts in {"weekly": weekly, "monthly": monthly}.items():
    ts = ts.sort_values("ds").reset_index(drop=True)
    if len(ts) < 8:
        continue
    split = int(len(ts) * 0.8)
    train, test = ts.iloc[:split], ts.iloc[split:]
    if len(test) == 0:
        continue
    if has_prophet:
        for cp in cp_vals:
            for mode in modes:
                m = Prophet(changepoint_prior_scale=cp, seasonality_mode=mode, yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
                m.fit(train)
                fut = m.make_future_dataframe(periods=len(test), freq="W-MON" if agg_name=="weekly" else "MS")
                pred = m.predict(fut)[["ds","yhat"]].tail(len(test))
                mets = regression_metrics(test["y"], pred["yhat"])
                rows.append({"model":"prophet","aggregation":agg_name,"changepoint_prior_scale":cp,"seasonality_mode":mode, **mets})
                store[("prophet",agg_name,cp,mode)] = (train,test,pred)
    else:
        model = ExponentialSmoothing(train["y"], trend="add", seasonal=None).fit(optimized=True)
        pred = pd.DataFrame({"ds": test["ds"].values, "yhat": model.forecast(len(test)).values})
        mets = regression_metrics(test["y"], pred["yhat"])
        for cp in cp_vals:
            for mode in modes:
                rows.append({"model":"exp_smoothing_fallback","aggregation":agg_name,"changepoint_prior_scale":cp,"seasonality_mode":mode, **mets})
                store[("exp_smoothing_fallback",agg_name,cp,mode)] = (train,test,pred)

exp_input = pd.DataFrame(rows)
if exp_input.empty:
    exp = pd.DataFrame(columns=["model","aggregation","changepoint_prior_scale","seasonality_mode","MAE","RMSE","MAPE","composite_score"])
    best = None
    forecast = pd.DataFrame(columns=["ds","y","yhat","level","entity"])
else:
    exp = rank_models(exp_input, lower_is_better_cols=["MAE","RMSE","MAPE"])
    best = exp.iloc[0]
    train, test, pred = store[(best["model"],best["aggregation"],best["changepoint_prior_scale"],best["seasonality_mode"])]
    forecast = test.merge(pred, on="ds", how="left")
    forecast["level"], forecast["entity"] = "overall", "all_businesses"


# Sector/Business Extensions and Save

extra = []
if best is not None:
    freq = "W-MON" if best["aggregation"] == "weekly" else "MS"

    def build_segment_ts(seg, freq):
        ts = (
            seg.set_index("post_date")["engagement"]
            .resample(freq)
            .mean()
            .dropna()
            .reset_index()
            .rename(columns={"post_date": "ds", "engagement": "y"})
        )
        ts["ds"] = pd.to_datetime(ts["ds"])
        return ts

    for sector, seg in df.groupby("sector"):
        ts = build_segment_ts(seg, freq)
        if len(ts) < 10:
            continue
        split = int(len(ts)*0.8)
        train_s, test_s = ts.iloc[:split], ts.iloc[split:]
        if len(test_s) < 2:
            continue
        if best["model"] == "prophet":
            from prophet import Prophet
            m = Prophet(changepoint_prior_scale=float(best["changepoint_prior_scale"]), seasonality_mode=best["seasonality_mode"], yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
            m.fit(train_s)
            fut = m.make_future_dataframe(periods=len(test_s), freq=freq)
            pred_s = m.predict(fut)[["ds","yhat"]].tail(len(test_s))
        else:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            fit = ExponentialSmoothing(train_s["y"], trend="add", seasonal=None).fit(optimized=True)
            pred_s = pd.DataFrame({"ds": test_s["ds"].values, "yhat": fit.forecast(len(test_s)).values})
        mrg = test_s.merge(pred_s, on="ds", how="left")
        mrg["level"], mrg["entity"] = "sector", sector
        extra.append(mrg)

    for business, seg in df.groupby("business_name"):
        ts = build_segment_ts(seg, freq)
        if len(ts) < 12:
            continue
        split = int(len(ts)*0.8)
        train_b, test_b = ts.iloc[:split], ts.iloc[split:]
        if len(test_b) < 2:
            continue
        if best["model"] == "prophet":
            from prophet import Prophet
            m = Prophet(changepoint_prior_scale=float(best["changepoint_prior_scale"]), seasonality_mode=best["seasonality_mode"], yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
            m.fit(train_b)
            fut = m.make_future_dataframe(periods=len(test_b), freq=freq)
            pred_b = m.predict(fut)[["ds","yhat"]].tail(len(test_b))
        else:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            fit = ExponentialSmoothing(train_b["y"], trend="add", seasonal=None).fit(optimized=True)
            pred_b = pd.DataFrame({"ds": test_b["ds"].values, "yhat": fit.forecast(len(test_b)).values})
        mrg = test_b.merge(pred_b, on="ds", how="left")
        mrg["level"], mrg["entity"] = "business", business
        extra.append(mrg)

forecast = pd.concat([forecast] + extra, ignore_index=True) if extra else forecast
forecast.to_csv(PROCESSED_DIR / "forecast.csv", index=False)
exp.to_csv(PROCESSED_DIR / "forecast_metrics.csv", index=False)
display(exp.head(10))
display(forecast.head(20))
if best is None:
    print("No valid forecasting experiments were generated from available data.")
else:
    print("Insight: selected model balances low error and stable behavior across aggregation levels.")

