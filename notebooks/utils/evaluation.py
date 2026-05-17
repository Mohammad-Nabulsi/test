import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    denom = np.where(np.abs(y_true) < 1e-9, 1.0, np.abs(y_true))
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def _minmax(series):
    s = pd.Series(series)
    if s.max() - s.min() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def rank_models(df, higher_is_better_cols=None, lower_is_better_cols=None):
    out = df.copy()
    higher_is_better_cols = higher_is_better_cols or []
    lower_is_better_cols = lower_is_better_cols or []
    score = np.zeros(len(out))
    for col in higher_is_better_cols:
        score += _minmax(out[col]).to_numpy()
    for col in lower_is_better_cols:
        score += (1 - _minmax(out[col])).to_numpy()
    out["composite_score"] = score
    return out.sort_values("composite_score", ascending=False).reset_index(drop=True)
