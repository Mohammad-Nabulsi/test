from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def weekly_trends(kpis_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = kpis_df.copy()
    if "post_date" not in df.columns:
        return pd.DataFrame(), {"ok": False, "message": "post_date missing."}
    df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")
    df = df.dropna(subset=["post_date"]).copy()
    if df.empty:
        return pd.DataFrame(), {"ok": False, "message": "No valid dates."}

    df["week"] = df["post_date"].dt.to_period("W").dt.start_time
    g = df.groupby("week").agg(
        post_count=("business_name", "size"),
        avg_engagement_rate=("engagement_rate_followers", "mean"),
        avg_views_per_follower=("views_per_follower", "mean"),
        reels_share=("post_type", lambda s: float((s == "reel").mean())),
        promo_share=("promo_post", "mean"),
        CTA_share=("CTA_present", "mean"),
    )
    out = g.reset_index().sort_values("week")
    out["week"] = out["week"].astype(str)
    return out, {"ok": True, "weeks": int(len(out))}


def business_momentum(kpis_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = kpis_df.copy()
    df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")
    df = df.dropna(subset=["post_date"]).copy()
    if df.empty:
        return pd.DataFrame(), {"ok": False, "message": "No valid dates."}

    rows = []
    for (biz, sector), g in df.groupby(["business_name", "sector"], dropna=False):
        g = g.sort_values("post_date")
        if len(g) < 10:
            continue
        last10 = g.tail(10)["engagement_rate_followers"].mean()
        prev10 = g.tail(20).head(10)["engagement_rate_followers"].mean() if len(g) >= 20 else g.head(max(len(g)-10,1))["engagement_rate_followers"].mean()
        rows.append(
            {
                "business_name": biz,
                "sector": sector,
                "posts": int(len(g)),
                "recent_10_avg_engagement_rate": float(last10),
                "previous_10_avg_engagement_rate": float(prev10),
                "momentum": float(last10 - prev10),
            }
        )
    out = pd.DataFrame(rows).sort_values("momentum", ascending=False) if rows else pd.DataFrame(
        columns=[
            "business_name",
            "sector",
            "posts",
            "recent_10_avg_engagement_rate",
            "previous_10_avg_engagement_rate",
            "momentum",
        ]
    )
    return out, {"ok": True, "businesses": int(len(out))}


def simple_forecast(weekly_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    if weekly_df.empty or "week" not in weekly_df.columns:
        return pd.DataFrame(), {"ok": False, "message": "No weekly data."}
    if len(weekly_df) < 6:
        return pd.DataFrame(), {"ok": False, "message": "Too few weeks to forecast (need ~6+)."}

    df = weekly_df.copy()
    df["week_dt"] = pd.to_datetime(df["week"], errors="coerce")
    df = df.dropna(subset=["week_dt"]).sort_values("week_dt").copy()
    if len(df) < 6:
        return pd.DataFrame(), {"ok": False, "message": "Too few valid weeks."}

    horizon = 4
    last = df["week_dt"].max()
    future_weeks = [last + pd.Timedelta(days=7 * i) for i in range(1, horizon + 1)]

    # Moving average forecast for engagement rate and views per follower
    ma_window = min(4, len(df))
    er_ma = float(df["avg_engagement_rate"].tail(ma_window).mean())
    vpf_ma = float(df["avg_views_per_follower"].tail(ma_window).mean())
    pc_ma = float(df["post_count"].tail(ma_window).mean())

    out = pd.DataFrame(
        {
            "week": [w.date().isoformat() for w in future_weeks],
            "forecast_post_count": [pc_ma] * horizon,
            "forecast_avg_engagement_rate": [er_ma] * horizon,
            "forecast_avg_views_per_follower": [vpf_ma] * horizon,
            "method": ["moving_average"] * horizon,
        }
    )
    return out, {"ok": True, "horizon_weeks": horizon}

