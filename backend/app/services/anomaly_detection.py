from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(kpis_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = kpis_df.copy()
    if df.empty:
        return pd.DataFrame(), {"ok": False, "message": "Empty dataset."}

    rows = []

    # Percentile rules
    vpf = df["views_per_follower"].fillna(0)
    er = df["engagement_rate_followers"].fillna(0)
    cr = df["comments_rate_followers"].fillna(0)

    vpf_p95 = float(vpf.quantile(0.95))
    er_p95 = float(er.quantile(0.95))
    cr_p98 = float(cr.quantile(0.98))

    for idx, r in df.iterrows():
        flags = []
        if float(r.get("views_per_follower", 0)) >= max(vpf_p95, 0.01):
            flags.append("viral_spike")
        if float(r.get("engagement_rate_followers", 0)) >= max(er_p95, 0.0005):
            flags.append("engagement_spike")
        if float(r.get("comments_rate_followers", 0)) >= max(cr_p98, 0.0001):
            flags.append("comment_spike")
        if float(r.get("followers_count", 0)) >= df["followers_count"].quantile(0.9) and float(
            r.get("engagement_rate_followers", 0)
        ) <= df["engagement_rate_followers"].quantile(0.2):
            flags.append("weak_big_account_post")
        if str(r.get("post_type", "")).lower() in {"reel", "video"} and float(r.get("views_per_follower", 0)) <= df[
            "views_per_follower"
        ].quantile(0.2):
            flags.append("underperforming_video")
        if flags:
            rows.append(
                {
                    "row_id": int(idx),
                    "business_name": r.get("business_name"),
                    "sector": r.get("sector"),
                    "post_date": str(r.get("post_date")),
                    "post_type": r.get("post_type"),
                    "flags": "|".join(flags),
                    "engagement_rate_followers": float(r.get("engagement_rate_followers", 0)),
                    "views_per_follower": float(r.get("views_per_follower", 0)),
                    "comments_rate_followers": float(r.get("comments_rate_followers", 0)),
                }
            )

    rules_df = pd.DataFrame(rows)

    # IsolationForest (numeric)
    iso_rows = []
    if len(df) >= 40:
        feats = df[
            [
                "followers_count",
                "likes_count",
                "comments_count",
                "views_count",
                "engagement_rate_followers",
                "views_per_follower",
                "comments_rate_followers",
                "discount_percent",
            ]
        ].fillna(0.0)
        iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        pred = iso.fit_predict(feats)
        scores = iso.score_samples(feats)
        outliers = np.where(pred == -1)[0]
        for i in outliers[:2000]:
            r = df.iloc[int(i)]
            iso_rows.append(
                {
                    "row_id": int(i),
                    "business_name": r.get("business_name"),
                    "sector": r.get("sector"),
                    "post_date": str(r.get("post_date")),
                    "post_type": r.get("post_type"),
                    "flags": "isolation_forest_outlier",
                    "anomaly_score": float(scores[int(i)]),
                }
            )
    iso_df = pd.DataFrame(iso_rows)

    out = pd.concat([rules_df, iso_df], ignore_index=True) if (not rules_df.empty or not iso_df.empty) else pd.DataFrame(
        columns=[
            "row_id",
            "business_name",
            "sector",
            "post_date",
            "post_type",
            "flags",
            "engagement_rate_followers",
            "views_per_follower",
            "comments_rate_followers",
            "anomaly_score",
        ]
    )
    return out, {"ok": True, "rule_anomalies": int(len(rules_df)), "iforest_anomalies": int(len(iso_df))}

