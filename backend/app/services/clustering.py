from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ClusterResult:
    df: pd.DataFrame
    model_info: Dict


def _choose_k(X: np.ndarray, k_min: int = 3, k_max: int = 6, random_state: int = 42) -> Tuple[int, Dict]:
    best_k = k_min
    best_score = -1.0
    scores: Dict[int, float] = {}

    # silhouette requires at least 2 clusters and <= n_samples-1
    n = X.shape[0]
    k_max2 = min(k_max, n - 1) if n > 2 else 2
    if n < 10:
        return min(3, k_max2), {"reason": "too_few_samples", "scores": scores}

    for k in range(k_min, max(k_min, k_max2) + 1):
        try:
            km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            labels = km.fit_predict(X)
            sc = float(silhouette_score(X, labels)) if len(set(labels)) > 1 else -1.0
            scores[k] = sc
            if sc > best_score:
                best_score = sc
                best_k = k
        except Exception:
            continue
    return best_k, {"scores": scores, "best_score": best_score}


def _profile_label(df: pd.DataFrame) -> str:
    # Simple rule-based label from average metrics in the cluster.
    er = float(df["engagement_rate_followers"].mean()) if "engagement_rate_followers" in df.columns else 0.0
    vpf = float(df["views_per_follower"].mean()) if "views_per_follower" in df.columns else 0.0
    promo = float(df["promo_post"].mean()) if "promo_post" in df.columns else 0.0
    cta = float(df["CTA_present"].mean()) if "CTA_present" in df.columns else 0.0

    if er > 0.03 and vpf > 1.5:
        return "High-Impact Viral"
    if er > 0.02 and cta > 0.55 and promo < 0.4:
        return "Engaging With Clear CTAs"
    if promo > 0.6 and er < 0.01:
        return "Salesy Low Engagement"
    if vpf > 2.5 and er < 0.012:
        return "Viewed But Not Loved"
    return "Steady Baseline"


def post_clustering(kpis_df: pd.DataFrame) -> ClusterResult:
    df = kpis_df.copy()
    if len(df) < 20:
        df["post_cluster"] = 0
        df["post_cluster_label"] = "Too Few Posts"
        return ClusterResult(df=df, model_info={"ok": False, "message": "Too few rows for clustering (need ~20+)."})

    numeric = [
        "followers_count",
        "posting_hour",
        "caption_length",
        "hashtags_count",
        "emoji_count",
        "likes_count",
        "comments_count",
        "views_count",
        "engagement_rate_followers",
        "views_per_follower",
        "discount_percent",
    ]
    cat = ["sector", "post_type", "language", "day_of_week", "caption_length_group", "hashtag_group", "emoji_group", "discount_range"]
    bools = ["CTA_present", "promo_post", "religious_theme", "patriotic_theme", "arabic_dialect_style", "mentions_location"]

    for c in numeric:
        if c not in df.columns:
            df[c] = 0
    for c in cat:
        if c not in df.columns:
            df[c] = "unknown"
    for c in bools:
        if c not in df.columns:
            df[c] = False

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("bool", "passthrough", bools),
        ],
        remainder="drop",
    )

    X = pre.fit_transform(df)
    # Ensure dense for silhouette on small sets
    Xd = X.toarray() if hasattr(X, "toarray") else X
    k, kinfo = _choose_k(Xd, 3, 6)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(Xd)
    df["post_cluster"] = labels.astype(int)

    labels_map = {}
    for cid in sorted(df["post_cluster"].unique()):
        labels_map[int(cid)] = _profile_label(df[df["post_cluster"] == cid])
    df["post_cluster_label"] = df["post_cluster"].map(labels_map)
    return ClusterResult(df=df, model_info={"ok": True, "k": k, **kinfo, "cluster_labels": labels_map})


def business_clustering(kpis_df: pd.DataFrame) -> ClusterResult:
    df = kpis_df.copy()
    if len(df) < 30 or df["business_name"].nunique() < 6:
        agg = _business_aggregate(df)
        agg["business_cluster"] = 0
        agg["business_cluster_name"] = "Too Few Businesses"
        return ClusterResult(df=agg, model_info={"ok": False, "message": "Too few businesses for clustering (need ~6+)."})

    agg = _business_aggregate(df)

    numeric = [
        "followers_count_max",
        "avg_engagement_rate",
        "avg_comments_rate",
        "avg_views_per_follower",
        "pct_reels",
        "pct_CTA",
        "pct_promo",
        "pct_Arabic",
        "pct_dialect",
        "posts_per_week",
    ]
    pre = Pipeline([("scaler", StandardScaler())])
    X = pre.fit_transform(agg[numeric].fillna(0))
    k, kinfo = _choose_k(X, 3, 6)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    agg["business_cluster"] = labels.astype(int)

    # Name clusters based on profiles (simple, explainable)
    names = {}
    for cid in sorted(agg["business_cluster"].unique()):
        cdf = agg[agg["business_cluster"] == cid]
        er = float(cdf["avg_engagement_rate"].mean())
        freq = float(cdf["posts_per_week"].mean())
        followers = float(cdf["followers_count_max"].mean())
        promo = float(cdf["pct_promo"].mean())
        dialect = float(cdf["pct_dialect"].mean())
        if er > 0.02 and freq > 3:
            names[int(cid)] = "Strong Content Machines"
        elif followers > 15000 and er < 0.008:
            names[int(cid)] = "Sleeping Giants"
        elif followers < 4000 and er > 0.018:
            names[int(cid)] = "Hidden Gems"
        elif promo > 0.55 and er < 0.012:
            names[int(cid)] = "Salesy Weak Brands"
        elif dialect > 0.55 and er > 0.013:
            names[int(cid)] = "Community Builders"
        else:
            names[int(cid)] = "Balanced Players"
    agg["business_cluster_name"] = agg["business_cluster"].map(names)
    return ClusterResult(df=agg, model_info={"ok": True, "k": k, **kinfo, "cluster_names": names})


def _business_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    # posting frequency: posts per week (based on min/max post_date)
    temp = df.copy()
    if "post_date" in temp.columns:
        temp["post_date"] = pd.to_datetime(temp["post_date"], errors="coerce")
    else:
        temp["post_date"] = pd.NaT

    def _posts_per_week(g: pd.DataFrame) -> float:
        dates = g["post_date"].dropna().sort_values()
        if len(dates) < 2:
            return float(len(g))
        weeks = max((dates.max() - dates.min()).days / 7.0, 1.0)
        return float(len(g) / weeks)

    agg = (
        temp.groupby(["business_name", "sector"], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "posts": len(g),
                    "followers_count_max": float(g["followers_count"].max()) if "followers_count" in g else 0.0,
                    "avg_engagement_rate": float(g["engagement_rate_followers"].mean()) if "engagement_rate_followers" in g else 0.0,
                    "avg_comments_rate": float(g["comments_rate_followers"].mean()) if "comments_rate_followers" in g else 0.0,
                    "avg_views_per_follower": float(g["views_per_follower"].mean()) if "views_per_follower" in g else 0.0,
                    "pct_reels": float((g["post_type"] == "reel").mean()) if "post_type" in g else 0.0,
                    "pct_CTA": float(g["CTA_present"].mean()) if "CTA_present" in g else 0.0,
                    "pct_promo": float(g["promo_post"].mean()) if "promo_post" in g else 0.0,
                    "pct_Arabic": float((g["language"] == "Arabic").mean()) if "language" in g else 0.0,
                    "pct_dialect": float(g["arabic_dialect_style"].mean()) if "arabic_dialect_style" in g else 0.0,
                    "posts_per_week": _posts_per_week(g),
                }
            )
        )
        .reset_index()
    )
    return agg

