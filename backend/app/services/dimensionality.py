from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def pca_posts(kpis_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = kpis_df.copy()
    if len(df) < 10:
        out = pd.DataFrame({"pca1": [], "pca2": []})
        return out, {"ok": False, "message": "Too few rows for PCA (need ~10+)."}

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
    ]
    cat = ["sector", "post_type", "language", "day_of_week", "caption_length_group", "hashtag_group", "emoji_group"]
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
        ]
    )
    X = pre.fit_transform(df)
    Xd = X.toarray() if hasattr(X, "toarray") else X

    pca = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(Xd)
    out = pd.DataFrame({"pca1": comps[:, 0], "pca2": comps[:, 1]})
    return out, {"ok": True, "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_]}


def pca_businesses(business_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = business_df.copy()
    if len(df) < 6:
        out = pd.DataFrame({"pca1": [], "pca2": []})
        return out, {"ok": False, "message": "Too few businesses for PCA (need ~6+)."}

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
    X = df[numeric].fillna(0.0)
    pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=2, random_state=42))])
    comps = pipe.fit_transform(X)
    pca = pipe.named_steps["pca"]
    out = pd.DataFrame({"pca1": comps[:, 0], "pca2": comps[:, 1]})
    return out, {"ok": True, "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_]}

