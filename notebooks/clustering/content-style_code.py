from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


CONTENT_STYLE_FEATURES: List[str] = [
    "caption_length",
    "hashtags_count",
    "emoji_count",
    "CTA_present",
    "promo_post",
    "mentions_location",
    "religious_theme",
    "patriotic_theme",
    "arabic_dialect_style",
    "discount_percent",
]

BINARY_STYLE_COLS: List[str] = [
    "CTA_present",
    "promo_post",
    "mentions_location",
    "religious_theme",
    "patriotic_theme",
    "arabic_dialect_style",
]

NUMERIC_STYLE_COLS: List[str] = [
    "caption_length",
    "hashtags_count",
    "emoji_count",
    "discount_percent",
]


def _to_binary_flag(value) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    if isinstance(value, (int, float)):
        return int(value != 0)

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "present", "on"}:
        return 1
    if text in {"0", "false", "no", "n", "none", "nan", ""}:
        return 0
    return 1


def _load_json_cluster_settings(cluster_json_path: str | Path) -> Dict[str, float]:
    with open(cluster_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        raise ValueError(
            f"Expected cluster settings JSON object at {cluster_json_path}, got a JSON array."
        )

    params = payload.get("dbscan_params", {})
    if "eps" not in params or "min_samples" not in params:
        raise ValueError("JSON is missing dbscan_params.eps or dbscan_params.min_samples")

    return {
        "eps": float(params["eps"]),
        "min_samples": int(params["min_samples"]),
    }


def load_posts_dataset(posts_path: str | Path) -> pd.DataFrame:
    try:
        return pd.read_json(posts_path)
    except ValueError:
        return pd.read_json(posts_path, lines=True)


def preprocess_content_style_features(posts_df: pd.DataFrame) -> pd.DataFrame:
    work = posts_df.copy()

    for col in BINARY_STYLE_COLS:
        if col not in work.columns:
            work[col] = 0
        work[col] = work[col].apply(_to_binary_flag).astype(int)

    for col in NUMERIC_STYLE_COLS:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["caption_length"] = work["caption_length"].fillna(work["caption_length"].median())
    work["hashtags_count"] = work["hashtags_count"].fillna(0)
    work["emoji_count"] = work["emoji_count"].fillna(0)
    work["discount_percent"] = work["discount_percent"].fillna(0)

    for col in CONTENT_STYLE_FEATURES:
        if work[col].dtype.kind in "biufc":
            med = work[col].median()
            work[col] = work[col].fillna(0 if pd.isna(med) else med)
        else:
            work[col] = work[col].fillna(0)

    return work


def assign_content_style_clusters(
    posts_df: pd.DataFrame,
    cluster_json_path: str | Path,
    random_state: int = 42,
) -> pd.DataFrame:
    if posts_df.empty:
        raise ValueError("Input posts dataframe is empty.")

    work = preprocess_content_style_features(posts_df)
    X = work[CONTENT_STYLE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    perplexity = max(5, min(30, len(work) // 10))
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
    )
    X_tsne = tsne.fit_transform(X_scaled)

    dbscan_settings = _load_json_cluster_settings(cluster_json_path)
    dbscan = DBSCAN(
        eps=dbscan_settings["eps"],
        min_samples=dbscan_settings["min_samples"],
    )
    labels = dbscan.fit_predict(X_tsne)

    out = posts_df.copy()
    out["tsne_1"] = X_tsne[:, 0]
    out["tsne_2"] = X_tsne[:, 1]
    out["dbscan_fixed_cluster"] = labels

    return out


def export_json_records_with_clusters(
    cluster_json_path: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    with open(cluster_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        records = payload
    else:
        records = payload.get("records", [])

    if not records:
        raise ValueError("No records found inside cluster JSON.")

    df = pd.DataFrame(records)
    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(out_path, orient="records", force_ascii=False, indent=2)

    return df


CONTENT_STYLE_DEFAULT_JSON = r"C:\Users\hanib\data-mining-project\marketing\notebooks\clustering\artifacts\content_style_clustering\json\content_style_cluster_coordinates.json"


def summarize_content_style_clusters(
    posts_df: pd.DataFrame,
    cluster_json_path: str | Path = CONTENT_STYLE_DEFAULT_JSON,
    random_state: int = 42,
) -> pd.DataFrame:
    clustered = assign_content_style_clusters(posts_df, cluster_json_path, random_state)

    rate_cols = ["engagement_rate", "view_rate", "comment_rate"]
    for c in rate_cols:
        if c in clustered.columns:
            clustered[c] = pd.to_numeric(clustered[c], errors="coerce")

    summary = (
        clustered.groupby("dbscan_fixed_cluster")
        .agg(
            post_count=("dbscan_fixed_cluster", "size"),
            avg_engagement_rate=("engagement_rate", "mean"),
            avg_view_rate=("view_rate", "mean"),
            avg_comment_rate=("comment_rate", "mean"),
        )
        .reset_index()
    )

    summary.columns = ["cluster", "post_count", "avg_engagement_rate", "avg_view_rate", "avg_comment_rate"]
    return summary


def best_worst_posts_per_cluster(
    posts_df: pd.DataFrame,
    cluster_json_path: str | Path = CONTENT_STYLE_DEFAULT_JSON,
    random_state: int = 42,
) -> pd.DataFrame:
    clustered = assign_content_style_clusters(posts_df, cluster_json_path, random_state)

    for c in ["engagement_rate", "view_rate", "comment_rate"]:
        if c in clustered.columns:
            clustered[c] = pd.to_numeric(clustered[c], errors="coerce")

    clustered["_combined_score"] = (
        clustered["engagement_rate"].fillna(0) + clustered["view_rate"].fillna(0)
    )

    rows = []
    for cid in sorted(clustered["dbscan_fixed_cluster"].unique()):
        cluster_posts = clustered[clustered["dbscan_fixed_cluster"] == cid].copy()
        if cluster_posts.empty:
            continue

        best_idx = cluster_posts["_combined_score"].idxmax()
        worst_idx = cluster_posts["_combined_score"].idxmin()

        for rank, idx in [("best", best_idx), ("worst", worst_idx)]:
            row = clustered.loc[idx]
            rows.append({
                "cluster": int(cid),
                "rank": rank,
                "engagement_rate": float(row.get("engagement_rate", np.nan)),
                "view_rate": float(row.get("view_rate", np.nan)),
                "comment_rate": float(row.get("comment_rate", np.nan)),
                "combined_score": float(row["_combined_score"]),
                "post_type": str(row.get("post_type", "")),
                "caption_length": float(row.get("caption_length", np.nan)) if pd.notna(row.get("caption_length")) else None,
                "hashtags_count": float(row.get("hashtags_count", np.nan)) if pd.notna(row.get("hashtags_count")) else None,
            })

    return pd.DataFrame(rows)


def analyze_business_content_style(
    data_path: str,
    business_name: str,
    cluster_json_path: str | Path = CONTENT_STYLE_DEFAULT_JSON,
    random_state: int = 42,
) -> dict:
    df = load_posts_dataset(data_path)

    name_norm = str(business_name).strip().lower()
    user_mask = df["business_name"].astype(str).str.strip().str.lower().eq(name_norm)
    user_posts = df[user_mask].copy()

    if user_posts.empty:
        raise ValueError(f"No posts found for business '{business_name}'.")

    user_clustered = assign_content_style_clusters(user_posts, cluster_json_path, random_state)

    for c in ["engagement_rate", "view_rate", "comment_rate"]:
        if c in user_clustered.columns:
            user_clustered[c] = pd.to_numeric(user_clustered[c], errors="coerce")

    user_summary = (
        user_clustered.groupby("dbscan_fixed_cluster")
        .agg(
            user_post_count=("dbscan_fixed_cluster", "size"),
            user_avg_engagement_rate=("engagement_rate", "mean"),
            user_avg_view_rate=("view_rate", "mean"),
            user_avg_comment_rate=("comment_rate", "mean"),
        )
        .reset_index()
    )
    user_summary.columns = [
        "cluster", "user_post_count", "user_avg_engagement_rate",
        "user_avg_view_rate", "user_avg_comment_rate",
    ]

    user_clustered["_combined_score"] = (
        user_clustered["engagement_rate"].fillna(0) + user_clustered["view_rate"].fillna(0)
    )

    best_worst_rows = []
    for cid in sorted(user_clustered["dbscan_fixed_cluster"].unique()):
        cp = user_clustered[user_clustered["dbscan_fixed_cluster"] == cid]
        if cp.empty:
            continue
        best_idx = cp["_combined_score"].idxmax()
        worst_idx = cp["_combined_score"].idxmin()
        for rank, idx in [("best", best_idx), ("worst", worst_idx)]:
            r = user_clustered.loc[idx]
            best_worst_rows.append({
                "cluster": int(cid),
                "rank": rank,
                "engagement_rate": float(r.get("engagement_rate", np.nan)),
                "view_rate": float(r.get("view_rate", np.nan)),
                "comment_rate": float(r.get("comment_rate", np.nan)),
                "combined_score": float(r["_combined_score"]),
                "post_type": str(r.get("post_type", "")),
            })

    return {
        "business_name": business_name,
        "total_posts": len(user_posts),
        "cluster_summary": user_summary,
        "best_worst_posts": pd.DataFrame(best_worst_rows),
        "clustered_posts": user_clustered.drop(columns=["_combined_score"], errors="ignore"),
    }


POSTS_WITH_CLUSTERS_JSON = r"C:\Users\hanib\data-mining-project\marketing\notebooks\clustering\artifacts\content_style_clustering\json\posts_with_content_style_clusters.json"


def analyze_business_content_style_pure_inference(
    business_name: str,
    cached_clustered_json: str | Path = POSTS_WITH_CLUSTERS_JSON,
) -> dict:
    df = load_posts_dataset(cached_clustered_json)

    if "dbscan_fixed_cluster" not in df.columns:
        raise ValueError("Cached JSON missing dbscan_fixed_cluster column.")

    name_norm = str(business_name).strip().lower()
    user_mask = df["business_name"].astype(str).str.strip().str.lower().eq(name_norm)

    if not user_mask.any():
        raise ValueError(f"No posts found for business '{business_name}' in cached data.")

    for c in ["engagement_rate", "view_rate", "comment_rate"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    cluster_overall = (
        df.groupby("dbscan_fixed_cluster")
        .agg(
            cluster_post_count=("dbscan_fixed_cluster", "size"),
            cluster_avg_engagement_rate=("engagement_rate", "mean"),
            cluster_avg_view_rate=("view_rate", "mean"),
            cluster_avg_comment_rate=("comment_rate", "mean"),
        )
        .reset_index()
    )
    cluster_overall.columns = [
        "cluster", "cluster_post_count", "cluster_avg_engagement_rate",
        "cluster_avg_view_rate", "cluster_avg_comment_rate",
    ]

    user_posts = df[user_mask].copy()
    user_posts["_combined_score"] = (
        user_posts["engagement_rate"].fillna(0) + user_posts["view_rate"].fillna(0)
    )

    user_summary = (
        user_posts.groupby("dbscan_fixed_cluster")
        .agg(
            user_post_count=("dbscan_fixed_cluster", "size"),
            user_avg_engagement_rate=("engagement_rate", "mean"),
            user_avg_view_rate=("view_rate", "mean"),
            user_avg_comment_rate=("comment_rate", "mean"),
        )
        .reset_index()
    )
    user_summary.columns = [
        "cluster", "user_post_count", "user_avg_engagement_rate",
        "user_avg_view_rate", "user_avg_comment_rate",
    ]

    best_worst_rows = []
    for cid in sorted(user_posts["dbscan_fixed_cluster"].unique()):
        cp = user_posts[user_posts["dbscan_fixed_cluster"] == cid]
        if cp.empty:
            continue
        best_idx = cp["_combined_score"].idxmax()
        worst_idx = cp["_combined_score"].idxmin()
        for rank, idx in [("best", best_idx), ("worst", worst_idx)]:
            r = user_posts.loc[idx]
            best_worst_rows.append({
                "cluster": int(cid),
                "rank": rank,
                "engagement_rate": float(r.get("engagement_rate", np.nan)),
                "view_rate": float(r.get("view_rate", np.nan)),
                "comment_rate": float(r.get("comment_rate", np.nan)),
                "combined_score": float(r["_combined_score"]),
                "post_type": str(r.get("post_type", "")),
            })

    return {
        "business_name": business_name,
        "total_user_posts": len(user_posts),
        "cluster_overall": cluster_overall,
        "user_summary": user_summary,
        "best_worst_posts": pd.DataFrame(best_worst_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign content-style DBSCAN clusters to posts using settings from cluster JSON.",
    )
    parser.add_argument(
        "--cluster-json",
        default=r"C:\users\hanib\data_mining -project\marketing\notebooks\clustering\artifacts\content_style_clustering\json\content_style_cluster_coordinates.json",
        help="Path to content-style cluster JSON.",
    )
    parser.add_argument("--input", help="Input posts JSON path.")
    parser.add_argument(
        "--output",
        default=r"C:\users\hanib\data_mining -project\marketing\notebooks\clustering\artifacts\content_style_clustering\json\posts_with_content_style_clusters.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--export-json-records-only",
        action="store_true",
        help="Only export records from cluster JSON (already include cluster labels).",
    )
    args = parser.parse_args()

    if args.export_json_records_only:
        df_records = export_json_records_with_clusters(args.cluster_json, args.output)
        print(f"Exported {len(df_records)} records to: {args.output}")
        return

    if not args.input:
        raise ValueError("--input is required unless --export-json-records-only is used.")

    posts_df = load_posts_dataset(args.input)
    clustered = assign_content_style_clusters(posts_df, args.cluster_json)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clustered.to_json(out_path, orient="records", force_ascii=False, indent=2)

    print(f"Input rows: {len(posts_df)}")
    print(f"Output saved to: {args.output}")
    print("Cluster counts:")
    print(clustered["dbscan_fixed_cluster"].value_counts().sort_index())


if __name__ == "__main__":
    main()
