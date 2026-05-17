"""
Adaptive hashtag association recommendation pipeline.

This script takes a pandas DataFrame of social media posts and generates:
1. Association-rule-based hashtag recommendations.
2. Dataset-size-aware reliability labels using count, confidence, lift, and sample size.
3. Fallback recommendations when the uploaded dataset is too small or no rule reaches the minimum count.
4. Category-level recommendations showing which hashtags to use and which to avoid for each sector/post type/business/category.

Required input columns
----------------------
At least one of:
- hashtags: list-like column, stringified list, comma-separated string, pipe-separated string, or text containing hashtags.
- caption_text: caption text containing hashtags such as "#vanilla #cake".

Recommended columns for KPI labeling:
- views_count: numeric post views.
- followers_count: numeric business/page followers.

Optional columns:
- likes_count: numeric likes.
- comments_count: numeric comments.
- business_name: used for business-normalized performance labeling and category recommendations.
- sector: used for category-level recommendations.
- post_type: used for category-level recommendations.
- performance_label: if already exists, values should be high_view_rate, avg_view_rate, or low_view_rate.

Main function
-------------
    generate_hashtag_association_recommendations(df)

CLI usage
---------
    python hashtag_association_recommendations.py --input data.csv --output-dir reports
    python hashtag_association_recommendations.py --input data.json --output-dir reports --category-cols sector post_type business_name

Outputs
-------
CSV files are saved to --output-dir:
- hashtag_frequency.csv
- hashtag_association_all_rules.csv
- hashtag_association_recommended_rules.csv
- hashtag_association_warning_rules.csv
- hashtag_association_top_recommendations.csv
- hashtag_association_category_recommendations.csv
- hashtag_association_fallback_recommendations.csv
- hashtag_recommendation_summary.csv
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
except ImportError as exc:  # pragma: no cover - helpful runtime message
    raise ImportError(
        "This script requires mlxtend. Install it with: pip install mlxtend"
    ) from exc


HASHTAG_RE = re.compile(r"(?<!\w)#([^\s#]+)", flags=re.UNICODE)
VALID_PERFORMANCE_LABELS = {"high_view_rate", "avg_view_rate", "low_view_rate"}
GOOD_CONSEQUENTS = {"performance=high_view_rate", "performance=avg_view_rate"}
BAD_CONSEQUENTS = {"performance=low_view_rate"}


@dataclass(frozen=True)
class AdaptiveThresholds:
    """Dataset-size-aware thresholds for rule mining and recommendation filtering."""

    n_posts: int
    profile: str
    min_support: float
    min_confidence: float
    min_lift: float
    min_count: int
    strong_count: int
    medium_count: int
    experimental_count: int
    fallback_min_count: int
    fallback_min_confidence: float
    fallback_min_lift: float


def compute_adaptive_thresholds(n_posts: int) -> AdaptiveThresholds:
    """
    Compute thresholds that adapt to uploaded dataset size.

    Why this matters:
    - Count is the most size-sensitive reliability measure.
    - Confidence can look perfect on tiny samples, so it must be judged together with count.
    - Larger datasets should require more observed examples before calling a rule reliable.
    - Small datasets can still produce experimental suggestions, but they should be labeled honestly.
    """
    n = max(int(n_posts), 0)

    if n < 30:
        profile = "tiny"
        min_support = 0.01
        min_confidence = 0.35
        min_lift = 1.10
        min_count = 2
        strong_count = 8
        medium_count = 5
        experimental_count = 2
    elif n < 100:
        profile = "small"
        min_support = 0.015
        min_confidence = 0.35
        min_lift = 1.15
        min_count = 3
        strong_count = 10
        medium_count = 6
        experimental_count = 3
    elif n < 500:
        profile = "medium"
        min_support = 0.01
        min_confidence = 0.35
        min_lift = 1.20
        min_count = max(4, math.ceil(0.010 * n))
        strong_count = max(12, math.ceil(0.025 * n))
        medium_count = max(7, math.ceil(0.015 * n))
        experimental_count = max(3, math.ceil(0.006 * n))
    elif n < 2000:
        profile = "large"
        min_support = 0.006
        min_confidence = 0.30
        min_lift = 1.20
        min_count = max(8, math.ceil(0.0075 * n))
        strong_count = max(20, math.ceil(0.020 * n))
        medium_count = max(12, math.ceil(0.010 * n))
        experimental_count = max(5, math.ceil(0.004 * n))
    else:
        profile = "very_large"
        min_support = 0.003
        min_confidence = 0.30
        min_lift = 1.20
        min_count = max(15, min(50, math.ceil(0.005 * n)))
        strong_count = max(35, min(120, math.ceil(0.012 * n)))
        medium_count = max(20, min(80, math.ceil(0.007 * n)))
        experimental_count = max(8, min(35, math.ceil(0.003 * n)))

    # Fallback is more permissive on count, but stricter on confidence.
    fallback_min_count = max(2, min(min_count - 1, math.ceil(0.003 * max(n, 1))))
    fallback_min_confidence = max(0.65, min_confidence + 0.25)
    fallback_min_lift = max(1.05, min_lift - 0.10)

    return AdaptiveThresholds(
        n_posts=n,
        profile=profile,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        min_count=min_count,
        strong_count=strong_count,
        medium_count=medium_count,
        experimental_count=experimental_count,
        fallback_min_count=fallback_min_count,
        fallback_min_confidence=fallback_min_confidence,
        fallback_min_lift=fallback_min_lift,
    )


def normalize_post_type(x: Any) -> str:
    """Normalize common post type names."""
    s = str(x).strip().lower()
    if s in {"image", "img", "post", "photo"}:
        return "post"
    if s in {"reel", "reels"}:
        return "reel"
    if s in {"video", "vid"}:
        return "video"
    if s in {"carousel", "album"}:
        return "carousel"
    return s if s and s != "nan" else "unknown"


def normalize_hashtag(tag: Any) -> str:
    """Normalize one hashtag while preserving Arabic text."""
    s = str(tag).strip().lstrip("#").strip()
    s = s.strip(".,!?;:'\"()[]{}<>")
    return s.lower() if s else ""


def extract_hashtags(text: Any) -> list[str]:
    """Extract hashtags from caption text."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    seen, out = set(), []
    for match in HASHTAG_RE.findall(str(text)):
        tag = normalize_hashtag(match)
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def parse_hashtags(value: Any) -> list[str]:
    """Parse hashtags from lists, stringified lists, comma/pipe strings, or text."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, (list, tuple, set)):
        raw = list(value)
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            raw = []
        else:
            try:
                parsed = ast.literal_eval(s)
                raw = parsed if isinstance(parsed, list) else extract_hashtags(s)
            except Exception:
                if "#" in s:
                    raw = extract_hashtags(s)
                elif "|" in s:
                    raw = [x.strip() for x in s.split("|")]
                elif "," in s:
                    raw = [x.strip() for x in s.split(",")]
                else:
                    raw = [s]
    else:
        raw = [value]

    seen, out = set(), []
    for item in raw:
        tag = normalize_hashtag(item)
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def ensure_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and standardize columns.

    The script needs hashtags from either `hashtags` or `caption_text`.
    KPI columns are optional but strongly recommended.
    """
    out = df.copy()

    if "hashtags" not in out.columns and "caption_text" not in out.columns:
        raise ValueError("Input DataFrame must contain at least one of: 'hashtags' or 'caption_text'.")

    defaults = {
        "caption_text": "",
        "hashtags": np.nan,
        "business_name": "Unknown Business",
        "sector": "Unknown",
        "post_type": "unknown",
        "views_count": 0,
        "followers_count": 0,
        "likes_count": 0,
        "comments_count": 0,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    out["caption_text"] = out["caption_text"].fillna("").astype(str)
    out["business_name"] = out["business_name"].fillna("Unknown Business").astype(str).str.strip()
    out["sector"] = out["sector"].fillna("Unknown").astype(str).str.strip()
    out["post_type"] = out["post_type"].apply(normalize_post_type)

    for col in ["views_count", "followers_count", "likes_count", "comments_count"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).clip(lower=0)

    return out


def prepare_hashtags(df: pd.DataFrame) -> pd.DataFrame:
    """Merge hashtags from the hashtags column and from caption text."""
    out = df.copy()
    from_col = out["hashtags"].apply(parse_hashtags)
    from_caption = out["caption_text"].apply(extract_hashtags)

    merged = []
    for tags_a, tags_b in zip(from_col.tolist(), from_caption.tolist()):
        seen, tags = set(), []
        for tag in tags_a + tags_b:
            if tag and tag not in seen:
                seen.add(tag)
                tags.append(tag)
        merged.append(tags)

    out["hashtags"] = merged
    out["hashtag_count"] = out["hashtags"].apply(len)
    return out


def create_performance_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create high/average/low view-rate labels if `performance_label` is missing or invalid.

    Logic:
    - view_rate = views_count / followers_count.
    - Compare each post to its business median view_rate.
    - high_view_rate: at least 1.5x business median.
    - avg_view_rate: at least 0.8x business median.
    - low_view_rate: below 0.8x business median.
    """
    out = df.copy()

    if "performance_label" in out.columns:
        existing = out["performance_label"].astype(str)
        if existing.isin(VALID_PERFORMANCE_LABELS).mean() >= 0.8:
            out["performance_label"] = existing.where(existing.isin(VALID_PERFORMANCE_LABELS), "avg_view_rate")
            return out

    denom = out["followers_count"].replace(0, np.nan)
    out["view_rate"] = (out["views_count"] / denom).replace([np.inf, -np.inf], np.nan)
    out["engagement"] = out["likes_count"] + out["comments_count"]
    out["engagement_rate"] = (out["engagement"] / denom).replace([np.inf, -np.inf], np.nan)

    business_median = out.groupby("business_name")["view_rate"].transform("median").replace(0, np.nan)
    out["view_rate_ratio"] = out["view_rate"] / business_median

    # If follower/view data is missing, fall back to global rank of views_count.
    if out["view_rate_ratio"].notna().sum() < 3:
        ranks = out["views_count"].rank(pct=True, method="average")
        out["performance_label"] = np.select(
            [ranks >= 0.67, ranks >= 0.34],
            ["high_view_rate", "avg_view_rate"],
            default="low_view_rate",
        )
        return out

    out["performance_label"] = np.select(
        [out["view_rate_ratio"] >= 1.5, out["view_rate_ratio"] >= 0.8],
        ["high_view_rate", "avg_view_rate"],
        default="low_view_rate",
    )
    out.loc[out["view_rate_ratio"].isna(), "performance_label"] = "avg_view_rate"
    return out


def build_transactions(df: pd.DataFrame) -> list[list[str]]:
    """Create association-rule transactions from hashtags and performance labels."""
    transactions: list[list[str]] = []
    for _, row in df.iterrows():
        tags = row.get("hashtags", [])
        label = row.get("performance_label")
        if not isinstance(tags, list) or not tags or label not in VALID_PERFORMANCE_LABELS:
            continue
        transactions.append([f"hashtag={tag}" for tag in tags] + [f"performance={label}"])
    return transactions


def mine_hashtag_rules(df: pd.DataFrame, thresholds: AdaptiveThresholds | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mine hashtag -> performance association rules."""
    transactions = build_transactions(df)
    n_posts = len(transactions)
    thresholds = thresholds or compute_adaptive_thresholds(n_posts)

    diagnostics: dict[str, Any] = {
        **asdict(thresholds),
        "valid_transaction_posts": n_posts,
        "frequent_itemsets": 0,
        "raw_rules": 0,
        "filtered_rules": 0,
    }

    if n_posts == 0:
        return pd.DataFrame(), diagnostics

    te = TransactionEncoder()
    one_hot = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
    itemsets = apriori(one_hot, min_support=thresholds.min_support, use_colnames=True)
    diagnostics["frequent_itemsets"] = int(len(itemsets))

    if itemsets.empty:
        return pd.DataFrame(), diagnostics

    rules = association_rules(itemsets, metric="confidence", min_threshold=thresholds.min_confidence)
    diagnostics["raw_rules"] = int(len(rules))

    if rules.empty:
        return pd.DataFrame(), diagnostics

    rules = rules.copy()
    rules["antecedent_size"] = rules["antecedents"].apply(len)
    rules["consequent_size"] = rules["consequents"].apply(len)
    rules["consequent_item"] = rules["consequents"].apply(lambda s: next(iter(s)) if len(s) == 1 else None)

    valid_consequents = {"performance=high_view_rate", "performance=avg_view_rate", "performance=low_view_rate"}
    mask = (
        (rules["consequent_size"] == 1)
        & (rules["consequent_item"].isin(valid_consequents))
        & (rules["antecedent_size"].between(1, 3))
        & rules["antecedents"].apply(lambda items: all(str(x).startswith("hashtag=") for x in items))
    )

    out = rules.loc[mask].copy()
    if out.empty:
        return out, diagnostics

    out["count"] = (out["support"] * n_posts).round().astype(int)
    out["antecedent_count"] = (out["antecedent support"] * n_posts).round().astype(int)
    out["antecedents_str"] = out["antecedents"].apply(lambda s: "{" + ", ".join(sorted(s)) + "}")
    out["hashtags_clean"] = out["antecedents"].apply(lambda s: ", ".join(sorted(x.replace("hashtag=", "") for x in s)))

    diagnostics["filtered_rules"] = int(len(out))
    return out.reset_index(drop=True), diagnostics


def reliability_level(count: int, confidence: float, lift: float, thresholds: AdaptiveThresholds) -> str:
    """Assign reliability by combining dataset-size-aware count + confidence + lift."""
    count = int(count)
    confidence = float(confidence)
    lift = float(lift)

    if count >= thresholds.strong_count and confidence >= 0.55 and lift >= thresholds.min_lift:
        return "strong"
    if count >= thresholds.medium_count and confidence >= 0.50 and lift >= thresholds.min_lift:
        return "medium"
    if count >= thresholds.min_count and confidence >= thresholds.min_confidence and lift >= thresholds.min_lift:
        return "directional"
    if count >= thresholds.experimental_count and confidence >= thresholds.fallback_min_confidence and lift >= thresholds.fallback_min_lift:
        return "experimental"
    return "weak"


def score_rules(rules: pd.DataFrame, thresholds: AdaptiveThresholds) -> pd.DataFrame:
    """Score rules for ranking recommendations."""
    if rules.empty:
        return rules.copy()

    out = rules.copy()
    out["reliability"] = [
        reliability_level(c, conf, lift, thresholds)
        for c, conf, lift in zip(out["count"], out["confidence"], out["lift"])
    ]

    target_map = {
        "performance=high_view_rate": 1.00,
        "performance=avg_view_rate": 0.55,
        "performance=low_view_rate": -1.00,
    }
    rel_map = {"strong": 1.00, "medium": 0.80, "directional": 0.60, "experimental": 0.40, "weak": 0.10}

    out["target_score"] = out["consequent_item"].map(target_map).fillna(0.0)
    out["count_score"] = (out["count"] / max(thresholds.strong_count, 1)).clip(upper=1)
    out["lift_score"] = ((out["lift"] - 1) / 2).clip(lower=0, upper=1)
    out["reliability_score"] = out["reliability"].map(rel_map).fillna(0.0)
    out["strength_score"] = out["confidence"] * out["lift"]

    # Count is weighted heavily because it is the most affected by dataset size.
    out["recommendation_score"] = (
        0.30 * out["confidence"]
        + 0.25 * out["count_score"]
        + 0.20 * out["lift_score"]
        + 0.15 * out["reliability_score"]
        + 0.10 * out["target_score"]
    )
    return out


def explain_rule(row: pd.Series, action: str) -> str:
    """Create human-readable recommendation text."""
    tags = row.get("hashtags_clean", row.get("antecedents_str", ""))
    consequent = str(row.get("consequent_item", "")).replace("performance=", "")
    count = int(row.get("count", 0))
    antecedent_count = int(row.get("antecedent_count", 0))
    confidence = float(row.get("confidence", 0))
    lift = float(row.get("lift", 0))
    reliability = row.get("reliability", "weak")

    if action == "avoid":
        return (
            f"Avoid or test carefully: {tags}. It appeared in {antecedent_count} posts; "
            f"{count} were {consequent}. Confidence={confidence:.0%}, lift={lift:.2f}x, "
            f"reliability={reliability}. This is a risk signal, not proof of causality."
        )

    return (
        f"Use/test: {tags}. It appeared in {antecedent_count} posts; {count} were {consequent}. "
        f"Confidence={confidence:.0%}, lift={lift:.2f}x, reliability={reliability}."
    )


def top_with_diversity(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Keep diverse recommendations instead of many duplicates of the same hashtag set."""
    if df.empty:
        return df.copy()

    tmp = df.copy()
    tmp["antecedent_len"] = tmp["antecedents"].apply(len)
    tmp = tmp.sort_values(
        ["recommendation_score", "reliability_score", "count", "confidence", "lift", "antecedent_len"],
        ascending=[False, False, False, False, False, True],
    )

    seen: set[tuple[str, ...]] = set()
    rows = []
    for _, row in tmp.iterrows():
        key = tuple(sorted(x.replace("hashtag=", "") for x in row["antecedents"]))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
        if len(rows) >= k:
            break

    return pd.DataFrame(rows) if rows else tmp.head(0)


def hashtag_level_fallback(df: pd.DataFrame, min_count: int = 2) -> pd.DataFrame:
    """
    Fallback when association rules are too sparse.

    Instead of mining combinations, score individual hashtags by their observed high/avg/low rates.
    This is less powerful than association rules but safer for tiny or sparse uploads.
    """
    rows = []
    baseline_high = (df["performance_label"] == "high_view_rate").mean() if len(df) else 0
    baseline_low = (df["performance_label"] == "low_view_rate").mean() if len(df) else 0

    for _, row in df.iterrows():
        tags = row.get("hashtags", [])
        if not isinstance(tags, list):
            continue
        for tag in tags:
            rows.append({"hashtag": tag, "performance_label": row.get("performance_label")})

    tag_df = pd.DataFrame(rows)
    if tag_df.empty:
        return pd.DataFrame()

    summary = tag_df.groupby("hashtag").agg(
        count=("performance_label", "size"),
        high_rate=("performance_label", lambda s: float((s == "high_view_rate").mean())),
        avg_rate=("performance_label", lambda s: float((s == "avg_view_rate").mean())),
        low_rate=("performance_label", lambda s: float((s == "low_view_rate").mean())),
    ).reset_index()

    summary = summary[summary["count"] >= min_count].copy()
    if summary.empty:
        return summary

    summary["high_lift_like"] = summary["high_rate"] / max(baseline_high, 1e-9)
    summary["low_lift_like"] = summary["low_rate"] / max(baseline_low, 1e-9)
    summary["fallback_score"] = (
        0.45 * summary["high_rate"]
        + 0.25 * np.log1p(summary["count"]) / np.log1p(max(summary["count"].max(), 1))
        + 0.20 * summary["high_lift_like"].clip(upper=3) / 3
        - 0.10 * summary["low_rate"]
    )
    summary["recommendation_type"] = np.where(summary["low_rate"] >= summary["high_rate"], "avoid", "use")
    summary["reliability"] = np.where(summary["count"] >= max(5, min_count * 2), "fallback_directional", "fallback_experimental")
    summary["recommendation_text"] = summary.apply(
        lambda r: (
            f"{('Use/test' if r['recommendation_type'] == 'use' else 'Avoid/test carefully')}: #{r['hashtag']} "
            f"based on fallback hashtag-level stats. Count={int(r['count'])}, "
            f"high_rate={r['high_rate']:.0%}, low_rate={r['low_rate']:.0%}, "
            f"reliability={r['reliability']}."
        ),
        axis=1,
    )
    return summary.sort_values("fallback_score", ascending=False).reset_index(drop=True)


def generate_recommendations_for_subset(df: pd.DataFrame, top_k: int = 10) -> dict[str, Any]:
    """Generate global recommendations for one DataFrame/subset."""
    transactions = build_transactions(df)
    thresholds = compute_adaptive_thresholds(len(transactions))
    rules, diagnostics = mine_hashtag_rules(df, thresholds)
    rules = score_rules(rules, thresholds)

    if rules.empty:
        fallback = hashtag_level_fallback(df, min_count=thresholds.fallback_min_count)
        return {
            "thresholds": asdict(thresholds),
            "diagnostics": diagnostics,
            "all_rules": rules,
            "recommended_rules": pd.DataFrame(),
            "warning_rules": pd.DataFrame(),
            "top_recommendations": pd.DataFrame(),
            "fallback_recommendations": fallback,
            "used_fallback": True,
        }

    # Main recommendation filter: count threshold adapts with dataset size.
    recommended = rules[
        rules["consequent_item"].isin(GOOD_CONSEQUENTS)
        & (rules["lift"] >= thresholds.min_lift)
        & (rules["count"] >= thresholds.min_count)
        & (rules["confidence"] >= thresholds.min_confidence)
        & (rules["reliability"].isin(["strong", "medium", "directional", "experimental"]))
    ].copy()

    warnings = rules[
        rules["consequent_item"].isin(BAD_CONSEQUENTS)
        & (rules["lift"] >= thresholds.min_lift)
        & (rules["count"] >= thresholds.min_count)
        & (rules["confidence"] >= thresholds.min_confidence)
    ].copy()

    used_fallback = False
    fallback = pd.DataFrame()

    # Fallback logic if uploaded data does not match assigned min_count.
    if recommended.empty:
        relaxed = rules[
            rules["consequent_item"].isin(GOOD_CONSEQUENTS)
            & (rules["lift"] >= thresholds.fallback_min_lift)
            & (rules["count"] >= thresholds.fallback_min_count)
            & (rules["confidence"] >= thresholds.fallback_min_confidence)
        ].copy()
        if not relaxed.empty:
            relaxed["reliability"] = "experimental_fallback"
            relaxed["recommendation_score"] = relaxed["recommendation_score"] * 0.85
            recommended = relaxed
            used_fallback = True
        else:
            fallback = hashtag_level_fallback(df, min_count=thresholds.fallback_min_count)
            used_fallback = True

    recommended = recommended.sort_values(
        ["recommendation_score", "count", "confidence", "lift"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    warnings = warnings.sort_values(
        ["recommendation_score", "count", "confidence", "lift"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    top_recs = top_with_diversity(recommended, k=top_k)

    for frame, action in [(rules, "use"), (recommended, "use"), (warnings, "avoid"), (top_recs, "use")]:
        if not frame.empty:
            frame["recommendation_text"] = frame.apply(lambda r: explain_rule(r, action=action), axis=1)

    return {
        "thresholds": asdict(thresholds),
        "diagnostics": diagnostics,
        "all_rules": rules,
        "recommended_rules": recommended,
        "warning_rules": warnings,
        "top_recommendations": top_recs,
        "fallback_recommendations": fallback,
        "used_fallback": used_fallback,
    }


def generate_category_recommendations(
    df: pd.DataFrame,
    category_cols: Iterable[str] = ("sector", "post_type"),
    min_category_posts: int | None = None,
    top_k_per_category: int = 5,
) -> pd.DataFrame:
    """Generate best/avoid hashtag recommendations within each category value."""
    parts = []

    for category_col in category_cols:
        if category_col not in df.columns:
            continue

        for category_value, sub in df.groupby(category_col, dropna=False):
            sub = sub.copy()
            valid_posts = len(build_transactions(sub))
            thresholds = compute_adaptive_thresholds(valid_posts)
            min_rows = min_category_posts if min_category_posts is not None else max(10, thresholds.min_count * 2)

            if valid_posts < min_rows:
                fallback = hashtag_level_fallback(sub, min_count=thresholds.fallback_min_count)
                if fallback.empty:
                    continue
                fallback = fallback.head(top_k_per_category).copy()
                fallback["category_col"] = category_col
                fallback["category_value"] = str(category_value)
                fallback["action"] = fallback["recommendation_type"]
                fallback["source"] = "fallback_hashtag_level_category"
                parts.append(fallback)
                continue

            result = generate_recommendations_for_subset(sub, top_k=top_k_per_category)

            best = result["top_recommendations"].head(top_k_per_category).copy()
            if not best.empty:
                best["category_col"] = category_col
                best["category_value"] = str(category_value)
                best["action"] = "use"
                best["source"] = "association_rule_category"
                parts.append(best)

            avoid = result["warning_rules"].head(top_k_per_category).copy()
            if not avoid.empty:
                avoid["category_col"] = category_col
                avoid["category_value"] = str(category_value)
                avoid["action"] = "avoid"
                avoid["source"] = "association_rule_category"
                parts.append(avoid)

            # If rules were too sparse inside this category, include fallback output.
            fallback = result.get("fallback_recommendations", pd.DataFrame())
            if isinstance(fallback, pd.DataFrame) and not fallback.empty:
                fallback = fallback.head(top_k_per_category).copy()
                fallback["category_col"] = category_col
                fallback["category_value"] = str(category_value)
                fallback["action"] = fallback["recommendation_type"]
                fallback["source"] = "fallback_hashtag_level_category"
                parts.append(fallback)

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True, sort=False)


def hashtag_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Return hashtag frequency table."""
    tags = [tag for tags in df["hashtags"] for tag in tags]
    if not tags:
        return pd.DataFrame(columns=["hashtag", "frequency"])
    return pd.Series(tags, dtype="object").value_counts().rename_axis("hashtag").reset_index(name="frequency")


def build_summary(result: dict[str, Any], category_df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact dashboard summary."""
    thresholds = result["thresholds"]
    rows = [
        {"metric": "dataset_size_profile", "value": thresholds["profile"]},
        {"metric": "valid_transaction_posts", "value": result["diagnostics"].get("valid_transaction_posts", 0)},
        {"metric": "adaptive_min_count", "value": thresholds["min_count"]},
        {"metric": "adaptive_min_confidence", "value": thresholds["min_confidence"]},
        {"metric": "adaptive_min_lift", "value": thresholds["min_lift"]},
        {"metric": "used_fallback", "value": result["used_fallback"]},
        {"metric": "all_rules", "value": len(result["all_rules"])},
        {"metric": "recommended_rules", "value": len(result["recommended_rules"])},
        {"metric": "warning_rules", "value": len(result["warning_rules"])},
        {"metric": "category_recommendations", "value": len(category_df)},
    ]
    return pd.DataFrame(rows)


def generate_hashtag_association_recommendations(
    df: pd.DataFrame,
    category_cols: Iterable[str] = ("sector", "post_type"),
    output_dir: str | Path | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    End-to-end adaptive hashtag recommendation pipeline.

    Parameters
    ----------
    df:
        Input DataFrame. Must include `hashtags` and/or `caption_text`.
        Recommended metric columns are `views_count` and `followers_count`.
    category_cols:
        Columns used to create category-specific recommendations, such as sector,
        post_type, or business_name.
    output_dir:
        Optional directory where CSV outputs will be saved. If None, files are not saved.
    top_k:
        Number of top global recommendations to keep.

    Returns
    -------
    dict with:
        - prepared_df
        - hashtag_frequency
        - thresholds
        - diagnostics
        - all_rules
        - recommended_rules
        - warning_rules
        - top_recommendations
        - fallback_recommendations
        - category_recommendations
        - summary
    """
    prepared = ensure_input_columns(df)
    prepared = prepare_hashtags(prepared)
    prepared = create_performance_labels(prepared)

    result = generate_recommendations_for_subset(prepared, top_k=top_k)
    category_df = generate_category_recommendations(prepared, category_cols=category_cols, top_k_per_category=5)
    freq = hashtag_frequency(prepared)
    summary = build_summary(result, category_df)

    output = {
        "prepared_df": prepared,
        "hashtag_frequency": freq,
        "thresholds": result["thresholds"],
        "diagnostics": result["diagnostics"],
        "all_rules": result["all_rules"],
        "recommended_rules": result["recommended_rules"],
        "warning_rules": result["warning_rules"],
        "top_recommendations": result["top_recommendations"],
        "fallback_recommendations": result["fallback_recommendations"],
        "category_recommendations": category_df,
        "summary": summary,
    }

    if output_dir is not None:
        save_outputs(output, output_dir)

    return output


def save_outputs(output: dict[str, Any], output_dir: str | Path) -> None:
    """Save all output tables as CSV files."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_map = {
        "prepared_posts_with_labels.csv": output["prepared_df"],
        "hashtag_frequency.csv": output["hashtag_frequency"],
        "hashtag_association_all_rules.csv": output["all_rules"],
        "hashtag_association_recommended_rules.csv": output["recommended_rules"],
        "hashtag_association_warning_rules.csv": output["warning_rules"],
        "hashtag_association_top_recommendations.csv": output["top_recommendations"],
        "hashtag_association_fallback_recommendations.csv": output["fallback_recommendations"],
        "hashtag_association_category_recommendations.csv": output["category_recommendations"],
        "hashtag_recommendation_summary.csv": output["summary"],
    }

    for filename, frame in save_map.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(out_dir / filename, index=False)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a CSV or JSON file into a DataFrame."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported input file type: {path.suffix}. Use CSV or JSON.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive hashtag association recommendations")
    parser.add_argument("--input", required=True, help="Path to input CSV or JSON dataset")
    parser.add_argument("--output-dir", default="reports", help="Directory to save output CSV files")
    parser.add_argument(
        "--category-cols",
        nargs="*",
        default=["sector", "post_type"],
        help="Category columns for category-specific best/avoid hashtag recommendations",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of top global recommendations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataframe(args.input)
    output = generate_hashtag_association_recommendations(
        df,
        category_cols=args.category_cols,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )

    print("Adaptive hashtag recommendation pipeline finished.")
    print(output["summary"].to_string(index=False))
    print(f"\nSaved outputs to: {Path(args.output_dir).resolve()}")

    top = output["top_recommendations"]
    fallback = output["fallback_recommendations"]
    category = output["category_recommendations"]

    if not top.empty:
        print("\nTop global recommendations:")
        cols = [c for c in ["hashtags_clean", "consequent_item", "count", "confidence", "lift", "reliability", "recommendation_text"] if c in top.columns]
        print(top[cols].head(args.top_k).to_string(index=False))
    elif not fallback.empty:
        print("\nFallback global recommendations:")
        cols = [c for c in ["hashtag", "count", "high_rate", "low_rate", "reliability", "recommendation_text"] if c in fallback.columns]
        print(fallback[cols].head(args.top_k).to_string(index=False))
    else:
        print("\nNo reliable hashtag recommendations were found. Try uploading more posts with hashtags and view metrics.")

    if not category.empty:
        print("\nCategory-level best/avoid recommendations saved. Preview:")
        preview_cols = [c for c in ["category_col", "category_value", "action", "hashtags_clean", "hashtag", "count", "confidence", "lift", "reliability", "recommendation_text"] if c in category.columns]
        print(category[preview_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
