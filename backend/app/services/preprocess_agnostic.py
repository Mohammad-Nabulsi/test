from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreprocessSpec:
    required_inputs: list[str]
    optional_inputs: list[str]
    outputs: list[str]


PREPROCESS_SPEC = PreprocessSpec(
    required_inputs=["At least one row of tabular data (CSV or JSON)."],
    optional_inputs=[
        "business_name",
        "sector",
        "followers_count",
        "post_date",
        "posting_hour",
        "day_of_week",
        "month",
        "post_type",
        "caption_text",
        "hashtags",
        "caption_length",
        "hashtags_count",
        "emoji_count",
        "likes_count",
        "comments_count",
        "views_count",
        "language",
        "CTA_present",
        "promo_post",
        "discount_percent",
        "mentions_location",
        "religious_theme",
        "patriotic_theme",
        "arabic_dialect_style",
        "sponsored",
    ],
    outputs=[
        "dataset with normalized schema",
        "missingness/consistency report",
        "cleaned JSON records",
    ],
)


_COLUMN_ALIASES: dict[str, list[str]] = {
    "business_name": ["business", "brand_name", "account_name", "page_name", "name"],
    "sector": ["industry", "category", "business_sector", "vertical"],
    "followers_count": ["followers", "follower_count", "subscribers", "audience_size"],
    "post_date": ["date", "posted_at", "created_at", "timestamp", "post_timestamp"],
    "posting_hour": ["hour", "post_hour", "published_hour"],
    "day_of_week": ["weekday", "day_name"],
    "month": ["post_month"],
    "post_type": ["content_type", "media_type", "type"],
    "caption_text": ["caption", "text", "post_text", "description", "content"],
    "hashtags": ["tags", "hash_tags"],
    "caption_length": ["caption_len", "text_length", "content_length"],
    "hashtags_count": ["hashtag_count", "num_hashtags"],
    "emoji_count": ["num_emojis", "emoji_cnt"],
    "likes_count": ["likes", "like_count"],
    "comments_count": ["comments", "comment_count"],
    "views_count": ["views", "view_count", "impressions", "reach"],
    "language": ["lang", "post_language"],
    "cta_present": ["cta_present", "has_cta", "cta"],
    "promo_post": ["is_promo", "promotional_post", "promo"],
    "discount_percent": ["discount", "discount_pct"],
    "mentions_location": ["has_location", "location_mentioned"],
    "religious_theme": ["is_religious", "religion_theme"],
    "patriotic_theme": ["is_patriotic", "national_theme"],
    "arabic_dialect_style": ["dialect_style", "arabic_dialect"],
    "sponsored": ["is_sponsored", "paid_partnership"],
}


_HASHTAG_RE = re.compile(r"(?<!\w)#([^\s#]+)", flags=re.UNICODE)


def _normalize_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    values = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "t": True,
        "f": False,
    }
    return values.map(mapping).fillna(False).astype(bool)


def _find_existing_column(df: pd.DataFrame, target: str) -> str | None:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    if target in lowered:
        return lowered[target]
    for alias in _COLUMN_ALIASES.get(target, []):
        if alias in lowered:
            return lowered[alias]
    return None


def _extract_hashtags(text: str) -> list[str]:
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for token in _HASHTAG_RE.findall(text):
        tag = str(token).strip().lower().lstrip("#")
        tag = re.sub(r"[^\w\u0600-\u06FF]+", "", tag)
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def _parse_hashtags(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        raw = value
    else:
        s = str(value).strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s.replace("'", '"'))
                raw = parsed if isinstance(parsed, list) else [s]
            except Exception:
                raw = [s]
        elif "|" in s:
            raw = [x.strip() for x in s.split("|")]
        elif "," in s:
            raw = [x.strip() for x in s.split(",")]
        elif "#" in s:
            raw = [f"#{x}" for x in _HASHTAG_RE.findall(s)]
        else:
            raw = [s]
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        tag = str(item).strip().lower().lstrip("#")
        tag = re.sub(r"[^\w\u0600-\u06FF]+", "", tag)
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def _safe_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    return json.loads(out.to_json(orient="records", force_ascii=False))


def preprocess_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Dataset-agnostic preprocessing derived from the notebook flow.

    The function maps common column aliases to a canonical schema, fills missing
    values consistently, runs lightweight consistency checks, and prepares a
    cleaned dataframe that downstream KPI/insights stages can consume.
    """
    original = df.copy()
    working = df.copy()

    mapped_columns: dict[str, str] = {}
    for target in _COLUMN_ALIASES:
        existing = _find_existing_column(working, target)
        if existing is not None and existing != target:
            working[target] = working[existing]
            mapped_columns[target] = existing

    if "business_name" not in working.columns:
        working["business_name"] = "Unknown Business"
    if "sector" not in working.columns:
        working["sector"] = "Unknown"
    if "caption_text" not in working.columns:
        working["caption_text"] = ""

    raw_post_date = working.get("post_date", pd.Series([None] * len(working)))
    post_date_num = pd.to_numeric(raw_post_date, errors="coerce")
    working["post_date"] = pd.to_datetime(post_date_num, unit="ms", errors="coerce")
    non_numeric_mask = post_date_num.isna()
    if non_numeric_mask.any():
        working.loc[non_numeric_mask, "post_date"] = pd.to_datetime(
            raw_post_date.loc[non_numeric_mask], errors="coerce"
        )

    before_drop = len(working)
    working = working.dropna(subset=["post_date"]).copy()
    dropped_bad_dates = before_drop - len(working)

    for col, default in {
        "business_name": "Unknown Business",
        "sector": "Unknown",
        "post_type": "unknown",
        "language": "Unknown",
        "caption_text": "",
        "day_of_week": "Unknown",
    }.items():
        if col not in working.columns:
            working[col] = default
        working[col] = working[col].fillna(default).astype(str).str.strip()
        working.loc[working[col] == "", col] = default

    numeric_cols = [
        "followers_count",
        "posting_hour",
        "month",
        "caption_length",
        "hashtags_count",
        "emoji_count",
        "likes_count",
        "comments_count",
        "views_count",
        "discount_percent",
    ]
    for col in numeric_cols:
        if col not in working.columns:
            working[col] = np.nan
        working[col] = pd.to_numeric(working[col], errors="coerce")

    follower_median = pd.to_numeric(working["followers_count"], errors="coerce").median()
    follower_fill = 0 if pd.isna(follower_median) else float(follower_median)
    working["followers_count"] = working["followers_count"].fillna(follower_fill).clip(lower=0)
    working["posting_hour"] = working["posting_hour"].fillna(working["post_date"].dt.hour).clip(lower=0, upper=23)
    working["month"] = working["month"].fillna(working["post_date"].dt.month).clip(lower=1, upper=12)
    working["caption_length"] = working["caption_length"].fillna(working["caption_text"].str.len())
    working["likes_count"] = working["likes_count"].fillna(0).clip(lower=0)
    working["comments_count"] = working["comments_count"].fillna(
        working.groupby("business_name")["comments_count"].transform("median")
    )
    working["comments_count"] = working["comments_count"].fillna(0).clip(lower=0)
    working["views_count"] = working["views_count"].fillna(0).clip(lower=0)
    working["discount_percent"] = working["discount_percent"].fillna(0).clip(lower=0, upper=100)

    if "hashtags" not in working.columns:
        working["hashtags"] = np.nan
    parsed_hashtags = working["hashtags"].apply(_parse_hashtags)
    caption_hashtags = working["caption_text"].apply(_extract_hashtags)
    merged_hashtags: list[list[str]] = []
    for first, second in zip(parsed_hashtags.tolist(), caption_hashtags.tolist()):
        seen: set[str] = set()
        out_tags: list[str] = []
        for tag in first + second:
            if tag and tag not in seen:
                seen.add(tag)
                out_tags.append(tag)
        merged_hashtags.append(out_tags)
    working["hashtags"] = merged_hashtags
    working["hashtags_count"] = working["hashtags_count"].fillna(working["hashtags"].apply(len)).clip(lower=0)
    working["emoji_count"] = working["emoji_count"].fillna(0).clip(lower=0)

    for bcol in [
        "cta_present",
        "promo_post",
        "mentions_location",
        "religious_theme",
        "patriotic_theme",
        "arabic_dialect_style",
        "sponsored",
    ]:
        if bcol not in working.columns:
            working[bcol] = False
        working[bcol] = _normalize_bool_series(working[bcol])

    working["CTA_present"] = working["cta_present"]

    working["day_of_week"] = (
        working["day_of_week"].replace("Unknown", np.nan).fillna(working["post_date"].dt.day_name())
    )
    working["month"] = pd.to_numeric(working["month"], errors="coerce").fillna(working["post_date"].dt.month).clip(1, 12)

    dedupe_frame = working.copy()
    if "hashtags" in dedupe_frame.columns:
        dedupe_frame["hashtags"] = dedupe_frame["hashtags"].apply(
            lambda x: tuple(x) if isinstance(x, list) else x
        )
    dedupe_mask = ~dedupe_frame.duplicated()
    cleaned = working.loc[dedupe_mask].reset_index(drop=True)
    missing_before = original.isna().sum().to_dict()
    missing_after = cleaned.isna().sum().to_dict()

    consistency_warnings: list[str] = []
    if (cleaned["posting_hour"] < 0).any() or (cleaned["posting_hour"] > 23).any():
        consistency_warnings.append("Some posting_hour values are outside 0..23 after cleaning.")
    if (cleaned["discount_percent"] < 0).any() or (cleaned["discount_percent"] > 100).any():
        consistency_warnings.append("Some discount_percent values are outside 0..100 after cleaning.")
    if cleaned["business_name"].eq("Unknown Business").mean() > 0.5:
        consistency_warnings.append("More than 50% of rows have unknown business_name.")

    report = {
        "rows_before": int(len(original)),
        "rows_after": int(len(cleaned)),
        "rows_dropped_invalid_post_date": int(dropped_bad_dates),
        "deduplicated_rows": int(len(working) - len(cleaned)),
        "mapped_columns": mapped_columns,
        "missing_before": {k: int(v) for k, v in missing_before.items()},
        "missing_after": {k: int(v) for k, v in missing_after.items()},
        "consistency_warnings": consistency_warnings,
        "output_columns": list(cleaned.columns),
    }
    return cleaned, report


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return _safe_json_records(df)
