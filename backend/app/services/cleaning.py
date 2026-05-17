from __future__ import annotations

from typing import Tuple

import pandas as pd


def _to_bool_series(s: pd.Series) -> pd.Series:
    # Accept: True/False, 1/0, yes/no, y/n, "true"/"false"
    if s.dtype == bool:
        return s
    v = s.astype(str).str.strip().str.lower()
    true_vals = {"true", "1", "yes", "y", "t"}
    false_vals = {"false", "0", "no", "n", "f"}
    out = v.map(lambda x: True if x in true_vals else (False if x in false_vals else None))
    return out.astype("boolean")


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    notes: dict = {"dropped_rows": 0, "drop_reasons": {}}
    out = df.copy()

    # Parse date
    out["post_date"] = pd.to_datetime(out["post_date"], errors="coerce")
    bad_date = out["post_date"].isna().sum()
    if bad_date:
        notes["drop_reasons"]["invalid_post_date"] = int(bad_date)

    # Fill defaults
    out["views_count"] = pd.to_numeric(out["views_count"], errors="coerce").fillna(0).clip(lower=0)
    out["discount_percent"] = pd.to_numeric(out["discount_percent"], errors="coerce").fillna(0).clip(lower=0, upper=100)

    # Normalize booleans
    for bcol in [
        "CTA_present",
        "promo_post",
        "mentions_location",
        "religious_theme",
        "patriotic_theme",
        "arabic_dialect_style",
    ]:
        out[bcol] = _to_bool_series(out[bcol]).fillna(False).astype(bool)

    # Standardize casing
    out["sector"] = out["sector"].astype(str).str.strip().str.title()
    out["business_name"] = out["business_name"].astype(str).str.strip()
    out["post_type"] = out["post_type"].astype(str).str.strip().str.lower()
    out["language"] = out["language"].astype(str).str.strip()
    out["day_of_week"] = out["day_of_week"].astype(str).str.strip().str.title()

    # Numeric conversions
    for col in [
        "followers_count",
        "posting_hour",
        "month",
        "caption_length",
        "hashtags_count",
        "emoji_count",
        "likes_count",
        "comments_count",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Drop impossible rows
    before = len(out)
    mask_ok = (
        out["followers_count"].notna()
        & (out["followers_count"] >= 0)
        & out["posting_hour"].notna()
        & (out["posting_hour"].between(0, 23))
        & out["month"].notna()
        & (out["month"].between(1, 12))
        & out["likes_count"].notna()
        & (out["likes_count"] >= 0)
        & out["comments_count"].notna()
        & (out["comments_count"] >= 0)
        & (out["views_count"] >= 0)
        & out["post_date"].notna()
    )
    out = out.loc[mask_ok].copy()
    dropped = before - len(out)
    notes["dropped_rows"] = int(dropped)

    out = out.reset_index(drop=True)
    return out, notes

