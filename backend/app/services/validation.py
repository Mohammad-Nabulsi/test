from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd

from app.schemas import ValidationIssue, ValidationReport
from app.utils.constants import REQUIRED_COLUMNS, VALID_LANGUAGES, VALID_POST_TYPES


def _issue(
    type_: str,
    message: str,
    column: Optional[str] = None,
    count: Optional[int] = None,
    examples: Optional[List[Any]] = None,
) -> ValidationIssue:
    return ValidationIssue(type=type_, message=message, column=column, count=count, examples=examples)


def validate_dataframe(df: pd.DataFrame) -> ValidationReport:
    issues: list[ValidationIssue] = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return ValidationReport(
            ok=False,
            dataset_rows=int(len(df)),
            dataset_columns=int(df.shape[1]),
            missing_required_columns=missing_cols,
            issues=[_issue("schema", "Missing required columns.", count=len(missing_cols), examples=missing_cols[:20])],
        )

    # Missing values check (only on required columns)
    na_counts = df[REQUIRED_COLUMNS].isna().sum()
    high_na = na_counts[na_counts > 0].sort_values(ascending=False)
    if not high_na.empty:
        for col, cnt in high_na.items():
            issues.append(_issue("missing", f"Column has missing values: {col}", column=col, count=int(cnt)))

    # Duplicate rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues.append(_issue("duplicate", "Dataset contains duplicate rows.", count=dup_count))

    # Numeric ranges
    def _invalid_numeric(col: str, cond, msg: str):
        bad = df.loc[cond, col]
        if len(bad) > 0:
            ex = bad.dropna().astype(str).head(5).tolist()
            issues.append(_issue("range", msg, column=col, count=int(len(bad)), examples=ex))

    # Coerce to numeric for checks (do not mutate df)
    num = {}
    for c in [
        "followers_count",
        "posting_hour",
        "month",
        "likes_count",
        "comments_count",
        "views_count",
        "discount_percent",
    ]:
        num[c] = pd.to_numeric(df[c], errors="coerce")
        if num[c].isna().any():
            n = int(num[c].isna().sum())
            issues.append(_issue("type", "Non-numeric values found where numeric expected.", column=c, count=n))

    _invalid_numeric("followers_count", num["followers_count"] < 0, "followers_count must be >= 0")
    _invalid_numeric(
        "posting_hour",
        (num["posting_hour"] < 0) | (num["posting_hour"] > 23),
        "posting_hour must be between 0 and 23",
    )
    _invalid_numeric("month", (num["month"] < 1) | (num["month"] > 12), "month must be between 1 and 12")
    for c in ["likes_count", "comments_count", "views_count"]:
        _invalid_numeric(c, num[c] < 0, f"{c} must be >= 0")
    _invalid_numeric(
        "discount_percent",
        (num["discount_percent"] < 0) | (num["discount_percent"] > 100),
        "discount_percent must be between 0 and 100",
    )

    # Categorical values
    post_types = set(df["post_type"].dropna().astype(str).str.lower().unique().tolist())
    invalid_post_types = sorted([p for p in post_types if p not in VALID_POST_TYPES])
    if invalid_post_types:
        issues.append(
            _issue(
                "category",
                f"Invalid post_type values. Expected one of: {sorted(VALID_POST_TYPES)}",
                column="post_type",
                count=len(invalid_post_types),
                examples=invalid_post_types[:20],
            )
        )

    langs = set(df["language"].dropna().astype(str).unique().tolist())
    invalid_langs = sorted([l for l in langs if l not in VALID_LANGUAGES])
    if invalid_langs:
        issues.append(
            _issue(
                "category",
                f"Invalid language values. Expected one of: {sorted(VALID_LANGUAGES)}",
                column="language",
                count=len(invalid_langs),
                examples=invalid_langs[:20],
            )
        )

    ok = (len(missing_cols) == 0) and all(i.type != "schema" for i in issues)
    return ValidationReport(
        ok=ok,
        dataset_rows=int(len(df)),
        dataset_columns=int(df.shape[1]),
        missing_required_columns=missing_cols,
        issues=issues,
    )

