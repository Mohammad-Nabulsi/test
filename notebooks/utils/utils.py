from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DEFAULT = PROJECT_ROOT / "synthetic_generator" / "synthetic_social_media_posts.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

EXPECTED_COLUMNS = [
    "business_name",
    "sector",
    "followers_count",
    "post_date",
    "posting_hour",
    "day_of_week",
    "month",
    "post_type",
    "caption_text",
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
]


def ensure_project_dirs():
    for d in [PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def safe_divide(numerator, denominator, default=0.0):
    n = np.asarray(numerator, dtype=float)
    d = np.asarray(denominator, dtype=float)
    return np.where(d == 0, default, n / d)


def load_raw_dataset(path=None):
    csv_path = Path(path) if path is not None else RAW_DATA_DEFAULT
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    return pd.read_csv(csv_path)


def _normalize_bool_series(s: pd.Series) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    return s.astype(str).str.strip().str.lower().map(mapping).fillna(False).astype(bool)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in EXPECTED_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out["post_date"] = pd.to_datetime(out["post_date"], errors="coerce")
    out = out.dropna(subset=["post_date"]).copy()

    default_cat = {
        "business_name": "Unknown Business",
        "sector": "Unknown",
        "day_of_week": "Unknown",
        "post_type": "unknown",
        "caption_text": "",
        "language": "Unknown",
    }
    for col, default in default_cat.items():
        out[col] = out[col].fillna(default).astype(str).str.strip()
        out.loc[out[col] == "", col] = default

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
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["followers_count"] = out["followers_count"].fillna(out["followers_count"].median()).clip(lower=0)
    out["posting_hour"] = out["posting_hour"].fillna(out["post_date"].dt.hour).clip(lower=0, upper=23)
    out["month"] = out["month"].fillna(out["post_date"].dt.month).clip(lower=1, upper=12)
    out["caption_length"] = out["caption_length"].fillna(out["caption_text"].str.len())
    out["hashtags_count"] = out["hashtags_count"].fillna(0).clip(lower=0)
    out["emoji_count"] = out["emoji_count"].fillna(0).clip(lower=0)
    out["likes_count"] = out["likes_count"].fillna(0).clip(lower=0)
    out["comments_count"] = out["comments_count"].fillna(0).clip(lower=0)
    out["views_count"] = out["views_count"].fillna(0).clip(lower=0)
    out["discount_percent"] = out["discount_percent"].fillna(0).clip(lower=0, upper=100)

    out["day_of_week"] = out["day_of_week"].replace("Unknown", np.nan).fillna(out["post_date"].dt.day_name())

    bool_cols = [
        "CTA_present",
        "promo_post",
        "mentions_location",
        "religious_theme",
        "patriotic_theme",
        "arabic_dialect_style",
    ]
    for col in bool_cols:
        out[col] = _normalize_bool_series(out[col])

    return out.drop_duplicates().reset_index(drop=True)
