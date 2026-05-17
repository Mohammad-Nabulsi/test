from __future__ import annotations

import hashlib
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, UploadFile

from app.config import settings
from app.schemas import MeVisibleInfoResponse
from app.utils.file_utils import safe_read_csv


router = APIRouter(prefix="/me")

MISSING_AR = "غير محدد"

COLUMN_ALIASES: dict[str, list[str]] = {
    "business_name": ["business_name", "business", "brand_name", "account_name"],
    "sector": ["sector", "category", "industry"],
    "location": ["location", "business_location", "city", "region", "country"],
    "followers_count": ["followers_count", "followers", "follower_count"],
    "post_date": ["post_date", "date", "posted_at", "created_at"],
    "posting_hour": ["posting_hour", "hour", "post_hour"],
    "post_type": ["post_type", "content_type", "type"],
    "engagement_rate": ["engagement_rate", "engagement_rate_followers", "er"],
    "likes_count": ["likes_count", "likes", "like_count"],
    "comments_count": ["comments_count", "comments", "comment_count"],
    "views_count": ["views_count", "views", "view_count"],
}


AR_LABELS = {
    "engagement": "نسبة التفاعل",
    "likes": "متوسط اللايكات",
    "comments": "متوسط الكومنتات",
    "views": "متوسط مشاهدات الريلز",
    "growth": "نمو المتابعين",
    "frequency": "بوستات بالأسبوع",
    "besttime": "أفضل وقت للنشر",
    "topcontent": "أقوى نوع محتوى",
}


CONTENT_TYPE_AR = {
    "reel": "ريلز",
    "reels": "ريلز",
    "video": "فيديو",
    "image": "صورة",
    "photo": "صورة",
    "carousel": "كاروسيل",
}


def _norm_col(name: str) -> str:
    return str(name).strip().lower()


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    normalized_to_original = {_norm_col(c): c for c in out.columns}
    rename_map: dict[str, str] = {}

    for canonical, candidates in COLUMN_ALIASES.items():
        if canonical in out.columns:
            continue
        for cand in candidates:
            hit = normalized_to_original.get(_norm_col(cand))
            if hit and canonical not in out.columns:
                rename_map[hit] = canonical
                break

    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def _num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _spark(values: pd.Series, length: int = 10) -> list[float]:
    arr = values.dropna().astype(float).tolist()
    if not arr:
        return [0.0] * length
    arr = arr[-length:]
    if len(arr) < length:
        arr = [arr[0]] * (length - len(arr)) + arr
    return [round(float(v), 4) for v in arr]


def _delta_pct(values: pd.Series) -> float:
    arr = values.dropna().astype(float).tolist()
    if len(arr) < 4:
        return 0.0
    last = np.mean(arr[-3:])
    prev = np.mean(arr[-6:-3]) if len(arr) >= 6 else np.mean(arr[:-3])
    if prev == 0:
        return 0.0
    return round(((last - prev) / abs(prev)) * 100.0, 2)


def _mode_str(df: pd.DataFrame, col: str, default: str) -> str:
    if col not in df.columns:
        return default
    s = df[col].dropna().astype(str).str.strip().replace("", np.nan).dropna()
    mode = s.mode()
    if mode.empty:
        return default
    return str(mode.iloc[0])


def _handle_from_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", ".", str(name).strip().lower())
    slug = re.sub(r"\.+", ".", slug).strip(".")
    return f"@{slug}" if slug else "@business"


def _avatar_color(name: str) -> str:
    palette = [
        "from-fuchsia-500 to-violet-600",
        "from-sky-500 to-indigo-600",
        "from-amber-500 to-orange-600",
        "from-emerald-500 to-teal-600",
        "from-rose-500 to-pink-600",
    ]
    digest = hashlib.md5(str(name).strip().lower().encode("utf-8")).hexdigest()
    return palette[int(digest[:8], 16) % len(palette)]


def _fmt_hour(hour: int) -> str:
    suffix = "صباحًا" if hour < 12 else "مساءً"
    hour12 = hour % 12
    if hour12 == 0:
        hour12 = 12
    return f"{hour12}:00 {suffix}"


def _best_posting_time(df: pd.DataFrame) -> str:
    if "posting_hour" not in df.columns:
        return MISSING_AR

    hours_num = pd.to_numeric(df["posting_hour"], errors="coerce")
    if "engagement_rate" in df.columns:
        tmp = pd.DataFrame(
            {
                "hour": hours_num,
                "engagement_rate": pd.to_numeric(df["engagement_rate"], errors="coerce"),
            }
        ).dropna(subset=["hour", "engagement_rate"])
        if not tmp.empty:
            tmp["hour"] = tmp["hour"].astype(int)
            by_hour = tmp.groupby("hour", as_index=False)["engagement_rate"].mean()
            best_hour = int(by_hour.sort_values("engagement_rate", ascending=False).iloc[0]["hour"])
            return _fmt_hour(best_hour)

    hours = hours_num.dropna().astype(int)
    if hours.empty:
        return MISSING_AR
    return _fmt_hour(int(hours.mode().iloc[0]))


def _top_content_type(df: pd.DataFrame) -> str:
    top: str
    if "post_type" in df.columns and "engagement_rate" in df.columns:
        tmp = df.copy()
        tmp["post_type"] = tmp["post_type"].astype(str).str.strip().replace("", np.nan)
        tmp["engagement_rate"] = pd.to_numeric(tmp["engagement_rate"], errors="coerce")
        tmp = tmp.dropna(subset=["post_type", "engagement_rate"])
        if not tmp.empty:
            by_type = tmp.groupby("post_type", as_index=False)["engagement_rate"].mean()
            top = str(by_type.sort_values("engagement_rate", ascending=False).iloc[0]["post_type"]).lower()
        else:
            top = _mode_str(df, "post_type", MISSING_AR).lower()
    else:
        top = _mode_str(df, "post_type", MISSING_AR).lower()

    return CONTENT_TYPE_AR.get(top, top if top else MISSING_AR)


def _followers_growth_pct(df: pd.DataFrame) -> float:
    if "followers_count" not in df.columns:
        return 0.0
    work = df.copy()
    work["followers_count"] = pd.to_numeric(work["followers_count"], errors="coerce")
    if "post_date" in work.columns:
        work["post_date"] = pd.to_datetime(work["post_date"], errors="coerce")
        work = work.sort_values("post_date")
    s = work["followers_count"].dropna()
    if len(s) < 2:
        return 0.0
    start = float(s.iloc[0])
    end = float(s.iloc[-1])
    if start <= 0:
        return 0.0
    return round(((end - start) / start) * 100.0, 2)


def _posts_per_week(df: pd.DataFrame) -> float:
    total = float(len(df))
    if total == 0:
        return 0.0
    if "post_date" not in df.columns:
        return round(total / 4.0, 2)
    d = pd.to_datetime(df["post_date"], errors="coerce").dropna()
    if d.empty:
        return round(total / 4.0, 2)
    days = max(1, int((d.max() - d.min()).days) + 1)
    return round(total * 7.0 / days, 2)


def _sorted_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "post_date" in out.columns:
        out["post_date"] = pd.to_datetime(out["post_date"], errors="coerce")
        out = out.sort_values("post_date")
    return out


def _series_sorted_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    if "post_date" not in df.columns:
        return _num_series(df, col)
    tmp = pd.DataFrame(
        {
            "post_date": pd.to_datetime(df["post_date"], errors="coerce"),
            col: pd.to_numeric(df[col], errors="coerce"),
        }
    ).sort_values("post_date")
    return tmp[col]


def _resolve_me_source_path(dataset_id: str) -> Path:
    root = Path(settings.storage_path())
    candidates = [
        root / "raw" / dataset_id / "raw.csv",
        root / "cleaned" / dataset_id / "cleaned_dataset.csv",
        root / "outputs" / dataset_id / "kpi_dataset.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return root / "raw" / dataset_id / "raw.csv"


def _ensure_derived_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "engagement_rate" not in out.columns:
        likes = pd.to_numeric(out.get("likes_count", 0), errors="coerce").fillna(0.0)
        comments = pd.to_numeric(out.get("comments_count", 0), errors="coerce").fillna(0.0)
        followers = pd.to_numeric(out.get("followers_count", 0), errors="coerce").replace(0, np.nan)
        out["engagement_rate"] = ((likes + comments) / followers).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _followers_value(df: pd.DataFrame) -> int:
    s = _num_series(df, "followers_count")
    if s.dropna().empty:
        return 0
    if "post_date" not in df.columns:
        return int(s.dropna().max())
    tmp = pd.DataFrame(
        {
            "post_date": pd.to_datetime(df["post_date"], errors="coerce"),
            "followers_count": s,
        }
    ).sort_values("post_date")
    last_valid = tmp["followers_count"].dropna()
    if not last_valid.empty:
        return int(last_valid.iloc[-1])
    return int(s.dropna().max())


@router.get("/{dataset_id}", response_model=MeVisibleInfoResponse)
def get_me_visible_info(
    dataset_id: str,
    business_name: Optional[str] = Query(
        default=None,
        description="Optional business name filter. If omitted, the most frequent business is used.",
    ),
) -> MeVisibleInfoResponse:
    if not isinstance(business_name, (str, type(None))):
        business_name = None

    source_path = _resolve_me_source_path(dataset_id)
    if not source_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    try:
        df = safe_read_csv(source_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded dataset is empty.")

    work = _canonicalize_columns(df.copy())
    work = work.replace(r"^\s*$", np.nan, regex=True)
    work = _ensure_derived_kpis(work)

    if "business_name" in work.columns and not work["business_name"].dropna().empty:
        if business_name:
            target = business_name.strip()
            filtered = work[work["business_name"].astype(str).str.strip() == target]
            if filtered.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"business_name '{business_name}' not found in dataset.",
                )
            work = filtered
        else:
            top_business = work["business_name"].astype(str).str.strip().value_counts().index[0]
            work = work[work["business_name"].astype(str).str.strip() == top_business]

    sorted_work = _sorted_df(work)

    biz_name = _mode_str(sorted_work, "business_name", "Business")
    category = _mode_str(sorted_work, "sector", MISSING_AR)
    location = _mode_str(sorted_work, "location", MISSING_AR)

    followers_series = _num_series(sorted_work, "followers_count")
    followers = _followers_value(sorted_work)
    followers_growth = _followers_growth_pct(sorted_work)
    posts = int(len(sorted_work))

    engagement_series = _series_sorted_by_date(sorted_work, "engagement_rate")
    engagement_value = float(engagement_series.mean()) if not engagement_series.dropna().empty else 0.0
    if 0 < engagement_value <= 1:
        engagement_value *= 100.0
    engagement_value = round(engagement_value, 2)

    likes_series = _series_sorted_by_date(sorted_work, "likes_count")
    comments_series = _series_sorted_by_date(sorted_work, "comments_count")

    views_base = sorted_work
    if "post_type" in sorted_work.columns:
        post_type_norm = sorted_work["post_type"].astype(str).str.strip().str.lower()
        reels_videos = sorted_work[post_type_norm.isin(["reel", "reels", "video"])]
        if not reels_videos.empty:
            views_base = reels_videos
    views_series = _series_sorted_by_date(views_base, "views_count")

    likes_value = int(round(float(likes_series.mean()))) if not likes_series.dropna().empty else 0
    comments_value = int(round(float(comments_series.mean()))) if not comments_series.dropna().empty else 0
    views_value = int(round(float(views_series.mean()))) if not views_series.dropna().empty else 0

    frequency_value = _posts_per_week(sorted_work)
    best_time = _best_posting_time(sorted_work)
    top_content = _top_content_type(sorted_work)
    followers_spark_series = _series_sorted_by_date(sorted_work, "followers_count")

    return MeVisibleInfoResponse(
        business={
            "name": biz_name,
            "nameEn": biz_name,
            "handle": _handle_from_name(biz_name),
            "category": category,
            "location": location,
            "followers": followers,
            "followersGrowth": followers_growth,
            "posts": posts,
            "avatarColor": _avatar_color(biz_name),
        },
        kpis=[
            {
                "key": "engagement",
                "label": AR_LABELS["engagement"],
                "value": engagement_value,
                "suffix": "%",
                "delta": _delta_pct(engagement_series),
                "spark": _spark(engagement_series),
            },
            {
                "key": "likes",
                "label": AR_LABELS["likes"],
                "value": likes_value,
                "delta": _delta_pct(likes_series),
                "spark": _spark(likes_series),
            },
            {
                "key": "comments",
                "label": AR_LABELS["comments"],
                "value": comments_value,
                "delta": _delta_pct(comments_series),
                "spark": _spark(comments_series),
            },
            {
                "key": "views",
                "label": AR_LABELS["views"],
                "value": views_value,
                "delta": _delta_pct(views_series),
                "spark": _spark(views_series),
            },
            {
                "key": "growth",
                "label": AR_LABELS["growth"],
                "value": followers_growth,
                "suffix": "%",
                "delta": _delta_pct(followers_spark_series),
                "spark": _spark(followers_spark_series),
            },
            {
                "key": "frequency",
                "label": AR_LABELS["frequency"],
                "value": frequency_value,
                "delta": 0.0,
                "spark": _spark(pd.Series([frequency_value] * 10, dtype=float)),
            },
            {
                "key": "besttime",
                "label": AR_LABELS["besttime"],
                "value": best_time,
                "delta": 0.0,
                "spark": _spark(engagement_series),
                "string": True,
            },
            {
                "key": "topcontent",
                "label": AR_LABELS["topcontent"],
                "value": top_content,
                "delta": 0.0,
                "spark": _spark(views_series),
                "string": True,
            },
        ],
    )


@router.get("/{dataset_id}/columns")
def get_me_columns_debug(dataset_id: str) -> dict[str, Any]:
    source_path = _resolve_me_source_path(dataset_id)
    if not source_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    try:
        df = safe_read_csv(source_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    mapped = _canonicalize_columns(df.copy()).replace(r"^\s*$", np.nan, regex=True)

    expected = list(COLUMN_ALIASES.keys())
    missing_expected = [c for c in expected if c not in mapped.columns]

    business_names: list[str] = []
    if "business_name" in mapped.columns:
        business_names = (
            mapped["business_name"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .drop_duplicates()
            .head(50)
            .tolist()
        )

    return {
        "dataset_id": dataset_id,
        "source_file": str(source_path),
        "row_count": int(len(df)),
        "original_columns": [str(c) for c in df.columns],
        "detected_columns": [str(c) for c in mapped.columns],
        "missing_expected_columns": missing_expected,
        "business_names_sample": business_names,
    }


def _parse_uploaded_file(file: UploadFile) -> pd.DataFrame:
    content = file.file.read()
    filename = file.filename or "uploaded.csv"
    suffix = Path(filename).suffix.lower()
    if suffix == ".json":
        return pd.read_json(BytesIO(content))
    if suffix == ".csv":
        return pd.read_csv(BytesIO(content))
    raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'. Use .json or .csv.")


def _build_me_response(df: pd.DataFrame, business_name: str | None) -> MeVisibleInfoResponse:
    work = df.copy()
    if "business_name" in work.columns and not work["business_name"].dropna().empty:
        if business_name:
            filtered = work[work["business_name"].astype(str).str.strip() == business_name]
            if filtered.empty:
                raise HTTPException(status_code=404, detail=f"business_name '{business_name}' not found in dataset.")
            work = filtered
        else:
            top_business = work["business_name"].astype(str).str.strip().value_counts().index[0]
            work = work[work["business_name"].astype(str).str.strip() == top_business]

    biz_name = _mode_str(work, "business_name", "Business")
    category = _mode_str(work, "sector", "غير محدد")
    location = _mode_str(work, "location", "غير محدد")
    followers_series = _num_series(work, "followers_count")
    followers = int(followers_series.dropna().max()) if not followers_series.dropna().empty else 0
    followers_growth = _followers_growth_pct(work)
    posts = int(len(work))

    engagement_series = _num_series(work, "engagement_rate")
    engagement_value = float(engagement_series.mean()) if not engagement_series.dropna().empty else 0.0
    if 0 < engagement_value <= 1:
        engagement_value *= 100.0
    engagement_value = round(engagement_value, 2)

    likes_series = _num_series(work, "likes_count")
    comments_series = _num_series(work, "comments_count")
    views_series = _num_series(work, "views_count")

    likes_value = int(round(float(likes_series.mean()))) if not likes_series.dropna().empty else 0
    comments_value = int(round(float(comments_series.mean()))) if not comments_series.dropna().empty else 0
    views_value = int(round(float(views_series.mean()))) if not views_series.dropna().empty else 0

    frequency_value = _posts_per_week(work)
    best_time = _best_posting_time(work)
    top_content = _top_content_type(work)

    return MeVisibleInfoResponse(
        business={
            "name": biz_name,
            "nameEn": biz_name,
            "handle": _handle_from_name(biz_name),
            "category": category,
            "location": location,
            "followers": followers,
            "followersGrowth": followers_growth,
            "posts": posts,
            "avatarColor": _avatar_color(biz_name),
        },
        kpis=[
            {"key": "engagement", "label": "نسبة التفاعل", "value": engagement_value, "suffix": "%", "delta": _delta_pct(engagement_series), "spark": _spark(engagement_series)},
            {"key": "likes", "label": "متوسط اللايكات", "value": likes_value, "delta": _delta_pct(likes_series), "spark": _spark(likes_series)},
            {"key": "comments", "label": "متوسط الكومنتات", "value": comments_value, "delta": _delta_pct(comments_series), "spark": _spark(comments_series)},
            {"key": "views", "label": "متوسط مشاهدات الريلز", "value": views_value, "delta": _delta_pct(views_series), "spark": _spark(views_series)},
            {"key": "growth", "label": "نمو المتابعين", "value": followers_growth, "suffix": "%", "delta": _delta_pct(followers_series), "spark": _spark(followers_series)},
            {"key": "frequency", "label": "بوستات بالأسبوع", "value": frequency_value, "delta": 0.0, "spark": _spark(pd.Series([frequency_value] * 10, dtype=float))},
            {"key": "besttime", "label": "أفضل وقت للنشر", "value": best_time, "delta": 0.0, "spark": _spark(engagement_series), "string": True},
            {"key": "topcontent", "label": "أقوى نوع محتوى", "value": top_content, "delta": 0.0, "spark": _spark(views_series), "string": True},
        ],
    )


@router.post("/upload", response_model=MeVisibleInfoResponse)
def upload_and_get_me(
    file: UploadFile,
    business_name: Optional[str] = Query(default=None, description="Optional business name filter."),
) -> MeVisibleInfoResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")
    try:
        df = _parse_uploaded_file(file)
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded dataset is empty.")

        dataset_id = str(uuid.uuid4())
        raw_dir = settings.storage_path() / "raw" / dataset_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(file.filename or "uploaded.csv").suffix.lower() or ".csv"
        file.file.seek(0)
        (raw_dir / f"raw{suffix}").write_bytes(file.file.read())

        return _build_me_response(df, business_name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Me analysis failed: {e}")
