from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import math
import pandas as pd
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["anomalies-single"])


class UploadedDatasetRequest(BaseModel):
    uploaded_file_path: str


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _storage_root() -> Path:
    return (_backend_root() / "storage").resolve()


def _anomaly_outputs_root() -> Path:
    return (_storage_root() / "outputs" / "anomaly_outputs").resolve()


def _safe_dataset_name(path: Path, original_input: str) -> str:
    candidate = path.stem
    if candidate.lower() == "raw" and path.parent.name:
        candidate = path.parent.name
    if not candidate:
        candidate = Path(str(original_input)).stem
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", candidate).strip("_").lower()
    return safe or "uploaded_dataset"


def _resolve_dataset_path(uploaded_file_path: str) -> Path:
    p = Path(uploaded_file_path)
    if p.is_absolute() and p.exists():
        return p.resolve()

    project_root = _project_root()
    backend_root = _backend_root()
    candidates = [
        backend_root / uploaded_file_path,
        backend_root / "data" / uploaded_file_path,
        project_root / uploaded_file_path,
        project_root / "data" / uploaded_file_path,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(f"Uploaded file not found: {uploaded_file_path}")


def _load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported dataset format: {suffix}")


def _as_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []

    def clean(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if pd.isna(value):
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return value

    return [{key: clean(value) for key, value in row.items()} for row in df.to_dict(orient="records")]


def _median_numeric(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns or df.empty:
        return None
    value = pd.to_numeric(df[column], errors="coerce").median()
    if pd.isna(value):
        return None
    return float(value)


def _mean_numeric(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns or df.empty:
        return None
    value = pd.to_numeric(df[column], errors="coerce").mean()
    if pd.isna(value):
        return None
    return float(value)


def _mode_text(df: pd.DataFrame, column: str) -> str | None:
    if column not in df.columns or df.empty:
        return None
    values = df[column].dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return None
    return str(values.mode().iloc[0])


def _describe_patterns(df: pd.DataFrame, polarity: str) -> List[str]:
    if df.empty:
        return []

    patterns: List[str] = []
    post_type = _mode_text(df, "post_type")
    if post_type:
        patterns.append(f"mostly {post_type} posts")

    captions = _median_numeric(df, "caption_length")
    if captions is not None:
        if captions <= 40:
            patterns.append("short captions")
        elif captions >= 160:
            patterns.append("longer captions")

    hashtags = _median_numeric(df, "hashtags_count")
    if hashtags is not None:
        if hashtags >= 6:
            patterns.append("heavy hashtag use")
        elif hashtags <= 1:
            patterns.append("few hashtags")

    emojis = _median_numeric(df, "emoji_count")
    if emojis is not None:
        if emojis >= 3:
            patterns.append("emoji-rich captions")
        elif emojis == 0:
            patterns.append("no emoji use")

    if "promo_post" in df.columns:
        promo = df["promo_post"].astype(str).str.lower().isin({"true", "1", "yes", "y"}).mean()
        if promo >= 0.6:
            patterns.append("mostly promotional posts")
        elif promo <= 0.2:
            patterns.append("mostly non-promotional posts")

    views = _median_numeric(df, "views_count")
    likes = _median_numeric(df, "likes_count")
    comments = _median_numeric(df, "comments_count")
    engagement = _median_numeric(df, "engagement_rate")
    if views is not None:
        patterns.append(f"median views around {views:.0f}")
    if likes is not None:
        patterns.append(f"median likes around {likes:.0f}")
    if comments is not None:
        patterns.append(f"median comments around {comments:.0f}")
    if engagement is not None:
        label = "high" if polarity == "positive" else "low"
        patterns.append(f"{label} median engagement rate of {engagement:.4f}")

    return patterns[:6]


def _title_for_pattern(pattern: str, rec_type: str) -> str:
    base = pattern.replace("mostly ", "").replace("median ", "").strip()
    if "post" in base and rec_type == "do_more":
        return f"Use {base} more often"
    if "post" in base:
        return f"Reduce reliance on {base}"
    if rec_type == "do_more":
        return f"Lean into {base}"
    return f"Avoid {base}"


def _generate_recommendations(top_pos: pd.DataFrame, top_neg: pd.DataFrame) -> List[Dict[str, str]]:
    recommendations: List[Dict[str, str]] = []
    positive_patterns = _describe_patterns(top_pos, "positive")
    negative_patterns = _describe_patterns(top_neg, "negative")

    if not top_pos.empty:
        pattern = positive_patterns[0] if positive_patterns else "the strongest anomaly pattern"
        detail = ", ".join(positive_patterns[:4]) if positive_patterns else "they show above-normal engagement"
        recommendations.append(
            {
                "type": "do_more",
                "title": _title_for_pattern(pattern, "do_more"),
                "reason": f"Top positive anomalies are {detail}.",
            }
        )

    if not top_neg.empty:
        pattern = negative_patterns[0] if negative_patterns else "the weakest anomaly pattern"
        detail = ", ".join(negative_patterns[:4]) if negative_patterns else "they show below-normal engagement"
        recommendations.append(
            {
                "type": "avoid",
                "title": _title_for_pattern(pattern, "avoid"),
                "reason": f"Top negative anomalies are {detail}.",
            }
        )

    return recommendations


def _generate_sector_summary(sector: str, top_pos: pd.DataFrame, top_neg: pd.DataFrame) -> List[Dict[str, Any]]:
    if top_pos.empty and top_neg.empty:
        return []
    return [
        {
            "sector": sector,
            "positive_count": int(len(top_pos)),
            "negative_count": int(len(top_neg)),
            "avg_positive_engagement_rate": _mean_numeric(top_pos, "engagement_rate"),
            "avg_negative_engagement_rate": _mean_numeric(top_neg, "engagement_rate"),
            "best_post_type": _mode_text(top_pos, "post_type"),
            "weakest_post_type": _mode_text(top_neg, "post_type"),
            "common_positive_patterns": _describe_patterns(top_pos, "positive"),
            "common_negative_patterns": _describe_patterns(top_neg, "negative"),
        }
    ]


def _url_for_output(path: Path) -> str:
    rel = path.resolve().relative_to(_anomaly_outputs_root())
    return f"/anomaly_outputs/{rel.as_posix()}"


def _write_csv_outputs(
    output_dir: Path,
    top_pos: pd.DataFrame,
    top_neg: pd.DataFrame,
    recommendations: List[Dict[str, Any]],
    sector_summary: List[Dict[str, Any]],
) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "top_positive_anomalies.csv": top_pos,
        "top_negative_anomalies.csv": top_neg,
        "anomaly_recommendations.csv": pd.DataFrame(recommendations),
        "sector_anomaly_summary.csv": pd.DataFrame(sector_summary),
    }
    urls: List[str] = []
    for filename, frame in outputs.items():
        path = output_dir / filename
        frame.to_csv(path, index=False)
        urls.append(_url_for_output(path))
    return urls


def _write_anomaly_chart(output_dir: Path, business_name: str, top_pos: pd.DataFrame, top_neg: pd.DataFrame) -> str | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_df = pd.concat([top_pos, top_neg], ignore_index=True)
    if chart_df.empty or "engagement_rate" not in chart_df.columns or "views_count" not in chart_df.columns:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chart_df["engagement_rate"] = pd.to_numeric(chart_df["engagement_rate"], errors="coerce")
    chart_df["views_count"] = pd.to_numeric(chart_df["views_count"], errors="coerce")
    size_col = "likes_count" if "likes_count" in chart_df.columns else "comments_count" if "comments_count" in chart_df.columns else None
    if size_col:
        chart_df[size_col] = pd.to_numeric(chart_df[size_col], errors="coerce").fillna(0)

    chart_df = chart_df.dropna(subset=["views_count", "engagement_rate"]).copy()
    if chart_df.empty:
        return None

    if size_col:
        raw_sizes = chart_df[size_col].astype(float)
        min_size = float(raw_sizes.min())
        max_size = float(raw_sizes.max())
        if max_size > min_size:
            chart_df["bubble_size"] = 90 + ((raw_sizes - min_size) / (max_size - min_size)) * 460
        else:
            chart_df["bubble_size"] = 220
    else:
        chart_df["bubble_size"] = 220

    def short_label(value: Any) -> str:
        text = str(value or "").strip()
        if not text or text.lower() == "nan":
            return "Top anomaly"
        return text[:42].rstrip() + ("..." if len(text) > 42 else "")

    fig, ax = plt.subplots(figsize=(10, 6))
    groups = [
        ("positive_anomaly", "Positive Anomaly", "#178a5a"),
        ("negative_anomaly", "Negative Anomaly", "#c83c3c"),
    ]
    for anomaly_type, label, color in groups:
        group = chart_df[chart_df["anomaly_type"].astype(str) == anomaly_type]
        if group.empty:
            continue
        ax.scatter(
            group["views_count"].astype(float),
            group["engagement_rate"].astype(float),
            s=group["bubble_size"].astype(float),
            c=color,
            label=label,
            alpha=0.72,
            edgecolors="white",
            linewidths=1.2,
        )

    positive = chart_df[chart_df["anomaly_type"].astype(str) == "positive_anomaly"]
    negative = chart_df[chart_df["anomaly_type"].astype(str) == "negative_anomaly"]
    annotation_specs = []
    if not positive.empty:
        annotation_specs.append((positive.sort_values("engagement_rate", ascending=False).iloc[0], (12, 12)))
    if not negative.empty:
        annotation_specs.append((negative.sort_values("engagement_rate", ascending=True).iloc[0], (12, -22)))
    for row, offset in annotation_specs:
        ax.annotate(
            short_label(row.get("caption_text")),
            (float(row["views_count"]), float(row["engagement_rate"])),
            textcoords="offset points",
            xytext=offset,
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cccccc", "alpha": 0.88},
            arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.8},
        )

    ax.set_title(f"Anomaly Performance Map - {business_name}", fontsize=14, pad=14)
    ax.set_xlabel("Views Count")
    ax.set_ylabel("Engagement Rate")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)
    if size_col:
        ax.text(
            0.99,
            0.01,
            f"Bubble size: {size_col.replace('_', ' ')}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
    plt.tight_layout()
    chart_path = output_dir / "anomaly_chart.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    return _url_for_output(chart_path)


@router.post("/anomalies/analyze-single", status_code=status.HTTP_200_OK)
def analyze_anomalies_single(request: UploadedDatasetRequest) -> Dict[str, Any]:
    try:
        path = _resolve_dataset_path(request.uploaded_file_path)
        df = _load_dataset(path)

        if "engagement_rate" not in df.columns:
            raise ValueError("Dataset must include 'engagement_rate'.")

        for col in ["business_name", "sector", "post_type", "caption_text"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

        df["engagement_rate"] = pd.to_numeric(df["engagement_rate"], errors="coerce")
        for col in ["views_count", "likes_count", "comments_count", "caption_length", "hashtags_count", "emoji_count"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["engagement_rate"]).copy()
        if df.empty:
            return {
                "business_name": "",
                "sector": "",
                "message": "Anomaly detection finished but no valid engagement_rate values were found.",
                "top_positive_anomalies": [],
                "top_negative_anomalies": [],
                "recommendations": [],
                "sector_anomaly_summary": [],
                "chart_url": None,
                "csv_outputs": [],
            }

        business_name = (
            str(df["business_name"].dropna().iloc[0]).strip()
            if "business_name" in df.columns and not df["business_name"].dropna().empty
            else "Uploaded Dataset"
        )
        sector = (
            str(df["sector"].dropna().iloc[0]).strip()
            if "sector" in df.columns and not df["sector"].dropna().empty
            else ""
        )

        top_pos = df.sort_values("engagement_rate", ascending=False).head(5).copy()
        top_neg = df.sort_values("engagement_rate", ascending=True).head(5).copy()
        top_pos["anomaly_type"] = "positive_anomaly"
        top_neg["anomaly_type"] = "negative_anomaly"
        top_pos["best_method"] = "rank"
        top_neg["best_method"] = "rank"
        top_pos["best_setting"] = "top_5"
        top_neg["best_setting"] = "bottom_5"

        keep_cols = [
            "business_name",
            "sector",
            "post_date",
            "post_type",
            "caption_text",
            "engagement_rate",
            "views_count",
            "likes_count",
            "comments_count",
            "caption_length",
            "hashtags_count",
            "emoji_count",
            "promo_post",
            "anomaly_type",
            "best_method",
            "best_setting",
        ]
        top_pos = top_pos[[c for c in keep_cols if c in top_pos.columns]]
        top_neg = top_neg[[c for c in keep_cols if c in top_neg.columns]]

        recommendations = _generate_recommendations(top_pos, top_neg)
        sector_anomaly_summary = _generate_sector_summary(sector, top_pos, top_neg)
        output_dir = _anomaly_outputs_root() / _safe_dataset_name(path, request.uploaded_file_path)
        chart_url = None
        csv_outputs: List[str] = []
        warnings: List[str] = []

        try:
            chart_url = _write_anomaly_chart(output_dir, business_name, top_pos, top_neg)
        except Exception as exc:
            warnings.append(f"Chart generation failed: {type(exc).__name__}: {exc}")

        try:
            csv_outputs = _write_csv_outputs(output_dir, top_pos, top_neg, recommendations, sector_anomaly_summary)
        except Exception as exc:
            warnings.append(f"CSV output saving failed: {type(exc).__name__}: {exc}")

        message = "Anomaly detection completed successfully."
        if warnings:
            message = f"{message} Warnings: {'; '.join(warnings)}"

        return {
            "business_name": business_name,
            "sector": sector,
            "message": message,
            "top_positive_anomalies": _as_records(top_pos),
            "top_negative_anomalies": _as_records(top_neg),
            "recommendations": recommendations,
            "sector_anomaly_summary": sector_anomaly_summary,
            "chart_url": chart_url,
            "csv_outputs": csv_outputs,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@router.get("/anomalies/sector-summary-single", status_code=status.HTTP_200_OK)
def anomaly_sector_summary_single(sector: str) -> List[Dict[str, Any]]:
    _ = sector
    return []
