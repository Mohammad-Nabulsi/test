from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class HashtagStageSpec:
    required_inputs: list[str]
    optional_inputs: list[str]
    outputs: list[str]


HASHTAG_STAGE_SPEC = HashtagStageSpec(
    required_inputs=["hashtags OR caption_text"],
    optional_inputs=[
        "business_name",
        "sector",
        "post_type",
        "views_count",
        "followers_count",
        "likes_count",
        "comments_count",
        "performance_label",
    ],
    outputs=[
        "hashtag_frequency",
        "all_rules",
        "recommended_rules",
        "warning_rules",
        "top_recommendations",
        "fallback_recommendations",
        "category_recommendations",
        "summary",
    ],
)


def _load_hashtag_module() -> Any:
    module_path = Path(__file__).resolve().parents[3] / "notebooks" / "hashtag_association_recommendations.py"
    spec = importlib.util.spec_from_file_location("hashtag_association_recommendations", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load hashtag module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(v) for v in sorted(value, key=lambda x: str(x))]
    return value


def run_hashtag_stage(
    kpi_df: pd.DataFrame,
    category_cols: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    module = _load_hashtag_module()
    result = module.generate_hashtag_association_recommendations(
        kpi_df,
        category_cols=category_cols or ["sector", "post_type", "business_name"],
        output_dir=None,
        top_k=10,
    )

    used_fallback = False
    summary_df = result.get("summary", pd.DataFrame())
    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty and {"metric", "value"}.issubset(summary_df.columns):
        fallback_rows = summary_df.loc[summary_df["metric"] == "used_fallback", "value"]
        if not fallback_rows.empty:
            used_fallback = str(fallback_rows.iloc[0]).strip().lower() in {"true", "1", "yes"}

    frames = {
        "prepared_posts_with_labels": result["prepared_df"],
        "hashtag_frequency": result["hashtag_frequency"],
        "hashtag_association_all_rules": result["all_rules"],
        "hashtag_association_recommended_rules": result["recommended_rules"],
        "hashtag_association_warning_rules": result["warning_rules"],
        "hashtag_association_top_recommendations": result["top_recommendations"],
        "hashtag_association_fallback_recommendations": result["fallback_recommendations"],
        "hashtag_association_category_recommendations": result["category_recommendations"],
        "hashtag_recommendation_summary": result["summary"],
    }

    response = {
        "thresholds": result.get("thresholds", {}),
        "diagnostics": result.get("diagnostics", {}),
        "used_fallback": used_fallback,
        "summary": summary_df.to_dict(orient="records") if isinstance(summary_df, pd.DataFrame) else [],
        "top_recommendations": result.get("top_recommendations", pd.DataFrame()).to_dict(orient="records"),
        "warning_recommendations": result.get("warning_rules", pd.DataFrame()).to_dict(orient="records"),
    }
    return _jsonable(response), frames
