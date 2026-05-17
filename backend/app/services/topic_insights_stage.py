from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import time
import logging
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TopicStageSpec:
    required_inputs: list[str]
    optional_inputs: list[str]
    outputs: list[str]


TOPIC_STAGE_SPEC = TopicStageSpec(
    required_inputs=["JSON records with caption text (caption_text/caption/text/post_text)."],
    optional_inputs=[
        "hashtags",
        "engagement_rate",
        "view_rate",
        "likes_count",
        "comments_count",
        "views_count",
        "post_type",
        "language",
    ],
    outputs=[
        "summary",
        "topic_labels",
        "topic_kpis",
        "topic_recommendations",
        "insight_cards",
        "posts_with_topics_preview",
        "fit_attempts",
        "files",
    ],
)
logger = logging.getLogger(__name__)


@contextmanager
def _temporary_env(updates: dict[str, str | None]):
    old: dict[str, str | None] = {}
    try:
        for key, value in updates.items():
            old[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_business_topic_module() -> Any:
    module_path = Path(__file__).resolve().parents[3] / "notebooks" / "business_topic_insights_backend.py"
    spec = importlib.util.spec_from_file_location("business_topic_insights_backend", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load topic backend module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    return json.loads(out.to_json(orient="records", force_ascii=False))


def _records_to_df(records: Any) -> pd.DataFrame:
    if isinstance(records, list):
        return pd.DataFrame(records)
    return pd.DataFrame()


def _result_to_frames(result: dict[str, Any]) -> dict[str, pd.DataFrame]:
    return {
        "business_topic_labels": _records_to_df(result.get("topic_labels", [])),
        "business_topic_kpis": _records_to_df(result.get("topic_kpis", [])),
        "business_topic_recommendations": _records_to_df(result.get("topic_recommendations", [])),
        "business_topic_insight_cards": _records_to_df(result.get("insight_cards", [])),
        "business_topic_posts_preview": _records_to_df(result.get("posts_with_topics_preview", [])),
    }


def run_topic_insights_stage(
    kpi_df: pd.DataFrame,
    *,
    output_dir: str | None = None,
    include_posts: bool = False,
    require_openai: bool | None = None,
    disable_openai: bool = False,
    allow_kmeans_fallback: bool = True,
    require_bertopic: bool = False,
    bert_device: str | None = None,
    bert_model_dir: str | None = None,
    bert_local_only: bool | None = None,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """
    Run business-topic insights using notebooks/business_topic_insights_backend.py.

    OPENAI_API_KEY is automatically read from .env by that script.
    Set TOPIC_REQUIRE_OPENAI=1 to force hard failure when OpenAI key/client is missing.
    """
    module = _load_business_topic_module()
    require_openai_effective = (
        require_openai
        if require_openai is not None
        else os.getenv("TOPIC_REQUIRE_OPENAI", "0").strip().lower() in {"1", "true", "yes"}
    )
    started_at = time.perf_counter()
    logger.info(
        "topic-insights stage started rows=%s output_dir=%s require_openai=%s",
        len(kpi_df),
        output_dir or "business_topic_outputs",
        require_openai_effective,
    )

    records = _json_records(kpi_df)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False) as tmp:
        json.dump({"records": records}, tmp, ensure_ascii=False, indent=2)
        temp_input_path = tmp.name

    topic_output_dir = output_dir or "business_topic_outputs"
    try:
        env_updates = {
            "BERT_EMBEDDING_DEVICE": bert_device,
            "BERT_EMBEDDING_MODEL_DIR": bert_model_dir,
            "BERT_LOCAL_ONLY": None if bert_local_only is None else ("1" if bert_local_only else "0"),
        }
        with _temporary_env(env_updates):
            result = module.analyze_business_json(
                input_path=temp_input_path,
                output_dir=topic_output_dir,
                text_col=None,
                hashtags_col="hashtags",
                openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                require_openai=require_openai_effective,
                disable_openai=disable_openai,
                include_posts=include_posts,
                allow_kmeans_fallback=allow_kmeans_fallback,
                require_bertopic=require_bertopic,
                verbose=False,
            )
    except AssertionError as exc:
        message = str(exc)
        if "Torch not compiled with CUDA enabled" in message:
            raise RuntimeError(
                "CUDA was requested but your installed PyTorch build is CPU-only. "
                "Set BERT_EMBEDDING_DEVICE=cpu (or auto), or install a CUDA-enabled torch build."
            ) from exc
        raise
    finally:
        elapsed = time.perf_counter() - started_at
        logger.info("topic-insights stage finished elapsed_seconds=%.2f", elapsed)

    result["mode"] = "business_topic_insights_backend"
    result["message"] = (
        "Topic insights generated using notebooks/business_topic_insights_backend.py "
        "with OPENAI_API_KEY loaded from .env."
    )

    frames = _result_to_frames(result)
    logger.info(
        "topic-insights outputs labels=%s recommendations=%s insight_cards=%s",
        len(frames["business_topic_labels"]),
        len(frames["business_topic_recommendations"]),
        len(frames["business_topic_insight_cards"]),
    )
    return result, frames
