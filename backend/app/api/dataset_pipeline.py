from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from app.utils.file_utils import read_json, safe_read_csv
from app.services.dataset_staged_pipeline import (
    load_stage_json,
    run_hashtag_stage_for_dataset,
    run_hashtag_stage_from_upload,
    run_preprocess_kpi_pipeline,
    run_staged_pipeline,
    run_topic_stage_for_dataset,
    run_topic_stage_from_upload,
)


router = APIRouter()
logger = logging.getLogger(__name__)


TOPIC_VISUALIZATION_FILES = {
    "intertopic": "business_topic_intertopic_map.html",
    "barchart": "business_topic_barchart.html",
    "heatmap": "business_topic_heatmap.html",
}


@router.get("/datasets/static-topic-content")
def get_static_topic_content() -> dict:
    out = settings.storage_path() / "outputs" / "static_topic_content"
    required = {
        "dataset_with_topics": out / "dataset_with_topics.csv",
        "topic_summary": out / "topic_summary.csv",
        "topic_terms": out / "topic_terms_exact_1_2_3grams.csv",
        "recommendations": out / "dynamic_recommendations.csv",
        "topics_json": out / "topics.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise HTTPException(status_code=404, detail=f"Missing static files: {', '.join(missing)}")

    df_dataset = safe_read_csv(required["dataset_with_topics"])
    df_summary = safe_read_csv(required["topic_summary"])
    df_terms = safe_read_csv(required["topic_terms"])
    df_recs = safe_read_csv(required["recommendations"])
    topics_payload = read_json(required["topics_json"])
    topic_representations = topics_payload.get("topic_representations", {}) if isinstance(topics_payload, dict) else {}
    topics_preview = []
    for topic_id, terms in list(topic_representations.items())[:8]:
        preview_terms = [str(item[0]) for item in terms[:8] if isinstance(item, list) and item]
        topics_preview.append({"topic_id": str(topic_id), "terms": preview_terms})

    return {
        "cards": {
            "topics_total": int(df_summary["topic_id"].nunique()) if "topic_id" in df_summary.columns else int(len(df_summary)),
            "posts_with_topics": int(len(df_dataset)),
            "recommendations_total": int(len(df_recs)),
            "terms_total": int(len(df_terms)),
        },
        "topic_summary": df_summary.head(8).to_dict(orient="records"),
        "recommendations": df_recs.head(8).to_dict(orient="records"),
        "topics_preview": topics_preview,
        "intertopic_map_url": "/api/datasets/static-topic-intertopic-map",
    }


@router.get("/datasets/static-topic-intertopic-map")
def get_static_topic_intertopic_map() -> FileResponse:
    path = settings.storage_path() / "outputs" / "static_topic_content" / "intertopic_distance_map.html"
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Static intertopic map not found.")
    return FileResponse(
        path,
        media_type="text/html; charset=utf-8",
        filename=path.name,
        headers={"Content-Disposition": f'inline; filename="{path.name}"'},
    )


@router.post("/datasets/upload-and-run")
async def upload_and_run_dataset(file: UploadFile = File(...)) -> dict:
    """
    Upload a raw dataset and run the production flow:
    preprocess -> KPI -> hashtag + business-topic-insights.
    """
    logger.info("upload-and-run started filename=%s", file.filename)
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        result = run_staged_pipeline(file.filename, content)
        logger.info(
            "upload-and-run completed dataset_id=%s rows_received=%s",
            result.get("dataset_id"),
            result.get("rows_received"),
        )
        return result
    except ValueError as exc:
        logger.warning("upload-and-run validation failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("upload-and-run unexpected failure")
        raise HTTPException(status_code=500, detail=f"Upload pipeline failed: {exc}")


@router.post("/datasets/upload-preprocess-kpi")
async def upload_preprocess_kpi_dataset(file: UploadFile = File(...)) -> dict:
    """
    Upload a raw dataset and run only preprocess + KPI stages.
    """
    logger.info("upload-preprocess-kpi started filename=%s", file.filename)
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        result = run_preprocess_kpi_pipeline(file.filename, content)
        logger.info(
            "upload-preprocess-kpi completed dataset_id=%s rows_received=%s",
            result.get("dataset_id"),
            result.get("rows_received"),
        )
        return result
    except ValueError as exc:
        logger.warning("upload-preprocess-kpi validation failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("upload-preprocess-kpi unexpected failure")
        raise HTTPException(status_code=500, detail=f"Upload preprocess+kpi failed: {exc}")


@router.post("/datasets/stages/hashtag")
async def hashtag_stage_from_upload(file: UploadFile = File(...)) -> dict:
    """Run the production hashtag association recommendation stage from an upload."""
    logger.info("hashtag-stage-from-upload started filename=%s", file.filename)
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        result = run_hashtag_stage_from_upload(file.filename, content)
        logger.info(
            "hashtag-stage-from-upload completed dataset_id=%s",
            result.get("dataset_id"),
        )
        return result
    except ValueError as exc:
        logger.warning("hashtag-stage-from-upload validation failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("hashtag-stage-from-upload unexpected failure")
        raise HTTPException(status_code=500, detail=f"Hashtag stage failed: {exc}")


@router.post("/datasets/{dataset_id}/stages/hashtag")
def hashtag_stage_for_dataset(dataset_id: str) -> dict:
    """Run the production hashtag association recommendation stage for an existing dataset."""
    logger.info("hashtag-stage-for-dataset started dataset_id=%s", dataset_id)
    try:
        result = run_hashtag_stage_for_dataset(dataset_id)
        logger.info("hashtag-stage-for-dataset completed dataset_id=%s", dataset_id)
        return result
    except FileNotFoundError as exc:
        logger.warning("hashtag-stage-for-dataset file not found dataset_id=%s", dataset_id)
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        logger.warning("hashtag-stage-for-dataset validation failed dataset_id=%s: %s", dataset_id, exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("hashtag-stage-for-dataset unexpected failure dataset_id=%s", dataset_id)
        raise HTTPException(status_code=500, detail=f"Hashtag stage failed: {exc}")


@router.post("/datasets/stages/business-topic-insights")
async def business_topic_insights_from_upload(file: UploadFile = File(...)) -> dict:
    """
    Run the production BERTopic business-topic insight stage from an upload.
    """
    logger.info("business-topic-insights-from-upload started filename=%s", file.filename)
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        stage_result = run_topic_stage_from_upload(file.filename, content)
        response = {
            "dataset_id": stage_result["dataset_id"],
            "output_files": stage_result["output_files"],
            "result": stage_result["topic_stage"],
        }
        logger.info(
            "business-topic-insights-from-upload completed dataset_id=%s",
            response.get("dataset_id"),
        )
        return response
    except ValueError as exc:
        logger.warning("business-topic-insights-from-upload validation failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("business-topic-insights-from-upload unexpected failure")
        raise HTTPException(status_code=500, detail=f"Business topic insights failed: {exc}")


@router.post("/datasets/{dataset_id}/stages/business-topic-insights")
def business_topic_insights_for_dataset(dataset_id: str) -> dict:
    """
    Run the production BERTopic business-topic insight stage for an existing dataset.
    """
    logger.info("business-topic-insights-for-dataset started dataset_id=%s", dataset_id)
    try:
        stage_result = run_topic_stage_for_dataset(dataset_id)
        response = {
            "dataset_id": stage_result["dataset_id"],
            "output_files": stage_result["output_files"],
            "result": stage_result["topic_stage"],
        }
        logger.info("business-topic-insights-for-dataset completed dataset_id=%s", dataset_id)
        return response
    except ValueError as exc:
        logger.warning("business-topic-insights-for-dataset validation failed dataset_id=%s: %s", dataset_id, exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("business-topic-insights-for-dataset unexpected failure dataset_id=%s", dataset_id)
        raise HTTPException(status_code=500, detail=f"Business topic insights failed: {exc}")


@router.get("/datasets/{dataset_id}/hashtag-recommendations")
def get_hashtag_stage(dataset_id: str) -> dict:
    """Read the saved production hashtag recommendation result."""
    logger.info("get-hashtag-stage started dataset_id=%s", dataset_id)
    try:
        result = load_stage_json(dataset_id, "hashtag")
        logger.info("get-hashtag-stage completed dataset_id=%s", dataset_id)
        return result
    except FileNotFoundError as exc:
        logger.warning("get-hashtag-stage not found dataset_id=%s", dataset_id)
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/datasets/{dataset_id}/dynamic-ngram-insights")
def get_topic_stage(dataset_id: str) -> dict:
    """Read the saved production BERTopic business-topic insight result."""
    logger.info("get-topic-stage started dataset_id=%s", dataset_id)
    try:
        result = load_stage_json(dataset_id, "topic")
        logger.info("get-topic-stage completed dataset_id=%s", dataset_id)
        return result
    except FileNotFoundError as exc:
        logger.warning("get-topic-stage not found dataset_id=%s", dataset_id)
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/datasets/{dataset_id}/business-topic-visualization")
def get_business_topic_visualization(
    dataset_id: str,
    view: Literal["intertopic", "barchart", "heatmap"] = Query("intertopic"),
) -> FileResponse:
    """
    Open a saved interactive BERTopic visualization HTML file.

    Available views:
    - intertopic: BERTopic intertopic distance map
    - barchart: top topic words bar chart
    - heatmap: topic similarity heatmap
    """
    logger.info("get-business-topic-visualization started dataset_id=%s view=%s", dataset_id, view)
    outputs_dir = settings.storage_path() / "outputs" / dataset_id / "business_topic_outputs"
    path = (outputs_dir / TOPIC_VISUALIZATION_FILES[view]).resolve()
    try:
        path.relative_to(outputs_dir.resolve())
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid visualization path.") from exc

    if not path.exists() or not path.is_file():
        logger.warning("visualization file not found dataset_id=%s view=%s path=%s", dataset_id, view, path)
        raise HTTPException(
            status_code=404,
            detail=(
                "Visualization file was not found. Run the business-topic-insights POST endpoint first. "
                "If this dataset was generated before visualization support was added, rerun the topic stage."
            ),
        )

    logger.info("get-business-topic-visualization completed dataset_id=%s view=%s", dataset_id, view)
    return FileResponse(
        Path(path),
        media_type="text/html; charset=utf-8",
        filename=path.name,
        headers={"Content-Disposition": f'inline; filename="{path.name}"'},
    )
