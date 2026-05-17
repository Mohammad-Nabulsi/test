from __future__ import annotations

import base64
import logging
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict
from uuid import UUID

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse

from app.config import settings
from app.services.cluster_visualization_api import build_cluster_visualization_response


router = APIRouter(prefix="/cluster-visualization")
_BUSINESS_COMPARISON_DEFAULT = "data/processed/kpi_dataset.json"
logger = logging.getLogger(__name__)


def _load_dataset(path: Path) -> pd.DataFrame:
    logger.info("cluster_visualization: loading dataset path=%s", path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type '{suffix}'. Use .json or .csv.",
    )


def _save_plot_images(dataset_id: str, result: Dict[str, Any]) -> Dict[str, str]:
    logger.info("cluster_visualization: saving plot images dataset_id=%s", dataset_id)
    output_dir = settings.storage_path() / "outputs" / dataset_id / "cluster_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)

    content_plot = result.get("content_cluster_plot", {})
    business_plot = result.get("business_cluster_plot", {})
    content_b64 = content_plot.get("image_base64")
    business_b64 = business_plot.get("image_base64")

    if not content_b64 or not business_b64:
        logger.error("cluster_visualization: missing base64 images dataset_id=%s", dataset_id)
        raise HTTPException(status_code=500, detail="Missing image_base64 in visualization result.")

    content_path = output_dir / "content_cluster_plot.png"
    business_path = output_dir / "business_cluster_plot.png"

    try:
        content_path.write_bytes(base64.b64decode(content_b64))
        business_path.write_bytes(base64.b64decode(business_b64))
    except Exception as e:
        logger.exception("cluster_visualization: failed writing plot images dataset_id=%s", dataset_id)
        raise HTTPException(status_code=500, detail=f"Failed to save cluster plot images: {e}")

    content_plot["image_path"] = str(content_path.resolve())
    business_plot["image_path"] = str(business_path.resolve())

    result["saved_files"] = {
        "content_cluster_plot_path": str(content_path.resolve()),
        "business_cluster_plot_path": str(business_path.resolve()),
    }
    return result


def _compute_visualization(
    dataset_id: str,
    original_file_path: str,
    include_base64: bool = False,
) -> Dict[str, Any]:
    logger.info(
        "cluster_visualization: compute start dataset_id=%s original_file_path=%s include_base64=%s",
        dataset_id,
        original_file_path,
        include_base64,
    )
    raw_dir = settings.storage_path() / "raw" / dataset_id
    uploaded_candidates = [
        raw_dir / "raw.csv",
        raw_dir / "raw.json",
    ]
    uploaded_path = next((p for p in uploaded_candidates if p.exists()), None)
    if uploaded_path is None:
        logger.warning("cluster_visualization: uploaded dataset missing dataset_id=%s path=%s", dataset_id, raw_dir)
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    original_path = project_root / original_file_path
    if not original_path.exists():
        logger.warning("cluster_visualization: original dataset missing path=%s", original_path)
        raise HTTPException(status_code=404, detail=f"Original dataset not found: {original_path}")

    content_pipeline_path = (
        project_root
        / "notebooks"
        / "clustering"
        / "artifacts"
        / "content_style_clustering"
        / "models"
        / "content_style_svm_best_pipeline.joblib"
    )
    business_module_path = project_root / "notebooks" / "clustering" / "business_clustring_code.py"
    business_cluster_json_path = (
        project_root
        / "notebooks"
        / "clustering"
        / "artifacts"
        / "business_clustering"
        / "json"
        / "business_cluster_coordinates.json"
    )

    for p, label in [
        (content_pipeline_path, "content model"),
        (business_module_path, "business module"),
        (business_cluster_json_path, "business cluster JSON"),
    ]:
        if not p.exists():
            logger.error("cluster_visualization: dependency missing label=%s path=%s", label, p)
            raise HTTPException(status_code=500, detail=f"Missing {label}: {p}")

    original_df = _load_dataset(original_path)
    business_original_path = project_root / _BUSINESS_COMPARISON_DEFAULT
    if not business_original_path.exists():
        logger.warning("cluster_visualization: business comparison dataset missing path=%s", business_original_path)
        raise HTTPException(
            status_code=404,
            detail=f"Business comparison dataset not found: {business_original_path}",
        )
    business_original_df = _load_dataset(business_original_path)
    uploaded_df = _load_dataset(uploaded_path)
    if uploaded_df.empty:
        logger.warning("cluster_visualization: uploaded dataset empty dataset_id=%s", dataset_id)
        raise HTTPException(status_code=400, detail="Uploaded dataset is empty.")

    result = build_cluster_visualization_response(
        original_kpi_dataset=original_df,
        business_original_kpi_dataset=business_original_df,
        uploaded_dataset=uploaded_df,
        content_pipeline_path=content_pipeline_path,
        business_module_path=business_module_path,
        business_cluster_json_path=business_cluster_json_path,
    )
    result = _save_plot_images(dataset_id=dataset_id, result=result)

    if not include_base64:
        result.get("content_cluster_plot", {}).pop("image_base64", None)
        result.get("business_cluster_plot", {}).pop("image_base64", None)

    logger.info("cluster_visualization: compute success dataset_id=%s", dataset_id)
    return result


@router.get("/{dataset_id}")
def visualize_clusters(
    dataset_id: UUID,
    original_file_path: str = Query(
        default="data/vanilla_kpi_dataset.json",
        description="Path to the original KPI dataset (.json or .csv).",
    ),
    include_base64: bool = Query(
        default=False,
        description="Include raw base64 image in response. Defaults to false.",
    ),
) -> Dict[str, Any]:
    logger.info(
        "cluster_visualization: GET /{dataset_id} called dataset_id=%s original_file_path=%s include_base64=%s",
        dataset_id,
        original_file_path,
        include_base64,
    )
    try:
        result = _compute_visualization(
            dataset_id=str(dataset_id),
            original_file_path=original_file_path,
            include_base64=include_base64,
        )
        return jsonable_encoder(result)
    except HTTPException:
        logger.warning("cluster_visualization: request failed with HTTPException dataset_id=%s", dataset_id)
        raise
    except Exception as e:
        logger.exception("cluster_visualization: unexpected failure dataset_id=%s", dataset_id)
        raise HTTPException(status_code=500, detail=f"Cluster visualization failed: {e}")


@router.get("/{dataset_id}/preview", response_class=HTMLResponse)
def preview_clusters(
    dataset_id: UUID,
    original_file_path: str = Query(
        default="data/vanilla_kpi_dataset.json",
        description="Path to the original KPI dataset (.json or .csv).",
    ),
) -> HTMLResponse:
    logger.info(
        "cluster_visualization: GET /{dataset_id}/preview called dataset_id=%s original_file_path=%s",
        dataset_id,
        original_file_path,
    )
    try:
        result = _compute_visualization(
            dataset_id=str(dataset_id),
            original_file_path=original_file_path,
            include_base64=True,
        )
    except HTTPException:
        logger.warning("cluster_visualization: preview failed with HTTPException dataset_id=%s", dataset_id)
        raise
    except Exception as e:
        logger.exception("cluster_visualization: preview unexpected failure dataset_id=%s", dataset_id)
        raise HTTPException(status_code=500, detail=f"Cluster preview failed: {e}")

    content_image = result.get("content_cluster_plot", {}).get("image_base64")
    business_image = result.get("business_cluster_plot", {}).get("image_base64")
    if not content_image or not business_image:
        logger.error("cluster_visualization: preview missing images dataset_id=%s", dataset_id)
        raise HTTPException(status_code=500, detail="Missing image data in visualization result.")

    html = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Cluster Visualization Preview</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 24px; background: #fafafa; }}
          h1 {{ margin-bottom: 8px; }}
          p {{ color: #444; }}
          .panel {{ margin: 20px 0; padding: 16px; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; }}
          img {{ width: 100%; max-width: 1200px; height: auto; border: 1px solid #ddd; border-radius: 6px; background: #fff; }}
          code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 4px; }}
        </style>
      </head>
      <body>
        <h1>Cluster Visualization Preview</h1>
        <p><code>dataset_id</code>: {dataset_id}</p>
        <div class="panel">
          <h2>Content Cluster Plot</h2>
          <img alt="Content cluster plot" src="data:image/png;base64,{content_image}" />
        </div>
        <div class="panel">
          <h2>Business Cluster Plot</h2>
          <img alt="Business cluster plot" src="data:image/png;base64,{business_image}" />
        </div>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


def _load_uploaded_file(file: UploadFile) -> pd.DataFrame:
    logger.info("cluster_visualization: reading uploaded file filename=%s", file.filename)
    content = file.file.read()
    filename = file.filename or "uploaded"
    suffix = Path(filename).suffix.lower()
    if suffix in (".json",):
        return pd.read_json(BytesIO(content))
    if suffix in (".csv",):
        return pd.read_csv(BytesIO(content))
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type '{suffix}'. Use .json or .csv.",
    )


def _save_uploaded_file(file: UploadFile, dataset_id: str) -> str:
    suffix = Path(file.filename or "uploaded.csv").suffix.lower() or ".csv"
    raw_dir = settings.storage_path() / "raw" / dataset_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"raw{suffix}"
    file.file.seek(0)
    content = file.file.read()
    raw_path.write_bytes(content)
    logger.info("cluster_visualization: saved uploaded file dataset_id=%s path=%s", dataset_id, raw_path)
    return str(raw_path)


@router.post("/upload")
def visualize_clusters_from_file(
    file: UploadFile = File(...),
    original_file_path: str = Query(
        default="data/vanilla_kpi_dataset.json",
        description="Path to the original KPI dataset (.json or .csv).",
    ),
) -> Dict[str, Any]:
    logger.info(
        "cluster_visualization: POST /upload called filename=%s original_file_path=%s",
        file.filename,
        original_file_path,
    )
    if not file.filename:
        logger.warning("cluster_visualization: upload without filename")
        raise HTTPException(status_code=400, detail="No file provided.")
    try:
        dataset_id = str(uuid.uuid4())
        logger.info("cluster_visualization: upload assigned dataset_id=%s", dataset_id)
        uploaded_df = _load_uploaded_file(file)
        if uploaded_df.empty:
            logger.warning("cluster_visualization: uploaded dataframe empty dataset_id=%s", dataset_id)
            raise HTTPException(status_code=400, detail="Uploaded dataset is empty.")

        _save_uploaded_file(file, dataset_id)

        project_root = Path(__file__).resolve().parent.parent.parent.parent
        original_path = project_root / original_file_path
        if not original_path.exists():
            logger.warning("cluster_visualization: upload original dataset missing path=%s", original_path)
            raise HTTPException(status_code=404, detail=f"Original dataset not found: {original_path}")

        content_pipeline_path = (
            project_root
            / "notebooks"
            / "clustering"
            / "artifacts"
            / "content_style_clustering"
            / "models"
            / "content_style_svm_best_pipeline.joblib"
        )
        business_module_path = project_root / "notebooks" / "clustering" / "business_clustring_code.py"
        business_cluster_json_path = (
            project_root
            / "notebooks"
            / "clustering"
            / "artifacts"
            / "business_clustering"
            / "json"
            / "business_cluster_coordinates.json"
        )

        for p, label in [
            (content_pipeline_path, "content model"),
            (business_module_path, "business module"),
            (business_cluster_json_path, "business cluster JSON"),
        ]:
            if not p.exists():
                logger.error("cluster_visualization: upload dependency missing label=%s path=%s", label, p)
                raise HTTPException(status_code=500, detail=f"Missing {label}: {p}")

        original_df = _load_dataset(original_path)
        business_original_path = project_root / _BUSINESS_COMPARISON_DEFAULT
        if not business_original_path.exists():
            logger.warning("cluster_visualization: upload business comparison missing path=%s", business_original_path)
            raise HTTPException(
                status_code=404,
                detail=f"Business comparison dataset not found: {business_original_path}",
            )
        business_original_df = _load_dataset(business_original_path)

        result = build_cluster_visualization_response(
            original_kpi_dataset=original_df,
            business_original_kpi_dataset=business_original_df,
            uploaded_dataset=uploaded_df,
            content_pipeline_path=content_pipeline_path,
            business_module_path=business_module_path,
            business_cluster_json_path=business_cluster_json_path,
        )
        result = _save_plot_images(dataset_id=dataset_id, result=result)
        logger.info("cluster_visualization: upload success dataset_id=%s", dataset_id)

        return jsonable_encoder(result)
    except HTTPException:
        logger.warning("cluster_visualization: upload failed with HTTPException filename=%s", file.filename)
        raise
    except Exception as e:
        logger.exception("cluster_visualization: upload unexpected failure filename=%s", file.filename)
        raise HTTPException(status_code=500, detail=f"Cluster visualization failed: {e}")
