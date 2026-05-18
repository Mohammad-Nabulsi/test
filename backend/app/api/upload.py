from __future__ import annotations

import uuid
from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.schemas import UploadResponse
from app.services.validation import validate_dataframe
from app.utils.file_utils import ensure_dir, safe_read_csv, write_json


router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)) -> UploadResponse:
    filename = (file.filename or "").lower()
    if not (filename.endswith(".csv") or filename.endswith(".json")):
        raise HTTPException(status_code=400, detail="Only CSV or JSON uploads are supported.")

    dataset_id = str(uuid.uuid4())
    storage = settings.storage_path()

    raw_dir = ensure_dir(storage / "raw" / dataset_id)

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    try:
        if filename.endswith(".json"):
            raw_path = raw_dir / "raw.json"
            raw_path.write_bytes(content)
            df = pd.read_json(BytesIO(content))
        else:
            raw_path = raw_dir / "raw.csv"
            raw_path.write_bytes(content)
            df = safe_read_csv(raw_path)
    except Exception as e:
        kind = "JSON" if filename.endswith(".json") else "CSV"
        raise HTTPException(status_code=400, detail=f"Failed to parse {kind}: {e}")

    report = validate_dataframe(df)

    reports_dir = ensure_dir(storage / "reports" / dataset_id)
    write_json(reports_dir / "validation_report.json", report.model_dump())

    return UploadResponse(dataset_id=dataset_id, validation_report=report)
