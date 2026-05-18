from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings
from app.utils.file_utils import resolve_output_file


router = APIRouter(prefix="/exports")


@router.get("/{dataset_id}/{file_name}")
def export_file(dataset_id: str, file_name: str):
    outputs_dir = settings.storage_path() / "outputs" / dataset_id
    p = resolve_output_file(outputs_dir, file_name)
    if p is None:
        raise HTTPException(status_code=404, detail="File not found.")
    media_type = "text/csv"
    if p.suffix.lower() == ".json":
        media_type = "application/json"
    return FileResponse(path=str(p), media_type=media_type, filename=p.name)

