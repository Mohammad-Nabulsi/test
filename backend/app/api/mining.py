from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import PipelineSummary
from app.services.pipeline import run_full_pipeline


router = APIRouter()


@router.post("/run-pipeline/{dataset_id}", response_model=PipelineSummary)
def run_pipeline(dataset_id: str) -> PipelineSummary:
    try:
        return run_full_pipeline(dataset_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

