from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api import (
    business_clustering,
    cluster_visualization,
    clustering,
    clustering_ai,
    dashboard,
    dataset_pipeline,
    exports,
    me,
    mining,
    recommendations,
    upload,
)
from app.api.recommendation_apis_final import router as recommendation_apis_single_router
from app.api.forecast_final import router as forecast_single_router
from app.api.anomalies_final import router as anomalies_single_router
from app.api.business_momentum_final import router as business_momentum_single_router
from app.utils.file_utils import ensure_dir


def create_app() -> FastAPI:
    app = FastAPI(
        title="Palestine SME Social Media Intelligence Platform",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Ensure storage folders exist
    storage = settings.storage_path()
    ensure_dir(storage / "raw")
    ensure_dir(storage / "cleaned")
    ensure_dir(storage / "outputs")
    ensure_dir(storage / "reports")

    @app.get("/health")
    def health():
        return {"status": "ok", "env": settings.app_env}

    app.include_router(upload.router, prefix="/api", tags=["upload"])
    app.include_router(mining.router, prefix="/api", tags=["pipeline"])
    app.include_router(dashboard.router, prefix="/api", tags=["dashboard"])
    app.include_router(dataset_pipeline.router, prefix="/api", tags=["dataset-pipeline"])
    app.include_router(recommendations.router, prefix="/api", tags=["recommendations"])
    app.include_router(exports.router, prefix="/api", tags=["exports"])
    app.include_router(clustering.router, prefix="/api", tags=["clustering"])
    app.include_router(clustering_ai.router, prefix="/api", tags=["clustering-ai"])
    app.include_router(business_clustering.router, prefix="/api", tags=["business-clustering"])
    app.include_router(cluster_visualization.router, prefix="/api", tags=["cluster-visualization"])
    app.include_router(me.router, prefix="/api", tags=["me"])
    app.include_router(recommendation_apis_single_router)
    app.include_router(forecast_single_router)
    app.include_router(anomalies_single_router)
    app.include_router(business_momentum_single_router)

    return app


app = create_app()
