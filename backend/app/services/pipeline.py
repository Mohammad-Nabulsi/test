from __future__ import annotations

import traceback
from pathlib import Path
from typing import List

import pandas as pd

from app.config import settings
from app.schemas import PipelineStepStatus, PipelineSummary
from app.services.anomaly_detection import detect_anomalies
from app.services.association_rules import build_transactions, business_value_rules, mine_association_rules
from app.services.cleaning import clean_dataset
from app.services.clustering import business_clustering, post_clustering
from app.services.dimensionality import pca_businesses, pca_posts
from app.services.eda import build_eda_summary
from app.services.kpi_engineering import engineer_kpis
from app.services.network_analysis import build_cooccurrence_network
from app.services.recommendations import generate_recommendations
from app.services.time_series import business_momentum, simple_forecast, weekly_trends
from app.services.validation import validate_dataframe
from app.utils.file_utils import ensure_dir, safe_read_csv, write_csv, write_json


def _raw_path(dataset_id: str) -> Path:
    return settings.storage_path() / "raw" / dataset_id / "raw.csv"


def _outputs_dir(dataset_id: str) -> Path:
    return ensure_dir(settings.storage_path() / "outputs" / dataset_id)


def _cleaned_dir(dataset_id: str) -> Path:
    return ensure_dir(settings.storage_path() / "cleaned" / dataset_id)


def _step(name: str, fn) -> PipelineStepStatus:
    try:
        res = fn()
        # Allow fn to return (message, files) or a dict
        if isinstance(res, tuple):
            msg, files = res
            return PipelineStepStatus(step=name, ok=True, message=str(msg), output_files=list(files))
        return PipelineStepStatus(step=name, ok=True, message="ok", output_files=[])
    except Exception as e:
        return PipelineStepStatus(step=name, ok=False, message=f"{e}", output_files=[])


def run_full_pipeline(dataset_id: str) -> PipelineSummary:
    raw_path = _raw_path(dataset_id)
    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset not found: {raw_path}")

    out_dir = _outputs_dir(dataset_id)
    cleaned_dir = _cleaned_dir(dataset_id)

    steps: List[PipelineStepStatus] = []

    # Load raw
    df_raw = safe_read_csv(raw_path)

    # Step 1: Validation
    report = validate_dataframe(df_raw)
    write_json(out_dir / "validation_report.json", report.model_dump())
    steps.append(
        PipelineStepStatus(
            step="validation",
            ok=bool(report.ok),
            message="ok" if report.ok else "Validation failed (see validation_report.json). Pipeline will still attempt cleaning.",
            output_files=["validation_report.json"],
        )
    )

    # Step 2: Cleaning
    df_clean, clean_notes = clean_dataset(df_raw)
    write_csv(cleaned_dir / "clean_dataset.csv", df_clean)
    write_csv(out_dir / "clean_dataset.csv", df_clean)
    write_json(out_dir / "cleaning_notes.json", clean_notes)
    steps.append(PipelineStepStatus(step="cleaning", ok=True, message=f"cleaned_rows={len(df_clean)}", output_files=["clean_dataset.csv", "cleaning_notes.json"]))

    # Step 3: KPI Engineering
    df_kpis = engineer_kpis(df_clean)
    write_csv(out_dir / "kpis.csv", df_kpis)
    steps.append(PipelineStepStatus(step="kpi_engineering", ok=True, message="kpis generated", output_files=["kpis.csv"]))

    # Step 4: EDA
    eda = build_eda_summary(df_kpis)
    write_json(out_dir / "eda_summary.json", eda)
    steps.append(PipelineStepStatus(step="eda", ok=True, message="eda summary generated", output_files=["eda_summary.json"]))

    # Step 5: Post clustering
    post_cl = post_clustering(df_kpis)
    post_cols = post_cl.df[["business_name", "sector", "post_date", "post_type", "language", "post_cluster", "post_cluster_label", "engagement_rate_followers", "views_per_follower", "total_engagement"]].copy()
    post_cols["post_date"] = post_cols["post_date"].astype(str)
    write_csv(out_dir / "post_clusters.csv", post_cols)
    write_json(out_dir / "post_clustering_info.json", post_cl.model_info)
    steps.append(PipelineStepStatus(step="post_clustering", ok=bool(post_cl.model_info.get("ok", True)), message=str(post_cl.model_info.get("message", "ok")), output_files=["post_clusters.csv", "post_clustering_info.json"]))

    # Step 6: Business clustering
    biz_cl = business_clustering(df_kpis)
    write_csv(out_dir / "business_clusters.csv", biz_cl.df)
    write_json(out_dir / "business_clustering_info.json", biz_cl.model_info)
    steps.append(PipelineStepStatus(step="business_clustering", ok=bool(biz_cl.model_info.get("ok", True)), message=str(biz_cl.model_info.get("message", "ok")), output_files=["business_clusters.csv", "business_clustering_info.json"]))

    # Step 7: PCA
    post_pca_df, post_pca_info = pca_posts(df_kpis)
    if not post_pca_df.empty:
        post_pca_df = post_pca_df.assign(
            business_name=df_kpis["business_name"].astype(str),
            sector=df_kpis["sector"].astype(str),
            post_type=df_kpis["post_type"].astype(str),
            language=df_kpis["language"].astype(str),
            engagement_rate_followers=df_kpis["engagement_rate_followers"].astype(float),
        )
    write_csv(out_dir / "post_pca.csv", post_pca_df)
    write_json(out_dir / "post_pca_info.json", post_pca_info)

    biz_pca_df, biz_pca_info = pca_businesses(biz_cl.df)
    if not biz_pca_df.empty:
        biz_pca_df = biz_pca_df.assign(
            business_name=biz_cl.df["business_name"].astype(str),
            sector=biz_cl.df["sector"].astype(str),
            business_cluster=biz_cl.df.get("business_cluster", 0),
            business_cluster_name=biz_cl.df.get("business_cluster_name", "Unknown"),
            avg_engagement_rate=biz_cl.df.get("avg_engagement_rate", 0.0),
        )
    write_csv(out_dir / "business_pca.csv", biz_pca_df)
    write_json(out_dir / "business_pca_info.json", biz_pca_info)
    steps.append(PipelineStepStatus(step="pca", ok=True, message="pca computed", output_files=["post_pca.csv", "business_pca.csv", "post_pca_info.json", "business_pca_info.json"]))

    # Step 8: Association rules
    tx = build_transactions(df_kpis)
    write_csv(out_dir / "transactions.csv", tx)
    rules_df, rules_info = mine_association_rules(tx)
    write_csv(out_dir / "association_rules.csv", rules_df)
    bv = business_value_rules(rules_df)
    write_csv(out_dir / "business_value_rules.csv", bv)
    write_json(out_dir / "association_rules_info.json", rules_info)
    steps.append(PipelineStepStatus(step="association_rules", ok=bool(rules_info.get("ok", False)), message=str(rules_info.get("message", "ok")), output_files=["transactions.csv", "association_rules.csv", "business_value_rules.csv", "association_rules_info.json"]))

    # Step 9: Trends
    weekly_df, weekly_info = weekly_trends(df_kpis)
    write_csv(out_dir / "weekly_trends.csv", weekly_df)
    momentum_df, momentum_info = business_momentum(df_kpis)
    write_csv(out_dir / "business_momentum.csv", momentum_df)
    forecast_df, forecast_info = simple_forecast(weekly_df)
    write_csv(out_dir / "forecast.csv", forecast_df)
    write_json(out_dir / "trends_info.json", {"weekly": weekly_info, "momentum": momentum_info, "forecast": forecast_info})
    steps.append(PipelineStepStatus(step="trends_and_forecast", ok=bool(weekly_info.get("ok", False)), message=str(weekly_info.get("message", "ok")), output_files=["weekly_trends.csv", "business_momentum.csv", "forecast.csv", "trends_info.json"]))

    # Step 10: Anomalies
    anomalies_df, anomalies_info = detect_anomalies(df_kpis)
    write_csv(out_dir / "anomalies.csv", anomalies_df)
    write_json(out_dir / "anomalies_info.json", anomalies_info)
    steps.append(PipelineStepStatus(step="anomaly_detection", ok=bool(anomalies_info.get("ok", False)), message=str(anomalies_info.get("message", "ok")), output_files=["anomalies.csv", "anomalies_info.json"]))

    # Step 11: Network analysis
    nodes_df, edges_df, net_summary = build_cooccurrence_network(tx)
    write_csv(out_dir / "network_nodes.csv", nodes_df)
    write_csv(out_dir / "network_edges.csv", edges_df)
    write_json(out_dir / "network_summary.json", net_summary)
    steps.append(PipelineStepStatus(step="network_analysis", ok=bool(net_summary.get("ok", False)), message=str(net_summary.get("message", "ok")), output_files=["network_nodes.csv", "network_edges.csv", "network_summary.json"]))

    # Step 12: Recommendations
    recs_df, recs_info = generate_recommendations(
        kpis_df=df_kpis,
        business_clusters_df=biz_cl.df,
        rules_df=bv if not bv.empty else rules_df,
        trends_df=weekly_df,
        anomalies_df=anomalies_df,
        network_nodes_df=nodes_df,
    )
    write_csv(out_dir / "recommendations.csv", recs_df)
    write_json(out_dir / "recommendations_info.json", recs_info)
    steps.append(PipelineStepStatus(step="recommendations", ok=bool(recs_info.get("ok", False)), message=str(recs_info.get("message", "ok")), output_files=["recommendations.csv", "recommendations_info.json"]))

    ok = all(s.ok for s in steps if s.step in {"cleaning", "kpi_engineering", "eda"})
    summary = PipelineSummary(
        dataset_id=dataset_id,
        ok=ok,
        message="Pipeline complete" if ok else "Pipeline completed with some issues (see step statuses).",
        steps=steps,
        outputs_dir=str(out_dir),
    )
    write_json(out_dir / "pipeline_summary.json", summary.model_dump())
    return summary

