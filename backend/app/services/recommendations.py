from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def generate_recommendations(
    kpis_df: pd.DataFrame,
    business_clusters_df: pd.DataFrame,
    rules_df: pd.DataFrame,
    trends_df: pd.DataFrame,
    anomalies_df: pd.DataFrame,
    network_nodes_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Explainable recommendation generator.

    Design goal:
    - simple and deterministic
    - uses evidence from computed outputs
    - produces actionable text suitable for a student dashboard demo
    """

    df = kpis_df.copy()
    if df.empty:
        empty = pd.DataFrame(
            columns=[
                "business_name",
                "sector",
                "recommendation",
                "reason",
                "evidence_source",
                "evidence_metric",
                "priority",
                "expected_impact",
            ]
        )
        return empty, {"ok": False, "message": "Empty dataset."}

    # Business-level aggregates for decisioning
    biz = (
        df.groupby(["business_name", "sector"], dropna=False)
        .agg(
            posts=("business_name", "size"),
            followers=("followers_count", "max"),
            avg_engagement_rate=("engagement_rate_followers", "mean"),
            avg_views_per_follower=("views_per_follower", "mean"),
            pct_reels=("post_type", lambda s: float((s == "reel").mean())),
            pct_CTA=("CTA_present", "mean"),
            pct_promo=("promo_post", "mean"),
            pct_dialect=("arabic_dialect_style", "mean"),
            pct_Arabic=("language", lambda s: float((s == "Arabic").mean())),
        )
        .reset_index()
    )

    # Merge cluster names if present
    if not business_clusters_df.empty and "business_cluster_name" in business_clusters_df.columns:
        biz = biz.merge(
            business_clusters_df[["business_name", "sector", "business_cluster_name"]],
            on=["business_name", "sector"],
            how="left",
        )
    else:
        biz["business_cluster_name"] = "Unknown"

    # Global comparisons (EDA-like)
    def _mean_by(col: str, target: str) -> pd.Series:
        if col not in df.columns or target not in df.columns:
            return pd.Series(dtype=float)
        return df.groupby(col)[target].mean()

    by_type = _mean_by("post_type", "engagement_rate_followers")
    reels_better = ("reel" in by_type.index) and ("image" in by_type.index) and (float(by_type["reel"]) > float(by_type["image"]) * 1.15)

    by_cta = _mean_by("CTA_present", "engagement_rate_followers")
    cta_better = (True in by_cta.index) and (False in by_cta.index) and (float(by_cta[True]) > float(by_cta[False]) * 1.10)

    by_dialect = _mean_by("arabic_dialect_style", "engagement_rate_followers")
    dialect_better = (True in by_dialect.index) and (False in by_dialect.index) and (float(by_dialect[True]) > float(by_dialect[False]) * 1.08)

    by_promo = _mean_by("promo_post", "engagement_rate_followers")
    promo_worse = (True in by_promo.index) and (False in by_promo.index) and (float(by_promo[True]) < float(by_promo[False]) * 0.92)

    # Top rule that leads to a "result=" consequent
    top_rule_text = None
    if not rules_df.empty and {"antecedents", "consequents", "lift", "confidence"}.issubset(rules_df.columns):
        rr = rules_df[rules_df["consequents"].astype(str).str.contains("result=")]
        if not rr.empty:
            rr = rr.sort_values(["lift", "confidence"], ascending=False).head(1).iloc[0]
            top_rule_text = f"{rr['antecedents']} -> {rr['consequents']} (lift={float(rr['lift']):.2f}, conf={float(rr['confidence']):.2f})"

    # Recommendation rows
    recs = []

    def add(bname: str, sector: str, rec: str, reason: str, src: str, metric: str, priority: str, impact: str):
        recs.append(
            {
                "business_name": bname,
                "sector": sector,
                "recommendation": rec,
                "reason": reason,
                "evidence_source": src,
                "evidence_metric": metric,
                "priority": priority,
                "expected_impact": impact,
            }
        )

    for _, r in biz.iterrows():
        bname = str(r["business_name"])
        sector = str(r["sector"])
        followers = float(r.get("followers", 0))
        er = float(r.get("avg_engagement_rate", 0))
        promo = float(r.get("pct_promo", 0))
        cta = float(r.get("pct_CTA", 0))
        reels = float(r.get("pct_reels", 0))
        dialect = float(r.get("pct_dialect", 0))

        if followers >= 15000 and er < 0.008:
            add(
                bname,
                sector,
                "Redesign content mix and test new formats",
                "High followers but low engagement suggests content is not resonating.",
                "KPIs",
                f"followers={followers:.0f}, avg_engagement_rate={er:.4f}",
                "High",
                "High",
            )

        if reels_better and reels < 0.25:
            add(
                bname,
                sector,
                "Increase Reels share",
                "Reels outperform images in this dataset; you are under-using them.",
                "EDA",
                f"your_pct_reels={reels:.2f}",
                "Medium",
                "Medium",
            )

        if cta_better and cta < 0.35:
            add(
                bname,
                sector,
                "Add clearer CTAs to more posts",
                "CTA posts show higher engagement rate overall; your CTA usage is low.",
                "EDA",
                f"your_pct_CTA={cta:.2f}",
                "Medium",
                "Medium",
            )

        if dialect_better and dialect < 0.35:
            add(
                bname,
                sector,
                "Use local Arabic dialect more often",
                "Dialect-style posts correlate with higher engagement in this dataset.",
                "EDA",
                f"your_pct_dialect={dialect:.2f}",
                "Low",
                "Medium",
            )

        if promo_worse and promo > 0.55:
            add(
                bname,
                sector,
                "Reduce sales-only content and add value posts",
                "Promo posts underperform overall; your promo share is high.",
                "EDA",
                f"your_pct_promo={promo:.2f}",
                "High",
                "Medium",
            )

        # Viral anomalies evidence
        if not anomalies_df.empty and {"business_name", "flags"}.issubset(anomalies_df.columns):
            a = anomalies_df[anomalies_df["business_name"].astype(str) == bname]
            if not a.empty and a["flags"].astype(str).str.contains("viral_spike").any():
                add(
                    bname,
                    sector,
                    "Replicate the format of your viral post(s)",
                    "You have viral spikes; analyze hook, timing, and format, then replicate intentionally.",
                    "Anomalies",
                    "viral_spike flags",
                    "Medium",
                    "High",
                )

        if top_rule_text:
            add(
                bname,
                sector,
                "Test a top-performing pattern from association rules",
                "Association rules suggest a pattern frequently linked to high engagement/virality. Run an A/B test.",
                "Association Rules",
                top_rule_text,
                "Low",
                "Medium",
            )

        # Cluster framing (always include one low-priority row)
        cname = str(r.get("business_cluster_name", "Unknown")) if pd.notna(r.get("business_cluster_name", None)) else "Unknown"
        add(
            bname,
            sector,
            f"Focus based on your cluster: {cname}",
            "Cluster names summarize your current content behavior; use this to choose experiments.",
            "Clustering",
            f"cluster={cname}",
            "Low",
            "Low",
        )

    out = pd.DataFrame(recs)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "business_name",
                "sector",
                "recommendation",
                "reason",
                "evidence_source",
                "evidence_metric",
                "priority",
                "expected_impact",
            ]
        )
    return out, {"ok": True, "recommendations": int(len(out))}

