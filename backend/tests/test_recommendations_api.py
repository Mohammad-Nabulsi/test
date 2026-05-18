import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

from app.api.recommendation_apis_final import (
    _ensure_benchmark_text_quality,
    _fallback_benchmark_pipeline,
    _fallback_similar_pipeline,
)
from app.api.recommendations import get_recommendations, get_recommendation_rules
from app.config import settings


class RecommendationsApiTest(unittest.TestCase):
    def setUp(self):
        self.original_storage_dir = settings.storage_dir
        self.temp_dir = Path(tempfile.mkdtemp())
        settings.storage_dir = str(self.temp_dir / "storage")

    def tearDown(self):
        settings.storage_dir = self.original_storage_dir
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_recommendations_endpoint_returns_data(self):
        dataset_id = "test-dataset"
        outputs_dir = Path(settings.storage_dir) / "outputs" / dataset_id
        outputs_dir.mkdir(parents=True, exist_ok=True)

        recs_df = pd.DataFrame(
            [
                {
                    "business_name": "Vanilla Cafe",
                    "sector": "Cafe",
                    "recommendation": "Use more reels",
                    "reason": "Reels perform better in your dataset.",
                    "evidence_source": "KPIs",
                    "evidence_metric": "pct_reels=0.10",
                    "priority": "Medium",
                    "expected_impact": "Medium",
                }
            ]
        )
        recs_df.to_csv(outputs_dir / "recommendations.csv", index=False, encoding="utf-8")

        response = get_recommendations(dataset_id)

        self.assertEqual(response.dataset_id, dataset_id)
        self.assertEqual(response.data["recommendations"][0]["business_name"], "Vanilla Cafe")

    def test_recommendation_rules_endpoint_returns_404_when_missing(self):
        dataset_id = "missing-ds"

        with self.assertRaises(HTTPException) as context:
            get_recommendation_rules(dataset_id)

        self.assertEqual(context.exception.status_code, 404)
        self.assertEqual(context.exception.detail, "Association rules not found. Run pipeline first.")

    def test_benchmark_text_quality_fallback_uses_palestinian_arabic(self):
        payload = {
            "business_summary": {
                "business_name": "Vanilla Cafe",
                "sector": "Cafe",
                "sector_rank": 2,
                "sector_percentile": 80,
                "total_sector_businesses": 10,
                "summary_text": "",
            },
            "kpi_comparisons": [
                {
                    "metric_key": "engagement",
                    "metric_name": "Engagement",
                    "business_value": 12,
                    "sector_average": 10,
                    "top_sector_value": 13,
                    "formatted_text": "",
                    "gpt_insight": "",
                }
            ],
            "sector_insights": [],
        }

        result = _ensure_benchmark_text_quality(payload)

        self.assertIn("بمؤشر", result["kpi_comparisons"][0]["formatted_text"])
        self.assertIn("مؤشر التفاعل", result["kpi_comparisons"][0]["gpt_insight"])
        self.assertIn("ترتيبه", result["business_summary"]["summary_text"])
        self.assertTrue(any("القطاع" in item or "CTA" in item for item in result["sector_insights"]))

    def test_similar_business_fallback_returns_same_sector_peers_for_vanilla_dataset(self):
        result = _fallback_similar_pipeline("data/processed/vanilla_kpi_dataset.json")

        self.assertGreater(len(result["similar_businesses"]), 0)
        self.assertTrue(
            all(item["sector"] == "Cafes/Restaurants" for item in result["similar_businesses"])
        )

    def test_benchmark_fallback_builds_normalized_sector_comparison_for_vanilla_dataset(self):
        result = _fallback_benchmark_pipeline("data/processed/vanilla_kpi_dataset.json")

        self.assertGreater(result["business_summary"]["total_sector_businesses"], 1)
        self.assertLessEqual(result["business_summary"]["business_score"], 1.0)
        quality_item = next(item for item in result["kpi_comparisons"] if item["metric_key"] == "quality")
        self.assertLessEqual(float(quality_item["business_value"]), 1.0)
