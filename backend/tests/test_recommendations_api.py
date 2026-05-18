import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

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
