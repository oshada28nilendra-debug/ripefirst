import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from usefirst.data import deterministic_split, read_manifest
from usefirst.features import extract_features, quality_flags
from usefirst.ranking import append_observation, priority_score, rank_manifest


class UseFirstTests(unittest.TestCase):
    def test_features_are_finite_and_stable_length(self):
        image = np.full((100, 120, 3), [220, 180, 35], dtype=np.uint8)
        features = extract_features(image)
        self.assertEqual(features.shape, (130,))
        self.assertTrue(np.isfinite(features).all())

    def test_low_resolution_is_flagged(self):
        image = np.full((40, 40, 3), 100, dtype=np.uint8)
        self.assertIn("low_resolution", quality_flags(image))

    def test_split_is_deterministic(self):
        self.assertEqual(deterministic_split("42", "fresh_banana"), deterministic_split("42", "fresh_banana"))

    def test_manifest_rejects_source_image_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["path", "label", "stage", "split", "source_image"])
                writer.writeheader(); writer.writerows([
                    {"path": "a.jpg", "label": "fresh_banana", "stage": 0, "split": "train", "source_image": "same"},
                    {"path": "b.jpg", "label": "fresh_banana", "stage": 0, "split": "test", "source_image": "same"},
                ])
            with self.assertRaisesRegex(ValueError, "leakage"):
                read_manifest(path)

    def test_storage_is_only_a_bounded_tiebreaker(self):
        self.assertEqual(priority_score(80), 80)
        self.assertAlmostEqual(priority_score(80, 100), 83.0)
        self.assertGreater(priority_score(40, 7), priority_score(40, 0))

    def test_negative_storage_days_rejected(self):
        with self.assertRaises(ValueError):
            priority_score(50, -1)

    @patch("usefirst.ranking.predict")
    def test_uncertain_items_are_sent_to_review(self, fake_predict):
        fake_predict.side_effect = [
            {"visible_deterioration_score": 90, "uncertain": True},
            {"visible_deterioration_score": 60, "uncertain": False},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "items.csv"
            manifest.write_text("item_id,image\na,a.jpg\nb,b.jpg\n")
            result = rank_manifest("model", manifest)
        self.assertEqual(result[0]["item_id"], "b")
        self.assertEqual(result[0]["action"], "use_first")
        self.assertEqual(result[1]["action"], "review_image")

    @patch("usefirst.ranking.predict")
    def test_observation_reports_score_change_per_day(self, fake_predict):
        fake_predict.side_effect = [
            {"stage": "fresh", "visible_deterioration_score": 10, "confidence": .9, "uncertain": False},
            {"stage": "normal", "visible_deterioration_score": 30, "confidence": .8, "uncertain": False},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            store = Path(temporary) / "history.json"
            append_observation(store, "m", "a.jpg", "banana-1", "2026-11-18T09:00:00")
            second = append_observation(store, "m", "b.jpg", "banana-1", "2026-11-20T09:00:00")
            self.assertEqual(second["score_change_per_day"], 10)
            self.assertEqual(len(json.loads(store.read_text())), 2)

    @patch("usefirst.ranking.predict")
    def test_duplicate_observation_rejected(self, fake_predict):
        fake_predict.return_value = {"stage": "fresh", "visible_deterioration_score": 10, "confidence": .9, "uncertain": False}
        with tempfile.TemporaryDirectory() as temporary:
            store = Path(temporary) / "history.json"
            append_observation(store, "m", "a.jpg", "banana-1", "2026-11-18T09:00:00")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                append_observation(store, "m", "a.jpg", "banana-1", "2026-11-18T09:00:00")


if __name__ == "__main__":
    unittest.main()
