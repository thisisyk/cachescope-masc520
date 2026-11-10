"""Tests of feature boundaries, leakage protection, and real training artifacts."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from cachescope.features import FEATURE_NAMES, extract_features


class FeaturesTests(unittest.TestCase):
    def test_information_loss_features_and_empty_sets_are_finite(self):
        original = "Project Cedar starts on 2026-04-17. Its budget is 8000 dollars. It is not canceled."
        candidate = "Project Cedar starts on 2026-04-17."
        features = extract_features(original, "What is the budget for Project Cedar?", candidate, 50, 25)
        self.assertEqual(tuple(features), FEATURE_NAMES)
        self.assertEqual(features["removed_fraction"], 0.5)
        self.assertLess(features["numeric_retention"], 1)
        self.assertEqual(features["negation_retention"], 0)
        no_numbers = extract_features("Cedar belongs to Mira.", "Who owns Cedar?", "Cedar belongs to Mira.", 10, 10)
        self.assertEqual(no_numbers["numeric_retention"], 1)
        terminal_number = extract_features("Cedar has 42. It belongs to Mira.", "Who owns Cedar?", "It belongs to Mira.", 20, 10)
        self.assertEqual(terminal_number["numeric_retention"], 0)
        with self.assertRaises(ValueError):
            extract_features("a", "b", "a", 0, 0)


_ML_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "sklearn", "torch", "joblib"))


@unittest.skipUnless(_ML_AVAILABLE, "ML dependencies unavailable")
class ModelTests(unittest.TestCase):
    def _records(self):
        records = []
        for split, count in (("train", 24), ("validation", 8), ("test", 8)):
            for index in range(count):
                dangerous = bool(index % 2)
                features = {name: float(index % 3) for name in FEATURE_NAMES}
                features["numeric_retention"] = 0.0 if dangerous else 1.0
                features["original_tokens"] = 20.0 if split == "train" else 1000.0
                records.append({
                    "task_id": f"{split}-{index}", "group_id": f"{split}-family-{index // 4}",
                    "split": split, "candidate_id": "60", "features": features,
                    "original_correct": index != 0, "compressed_correct": not dangerous,
                    "regression": int(index != 0 and dangerous),
                })
        return records

    def test_outcomes_and_gold_cannot_change_feature_vector(self):
        from cachescope.models import feature_row
        record = {"original_context": "Cedar has 42 units.", "question": "How many units?",
                  "compressed_context": "Cedar has 42 units.", "original_tokens": 15, "compressed_tokens": 15}
        expected = feature_row(record)
        record.update(answer="DO NOT USE", original_correct=False, compressed_correct=True,
                      regression=1, model_response="DO NOT USE")
        self.assertEqual(expected, feature_row(record))

    def test_group_leakage_and_one_class_are_rejected(self):
        from cachescope.models import train_models
        records = self._records()
        records[-1]["group_id"] = records[0]["group_id"]
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "Data leakage"):
                train_models(records, temp, seeds=(520,), max_epochs=2, run_ablations=False)
        records = self._records()
        for record in records:
            record["compressed_correct"] = True
            record["regression"] = 0
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "both observed"):
                train_models(records, temp, seeds=(520,), max_epochs=2, run_ablations=False)

    def test_saved_models_use_train_scaler_and_roundtrip(self):
        import numpy as np
        from cachescope.models import load_bundle, predict_scores, train_models
        records = self._records()
        with tempfile.TemporaryDirectory() as temp:
            report = train_models(records, temp, seeds=(520,), max_epochs=3, patience=2, run_ablations=False)
            self.assertEqual(report["split_counts"]["train"]["original_correct_candidates"], 23)
            for name in ("lr", "rf", "mlp"):
                bundle = load_bundle(temp, name)
                self.assertEqual(bundle["scaler"].mean_[0], 20.0)
                scores = predict_scores(bundle, records[-8:])
                self.assertEqual(scores.shape, (8,))
                self.assertTrue(np.isfinite(scores).all())
                self.assertTrue(((scores >= 0) & (scores <= 1)).all())
            self.assertTrue((Path(temp) / "mlp_seed_520.pt").exists())


if __name__ == "__main__":
    unittest.main()
