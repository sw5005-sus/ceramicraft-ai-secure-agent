"""unittest-style tests for ml_model.

Updated for the JSON-based logistic regression inference implementation.
"""

from __future__ import annotations

import importlib
import json
import math
import unittest
from unittest.mock import mock_open, patch


MODULE_UNDER_TEST = "ceramicraft_ai_secure_agent.service.ml_model"


class TestMLModel(unittest.TestCase):
    def setUp(self):
        self.ml_model = importlib.import_module(MODULE_UNDER_TEST)
        self.ml_model._model = None

    def tearDown(self):
        self.ml_model._model = None

    def test_load_model_loads_once_and_caches_instance(self):
        fake_model = {
            "feature_columns": ["f1", "f2"],
            "coef": [0.1, 0.2],
            "intercept": -0.3,
        }

        with patch(
            "builtins.open",
            mock_open(read_data=json.dumps(fake_model)),
            create=True,
        ) as mock_file:
            first = self.ml_model._load_model()
            second = self.ml_model._load_model()

        self.assertEqual(first, fake_model)
        self.assertIs(first, second)
        mock_file.assert_called_once()

    def test_load_model_uses_env_path_override(self):
        fake_model = {
            "feature_columns": ["f1"],
            "coef": [0.5],
            "intercept": 0.0,
        }
        custom_path = "/tmp/custom_model_weights.json"

        with (
            patch.dict("os.environ", {"FRAUD_MODEL_PATH": custom_path}, clear=False),
            patch(
                "builtins.open",
                mock_open(read_data=json.dumps(fake_model)),
                create=True,
            ) as mock_file,
        ):
            result = self.ml_model._load_model()

        self.assertEqual(result, fake_model)
        opened_path = mock_file.call_args[0][0]
        self.assertEqual(str(opened_path), custom_path)

    def test_sigmoid_zero(self):
        self.assertAlmostEqual(self.ml_model._sigmoid(0.0), 0.5, places=6)

    def test_sigmoid_positive_and_negative(self):
        self.assertGreater(self.ml_model._sigmoid(2.0), 0.5)
        self.assertLess(self.ml_model._sigmoid(-2.0), 0.5)

    def test_predict_proba_from_features_computes_expected_probability(self):
        fake_model = {
            "feature_columns": ["a", "b", "c"],
            "coef": [0.5, -0.25, 1.0],
            "intercept": -1.0,
        }
        features = {
            "a": 2.0,
            "b": 4.0,
            "c": 1.0,
        }

        # score = -1.0 + 2*0.5 + 4*(-0.25) + 1*1.0 = 0.0
        # sigmoid(0) = 0.5
        with patch.object(self.ml_model, "_load_model", return_value=fake_model):
            prob = self.ml_model.predict_proba_from_features(features)

        self.assertAlmostEqual(prob, 0.5, places=6)

    def test_predict_proba_from_features_fills_missing_features_with_zero(self):
        fake_model = {
            "feature_columns": ["a", "b"],
            "coef": [1.0, 2.0],
            "intercept": 0.0,
        }
        features = {
            "a": 1.0,
            # "b" is missing -> should default to 0.0
        }

        with patch.object(self.ml_model, "_load_model", return_value=fake_model):
            prob = self.ml_model.predict_proba_from_features(features)

        expected = 1.0 / (1.0 + math.exp(-1.0))
        self.assertAlmostEqual(prob, expected, places=6)

    def test_predict_returns_probability_and_prediction_one(self):
        features = {"some_feature": 1}

        with patch.object(
            self.ml_model,
            "predict_proba_from_features",
            return_value=0.8,
        ):
            result = self.ml_model.predict(features)

        self.assertEqual(
            result,
            {
                "fraud_probability": 0.8,
                "prediction": 1,
            },
        )

    def test_predict_returns_probability_and_prediction_zero(self):
        features = {"some_feature": 1}

        with patch.object(
            self.ml_model,
            "predict_proba_from_features",
            return_value=0.3,
        ):
            result = self.ml_model.predict(features)

        self.assertEqual(
            result,
            {
                "fraud_probability": 0.3,
                "prediction": 0,
            },
        )

    def test_predict_returns_safe_default_when_prediction_fails(self):
        with patch.object(
            self.ml_model,
            "predict_proba_from_features",
            side_effect=RuntimeError("boom"),
        ):
            result = self.ml_model.predict({"some_feature": 1})

        # Note:
        # Current implementation returns `ml_prediction` on the error path,
        # while the success path returns `prediction`.
        self.assertEqual(
            result,
            {
                "fraud_probability": 0.5,
                "prediction": 0,
            },
        )

    def test_predict_tool_delegates_to_predict(self):
        features = {"a": 1}
        expected = {"fraud_probability": 0.77, "prediction": 1}

        with patch.object(
            self.ml_model,
            "predict",
            return_value=expected,
        ) as mock_predict:
            result = self.ml_model.predict_tool.func(features)

        self.assertEqual(result, expected)
        mock_predict.assert_called_once_with(features)


if __name__ == "__main__":
    unittest.main()
