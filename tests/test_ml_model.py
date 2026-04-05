"""unittest-style tests for ml_model.

This version is designed for VS Code unittest discovery.
Adjust MODULE_UNDER_TEST if your real import path is different.
"""

from __future__ import annotations

import importlib
import unittest
from unittest.mock import MagicMock, mock_open, patch


MODULE_UNDER_TEST = "ceramicraft_ai_secure_agent.service.ml_model"


class TestMLModel(unittest.TestCase):
    def setUp(self):
        self.ml_model = importlib.import_module(MODULE_UNDER_TEST)
        self.ml_model._model = None

    def tearDown(self):
        self.ml_model._model = None

    def test_load_model_loads_once_and_caches_instance(self):
        fake_model = MagicMock()

        with (
            patch.object(self.ml_model, "joblib") as mock_joblib,
            patch(
                "builtins.open", mock_open(read_data=b"fake-binary"), create=True
            ) as mock_file,
        ):
            mock_joblib.load.return_value = fake_model

            first = self.ml_model._load_model()
            second = self.ml_model._load_model()

        self.assertIs(first, fake_model)
        self.assertIs(second, fake_model)
        mock_joblib.load.assert_called_once()
        mock_file.assert_called_once()

    def test_load_model_uses_env_path_override(self):
        fake_model = MagicMock()
        custom_path = "/tmp/custom_fraud_model.joblib"

        with (
            patch.dict("os.environ", {"FRAUD_MODEL_PATH": custom_path}, clear=False),
            patch.object(self.ml_model, "joblib") as mock_joblib,
            patch(
                "builtins.open", mock_open(read_data=b"fake-binary"), create=True
            ) as mock_file,
        ):
            mock_joblib.load.return_value = fake_model

            result = self.ml_model._load_model()

        self.assertIs(result, fake_model)
        opened_path = mock_file.call_args[0][0]
        self.assertEqual(str(opened_path), custom_path)

    def test_predict_returns_probability_and_label(self):
        fake_model = MagicMock()
        fake_model.predict_proba.return_value = [[0.2, 0.8]]
        fake_model.predict.return_value = [1]

        features = {
            col: idx + 1 for idx, col in enumerate(self.ml_model.FEATURE_COLUMNS)
        }

        with patch.object(self.ml_model, "_load_model", return_value=fake_model):
            result = self.ml_model.predict(features)

        self.assertEqual(
            result,
            {"fraud_probability": 0.8, "ml_prediction": 1},
        )

        expected_vector = [[features[col] for col in self.ml_model.FEATURE_COLUMNS]]
        self.assertEqual(fake_model.predict_proba.call_args[0][0], expected_vector)
        self.assertEqual(fake_model.predict.call_args[0][0], expected_vector)

    def test_predict_fills_missing_features_with_zero(self):
        fake_model = MagicMock()
        fake_model.predict_proba.return_value = [[0.9, 0.1]]
        fake_model.predict.return_value = [0]

        partial_features = {}
        if self.ml_model.FEATURE_COLUMNS:
            partial_features[self.ml_model.FEATURE_COLUMNS[0]] = 123

        with patch.object(self.ml_model, "_load_model", return_value=fake_model):
            result = self.ml_model.predict(partial_features)

        self.assertEqual(
            result,
            {"fraud_probability": 0.1, "ml_prediction": 0},
        )

        sent_vector = fake_model.predict_proba.call_args[0][0][0]
        self.assertEqual(len(sent_vector), len(self.ml_model.FEATURE_COLUMNS))
        if self.ml_model.FEATURE_COLUMNS:
            self.assertEqual(sent_vector[0], 123)
            self.assertTrue(all(value == 0.0 for value in sent_vector[1:]))

    def test_predict_returns_safe_default_when_model_call_fails(self):
        fake_model = MagicMock()
        fake_model.predict_proba.side_effect = RuntimeError("boom")

        with patch.object(self.ml_model, "_load_model", return_value=fake_model):
            result = self.ml_model.predict({"some_feature": 1})

        self.assertEqual(
            result,
            {"fraud_probability": 0.5, "ml_prediction": 0},
        )

    def test_predict_tool_delegates_to_predict(self):
        features = {"a": 1}
        expected = {"fraud_probability": 0.77, "ml_prediction": 1}

        with patch.object(
            self.ml_model, "predict", return_value=expected
        ) as mock_predict:
            result = self.ml_model.predict_tool.func(features)

        self.assertEqual(result, expected)
        mock_predict.assert_called_once_with(features)


if __name__ == "__main__":
    unittest.main()
