"""ML model service.

Loads the serialised fraud-detection model from disk and exposes a
``predict`` function that returns the fraud probability for a feature set.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
from langchain_core.tools import tool

from ceramicraft_ai_secure_agent.model.train_model import FEATURE_COLUMNS
from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)

# Resolve path relative to this file so the service works regardless of the
# working directory.
_MODEL_PATH: Path = (
    Path(__file__).resolve().parent.parent / "model" / "fraud_logistic_regression.pkl"
)

_model = None


def _load_model():
    """Lazy-load the model from disk (singleton)."""
    global _model
    if _model is None:
        model_path = Path(os.environ.get("FRAUD_MODEL_PATH", str(_MODEL_PATH)))
        logger.info("Loading fraud model from %s", model_path)
        with open(model_path, "rb") as f:
            _model = joblib.load(f)  # noqa: S301
        logger.info("Fraud model loaded successfully.")
    return _model


def predict(features: dict[str, Any]) -> dict[str, Any]:
    """Return fraud probability and predicted label for a feature dict.

    Args:
        features: Feature dictionary produced by ``feature_service.extract_features``.

    Returns:
        Dictionary with keys:
          - ``fraud_probability`` (float): probability in [0, 1].
          - ``ml_prediction`` (int): 1 = fraud, 0 = legitimate.
    """
    model = _load_model()
    try:
        feature_vector = [[features.get(col, 0.0) for col in FEATURE_COLUMNS]]
        fraud_probability: float = float(model.predict_proba(feature_vector)[0][1])
        ml_prediction: int = int(model.predict(feature_vector)[0])

        logger.info(
            "ML prediction: label=%d, probability=%.4f",
            ml_prediction,
            fraud_probability,
        )

        return {
            "fraud_probability": fraud_probability,
            "ml_prediction": ml_prediction,
        }
    except Exception as e:
        logger.error(f"ML Prediction failed: {str(e)}")
        # 容错处理：发生异常时返回中立结果，防止整个 Agent 崩溃
        return {"fraud_probability": 0.5, "ml_prediction": 0}


@tool
def predict_tool(features: dict) -> dict:
    """Run the ML fraud-detection model on a feature set.

    Args:
        features: Feature dictionary produced by ``extract_features_tool``,
            with keys matching the model's expected feature columns.

    Returns:
        Dictionary with ``fraud_probability`` (float in [0, 1]) and
        ``ml_prediction`` (int: 1 = fraud, 0 = legitimate).
    """
    return predict(features)
