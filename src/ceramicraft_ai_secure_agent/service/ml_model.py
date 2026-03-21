"""ML model service.

Loads the serialised fraud-detection model from disk and exposes a
``predict`` function that returns the fraud probability for a feature set.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)

# Resolve path relative to this file so the service works regardless of the
# working directory.
_MODEL_PATH: Path = Path(__file__).resolve().parent.parent / "model" / "fraud_model.pkl"

# Feature column order must match what was used during training.
FEATURE_COLUMNS: list[str] = [
    "amount",
    "is_high_risk_country",
    "is_high_risk_category",
    "amount_log",
    "is_large_amount",
]

_model = None


def _load_model():
    """Lazy-load the model from disk (singleton)."""
    global _model
    if _model is None:
        model_path = Path(os.environ.get("FRAUD_MODEL_PATH", str(_MODEL_PATH)))
        logger.info("Loading fraud model from %s", model_path)
        with open(model_path, "rb") as f:
            _model = pickle.load(f)  # noqa: S301
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

    feature_vector = [[features.get(col, 0.0) for col in FEATURE_COLUMNS]]
    fraud_probability: float = float(model.predict_proba(feature_vector)[0][1])
    ml_prediction: int = int(model.predict(feature_vector)[0])

    logger.info("ML prediction: label=%d, probability=%.4f", ml_prediction, fraud_probability)

    return {
        "fraud_probability": fraud_probability,
        "ml_prediction": ml_prediction,
    }
