"""ML model service.

Loads the serialised fraud-detection model from disk and exposes a
``predict`` function that returns the fraud probability for a feature set.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)

# Resolve path relative to this file so the service works regardless of the
# working directory.
_MODEL_PATH: Path = (
    Path(__file__).resolve().parent.parent / "model" / "model_weights.json"
)

_MODEL_VERSION = "v3"
_model = None


def _load_model():
    """Lazy-load the model from disk (singleton)."""
    global _model
    if _model is None:
        model_path = Path(os.environ.get("FRAUD_MODEL_PATH", str(_MODEL_PATH)))
        logger.info("Loading fraud model from %s", model_path)
        with open(model_path, "r", encoding="utf-8") as f:
            _model = json.load(f)
        logger.info("Fraud model loaded successfully.")
    return _model


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def predict_proba_from_features(features: dict[str, float]) -> tuple[float, list[dict]]:
    model = _load_model()
    coef = model["coef"]
    intercept = model["intercept"]
    means = model["mean"]
    stds = model["std"]
    feature_contributions = []
    total_abs_score = abs(intercept)
    feature_contributions.append({"feature": "intercept", "contribution": intercept})
    score = intercept
    for c in model["feature_columns"]:
        raw_val = float(features.get(c, 0.0))

        mean_val = means[c]
        std_val = stds[c]
        w = coef[c]

        scaled_val = (raw_val - mean_val) / (std_val if std_val != 0 else 1.0)
        contribution = scaled_val * w
        feature_contributions.append({"feature": c, "contribution": contribution})
        total_abs_score += abs(contribution)
        score += contribution

    top_features = sorted(
        [item for item in feature_contributions if item["feature"] != "intercept"],
        key=lambda x: abs(x["contribution"]),
        reverse=True,
    )[:3]
    explanation = []
    for item in top_features:
        percentage = (
            (abs(item["contribution"]) / total_abs_score) * 100
            if total_abs_score != 0
            else 0
        )
        explanation.append(
            {
                "name": item["feature"],
                "impact": round(item["contribution"], 4),
                "ratio": f"{round(percentage, 2)}%",
            }
        )
    return _sigmoid(score), explanation


def predict(features: dict[str, Any]) -> dict[str, Any]:
    """Return fraud probability and predicted label for a feature dict.

    Args:
        features: Feature dictionary produced by ``feature_service.extract_features``.
    Returns:
        Dictionary with keys:
          - ``fraud_probability`` (float): probability in [0, 1].
          - ``prediction`` (int): 1 = fraud, 0 = legitimate.
    """
    try:
        prob, explanation = predict_proba_from_features(features)
        return {
            "fraud_probability": prob,
            "prediction": 1 if prob >= 0.5 else 0,
            "explanation": explanation,
            "model_version": _MODEL_VERSION,
        }
    except Exception as e:
        logger.error(f"ML Prediction failed: {str(e)}")
        return {"fraud_probability": 0.5, "prediction": 0}


@tool
def predict_tool(features: dict) -> dict:
    """Run the ML fraud-detection model on a feature set.

    Args:
        features: Feature dictionary produced by ``extract_features_tool``,
            with keys matching the model's expected feature columns.

    Returns:
        Dictionary with ``fraud_probability`` (float in [0, 1]) and
        ``prediction`` (int: 1 = fraud, 0 = legitimate), ``explanation`` (list).
    """
    return predict(features)
