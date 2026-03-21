"""Risk scoring service.

Combines the outputs of the rule engine and the ML model into a single
normalised risk score and a human-readable risk level.
"""

from __future__ import annotations

from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

# Weight applied to the rule-based signal (0–1)
RULE_WEIGHT: float = 0.4

# Weight applied to the ML fraud probability (0–1)
ML_WEIGHT: float = 0.6

# Thresholds that map a numeric score to a risk level label
RISK_THRESHOLDS: dict[str, float] = {
    "HIGH": 0.65,
    "MEDIUM": 0.35,
    "LOW": 0.0,
}


def compute_score(
    rule_result: dict[str, Any],
    ml_result: dict[str, Any],
) -> dict[str, Any]:
    """Compute a combined risk score from rule and ML signals.

    Args:
        rule_result: Output of ``rule_engine.evaluate_rules``.
        ml_result:   Output of ``ml_model.predict``.

    Returns:
        Dictionary with keys:
          - ``risk_score`` (float): weighted composite score in [0, 1].
          - ``risk_level`` (str): one of ``"HIGH"``, ``"MEDIUM"``, ``"LOW"``.
          - ``triggered_rules`` (list[str]): rules that fired.
          - ``fraud_probability`` (float): raw ML probability.
    """
    rule_signal: float = 1.0 if rule_result.get("rule_risk") else 0.0
    ml_signal: float = float(ml_result.get("fraud_probability", 0.0))

    risk_score: float = round(RULE_WEIGHT * rule_signal + ML_WEIGHT * ml_signal, 4)

    risk_level: str = _score_to_level(risk_score)

    logger.info(
        "Risk score computed: %.4f (%s) | rule_signal=%.1f | ml_probability=%.4f",
        risk_score,
        risk_level,
        rule_signal,
        ml_signal,
    )

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "triggered_rules": rule_result.get("triggered_rules", []),
        "fraud_probability": ml_signal,
    }


def _score_to_level(score: float) -> str:
    """Map a numeric score to a risk level string."""
    for level, threshold in RISK_THRESHOLDS.items():
        if score >= threshold:
            return level
    return "LOW"
