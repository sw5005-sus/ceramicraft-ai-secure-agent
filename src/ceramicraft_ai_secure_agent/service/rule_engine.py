"""Rule-based risk engine.

Evaluates a feature vector against hard-coded business rules and
returns a list of triggered rule names plus a boolean risk flag.
"""

from __future__ import annotations

from typing import Any

from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)

# Amount threshold (USD) that triggers the large-transaction rule
LARGE_AMOUNT_THRESHOLD: float = 5_000.0


def evaluate_rules(features: dict[str, Any]) -> dict[str, Any]:
    """Apply all business rules to the given feature set.

    Args:
        features: Feature dictionary produced by ``feature_service.extract_features``.

    Returns:
        Dictionary with keys:
          - ``triggered_rules`` (list[str]): names of rules that fired.
          - ``rule_risk`` (bool): ``True`` if *any* rule triggered.
    """
    triggered: list[str] = []

    if features.get("is_large_amount"):
        triggered.append("large_amount")

    if features.get("is_high_risk_country"):
        triggered.append("high_risk_country")

    if features.get("is_high_risk_category"):
        triggered.append("high_risk_category")

    # Combined rule: large amount AND high-risk country is an immediate flag
    if features.get("is_large_amount") and features.get("is_high_risk_country"):
        triggered.append("large_amount_in_high_risk_country")

    rule_risk = len(triggered) > 0

    logger.info("Rule evaluation complete. Triggered rules: %s", triggered)

    return {
        "triggered_rules": triggered,
        "rule_risk": rule_risk,
    }
