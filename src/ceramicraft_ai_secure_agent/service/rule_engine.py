"""Rule-based risk engine.

Evaluates a feature vector against hard-coded business rules and
returns a list of triggered rule names plus a boolean risk flag.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)

# Amount threshold (USD) that triggers the large-transaction rule
LARGE_AMOUNT_THRESHOLD: float = 5_000.0


class RuleEvaluationResult:
    def __init__(self, rule_score: float, hits: list[str], reasons: list[str]):
        self.rule_score = rule_score
        self.hits = hits
        self.reasons = reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_score": self.rule_score,
            "hits": self.hits,
            "reasons": self.reasons,
        }


def evaluate_rules(features: dict[str, Any]) -> dict[str, Any]:
    """Apply all business rules to the given feature set.

    Args:
        features: Feature dictionary produced by ``feature_service.extract_features``.

    Returns:
        Dictionary with keys:
          - ``triggered_rules`` (list[str]): names of rules that fired.
          - ``rule_risk`` (bool): ``True`` if *any* rule triggered.
    """
    result = RuleEvaluationResult(rule_score=0.0, hits=[], reasons=[])

    if float(features.get("order_count_last_1h", 0)) >= 10:
        result.hits.append("high_order_count_last_1h")
        result.rule_score += 0.3
        result.reasons.append("More than 10 orders in the last hour")
        logger.debug(
            "Rule triggered: high_order_count_last_1h (order_count_last_1h=%s)",
            features.get("order_count_last_1h"),
        )

    if float(features.get("unique_ip_count", 0)) >= 5:
        result.hits.append("multiple_unique_ips")
        result.rule_score += 0.25
        result.reasons.append("Multiple unique IPs associated with the account")
        logger.debug(
            "Rule triggered: multiple_unique_ips (unique_ip_count=%s)",
            features.get("unique_ip_count"),
        )

    if (
        float(features.get("avg_order_amount", 0.0)) < 20.0
        and float(features.get("order_count_last_24h", 0.0)) >= 15
    ):
        result.hits.append("suspicious_order_pattern")
        result.rule_score += 0.2
        result.reasons.append("Many low-value orders in the last 24 hours")
        logger.debug(
            "Rule triggered: suspicious_order_pattern "
            "(avg_order_amount=%s, order_count_last_24h=%s)",
            features.get("avg_order_amount"),
            features.get("order_count_last_24h"),
        )

    if (
        float(features.get("account_age_days", 0.0)) <= 30.0
        and float(features.get("order_count_last_1h", 0.0)) >= 6
    ):
        result.hits.append("new_account_high_activity")
        result.rule_score += 0.15
        result.reasons.append("New account with high order activity in the last hour")
        logger.debug(
            "Rule triggered: new_account_high_activity "
            "(account_age_days=%s, order_count_last_1h=%s)",
            features.get("account_age_days"),
            features.get("order_count_last_1h"),
        )

    if (
        float(features.get("device_count", 0.0)) >= 4.0
        and float(features.get("unique_ip_count", 0.0)) >= 3.0
    ):
        result.hits.append("multiple_devices_and_ips")
        result.rule_score += 0.1
        result.reasons.append("Multiple devices and IPs associated with the account")
        logger.debug(
            "Rule triggered: multiple_devices_and_ips "
            "(device_count=%s, unique_ip_count=%s)",
            features.get("device_count"),
            features.get("unique_ip_count"),
        )

    if result.rule_score > 1.0:
        result.rule_score = 1.0
    return result.to_dict()


@tool
def evaluate_rules_tool(features: dict) -> dict:
    """Apply all business rules to a feature set and return triggered rules.

    Args:
        features: Feature dictionary produced by ``extract_features_tool``,
            containing numeric flags such as ``is_large_amount`` and
            ``is_high_risk_country``.

    Returns:
        Dictionary with ``triggered_rules`` (list of rule names that fired)
        and ``rule_risk`` (bool indicating whether any rule triggered).
    """
    return evaluate_rules(features)
