"""Rule-based risk engine.

Evaluates a feature vector against hard-coded business rules and
returns a weighted rule score, triggered rule names, and reasons.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def evaluate_rules(features: dict[str, Any]) -> dict[str, Any]:
    """Apply business rules to the given feature set.

    Expected feature keys:
      - order_count_last_1h
      - order_count_last_24h
      - unique_ip_count
      - avg_order_amount_global
      - avg_order_amount_today
      - account_age_days
      - receive_address_count
      - last_status

    Args:
        features: Feature dictionary produced by feature extraction.

    Returns:
        Dictionary with:
          - rule_score: float in [0, 1]
          - hits: list[str]
          - reasons: list[str]
    """
    result = RuleEvaluationResult(rule_score=0.0, hits=[], reasons=[])

    order_count_last_1h = _safe_float(features.get("order_count_last_1h"))
    order_count_last_24h = _safe_float(features.get("order_count_last_24h"))
    unique_ip_count = _safe_float(features.get("unique_ip_count"))
    avg_order_amount_global = _safe_float(features.get("avg_order_amount_global"))
    avg_order_amount_today = _safe_float(features.get("avg_order_amount_today"))
    account_age_days = _safe_float(features.get("account_age_days"))
    receive_address_count = _safe_float(features.get("receive_address_count"))
    last_status = features.get("last_status")

    amount_shift_ratio = (
        avg_order_amount_today / avg_order_amount_global
        if avg_order_amount_global > 0
        else 1.0
    )

    # Rule 1: burst order activity in short window
    if order_count_last_1h >= 10:
        result.hits.append("high_order_count_last_1h")
        result.rule_score += 0.28
        result.reasons.append("More than 10 orders in the last hour")
        logger.debug(
            "Rule triggered: high_order_count_last_1h (order_count_last_1h=%s)",
            order_count_last_1h,
        )

    # Rule 2: unusually high daily order volume
    if order_count_last_24h >= 20:
        result.hits.append("high_order_count_last_24h")
        result.rule_score += 0.18
        result.reasons.append("Unusually high order volume in the last 24 hours")
        logger.debug(
            "Rule triggered: high_order_count_last_24h (order_count_last_24h=%s)",
            order_count_last_24h,
        )

    # Rule 3: multiple IPs tied to one account
    if unique_ip_count >= 5:
        result.hits.append("multiple_unique_ips")
        result.rule_score += 0.24
        result.reasons.append("Multiple unique IPs associated with the account")
        logger.debug(
            "Rule triggered: multiple_unique_ips (unique_ip_count=%s)",
            unique_ip_count,
        )

    # Rule 4: many low-value orders today compared with historical average
    if avg_order_amount_today < 20.0 and order_count_last_24h >= 15:
        result.hits.append("many_low_value_orders_today")
        result.rule_score += 0.16
        result.reasons.append("Many low-value orders placed in the last 24 hours")
        logger.debug(
            "Rule triggered: many_low_value_orders_today "
            "(avg_order_amount_today=%s, order_count_last_24h=%s)",
            avg_order_amount_today,
            order_count_last_24h,
        )

    # Rule 5: today's order amount drops significantly from historical behavior
    if (
        avg_order_amount_global > 0
        and amount_shift_ratio <= 0.35
        and order_count_last_24h >= 8
    ):
        result.hits.append("amount_drop_vs_history")
        result.rule_score += 0.14
        result.reasons.append(
            "Today's average order amount is significantly lower than "
            "historical average"
        )
        logger.debug(
            "Rule triggered: amount_drop_vs_history "
            "(avg_order_amount_today=%s, avg_order_amount_global=%s, ratio=%s)",
            avg_order_amount_today,
            avg_order_amount_global,
            amount_shift_ratio,
        )

    # Rule 6: new account with high activity
    if account_age_days <= 30 and order_count_last_1h >= 6:
        result.hits.append("new_account_high_activity")
        result.rule_score += 0.16
        result.reasons.append("New account with high order activity in the last hour")
        logger.debug(
            "Rule triggered: new_account_high_activity "
            "(account_age_days=%s, order_count_last_1h=%s)",
            account_age_days,
            order_count_last_1h,
        )

    # Rule 7: one account linked to many receiving addresses
    if receive_address_count >= 4:
        result.hits.append("multiple_receive_addresses")
        result.rule_score += 0.18
        result.reasons.append("Account is associated with multiple receiving addresses")
        logger.debug(
            "Rule triggered: multiple_receive_addresses (receive_address_count=%s)",
            receive_address_count,
        )

    # Rule 8: combined network/address anomaly
    if unique_ip_count >= 3 and receive_address_count >= 3:
        result.hits.append("ip_address_combination_anomaly")
        result.rule_score += 0.12
        result.reasons.append(
            "Account is associated with both multiple IPs "
            "and multiple receiving addresses"
        )
        logger.debug(
            "Rule triggered: ip_address_combination_anomaly "
            "(unique_ip_count=%s, receive_address_count=%s)",
            unique_ip_count,
            receive_address_count,
        )

    # Rule 9: combined watchlist user punishment
    if last_status == "watchlist":
        result.hits.append("watchlist_user")
        result.rule_score += 0.10
        result.reasons.append(
            "User is on the watchlist, indicating prior suspicious activity"
        )
        logger.debug(
            "Rule triggered: watchlist_user (last_status=%s)",
            last_status,
        )

    if result.rule_score > 1.0:
        result.rule_score = 1.0

    return result.to_dict()


@tool
def evaluate_rules_tool(features: dict) -> dict:
    """Apply business rules to a feature set and return the evaluation result.

    Args:
        features: Feature dictionary with the following numeric fields:
            - order_count_last_1h
            - order_count_last_24h
            - unique_ip_count
            - avg_order_amount_global
            - avg_order_amount_today
            - account_age_days
            - receive_address_count

    Returns:
        Dictionary with:
            - rule_score: weighted score in [0, 1]
            - hits: list of triggered rule names
            - reasons: list of human-readable explanations
    """
    return evaluate_rules(features)
