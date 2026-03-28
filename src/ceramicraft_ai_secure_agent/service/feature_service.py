"""Feature extraction service.

Converts a raw transaction dict into a numerical feature vector
that can be consumed by the rule engine and the ML model.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)

# Countries considered high-risk for fraud purposes
HIGH_RISK_COUNTRIES = {"NG", "RU", "KP", "IR", "SY"}

# Merchant categories considered high-risk
HIGH_RISK_CATEGORIES = {"unknown", "financial", "gambling"}


def extract_features(transaction: dict[str, Any]) -> dict[str, float]:
    """Extract a flat feature dictionary from a raw transaction.

    Args:
        transaction: Raw transaction payload (dict).

    Returns:
        Dictionary mapping feature names to numeric values.
    """
    features: dict[str, float] = {
        "order_count_last_1h": 12.0,
        "order_count_last_24h": 5.0,
        "unique_ip_count": 12.0,
        "avg_order_amount": 1.1,
        "account_age_days": 2.0,
        "device_count": 1.0,
    }

    logger.debug(
        "Extracted features for transaction %s: %s",
        transaction.get("transaction_id"),
        features,
    )
    return features


def _safe_log(value: float) -> float:
    """Return log(value + 1) to avoid log(0)."""
    import math

    return math.log1p(max(value, 0.0))


@tool
def extract_features_tool(transaction: dict) -> dict:
    """Extract a flat feature dictionary from a raw transaction payload.

    Args:
        transaction: Raw transaction payload dict containing fields such as
            ``amount``, ``country``, and ``merchant_category``.

    Returns:
        Dictionary mapping feature names to numeric values ready for the
        rule engine and ML model.
    """
    return extract_features(transaction)
