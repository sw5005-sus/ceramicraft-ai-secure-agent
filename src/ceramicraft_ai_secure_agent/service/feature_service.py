"""Feature extraction service.

Converts a raw transaction dict into a numerical feature vector
that can be consumed by the rule engine and the ML model.
"""

from __future__ import annotations

from typing import Any

from ceramicraft_ai_secure_agent.service.rule_engine import LARGE_AMOUNT_THRESHOLD
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
    amount: float = float(transaction.get("amount", 0.0))
    country: str = str(transaction.get("country", "")).upper()
    category: str = str(transaction.get("merchant_category", "")).lower()

    features: dict[str, float] = {
        "amount": amount,
        "is_high_risk_country": 1.0 if country in HIGH_RISK_COUNTRIES else 0.0,
        "is_high_risk_category": 1.0 if category in HIGH_RISK_CATEGORIES else 0.0,
        "amount_log": _safe_log(amount),
        "is_large_amount": 1.0 if amount >= LARGE_AMOUNT_THRESHOLD else 0.0,
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
