"""Agent service.

Orchestrates the full risk-assessment pipeline:
  feature extraction → rule evaluation → ML prediction → risk scoring.

This is the single entry point used by the API layer.
"""

from __future__ import annotations

from typing import Any

from ceramicraft_ai_secure_agent.service import feature_service, ml_model, risk_scoring, rule_engine
from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)


def assess_risk(transaction: dict[str, Any]) -> dict[str, Any]:
    """Run the complete risk-assessment pipeline for a transaction.

    Args:
        transaction: Raw transaction payload.

    Returns:
        Final risk assessment dict containing:
          - ``transaction_id`` (str)
          - ``risk_score`` (float)
          - ``risk_level`` (str)
          - ``triggered_rules`` (list[str])
          - ``fraud_probability`` (float)
          - ``recommendation`` (str)
    """
    txn_id: str = str(transaction.get("transaction_id", "unknown"))
    logger.info("Starting risk assessment for transaction: %s", txn_id)

    # Step 1: Feature extraction
    features = feature_service.extract_features(transaction)

    # Step 2: Rule-based evaluation
    rule_result = rule_engine.evaluate_rules(features)

    # Step 3: ML model prediction
    ml_result = ml_model.predict(features)

    # Step 4: Composite risk scoring
    score_result = risk_scoring.compute_score(rule_result, ml_result)

    # Step 5: Build final response
    recommendation = _build_recommendation(score_result["risk_level"])

    assessment: dict[str, Any] = {
        "transaction_id": txn_id,
        **score_result,
        "recommendation": recommendation,
    }

    logger.info(
        "Risk assessment complete for %s: level=%s score=%.4f",
        txn_id,
        score_result["risk_level"],
        score_result["risk_score"],
    )

    return assessment


def _build_recommendation(risk_level: str) -> str:
    """Return a human-readable recommendation based on the risk level."""
    recommendations: dict[str, str] = {
        "HIGH": "Block transaction and alert fraud team immediately.",
        "MEDIUM": "Flag for manual review before processing.",
        "LOW": "Transaction appears legitimate. Approve.",
    }
    return recommendations.get(risk_level, "Insufficient data. Review manually.")
