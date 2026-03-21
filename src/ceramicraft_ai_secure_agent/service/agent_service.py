"""Agent service.

Orchestrates the full risk-assessment pipeline using LangGraph:
  feature extraction → rule evaluation → ML prediction → risk scoring → LLM judgment.

The LangGraph StateGraph defines each processing step as a node and wires them
into a deterministic sequential workflow. An OpenAI LLM provides the final risk
judgment and human-readable recommendation.

This is the single entry point used by the API layer.
"""

from __future__ import annotations

import os
from typing import Any, cast

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from ceramicraft_ai_secure_agent.service import risk_scoring
from ceramicraft_ai_secure_agent.service.feature_service import extract_features_tool
from ceramicraft_ai_secure_agent.service.ml_model import predict_tool
from ceramicraft_ai_secure_agent.service.rule_engine import evaluate_rules_tool
from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# LangGraph state definition
# ---------------------------------------------------------------------------


class _AssessmentState(TypedDict):
    """Mutable state passed between LangGraph nodes."""

    transaction: dict[str, Any]
    features: dict[str, float]
    rule_result: dict[str, Any]
    ml_result: dict[str, Any]
    score_result: dict[str, Any]
    recommendation: str


# ---------------------------------------------------------------------------
# LangGraph node functions
# ---------------------------------------------------------------------------


def _extract_features_node(state: _AssessmentState) -> dict[str, Any]:
    """Node: extract features from the raw transaction."""
    tool_input = cast(Any, {"transaction": state["transaction"]})
    res = extract_features_tool.invoke(tool_input)
    return {"features": res}


def _evaluate_rules_node(state: _AssessmentState) -> dict[str, Any]:
    """Node: apply business rules to the feature set."""
    tool_input = cast(Any, {"features": state["features"]})
    res = evaluate_rules_tool.invoke(tool_input)
    return {"rule_result": res}


def _predict_node(state: _AssessmentState) -> dict[str, Any]:
    """Node: run the ML model on the feature set."""
    tool_input = cast(Any, {"features": state["features"]})
    res = predict_tool.invoke(tool_input)
    return {"ml_result": res}


def _compute_score_node(state: _AssessmentState) -> dict[str, Any]:
    """Node: combine rule and ML signals into a composite risk score."""
    # state already unpacked at upstream node, use directly
    score_result = risk_scoring.compute_score(state["rule_result"], state["ml_result"])
    return {"score_result": score_result}


def _llm_judge_node(state: _AssessmentState) -> dict[str, Any]:
    """Node: use an OpenAI LLM to produce a risk judgment and recommendation.

    Falls back to a rule-based recommendation when ``OPENAI_API_KEY`` is not
    set so that the service degrades gracefully in environments without LLM
    access.
    """
    score = state["score_result"]
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning(
            "OPENAI_API_KEY not set – using rule-based recommendation fallback."
        )
        return {"recommendation": _build_recommendation(score["risk_level"])}

    triggered = (
        ", ".join(score["triggered_rules"]) if score["triggered_rules"] else "none"
    )
    prompt = (
        "You are a fraud risk assessment expert. "
        "Based on the following automated risk analysis, provide a single concise "
        "recommendation sentence for this transaction.\n\n"
        f"Risk Score: {score['risk_score']:.4f}\n"
        f"Risk Level: {score['risk_level']}\n"
        f"Triggered Rules: {triggered}\n"
        f"Fraud Probability: {score['fraud_probability']:.4f}"
    )

    llm = _get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"recommendation": response.content}


# ---------------------------------------------------------------------------
# Build and compile the LangGraph workflow (module-level singleton)
# ---------------------------------------------------------------------------

_builder: StateGraph = StateGraph(cast(Any, _AssessmentState))
_builder.add_node("extract_features", _extract_features_node)
_builder.add_node("evaluate_rules", _evaluate_rules_node)
_builder.add_node("predict", _predict_node)
_builder.add_node("compute_score", _compute_score_node)
_builder.add_node("llm_judge", _llm_judge_node)

_builder.set_entry_point("extract_features")
_builder.add_edge("extract_features", "evaluate_rules")
_builder.add_edge("extract_features", "predict")
_builder.add_edge("predict", "compute_score")
_builder.add_edge("compute_score", "llm_judge")
_builder.add_edge("llm_judge", END)

_graph = _builder.compile()

# Lazy LLM singleton – initialised only when an API key is available
_llm: ChatOpenAI | None = None


def _get_llm() -> ChatOpenAI:
    """Return a module-level ChatOpenAI singleton (created on first use)."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    return _llm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assess_risk(transaction: dict[str, Any]) -> dict[str, Any]:
    """Run the complete risk-assessment pipeline for a transaction.

    The pipeline is implemented as a LangGraph StateGraph:
      1. ``extract_features`` – convert raw fields to numeric features.
      2. ``evaluate_rules``   – apply hard-coded business rules.
      3. ``predict``          – run the ML fraud-detection model.
      4. ``compute_score``    – combine signals into a composite score.
      5. ``llm_judge``        – use an LLM to generate a recommendation.

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

    initial_state: _AssessmentState = {
        "transaction": transaction,
        "features": {},
        "rule_result": {},
        "ml_result": {},
        "score_result": {},
        "recommendation": "",
    }

    final_state = cast(_AssessmentState, _graph.invoke(initial_state))
    score = final_state["score_result"]

    assessment: dict[str, Any] = {
        "transaction_id": txn_id,
        "risk_score": score["risk_score"],
        "risk_level": score["risk_level"],
        "triggered_rules": score["triggered_rules"],
        "fraud_probability": score["fraud_probability"],
        "recommendation": final_state["recommendation"],
    }

    logger.info(
        "Risk assessment complete for %s: level=%s score=%.4f",
        txn_id,
        score["risk_level"],
        score["risk_score"],
    )

    return assessment


def _build_recommendation(risk_level: str) -> str:
    """Return a rule-based recommendation when the LLM is unavailable."""
    recommendations: dict[str, str] = {
        "HIGH": "Block transaction and alert fraud team immediately.",
        "MEDIUM": "Flag for manual review before processing.",
        "LOW": "Transaction appears legitimate. Approve.",
    }
    return recommendations.get(risk_level, "Insufficient data. Review manually.")
