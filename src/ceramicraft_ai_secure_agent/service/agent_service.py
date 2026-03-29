"""Agent service.

Orchestrates the full risk-assessment pipeline using LangGraph:
  feature extraction → rule evaluation → ML prediction → risk scoring → LLM judgment.

The LangGraph StateGraph defines each processing step as a node and wires them
into a deterministic sequential workflow. An OpenAI LLM provides the final risk
judgment and human-readable recommendation.

This is the single entry point used by the API layer.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any, TypeVar, cast

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from ceramicraft_ai_secure_agent.service import risk_scoring
from ceramicraft_ai_secure_agent.service.feature_service import extract_features_tool
from ceramicraft_ai_secure_agent.service.ml_model import predict_tool
from ceramicraft_ai_secure_agent.service.rule_engine import evaluate_rules_tool
from ceramicraft_ai_secure_agent.utils.logger import get_logger
from ceramicraft_ai_secure_agent.utils.mlflow_trace import (
    LLM_MODEL_NAME,
    PROMPT_NAME,
    PROMPT_VERSION,
    safe_update_current_trace,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy globals
# ---------------------------------------------------------------------------

_PROMPT_URI = "prompts:/fraud_recommendation_prompt@production"

_mlflow_initialized = False
_loaded_prompt: Any | None = None
_graph: Any | None = None
_llm: ChatOpenAI | None = None

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# MLflow helpers (lazy / optional)
# ---------------------------------------------------------------------------


def _mlflow_enabled() -> bool:
    """Whether MLflow tracing/integration is enabled.

    Default is disabled so tests/CI do not hang on import or require external
    MLflow services unless explicitly opted in.
    """
    return os.environ.get("ENABLE_MLFLOW_TRACING", "false").lower() == "true"


def _init_mlflow_once() -> None:
    """Initialise MLflow lazily and only once."""
    global _mlflow_initialized

    if _mlflow_initialized or not _mlflow_enabled():
        return

    import mlflow

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow.set_experiment("ai-secure-agent-llm-traces")
    mlflow.langchain.autolog()
    _mlflow_initialized = True


def _trace(name: str):
    """Lazy MLflow trace decorator.

    When tracing is disabled, this becomes a no-op decorator so module import
    stays side-effect free in tests and CI.
    """

    def decorator(func: F) -> F:
        if not _mlflow_enabled():
            return func

        import mlflow

        return mlflow.trace(name=name)(func)  # type: ignore[return-value]

    return decorator


def _safe_update_trace(metadata: dict[str, Any]) -> None:
    """Update current trace metadata only when MLflow tracing is enabled."""
    if not _mlflow_enabled():
        return

    _init_mlflow_once()
    safe_update_current_trace(metadata=metadata)


def _get_loaded_prompt() -> Any:
    """Return the loaded prompt template lazily.

    If MLflow is disabled, use a local fallback template string so tests can run
    without external dependencies.
    """
    global _loaded_prompt

    if _loaded_prompt is not None:
        return _loaded_prompt

    if not _mlflow_enabled():
        _loaded_prompt = (
            "You are a fraud risk assessment expert.\n\n"
            "Based on the following automated risk analysis, return JSON with keys:\n"
            "recommended_action, reason, analyst_summary, confidence.\n\n"
            "Risk Score: {risk_score}\n"
            "Risk Level: {risk_level}\n"
            "Triggered Rules: {triggered_rules}\n"
            "Fraud Probability: {fraud_probability}\n"
            "Previous Status: {previous_status}\n"
            "Feature Snapshot: {feature_snapshot}\n"
        )
        return _loaded_prompt

    _init_mlflow_once()

    import mlflow

    _loaded_prompt = mlflow.genai.load_prompt(_PROMPT_URI)
    return _loaded_prompt


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
    score_result = risk_scoring.compute_score(state["rule_result"], state["ml_result"])
    return {"score_result": score_result}


@_trace(name="llm_judge_node")
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

    _update_trace_with_score(state)
    prompt = _build_llm_prompt(state)
    llm = _get_llm()
    response = llm.invoke(prompt)
    return {"recommendation": response.content}


class Recommendation:
    """Class to represent a recommendation from the deserialized content."""

    def __init__(
        self,
        recommended_action: str,
        reason: str,
        analyst_summary: str,
        confidence: str,
    ) -> None:
        self.recommended_action = recommended_action
        self.reason = reason
        self.analyst_summary = analyst_summary
        self.confidence = confidence

    @classmethod
    def from_json(cls, json_str: str) -> Recommendation:
        """Deserialize from a JSON string."""
        if not json_str:
            logger.error("Empty recommendation string received from LLM.")
            return fallback_return
        try:
            data = json.loads(json_str)
            return cls(
                recommended_action=data["recommended_action"],
                reason=data["reason"],
                analyst_summary=data["analyst_summary"],
                confidence=data["confidence"],
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.error("Failed to decode JSON recommendation: %s", json_str)
            return fallback_return


fallback_return = Recommendation(
    recommended_action="manual_review",
    reason="LLM unavailable",
    analyst_summary="LLM API key not set. Defaulting to manual review.",
    confidence="low",
)


class Action:
    """Base class for actions."""

    def run(self, state: _AssessmentState) -> None:
        raise NotImplementedError("Subclasses should implement this method.")


class ManualReviewAction(Action):
    def run(self, state: _AssessmentState) -> None:
        logger.info("Flagging transaction for manual review.")


class BlockAction(Action):
    def run(self, state: _AssessmentState) -> None:
        logger.info("Blocking transaction.")


class WatchlistAction(Action):
    def run(self, state: _AssessmentState) -> None:
        logger.info("Adding transaction to watchlist.")


class AllowAction(Action):
    def run(self, state: _AssessmentState) -> None:
        logger.info("Allowing transaction to proceed.")


action_map = {
    "manual_review": ManualReviewAction(),
    "block": BlockAction(),
    "watchlist": WatchlistAction(),
    "allow": AllowAction(),
}


def _action_node(state: _AssessmentState) -> dict[str, Any]:
    """Node: execute the recommended action (placeholder)."""
    recommendation = state["recommendation"]
    logger.info("Executing action based on recommendation: %s", recommendation)
    action_key = Recommendation.from_json(recommendation).recommended_action.lower()
    action = action_map.get(action_key, AllowAction())
    action.run(state)
    return {}


def _update_trace_with_score(state: _AssessmentState) -> None:
    """Attach score metadata to the current trace if tracing is enabled."""
    transaction = state["transaction"]
    score = state["score_result"]

    trace_metadata: dict[str, Any] = {
        "transaction_id": transaction.get("transaction_id", "unknown"),
        "risk_level": score["risk_level"],
        "risk_score": score["risk_score"],
        "fraud_probability": score["fraud_probability"],
        "triggered_rules_count": len(score["triggered_rules"]),
        "prompt_name": PROMPT_NAME,
        "prompt_version": PROMPT_VERSION,
        "llm_model": LLM_MODEL_NAME,
        "fallback_used": not bool(os.environ.get("OPENAI_API_KEY", "")),
    }
    _safe_update_trace(trace_metadata)


def _build_llm_prompt(state: _AssessmentState) -> str:
    """Build the recommendation prompt for the LLM."""
    score = state["score_result"]
    triggered = (
        ", ".join(score["triggered_rules"]) if score["triggered_rules"] else "none"
    )
    feature_snapshot = state["features"] if state["features"] else "N/A"

    prompt_template = _get_loaded_prompt()
    return prompt_template.format(
        risk_score=f"{score['risk_score']:.4f}",
        risk_level=score["risk_level"],
        triggered_rules=triggered,
        fraud_probability=f"{score['fraud_probability']:.4f}",
        previous_status="N/A",  # todo read from redis
        feature_snapshot=feature_snapshot,
    )


# ---------------------------------------------------------------------------
# Lazy graph / llm singletons
# ---------------------------------------------------------------------------


def _get_graph():
    """Build and compile the LangGraph workflow lazily."""
    global _graph

    if _graph is not None:
        return _graph

    builder: StateGraph = StateGraph(cast(Any, _AssessmentState))
    builder.add_node("extract_features", _extract_features_node)
    builder.add_node("evaluate_rules", _evaluate_rules_node)
    builder.add_node("predict", _predict_node)
    builder.add_node("compute_score", _compute_score_node)
    builder.add_node("llm_judge", _llm_judge_node)
    builder.add_node("execute_action", _action_node)

    builder.set_entry_point("extract_features")
    builder.add_edge("extract_features", "evaluate_rules")
    builder.add_edge("extract_features", "predict")
    builder.add_edge("predict", "compute_score")
    builder.add_edge("compute_score", "llm_judge")
    builder.add_edge("llm_judge", "execute_action")
    builder.add_edge("execute_action", END)

    _graph = builder.compile()
    return _graph


def _get_llm() -> ChatOpenAI:
    """Return a module-level ChatOpenAI singleton (created on first use)."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    return _llm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@_trace(name="assess_risk")
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

    _safe_update_trace(
        {
            "transaction_id": txn_id,
            "service": "ai-secure-agent",
            "workflow": "risk_assessment_graph",
            "prompt_name": PROMPT_NAME,
            "prompt_version": PROMPT_VERSION,
            "llm_model": LLM_MODEL_NAME,
        }
    )

    initial_state: _AssessmentState = {
        "transaction": transaction,
        "features": {},
        "rule_result": {},
        "ml_result": {},
        "score_result": {},
        "recommendation": "",
    }

    final_state = cast(_AssessmentState, _get_graph().invoke(initial_state))
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
