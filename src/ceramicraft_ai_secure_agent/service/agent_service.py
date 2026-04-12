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
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

from langchain_openai import ChatOpenAI

from ceramicraft_ai_secure_agent.data.feature_columns import FEATURE_COLUMNS
from ceramicraft_ai_secure_agent.data.state import (
    AssessmentState as _AssessmentState,
)
from ceramicraft_ai_secure_agent.data.state import (
    Recommendation,
    build_risk_user_review_from_state,
    direct_block_recommendation,
    fallback_return,
    no_risk_recommendation,
)
from ceramicraft_ai_secure_agent.mysqlcli.risk_user_review_storage import (
    create_risk_user_review,
)
from ceramicraft_ai_secure_agent.rediscli import (
    blacklist_storage,
    user_last_status_storage,
    watchlist_storage,
    whitelist_storage,
)
from ceramicraft_ai_secure_agent.service import risk_scoring
from ceramicraft_ai_secure_agent.service.feature_service import extract_features_tool
from ceramicraft_ai_secure_agent.service.ml_model import predict_tool
from ceramicraft_ai_secure_agent.service.policy_engine import (
    need_llm_judgment,
    should_block_directly,
)
from ceramicraft_ai_secure_agent.service.rule_engine import evaluate_rules_tool
from ceramicraft_ai_secure_agent.utils.logger import get_logger
from ceramicraft_ai_secure_agent.utils.mlflow_trace import (
    LLM_MODEL_NAME,
    PROMPT_NAME,
    PROMPT_VERSION,
    safe_update_trace,
    trace,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy globals
# ---------------------------------------------------------------------------

_loaded_prompt: str | None = None
_prompt_lock = threading.Lock()
_graph: Any | None = None
_graph_lock = threading.Lock()
_llm: Any | None = None
_llm_lock = threading.Lock()

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# MLflow helpers (lazy / optional)
# ---------------------------------------------------------------------------


def _get_loaded_prompt() -> str:
    """Return the loaded prompt template lazily.

    If MLflow is disabled, use a local fallback template string so tests can run
    without external dependencies.
    """
    global _loaded_prompt
    if _loaded_prompt is not None:
        return _loaded_prompt

    with _prompt_lock:
        if _loaded_prompt is None:
            try:
                base_dir = Path(__file__).resolve().parent.parent
                file_name = f"{PROMPT_NAME}_{PROMPT_VERSION}.txt"
                prompt_path = base_dir / "prompts" / file_name

                if not prompt_path.exists():
                    logger.warning(
                        f"Prompt file not found at {prompt_path}. Using fallback."
                    )
                    raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
                else:
                    with open(prompt_path, "r", encoding="utf-8") as f:
                        _loaded_prompt = f.read().strip()

                logger.info("Prompt loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load prompt: {e}")
                return "Emergency fallback prompt template..."

    return _loaded_prompt


# ---------------------------------------------------------------------------
# LangGraph node functions
# ---------------------------------------------------------------------------


@trace(name="extract_features_node")
def _extract_features_node(state: _AssessmentState) -> dict[str, Any]:
    """Node: extract features from the raw input."""
    tool_input = cast(Any, {"user_id": state["user_id"]})
    res = extract_features_tool.invoke(tool_input)
    return {"features": res}


@trace(name="evaluate_rules_node")
def _evaluate_rules_node(state: _AssessmentState) -> dict[str, Any]:
    """Node: apply business rules to the feature set."""
    res = evaluate_rules_tool.invoke({"features": state["features"]})
    return {"rule_result": res}


@trace(name="predict_node")
def _predict_node(state: _AssessmentState) -> dict[str, Any]:
    """Node: run the ML model on the feature set."""
    res = predict_tool.invoke({"features": state["features"]})
    return {"ml_result": res}


@trace(name="compute_score_node")
def _compute_score_node(state: _AssessmentState) -> dict[str, Any]:
    """Node: combine rule and ML signals into a composite risk score."""
    score_result = risk_scoring.compute_score(state["rule_result"], state["ml_result"])
    return {"score_result": score_result}


@trace(name="llm_judge_node")
def _llm_judge_node(state: _AssessmentState) -> dict[str, Any]:
    """Node: use an OpenAI LLM to produce a risk judgment and recommendation.

    Falls back to a rule-based recommendation when ``OPENAI_API_KEY`` is not
    set so that the service degrades gracefully in environments without LLM
    access.
    """
    if should_block_directly(state):
        logger.info(
            "High-risk signals detected. Blocking user %s directly.", state["user_id"]
        )
        return {"recommendation": direct_block_recommendation}

    if not need_llm_judgment(state):
        logger.info(
            'No significant risk signals. Using "no risk" recommendation for user %s.',
            state["user_id"],
        )
        return {"recommendation": no_risk_recommendation.to_json()}

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning(
            "OPENAI_API_KEY not set – using rule-based recommendation fallback."
        )
        return {"recommendation": fallback_return.to_json()}

    _update_trace_with_score(state)
    prompt = _build_llm_prompt(state)
    llm = _get_llm()
    try:
        response = llm.invoke(prompt)
        return {"recommendation": str(response.content)}
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}. Using fallback recommendation.")
        return {"recommendation": fallback_return.to_json()}


class Action:
    """Base class for actions."""

    def run(self, state: _AssessmentState) -> None:
        raise NotImplementedError("Subclasses should implement this method.")


class ManualReviewAction(Action):
    def run(self, state: _AssessmentState) -> None:
        create_risk_user_review(build_risk_user_review_from_state(state=state))


class BlockAction(Action):
    def run(self, state: _AssessmentState) -> None:
        user_id = state["user_id"]
        blacklist_storage.add_blacklist(user_id=user_id)
        create_risk_user_review(build_risk_user_review_from_state(state=state))


class WatchlistAction(Action):
    def run(self, state: _AssessmentState) -> None:
        user_id = state["user_id"]
        watchlist_storage.add_watchlist(user_id=user_id)
        create_risk_user_review(build_risk_user_review_from_state(state=state))


class AllowAction(Action):
    def run(self, state: _AssessmentState) -> None:
        return


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
    action = action_map.get(action_key, ManualReviewAction())
    user_id = state["user_id"]
    user_last_status_storage.set_user_last_status(user_id, action_key)
    action.run(state)
    return {}


def _update_trace_with_score(state: _AssessmentState) -> None:
    """Attach score metadata to the current trace if tracing is enabled."""
    score = state["score_result"]

    trace_metadata: dict[str, Any] = {
        "user_id": str(state["user_id"]),
        "risk_level": score["risk_level"],
        "risk_score": score["risk_score"],
        "fraud_probability": score["fraud_probability"],
        "triggered_rules_count": len(score["triggered_rules"]),
        "prompt_name": PROMPT_NAME,
        "prompt_version": PROMPT_VERSION,
        "llm_model": LLM_MODEL_NAME,
        "fallback_used": not bool(os.environ.get("OPENAI_API_KEY", "")),
    }
    safe_update_trace(trace_metadata)


def _build_llm_prompt(state: _AssessmentState) -> str:
    """Build the recommendation prompt for the LLM."""
    score = state["score_result"]
    triggered = (
        ", ".join(score["triggered_rules"]) if score["triggered_rules"] else "none"
    )
    filtered_features = {
        key: state["features"][key]
        for key in FEATURE_COLUMNS
        if key in state["features"]
    }

    prompt_template = _get_loaded_prompt()
    return prompt_template.format(
        risk_score=f"{score['risk_score']:.4f}",
        risk_level=score["risk_level"],
        triggered_rules=triggered,
        fraud_probability=f"{score['fraud_probability']:.4f}",
        previous_status=state["features"].get("last_status", "N/A"),
        feature_snapshot=filtered_features,
    )


# ---------------------------------------------------------------------------
# Lazy graph / llm singletons
# ---------------------------------------------------------------------------


def _get_graph():
    """Build and compile the LangGraph workflow lazily."""
    global _graph
    if _graph is not None:
        return _graph
    with _graph_lock:
        if _graph is not None:
            return _graph
        from langgraph.graph import END, StateGraph

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
        builder.add_edge("evaluate_rules", "compute_score")
        builder.add_edge("predict", "compute_score")
        builder.add_edge("compute_score", "llm_judge")
        builder.add_edge("llm_judge", "execute_action")
        builder.add_edge("execute_action", END)

        _graph = builder.compile()
    return _graph


def _get_llm() -> ChatOpenAI:
    """Return a module-level ChatOpenAI singleton (created on first use)."""
    global _llm
    if _llm is not None:
        return _llm

    with _llm_lock:
        if _llm is None:
            _llm = ChatOpenAI(
                model_name="gpt-4o-mini", temperature=0, request_timeout=5
            )
        return _llm


def _do_skip(user_id: int) -> bool:
    if whitelist_storage.is_whitelisted(
        user_id=user_id
    ) or blacklist_storage.is_blacklisted(user_id=user_id):
        logger.info(
            f"User {user_id} is in whitelist or blacklist. Skipping risk assessment."
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@trace(name="assess_risk")
def assess_risk(user_id: int) -> dict[str, Any]:
    """Run the complete risk-assessment pipeline for a transaction.

    The pipeline is implemented as a LangGraph StateGraph:
      1. ``extract_features`` - convert raw fields to numeric features.
      2. ``evaluate_rules``   - apply hard-coded business rules.
      3. ``predict``          - run the ML fraud-detection model.
      4. ``compute_score``    - combine signals into a composite score.
      5. ``llm_judge``        - use an LLM to generate a recommendation.

    Args:
        transaction: Raw transaction payload.

    Returns:
        Final risk assessment dict containing:
          - ``user_id`` (int)
          - ``risk_score`` (float)
          - ``risk_level`` (str)
          - ``triggered_rules`` (list[str])
          - ``fraud_probability`` (float)
          - ``recommendation`` (str)
    """
    logger.info("Starting risk assessment for user %s", user_id)
    if _do_skip(user_id=user_id):
        return {}

    safe_update_trace(
        {
            "user_id": str(user_id),
            "service": "ai-secure-agent",
            "workflow": "risk_assessment_graph",
            "prompt_name": PROMPT_NAME,
            "prompt_version": PROMPT_VERSION,
            "llm_model": LLM_MODEL_NAME,
        }
    )

    initial_state: _AssessmentState = {
        "user_id": user_id,
        "features": {},
        "rule_result": {},
        "ml_result": {},
        "score_result": {},
        "recommendation": "",
    }

    final_state = cast(_AssessmentState, _get_graph().invoke(initial_state))
    score = final_state["score_result"]

    assessment: dict[str, Any] = {
        "user_id": user_id,
        "risk_score": score["risk_score"],
        "risk_level": score["risk_level"],
        "triggered_rules": score["triggered_rules"],
        "fraud_probability": score["fraud_probability"],
        "recommendation": final_state["recommendation"],
    }
    logger.info(
        "Risk assessment complete for User %s: level=%s score=%.4f",
        user_id,
        score["risk_level"],
        score["risk_score"],
    )

    return assessment
