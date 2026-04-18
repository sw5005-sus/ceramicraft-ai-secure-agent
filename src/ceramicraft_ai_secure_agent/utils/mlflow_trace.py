import os
from typing import Any, Callable, TypeVar, cast

import mlflow

from ceramicraft_ai_secure_agent.utils.logger import get_logger

F = TypeVar("F", bound=Callable[..., Any])

logger = get_logger(__name__)

MLFLOW_EXPERIMENT_NAME = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME",
    "ai-secure-agent-llm-traces",
)
PROMPT_NAME = os.environ.get(
    "FRAUD_RECOMMENDATION_PROMPT_NAME",
    "fraud_recommendation_prompt",
)
PROMPT_VERSION = os.environ.get(
    "FRAUD_RECOMMENDATION_PROMPT_VERSION",
    "v4",
)
LLM_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")

_tracing_initialized = False


def _is_tracing_enabled() -> bool:
    return os.environ.get("ENABLE_MLFLOW_TRACING", "false").lower() == "true"


def init_tracing_context() -> None:
    global _tracing_initialized

    if _tracing_initialized or not _is_tracing_enabled() or mlflow is None:
        return

    try:
        mlflow.set_tracking_uri(
            os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        _tracing_initialized = True
    except Exception as e:
        logger.error("Failed to initialize lightweight tracing: %s", e)


def trace(name: str):
    """Lazy MLflow trace decorator.

    When tracing is disabled, this becomes a no-op decorator so module import
    stays side-effect free in tests and CI.
    """

    def decorator(func: F) -> F:
        if not _is_tracing_enabled() or mlflow is None:
            return func

        init_tracing_context()
        return cast(F, mlflow.trace(name=name)(func))

    return decorator


def safe_update_trace(metadata: dict[str, Any]) -> None:
    """Update current trace metadata only when MLflow tracing is enabled."""
    if not _is_tracing_enabled():
        return

    try:
        mlflow.update_current_trace(metadata=metadata)
    except Exception as exc:
        logger.warning("Trace metadata update skipped: %s", exc)
