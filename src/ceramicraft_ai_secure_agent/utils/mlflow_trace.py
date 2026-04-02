import os
from typing import Any

from ceramicraft_ai_secure_agent.utils.logger import get_logger

try:
    import mlflow
except Exception:  # pragma: no cover - degrade gracefully when MLflow is absent
    mlflow = None  # type: ignore[assignment]

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
    "v1",
)
LLM_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini")


def mlflow_enabled() -> bool:
    """Whether MLflow tracing/integration is enabled.

    Default is disabled so tests/CI do not hang on import or require external
    MLflow services unless explicitly opted in.
    """
    return os.environ.get("ENABLE_MLFLOW_TRACING", "false").lower() == "true"


def _safe_update_current_trace(metadata: dict[str, Any] | None = None) -> None:
    """Best-effort update of current MLflow trace metadata."""
    if mlflow is None or not hasattr(mlflow, "update_current_trace"):
        return

    try:
        mlflow.update_current_trace(metadata=metadata or {})
    except Exception as exc:  # pragma: no cover - tracing must not break business flow
        logger.warning("Failed to update MLflow trace metadata: %s", exc)


def init_mlflow_once() -> None:
    """Initialise MLflow lazily and only once."""
    global _mlflow_initialized

    if _mlflow_initialized or not mlflow_enabled():
        return

    import mlflow

    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.langchain.autolog()
    _mlflow_initialized = True


def trace(name: str):
    """Lazy MLflow trace decorator.

    When tracing is disabled, this becomes a no-op decorator so module import
    stays side-effect free in tests and CI.
    """

    def decorator(func: F) -> F:
        if not mlflow_enabled():
            return func

        import mlflow

        return mlflow.trace(name=name)(func)  # type: ignore[return-value]

    return decorator


def safe_update_trace(metadata: dict[str, Any]) -> None:
    """Update current trace metadata only when MLflow tracing is enabled."""
    if not mlflow_enabled():
        return

    init_mlflow_once()
    _safe_update_current_trace(metadata=metadata)
