import os
from typing import Any, Callable, cast

from ceramicraft_ai_secure_agent.utils.logger import get_logger

try:
    import mlflow
except Exception:  # pragma: no cover - degrade gracefully when MLflow is absent
    mlflow = None  # type: ignore[assignment]

logger = get_logger(__name__)

_MLFLOW_INITIALIZED = False

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


def trace(
    *args: Any, **kwargs: Any
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Return mlflow.trace decorator if available, else no-op."""

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if mlflow is None or not hasattr(mlflow, "trace"):
            return func
        return cast(Callable[..., Any], mlflow.trace(*args, **kwargs)(func))

    return _decorator


def safe_update_current_trace(metadata: dict[str, Any] | None = None) -> None:
    """Best-effort update of current MLflow trace metadata."""
    if mlflow is None or not hasattr(mlflow, "update_current_trace"):
        return

    try:
        mlflow.update_current_trace(metadata=metadata or {})
    except Exception as exc:  # pragma: no cover - tracing must not break business flow
        logger.warning("Failed to update MLflow trace metadata: %s", exc)


def init_mlflow_tracing() -> None:
    """Initialize MLflow tracing once, if MLflow is configured."""
    global _MLFLOW_INITIALIZED

    if _MLFLOW_INITIALIZED or mlflow is None:
        return

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        logger.info("MLFLOW_TRACKING_URI not set; MLflow tracing disabled.")
        _MLFLOW_INITIALIZED = True
        return

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        # LangGraph tracing is supported via LangChain autologging.
        # If this MLflow version does not support it, keep service functional.
        if hasattr(mlflow, "langchain") and hasattr(mlflow.langchain, "autolog"):
            try:
                mlflow.langchain.autolog(log_traces=True)
            except TypeError:
                # Backward compatibility for older signatures
                mlflow.langchain.autolog()

        logger.info(
            "MLflow tracing initialized. tracking_uri=%s experiment=%s",
            tracking_uri,
            MLFLOW_EXPERIMENT_NAME,
        )
    except Exception as exc:  # pragma: no cover - tracing must not break business flow
        logger.warning("Failed to initialize MLflow tracing: %s", exc)

    _MLFLOW_INITIALIZED = True
