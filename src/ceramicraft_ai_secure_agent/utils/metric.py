import functools
import time

from opentelemetry import metrics

meter = metrics.get_meter("ceramicraft_ai_secure_agent.agent_service")

risk_decision_counter = meter.create_counter(
    "agent_risk_decisions_total",
    description="Total risk decisions by action and risk level",
)

risk_score_histogram = meter.create_histogram(
    "agent_risk_score",
    description="Risk score distribution",
)

fraud_probability_histogram = meter.create_histogram(
    "agent_fraud_probability",
    description="Fraud probability distribution",
)


def record_risk_decision(
    action: str,
    risk_level: str,
    model_version: str,
    prompt_version: str,
) -> None:
    risk_decision_counter.add(
        1,
        {
            "action": action,
            "risk_level": risk_level,
            "model_version": model_version,
            "prompt_version": prompt_version,
        },
    )


def record_risk_score(
    risk_level: str,
    risk_score: float,
    fraud_probability: float,
) -> None:
    attrs = {"risk_level": risk_level}

    risk_score_histogram.record(risk_score, attrs)
    fraud_probability_histogram.record(fraud_probability, attrs)


def metric_timed(name: str):
    counter = meter.create_counter(f"{name}_total")
    duration = meter.create_histogram(f"{name}_duration_ms", unit="ms")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            counter.add(1)
            try:
                return func(*args, **kwargs)
            finally:
                duration.record((time.perf_counter() - start) * 1000)

        return wrapper

    return decorator
