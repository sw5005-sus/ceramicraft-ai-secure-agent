import time

from opentelemetry import metrics

meter = metrics.get_meter("ceramicraft_ai_secure_agent.agent_service")


def metric_timed(name: str):
    counter = meter.create_counter(f"{name}_total")
    duration = meter.create_histogram(f"{name}_duration_ms", unit="ms")

    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            counter.add(1)
            try:
                return func(*args, **kwargs)
            finally:
                duration.record((time.perf_counter() - start) * 1000)

        return wrapper

    return decorator
