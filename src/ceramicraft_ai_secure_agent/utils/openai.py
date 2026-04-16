import time
from typing import Any

from langchain_openai import ChatOpenAI

from ceramicraft_ai_secure_agent.utils.metric import meter
from ceramicraft_ai_secure_agent.utils.mlflow_trace import safe_update_trace

llm_requests_total = meter.create_counter(
    "agent_llm_requests_total",
    description="Total number of LLM requests",
)

llm_errors_total = meter.create_counter(
    "agent_llm_errors_total",
    description="Total number of failed LLM requests",
)

llm_duration_ms = meter.create_histogram(
    "agent_llm_duration_ms",
    unit="ms",
    description="LLM request duration in milliseconds",
)

llm_input_tokens_total = meter.create_counter(
    "agent_llm_input_tokens_total",
    description="Total input tokens sent to the LLM",
)

llm_output_tokens_total = meter.create_counter(
    "agent_llm_output_tokens_total",
    description="Total output tokens returned by the LLM",
)

llm_total_tokens_total = meter.create_counter(
    "agent_llm_total_tokens_total",
    description="Total tokens used by the LLM",
)


def extract_token_usage(response: Any) -> tuple[int, int, int]:
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        input_tokens = int(usage_metadata.get("input_tokens", 0) or 0)
        output_tokens = int(usage_metadata.get("output_tokens", 0) or 0)
        total_tokens = int(usage_metadata.get("total_tokens", 0) or 0)

    if total_tokens == 0:
        response_metadata = getattr(response, "response_metadata", None)
        if isinstance(response_metadata, dict):
            token_usage = response_metadata.get("token_usage", {})
            if isinstance(token_usage, dict):
                input_tokens = int(token_usage.get("prompt_tokens", input_tokens) or 0)
                output_tokens = int(
                    token_usage.get("completion_tokens", output_tokens) or 0
                )
                total_tokens = int(token_usage.get("total_tokens", 0) or 0)

    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens

    return input_tokens, output_tokens, total_tokens


def invoke_llm_with_metrics(llm: ChatOpenAI, prompt: str, attrs: dict[str, str]) -> Any:
    start = time.perf_counter()
    llm_requests_total.add(1, attrs)

    try:
        response = llm.invoke(prompt)
        duration_ms = (time.perf_counter() - start) * 1000
        llm_duration_ms.record(duration_ms, attrs)

        input_tokens, output_tokens, total_tokens = extract_token_usage(response)

        if input_tokens > 0:
            llm_input_tokens_total.add(input_tokens, attrs)
        if output_tokens > 0:
            llm_output_tokens_total.add(output_tokens, attrs)
        if total_tokens > 0:
            llm_total_tokens_total.add(total_tokens, attrs)
        safe_update_trace(
            {
                "llm_duration_ms": round(duration_ms, 2),
                "llm_input_tokens": input_tokens,
                "llm_output_tokens": output_tokens,
                "llm_total_tokens": total_tokens,
            }
        )
        return response
    except Exception:
        llm_errors_total.add(1, attrs)
        llm_duration_ms.record((time.perf_counter() - start) * 1000, attrs)
        raise
