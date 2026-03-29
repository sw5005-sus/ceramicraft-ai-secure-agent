# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --locked --no-dev --no-editable \
 && rm -rf /root/.cache /tmp/* \
 && find /app/.venv -type d -name "__pycache__" -prune -exec rm -rf {} + \
 && find /app/.venv -type f -name "*.pyc" -delete \
 && find /app/.venv -type f -name "*.pyo" -delete

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src"

WORKDIR /app

RUN groupadd --system --gid 10001 appuser \
 && useradd --system --uid 10001 --gid 10001 --create-home appuser

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src
COPY --from=builder --chown=appuser:appuser /app/README.md /app/README.md
COPY --from=builder --chown=appuser:appuser /app/pyproject.toml /app/pyproject.toml

USER appuser

EXPOSE 8080

CMD ["uvicorn", "ceramicraft_ai_secure_agent.app:app", "--host", "0.0.0.0", "--port", "8080"]