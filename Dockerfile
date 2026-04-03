FROM python:3.12-alpine AS builder

RUN apk add --no-cache \
    gcc \
    musl-dev \
    python3-dev \
    libffi-dev \
    zlib-dev \
    curl
    
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
    
COPY pyproject.toml uv.lock README.md ./
    
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev
    
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev && \
    find .venv -type d -name "__pycache__" -exec rm -rf {} +
    
FROM python:3.12-alpine
    
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"
    
WORKDIR /app
    
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
    
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appgroup /app/src /app/src
    
USER appuser
EXPOSE 8000
CMD ["python", "-m", "ceramicraft_ai_secure_agent.app"]