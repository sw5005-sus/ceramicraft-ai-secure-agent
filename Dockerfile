FROM python:3.12-alpine AS builder

RUN apk add --no-cache \
    gcc \
    musl-dev \
    python3-dev \
    libffi-dev \
    zlib>=1.3.2-r0 \
    zlib-dev>=1.3.2-r0 \
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

RUN apk update && apk upgrade --no-cache && \
    apk add --no-cache zlib>=1.3.2-r0 && \
    rm -rf /var/cache/apk/*

WORKDIR /app
    
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
    
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appgroup /app/src /app/src
    
USER appuser
EXPOSE 8080
CMD ["python", "-m", "ceramicraft_ai_secure_agent.app"]