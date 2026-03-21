"""FastAPI application entry point for the AI Secure Agent."""

from __future__ import annotations

from fastapi import FastAPI

from api.risk_api import router as risk_router
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="AI Secure Agent – Fraud Risk API",
    description=(
        "Real-time fraud risk assessment combining rule-based heuristics "
        "and a machine-learning model to score financial transactions."
    ),
    version="1.0.0",
)

app.include_router(risk_router)


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Return a simple health-check response."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting AI Secure Agent API server …")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
