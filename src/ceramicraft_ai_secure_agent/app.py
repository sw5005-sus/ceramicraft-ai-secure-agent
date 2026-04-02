"""FastAPI application entry point for the AI Secure Agent."""

from __future__ import annotations
from typing import Optional

from fastapi import FastAPI, Request, Header, HTTPException
import contextlib
import asyncio

from ceramicraft_ai_secure_agent.api.risk_api import router as risk_router
from ceramicraft_ai_secure_agent.utils.logger import get_logger
from ceramicraft_ai_secure_agent.kafka.consumer import create_consumer
from ceramicraft_ai_secure_agent.config.config import load_config
from ceramicraft_ai_secure_agent.service.feature_service import (
    validate_and_update_feature_with_request,
    UserRequest,
)


logger = get_logger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_config("config/config.yaml")
        logger.info("config loaded successfully")
        # import for initialization
        import ceramicraft_ai_secure_agent.redis

        logger.info("redis client initialized successfully")

        task = asyncio.create_task(create_consumer())
        logger.info("start to create kafka consumer task")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise e
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        logger.info("kafka consumer task cancelled successfully")


app = FastAPI(
    title="AI Secure Agent – Fraud Risk API",
    description=(
        "Real-time fraud risk assessment combining rule-based heuristics "
        "and a machine-learning model to score financial transactions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(risk_router)


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Return a simple health-check response."""
    return {"status": "ok"}


@app.get("/risk/verify")
async def verify_user(
    request: Request,
    x_user_id: Optional[str] = Header(None, alias="X-Original-User-ID"),
    x_real_ip: Optional[str] = Header(None, alias="X-Real-IP"),
):
    user_id = x_user_id
    if user_id is None:
        return

    client_ip = x_real_ip or request.client.host
    original_url = request.headers.get("X-Forwarded-Uri") or str(request.url)
    method = request.headers.get("X-Forwarded-Method") or request.method

    logger.info(
        f"User: {user_id} | IP: {client_ip} | Method: {method} | URL: {original_url}"
    )

    if not user_id:
        return

    userRequest = UserRequest(
        user_id=int(user_id), ip=client_ip, uri=original_url, method=method
    )
    if not validate_and_update_feature_with_request(userRequest=userRequest):
        raise HTTPException(status_code=403, detail="User is blacklisted")
    return


if __name__ == "__main__":
    import os

    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    logger.info("Starting AI Secure Agent API server on %s:%s ...", host, port)

    uvicorn.run(
        "ceramicraft_ai_secure_agent.app:app", host=host, port=port, reload=False
    )
