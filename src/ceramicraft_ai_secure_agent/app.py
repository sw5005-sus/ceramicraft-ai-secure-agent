"""FastAPI application entry point for the AI Secure Agent."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ceramicraft_ai_secure_agent.config.config import get_config
from ceramicraft_ai_secure_agent.kafka.consumer import consume
from ceramicraft_ai_secure_agent.routers.demo_api import router as demo_router
from ceramicraft_ai_secure_agent.service.feature_service import (
    UserRequest,
    validate_and_update_feature_with_request,
)
from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security-related HTTP response headers to every response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'"
        )
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=()"
        )
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        return response


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    task = None
    try:
        task = asyncio.create_task(consume())
        logger.info("start to create kafka consumer task")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise e
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            logger.info("kafka consumer task cancelled successfully")


PREFIX = "/ai-secure-agent-ms/v1"

app = FastAPI(
    title="AI Secure Agent - Fraud Risk API",
    description=(
        "Real-time fraud risk assessment combining rule-based heuristics "
        "and a machine-learning model to score financial transactions."
    ),
    docs_url=f"{PREFIX}/docs",
    redoc_url=f"{PREFIX}/redoc",
    openapi_url=f"{PREFIX}/openapi.json",
    version="1.0.0",
    lifespan=lifespan,
)
FastAPIInstrumentor.instrument_app(app)
app.add_middleware(SecurityHeadersMiddleware)
app.include_router(demo_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return a generic 422 response to avoid leaking validation details."""
    return JSONResponse(status_code=422, content={"detail": "Invalid request"})


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catch-all handler that prevents stack-trace leakage in 500 responses."""
    logger.error("Unhandled exception: %s", exc, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/ai-secure-agent-ms/v1/ping", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Return a simple health-check response."""
    return {"message": "pong"}


@app.get("/risk/verify")
async def verify_user(
    request: Request,
    x_user_id: Optional[str] = Header(None, alias="X-Original-User-ID"),
    x_real_ip: Optional[str] = Header(None, alias="X-Real-IP"),
):
    user_id = x_user_id
    if user_id is None or user_id == 0:
        return

    client_ip = (
        x_real_ip or (request.client.host if request.client else None) or "127.0.0.1"
    )
    original_url = request.headers.get("X-Forwarded-Uri") or str(request.url)
    method = request.headers.get("X-Forwarded-Method") or request.method

    logger.info(
        f"User: {user_id} | IP: {client_ip} | Method: {method} | URL: {original_url}"
    )

    userRequest = UserRequest(
        user_id=int(user_id), ip=client_ip, uri=original_url, method=method
    )
    if not validate_and_update_feature_with_request(user_request=userRequest):
        raise HTTPException(status_code=403, detail="User is blacklisted")
    return


if __name__ == "__main__":
    import uvicorn

    host = get_config().http.host
    port = get_config().http.port
    logger.info("Starting AI Secure Agent API server on %s:%s ...", host, port)

    uvicorn.run(
        "ceramicraft_ai_secure_agent.app:app", host=host, port=port, reload=False
    )
