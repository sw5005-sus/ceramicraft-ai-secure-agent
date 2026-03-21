"""FastAPI router for the /risk/check endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from service.agent_service import assess_risk
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/risk", tags=["Risk"])


class TransactionRequest(BaseModel):
    """Schema for an incoming transaction risk-check request."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="Identifier of the user initiating the transaction")
    amount: float = Field(..., gt=0, description="Transaction amount (must be positive)")
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    merchant: str = Field(..., description="Merchant name")
    merchant_category: str = Field(default="unknown", description="Merchant category code or label")
    country: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    ip_address: str = Field(default="", description="Client IP address")
    device_id: str = Field(default="", description="Client device identifier")
    timestamp: str = Field(default="", description="Transaction timestamp (ISO 8601)")


class RiskCheckResponse(BaseModel):
    """Schema for the /risk/check response."""

    transaction_id: str
    risk_score: float
    risk_level: str
    triggered_rules: list[str]
    fraud_probability: float
    recommendation: str


@router.post("/check", response_model=RiskCheckResponse, summary="Check transaction risk")
async def check_risk(request: TransactionRequest) -> Any:
    """Assess the fraud risk of an incoming transaction.

    Returns a risk score, risk level (HIGH / MEDIUM / LOW), the list of
    triggered business rules, the raw ML fraud probability, and a
    recommended action.
    """
    logger.info("Received risk check request for transaction: %s", request.transaction_id)
    try:
        result = assess_risk(request.model_dump())
    except Exception as exc:
        logger.error("Risk assessment failed for %s: %s", request.transaction_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Risk assessment failed.") from exc

    return result
