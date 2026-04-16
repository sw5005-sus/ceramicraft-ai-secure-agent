from enum import Enum
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(
    prefix="/demo",
    tags=["demo", "mock"],
)

DEMO_HTML_PATH = Path(__file__).parent / "demo_page.html"
DEMO_HTML = DEMO_HTML_PATH.read_text(encoding="utf-8")


class OpType(str, Enum):
    clear = "clear"
    normal = "normal"
    block = "block"
    manual_review = "manual_review"
    watchlist = "watchlist"


@router.post("/mock-data")
def gen_mock_data(user_id: int, op_type: OpType):
    """Generate mock data for testing."""
    import ceramicraft_ai_secure_agent.mock.gen_mock_data as gen_mock_data
    from ceramicraft_ai_secure_agent.service.feature_service import extract_features

    if op_type == OpType.clear:
        gen_mock_data.clear_mock_data(user_id=user_id)
    elif op_type == OpType.normal:
        gen_mock_data.gen_mock_normal(user_id=user_id)
    elif op_type == OpType.block:
        gen_mock_data.gen_mock_block(user_id=user_id)
    elif op_type == OpType.manual_review:
        gen_mock_data.gen_mock_manual_review(user_id=user_id)
    elif op_type == OpType.watchlist:
        gen_mock_data.gen_mock_watchlist(user_id=user_id)
    return extract_features(user_id=user_id)


@router.post("/risk-access")
def risk_access(user_id: int):
    """Assess risk for the specified user."""
    from ceramicraft_ai_secure_agent.service.agent_service import assess_risk

    return assess_risk(user_id=user_id)


@router.get("/page", response_class=HTMLResponse)
def demo_page():
    """Interactive HTML page for demo APIs."""
    return HTMLResponse(content=DEMO_HTML)
