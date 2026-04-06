import json
from datetime import datetime
from typing import Any

from typing_extensions import TypedDict

from ceramicraft_ai_secure_agent.data.const import RiskUserReviewDecision
from ceramicraft_ai_secure_agent.data.risk_user_review import RiskUserReview


class AssessmentState(TypedDict):
    """Mutable state passed between LangGraph nodes."""

    user_id: int
    features: dict[str, int | float]
    rule_result: dict[str, Any]
    ml_result: dict[str, Any]
    score_result: dict[str, Any]
    recommendation: str


def build_risk_user_review_from_state(state: AssessmentState) -> RiskUserReview:
    user_id = state["user_id"]
    recommendation = Recommendation.from_json(state["recommendation"])
    score = state["score_result"]
    return RiskUserReview(
        user_id=user_id,
        confidence=recommendation.confidence,
        create_time=int(datetime.now().timestamp()),
        analyst_summary=recommendation.analyst_summary,
        decision=RiskUserReviewDecision.from_str(recommendation.recommended_action),
        risk_score=score.get("risk_score", 0.0),
        risk_level=score.get("risk_level", ""),
        rule_score=state["rule_result"].get("rule_score", 0.0),
        fraud_probability=state["ml_result"].get("fraud_probability", 0.0),
        triggered_rules=score.get("triggered_rules", []),
        decision_source=(
            "LLM" if recommendation.reason != "LLM unavailable" else "SystemRule"
        ),
    )


class Recommendation:
    """Class to represent a recommendation from the deserialized content."""

    def __init__(
        self,
        recommended_action: str,
        reason: str,
        analyst_summary: str,
        confidence: str,
    ) -> None:
        self.recommended_action = recommended_action
        self.reason = reason
        self.analyst_summary = analyst_summary
        self.confidence = confidence

    @classmethod
    def from_json(cls, json_str: str) -> "Recommendation":
        """Deserialize from a JSON string."""
        if not json_str:
            return fallback_return
        try:
            data = json.loads(json_str)
            return cls(
                recommended_action=data["recommended_action"],
                reason=data["reason"],
                analyst_summary=data["analyst_summary"],
                confidence=data["confidence"],
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            return fallback_return

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.__dict__)


no_risk_recommendation = Recommendation(
    "allow",
    "Low risk score and no triggered rules",
    "Legitimate transaction: low risk, no rules triggered.",
    "high",
)

fallback_return = Recommendation(
    recommended_action="manual_review",
    reason="LLM unavailable",
    analyst_summary="LLM API key not set. Defaulting to manual review.",
    confidence="low",
)

analyst_msg = (
    "The transaction exhibits multiple high-risk indicators, "
    "including a high composite risk score and several triggered rules. "
    "Blocking is recommended based on these strong signals."
)

direct_block_recommendation = Recommendation(
    recommended_action="block",
    reason="High risk score and/or triggered rules",
    analyst_summary=analyst_msg,
    confidence="high",
).to_json()
