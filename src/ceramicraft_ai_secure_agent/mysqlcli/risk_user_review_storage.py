from ceramicraft_ai_secure_agent.mysqlcli import get_connection
from ceramicraft_ai_secure_agent.utils.logger import get_logger
from ceramicraft_ai_secure_agent.data.const import RiskUserReviewStatus

logger = get_logger(__name__)


class RiskUserReview:
    def __init__(
        self,
        user_id: int,
        create_time: int,
        confidence: str,
        analyst_summary: str,
        state: RiskUserReviewStatus,
        detail: str = "",
    ):
        self.user_id = user_id
        self.create_time = create_time
        self.confidence = confidence
        self.analyst_summary = analyst_summary
        self.state = state
        self.detail = detail


def create_risk_user_review(review: RiskUserReview) -> None:
    """Insert a new risk user review into the database."""
    try:
        sql = """
        INSERT IGNORE INTO risk_user_review (user_id, create_time, confidence, anlyst_summary, state)
        VALUES (%s, %s, %s, %s)
        """
        get_connection().cursor().execute(
            sql,
            (
                review.user_id,
                review.create_time,
                review.confidence,
                review.analyst_summary,
                review.state,
            ),
        )
        logger.info(f"Risk user review created for user {review.user_id}")
    except Exception as e:
        logger.error(
            f"Failed to create risk user review for user {review.user_id}: {e}"
        )
