from ceramicraft_ai_secure_agent.mysqlcli import get_connection
from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)


class RiskUserReview:
    def __init__(
        self, user_id: int, create_time: int, confidence: str, anlyst_summary: str
    ):
        self.user_id = user_id
        self.create_time = create_time
        self.confidence = confidence
        self.anlyst_summary = anlyst_summary


def create_risk_user_review(review: RiskUserReview) -> None:
    """Insert a new risk user review into the database."""
    try:
        sql = """
        INSERT IGNORE INTO risk_user_review (user_id, create_time, confidence, anlyst_summary)
        VALUES (%s, %s, %s, %s)
        """
        get_connection().cursor().execute(
            sql,
            (
                review.user_id,
                review.create_time,
                review.confidence,
                review.anlyst_summary,
            ),
        )
        logger.info(f"Risk user review created for user {review.user_id}")
    except Exception as e:
        logger.error(
            f"Failed to create risk user review for user {review.user_id}: {e}"
        )
