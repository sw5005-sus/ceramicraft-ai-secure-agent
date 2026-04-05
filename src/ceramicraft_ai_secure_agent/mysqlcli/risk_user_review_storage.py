from ceramicraft_ai_secure_agent.mysqlcli import get_connection
from ceramicraft_ai_secure_agent.utils.logger import get_logger
from ceramicraft_ai_secure_agent.data.const import RiskUserReviewDecision

logger = get_logger(__name__)


class RiskUserReview:
    def __init__(
        self,
        user_id: int,
        create_time: int,
        confidence: str,
        analyst_summary: str,
        decision: int,
        decision_source: str = "",
        risk_score: float = 0.0,
        risk_level: str = "",
        rule_score: float = 0.0,
        fraud_probability: float = 0.0,
        triggered_rules: list[str] = [],
    ):
        self.user_id = user_id
        self.create_time = create_time
        self.confidence = confidence
        self.analyst_summary = analyst_summary
        self.decision_source = decision_source
        self.decision = decision
        self.risk_score = risk_score
        self.risk_level = risk_level
        self.rule_score = rule_score
        self.fraud_probability = fraud_probability
        self.rules = ",".join(triggered_rules)


def create_risk_user_review(review: RiskUserReview) -> None:
    """Insert a new risk user review into the database."""
    try:
        sql = """
        REPLACE INTO risk_user_reviews (
            user_id, 
            create_time, 
            confidence, 
            analyst_summary, 
            decision, 
            decision_source, 
            risk_score, 
            risk_level, 
            rule_score, 
            fraud_probability, 
            rules
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with get_connection() as conn:
            params = (
                review.user_id,
                review.create_time,
                review.confidence,
                review.analyst_summary,
                review.decision,
                review.decision_source,
                review.risk_score,
                review.risk_level,
                review.rule_score,
                review.fraud_probability,
                review.rules,
            )
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                affected_rows = cursor.rowcount
                logger.info(f"SQL affected rows: {affected_rows}")
            conn.commit()
        logger.info(f"Risk user review created for user {review.user_id}")
    except Exception as e:
        logger.error(
            f"Failed to create risk user review for user {review.user_id}: {e}"
        )
