from ceramicraft_ai_secure_agent.data.risk_user_review import RiskUserReview
from ceramicraft_ai_secure_agent.mysqlcli import get_connection
from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)


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
            rules,
            ml_top_contributor
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                review.ml_top_contributor,
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
