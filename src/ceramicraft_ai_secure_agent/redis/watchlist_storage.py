from ceramicraft_ai_secure_agent.redis import redis_client
from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)


def add_watechlist(user_id: int) -> None:
    """Add a user ID to the watchlist."""
    try:
        redis_client.sadd("watchlist", user_id)
        logger.info(f"Added user {user_id} to watchlist")
    except Exception as e:
        logger.error(f"Failed to add user {user_id} to watchlist: {e}")


def is_watchlisted(user_id: int) -> bool:
    """Check if a user ID is in the watchlist."""
    try:
        return redis_client.sismember("watchlist", user_id)
    except Exception as e:
        logger.error(f"Failed to check if user {user_id} is watchlist: {e}")
        return False
