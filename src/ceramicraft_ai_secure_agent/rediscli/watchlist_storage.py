from datetime import datetime, timedelta
from ceramicraft_ai_secure_agent.rediscli import get_redis_client
from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)


def add_watechlist(user_id: int) -> None:
    """Add a user ID to the watchlist."""
    try:
        expire_time = datetime.now() + timedelta(days=1)
        get_redis_client().zadd("watchlist", {user_id: expire_time.timestamp()})
        logger.info(f"Added user {user_id} to watchlist")
    except Exception as e:
        logger.error(f"Failed to add user {user_id} to watchlist: {e}")


def is_watchlisted(user_id: int) -> bool:
    """Check if a user ID is in the watchlist."""
    try:
        expireTime = get_redis_client().zscore("watchlist", user_id)
        if expireTime is None:
            return False
        if expireTime < datetime.now().timestamp():
            get_redis_client().zrem("watchlist", user_id)
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to check if user {user_id} is watchlist: {e}")
        return False
