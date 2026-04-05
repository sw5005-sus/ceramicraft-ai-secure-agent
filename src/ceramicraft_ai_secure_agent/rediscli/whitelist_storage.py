from datetime import datetime, timedelta
from ceramicraft_ai_secure_agent.rediscli import get_redis_client
from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)


def add_whitelist(user_id: int) -> None:
    """Add a user ID to the whitelist."""
    try:
        expire_time = datetime.now() + timedelta(days=1)
        get_redis_client().zadd("whitelist", {user_id: expire_time.timestamp()})
        logger.info(f"Added user {user_id} to whitelist")
    except Exception as e:
        print(f"Failed to add user {user_id} to whitelist: {e}")


def is_whitelisted(user_id: int) -> bool:
    """Check if a user ID is in the whitelist."""
    try:
        expireTime = get_redis_client().zscore("whitelist", user_id)
        if expireTime is None:
            return False
        if expireTime < datetime.now().timestamp():
            get_redis_client().zrem("whitelist", user_id)
            return False
        return True
    except Exception as e:
        print(f"Failed to check if user {user_id} is whitelist: {e}")
        return False
