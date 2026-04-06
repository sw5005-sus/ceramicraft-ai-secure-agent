from datetime import datetime, timedelta
from typing import cast

from ceramicraft_ai_secure_agent.rediscli import get_redis_client
from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)


def add_blacklist(user_id: int) -> None:
    """Add a user ID to the blacklist."""
    try:
        expire_time = datetime.now() + timedelta(
            days=1
        )  # Set expiration time for blacklist entry
        get_redis_client().zadd("blacklist", {str(user_id): expire_time.timestamp()})
        logger.info(f"Added user {user_id} to blacklist")
    except Exception as e:
        print(f"Failed to add user {user_id} to blacklist: {e}")


def is_blacklisted(user_id: int) -> bool:
    """Check if a user ID is in the blacklist."""
    try:
        expireTime = get_redis_client().zscore("blacklist", user_id)
        if expireTime is None:
            return False
        if float(cast(float, expireTime)) < datetime.now().timestamp():
            get_redis_client().zrem("blacklist", user_id)
            return False
        return True
    except Exception as e:
        print(f"Failed to check if user {user_id} is blacklisted: {e}")
        return False
