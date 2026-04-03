from ceramicraft_ai_secure_agent.rediscli import get_redis_client

from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)


def add_blacklist(user_id: int) -> None:
    """Add a user ID to the blacklist."""
    try:
        get_redis_client().sadd("blacklist", user_id)
        logger.info(f"Added user {user_id} to blacklist")
    except Exception as e:
        print(f"Failed to add user {user_id} to blacklist: {e}")


def is_blacklisted(user_id: int) -> bool:
    """Check if a user ID is in the blacklist."""
    try:
        return get_redis_client().sismember("blacklist", user_id)
    except Exception as e:
        print(f"Failed to check if user {user_id} is blacklisted: {e}")
        return False
