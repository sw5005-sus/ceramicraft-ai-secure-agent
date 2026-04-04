from ceramicraft_ai_secure_agent.rediscli import get_redis_client
from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)


def set_user_last_status(user_id: int, status: str) -> None:
    """Set the last status for a user ID."""
    try:
        get_redis_client().set(f"user:{user_id}:ls", status)
        logger.info(f"Set last status for user {user_id} to {status}")
    except Exception as e:
        logger.error(f"Failed to set last status for user {user_id}: {e}")


def get_user_last_status(user_id: int) -> str:
    """Get the last status for a user ID."""
    try:
        status = get_redis_client().get(f"user:{user_id}:ls")
        if status is not None:
            status = status.decode("utf-8")
        logger.info(f"Got last status for user {user_id}: {status}")
        return status
    except Exception as e:
        logger.error(f"Failed to get last status for user {user_id}: {e}")
        return None
