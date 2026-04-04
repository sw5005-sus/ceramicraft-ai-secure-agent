from ceramicraft_ai_secure_agent.rediscli import get_redis_client
from ceramicraft_ai_secure_agent.utils.logger import get_logger
from ceramicraft_ai_secure_agent.data.event_data import UserActivatedEvent
from datetime import datetime


logger = get_logger(__name__)


def update_user_register_time(useractivatedEvent: UserActivatedEvent) -> None:
    """Store the user registration time in Redis."""
    user_id, activated_time = (
        useractivatedEvent.user_id,
        useractivatedEvent.activated_time,
    )
    try:
        get_redis_client().set(f"u:{user_id}:rt", activated_time)
        logger.info(
            f"User {user_id} registration time {activated_time} stored in Redis."
        )
    except Exception as e:
        logger.error(
            f"Failed to store registration time for user {user_id} in Redis: {e}"
        )


def get_user_register_time(user_id: int) -> int:
    """Get the user registration time from Redis."""
    try:
        activated_time = get_redis_client().get(f"u:{user_id}:rt")
        if activated_time is not None:
            activated_time = int(activated_time)
            logger.info(
                f"User {user_id} registration time retrieved from Redis: {activated_time}"
            )
            return activated_time
        else:
            logger.warning(f"No registration time found for user {user_id} in Redis.")
            return 0
    except Exception as e:
        logger.error(
            f"Failed to retrieve registration time for user {user_id} from Redis: {e}"
        )
        return -1


def update_user_ip(user_id: int, ip_address: str) -> None:
    """Store the user IP address in Redis."""
    latest_retain_time = int(datetime.now().timestamp()) - 24 * 3600
    try:
        redis_client = get_redis_client()
        pipeline = redis_client.pipeline()
        pipeline.zadd(f"u:{user_id}:ip", {ip_address: int(datetime.now().timestamp())})
        pipeline.zremrangebyscore(f"u:{user_id}:ip", 0, latest_retain_time)
        result = pipeline.execute()
        logger.info(
            f"Stored IP address {ip_address} for user {user_id} in Redis. result: {result}"
        )
    except Exception as e:
        logger.error(f"Failed to store IP address for user {user_id} in Redis: {e}")


def count_user_ip(user_id: int) -> int:
    """Count the number of unique IP addresses used by the user in the last 24 hours."""
    latest_retain_time = int(datetime.now().timestamp()) - 24 * 3600
    try:
        count = get_redis_client().zcount(
            f"u:{user_id}:ip", latest_retain_time, int(datetime.now().timestamp())
        )
        logger.info(
            f"User {user_id} has used {count} unique IP addresses in the last 24 hours."
        )
        return count
    except Exception as e:
        logger.error(f"Failed to count IP addresses for user {user_id} in Redis: {e}")
        return 0
