from ceramicraft_ai_secure_agent.redis import redis_client
from ceramicraft_ai_secure_agent.utils.logger import get_logger
from ceramicraft_ai_secure_agent.data import OrderMessage
from datetime import datetime


logger = get_logger(__name__)


def create_order(orderMsg: OrderMessage) -> None:
    """Store the order ID in Redis."""
    user_id, order_id, timestamp = (
        orderMsg.user_id,
        orderMsg.order_id,
        orderMsg.timestamp,
    )
    earlest_retain_time = int(datetime.now().timestamp()) - 24 * 3600
    try:
        redis_client.zadd(f"u:{user_id}:o", order_id, timestamp)
        logger.info(f"User {user_id} Order {order_id} stored in Redis.")
        redis_client.zremrangebyscore(f"u:{user_id}:o", 0, earlest_retain_time)
        logger.info(
            f"Removed orders for user {user_id} older than {earlest_retain_time}."
        )
    except Exception as e:
        logger.error(f"Failed to store order {order_id} in Redis: {e}")


def count_order_by_time(user_id: int, start_time: int, end_time: int) -> int:
    """Count the number of orders for a user within a time range."""
    try:
        count = redis_client.zcount(f"u:{user_id}:o", start_time, end_time)
        logger.info(
            f"User {user_id} has {count} orders between {start_time} and {end_time}."
        )
        return count
    except Exception as e:
        logger.error(f"Failed to count orders for user {user_id} in Redis: {e}")
        return 0


def update_total_order_amount(orderMsg: OrderMessage) -> None:
    """Update the total order amount for a user."""
    user_id, order_amount = orderMsg.user_id, orderMsg.total_amount
    try:
        redis_client.hincrbyfloat(f"u:{user_id}:toa", order_amount)
        redis_client.hincrby(f"u:{user_id}:toc", 1)
        logger.info(f"Updated total order amount for user {user_id} by {order_amount}.")
    except Exception as e:
        logger.error(
            f"Failed to update total order amount for user {user_id} in Redis: {e}"
        )


def get_global_avg_order_amount(user_id: int) -> float:
    """Get the global average order amount for a user."""
    try:
        total_amount = redis_client.hget(f"u:{user_id}:toa", 0.0)
        total_count = redis_client.hget(f"u:{user_id}:toc", 0)
        if total_count == 0:
            return 0.0
        avg_amount = float(total_amount) / int(total_count)
        logger.info(f"Global average order amount for user {user_id} is {avg_amount}.")
        return avg_amount
    except Exception as e:
        logger.error(
            f"Failed to get global average order amount for user {user_id} from Redis: {e}"
        )
        return 0.0


def update_today_order_amount(orderMsg: OrderMessage) -> None:
    """Update the total order amount for a user today."""
    user_id, order_amount = orderMsg.user_id, orderMsg.total_amount
    today = datetime.now().strftime("%Y%m%d")
    expireAt = datetime.now().replace(hour=23, minute=59, second=59)
    try:
        redis_client.hincrbyfloat(f"u:{user_id}:toa:{today}", order_amount)
        redis_client.expireat(f"u:{user_id}:toa:{today}", expireAt)
        redis_client.hincrby(f"u:{user_id}:toc:{today}", 1)
        redis_client.expireat(f"u:{user_id}:toc:{today}", expireAt)
        logger.info(
            f"Updated today's order amount for user {user_id} by {order_amount}."
        )
    except Exception as e:
        logger.error(
            f"Failed to update today's order amount for user {user_id} in Redis: {e}"
        )


def get_today_order_avg_amount(user_id: int) -> float:
    """Get today's average order amount for a user."""
    today = datetime.now().strftime("%Y%m%d")
    try:
        total_amount = redis_client.hget(f"u:{user_id}:toa:{today}", 0.0)
        total_count = redis_client.hget(f"u:{user_id}:toc:{today}", 0)
        if total_count == 0:
            return 0.0
        avg_amount = float(total_amount) / int(total_count)
        logger.info(f"Today's average order amount for user {user_id} is {avg_amount}.")
        return avg_amount
    except Exception as e:
        logger.error(
            f"Failed to get today's average order amount for user {user_id} from Redis: {e}"
        )
        return 0.0
