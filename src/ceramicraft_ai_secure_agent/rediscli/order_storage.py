from datetime import datetime
from typing import Any, List, cast

from ceramicraft_ai_secure_agent.data.event_data import OrderMessage
from ceramicraft_ai_secure_agent.rediscli import get_redis_client
from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)


def exist_user_order(orderMsg: OrderMessage) -> bool:
    """Check if the order ID already exists in Redis."""
    user_id, order_id = orderMsg.user_id, orderMsg.order_id
    try:
        exists = get_redis_client().zscore(f"u:{user_id}:o", order_id) is not None
        logger.info(f"Order {order_id} for user {user_id} exists in Redis: {exists}.")
        return exists
    except Exception as e:
        logger.error(f"Failed to check existence of order {order_id} in Redis: {e}")
        return False


def create_order(orderMsg: OrderMessage) -> None:
    """Store the order ID in Redis."""
    user_id, order_id, timestamp = (
        orderMsg.user_id,
        orderMsg.order_id,
        int(datetime.now().timestamp()),
    )
    earlest_retain_time = int(datetime.now().timestamp()) - 24 * 3600
    try:
        redis_client = get_redis_client()
        pipeline = redis_client.pipeline(transaction=True)
        pipeline.zadd(f"u:{user_id}:o", {str(order_id): timestamp})
        pipeline.zremrangebyscore(f"u:{user_id}:o", 0, earlest_retain_time)
        result = pipeline.execute()
        logger.info(
            f"Stored order {order_id} for user {user_id} "
            f"in Redis with timestamp {timestamp}. result: {result}"
        )
    except Exception as e:
        logger.error(f"Failed to store order {order_id} in Redis: {e}")


def count_order_by_time(user_id: int, start_time: int, end_time: int) -> int:
    """Count the number of orders for a user within a time range."""
    try:
        count = get_redis_client().zcount(f"u:{user_id}:o", start_time, end_time)
        logger.info(
            f"User {user_id} has {count} orders between {start_time} and {end_time}."
        )
        return int(cast(int, count))
    except Exception as e:
        logger.error(
            "Failed to count orders for user %s in Redis: %s",
            user_id,
            e,
        )
        return 0


def update_total_order_amount(orderMsg: OrderMessage) -> None:
    """Update the total order amount for a user."""
    user_id, order_amount = orderMsg.user_id, orderMsg.total_amount
    key = f"u:{user_id}:stat"
    try:
        redis_client = get_redis_client()
        pipeline = redis_client.pipeline(transaction=True)
        redis_client.hincrbyfloat(key, "amount", order_amount)
        redis_client.hincrby(key, "count", 1)
        pipeline.execute()
        logger.info(f"Updated total order amount for user {user_id} by {order_amount}.")
    except Exception as e:
        logger.error(
            f"Failed to update total order amount for user {user_id} in Redis: {e}"
        )


def get_global_avg_order_amount(user_id: int) -> float:
    """Get the global average order amount for a user."""
    key = f"u:{user_id}:stat"
    try:
        redis_client = get_redis_client()
        res = cast(List[Any], redis_client.hmget(key, ["amount", "count"]))
        total_amount = res[0] if res[0] is not None else 0.0
        total_count = res[1] if res[1] is not None else 0
        if total_count == 0:
            return 0.0
        avg_amount = float(total_amount) / int(total_count)
        logger.info(f"Global average order amount for user {user_id} is {avg_amount}.")
        return avg_amount
    except Exception as e:
        logger.error(
            "Failed to get global average order amount for user %s from Redis: %s",
            user_id,
            e,
        )
        return 0.0


def update_today_order_amount(orderMsg: OrderMessage) -> None:
    """Update the total order amount for a user today."""
    user_id, order_amount = orderMsg.user_id, orderMsg.total_amount
    today = datetime.now().strftime("%Y%m%d")
    expireAt = datetime.now().replace(hour=23, minute=59, second=59)
    key = f"u:{user_id}:stat:{today}"
    try:
        redis_client = get_redis_client()
        pipe = redis_client.pipeline()
        pipe.hincrbyfloat(key, "amount", order_amount)
        pipe.hincrby(key, "count", 1)
        pipe.expireat(key, expireAt)
        result = pipe.execute()
        logger.info(
            "Updated today's order amount for user %s by %s. result: %s",
            user_id,
            order_amount,
            result,
        )
    except Exception as e:
        logger.error(
            "Failed to update today's order amount for user %s in Redis: %s",
            user_id,
            e,
        )


def get_today_order_avg_amount(user_id: int) -> float:
    """Get today's average order amount for a user."""
    today = datetime.now().strftime("%Y%m%d")
    key = f"u:{user_id}:stat:{today}"
    try:
        redis_client = get_redis_client()
        res = cast(List[Any], redis_client.hmget(key, ["amount", "count"]))
        total_amount = res[0] if res[0] is not None else 0.0
        total_count = res[1] if res[1] is not None else 0
        if total_count == 0:
            return 0.0
        avg_amount = float(total_amount) / int(total_count)
        logger.info(f"Today's average order amount for user {user_id} is {avg_amount}.")
        return avg_amount
    except Exception as e:
        logger.error(
            "Failed to get today's average order amount for user %s from Redis: %s",
            user_id,
            e,
        )
        return 0.0


def update_receiver_address(orderMsg: OrderMessage) -> None:
    """Update the receiver address for a user."""
    user_id, receiver_zip_code = orderMsg.user_id, orderMsg.receiver_zip_code
    try:
        redis_client = get_redis_client()
        redis_client.sadd(f"u:{user_id}:ra", receiver_zip_code)
        logger.info(f"Updated receiver address for user {user_id}.")
    except Exception as e:
        logger.error(
            "Failed to update receiver address for user %s in Redis: %s",
            user_id,
            e,
        )


def count_user_receiver_address(user_id: int) -> int:
    """Count the number of unique receiver addresses for a user."""
    try:
        count = get_redis_client().scard(f"u:{user_id}:ra")
        logger.info(f"User {user_id} has {count} unique receiver addresses.")
        return int(cast(int, count))
    except Exception as e:
        logger.error(
            "Failed to count unique receiver addresses for user %s in Redis: %s",
            user_id,
            e,
        )
        return 0
