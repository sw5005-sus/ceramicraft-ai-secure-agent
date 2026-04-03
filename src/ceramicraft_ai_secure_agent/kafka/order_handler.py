from ceramicraft_ai_secure_agent.data.event_data import OrderMessage
from ceramicraft_ai_secure_agent.utils.logger import get_logger
import ceramicraft_ai_secure_agent.rediscli.order_storage as order_storage


logger = get_logger(__name__)


def handle(msg: bytes) -> None:
    try:
        order_message = OrderMessage.from_json(msg)
        order_storage.create_order(order_message)
        order_storage.update_today_order_amount(order_message)
        order_storage.update_total_order_amount(order_message)
        logger.info(
            f"Order created message handled: user_id={order_message.user_id}, order_id={order_message.order_id}"
        )
    except Exception as e:
        logger.error(f"handler order_created_event failed. {e}")
