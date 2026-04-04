from ceramicraft_ai_secure_agent.data.event_data import OrderMessage
from ceramicraft_ai_secure_agent.utils.logger import get_logger
import ceramicraft_ai_secure_agent.rediscli.order_storage as order_storage
import ceramicraft_ai_secure_agent.service.agent_service as agent_service


logger = get_logger(__name__)


def handle(msg: bytes) -> None:
    try:
        order_message = OrderMessage.from_json(msg)
        if order_storage.exist_user_order(order_message):
            return
        order_storage.create_order(order_message)
        order_storage.update_receiver_address(order_message)
        order_storage.update_today_order_amount(order_message)
        order_storage.update_total_order_amount(order_message)
        logger.info(
            f"Order created message handled: user_id={order_message.user_id}, order_id={order_message.order_id}"
        )
        access_ret = agent_service.assess_risk(order_message.user_id)
        logger.info(
            f"Risk assessment completed for user {order_message.user_id}. Access result: {access_ret}"
        )
    except Exception as e:
        logger.error(f"handler order_created_event failed. {e}")
