from ceramicraft_ai_secure_agent.data.event_data import UserActivatedEvent
from ceramicraft_ai_secure_agent.rediscli import user_storage
from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)


def handle(msg: str) -> None:
    """Handle a user activated event message."""
    try:
        event = UserActivatedEvent.from_json(msg)
        logger.info(
            f"User Activated: user_id={event.user_id}, "
            f"activated_time={event.activated_time}"
        )
        registered_time = user_storage.get_user_register_time(event.user_id)
        if registered_time > 0:
            return
        user_storage.update_user_register_time(event)
    except Exception as e:
        logger.error(f"Failed to handle user activated event: {e}")
