from ceramicraft_ai_secure_agent.utils.logger import get_logger
from ceramicraft_ai_secure_agent.data import UserActivatedEvent

logger = get_logger(__name__)


def handle(msg: bytes) -> None:
    """Handle a user activated event message."""
    try:
        event = UserActivatedEvent.from_json(msg.decode("utf-8"))
        logger.info(
            f"User Activated: user_id={event.user_id}, activated_time={event.activated_time}"
        )
    except Exception as e:
        logger.error(f"Failed to handle user activated event: {e}")
