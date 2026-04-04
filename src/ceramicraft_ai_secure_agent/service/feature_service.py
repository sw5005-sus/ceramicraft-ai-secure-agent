"""Feature extraction service.

Converts a raw transaction dict into a numerical feature vector
that can be consumed by the rule engine and the ML model.
"""

from __future__ import annotations

from typing import Any
from datetime import datetime
import math


from langchain_core.tools import tool
from concurrent.futures import ThreadPoolExecutor

from ceramicraft_ai_secure_agent.utils.logger import get_logger
from ceramicraft_ai_secure_agent.rediscli import (
    user_storage,
    order_storage,
    blacklist_storage,
    user_last_status_storage,
)

logger = get_logger(__name__)

# Countries considered high-risk for fraud purposes
HIGH_RISK_COUNTRIES = {"NG", "RU", "KP", "IR", "SY"}

# Merchant categories considered high-risk
HIGH_RISK_CATEGORIES = {"unknown", "financial", "gambling"}


class UserRequest:
    def __init__(self, user_id: int, ip: str, uri: str, method: str) -> None:
        self.user_id = user_id
        self.ip = ip
        self.uri = uri
        self.method = method


def validate_and_update_feature_with_request(userRequest: UserRequest) -> bool:
    """Validate and update user features based on the incoming request."""
    try:
        if blacklist_storage.is_blacklisted(user_id=userRequest.user_id):
            logger.warning(
                f"User {userRequest.user_id} is blacklisted. Skipping feature update."
            )
            return False
        user_storage.update_user_ip(
            user_id=userRequest.user_id, ip_address=userRequest.ip
        )
        logger.info(
            f"Updated user {userRequest.user_id} IP with {userRequest.ip} from request."
        )
    except Exception as e:
        logger.error(f"Failed to update user {userRequest.user_id} IP in Redis: {e}")
    return True


def extract_features(user_id: int) -> dict[str, float]:
    """Extract a flat feature dictionary from a raw transaction.

    Args:
        transaction: Raw transaction payload (dict).

    Returns:
        Dictionary mapping feature names to numeric values.
    """
    last_hour = datetime.now().timestamp() - 3600
    last_day = datetime.now().timestamp() - 24 * 3600
    now = datetime.now().timestamp()

    with ThreadPoolExecutor() as executor:
        features = {
            "order_count_last_1h": executor.submit(
                order_storage.count_order_by_time,
                user_id=user_id,
                start_time=last_hour,
                end_time=now,
            ),
            "order_count_last_24h": executor.submit(
                order_storage.count_order_by_time,
                user_id=user_id,
                start_time=last_day,
                end_time=now,
            ),
            "unique_ip_count": executor.submit(
                user_storage.count_user_ip, user_id=user_id
            ),
            "avg_order_amount_global": executor.submit(
                _get_avg_order_amount, user_id=user_id
            ),
            "avg_order_amount_today": executor.submit(
                _get_avg_order_amount_today, user_id=user_id
            ),
            "account_age_days": executor.submit(
                lambda: math.ceil(
                    (
                        datetime.now().timestamp()
                        - user_storage.get_user_register_time(user_id=user_id)
                    )
                    / (24 * 3600)
                )
            ),
            "receive_address_count": executor.submit(
                order_storage.count_user_receiver_address, user_id=user_id
            ),
            "last_status": executor.submit(
                user_last_status_storage.get_user_last_status, user_id=user_id
            ),
        }

        results = {k: v.result() for k, v in features.items()}

    logger.debug(f"Extracted features for user {user_id}: {features}")
    return results


def _get_avg_order_amount(user_id: int) -> float:
    return order_storage.get_global_avg_order_amount(user_id=user_id) / 100


def _get_avg_order_amount_today(user_id: int) -> float:
    return order_storage.get_today_order_avg_amount(user_id=user_id) / 100


@tool
def extract_features_tool(user_id: int) -> dict:
    """Extract a flat feature dictionary from a raw transaction payload.

    Args:
        transaction: Raw transaction payload dict containing fields such as
            ``amount``, ``country``, and ``merchant_category``.

    Returns:
        Dictionary mapping feature names to numeric values ready for the
        rule engine and ML model.
    """
    return extract_features(user_id=user_id)
