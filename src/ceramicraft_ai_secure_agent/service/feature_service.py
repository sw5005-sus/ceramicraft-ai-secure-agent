"""Feature extraction service.

Converts a raw transaction dict into a numerical feature vector
that can be consumed by the rule engine and the ML model.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from langchain_core.tools import tool

from ceramicraft_ai_secure_agent.rediscli import (
    blacklist_storage,
    order_storage,
    user_last_status_storage,
    user_storage,
)
from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)
feature_executor = ThreadPoolExecutor(max_workers=100)


class UserRequest:
    def __init__(self, user_id: int, ip: str, uri: str, method: str) -> None:
        self.user_id = user_id
        self.ip = ip
        self.uri = uri
        self.method = method


def validate_and_update_feature_with_request(user_request: UserRequest) -> bool:
    """Validate and update user features based on the incoming request."""
    valid: bool = True
    if not _is_customer_reuqest(user_request=user_request):
        return valid
    try:
        if blacklist_storage.is_blacklisted(user_id=user_request.user_id):
            valid = not _is_block_request(user_request=user_request)
            logger.warning(
                f"User {user_request.user_id} is blacklisted. Block: {not valid}"
            )
        user_storage.update_user_ip(
            user_id=user_request.user_id, ip_address=user_request.ip
        )
        logger.info(
            "Updated user %s IP with %s from request.",
            user_request.user_id,
            user_request.ip,
        )
    except Exception as e:
        logger.error(
            f"Failed to chack and update user {user_request.user_id} IP in Redis: {e}"
        )
    return valid


def _is_customer_reuqest(user_request: UserRequest) -> bool:
    return "/customer/" in user_request.uri


def _is_block_request(user_request: UserRequest) -> bool:
    return user_request.method.lower() in (
        "post",
        "put",
        "patch",
        "delete",
    )


def extract_features(user_id: int) -> dict[str, int | float | str]:
    """Extract a flat feature dictionary from a raw transaction.

    Args:
        user_id: The user ID for which to extract features.

    Returns:
        Dictionary mapping feature names to numeric values.
    """
    last_hour = int(datetime.now().timestamp() - 3600)
    last_day = int(datetime.now().timestamp() - 24 * 3600)
    now = int(datetime.now().timestamp())

    with feature_executor as executor:
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

    logger.debug(f"Extracted features for user {user_id}: {results}")
    return results


def _get_avg_order_amount(user_id: int) -> float:
    return order_storage.get_global_avg_order_amount(user_id=user_id) / 100


def _get_avg_order_amount_today(user_id: int) -> float:
    return order_storage.get_today_order_avg_amount(user_id=user_id) / 100


@tool
def extract_features_tool(user_id: int) -> dict:
    """Extract a flat feature dictionary from a raw transaction payload.

    Args:
        user_id: The user ID for which to extract features.

    Returns:
        Dictionary mapping feature names to numeric values ready for the
        rule engine and ML model.
    """
    return extract_features(user_id=user_id)
