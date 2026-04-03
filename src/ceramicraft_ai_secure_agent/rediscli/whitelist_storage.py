from ceramicraft_ai_secure_agent.rediscli import get_redis_client


def add_whitelist(user_id: int) -> None:
    """Add a user ID to the whitelist."""
    try:
        get_redis_client().sadd("whitelist", user_id)
    except Exception as e:
        print(f"Failed to add user {user_id} to whitelist: {e}")


def is_whitelisted(user_id: int) -> bool:
    """Check if a user ID is in the whitelist."""
    try:
        return get_redis_client().sismember("whitelist", user_id)
    except Exception as e:
        print(f"Failed to check if user {user_id} is whitelist: {e}")
        return False
