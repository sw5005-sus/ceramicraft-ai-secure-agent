from ceramicraft_ai_secure_agent.redis import redis_client


def add_blacklist(user_id: int) -> None:
    """Add a user ID to the blacklist."""
    try:
        redis_client.sadd("blacklist", user_id)
    except Exception as e:
        print(f"Failed to add user {user_id} to blacklist: {e}")


def is_blacklisted(user_id: int) -> bool:
    """Check if a user ID is in the blacklist."""
    try:
        return redis_client.sismember("blacklist", user_id)
    except Exception as e:
        print(f"Failed to check if user {user_id} is blacklisted: {e}")
        return False
