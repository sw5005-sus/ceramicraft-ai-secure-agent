import redis
from ceramicraft_ai_secure_agent.config.config import Config


def init_redis_client() -> redis.Redis:
    """Initialize and return a Redis client."""
    return redis.Redis(host=Config.redis.host, port=Config.redis.port, db=0)


redis_client = init_redis_client()
