import redis

from ceramicraft_ai_secure_agent.config.config import system_config

_redis_pool = None


def get_redis_client() -> redis.Redis:
    """Get a Redis client instance using a connection pool."""
    global _redis_pool

    if _redis_pool is None:
        _redis_pool = redis.ConnectionPool(
            host=system_config.redis.host,
            port=system_config.redis.port,
            db=0,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=2.0,
            max_connections=20,
        )

    return redis.Redis(connection_pool=_redis_pool)
