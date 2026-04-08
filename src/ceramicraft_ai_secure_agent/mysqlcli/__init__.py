import os
import threading

from mysql.connector import Error, pooling

from ceramicraft_ai_secure_agent.config.config import get_config
from ceramicraft_ai_secure_agent.utils.logger import get_logger

logger = get_logger(__name__)

_connetion_pool = None
_conn_lock = threading.Lock()


def get_mysql_connection_pool():
    global _connetion_pool
    if _connetion_pool is not None:
        return _connetion_pool
    with _conn_lock:
        if _connetion_pool is not None:
            return _connetion_pool
        try:
            pool = pooling.MySQLConnectionPool(
                pool_name="mypool",
                pool_size=get_config().mysql.max_connect_size,
                host=get_config().mysql.host,
                port=get_config().mysql.port,
                user=get_config().mysql.username,
                password=os.getenv("MYSQL_PASSWORD"),
                database=get_config().mysql.database,
            )
            logger.info("MySQL connection pool initialized successfully!")
            _connetion_pool = pool
            return _connetion_pool
        except Error as e:
            logger.error(f"Error while connecting to MySQL: {e}")
            raise e


def get_connection():
    return get_mysql_connection_pool().get_connection()
