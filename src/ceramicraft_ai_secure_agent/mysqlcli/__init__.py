from mysql.connector import pooling, Error
from ceramicraft_ai_secure_agent.config.config import system_config
from ceramicraft_ai_secure_agent.utils.logger import get_logger
import os
import threading

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
                pool_size=system_config.mysql.max_connect_size,
                host=system_config.mysql.host,
                port=system_config.mysql.port,
                user=system_config.mysql.username,
                password=os.getenv("MYSQL_PASSWORD"),
                database=system_config.mysql.database,
            )
            logger.info("MySQL connection pool initialized successfully!")
            _connetion_pool = pool
            return _connetion_pool
        except Error as e:
            logger.error(f"Error while connecting to MySQL: {e}")
            raise e


def get_connection():
    return get_mysql_connection_pool().get_connection()
