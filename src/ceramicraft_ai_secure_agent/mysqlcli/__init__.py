from mysql.connector import pooling, Error
from ceramicraft_ai_secure_agent.config.config import system_config
from ceramicraft_ai_secure_agent.utils.logger import get_logger
import os

logger = get_logger(__name__)


def init_mysql_connection_pool():
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
        return pool
    except Error as e:
        logger.error(f"Error while connecting to MySQL: {e}")
        raise e


connection_pool = init_mysql_connection_pool()


def get_connection():
    try:
        if connection_pool:
            return connection_pool.get_connection()
        else:
            logger.error("Connection pool is not initialized.")
            return None
    except Error as e:
        logger.error(f"Error while getting connection from pool: {e}")
        return None
