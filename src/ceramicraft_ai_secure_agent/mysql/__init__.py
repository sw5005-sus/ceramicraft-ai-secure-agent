import mysql.connector
from mysql.connector import Error
from ceramicraft_ai_secure_agent.config.config import Config
from ceramicraft_ai_secure_agent.utils.logger import get_logger
import os

logger = get_logger(__name__)


def init_mysql_connection():
    try:
        connection = mysql.connector.connect(
            host=Config.mysql.host,
            port=Config.mysql.port,
            user=Config.mysql.username,
            password=os.getenv("MYSQL_PASSWORD"),
            database=Config.mysql.database,
        )
        if connection.is_connected():
            logger.info("MySQL connection established successfully!")
        return connection
    except Error as e:
        logger.error(f"Error while connecting to MySQL: {e}")
        return None
