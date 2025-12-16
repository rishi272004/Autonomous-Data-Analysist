import mysql.connector
from backend.utils.config import DB_CONFIG
from backend.utils.logger import logger

def get_connection():
    try:
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"]
        )
        logger.info("Connected to MySQL successfully.")
        return conn
    except Exception as e:
        logger.error(f"MySQL connection failed: {e}")
        raise
