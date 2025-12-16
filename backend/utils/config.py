# backend/utils/config.py
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "root"),
    "database": os.getenv("DB_NAME", "data_analyst_db"),
}

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 200))