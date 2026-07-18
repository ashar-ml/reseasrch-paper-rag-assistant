import sqlite3
from pathlib import Path
from loguru import logger
from app.core.config import settings

DB_PATH = settings.vector_db_path.parent / "users.db"

def init_db():
    """Initializes the users database table."""
    logger.info(f"Initializing SQLite user database at {DB_PATH}...")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            citations TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users (username) ON DELETE CASCADE
        )
    """)
    conn.commit()
    conn.close()
    logger.info("User database initialized successfully.")

def get_db_connection():
    return sqlite3.connect(str(DB_PATH))

# Initialize database table on import
init_db()
