import sqlite3
from pathlib import Path


def connect(db_path: Path):
    """
    Opens (or creates) the database file.
    """
    conn = sqlite3.connect(str(db_path))
    return conn


def init_db(conn):
    """
    Creates the photos table if it doesn't exist.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            ext TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            mtime REAL NOT NULL,
            width INTEGER,
            height INTEGER
        );
    """)
    conn.commit()

def save_photo(conn, data):
    """
    Saves one photo into the database.
    """
    conn.execute("""
        INSERT OR REPLACE INTO photos
        (path, filename, ext, size_bytes, mtime, width, height)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, data)
    
def init_ai_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_tags (
            path TEXT PRIMARY KEY,
            caption TEXT,
            objects TEXT
        );
    """)
    conn.commit()
    
def init_ai_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_tags (
            path TEXT PRIMARY KEY,
            caption TEXT,
            objects TEXT
        );
    """)
    conn.commit()
    
def init_categories_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            name TEXT PRIMARY KEY
        );
    """)
    conn.commit()

