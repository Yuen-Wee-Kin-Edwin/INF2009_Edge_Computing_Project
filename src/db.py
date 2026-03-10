# File: src/db.py
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "lab_monitor.db"


def init_db():
    """Initialize a basic SQLite database file."""
    conn = sqlite3.connect(DB_PATH)
    # Tables will be added later.
    conn.commit()
    conn.close()


class Database:
    """Class to manage SQLite database connections, initialisation, and queries."""

    def __init__(self, db_name: str = "lab_monitor.db"):
        # Resolve the absolute path to ensure reliability regardless of where the script is run from
        self.db_path = Path(__file__).parent / db_name

    def connect(self):
        """Open a connection to the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        # Configure row_factory to return rows as dictionaries instead of plain tuples
        # This makes it much easier to convert the output directly to JSON for your Flask API
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """
        Initialise the database file and construct the required tables.
        Uses 'IF NOT EXISTS' to safely execute on every system boot.
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        camera_id TEXT NOT NULL,
                        location TEXT NOT NULL,
                        lab_id TEXT NOT NULL,
                        detection_timestamp TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        filename TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.commit()
                print(f"[SYSTEM] Database schema initialised successfully.")
        except sqlite3.Error as e:
            print(f"[DB ERROR] Critical failure initialising database: {e}")

    def insert_snapshot(
        self,
        camera_id: str,
        location: str,
        lab_id: str,
        timestamp: str,
        confidence: float,
        filename: str,
    ):
        """
        Logs a new detection event and its associated evidence filename into the database.
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO snapshots (camera_id, location, lab_id, detection_timestamp, confidence, filename)
                    VALUES (?, ?, ?, ? ,? , ?)
                """,
                    (camera_id, location, lab_id, timestamp, confidence, filename),
                )
                conn.commit()
        except sqlite3.Error as e:
            print(f"[DB ERROR] Failed to insert snapshot record: {e}")

    def get_recent_events(self, limit: int = 50):
        """
        Retrieves the most recent detection events to populate the Flask dashboard.
        """
        try:
            with self.connect() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, camera_id, location, lab_id, detection_timestamp, confidence, filename, created_at
                    FROM snapshots
                    ORDER BY id DESC
                    LIMIT ?
                """,
                    (limit,),
                )
                # Convert the sqlite3.Row objects into standard Python dictionaries
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"[DB ERROR] Failed to fetch recent events: {e}")
            return []

    def close(self):
        """Close the SQLite database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
