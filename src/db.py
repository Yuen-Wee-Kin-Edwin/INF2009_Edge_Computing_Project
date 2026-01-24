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
    """Class to manage SQLite database connections and initialization."""

    def __init__(self, db_name: str = "lab_monitor.db"):
        """
        Initialize the Database object.

        Args:
            db_name (str): Name of the SQLite database file.
        """
        self.db_path = Path(__file__).parent / db_name
        self.conn = None

    def connect(self):
        """Open a connection to the SQLite database."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        return self.conn

    def close(self):
        """Close the SQLite database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def init_db(self):
        """
        Initialize the database file.
        Placeholder for table creation � tables can be added later.
        """
        conn = self.connect()
        # Future table creation logic goes here.
        conn.commit()
        self.close()
