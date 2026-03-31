"""
database.py — Smart Count Tramway
===================================
SQLite schema design and all CRUD operations for passenger count storage.

Schema overview:
─────────────────────────────────────────────────────────────────
  TABLE stops
    id INTEGER PRIMARY KEY
    name TEXT UNIQUE          ← e.g. "Kharouba", "Salamandre", "Gare SNTF"

  TABLE counts
    id        INTEGER PRIMARY KEY AUTOINCREMENT
    stop_id   INTEGER  FK → stops.id
    timestamp TEXT     ISO-8601 with timezone (UTC)
    entries   INTEGER  passengers who boarded this recording window
    exits     INTEGER  passengers who alighted this recording window
    net       INTEGER  GENERATED: entries - exits
─────────────────────────────────────────────────────────────────

All inserts use parameterised queries (no f-string SQL) to prevent
SQL-injection — good practice even for a local SQLite file.

Author  : Smart Count Tramway Team
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CountRecord:
    """A single timestamped count record retrieved from the DB."""
    id        : int
    stop_name : str
    timestamp : str
    entries   : int
    exits     : int
    net       : int   # entries - exits


# ─────────────────────────────────────────────────────────────────────────────
# Database Manager
# ─────────────────────────────────────────────────────────────────────────────

class Database:
    """
    Manages SQLite connection, schema creation, and all data operations.

    Parameters
    ----------
    db_path : str | Path
        File path for the SQLite database.
        Defaults to 'data/tramway_counts.db'.
        Use ':memory:' for unit tests.
    """

    # Default tram stops on the Mostaganem line
    DEFAULT_STOPS = [
        "Kharouba",
        "Salamandre",
        "Gare SNTF",
        "Nouvelle Gare Routière",
        "Centre Ville",
    ]

    def __init__(self, db_path: str = "data/tramway_counts.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False allows the connection to be shared between
        # the main loop thread and the Streamlit dashboard thread.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row  # allows column access by name

        self._create_schema()
        self._seed_stops()
        logger.info(f"[Database] Connected to {self.db_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Schema
    # ─────────────────────────────────────────────────────────────────────────

    def _create_schema(self) -> None:
        """Create tables if they do not yet exist (idempotent)."""
        self._conn.executescript("""
            PRAGMA journal_mode = WAL;

            CREATE TABLE IF NOT EXISTS stops (
                id   INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT    NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS counts (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                stop_id   INTEGER NOT NULL REFERENCES stops(id),
                timestamp TEXT    NOT NULL,
                entries   INTEGER NOT NULL DEFAULT 0,
                exits     INTEGER NOT NULL DEFAULT 0,
                net       INTEGER GENERATED ALWAYS AS (entries - exits) STORED
            );

            CREATE INDEX IF NOT EXISTS idx_counts_stop
                ON counts(stop_id);

            CREATE INDEX IF NOT EXISTS idx_counts_timestamp
                ON counts(timestamp);
        """)
        self._conn.commit()

    def _seed_stops(self) -> None:
        """Insert default stops if they are not already in the database."""
        cursor = self._conn.cursor()
        for stop in self.DEFAULT_STOPS:
            cursor.execute(
                "INSERT OR IGNORE INTO stops (name) VALUES (?)", (stop,)
            )
        self._conn.commit()

    # ─────────────────────────────────────────────────────────────────────────
    # Writes
    # ─────────────────────────────────────────────────────────────────────────

    def insert_count(
        self,
        stop_name: str,
        entries: int,
        exits: int,
        timestamp: Optional[datetime] = None,
    ) -> int:
        """
        Record a passenger count event.

        Parameters
        ----------
        stop_name : str
            Name of the tram stop (must exist in the stops table).
        entries : int
            Number of passengers who boarded during this window.
        exits : int
            Number of passengers who alighted during this window.
        timestamp : datetime | None
            Defaults to now (UTC) if not provided.

        Returns
        -------
        int
            Row ID of the newly inserted record.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        iso_ts = timestamp.isoformat()

        # Resolve stop name → stop id
        row = self._conn.execute(
            "SELECT id FROM stops WHERE name = ?", (stop_name,)
        ).fetchone()

        if row is None:
            # Auto-create unknown stops so the system is resilient
            self._conn.execute(
                "INSERT INTO stops (name) VALUES (?)", (stop_name,)
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT id FROM stops WHERE name = ?", (stop_name,)
            ).fetchone()

        stop_id = row["id"]

        cursor = self._conn.execute(
            "INSERT INTO counts (stop_id, timestamp, entries, exits) VALUES (?, ?, ?, ?)",
            (stop_id, iso_ts, entries, exits),
        )
        self._conn.commit()

        inserted_id = cursor.lastrowid
        logger.debug(
            f"[Database] Inserted — stop={stop_name} entries={entries} "
            f"exits={exits} ts={iso_ts} id={inserted_id}"
        )
        return inserted_id

    # ─────────────────────────────────────────────────────────────────────────
    # Reads
    # ─────────────────────────────────────────────────────────────────────────

    def get_all_counts(self, stop_name: Optional[str] = None) -> List[CountRecord]:
        """
        Retrieve all count records, optionally filtered by stop name.

        Returns a list of CountRecord objects sorted by timestamp ascending.
        """
        if stop_name:
            rows = self._conn.execute("""
                SELECT c.id, s.name AS stop_name, c.timestamp, c.entries, c.exits, c.net
                FROM counts c
                JOIN stops  s ON c.stop_id = s.id
                WHERE s.name = ?
                ORDER BY c.timestamp ASC
            """, (stop_name,)).fetchall()
        else:
            rows = self._conn.execute("""
                SELECT c.id, s.name AS stop_name, c.timestamp, c.entries, c.exits, c.net
                FROM counts c
                JOIN stops s ON c.stop_id = s.id
                ORDER BY c.timestamp ASC
            """).fetchall()

        return [
            CountRecord(
                id=r["id"], stop_name=r["stop_name"],
                timestamp=r["timestamp"], entries=r["entries"],
                exits=r["exits"], net=r["net"],
            )
            for r in rows
        ]

    def get_totals_by_stop(self) -> List[Tuple[str, int, int]]:
        """
        Returns [(stop_name, total_entries, total_exits), ...] for all stops.
        Useful for the dashboard summary card.
        """
        rows = self._conn.execute("""
            SELECT s.name, SUM(c.entries) AS total_entries, SUM(c.exits) AS total_exits
            FROM counts c
            JOIN stops  s ON c.stop_id = s.id
            GROUP BY s.id
            ORDER BY total_entries DESC
        """).fetchall()

        return [(r["name"], r["total_entries"], r["total_exits"]) for r in rows]

    def get_hourly_traffic(self, stop_name: Optional[str] = None) -> List[dict]:
        """
        Aggregate entries + exits by hour-of-day across all records.
        Returns data ready for a bar chart in the dashboard.
        """
        if stop_name:
            rows = self._conn.execute("""
                SELECT strftime('%H', c.timestamp) AS hour,
                       SUM(c.entries) AS entries,
                       SUM(c.exits)   AS exits
                FROM counts c
                JOIN stops  s ON c.stop_id = s.id
                WHERE s.name = ?
                GROUP BY hour
                ORDER BY hour
            """, (stop_name,)).fetchall()
        else:
            rows = self._conn.execute("""
                SELECT strftime('%H', timestamp) AS hour,
                       SUM(entries) AS entries,
                       SUM(exits)   AS exits
                FROM counts
                GROUP BY hour
                ORDER BY hour
            """).fetchall()

        return [dict(r) for r in rows]

    def close(self) -> None:
        """Close the database connection gracefully."""
        self._conn.close()
        logger.info("[Database] Connection closed.")
