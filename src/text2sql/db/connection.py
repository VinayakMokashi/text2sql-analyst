"""Read-only SQLite connections.

The read-only guarantee comes from SQLite itself (``mode=ro`` in the URI), not from
our own checks, so even a query that slipped past the SQL guardrails cannot modify
the database. ``PRAGMA query_only`` is a second, independent layer.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open ``db_path`` read-only. Raises ``FileNotFoundError`` if it does not exist."""
    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Database not found: {path}. Run `python scripts/download_sample_db.py` first "
            "or point T2S_DB_PATH at your own SQLite file."
        )
    # Path.as_uri() percent-encodes spaces and handles Windows drive letters.
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, check_same_thread=False)
    conn.execute("PRAGMA query_only = ON")
    return conn
