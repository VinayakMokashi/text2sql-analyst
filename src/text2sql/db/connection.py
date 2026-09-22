"""Read-only SQLite connections.

The read-only guarantee comes from SQLite itself (``mode=ro`` in the URI), not from
our own checks, so even a query that slipped past the SQL guardrails cannot modify
the database. ``PRAGMA query_only`` is a second, independent layer.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from urllib.request import pathname2url


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open ``db_path`` read-only. Raises ``FileNotFoundError`` if it does not exist."""
    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Database not found: {path}. Run `python scripts/download_sample_db.py` first "
            "or point T2S_DB_PATH at your own SQLite file."
        )
    # pathname2url percent-encodes spaces, "#" and "?" and handles Windows drive letters
    # and network (UNC) paths, which Path.as_uri() turns into a form SQLite rejects.
    uri = f"file:{pathname2url(str(path))}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.execute("PRAGMA query_only = ON")
    return conn
