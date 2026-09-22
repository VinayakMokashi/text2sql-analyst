"""The bundled sample databases: download them as SQLite files.

  chinook  a digital music store: artists, albums, tracks, playlists, customers,
           employees, invoices (11 tables). https://github.com/lerocha/chinook-database
           (MIT License). The default.
  sakila   a DVD rental chain: films, actors, categories, inventory, rentals, payments,
           customers, stores, staff (15 tables). https://github.com/jOOQ/sakila
           (BSD 2-Clause License). A second, larger schema for testing that nothing
           is tuned to Chinook.

Used by ``scripts/download_sample_db.py`` and by the app, which fetches a sample
database by itself the first time it runs (for example on a fresh cloud deployment).
"""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
import urllib.request
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

CHINOOK_URL = (
    "https://github.com/lerocha/chinook-database/releases/download/v1.4.5/Chinook_Sqlite.sqlite"
)
SAKILA_BASE = "https://raw.githubusercontent.com/jOOQ/sakila/main/sqlite-sakila-db/"

Log = Callable[[str], None]


def _fetch(url: str, dest: Path, log: Log) -> Path:
    log(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def _build_chinook(out: Path, tmp: Path, log: Log) -> None:
    # Already a SQLite file. shutil.move, not Path.replace: the temp folder may be on
    # another drive or filesystem, where a rename fails.
    shutil.move(_fetch(CHINOOK_URL, tmp / "chinook.sqlite", log), out)


def _build_sakila(out: Path, tmp: Path, log: Log) -> None:
    """Sakila ships as SQL scripts, so the database is built locally."""
    schema = _fetch(SAKILA_BASE + "sqlite-sakila-schema.sql", tmp / "schema.sql", log)
    data = _fetch(SAKILA_BASE + "sqlite-sakila-insert-data.sql", tmp / "data.sql", log)
    log("Building the database...")
    part = tmp / "sakila.db"
    # closing(): sqlite3's own context manager only commits, and an open connection
    # keeps the file locked on Windows, so it could not be moved into place.
    with closing(sqlite3.connect(part)) as conn:
        conn.executescript(schema.read_text(encoding="utf-8"))
        # One transaction for ~50k INSERTs: committing each one separately is ~50x slower.
        conn.executescript("BEGIN;\n" + data.read_text(encoding="utf-8") + "\nCOMMIT;")
        # film_text is an always-empty full-text-search helper in this port. Left in, it
        # invites the model to query a table that can never return rows.
        conn.executescript("DROP TABLE IF EXISTS film_text; VACUUM;")
    shutil.move(part, out)


SAMPLES: dict[str, Callable[[Path, Path, Log], None]] = {
    "chinook": _build_chinook,
    "sakila": _build_sakila,
}


def download(name: str, out: Path, force: bool = False, log: Log = print) -> Path:
    """Create ``out`` unless it already exists, then sanity-check it."""
    if out.exists() and not force:
        log(f"Already present: {out} (use --force to re-download)")
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        SAMPLES[name](out, Path(tmp), log)

    # Fail loudly now rather than later in the pipeline if the file is not a database.
    with closing(sqlite3.connect(out)) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\'"
            )
        ]
    log(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB) with {len(tables)} tables:")
    log("  " + ", ".join(sorted(tables)))
    return out
