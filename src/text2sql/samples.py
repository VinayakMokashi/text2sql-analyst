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

import os
import shutil
import sqlite3
import tempfile
import urllib.request
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

from text2sql.db.connection import connect_readonly

CHINOOK_URL = (
    "https://github.com/lerocha/chinook-database/releases/download/v1.4.5/Chinook_Sqlite.sqlite"
)
SAKILA_BASE = "https://raw.githubusercontent.com/jOOQ/sakila/main/sqlite-sakila-db/"
# Per network operation, not in total. Without it a stalled connection would hang the
# app's first start, and every visitor waits on that.
TIMEOUT_S = 60.0

Log = Callable[[str], None]


def _fetch(url: str, dest: Path, log: Log) -> Path:
    log(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resp, dest.open("wb") as fh:
        shutil.copyfileobj(resp, fh)
        expected = resp.headers.get("Content-Length")
    # A connection that drops early can look like a finished download.
    if expected is not None and dest.stat().st_size != int(expected):
        raise OSError(
            f"Download of {url} stopped early ({dest.stat().st_size} of {expected} bytes)"
        )
    return dest


def _build_chinook(tmp: Path, log: Log) -> Path:
    return _fetch(CHINOOK_URL, tmp / "chinook.sqlite", log)  # already a SQLite file


def _build_sakila(tmp: Path, log: Log) -> Path:
    """Sakila ships as SQL scripts, so the database is built locally."""
    schema = _fetch(SAKILA_BASE + "sqlite-sakila-schema.sql", tmp / "schema.sql", log)
    data = _fetch(SAKILA_BASE + "sqlite-sakila-insert-data.sql", tmp / "data.sql", log)
    log("Building the database...")
    db = tmp / "sakila.db"
    # closing(): sqlite3's own context manager only commits, and an open connection
    # keeps the file locked on Windows, so it could not be moved into place.
    with closing(sqlite3.connect(db)) as conn:
        conn.executescript(schema.read_text(encoding="utf-8"))
        # One transaction for ~50k INSERTs: committing each one separately is ~50x slower.
        conn.executescript("BEGIN;\n" + data.read_text(encoding="utf-8") + "\nCOMMIT;")
        # film_text is an always-empty full-text-search helper in this port. Left in, it
        # invites the model to query a table that can never return rows.
        conn.executescript("DROP TABLE IF EXISTS film_text; VACUUM;")
    return db


SAMPLES: dict[str, Callable[[Path, Log], Path]] = {
    "chinook": _build_chinook,
    "sakila": _build_sakila,
}


def check_database(path: Path) -> list[str]:
    """The table names in a SQLite file; ValueError if it is not a usable database.

    Catches what a bad download leaves behind: an HTML error page saved as the file,
    a cut-off file, or an empty database.
    """
    try:
        with closing(connect_readonly(path)) as conn:
            if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError(f"{path} is damaged (PRAGMA quick_check failed)")
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\'"
                )
            ]
    except sqlite3.OperationalError:
        raise  # locked, unreadable, ...: a problem with access, not with the file
    except sqlite3.DatabaseError as exc:
        raise ValueError(f"{path} is not a usable SQLite database ({exc})") from exc
    if not tables:
        raise ValueError(f"{path} has no tables")
    return tables


def download(name: str, out: Path, force: bool = False, log: Log = print) -> Path:
    """Create ``out`` unless it already exists, checking the database before it is used."""
    if out.exists() and not force:
        log(f"Already present: {out} (use --force to re-download)")
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        built = SAMPLES[name](Path(tmp), log)
        tables = check_database(built)  # a bad file never reaches ``out``
        # Copy next to the target, then rename: the temp folder may be on another drive,
        # and a rename within one folder is atomic, so an interrupted copy never leaves
        # a half-written database in place.
        part = out.with_name(out.name + ".part")
        shutil.copyfile(built, part)
        os.replace(part, out)
    log(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB) with {len(tables)} tables:")
    log("  " + ", ".join(sorted(tables)))
    return out
