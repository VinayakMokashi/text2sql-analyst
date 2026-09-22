"""Download a sample database as SQLite.

  chinook  a digital music store: artists, albums, tracks, playlists, customers,
           employees, invoices (11 tables). https://github.com/lerocha/chinook-database
           (MIT License). The default.
  sakila   a DVD rental chain: films, actors, categories, inventory, rentals, payments,
           customers, stores, staff (15 tables). https://github.com/jOOQ/sakila
           (BSD 2-Clause License). A second, larger schema for testing that nothing
           is tuned to Chinook.

Usage:
    python scripts/download_sample_db.py [chinook|sakila] [--out PATH] [--force]
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import tempfile
import urllib.request
from contextlib import closing
from pathlib import Path

CHINOOK_URL = (
    "https://github.com/lerocha/chinook-database/releases/download/v1.4.5/Chinook_Sqlite.sqlite"
)
SAKILA_BASE = "https://raw.githubusercontent.com/jOOQ/sakila/main/sqlite-sakila-db/"


def fetch(url: str, dest: Path) -> Path:
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def build_chinook(out: Path, tmp: Path) -> None:
    # Already a SQLite file. shutil.move, not Path.replace: the temp folder may be on
    # another drive or filesystem, where a rename fails.
    shutil.move(fetch(CHINOOK_URL, tmp / "chinook.sqlite"), out)


def build_sakila(out: Path, tmp: Path) -> None:
    """Sakila ships as SQL scripts, so the database is built locally."""
    schema = fetch(SAKILA_BASE + "sqlite-sakila-schema.sql", tmp / "schema.sql")
    data = fetch(SAKILA_BASE + "sqlite-sakila-insert-data.sql", tmp / "data.sql")
    print("Building the database...")
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


BUILDERS = {"chinook": build_chinook, "sakila": build_sakila}


def download(name: str, out: Path, force: bool = False) -> Path:
    """Create ``out`` unless it already exists, then sanity-check it."""
    if out.exists() and not force:
        print(f"Already present: {out} (use --force to re-download)")
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        BUILDERS[name](out, Path(tmp))

    # Fail loudly now rather than later in the pipeline if the file is not a database.
    with closing(sqlite3.connect(out)) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB) with {len(tables)} tables:")
    print("  " + ", ".join(sorted(tables)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", nargs="?", default="chinook", choices=sorted(BUILDERS))
    parser.add_argument("--out", type=Path, help="default: data/<name>.db")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    download(args.name, args.out or Path("data") / f"{args.name}.db", args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
