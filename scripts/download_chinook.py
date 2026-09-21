"""Download the Chinook sample database (a digital music store) as SQLite.

Chinook models artists, albums, tracks, playlists, customers, employees and invoices,
which gives a realistic mix of joins and aggregations for Text-to-SQL questions.
Source: https://github.com/lerocha/chinook-database (MIT License).

Usage:
    python scripts/download_chinook.py [--out data/chinook.db] [--force]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import urllib.request
from pathlib import Path

CHINOOK_URL = (
    "https://github.com/lerocha/chinook-database/releases/download/v1.4.5/Chinook_Sqlite.sqlite"
)


def download(out: Path, force: bool = False) -> Path:
    """Download Chinook to ``out`` unless it already exists, then sanity-check it."""
    if out.exists() and not force:
        print(f"Already present: {out} (use --force to re-download)")
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    print(f"Downloading {CHINOOK_URL}")
    urllib.request.urlretrieve(CHINOOK_URL, tmp)
    tmp.replace(out)

    # Fail loudly now rather than later in the pipeline if the file is not a database.
    with sqlite3.connect(out) as conn:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB) with {len(tables)} tables:")
    print("  " + ", ".join(sorted(tables)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/chinook.db"))
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    download(args.out, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
