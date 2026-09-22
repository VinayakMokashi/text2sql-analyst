"""Download a sample database as SQLite (see text2sql/samples.py for what each one is).

Usage:
    python scripts/download_sample_db.py [chinook|sakila] [--out PATH] [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from text2sql.samples import SAMPLES, download


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", nargs="?", default="chinook", choices=sorted(SAMPLES))
    parser.add_argument("--out", type=Path, help="default: data/<name>.db")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()
    download(args.name, args.out or Path("data") / f"{args.name}.db", args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
