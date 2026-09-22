"""Fetch a subset of the BIRD mini-dev benchmark and fix the questions to evaluate.

BIRD (https://bird-bench.github.io, CC BY-SA 4.0) is a current, harder Text-to-SQL
benchmark: 500 curated "mini-dev" questions over 11 real SQLite databases, each with
an evidence hint written by the annotators. The archive is 800 MB, mostly three very
large databases, so this script reads the zip's table of contents over HTTP and
downloads only the eight smaller databases (about 50 MB) with range requests.

Then it samples the evaluation subset once, with a fixed seed, in BIRD's own
difficulty mix, and writes the chosen question ids to ``eval/bird_subset.json``. The
subset is committed before any model is run on it.

Usage:
    python eval/bird_prepare.py [--n 100] [--seed 42] [--resample]
"""

from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
ROOT = "minidev/MINIDEV/"
DATA_DIR = Path("data/bird")
SUBSET_FILE = Path(__file__).parent / "bird_subset.json"
# The eight databases under 75 MB; card_games, european_football_2 and
# codebase_community (1.3 GB together, uncompressed) are left out.
DATABASES = [
    "california_schools", "debit_card_specializing", "financial", "formula_1",
    "student_club", "superhero", "thrombosis_prediction", "toxicology",
]  # fmt: skip
# BIRD mini-dev's own mix: 148 simple, 250 moderate, 102 challenging of 500.
MIX = {"simple": 0.3, "moderate": 0.5, "challenging": 0.2}


class HttpRangeFile(io.RawIOBase):
    """A seekable, read-only remote file: zipfile reads only the parts it needs.

    Reads are served from a read-ahead buffer, so extracting a member takes a
    handful of requests rather than one per 4 KB chunk.
    """

    def __init__(self, url: str, block: int = 8 << 20) -> None:
        self.url, self.pos, self.block = url, 0, block
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.size = int(head.headers["Content-Length"])
        self.buf_start, self.buf = 0, b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        buffered = self.buf_start <= self.pos and self.pos + n <= self.buf_start + len(self.buf)
        if not buffered:
            end = min(self.size, self.pos + max(n, self.block)) - 1
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
            self.buf_start, self.buf = self.pos, urllib.request.urlopen(request).read()
        start = self.pos - self.buf_start
        data = self.buf[start : start + n]
        self.pos += len(data)
        return data

    def readinto(self, buffer) -> int:  # type: ignore[no-untyped-def]
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def download(archive: zipfile.ZipFile) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    questions = DATA_DIR / "mini_dev_sqlite.json"
    if not questions.exists():
        questions.write_bytes(archive.read(ROOT + "mini_dev_sqlite.json"))
    for db in DATABASES:
        out = DATA_DIR / f"{db}.sqlite"
        if out.exists():
            continue
        member = f"{ROOT}dev_databases/{db}/{db}.sqlite"
        size = archive.getinfo(member).compress_size / 1e6
        print(f"Downloading {db} ({size:.1f} MB compressed)")
        with archive.open(member) as src, open(out.with_suffix(".part"), "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        out.with_suffix(".part").replace(out)


def sample(n: int, seed: int) -> dict:
    questions = json.loads((DATA_DIR / "mini_dev_sqlite.json").read_text(encoding="utf-8"))
    pool: dict[str, list[int]] = defaultdict(list)
    for q in questions:
        if q["db_id"] in DATABASES:
            pool[q["difficulty"]].append(q["question_id"])
    rng = random.Random(seed)
    counts = {d: round(n * share) for d, share in MIX.items()}
    ids = sorted(i for d, k in counts.items() for i in rng.sample(sorted(pool[d]), k))
    return {
        "source": "BIRD mini-dev (SQLite), https://github.com/bird-bench/mini_dev",
        "license": "CC BY-SA 4.0",
        "databases": DATABASES,
        "seed": seed,
        "difficulty_counts": counts,
        "question_ids": ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a BIRD mini-dev subset.")
    parser.add_argument("--n", type=int, default=100, help="questions in the subset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resample", action="store_true", help="overwrite the subset file")
    args = parser.parse_args()

    download(zipfile.ZipFile(HttpRangeFile(URL)))
    if SUBSET_FILE.exists() and not args.resample:
        print(f"Keeping the existing subset in {SUBSET_FILE} (use --resample to replace it)")
    else:
        subset = sample(args.n, args.seed)
        SUBSET_FILE.write_text(json.dumps(subset, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {len(subset['question_ids'])} question ids to {SUBSET_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
