"""Running the app as a public demo: set itself up, and ration a free API quota.

A fresh deployment (say, on Streamlit Community Cloud) has no ``data/`` folder, since
databases and indexes are not committed. ``ensure_sample_ready`` downloads a sample
database and builds its index from table descriptions committed with the app, so
no LLM calls are needed. ``DailyBudget`` caps questions per day, because every
visitor spends the owner's free-tier tokens.
"""

from __future__ import annotations

import shutil
import threading
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

from text2sql.config import Settings
from text2sql.db.schema import read_schema
from text2sql.indexing import build_index, open_collection
from text2sql.indexing.embeddings import Embedder
from text2sql.indexing.indexer import DOCS_FILE
from text2sql.samples import SAMPLES, download


def index_ready(settings: Settings) -> bool:
    if not (settings.db_index_dir / "chroma").exists():
        return False
    return open_collection(settings.db_index_dir).count() > 0


def ensure_sample_ready(
    settings: Settings,
    descriptions_dir: Path,
    embedder: Callable[[], Embedder],
    log: Callable[[str], None] = lambda _msg: None,
) -> bool:
    """Download a bundled sample database and index it if either is missing.

    Only the sample databases are set up automatically: for any other database, a
    missing index should be built deliberately with ``python -m text2sql index``,
    which writes proper LLM descriptions. Returns True if anything was set up.
    """
    name = settings.db_path.stem.lower()
    seed = descriptions_dir / f"{name}.json"
    if name not in SAMPLES or not seed.exists():
        return False
    changed = False
    if not settings.db_path.exists():
        download(name, settings.db_path, log=log)
        changed = True
    if not index_ready(settings):
        log(f"Indexing {settings.db_path.name} with the bundled table descriptions")
        settings.db_index_dir.mkdir(parents=True, exist_ok=True)
        docs = settings.db_index_dir / DOCS_FILE
        if not docs.exists():
            shutil.copyfile(seed, docs)
        tables = read_schema(settings.db_path, settings.sample_rows)
        build_index(tables, settings.db_index_dir, embedder(), llm=None, db_path=settings.db_path)
        changed = True
    return changed


class DailyBudget:
    """A thread-safe count of questions per UTC day, shared by every visitor.

    ``limit=None`` means no limit (the default outside a demo).
    """

    def __init__(self, limit: int | None, today: Callable[[], date] | None = None) -> None:
        self.limit = limit
        self._today = today or (lambda: datetime.now(UTC).date())
        self._day = self._today()
        self._used = 0
        self._lock = threading.Lock()

    def _roll_over(self) -> None:
        if self._today() != self._day:
            self._day, self._used = self._today(), 0

    def remaining(self) -> int | None:
        if self.limit is None:
            return None
        with self._lock:
            self._roll_over()
            return max(0, self.limit - self._used)

    def take(self) -> bool:
        """Use one question from today's budget; False once it is spent."""
        if self.limit is None:
            return True
        with self._lock:
            self._roll_over()
            if self._used >= self.limit:
                return False
            self._used += 1
            return True
