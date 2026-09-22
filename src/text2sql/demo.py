"""Running the app as a public demo: set itself up, and ration a free API quota.

A fresh deployment (say, on Streamlit Community Cloud) has no ``data/`` folder, since
databases and indexes are not committed. ``ensure_sample_ready`` downloads a sample
database and builds its index from table descriptions committed with the app, so
no LLM calls are needed. ``DailyBudget`` caps questions per day, because every
visitor spends the owner's free-tier tokens.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

from text2sql.config import Settings
from text2sql.db.schema import read_schema
from text2sql.indexing import build_index, load_descriptions, load_index_info, open_collection
from text2sql.indexing.embeddings import Embedder
from text2sql.indexing.indexer import DOCS_FILE
from text2sql.samples import SAMPLES, check_database, download


def index_ready(settings: Settings) -> bool:
    """True if the index holds tables and was built for this database and embedding model."""
    if not (settings.db_index_dir / "chroma").exists():
        return False
    info = load_index_info(settings.db_index_dir)
    if info.get("db_path") not in (None, str(settings.db_path.resolve())):
        return False  # say, the project folder was moved
    if info.get("embedding_model") not in (None, settings.embedding_model):
        return False
    return open_collection(settings.db_index_dir).count() > 0


def ensure_sample_ready(
    settings: Settings,
    descriptions_dir: Path,
    embedder: Callable[[], Embedder],
    log: Callable[[str], None] = lambda _msg: None,
) -> bool:
    """Download a bundled sample database and index it if either is missing or unusable.

    Only the sample databases are set up automatically: for any other database, a
    missing index should be built deliberately with ``python -m text2sql index``,
    which writes proper LLM descriptions. Returns True if anything was set up.
    """
    name = settings.db_path.stem.lower()
    seed = descriptions_dir / f"{name}.json"
    if name not in SAMPLES or not seed.exists():
        return False
    changed = False
    if not _usable(settings.db_path):
        download(name, settings.db_path, force=True, log=log)
        changed = True
    if not index_ready(settings):
        log(f"Indexing {settings.db_path.name} with the bundled table descriptions")
        _merge_descriptions(seed, settings.db_index_dir)
        tables = read_schema(settings.db_path, settings.sample_rows)
        # force: an index left over from another copy of the file is rebuilt, not refused.
        build_index(
            tables, settings.db_index_dir, embedder(), llm=None, db_path=settings.db_path,
            force=True,
        )  # fmt: skip
        changed = True
    return changed


def _usable(db_path: Path) -> bool:
    if not db_path.exists():
        return False
    try:
        check_database(db_path)
        return True
    except ValueError:
        return False  # left by an interrupted or failed download: fetch it again


def _merge_descriptions(seed: Path, index_dir: Path) -> None:
    """Fill in the bundled descriptions without overwriting ones already there.

    Existing entries may be hand-edited, or come from an ``index`` run that stopped
    part-way; tables they lack take the bundled text. An unreadable file is replaced.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    try:
        existing = load_descriptions(index_dir)
    except ValueError:  # JSONDecodeError, or bytes that are not UTF-8
        existing = {}
    bundled = json.loads(seed.read_text(encoding="utf-8"))
    merged = {**bundled, **existing}
    (index_dir / DOCS_FILE).write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
    )


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

    def refund(self) -> None:
        """Give one question back, say one that the provider's quota stopped."""
        if self.limit is None:
            return
        with self._lock:
            self._used = max(0, self._used - 1)


# Kept in this module rather than in a Streamlit cache: any visitor's browser can ask
# the server to clear those caches, which would reset the day's count.
_budgets: dict[int | None, DailyBudget] = {}
_budgets_lock = threading.Lock()


def shared_budget(limit: int | None) -> DailyBudget:
    """The one budget per server process for ``limit``, shared by every visitor."""
    with _budgets_lock:
        if limit not in _budgets:
            _budgets[limit] = DailyBudget(limit)
        return _budgets[limit]
