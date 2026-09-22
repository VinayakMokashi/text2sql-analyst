"""Offline indexing: schema -> descriptions -> embeddings -> ChromaDB.

Output lives in ``<index_dir>/<db name>/``:
  * ``table_docs.json``  - the table descriptions (human-readable and hand-editable;
    edit a description and re-run ``index`` to improve retrieval for your data)
  * ``index_info.json``  - which database file and embedding model built the index
  * ``chroma/``          - the persistent vector store
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from text2sql.db.schema import TableSchema
from text2sql.indexing.describe import describe_table, index_document
from text2sql.indexing.embeddings import Embedder
from text2sql.llm.base import LLM

log = logging.getLogger(__name__)

COLLECTION = "tables"
DOCS_FILE = "table_docs.json"
INFO_FILE = "index_info.json"


@dataclass
class IndexedTable:
    name: str
    description: str


def open_collection(index_dir: Path, reset: bool = False) -> chromadb.Collection:
    """Open (or create) the persistent Chroma collection for one database."""
    client = chromadb.PersistentClient(
        path=str(index_dir / "chroma"), settings=ChromaSettings(anonymized_telemetry=False)
    )
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:  # noqa: BLE001 - the collection simply did not exist yet
            pass
    # Cosine distance suits normalised sentence embeddings; we pass our own vectors,
    # so Chroma's built-in embedding function is disabled.
    return client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None
    )


def _read_json(path: Path) -> dict[str, Any]:
    # utf-8-sig: editors on Windows may save hand-edited files with a byte-order mark.
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_descriptions(index_dir: Path) -> dict[str, str]:
    return _read_json(index_dir / DOCS_FILE)


def load_index_info(index_dir: Path) -> dict[str, Any]:
    return _read_json(index_dir / INFO_FILE)


def build_index(
    tables: list[TableSchema],
    index_dir: Path,
    embedder: Embedder,
    llm: LLM | None,
    refresh_descriptions: bool = False,
    on_progress: Callable[[str], None] | None = None,
    db_path: Path | None = None,
    force: bool = False,
) -> list[IndexedTable]:
    """Describe, embed and store every table.

    Existing descriptions are reused unless ``refresh_descriptions`` is set, so
    re-indexing costs no LLM calls. ``force`` allows reusing a folder that was built
    for a different database file (for example after moving the file).
    """
    if not tables:
        raise ValueError("The database has no tables to index.")
    info = load_index_info(index_dir)
    source = str(db_path.resolve()) if db_path else None
    if source and info.get("db_path") not in (None, source) and not force:
        raise ValueError(
            f"{index_dir} holds the index of {info['db_path']}. Set T2S_INDEX_DIR to "
            "another folder, or pass --force to replace it (its descriptions are reused)."
        )

    index_dir.mkdir(parents=True, exist_ok=True)
    cached = {} if refresh_descriptions else load_descriptions(index_dir)
    descriptions: dict[str, str] = {}
    for table in tables:
        if table.name in cached:
            descriptions[table.name] = cached[table.name]
            continue
        if on_progress:
            on_progress(f"Describing {table.name}")
        descriptions[table.name] = describe_table(table, llm)
        # Save as we go: if a later call fails (say, a daily rate limit), the
        # descriptions already paid for are kept and reused on the next run.
        _write_json(index_dir / DOCS_FILE, {**cached, **descriptions})
    _write_json(index_dir / DOCS_FILE, descriptions)  # drop tables that no longer exist

    if on_progress:
        on_progress("Embedding descriptions")
    docs = [index_document(t, descriptions[t.name]) for t in tables]
    vectors = embedder.embed_documents(docs)

    # Update the collection in place rather than recreating it: a running app holds a
    # handle to it, and a recreated collection would make that handle invalid. Vectors
    # from a different embedding model are incompatible, so that case starts afresh.
    model = getattr(embedder, "model_name", None)
    collection = open_collection(index_dir, reset=info.get("embedding_model") != model)
    stale = set(collection.get(include=[])["ids"]) - {t.name for t in tables}
    if stale:
        collection.delete(ids=sorted(stale))
    collection.upsert(
        ids=[t.name for t in tables],
        documents=docs,
        embeddings=vectors,
        metadatas=[{"table": t.name, "description": descriptions[t.name]} for t in tables],
    )
    _write_json(
        index_dir / INFO_FILE,
        {
            "db_path": source,
            "embedding_model": model,
            "tables": len(tables),
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    log.info("Indexed %d tables into %s", len(tables), index_dir)
    return [IndexedTable(t.name, descriptions[t.name]) for t in tables]
