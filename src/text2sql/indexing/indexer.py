"""Offline indexing: schema -> descriptions -> embeddings -> ChromaDB.

Output lives in ``<index_dir>/<db name>/``:
  * ``table_docs.json`` - the table descriptions (human-readable and hand-editable;
    edit a description and re-run ``index`` to improve retrieval for your data)
  * ``chroma/``        - the persistent vector store
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from text2sql.db.schema import TableSchema
from text2sql.indexing.describe import describe_table, index_document
from text2sql.indexing.embeddings import Embedder
from text2sql.llm.base import LLM

log = logging.getLogger(__name__)

COLLECTION = "tables"
DOCS_FILE = "table_docs.json"


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


def load_descriptions(index_dir: Path) -> dict[str, str]:
    path = index_dir / DOCS_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def build_index(
    tables: list[TableSchema],
    index_dir: Path,
    embedder: Embedder,
    llm: LLM | None,
    refresh_descriptions: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> list[IndexedTable]:
    """Describe, embed and store every table. Existing descriptions are reused
    unless ``refresh_descriptions`` is set, so re-indexing costs no LLM calls."""
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

    (index_dir / DOCS_FILE).write_text(json.dumps(descriptions, indent=2), encoding="utf-8")

    if on_progress:
        on_progress("Embedding descriptions")
    docs = [index_document(t, descriptions[t.name]) for t in tables]
    vectors = embedder.embed_documents(docs)

    collection = open_collection(index_dir, reset=True)
    collection.add(
        ids=[t.name for t in tables],
        documents=docs,
        embeddings=vectors,
        metadatas=[{"table": t.name, "description": descriptions[t.name]} for t in tables],
    )
    log.info("Indexed %d tables into %s", len(tables), index_dir)
    return [IndexedTable(t.name, descriptions[t.name]) for t in tables]
