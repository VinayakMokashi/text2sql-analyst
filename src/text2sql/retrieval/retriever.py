"""Stage 1 of schema linking: find candidate tables for a question.

Two retrievers share one small interface (``SchemaRetriever``), so the pipeline works
with either, or with one an application brings:

* ``TableRetriever`` searches a vector index of table descriptions. It suits schemas too
  large to show the model in full, and needs the ``index`` extra (ChromaDB).
* ``AllTablesRetriever`` offers every table. For a schema of a dozen tables or so the
  whole schema fits in the prompt, so a vector index adds a build step without adding
  recall.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from text2sql.db.schema import TableSchema
from text2sql.indexing.describe import fallback_description
from text2sql.indexing.embeddings import Embedder

if TYPE_CHECKING:
    import chromadb


@dataclass(frozen=True)
class RetrievedTable:
    name: str
    description: str
    score: float  # cosine similarity in [-1, 1]; higher = more relevant


class SchemaRetriever(Protocol):
    """What the pipeline needs from a retriever."""

    def count(self) -> int:
        """Number of tables it can return (0 means nothing to search)."""
        ...

    def search(self, question: str, n: int) -> list[RetrievedTable]:
        """Up to ``n`` candidate tables for the question, most relevant first."""
        ...


class TableRetriever:
    """Finds the tables whose descriptions are most similar to the question.

    This is deliberately recall-oriented (it returns more tables than needed); the
    LLM selection step afterwards trims the list for precision.
    """

    def __init__(
        self,
        collection: chromadb.Collection,
        embedder: Embedder,
        reopen: Callable[[], chromadb.Collection] | None = None,
    ) -> None:
        self._collection = collection
        self._embedder = embedder
        self._reopen = reopen

    def count(self) -> int:
        """Number of indexed tables (0 means the index was never built)."""
        return self._collection.count()

    def search(self, question: str, n: int) -> list[RetrievedTable]:
        from chromadb.errors import NotFoundError

        try:
            return self._search(question, n)
        except NotFoundError:
            # The index was rebuilt from scratch while this process was running (e.g.
            # after changing the embedding model); fetch the new collection once.
            if self._reopen is None:
                raise
            self._collection = self._reopen()
            return self._search(question, n)

    def _search(self, question: str, n: int) -> list[RetrievedTable]:
        total = self._collection.count()
        if total == 0:
            raise RuntimeError("The table index is empty. Run `python -m text2sql index` first.")
        result = self._collection.query(
            query_embeddings=[self._embedder.embed_query(question)],
            n_results=min(n, total),
            include=["metadatas", "distances"],
        )
        return [
            RetrievedTable(
                name=str(meta["table"]),
                description=str(meta.get("description", "")),
                score=round(1.0 - float(dist), 4),  # Chroma returns cosine *distance*
            )
            for meta, dist in zip(result["metadatas"][0], result["distances"][0], strict=True)
        ]


class AllTablesRetriever:
    """Offers every table, in schema order, for schemas small enough to show whole.

    ``descriptions`` maps table names to hand-written or LLM-written descriptions (for
    example a ``table_docs.json``); a table without one gets a template description.
    Every table scores 1.0, and ``n`` is ignored: the selection step that follows does
    the narrowing.
    """

    def __init__(
        self, schemas: Sequence[TableSchema], descriptions: Mapping[str, str] | None = None
    ) -> None:
        given = {name.lower(): text for name, text in (descriptions or {}).items()}
        self._tables = [
            RetrievedTable(t.name, given.get(t.name.lower()) or fallback_description(t), 1.0)
            for t in schemas
        ]

    def count(self) -> int:
        return len(self._tables)

    def search(self, question: str, n: int) -> list[RetrievedTable]:
        return list(self._tables)
