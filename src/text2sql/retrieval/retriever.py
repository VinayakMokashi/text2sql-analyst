"""Stage 1 of schema linking: vector search for candidate tables."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import chromadb
from chromadb.errors import NotFoundError

from text2sql.indexing.embeddings import Embedder


@dataclass(frozen=True)
class RetrievedTable:
    name: str
    description: str
    score: float  # cosine similarity in [-1, 1]; higher = more relevant


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
