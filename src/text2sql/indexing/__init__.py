"""Offline indexing: describe tables with an LLM and store their embeddings."""

from text2sql.indexing.describe import describe_table, fallback_description, index_document
from text2sql.indexing.embeddings import Embedder, FastEmbedEmbedder
from text2sql.indexing.indexer import (
    IndexedTable,
    build_index,
    load_descriptions,
    load_index_info,
    open_collection,
)

__all__ = [
    "Embedder",
    "FastEmbedEmbedder",
    "IndexedTable",
    "build_index",
    "describe_table",
    "fallback_description",
    "index_document",
    "load_descriptions",
    "load_index_info",
    "open_collection",
]
