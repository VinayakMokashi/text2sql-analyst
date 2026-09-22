"""Local text embeddings.

We use ``fastembed`` (ONNX runtime) instead of ``sentence-transformers`` because it
runs the same open-source models without installing PyTorch: about 70 MB of
dependencies and a ~65 MB quantized model instead of multiple gigabytes.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

# The model is fetched from the Hugging Face Hub; on Windows without Developer Mode the
# hub falls back to copying files and warns about it on every run. Harmless, so hide it.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# BGE models were trained with this instruction in front of *queries* (not documents);
# adding it measurably improves retrieval for short questions.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    """Anything that can turn text into vectors (a fake one is used in tests)."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str, cache_dir: Path | None = None) -> None:
        from fastembed import TextEmbedding  # imported lazily: loading ONNX takes a moment

        self.model_name = model_name
        self._model = TextEmbedding(
            model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None
        )
        self._query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([self._query_prefix + text]))).tolist()
