"""A deterministic stand-in LLM for tests and offline demos."""

from __future__ import annotations

from collections.abc import Callable

from text2sql.llm.base import LLM, LLMResponse

Responder = Callable[[str, str], str]


class FakeLLM(LLM):
    """Returns scripted replies and records every call.

    ``replies`` can be a list (returned in order; the last one repeats) or a function
    ``(system, user) -> reply`` for replies that depend on the prompt.
    """

    provider = "fake"

    def __init__(self, replies: list[str] | Responder | None = None, model: str = "fake") -> None:
        self.model = model
        self._replies = replies if replies is not None else ["SELECT 1"]
        self.calls: list[tuple[str, str]] = []

    def _complete(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        self.calls.append((system, user))
        if callable(self._replies):
            text = self._replies(system, user)
        else:
            idx = min(len(self.calls) - 1, len(self._replies) - 1)
            text = self._replies[idx]
        return LLMResponse(text=text, model=self.model, latency_s=0.0)
