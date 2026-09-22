"""The provider-agnostic LLM interface used by every pipeline stage.

Stages only ever call ``LLM.complete(system, user)``. Swapping Groq for Ollama (or a
fake model in tests) is therefore a configuration change, never a code change.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Reasoning models (Qwen3, DeepSeek-R1, ...) may prepend their chain of thought.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove a model's visible chain of thought, keeping only the final answer.

    Besides complete ``<think>...</think>`` blocks this handles two unpaired cases,
    because a draft query inside the reasoning must never be mistaken for the answer:
    a reply that starts inside the block (only ``</think>`` appears), and a reply cut
    off while still reasoning (``<think>`` is never closed).
    """
    text = _THINK_RE.sub("", text)
    text = _THINK_CLOSE_RE.split(text)[-1]
    return _THINK_OPEN_RE.split(text)[0].strip()


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_s: float  # time of the successful request only
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    waited_s: float = 0.0  # time spent waiting out rate limits / transient errors


class LLMError(RuntimeError):
    """Raised when the provider fails (network, auth, rate limit after retries...)."""


class LLM(ABC):
    """A chat model bound to one provider and one model name."""

    provider: str
    model: str

    @abstractmethod
    def _complete(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> LLMResponse: ...

    def complete(
        self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 1024
    ) -> LLMResponse:
        """Send one system + user message pair and return the model's reply.

        Temperature defaults to 0 because SQL generation should be as deterministic
        and reproducible as the provider allows. ``max_tokens`` is a ceiling, not a
        target: callers keep it generous because reasoning models spend part of it
        thinking before they write the answer.
        """
        response = self._complete(system, user, temperature, max_tokens)
        response.text = strip_reasoning(response.text)
        return response

    def __repr__(self) -> str:
        return f"{type(self).__name__}(provider={self.provider!r}, model={self.model!r})"
