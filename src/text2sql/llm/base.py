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


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


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
        and reproducible as the provider allows.
        """
        response = self._complete(system, user, temperature, max_tokens)
        response.text = _THINK_RE.sub("", response.text).strip()
        return response

    def __repr__(self) -> str:
        return f"{type(self).__name__}(provider={self.provider!r}, model={self.model!r})"
