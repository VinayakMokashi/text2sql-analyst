"""Client for any OpenAI-compatible chat endpoint.

Groq, Cerebras, OpenRouter, Ollama, vLLM and LM Studio all expose the same
``/chat/completions`` API, so a single implementation covers every provider; only
the base URL, the API key and the model name differ.
"""

from __future__ import annotations

import time
from typing import Any

import openai

from text2sql.llm.base import LLM, LLMError, LLMResponse

# Transient failures worth waiting for; anything else (bad key, unknown model, bad
# request) fails immediately.
RETRYABLE = (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)


def retry_delay(exc: Exception, attempt: int) -> float:
    """Seconds to wait before retrying: the server's Retry-After if given, else backoff."""
    response = getattr(exc, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    try:
        return max(0.0, float(header)) if header is not None else float(min(2**attempt, 30))
    except ValueError:
        return float(min(2**attempt, 30))


class OpenAICompatibleLLM(LLM):
    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str,
        api_key: str,
        timeout_s: float = 60.0,
        max_retries: int = 8,
        max_wait_s: float = 120.0,
        reasoning_effort: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.reasoning_effort = reasoning_effort or None
        self._max_retries = max_retries
        self._max_wait_s = max_wait_s
        # Retries are handled below rather than inside the SDK, so that time spent
        # waiting out a free-tier rate limit is reported separately from the model's
        # own response time.
        self._client = openai.OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0
        )

    def list_models(self) -> list[str]:
        """Model ids the endpoint currently serves (free-tier catalogues change often)."""
        try:
            return sorted(m.id for m in self._client.models.list().data)
        except openai.APIError as exc:
            raise LLMError(f"{self.provider}: could not list models: {exc}") from exc

    def _complete(self, system: str, user: str, temperature: float, max_tokens: int) -> LLMResponse:
        # Sent as a raw body field: not every provider or model knows the parameter,
        # so it is only included when explicitly configured.
        extra: dict[str, Any] = (
            {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        )
        waited = 0.0
        for attempt in range(self._max_retries + 1):
            start = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body=extra or None,
                )
                break
            except RETRYABLE as exc:
                delay = retry_delay(exc, attempt)
                if attempt == self._max_retries or waited + delay > self._max_wait_s:
                    raise LLMError(
                        f"{self.provider}/{self.model}: still failing after {attempt + 1} "
                        f"attempt(s) and {waited:.0f}s of waiting ({exc}). If this is a daily "
                        "free-tier limit, try again later or switch to another model."
                    ) from exc
                time.sleep(delay)
                waited += delay
            except openai.APIError as exc:  # auth, unknown model, bad request, ...
                raise LLMError(f"{self.provider}/{self.model}: {exc}") from exc

        usage = resp.usage
        return LLMResponse(
            # Reasoning models return their chain of thought in a separate field on
            # most providers, so ``content`` holds only the final answer.
            text=resp.choices[0].message.content or "",
            model=self.model,
            latency_s=time.perf_counter() - start,  # the successful call only
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            waited_s=waited,
        )
