"""Client for any OpenAI-compatible chat endpoint.

Groq, Cerebras, OpenRouter, Ollama, vLLM and LM Studio all expose the same
``/chat/completions`` API, so a single implementation covers every provider; only
the base URL, the API key and the model name differ.
"""

from __future__ import annotations

import time
from typing import Any

import openai
from openai.types.chat import ChatCompletion

from text2sql.llm.base import LLM, LLMError, LLMResponse

# Transient failures worth waiting for; anything else (bad key, unknown model, bad
# request) fails immediately.
RETRYABLE = (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)
# A rate limit clears with time, so it gets many tries. A connection error or timeout
# usually means the server is down or the URL is wrong, so it gets few.
CONNECTION_RETRIES = 2


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
        name = f"{self.provider}/{self.model}"
        first_try = time.monotonic()
        waited = 0.0
        attempt = 0
        while True:
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
                if isinstance(exc, openai.RateLimitError):
                    limit = self._max_retries
                    hint = "If this is a daily free-tier limit, try again later or switch model."
                else:
                    limit = min(CONNECTION_RETRIES, self._max_retries)
                    hint = "Check that the server is running and the base URL is correct."
                delay = retry_delay(exc, attempt)
                elapsed = time.monotonic() - first_try
                # The budget counts time spent in failed requests too, not just sleeps.
                if attempt >= limit or elapsed + delay > self._max_wait_s:
                    raise LLMError(
                        f"{name}: gave up after {attempt + 1} attempt(s) and {elapsed:.0f}s "
                        f"({exc}). {hint}"
                    ) from exc
                time.sleep(delay)
                waited += delay
                attempt += 1
            except openai.APIError as exc:  # auth, unknown model, bad request, ...
                raise LLMError(f"{name}: {exc}") from exc

        # Some servers answer HTTP 200 with an error body or an HTML page (a wrong base
        # URL pointing at a web UI); turn those into a readable LLMError, not a crash.
        if not isinstance(resp, ChatCompletion) or not resp.choices:
            detail = getattr(resp, "error", None) or str(resp)[:200]
            raise LLMError(f"{name}: unexpected response from the server: {detail}")
        choice = resp.choices[0]
        text = (choice.message.content if choice.message else None) or ""
        if not text.strip() and choice.finish_reason == "length":
            raise LLMError(
                f"{name}: the reply hit the {max_tokens}-token limit before any answer "
                "(a reasoning model spent it thinking). Try a lower reasoning effort."
            )
        usage = resp.usage
        return LLMResponse(
            # Reasoning models return their chain of thought in a separate field on
            # most providers, so ``content`` holds only the final answer.
            text=text,
            model=self.model,
            latency_s=time.perf_counter() - start,  # the successful call only
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            waited_s=waited,
        )
