"""Client for any OpenAI-compatible chat endpoint.

Groq, Cerebras, GitHub Models, OpenRouter and Ollama all expose the same
``/chat/completions`` API, so a single implementation covers every provider; only
the base URL, the API key and the model name differ.
"""

from __future__ import annotations

import time

import openai

from text2sql.llm.base import LLM, LLMError, LLMResponse


class OpenAICompatibleLLM(LLM):
    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str,
        api_key: str,
        timeout_s: float = 60.0,
        max_retries: int = 6,
    ) -> None:
        self.provider = provider
        self.model = model
        # The SDK retries 429/5xx with exponential backoff and honours Retry-After,
        # which is exactly what free-tier rate limits need.
        self._client = openai.OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=max_retries
        )

    def _complete(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
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
            )
        except openai.APIError as exc:  # covers auth, rate limit, connection, 5xx
            raise LLMError(f"{self.provider}/{self.model}: {exc}") from exc

        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=self.model,
            latency_s=time.perf_counter() - start,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )
