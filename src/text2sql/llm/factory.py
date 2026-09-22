"""Build ``LLM`` instances from settings.

Provider presets keep ``.env`` short: choosing ``groq`` fills in the endpoint and the
name of the environment variable that holds the key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from text2sql.config import Settings
from text2sql.llm.base import LLM
from text2sql.llm.fake import FakeLLM
from text2sql.llm.openai_compat import OpenAICompatibleLLM

Role = Literal["sql", "helper"]


@dataclass(frozen=True)
class ProviderPreset:
    base_url: str
    key_env: str | None  # None = no key needed (local servers)
    signup_url: str = ""


PRESETS: dict[str, ProviderPreset] = {
    "groq": ProviderPreset(
        "https://api.groq.com/openai/v1", "GROQ_API_KEY", "https://console.groq.com/keys"
    ),
    "cerebras": ProviderPreset(
        "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "https://cloud.cerebras.ai"
    ),
    "openrouter": ProviderPreset(
        "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "https://openrouter.ai/keys"
    ),
    "ollama": ProviderPreset("http://localhost:11434/v1", None, "https://ollama.com/download"),
}


def _resolve_api_key(settings: Settings, preset: ProviderPreset | None) -> str:
    # A blank T2S_LLM_API_KEY= line (easy to leave behind in .env) must not hide the
    # provider's own key variable.
    explicit = settings.llm_api_key.get_secret_value().strip() if settings.llm_api_key else ""
    if explicit:
        return explicit
    if preset is None or preset.key_env is None:
        return "not-needed"  # the OpenAI SDK insists on some value
    key = os.environ.get(preset.key_env, "").strip()
    if not key:
        raise ValueError(
            f"No API key for provider '{settings.llm_provider}'. Set {preset.key_env} "
            f"(or T2S_LLM_API_KEY) in your .env file. Get a free key at {preset.signup_url}."
        )
    return key


def create_llm(settings: Settings, role: Role = "sql", model: str | None = None) -> LLM:
    """Create the LLM for a pipeline role.

    ``sql`` is used for SQL generation and repair (accuracy matters most); ``helper`` is
    used for table descriptions, table selection and the final analysis, where a smaller
    and faster model is usually good enough. ``model`` overrides the configured name.
    """
    name = model or (settings.sql_model if role == "sql" else settings.helper_model)
    provider = settings.llm_provider.lower()

    if provider == "fake":
        return FakeLLM(model=name)

    preset = PRESETS.get(provider)
    if preset is None and provider != "openai_compatible":
        known = ", ".join([*PRESETS, "openai_compatible", "fake"])
        raise ValueError(f"Unknown LLM provider '{provider}'. Choose one of: {known}.")

    base_url = settings.llm_base_url or (preset.base_url if preset else None)
    if not base_url:
        raise ValueError("T2S_LLM_BASE_URL is required for provider 'openai_compatible'.")

    effort = settings.sql_reasoning_effort if role == "sql" else settings.helper_reasoning_effort
    return OpenAICompatibleLLM(
        provider=provider,
        model=name,
        base_url=base_url,
        api_key=_resolve_api_key(settings, preset),
        timeout_s=settings.llm_timeout_s,
        reasoning_effort=effort,
    )
