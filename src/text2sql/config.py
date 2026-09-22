"""Application settings, loaded from environment variables and a local ``.env`` file.

Everything that a user might want to change without touching code (provider, models,
paths, limits) lives here. The one exception is the provider's own API-key variable
(``GROQ_API_KEY`` and friends), which the LLM factory reads by name.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values, find_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# What load_env_file() copied into os.environ, so a later call can update those values
# without ever overriding a variable that was set outside .env.
_from_env_file: dict[str, str] = {}


def load_env_file() -> None:
    """Copy ``.env`` into ``os.environ``; variables set outside the file always win.

    Provider key variables such as GROQ_API_KEY have no T2S_ prefix, so the LLM factory
    reads them from os.environ. ``get_settings`` calls this each time it rebuilds the
    settings, so an edited file (a new key, another model) takes effect on reload.
    The search starts in the working directory, like the relative paths below.
    """
    path = find_dotenv(usecwd=True)
    values = {k: v for k, v in (dotenv_values(path) if path else {}).items() if v is not None}
    for key, value in list(_from_env_file.items()):
        if key not in values and os.environ.get(key) == value:
            del os.environ[key]  # the line was removed from .env
            del _from_env_file[key]
    for key, value in values.items():
        current = os.environ.get(key)
        if current is not None and _from_env_file.get(key) != current:
            continue  # set by the shell or the host (say, Streamlit secrets)
        os.environ[key] = value
        _from_env_file[key] = value


load_env_file()


class Settings(BaseSettings):
    """All runtime configuration. Every field can be set as ``T2S_<FIELD_NAME>``."""

    model_config = SettingsConfigDict(
        env_prefix="T2S_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # A blank line such as "T2S_SQL_MODEL=" means "use the default", not "".
        env_ignore_empty=True,
    )

    # --- LLM -----------------------------------------------------------------------
    llm_provider: str = Field(
        "groq",
        description="groq | cerebras | openrouter | ollama | openai_compatible | fake",
    )
    llm_base_url: str | None = Field(None, description="Override the provider's endpoint.")
    llm_api_key: SecretStr | None = Field(None, description="Overrides provider key vars.")
    sql_model: str = "openai/gpt-oss-120b"
    helper_model: str = "qwen/qwen3.8-27b"
    # Reasoning models (gpt-oss, Qwen3) accept low/medium/high; empty = model default.
    sql_reasoning_effort: str | None = None
    helper_reasoning_effort: str | None = None
    llm_timeout_s: float = 60.0

    # --- Data & index -----------------------------------------------------------------
    db_path: Path = Path("data/chinook.db")
    index_dir: Path = Path("data/index")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    model_cache_dir: Path = Path("data/models")
    sql_dialect: str = "sqlite"

    # --- Pipeline knobs -----------------------------------------------------------------
    top_n_tables: int = Field(6, ge=1, description="Candidates returned by vector search.")
    # 6, not 4: multi-hop questions on richer schemas (e.g. Sakila's payment -> rental ->
    # inventory -> film_category -> category) need five tables.
    top_k_tables: int = Field(6, ge=1, description="Tables kept after LLM selection.")
    sample_rows: int = Field(3, ge=0, description="Sample rows shown per table in prompts.")
    max_rows: int = Field(200, ge=1, description="Maximum rows returned by a query.")
    query_timeout_s: float = Field(10.0, gt=0)
    max_retries: int = Field(2, ge=0, description="SQL self-correction retries.")
    analysis_rows: int = Field(40, ge=1, description="Result rows shown to the analysis LLM.")

    # --- Public demo (unset = unlimited) -------------------------------------------------
    demo_daily_limit: int | None = Field(None, ge=0, description="Questions per day, all users.")
    demo_session_limit: int | None = Field(None, ge=0, description="Questions per visitor.")

    @property
    def db_index_dir(self) -> Path:
        """Per-database index folder, so several databases can be indexed side by side."""
        return self.index_dir / self.db_path.stem


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings. ``get_settings.cache_clear()`` re-reads ``.env``."""
    load_env_file()
    return Settings()
