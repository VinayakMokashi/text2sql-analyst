"""Settings: an edited .env takes effect on reload, and never overrides the environment."""

import os

from text2sql import config
from text2sql.config import Settings, get_settings

KEYS = ("T2S_SQL_MODEL", "T2S_HELPER_MODEL", "EXAMPLE_PROVIDER_KEY")


def test_reloading_settings_picks_up_an_edited_env_file(tmp_path, monkeypatch):
    for key in KEYS:
        monkeypatch.setenv(key, "placeholder")  # so teardown restores the real value
        monkeypatch.delenv(key)
    monkeypatch.setattr(config, "_from_env_file", {})
    monkeypatch.setenv("T2S_HELPER_MODEL", "from-the-shell")
    monkeypatch.chdir(tmp_path)
    env = tmp_path / ".env"
    try:
        env.write_text(
            "T2S_SQL_MODEL=first\nT2S_HELPER_MODEL=from-the-file\nEXAMPLE_PROVIDER_KEY=k1\n"
        )
        get_settings.cache_clear()
        assert get_settings().sql_model == "first"

        # The user fixes .env and clicks the app's reload button.
        env.write_text("T2S_SQL_MODEL=second\nEXAMPLE_PROVIDER_KEY=k2\n")
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.sql_model == "second"
        assert os.environ["EXAMPLE_PROVIDER_KEY"] == "k2"  # key variables are read from here
        assert settings.helper_model == "from-the-shell"  # set outside .env: left alone

        env.write_text("EXAMPLE_PROVIDER_KEY=k2\n")  # a removed line means the default again
        get_settings.cache_clear()
        assert get_settings().sql_model == Settings.model_fields["sql_model"].default
    finally:
        get_settings.cache_clear()
