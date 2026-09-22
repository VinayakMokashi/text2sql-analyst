import pytest

from text2sql.config import Settings
from text2sql.llm import FakeLLM, create_llm
from text2sql.llm.openai_compat import OpenAICompatibleLLM


def test_fake_llm_replays_scripted_replies_and_records_calls():
    llm = FakeLLM(["first", "second"])
    assert llm.complete("sys", "a").text == "first"
    assert llm.complete("sys", "b").text == "second"
    assert llm.complete("sys", "c").text == "second"  # last reply repeats
    assert [user for _, user in llm.calls] == ["a", "b", "c"]


def test_reasoning_blocks_are_stripped():
    llm = FakeLLM(["<think>let me reason...</think>\nSELECT 1"])
    assert llm.complete("s", "u").text == "SELECT 1"


def test_factory_uses_role_specific_models():
    settings = Settings(llm_provider="fake", sql_model="big", helper_model="small")
    assert create_llm(settings, "sql").model == "big"
    assert create_llm(settings, "helper").model == "small"
    assert create_llm(settings, "sql", model="override").model == "override"


def test_groq_preset_reads_key_from_provider_variable(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    llm = create_llm(Settings(llm_provider="groq", llm_api_key=None))
    assert isinstance(llm, OpenAICompatibleLLM)
    assert str(llm._client.base_url).startswith("https://api.groq.com/openai/v1")


def test_missing_key_gives_actionable_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        create_llm(Settings(llm_provider="groq", llm_api_key=None))


def test_ollama_needs_no_key():
    llm = create_llm(Settings(llm_provider="ollama", llm_api_key=None))
    assert str(llm._client.base_url).startswith("http://localhost:11434")


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_llm(Settings(llm_provider="nope"))


# ---------------------------------------------------------------- review fixes
@pytest.mark.parametrize(
    "reply, expected",
    [
        (
            "<think>draft ```sql\nSELECT 1\n```</think>```sql\nSELECT 2\n```",
            "```sql\nSELECT 2\n```",
        ),
        # starts inside the reasoning block: only the closing tag is present
        ("draft ```sql\nSELECT Name FROM Genre\n```\n</think>\nSELECT real", "SELECT real"),
        # cut off while still reasoning: the block never closes
        ("<think>\nOkay, let me look at the schema", ""),
    ],
)
def test_unpaired_reasoning_tags_are_removed(reply, expected):
    assert FakeLLM([reply]).complete("s", "u").text == expected


def test_blank_api_key_setting_does_not_hide_the_provider_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "real-key")
    monkeypatch.setenv("T2S_LLM_API_KEY", "")
    llm = create_llm(Settings(llm_provider="groq"))
    assert llm._client.api_key == "real-key"
