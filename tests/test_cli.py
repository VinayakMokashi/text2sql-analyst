"""CLI output and wiring, with a recording console and no model calls."""

import pytest
from rich.console import Console

from text2sql import cli
from text2sql.execution import QueryResult
from text2sql.pipeline import PipelineResult


@pytest.fixture()
def recorded(monkeypatch):
    console = Console(record=True, width=120)
    monkeypatch.setattr(cli, "console", console)
    return console


def test_brackets_in_data_and_messages_are_printed_literally(recorded):
    declined = PipelineResult("q", status="unanswerable", message="No [salary] column [/url].")
    cli.render(declined)
    rows = [("A [b] c", "[/url]")]
    answered = PipelineResult(
        "q", result=QueryResult("SELECT 1", ["name", "note"], rows, False, 0.0)
    )
    cli.render(answered)
    text = recorded.export_text()
    assert "No [salary] column [/url]." in text
    assert "A [b] c" in text and "[/url]" in text


def test_row_count_note_is_accurate(recorded):
    rows = [(i,) for i in range(20)]
    cli.render(PipelineResult("q", result=QueryResult("SELECT 1", ["n"], rows, True, 0.0)))
    assert "showing 15 of 20 rows, cut off at the row limit" in recorded.export_text()


def test_index_records_the_database_and_accepts_force(monkeypatch, shop_db, tmp_path, recorded):
    seen = {}

    def fake_build_index(tables, index_dir, embedder, llm, **kwargs):
        seen.update(kwargs, llm=llm)
        return []

    monkeypatch.setattr(cli, "build_index", fake_build_index)
    monkeypatch.setattr(cli, "load_embedder", lambda _settings: None)
    monkeypatch.setenv("T2S_INDEX_DIR", str(tmp_path))
    assert cli.main(["index", "--db", str(shop_db), "--provider", "fake", "--force"]) == 0
    assert seen["db_path"] == shop_db and seen["force"] is True
    assert seen["llm"] is None  # the fake provider must not write "SELECT 1" descriptions


def test_configuration_errors_are_one_line_not_tracebacks(tmp_path, recorded):
    not_sqlite = tmp_path / "notes.db"
    not_sqlite.write_text("definitely not a database")
    assert cli.main(["tables", "--db", str(not_sqlite)]) == 2
    assert "Error:" in recorded.export_text()
