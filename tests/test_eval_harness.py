"""The evaluation harness's bookkeeping: summaries and selection replay."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import run_eval  # noqa: E402


def record(qid, model="m", correct=True, **extra):
    base = {
        "model": model, "provider": "p", "helper_model": "h", "id": qid, "difficulty": "easy",
        "question": f"question {qid}", "answerable": True, "correct": correct, "status": "ok",
        "message": "", "error_kind": None, "llm_error": False, "attempts": 1, "repaired": False,
        "sql_latency_s": 1.0, "recall_at_n": 1.0, "schema_recall": 1.0, "tables_used": ["T"],
    }  # fmt: skip
    return base | extra


def write_log(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_summary_only_combines_logs_over_the_same_questions(tmp_path):
    write_log(tmp_path / "full.jsonl", [record("a"), record("b")])
    write_log(tmp_path / "partial.jsonl", [record("a", model="other")])
    write_log(tmp_path / "empty.jsonl", [])
    table, summaries = run_eval.build_summary(tmp_path, {"a", "b"}, 6)
    assert [s["model"] for s in summaries] == ["m"]
    assert "2 questions" in table


def test_provider_and_harness_errors_are_left_out_of_accuracy():
    records = [
        record("a"),
        record("b", correct=False, status="error", error_kind="llm", llm_error=True),
        record("c", correct=False, status="error", error_kind="harness"),
    ]
    summary = run_eval.summarize("m", records)
    assert summary["ex"] == 100.0
    assert summary["excluded"] == 2


def test_replay_skips_selections_that_never_completed(tmp_path):
    log = tmp_path / "log.jsonl"
    write_log(
        log,
        [
            record("a", selection_tables=["T"], selection_answerable=True),
            record("b", selection_tables=[], selection_answerable=False),  # a real decline
            record("c", selection_tables=None, selection_answerable=None),  # provider error
            # An older log without the selection fields: an empty list after an error is
            # ambiguous, so it cannot be replayed either.
            record("d", status="error", llm_error=True, tables_used=[]),
        ],
    )
    replay = run_eval.ReplaySelector(log)
    questions = [{"id": q, "question": f"question {q}"} for q in "abcd"]
    assert replay.missing(questions) == ["c", "d"]
    assert replay.select("question b").answerable is False
    assert replay.select("question a").tables == ["T"]


@pytest.mark.parametrize("value, expected", [(None, "-"), (1.5, "1.5s")])
def test_latency_cell(value, expected):
    summary = run_eval.summarize("m", [record("a")]) | {"avg_latency_s": value}
    assert f"| {expected} |" in run_eval.markdown_table([summary], 6)
