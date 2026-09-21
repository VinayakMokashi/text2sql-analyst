"""Evaluate the pipeline on eval/questions.jsonl and compare SQL models.

For each SQL model it reports:
  * execution accuracy (EX) on answerable questions, overall and by difficulty
  * abstention: unanswerable questions correctly declined, and answerable ones wrongly
    declined
  * retrieval recall: share of gold tables in the vector-search candidates (@N) and
    in the final schema given to the SQL model (after selection + join expansion)
  * average SQL-stage latency (generation + execution + repairs) and how often
    self-correction was needed

Table selection (helper model) runs once per question and its result is shared by all
SQL models, so every model sees exactly the same schema and differences come from SQL
generation alone. The analysis step is skipped because it does not affect EX.

Usage:
    python eval/run_eval.py                                   # model from .env
    python eval/run_eval.py --models openai/gpt-oss-120b qwen/qwen3.8-27b
    python eval/run_eval.py --limit 5 --sleep 0               # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from text2sql.config import get_settings
from text2sql.evaluation import recall, results_match, tables_in_sql
from text2sql.execution import execute_query
from text2sql.generation import SQLGenerator
from text2sql.llm import create_llm
from text2sql.pipeline import Pipeline, load_embedder
from text2sql.retrieval import TableSelection, TableSelector

EVAL_DIR = Path(__file__).parent
DIFFICULTIES = ["easy", "medium", "hard"]


def load_questions(path: Path, ids: list[str] | None, limit: int | None) -> list[dict[str, Any]]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if ids:
        items = [q for q in items if q["id"] in ids]
    return items[:limit] if limit else items


def gold_variants(item: dict[str, Any]) -> list[str]:
    gold = item.get("gold_sql")
    return [] if gold is None else [gold] if isinstance(gold, str) else list(gold)


def slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9.-]+", "_", model)


class CachedSelector:
    """Memoises table selection per question so all SQL models get the same tables."""

    def __init__(self, inner: TableSelector) -> None:
        self._inner = inner
        self._cache: dict[str, TableSelection] = {}

    def select(self, question: str, *args: Any, **kwargs: Any) -> TableSelection:
        if question not in self._cache:
            self._cache[question] = self._inner.select(question, *args, **kwargs)
        return self._cache[question]


def evaluate_model(
    pipe: Pipeline, model: str, questions: list[dict[str, Any]], gold_rows: dict[str, list], args
) -> list[dict[str, Any]]:
    s = pipe.settings
    pipe.generator = SQLGenerator(create_llm(s, "sql", model), s.sql_dialect, s.sample_rows)
    records = []
    for n, item in enumerate(questions, 1):
        out = pipe.ask(item["question"], analyze=False)
        variants = gold_variants(item)
        answerable = bool(variants)

        if answerable:
            correct = out.ok and any(
                results_match(rows, out.result.rows) for rows in gold_rows[item["id"]]
            )
            gold_tables = tables_in_sql(variants[0])
        else:
            correct = out.status == "unanswerable"
            gold_tables = set()

        record = {
            "id": item["id"],
            "difficulty": item["difficulty"],
            "question": item["question"],
            "answerable": answerable,
            "correct": bool(correct),
            "status": out.status,
            "message": out.message,
            "sql": out.sql,
            "attempts": len(out.attempts),
            "repaired": len(out.attempts) > 1 and out.ok,
            "llm_error": out.message.startswith("The language model request failed"),
            "latency_s": round(out.total_s, 3),
            # Model time for every SQL attempt plus query execution. Excludes time spent
            # waiting out free-tier rate limits, which says nothing about the model.
            "sql_latency_s": round(
                sum(a.latency_s for a in out.attempts)
                + (out.result.elapsed_s if out.result else 0.0),
                3,
            ),
            "candidates": [c.name for c in out.candidates],
            "tables_used": out.tables_used,
            "recall_at_n": recall(gold_tables, [c.name for c in out.candidates]),
            "schema_recall": recall(gold_tables, out.tables_used),
        }
        records.append(record)
        mark = "PASS" if correct else "FAIL"
        print(
            f"  [{n:>2}/{len(questions)}] {mark} {item['id']} {out.total_s:5.1f}s "
            f"{item['question'][:60]}"
        )
        if args.sleep:
            time.sleep(args.sleep)  # stay under free-tier requests-per-minute limits
    return records


def summarize(model: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    def pct(rows: list[dict[str, Any]], key: str = "correct") -> float | None:
        return round(100 * sum(bool(r[key]) for r in rows) / len(rows), 1) if rows else None

    by_diff: dict[str, list] = defaultdict(list)
    for r in answerable:
        by_diff[r["difficulty"]].append(r)
    # Only questions that reached SQL generation (declined ones stop earlier).
    latencies = [r["sql_latency_s"] for r in records if r["attempts"] and not r["llm_error"]]
    return {
        "model": model,
        "questions": len(records),
        "ex": pct(answerable),
        "ex_by_difficulty": {d: pct(by_diff[d]) for d in DIFFICULTIES if by_diff[d]},
        "declined_correctly": f"{sum(r['correct'] for r in unanswerable)}/{len(unanswerable)}",
        "false_refusals": sum(r["status"] == "unanswerable" for r in answerable),
        "recall_at_n": round(100 * statistics.mean(r["recall_at_n"] for r in answerable), 1)
        if answerable
        else None,
        "schema_recall": round(100 * statistics.mean(r["schema_recall"] for r in answerable), 1)
        if answerable
        else None,
        "avg_latency_s": round(statistics.mean(latencies), 2) if latencies else None,
        "median_latency_s": round(statistics.median(latencies), 2) if latencies else None,
        "self_corrected": sum(r["repaired"] for r in records),
        "llm_errors": sum(r["llm_error"] for r in records),
    }


def markdown_table(summaries: list[dict[str, Any]], top_n: int) -> str:
    header = (
        "| SQL model | EX (all) | Easy | Medium | Hard | Declined unanswerable | "
        f"False refusals | Table recall@{top_n} | Final schema recall | Avg SQL latency | "
        "Self-corrected |\n|---|---|---|---|---|---|---|---|---|---|---|"
    )

    def p(value: float | None) -> str:
        return "-" if value is None else f"{value}%"

    rows = []
    for s in summaries:
        d = s["ex_by_difficulty"]
        rows.append(
            f"| `{s['model']}` | **{p(s['ex'])}** | {p(d.get('easy'))} | "
            f"{p(d.get('medium'))} | {p(d.get('hard'))} | {s['declined_correctly']} | "
            f"{s['false_refusals']} | {p(s['recall_at_n'])} | {p(s['schema_recall'])} | "
            f"{s['avg_latency_s']}s | {s['self_corrected']} |"
        )
    return "\n".join([header, *rows])


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate SQL models on the question set.")
    parser.add_argument("--models", nargs="+", default=[settings.sql_model])
    parser.add_argument("--helper-model", default=None, help="default: T2S_HELPER_MODEL")
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.jsonl")
    parser.add_argument("--ids", nargs="*", help="only these question ids")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--sleep", type=float, default=1.0, help="pause between questions (s)")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "results")
    args = parser.parse_args()

    questions = load_questions(args.questions, args.ids, args.limit)
    print(
        f"{len(questions)} questions | provider {settings.llm_provider} | helper model "
        f"{args.helper_model or settings.helper_model} | db {settings.db_path}"
    )

    # Run every gold query once up front; a broken gold query should fail loudly.
    gold_rows = {
        q["id"]: [
            execute_query(settings.db_path, sql, max_rows=10_000).rows for sql in gold_variants(q)
        ]
        for q in questions
    }
    pipe = Pipeline.from_settings(
        settings, helper_model=args.helper_model, embedder=load_embedder(settings)
    )
    pipe.selector = CachedSelector(pipe.selector)  # type: ignore[assignment]
    args.out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for model in args.models:
        print(f"\n=== {model} ===")
        records = evaluate_model(pipe, model, questions, gold_rows, args)
        with (args.out / f"{slug(model)}.jsonl").open("w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in records)
        summaries.append(summarize(model, records))

    table = markdown_table(summaries, settings.top_n_tables)
    note = (
        f"\n\nProvider: `{settings.llm_provider}`, helper model: "
        f"`{args.helper_model or settings.helper_model}`, {len(questions)} questions, "
        f"run on {time.strftime('%Y-%m-%d')}.\n"
    )
    (args.out / "summary.md").write_text(table + note, encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("\n" + table + note)
    if any(s["llm_errors"] for s in summaries):
        print("Warning: some questions failed because of LLM/provider errors (see *.jsonl).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
