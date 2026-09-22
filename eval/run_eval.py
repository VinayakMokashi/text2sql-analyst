"""Evaluate the pipeline on a question set and compare SQL models.

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

Questions that fail because of the provider (rate limits, outages) or a harness bug are
not the SQL model's fault: they are left out of EX and reported with a warning.

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
from text2sql.pipeline import Pipeline, PipelineResult, load_embedder
from text2sql.retrieval import TableSelection, TableSelector

EVAL_DIR = Path(__file__).parent
DEFAULT_OUT = EVAL_DIR / "results"
DIFFICULTIES = ["easy", "medium", "hard"]
GOLD_ROW_LIMIT = 10_000


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


def read_log(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def error_kind(out: PipelineResult) -> str | None:
    """'llm' for provider failures, 'harness' for bugs; None when the model is to blame."""
    if out.message.startswith("The language model request failed"):
        return "llm"
    if out.message.startswith("Unexpected error"):
        return "harness"
    return None


class CachedSelector:
    """Memoises table selection per question so all SQL models get the same tables."""

    def __init__(self, inner: TableSelector) -> None:
        self._inner = inner
        self._cache: dict[str, TableSelection] = {}

    def select(self, question: str, *args: Any, **kwargs: Any) -> TableSelection:
        if question not in self._cache:
            self._cache[question] = self._inner.select(question, *args, **kwargs)
        return self._cache[question]


class ReplaySelector:
    """Replays the table selection of an earlier run instead of calling the helper model.

    Adding a model to an existing comparison then costs one LLM call per question
    instead of two, and the new model sees exactly the same tables as the old ones.
    """

    def __init__(self, log: Path) -> None:
        self._source = log.name
        self._selections: dict[str, TableSelection] = {}
        for r in read_log(log):
            if "selection_answerable" in r:  # logs that record the selection explicitly
                if r["selection_answerable"] is None:
                    continue  # the selection never completed (e.g. a provider error)
                tables, answerable = r["selection_tables"], r["selection_answerable"]
            else:  # older logs: an empty table list is ambiguous if the question failed
                if not r["tables_used"] and (r.get("llm_error") or r["status"] == "error"):
                    continue
                tables, answerable = r["tables_used"], bool(r["tables_used"])
            self._selections[r["question"]] = TableSelection(
                tables, answerable=answerable, reason=f"replayed from {self._source}"
            )

    def missing(self, questions: list[dict[str, Any]]) -> list[str]:
        return [q["id"] for q in questions if q["question"] not in self._selections]

    def select(self, question: str, *_args: Any, **_kwargs: Any) -> TableSelection:
        return self._selections[question]


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
            correct = (
                out.ok
                and not out.result.truncated  # a cut-off result cannot be verified
                and any(results_match(rows, out.result.rows) for rows in gold_rows[item["id"]])
            )
            gold_tables = tables_in_sql(variants[0])
        else:
            correct = out.status == "unanswerable"
            gold_tables = set()

        kind = error_kind(out)
        record = {
            "model": model,
            "provider": s.llm_provider,
            "helper_model": args.helper_model or s.helper_model,
            "questions_file": args.questions.name,
            "id": item["id"],
            "difficulty": item["difficulty"],
            "question": item["question"],
            "answerable": answerable,
            "correct": bool(correct),
            "status": out.status,
            "message": out.message,
            "error_kind": kind,
            "llm_error": kind == "llm",
            "sql": out.sql,
            "attempts": len(out.attempts),
            "repaired": len(out.attempts) > 1 and out.ok,
            "latency_s": round(out.total_s, 3),
            # Model time for every SQL attempt plus query execution. Excludes time spent
            # waiting out free-tier rate limits, which says nothing about the model.
            "sql_latency_s": round(
                sum(a.latency_s for a in out.attempts)
                + (out.result.elapsed_s if out.result else 0.0),
                3,
            ),
            "candidates": [c.name for c in out.candidates],
            # The selection itself (None if it never completed), so a later run can
            # replay it without guessing from tables_used.
            "selection_tables": out.selection.tables if out.selection else None,
            "selection_answerable": out.selection.answerable if out.selection else None,
            "tables_used": out.tables_used,
            "recall_at_n": recall(gold_tables, [c.name for c in out.candidates]),
            "schema_recall": recall(gold_tables, out.tables_used),
        }
        records.append(record)
        mark = "PASS" if correct else f"ERROR ({kind})" if kind else "FAIL"
        print(
            f"  [{n:>2}/{len(questions)}] {mark} {item['id']} {out.total_s:5.1f}s "
            f"{item['question'][:60]}"
        )
        if args.sleep:
            time.sleep(args.sleep)  # stay under free-tier requests-per-minute limits
    return records


def summarize(model: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    excluded = [r for r in records if r.get("error_kind") or r.get("llm_error")]
    scored = [r for r in records if r not in excluded]
    answerable = [r for r in scored if r["answerable"]]
    unanswerable = [r for r in scored if not r["answerable"]]

    def pct(rows: list[dict[str, Any]], key: str = "correct") -> float | None:
        return round(100 * sum(bool(r[key]) for r in rows) / len(rows), 1) if rows else None

    by_diff: dict[str, list] = defaultdict(list)
    for r in answerable:
        by_diff[r["difficulty"]].append(r)
    # Only questions that reached SQL generation (declined ones stop earlier).
    latencies = [r["sql_latency_s"] for r in scored if r["attempts"]]
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
        "self_corrected": sum(r["repaired"] for r in scored),
        "excluded": len(excluded),
        "llm_errors": sum(bool(r.get("llm_error")) for r in records),
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
        latency = "-" if s["avg_latency_s"] is None else f"{s['avg_latency_s']}s"
        rows.append(
            f"| `{s['model']}` | **{p(s['ex'])}** | {p(d.get('easy'))} | "
            f"{p(d.get('medium'))} | {p(d.get('hard'))} | {s['declined_correctly']} | "
            f"{s['false_refusals']} | {p(s['recall_at_n'])} | {p(s['schema_recall'])} | "
            f"{latency} | {s['self_corrected']} |"
        )
    return "\n".join([header, *rows])


def build_summary(out_dir: Path, question_ids: set[str], top_n: int) -> tuple[str, list]:
    """Summarise every log in ``out_dir`` that covers exactly this run's questions.

    Models can be added to a comparison one run at a time; logs over a different
    question set (say, a partial run) are left out rather than mixed in.
    """
    summaries, skipped, providers, helpers = [], [], set(), set()
    for log in sorted(out_dir.glob("*.jsonl")):
        records = read_log(log)
        if not records or {r["id"] for r in records} != question_ids:
            skipped.append(log.name)
            continue
        model = records[0].get("model") or log.stem.replace("_", "/", 1)  # older logs
        providers.update(r["provider"] for r in records if "provider" in r)
        helpers.update(r["helper_model"] for r in records if "helper_model" in r)
        summaries.append(summarize(model, records))

    note = (
        f"\n\nProvider: {', '.join(f'`{p}`' for p in sorted(providers)) or 'see logs'}; "
        f"helper model (table selection): "
        f"{', '.join(f'`{h}`' for h in sorted(helpers)) or 'see logs'}; "
        f"{len(question_ids)} questions; updated {time.strftime('%Y-%m-%d')}.\n"
    )
    excluded = sum(s["excluded"] for s in summaries)
    if excluded:
        note += (
            f"\n{excluded} question run(s) were left out of the scores because of provider "
            "or harness errors; see the logs.\n"
        )
    if skipped:
        print(f"Not in this summary (different question set or empty): {', '.join(skipped)}")
    return markdown_table(summaries, top_n) + note, summaries


def default_out(questions: Path) -> Path:
    """results/ for the dev set, results/<set name>/ for any other question set.

    So a run on another set never lands on the dev set's published logs:
    heldout.jsonl -> results/heldout, sakila_questions.jsonl -> results/sakila.
    """
    name = questions.stem.removesuffix("_questions")
    return DEFAULT_OUT if name == "questions" else DEFAULT_OUT / name


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate SQL models on the question set.")
    parser.add_argument("--models", nargs="+", default=[settings.sql_model])
    parser.add_argument("--helper-model", default=None, help="default: T2S_HELPER_MODEL")
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.jsonl")
    parser.add_argument("--ids", nargs="*", help="only these question ids")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--sleep", type=float, default=1.0, help="pause between questions (s)")
    parser.add_argument(
        "--out", type=Path, help="default: results/ for the dev set, else results/<set name>/"
    )
    parser.add_argument(
        "--reuse-selection",
        type=Path,
        metavar="LOG.jsonl",
        help="replay table selection from an earlier run's log (half the LLM calls)",
    )
    args = parser.parse_args()

    args.out = args.out or default_out(args.questions)
    questions = load_questions(args.questions, args.ids, args.limit)
    if not questions:
        parser.error("no questions match --ids/--limit")
    if args.ids or args.limit:
        # A partial run must never overwrite the full, published logs, whatever --out is.
        args.out = args.out / "partial"
        print(f"Partial run: writing to {args.out}")
    print(
        f"{len(questions)} questions | provider {settings.llm_provider} | helper model "
        f"{args.helper_model or settings.helper_model} | db {settings.db_path}"
    )

    # Run every gold query once up front; a broken gold query should fail loudly.
    gold_rows: dict[str, list] = {}
    for q in questions:
        results = [
            execute_query(settings.db_path, sql, max_rows=GOLD_ROW_LIMIT)
            for sql in gold_variants(q)
        ]
        if any(r.truncated for r in results):
            sys.exit(f"Gold result for {q['id']} has more than {GOLD_ROW_LIMIT} rows.")
        gold_rows[q["id"]] = [r.rows for r in results]
    # Let predictions return as many rows as the largest gold answer, so a question with
    # a big correct result can still be matched.
    biggest = max((len(rows) for v in gold_rows.values() for rows in v), default=0)
    settings = settings.model_copy(update={"max_rows": max(settings.max_rows, biggest + 1)})

    if args.reuse_selection:
        replay = ReplaySelector(args.reuse_selection)
        if missing := replay.missing(questions):
            sys.exit(
                f"{args.reuse_selection.name} has no usable table selection for: "
                f"{', '.join(missing)}. Run without --reuse-selection."
            )
    pipe = Pipeline.from_settings(
        settings, helper_model=args.helper_model, embedder=load_embedder(settings)
    )
    pipe.selector = replay if args.reuse_selection else CachedSelector(pipe.selector)  # type: ignore[assignment]
    args.out.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        print(f"\n=== {model} ===")
        records = evaluate_model(pipe, model, questions, gold_rows, args)
        with (args.out / f"{slug(model)}.jsonl").open("w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in records)

    table, summaries = build_summary(args.out, {q["id"] for q in questions}, settings.top_n_tables)
    (args.out / "summary.md").write_text(table, encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("\n" + table)
    if any(s["excluded"] for s in summaries):
        print("Warning: some questions failed because of provider or harness errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
