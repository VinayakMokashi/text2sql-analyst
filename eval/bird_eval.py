"""Run the pipeline on the BIRD mini-dev subset and score it (see bird_prepare.py).

Each question is asked together with BIRD's evidence hint, as in the benchmark's
standard setting. Two scores are reported:
  * strict EX, BIRD's official definition: the set of result rows must equal the
    gold set exactly (row order and duplicates ignored, values compared as-is);
  * lenient EX, this project's matcher (extra columns allowed, numbers rounded to
    2 decimals), for comparison with the other question sets.

Results are appended to the log one question at a time, and a rerun skips questions
already answered, so a large run can be spread over several days of free-tier quota:

    python eval/bird_eval.py --limit 50          # today
    python eval/bird_eval.py --limit 50          # tomorrow: continues where it stopped
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from bird_prepare import DATA_DIR, SUBSET_FILE
from run_eval import error_kind, read_log, slug

from text2sql.config import get_settings
from text2sql.db import read_schema
from text2sql.demo import index_ready
from text2sql.evaluation import results_match
from text2sql.execution import execute_query
from text2sql.indexing import build_index
from text2sql.llm import LLMError, create_llm
from text2sql.pipeline import Pipeline, load_embedder

OUT_DIR = Path(__file__).parent / "results" / "bird"
DIFFICULTIES = ["simple", "moderate", "challenging"]
GOLD_TIMEOUT_S = 120.0  # some gold queries scan large tables
GOLD_ROW_LIMIT = 100_000


def latest_records(log: Path) -> dict[int, dict[str, Any]]:
    """The newest record per question (a retried question is logged again)."""
    return {r["question_id"]: r for r in read_log(log)} if log.exists() else {}


def question_text(q: dict[str, Any]) -> str:
    hint = q.get("evidence", "").strip()
    return f"{q['question']}\nHint: {hint}" if hint else q["question"]


def summarize(model: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    excluded = [r for r in records if r.get("error_kind")]
    scored = [r for r in records if not r.get("error_kind")]

    def pct(rows: list[dict[str, Any]], key: str) -> float | None:
        return round(100 * sum(bool(r[key]) for r in rows) / len(rows), 1) if rows else None

    by_diff: dict[str, list] = defaultdict(list)
    by_db: dict[str, list] = defaultdict(list)
    for r in scored:
        by_diff[r["difficulty"]].append(r)
        by_db[r["db_id"]].append(r)
    latencies = [r["sql_latency_s"] for r in scored if r["attempts"]]
    return {
        "model": model,
        "answered": len(records),
        "strict_ex": pct(scored, "strict"),
        "lenient_ex": pct(scored, "lenient"),
        "strict_by_difficulty": {d: pct(by_diff[d], "strict") for d in DIFFICULTIES},
        "count_by_difficulty": {d: len(by_diff[d]) for d in DIFFICULTIES},
        "strict_by_db": {db: pct(rows, "strict") for db, rows in sorted(by_db.items())},
        "declined": sum(r["status"] == "unanswerable" for r in scored),
        "excluded": len(excluded),
        "avg_sql_latency_s": round(statistics.mean(latencies), 2) if latencies else None,
    }


def markdown(summary: dict[str, Any], total: int) -> str:
    def p(v: float | None) -> str:
        return "-" if v is None else f"{v}%"

    s, d, c = summary, summary["strict_by_difficulty"], summary["count_by_difficulty"]
    lines = [
        "| SQL model | Answered | Strict EX (BIRD) | Lenient EX | Simple | Moderate | "
        "Challenging | Declined | Avg SQL latency |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| `{s['model']}` | {s['answered']}/{total} | **{p(s['strict_ex'])}** | "
        f"{p(s['lenient_ex'])} | {p(d['simple'])} ({c['simple']}) | "
        f"{p(d['moderate'])} ({c['moderate']}) | {p(d['challenging'])} ({c['challenging']}) | "
        f"{s['declined']} | {s['avg_sql_latency_s']}s |",
        "",
        "Strict EX by database: "
        + ", ".join(f"{db} {p(v)}" for db, v in s["strict_by_db"].items()),
    ]
    if s["excluded"]:
        lines.append(f"\n{s['excluded']} question(s) left out because of provider errors.")
    return "\n".join(lines) + "\n"


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate on the BIRD mini-dev subset.")
    parser.add_argument("--model", default=settings.sql_model, help="SQL model")
    parser.add_argument("--helper-model", default=None, help="default: T2S_HELPER_MODEL")
    parser.add_argument("--limit", type=int, help="answer at most N new questions this run")
    parser.add_argument("--sleep", type=float, default=0.0, help="pause between questions (s)")
    args = parser.parse_args()

    subset = json.loads(SUBSET_FILE.read_text(encoding="utf-8"))
    bird = json.loads((DATA_DIR / "mini_dev_sqlite.json").read_text(encoding="utf-8"))
    by_id = {q["question_id"]: q for q in bird}
    questions = [by_id[i] for i in subset["question_ids"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = OUT_DIR / f"{slug(args.model)}.jsonl"
    # A question that failed because of the provider (say, the daily quota ran out) is
    # asked again on the next run; its newest record replaces the old one.
    done = {qid for qid, r in latest_records(log).items() if not r.get("error_kind")}
    todo = [q for q in questions if q["question_id"] not in done][: args.limit]
    print(f"{len(done)} of {len(questions)} already answered; answering {len(todo)} now")

    embedder = load_embedder(settings)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in todo:
        groups[q["db_id"]].append(q)
    for db, items in sorted(groups.items()):
        s = settings.model_copy(
            update={"db_path": DATA_DIR / f"{db}.sqlite", "query_timeout_s": 30.0}
        )
        if not index_ready(s):
            print(f"Indexing {db}")
            helper = create_llm(s, "helper", args.helper_model)
            tables = read_schema(s.db_path, s.sample_rows)
            try:
                build_index(tables, s.db_index_dir, embedder, helper, db_path=s.db_path)
            except LLMError as exc:
                # Finished descriptions are kept; the next run picks up from there.
                print(f"  skipping {db} for now: {exc}")
                continue
        gold = {
            q["question_id"]: execute_query(
                s.db_path, q["SQL"], max_rows=GOLD_ROW_LIMIT, timeout_s=GOLD_TIMEOUT_S
            ).rows
            for q in items
        }
        biggest = max(len(rows) for rows in gold.values())
        s = s.model_copy(update={"max_rows": max(s.max_rows, biggest + 1)})
        pipe = Pipeline.from_settings(
            s, sql_model=args.model, helper_model=args.helper_model, embedder=embedder
        )
        for q in items:
            out = pipe.ask(question_text(q), analyze=False)
            rows = out.result.rows if out.ok and not out.result.truncated else None
            gold_rows = gold[q["question_id"]]
            record = {
                "model": args.model,
                "question_id": q["question_id"],
                "db_id": db,
                "difficulty": q["difficulty"],
                "question": q["question"],
                "evidence": q.get("evidence", ""),
                "gold_sql": q["SQL"],
                "sql": out.sql,
                "status": out.status,
                "message": out.message,
                "error_kind": error_kind(out),
                "strict": rows is not None and set(rows) == set(gold_rows),
                "lenient": rows is not None and results_match(gold_rows, rows),
                "attempts": len(out.attempts),
                "sql_latency_s": round(
                    sum(a.latency_s for a in out.attempts)
                    + (out.result.elapsed_s if out.result else 0.0),
                    3,
                ),
                "tables_used": out.tables_used,
            }
            with log.open("a", encoding="utf-8") as fh:  # appended at once: runs can resume
                fh.write(json.dumps(record) + "\n")
            mark = "PASS" if record["strict"] else "pass*" if record["lenient"] else "FAIL"
            if record["error_kind"]:
                mark = f"ERROR ({record['error_kind']})"
            print(f"  {mark:<12} {q['question_id']:>4} {db:<24} {q['question'][:60]}")
            if args.sleep:
                time.sleep(args.sleep)

    records = list(latest_records(log).values())
    if not records:
        return 0
    summary = summarize(args.model, records)
    table = markdown(summary, len(questions))
    (OUT_DIR / f"summary_{slug(args.model)}.md").write_text(table, encoding="utf-8")
    (OUT_DIR / f"summary_{slug(args.model)}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\n" + table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
