"""Command-line interface.

python -m text2sql index              # build the table index (once per database)
python -m text2sql ask "question"     # answer a question
python -m text2sql tables             # show indexed tables and their descriptions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from text2sql.config import Settings
from text2sql.db import read_schema
from text2sql.indexing import build_index, load_descriptions
from text2sql.llm import LLMError, create_llm
from text2sql.pipeline import Pipeline, PipelineResult, load_embedder

console = Console()


def _settings(args: argparse.Namespace) -> Settings:
    """Command-line flags override .env values."""
    overrides = {
        "db_path": args.db,
        "llm_provider": args.provider,
        "sql_model": args.sql_model,
        "helper_model": args.helper_model,
    }
    return Settings(**{k: v for k, v in overrides.items() if v is not None})


# ------------------------------------------------------------------------ commands
def cmd_index(args: argparse.Namespace) -> int:
    s = _settings(args)
    tables = read_schema(s.db_path, s.sample_rows)
    llm = None if args.no_llm else create_llm(s, "helper")
    console.print(
        f"Indexing [bold]{len(tables)}[/] tables from {s.db_path} "
        f"({'template descriptions' if llm is None else f'descriptions by {llm.model}'})"
    )
    with console.status("Loading embedding model..."):
        embedder = load_embedder(s)
    indexed = build_index(
        tables, s.db_index_dir, embedder, llm,
        refresh_descriptions=args.refresh,
        on_progress=lambda msg: console.print(f"  {msg}"),
    )  # fmt: skip
    console.print(f"[green]Done.[/] {len(indexed)} tables indexed into {s.db_index_dir}")
    return 0


def cmd_tables(args: argparse.Namespace) -> int:
    s = _settings(args)
    docs = load_descriptions(s.db_index_dir)
    table = Table("Table", "Rows", "Description", show_lines=True)
    for t in read_schema(s.db_path, sample_rows=0):
        table.add_row(t.name, f"{t.row_count:,}", docs.get(t.name, "[dim](not indexed)[/]"))
    console.print(table)
    return 0


def render(out: PipelineResult, show_sql: bool = True, max_rows: int = 15) -> None:
    """Pretty-print a result: answer first, details after."""
    if out.status == "unanswerable":
        console.print(Panel(out.message, title="Can't answer that from this data", style="yellow"))
        return
    if out.status == "error":
        console.print(Panel(out.message, title="Something went wrong", style="red"))
        if out.sql and show_sql:
            console.print(Syntax(out.sql, "sql", word_wrap=True))
        return

    a = out.analysis
    if a is not None:
        body = a.answer
        if a.insights:
            body += "\n\n" + "\n".join(f"- {i}" for i in a.insights)
        if a.caveats:
            body += "\n\n*Caveats:* " + " ".join(a.caveats)
        console.print(Panel(Markdown(body), title="Answer", border_style="green"))

    res = out.result
    if res is not None and res.columns:
        table = Table(*res.columns, header_style="bold")
        for row in res.rows[:max_rows]:
            table.add_row(*("NULL" if v is None else str(v) for v in row))
        console.print(table)
        more = res.row_count - max_rows
        if more > 0 or res.truncated:
            console.print(
                f"[dim]... {res.row_count} rows shown in total"
                f"{' (truncated at the row limit)' if res.truncated else ''}[/]"
            )

    if show_sql and out.sql:
        console.print(Syntax(out.sql, "sql", word_wrap=True, theme="ansi_dark"))
    fixes = len(out.attempts) - 1
    console.print(
        f"[dim]tables: {', '.join(out.tables_used)} | "
        f"{'self-corrected ' + str(fixes) + 'x | ' if fixes else ''}"
        f"{out.total_s:.1f}s[/]"
    )


def cmd_ask(args: argparse.Namespace) -> int:
    s = _settings(args)
    pipeline = Pipeline.from_settings(s)
    with console.status("Thinking..."):
        out = pipeline.ask(" ".join(args.question))
    render(out, show_sql=not args.no_sql)
    return 0 if out.status != "error" else 1


# --------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="text2sql", description=__doc__.splitlines()[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", type=Path, help="SQLite file (default: T2S_DB_PATH)")
    common.add_argument("--provider", help="LLM provider (default: T2S_LLM_PROVIDER)")
    common.add_argument("--sql-model", help="model for SQL generation")
    common.add_argument("--helper-model", help="model for selection/analysis/descriptions")

    sub = parser.add_subparsers(dest="command", required=True)
    p_index = sub.add_parser("index", parents=[common], help="build the table index")
    p_index.add_argument("--refresh", action="store_true", help="regenerate descriptions")
    p_index.add_argument("--no-llm", action="store_true", help="template descriptions only")
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", parents=[common], help="answer a question")
    p_ask.add_argument("question", nargs="+")
    p_ask.add_argument("--no-sql", action="store_true", help="hide the SQL")
    p_ask.set_defaults(func=cmd_ask)

    p_tables = sub.add_parser("tables", parents=[common], help="list indexed tables")
    p_tables.set_defaults(func=cmd_tables)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError, LLMError) as exc:
        # Configuration problems (missing DB, key or index) get one clear line.
        console.print(f"[red]Error:[/] {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
