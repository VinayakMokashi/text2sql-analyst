"""Command-line interface.

python -m text2sql index              # build the table index (once per database)
python -m text2sql ask "question"     # answer a question
python -m text2sql chat               # a conversation; follow-up questions work
python -m text2sql tables             # show indexed tables and their descriptions
python -m text2sql models             # list the models your provider/key can use
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from text2sql.config import Settings
from text2sql.conversation import Turn
from text2sql.db import read_schema
from text2sql.indexing import build_index, load_descriptions
from text2sql.llm import LLMError, create_llm
from text2sql.llm.openai_compat import OpenAICompatibleLLM
from text2sql.pipeline import Pipeline, PipelineResult, load_embedder

console = Console()

# Rich treats [square brackets] in printed strings as style markup, so every value
# that comes from the data, a model or an exception goes through escape().


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
    # The fake provider answers every prompt with "SELECT 1"; caching that as a table
    # description would poison later runs with a real model, so use templates instead.
    use_llm = not args.no_llm and s.llm_provider.lower() != "fake"
    llm = create_llm(s, "helper") if use_llm else None
    source = "template text" if llm is None else llm.model
    console.print(
        f"Indexing [bold]{len(tables)}[/] tables from {escape(str(s.db_path))} "
        f"(new descriptions by {escape(source)}; existing ones are reused"
        f"{' unless --refresh' if not args.refresh else ''})"
    )
    with console.status("Loading embedding model..."):
        embedder = load_embedder(s)
    indexed = build_index(
        tables, s.db_index_dir, embedder, llm,
        refresh_descriptions=args.refresh,
        on_progress=lambda msg: console.print(f"  {escape(msg)}"),
        db_path=s.db_path,
        force=args.force,
    )  # fmt: skip
    console.print(
        f"[green]Done.[/] {len(indexed)} tables indexed into {escape(str(s.db_index_dir))}"
    )
    return 0


def cmd_tables(args: argparse.Namespace) -> int:
    s = _settings(args)
    docs = load_descriptions(s.db_index_dir)
    table = Table("Table", "Rows", "Description", show_lines=True)
    for t in read_schema(s.db_path, sample_rows=0):
        description = escape(docs[t.name]) if t.name in docs else "[dim](not indexed)[/]"
        table.add_row(escape(t.name), f"{t.row_count:,}", description)
    console.print(table)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    s = _settings(args)
    llm = create_llm(s, "helper")
    if not isinstance(llm, OpenAICompatibleLLM):
        console.print(f"Provider '{escape(s.llm_provider)}' has no model list.")
        return 0
    console.print(f"Models served by [bold]{escape(s.llm_provider)}[/] for your key:")
    for model_id in llm.list_models():
        roles = [r for r, m in (("SQL", s.sql_model), ("helper", s.helper_model)) if m == model_id]
        marker = f" <- {' + '.join(roles)} model" if roles else ""
        console.print(f"  {escape(model_id)}[green]{marker}[/]")
    return 0


def render(out: PipelineResult, show_sql: bool = True, max_rows: int = 15) -> None:
    """Pretty-print a result: answer first, details after."""
    if out.status == "unanswerable":
        title = "Can't answer that from this data"
        console.print(Panel(escape(out.message), title=title, style="yellow"))
        return
    if out.status == "error":
        console.print(Panel(escape(out.message), title="Something went wrong", style="red"))
        if out.sql and show_sql:
            console.print(Syntax(out.sql, "sql", word_wrap=True))
        return

    a = out.analysis
    if a is not None:
        # Markdown (not console markup) on purpose: the analysis is written in Markdown.
        body = a.answer
        if a.insights:
            body += "\n\n" + "\n".join(f"- {i}" for i in a.insights)
        if a.caveats:
            body += "\n\n*Caveats:* " + " ".join(a.caveats)
        console.print(Panel(Markdown(body), title="Answer", border_style="green"))

    res = out.result
    if res is not None and res.columns:
        table = Table(*(escape(c) for c in res.columns), header_style="bold")
        for row in res.rows[:max_rows]:
            table.add_row(*("NULL" if v is None else escape(str(v)) for v in row))
        console.print(table)
        if res.row_count > max_rows or res.truncated:
            limit_note = ", cut off at the row limit" if res.truncated else ""
            shown = min(max_rows, res.row_count)
            console.print(f"[dim]... showing {shown} of {res.row_count} rows{limit_note}[/]")

    if show_sql and out.sql:
        console.print(Syntax(out.sql, "sql", word_wrap=True, theme="ansi_dark"))
    fixes = len(out.attempts) - 1
    console.print(
        f"[dim]tables: {escape(', '.join(out.tables_used))} | "
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


def cmd_chat(args: argparse.Namespace) -> int:
    s = _settings(args)
    pipeline = Pipeline.from_settings(s)
    console.print(
        "Ask about the data; follow-ups such as [italic]and for 2012?[/] work. "
        "An empty line or [bold]exit[/] quits."
    )
    history: list[Turn] = []
    while True:
        try:
            question = console.input("[bold cyan]you>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"", "exit", "quit"}:
            break
        try:
            with console.status("Thinking..."):
                out = pipeline.ask(question, history=history)
        except KeyboardInterrupt:  # Ctrl+C stops this question, not the conversation
            console.print("[dim]Cancelled.[/]")
            continue
        if out.interpreted_as:
            console.print(f"[dim]Interpreted as: {escape(out.interpreted_as)}[/]")
        render(out, show_sql=not args.no_sql)
        if out.status != "error":
            history.append(out.as_turn())
    return 0


# --------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="text2sql", description=__doc__.splitlines()[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", type=Path, help="SQLite file (default: T2S_DB_PATH)")
    common.add_argument("--provider", help="LLM provider (default: T2S_LLM_PROVIDER)")
    common.add_argument("--sql-model", help="model for SQL generation")
    common.add_argument("--helper-model", help="model for selection, follow-ups, analysis, descriptions")

    sub = parser.add_subparsers(dest="command", required=True)
    p_index = sub.add_parser("index", parents=[common], help="build the table index")
    p_index.add_argument("--refresh", action="store_true", help="regenerate all descriptions")
    p_index.add_argument(
        "--no-llm", action="store_true", help="write template descriptions for new tables"
    )
    p_index.add_argument(
        "--force",
        action="store_true",
        help="reuse an index folder built for another database file (e.g. after a move)",
    )
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", parents=[common], help="answer a question")
    p_ask.add_argument("question", nargs="+")
    p_ask.add_argument("--no-sql", action="store_true", help="hide the SQL")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", parents=[common], help="ask questions in a conversation")
    p_chat.add_argument("--no-sql", action="store_true", help="hide the SQL")
    p_chat.set_defaults(func=cmd_chat)

    p_tables = sub.add_parser("tables", parents=[common], help="list indexed tables")
    p_tables.set_defaults(func=cmd_tables)

    p_models = sub.add_parser("models", parents=[common], help="list available models")
    p_models.set_defaults(func=cmd_models)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError, LLMError, sqlite3.Error) as exc:
        # Configuration problems (missing DB or key, not a SQLite file, stale index)
        # get one clear line instead of a traceback.
        console.print(f"[red]Error:[/] {escape(str(exc))}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
