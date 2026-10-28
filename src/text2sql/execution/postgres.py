"""Run validated SQL on PostgreSQL: read-only, with a row cap and a statement timeout.

The SQL may be written in another dialect, SQLite by default, and is then translated
with sqlglot. That lets a model keep the prompts it was evaluated with while the data
moves to a server.
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.tokens import Token, TokenType

from text2sql.db.postgres import connect_postgres
from text2sql.execution.executor import (
    QueryResult,
    QueryTimeoutError,
    SQLExecutionError,
    unique_column_names,
)
from text2sql.execution.guardrails import validate_sql


def _sqlite_rule(node: exp.Expression) -> exp.Expression:
    """Keep what a SQLite query means where sqlglot's translation alone would not."""
    # LIKE ignores ASCII case in SQLite; PostgreSQL's ILIKE does the same. Every argument
    # is kept: NOT LIKE is a LIKE with negate set.
    if isinstance(node, exp.Like):
        return exp.ILike(**node.args)
    # SQLite's REAL is an 8-byte float; PostgreSQL's REAL has only 4 bytes.
    if isinstance(node, exp.Cast) and node.to.this == exp.DataType.Type.FLOAT:
        node.set("to", exp.DataType.build(exp.DataType.Type.DOUBLE))
        return node
    # PostgreSQL rounds to n places only for numeric values, not for floats.
    if isinstance(node, exp.Round) and node.args.get("decimals") is not None:
        node.set("this", exp.cast(node.this, exp.DataType.Type.DECIMAL))
        return node
    # SQLite counts a comparison as 1 or 0 inside SUM and AVG; PostgreSQL has booleans.
    if isinstance(node, (exp.Sum, exp.Avg)) and isinstance(
        node.this, (exp.Predicate, exp.Connector, exp.Not)
    ):
        node.set("this", exp.cast(node.this, exp.DataType.Type.INT))
        return node
    # TOTAL() is SQLite's SUM that returns 0.0 for no rows and always a float.
    if (
        isinstance(node, exp.Anonymous)
        and node.name.lower() == "total"
        and len(node.expressions) == 1
    ):
        total = exp.Coalesce(
            this=exp.Sum(this=node.expressions[0]), expressions=[exp.Literal.number(0)]
        )
        return exp.cast(total, exp.DataType.Type.DOUBLE)
    return node


# Tokens that end a SELECT list when they appear outside parentheses.
_END_OF_SELECT_LIST = {
    TokenType.FROM, TokenType.WHERE, TokenType.GROUP_BY, TokenType.HAVING,
    TokenType.ORDER_BY, TokenType.LIMIT, TokenType.UNION, TokenType.EXCEPT,
    TokenType.INTERSECT, TokenType.WINDOW, TokenType.SEMICOLON,
}  # fmt: skip


def _column_texts(sql: str, read: str) -> list[str] | None:
    """The source text of each column in the query's first top-level SELECT list."""
    tokens = sqlglot.tokenize(sql, read=read)
    depth, start = 0, None
    for i, token in enumerate(tokens):
        if token.token_type == TokenType.L_PAREN:
            depth += 1
        elif token.token_type == TokenType.R_PAREN:
            depth -= 1
        elif token.token_type == TokenType.SELECT and depth == 0:
            start = i + 1
            break
    if start is None:
        return None
    while start < len(tokens) and tokens[start].token_type in (TokenType.DISTINCT, TokenType.ALL):
        start += 1
    columns: list[list[Token]] = [[]]
    depth = 0
    for token in tokens[start:]:
        if depth == 0 and token.token_type in _END_OF_SELECT_LIST:
            break
        if token.token_type == TokenType.L_PAREN:
            depth += 1
        elif token.token_type == TokenType.R_PAREN:
            depth -= 1
        if depth == 0 and token.token_type == TokenType.COMMA:
            columns.append([])
        else:
            columns[-1].append(token)
    return [sql[c[0].start : c[-1].end + 1] for c in columns if c]


def _name_columns_like_sqlite(tree: exp.Expression, sql: str) -> None:
    """Alias each unnamed expression column with its own text, as SQLite names it.

    SQLite calls ``ROUND(price / 7, 2)`` by that text; PostgreSQL would call it
    ``round``. Column names reach charts and the analysis prompt, so they must match.
    """
    texts = _column_texts(sql, "sqlite")
    selects = tree.selects
    if texts is None or len(texts) != len(selects):
        return
    for node, text in zip(selects, texts, strict=True):
        if not isinstance(node, (exp.Alias, exp.Column, exp.Star)):
            node.replace(exp.alias_(node.copy(), text, quoted=True))


def to_postgres(sql: str, read: str = "sqlite") -> str:
    """Translate one query from ``read``'s dialect to PostgreSQL, keeping its meaning.

    sqlglot handles most of it (NULLs sort first in SQLite, division by zero gives NULL,
    string and date functions differ). Rules of our own cover the rest for SQLite:
    LIKE becomes ILIKE, REAL becomes DOUBLE PRECISION, ROUND(x, n) rounds a NUMERIC,
    a comparison inside SUM or AVG counts as an integer, TOTAL() becomes a SUM, and
    expression columns keep SQLite's names. PostgreSQL cuts names at 63 bytes, so a
    longer expression's name can still differ.
    """
    if read in ("postgres", "postgresql"):
        return sql
    tree = sqlglot.parse_one(sql, read=read)
    if read == "sqlite":
        tree = tree.transform(_sqlite_rule)
        _name_columns_like_sqlite(tree, sql)
    return tree.sql(dialect="postgres")


def _plain(value: Any) -> Any:
    """NUMERIC values arrive as Decimal; SQLite would have returned a float."""
    return float(value) if isinstance(value, Decimal) else value


def execute_postgres_query(
    url: str,
    sql: str,
    max_rows: int = 200,
    timeout_s: float = 10.0,
    dialect: str = "sqlite",
) -> QueryResult:
    """Validate ``sql`` in its own dialect, translate it, and run it read-only.

    ``sql`` in the result is the query as written; ``executed_sql`` is what PostgreSQL
    ran. As with SQLite, at most ``max_rows + 1`` rows are fetched, so the extra row
    tells whether the result was cut.
    """
    safe_sql = validate_sql(sql, dialect)
    pg_sql = validate_sql(to_postgres(safe_sql, read=dialect), "postgres")

    import psycopg

    conn = connect_postgres(url, timeout_s)
    start = time.perf_counter()
    try:
        cursor = conn.execute(pg_sql.encode())
        rows = cursor.fetchmany(max_rows + 1)
        columns = unique_column_names([d.name for d in cursor.description or []])
    except psycopg.errors.QueryCanceled as exc:
        raise QueryTimeoutError(
            f"The query took longer than {timeout_s:g}s and was stopped."
        ) from exc
    except psycopg.Error as exc:
        # The server's own message ("column x does not exist") is what a model repairs from.
        message = exc.diag.message_primary if exc.diag.message_primary else str(exc)
        raise SQLExecutionError(message) from exc
    finally:
        conn.close()

    return QueryResult(
        sql=safe_sql,
        columns=columns,
        rows=[tuple(_plain(v) for v in row) for row in rows[:max_rows]],
        truncated=len(rows) > max_rows,
        elapsed_s=time.perf_counter() - start,
        executed_sql=pg_sql,
    )
