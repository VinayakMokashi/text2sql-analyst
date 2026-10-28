"""Run validated SQL on PostgreSQL: read-only, with a row cap and a statement timeout.

The SQL may be written in another dialect, SQLite by default, and is then translated
with sqlglot. That lets a model keep the prompts it was evaluated with while the data
moves to a server.
"""

from __future__ import annotations

import time
from collections.abc import Collection
from decimal import Decimal
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    import psycopg


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


def sqlite_column_names(sql: str) -> list[str | None] | None:
    """What SQLite calls each result column of ``sql``.

    An alias keeps the case it was written in (PostgreSQL folds it to lower case), and
    an expression is named by its source text (PostgreSQL would say ``round``). A column
    reference gets None: SQLite and PostgreSQL both name it after the table's column.
    Returns None when the list has a star, whose columns can't be paired up.
    """
    selects = sqlglot.parse_one(sql, read="sqlite").selects
    texts = _column_texts(sql, "sqlite")
    if texts is None or len(texts) != len(selects):
        return None
    names: list[str | None] = []
    for node, text in zip(selects, texts, strict=True):
        if isinstance(node, exp.Star) or (
            isinstance(node, exp.Column) and isinstance(node.this, exp.Star)
        ):
            return None
        if isinstance(node, exp.Alias):
            names.append(node.alias)
        elif isinstance(node.unnest(), exp.Column):
            names.append(None)
        else:
            names.append(text)
    return names


def _expand_alias_references(tree: exp.Expression, columns: Collection[str]) -> None:
    """Put an alias's expression where WHERE or HAVING names the alias.

    SQLite lets those clauses refer to a column alias of the same SELECT; PostgreSQL
    doesn't. A name is replaced only when it is no column of the database (``columns``)
    and no output of another SELECT in the query (a CTE or a subquery), since SQLite
    reads a real column's name as that column.
    """
    selects = list(tree.find_all(exp.Select))
    known = {c.lower() for c in columns}
    for select in selects:
        aliases = {
            node.alias.lower(): node.this
            for node in select.expressions
            if isinstance(node, exp.Alias)
        }
        if not aliases:
            continue
        others = {n.lower() for s in selects if s is not select for n in s.named_selects}
        for clause in (select.args.get("where"), select.args.get("having")):
            if clause is None:
                continue
            for column in list(clause.find_all(exp.Column)):
                name = column.name.lower()
                if (
                    column.table
                    or name not in aliases
                    or name in known
                    or name in others
                    or column.find_ancestor(exp.Select) is not select
                ):
                    continue
                column.replace(aliases[name].copy())


def to_postgres(sql: str, read: str = "sqlite", columns: Collection[str] | None = None) -> str:
    """Translate one query from ``read``'s dialect to PostgreSQL, keeping its meaning.

    sqlglot handles most of it (NULLs sort first in SQLite, division by zero gives NULL,
    string and date functions differ). Rules of our own cover the rest for SQLite:
    LIKE becomes ILIKE, REAL becomes DOUBLE PRECISION, ROUND(x, n) rounds a NUMERIC,
    a comparison inside SUM or AVG counts as an integer, and TOTAL() becomes a SUM.
    Given the database's column names, an alias that WHERE or HAVING refers to is
    replaced by its expression too. Result column names are kept by
    ``execute_postgres_query`` (see ``sqlite_column_names``).
    """
    if read in ("postgres", "postgresql"):
        return sql
    tree = sqlglot.parse_one(sql, read=read)
    if read == "sqlite":
        tree = tree.transform(_sqlite_rule)
        if columns is not None:
            _expand_alias_references(tree, columns)
    return tree.sql(dialect="postgres")


# Every column name in a database's current schema, read once per URL.
_KNOWN_COLUMNS: dict[str, frozenset[str]] = {}
_COLUMN_NAMES = """
SELECT DISTINCT column_name FROM information_schema.columns
WHERE table_schema = current_schema()
"""


def _known_columns(conn: psycopg.Connection[Any], url: str) -> frozenset[str]:
    if url not in _KNOWN_COLUMNS:
        _KNOWN_COLUMNS[url] = frozenset(r[0] for r in conn.execute(_COLUMN_NAMES).fetchall())
    return _KNOWN_COLUMNS[url]


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
    tells whether the result was cut, and the columns carry SQLite's names.
    """
    safe_sql = validate_sql(sql, dialect)
    # Checked before connecting too, so a refused query never reaches the server.
    validate_sql(to_postgres(safe_sql, read=dialect), "postgres")

    import psycopg

    conn = connect_postgres(url, timeout_s)
    start = time.perf_counter()
    try:
        known = _known_columns(conn, url)
        pg_sql = validate_sql(to_postgres(safe_sql, read=dialect, columns=known), "postgres")
        cursor = conn.execute(pg_sql.encode())
        rows = cursor.fetchmany(max_rows + 1)
        names = [d.name for d in cursor.description or []]
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

    if dialect == "sqlite":
        sqlite_names = sqlite_column_names(safe_sql)
        if sqlite_names is not None and len(sqlite_names) == len(names):
            names = [mine or theirs for mine, theirs in zip(sqlite_names, names, strict=True)]
    return QueryResult(
        sql=safe_sql,
        columns=unique_column_names(names),
        rows=[tuple(_plain(v) for v in row) for row in rows[:max_rows]],
        truncated=len(rows) > max_rows,
        elapsed_s=time.perf_counter() - start,
        executed_sql=pg_sql,
    )
