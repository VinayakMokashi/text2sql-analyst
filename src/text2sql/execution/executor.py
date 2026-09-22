"""Run validated SQL against the database with a row cap, a size cap and a timeout."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from text2sql.db.connection import connect_readonly
from text2sql.execution.guardrails import DENIED_FUNCTIONS, QueryError, validate_sql

# Largest single string or blob a query may produce. Without it, one expression such as
# hex(zeroblob(...)) could build a value of up to 1 GB, which neither the row cap nor
# the timeout (checked between VM steps) can stop.
MAX_VALUE_BYTES = 1_000_000

# What a read-only query legitimately needs SQLite to do; everything else is refused
# by the authorizer. This is the layer that holds even where sqlglot's grammar and
# SQLite's disagree (for example, it refuses pragma_table_info()).
_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


class SQLExecutionError(QueryError):
    """The database rejected the query (syntax, unknown column, ...)."""


class QueryTimeoutError(SQLExecutionError):
    """The query ran longer than the configured timeout."""


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool  # True if more rows existed than max_rows
    elapsed_s: float

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.columns)


def _authorize(action: int, arg1: str | None, arg2: str | None, *_: object) -> int:
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    # For a function call SQLite passes (None, function name), so the name is in arg2.
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in DENIED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def unique_column_names(names: list[str]) -> list[str]:
    """Make result column names unique (``Name``, ``Name_2``, ...).

    ``SELECT t.Name, g.Name`` returns two columns called ``Name``; pandas and the
    chart and statistics code need every column name to be distinct.
    """
    seen: set[str] = set()
    result = []
    for name in names:
        candidate, n = name, 1
        while candidate in seen:
            n += 1
            candidate = f"{name}_{n}"
        seen.add(candidate)
        result.append(candidate)
    return result


def execute_query(
    db_path: str | Path,
    sql: str,
    max_rows: int = 200,
    timeout_s: float = 10.0,
    dialect: str = "sqlite",
) -> QueryResult:
    """Validate and run ``sql`` read-only.

    Row limit: we fetch at most ``max_rows + 1`` rows instead of rewriting the query's
    LIMIT, so the SQL the user sees is exactly the SQL that ran, and the extra row
    tells us whether the result was truncated.

    Timeout: SQLite has no statement timeout, but it calls a "progress handler" every
    N virtual-machine instructions; returning non-zero from it aborts the query.
    """
    safe_sql = validate_sql(sql, dialect)
    conn = connect_readonly(db_path)
    conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_VALUE_BYTES)
    conn.set_authorizer(_authorize)
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)

    start = time.perf_counter()
    try:
        cursor = conn.execute(safe_sql)
        rows = cursor.fetchmany(max_rows + 1)
        columns = unique_column_names([d[0] for d in cursor.description or []])
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise QueryTimeoutError(
                f"The query took longer than {timeout_s:g}s and was stopped."
            ) from exc
        raise SQLExecutionError(str(exc)) from exc
    except sqlite3.Error as exc:  # DataError for oversized values, DatabaseError, ...
        raise SQLExecutionError(str(exc)) from exc
    finally:
        conn.close()

    return QueryResult(
        sql=safe_sql,
        columns=columns,
        rows=rows[:max_rows],
        truncated=len(rows) > max_rows,
        elapsed_s=time.perf_counter() - start,
    )
