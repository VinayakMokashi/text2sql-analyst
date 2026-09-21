"""Run validated SQL against the database with a row cap and a timeout."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from text2sql.db.connection import connect_readonly
from text2sql.execution.guardrails import QueryError, validate_sql


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
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)

    start = time.perf_counter()
    try:
        cursor = conn.execute(safe_sql)
        rows = cursor.fetchmany(max_rows + 1)
        columns = [d[0] for d in cursor.description or []]
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise QueryTimeoutError(
                f"The query took longer than {timeout_s:g}s and was stopped."
            ) from exc
        raise SQLExecutionError(str(exc)) from exc
    except sqlite3.Error as exc:
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
