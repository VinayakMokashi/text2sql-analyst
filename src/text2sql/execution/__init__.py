"""Safe execution: static guardrails plus a read-only, time-limited executor."""

from text2sql.execution.executor import (
    QueryResult,
    QueryTimeoutError,
    SQLExecutionError,
    execute_query,
)
from text2sql.execution.guardrails import QueryError, UnsafeSQLError, validate_sql

__all__ = [
    "QueryError",
    "QueryResult",
    "QueryTimeoutError",
    "SQLExecutionError",
    "UnsafeSQLError",
    "execute_query",
    "validate_sql",
]
