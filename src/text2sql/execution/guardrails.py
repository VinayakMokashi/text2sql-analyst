"""Static SQL guardrails: allow exactly one read-only query, nothing else.

This is the first of three layers. The parsed-query check gives the LLM a clear error
message to repair and rejects statements a read-only connection would still accept
(``ATTACH``, ``PRAGMA``, multiple statements). Because sqlglot's grammar is not
SQLite's, the executor adds an allow-list authorizer inside SQLite itself, and the
connection is read-only.
"""

from __future__ import annotations

import logging

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

# sqlglot logs a warning whenever it falls back to a generic Command node; we reject
# those anyway, so the warning is just noise.
logging.getLogger("sqlglot").setLevel(logging.ERROR)

ALLOWED_ROOTS = (exp.Select, exp.SetOperation, exp.Subquery)

_FORBIDDEN_NAMES = [
    "Insert", "Update", "Delete", "Merge", "Drop", "Create", "Alter", "TruncateTable",
    "Command", "Pragma", "Attach", "Detach", "Transaction", "Commit", "Rollback",
    "Set", "Use", "Copy", "LoadData", "Into",
]  # fmt: skip
# Resolve by name so the list keeps working across sqlglot versions.
FORBIDDEN_NODES = tuple(getattr(exp, n) for n in _FORBIDDEN_NAMES if hasattr(exp, n))

# SQLite functions that touch the file system or load native code.
DENIED_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}


class QueryError(Exception):
    """Base class for anything that stops a query from producing a result."""


class UnsafeSQLError(QueryError):
    """The SQL was rejected before execution."""


def validate_sql(sql: str | None, dialect: str = "sqlite") -> str:
    """Return the cleaned SQL if it is a single read-only query, else raise.

    Raises:
        UnsafeSQLError: with a message that is safe to show to users and to the LLM.
    """
    if not sql or not sql.strip():
        raise UnsafeSQLError("The query is empty.")
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except SqlglotError as exc:  # ParseError, and TokenError for e.g. 'Guns N' Roses'
        first_line = str(exc).splitlines()[0]
        raise UnsafeSQLError(f"The SQL could not be parsed: {first_line}") from exc

    if len(statements) != 1:
        raise UnsafeSQLError(f"Exactly one statement is allowed, but {len(statements)} were found.")
    root = statements[0]
    if not isinstance(root, ALLOWED_ROOTS):
        raise UnsafeSQLError(f"Only SELECT queries are allowed (got {root.key.upper()}).")

    for item in root.walk():
        node = item[0] if isinstance(item, tuple) else item  # older sqlglot yields tuples
        if isinstance(node, FORBIDDEN_NODES):
            raise UnsafeSQLError(f"Forbidden operation in query: {node.key.upper()}.")
        if isinstance(node, exp.Anonymous) and node.name.lower() in DENIED_FUNCTIONS:
            raise UnsafeSQLError(f"Function {node.name}() is not allowed.")

    return sql.strip().rstrip(";").strip()
