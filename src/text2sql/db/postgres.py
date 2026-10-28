"""PostgreSQL: read-only connections and schema introspection.

Needs the ``postgres`` extra (psycopg 3). A database is named by its URL
(``postgresql://user:password@host:5432/dbname``) wherever a SQLite path is accepted.

The read-only guarantee comes from the server, as SQLite's does from ``mode=ro``: every
connection starts read-only transactions, so even a query that slipped past the SQL
guardrails cannot modify anything. For a second, independent layer, connect as a role
that is granted SELECT and nothing else.
"""

from __future__ import annotations

import logging
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

from text2sql.db.schema import Column, ForeignKey, TableSchema, quote_ident

if TYPE_CHECKING:
    import psycopg

log = logging.getLogger(__name__)

URL_SCHEMES = ("postgresql://", "postgres://")


def is_postgres_url(target: object) -> bool:
    """Whether ``target`` names a PostgreSQL database rather than a SQLite file."""
    return isinstance(target, str) and target.startswith(URL_SCHEMES)


def redact_url(url: str) -> str:
    """The URL without its password, for messages and the index's records."""
    parts = urlsplit(url)
    if parts.password is None:
        return url
    netloc = parts.netloc.rsplit("@", 1)[1]
    user = parts.username or ""
    return urlunsplit(parts._replace(netloc=f"{user}:***@{netloc}"))


class DatabaseUnavailableError(RuntimeError):
    """The database server could not be reached. Not the query's fault, so no SQL repair
    can fix it."""


def database_id(target: str | Path) -> str:
    """How an index records its database: a file's full path, or a URL without its
    password."""
    if is_postgres_url(target):
        return redact_url(str(target))
    return str(Path(target).resolve())


def connect_postgres(url: str, timeout_s: float | None = None) -> psycopg.Connection[Any]:
    """Open a connection whose every transaction is read-only.

    ``timeout_s`` becomes the server's ``statement_timeout``, which cancels a query that
    runs too long, wherever it spends its time.
    """
    try:
        import psycopg
        from psycopg.conninfo import conninfo_to_dict
    except ImportError as exc:  # pragma: no cover - depends on the installed extras
        raise RuntimeError(
            "PostgreSQL needs the postgres extra: pip install 'text2sql-analyst[postgres]'"
        ) from exc
    # Added to any options the URL already sets (a search_path, say), not in their place.
    options = [
        str(conninfo_to_dict(url).get("options") or ""),
        "-c default_transaction_read_only=on",
    ]
    if timeout_s is not None:
        options.append(f"-c statement_timeout={max(1, round(timeout_s * 1000))}")
    try:
        return psycopg.connect(
            url, options=" ".join(o for o in options if o), autocommit=True, connect_timeout=10
        )
    except psycopg.OperationalError as exc:
        # The driver's message names the host, and this one may reach an app's users.
        log.warning("Could not connect to %s: %s", redact_url(url), exc)
        raise DatabaseUnavailableError("The database could not be reached.") from exc


def fetch_rows(url: str, sql: str) -> list[tuple[Any, ...]]:
    """Run one trusted query (the library's own, never a model's) and return its rows."""
    with connect_postgres(url) as conn:
        return [tuple(row) for row in conn.execute(sql.encode()).fetchall()]


_TABLES = """
SELECT c.relname
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r', 'p') AND n.nspname = current_schema()
ORDER BY c.relname COLLATE "C"
"""

_COLUMNS = """
SELECT a.attname, format_type(a.atttypid, a.atttypmod), a.attnotnull,
       COALESCE(a.attnum = ANY(pk.conkey), false)
FROM pg_attribute a
LEFT JOIN pg_constraint pk ON pk.conrelid = a.attrelid AND pk.contype = 'p'
WHERE a.attrelid = %s::regclass AND a.attnum > 0 AND NOT a.attisdropped
ORDER BY a.attnum
"""

# One row per column of each foreign key, in key order, so a composite key spans
# several rows sharing a constraint name (as SQLite's PRAGMA foreign_key_list does).
_FOREIGN_KEYS = """
SELECT con.conname, ref.relname, src_col.attname, ref_col.attname
FROM pg_constraint con
JOIN pg_class ref ON ref.oid = con.confrelid
CROSS JOIN LATERAL unnest(con.conkey, con.confkey) WITH ORDINALITY AS k(src, dst, ord)
JOIN pg_attribute src_col ON src_col.attrelid = con.conrelid AND src_col.attnum = k.src
JOIN pg_attribute ref_col ON ref_col.attrelid = con.confrelid AND ref_col.attnum = k.dst
WHERE con.contype = 'f' AND con.conrelid = %s::regclass
ORDER BY con.conname, k.ord
"""


def read_postgres_schema(url: str, sample_rows: int = 3) -> list[TableSchema]:
    """Introspect every table in the connection's current schema, as ``read_schema``
    does for SQLite (tables only, in byte order of their names)."""
    with connect_postgres(url) as conn:
        names = [r[0] for r in conn.execute(_TABLES).fetchall()]
        tables: list[TableSchema] = []
        for name in names:
            quoted = quote_ident(name)
            columns = [
                Column(name=r[0], type=r[1], not_null=bool(r[2]), primary_key=bool(r[3]))
                for r in conn.execute(_COLUMNS, (quoted,)).fetchall()
            ]
            fk_rows = conn.execute(_FOREIGN_KEYS, (quoted,)).fetchall()
            fks = []
            for _, group in groupby(fk_rows, key=lambda r: r[0]):
                rows = list(group)
                fks.append(
                    ForeignKey(tuple(r[2] for r in rows), rows[0][1], tuple(r[3] for r in rows))
                )
            samples = (
                [
                    tuple(row)
                    for row in conn.execute(
                        f"SELECT * FROM {quoted} LIMIT %s".encode(), (sample_rows,)
                    ).fetchall()
                ]
                if sample_rows
                else []
            )
            row = conn.execute(f"SELECT COUNT(*) FROM {quoted}".encode()).fetchone()
            count = int(row[0]) if row else 0
            tables.append(TableSchema(name, columns, fks, samples, count))
        return tables
