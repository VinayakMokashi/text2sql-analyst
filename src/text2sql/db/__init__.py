"""Database access: read-only connections and schema introspection, for SQLite files and
PostgreSQL URLs."""

from text2sql.db.connection import connect_readonly
from text2sql.db.postgres import (
    DatabaseUnavailableError,
    connect_postgres,
    database_id,
    is_postgres_url,
    read_postgres_schema,
    redact_url,
)
from text2sql.db.schema import Column, ForeignKey, TableSchema, read_schema, schema_by_name

__all__ = [
    "Column",
    "DatabaseUnavailableError",
    "ForeignKey",
    "TableSchema",
    "connect_postgres",
    "connect_readonly",
    "database_id",
    "is_postgres_url",
    "read_postgres_schema",
    "read_schema",
    "redact_url",
    "schema_by_name",
]
