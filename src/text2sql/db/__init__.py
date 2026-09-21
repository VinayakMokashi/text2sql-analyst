"""Database access: read-only connections and schema introspection."""

from text2sql.db.connection import connect_readonly
from text2sql.db.schema import Column, ForeignKey, TableSchema, read_schema, schema_by_name

__all__ = [
    "Column",
    "ForeignKey",
    "TableSchema",
    "connect_readonly",
    "read_schema",
    "schema_by_name",
]
