"""Schema introspection: turn a SQLite database into ``TableSchema`` objects.

These objects feed every later stage: the table descriptions used for indexing, the
schema text inside the SQL prompt, and the foreign-key graph used to add join tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from text2sql.db.connection import connect_readonly


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    primary_key: bool = False
    not_null: bool = False


@dataclass(frozen=True)
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str


@dataclass
class TableSchema:
    """Structure plus a few example rows of one table."""

    name: str
    columns: list[Column]
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    sample_rows: list[tuple[Any, ...]] = field(default_factory=list)
    row_count: int = 0

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def to_ddl(self) -> str:
        """Render a compact ``CREATE TABLE`` statement.

        LLMs have seen a huge amount of DDL during training, so this is the format they
        read most reliably; it also carries keys, which matter for picking joins.
        """
        lines = []
        pks = [c.name for c in self.columns if c.primary_key]
        for col in self.columns:
            parts = [f'  "{col.name}"', col.type or "TEXT"]
            if col.not_null:
                parts.append("NOT NULL")
            lines.append(" ".join(parts))
        if pks:
            lines.append("  PRIMARY KEY (" + ", ".join(f'"{p}"' for p in pks) + ")")
        for fk in self.foreign_keys:
            lines.append(
                f'  FOREIGN KEY ("{fk.column}") REFERENCES "{fk.ref_table}" ("{fk.ref_column}")'
            )
        return f'CREATE TABLE "{self.name}" (\n' + ",\n".join(lines) + "\n);"

    def samples_as_text(self, limit: int | None = None, max_chars: int = 40) -> str:
        """Render sample rows as a small pipe-separated table (values truncated)."""
        rows = self.sample_rows if limit is None else self.sample_rows[:limit]
        if not rows:
            return "(no rows)"

        def fmt(value: Any) -> str:
            text = "NULL" if value is None else str(value)
            return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

        header = " | ".join(self.column_names)
        body = "\n".join(" | ".join(fmt(v) for v in row) for row in rows)
        return f"{header}\n{body}"


def read_schema(db_path: str | Path, sample_rows: int = 3) -> list[TableSchema]:
    """Introspect every user table in ``db_path`` (SQLite internals are skipped)."""
    conn = connect_readonly(db_path)
    try:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        tables: list[TableSchema] = []
        for name in names:
            quoted = '"' + name.replace('"', '""') + '"'
            columns = [
                # PRAGMA table_info: cid, name, type, notnull, default, pk
                Column(name=r[1], type=r[2], not_null=bool(r[3]), primary_key=r[5] > 0)
                for r in conn.execute(f"PRAGMA table_info({quoted})")
            ]
            fks = [
                # PRAGMA foreign_key_list: id, seq, table, from, to, ...
                ForeignKey(column=r[3], ref_table=r[2], ref_column=r[4])
                for r in conn.execute(f"PRAGMA foreign_key_list({quoted})")
            ]
            samples = (
                conn.execute(f"SELECT * FROM {quoted} LIMIT ?", (sample_rows,)).fetchall()
                if sample_rows
                else []
            )
            count = conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
            tables.append(TableSchema(name, columns, fks, samples, count))
        return tables
    finally:
        conn.close()


def schema_by_name(tables: list[TableSchema]) -> dict[str, TableSchema]:
    """Case-insensitive lookup table, since LLMs do not always preserve name casing."""
    return {t.name.lower(): t for t in tables}
