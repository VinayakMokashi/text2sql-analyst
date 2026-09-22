"""Schema introspection: turn a SQLite database into ``TableSchema`` objects.

These objects feed every later stage: the table descriptions used for indexing, the
schema text inside the SQL prompt, and the foreign-key graph used to add join tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from typing import Any

from text2sql.db.connection import connect_readonly


def quote_ident(name: str) -> str:
    """Quote an SQL identifier, doubling embedded quotes (``my"table`` -> ``"my""table"``)."""
    return '"' + str(name).replace('"', '""') + '"'


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    primary_key: bool = False
    not_null: bool = False


@dataclass(frozen=True)
class ForeignKey:
    """A foreign key. Composite keys (``(order_id, line_no)``) have several columns."""

    columns: tuple[str, ...]
    ref_table: str
    ref_columns: tuple[str, ...]  # empty only if the referenced table does not exist


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
            parts = [f"  {quote_ident(col.name)}", col.type or "TEXT"]
            if col.not_null:
                parts.append("NOT NULL")
            lines.append(" ".join(parts))
        if pks:
            lines.append("  PRIMARY KEY (" + ", ".join(quote_ident(p) for p in pks) + ")")
        for fk in self.foreign_keys:
            refs = ", ".join(quote_ident(c) for c in fk.ref_columns)
            lines.append(
                f"  FOREIGN KEY ({', '.join(quote_ident(c) for c in fk.columns)}) "
                f"REFERENCES {quote_ident(fk.ref_table)}" + (f" ({refs})" if refs else "")
            )
        return f"CREATE TABLE {quote_ident(self.name)} (\n" + ",\n".join(lines) + "\n);"

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
        # ESCAPE: in LIKE, "_" matches any character, so 'sqlite_%' alone would also
        # hide user tables such as "SQLiteVersions".
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' ORDER BY name"
            )
        ]

        def primary_key(table: str) -> tuple[str, ...]:
            rows = conn.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
            return tuple(r[1] for r in sorted(rows, key=lambda r: r[5]) if r[5] > 0)

        tables: list[TableSchema] = []
        for name in names:
            quoted = quote_ident(name)
            columns = [
                # PRAGMA table_xinfo: cid, name, type, notnull, default, pk, hidden.
                # Unlike table_info it lists generated columns, which SELECT * returns;
                # hidden == 1 marks virtual-table columns that SELECT * leaves out.
                Column(name=r[1], type=r[2], not_null=bool(r[3]), primary_key=r[5] > 0)
                for r in conn.execute(f"PRAGMA table_xinfo({quoted})")
                if r[6] != 1
            ]
            # PRAGMA foreign_key_list: id, seq, table, from, to, ... One row per column,
            # so a composite key spans several rows sharing an id.
            fk_rows = conn.execute(f"PRAGMA foreign_key_list({quoted})").fetchall()
            fks = []
            ordered = sorted(fk_rows, key=lambda r: (r[0], r[1]))
            for _, group in groupby(ordered, key=lambda r: r[0]):
                rows = list(group)
                ref_cols = tuple(r[4] for r in rows)
                if any(c is None for c in ref_cols):
                    # "REFERENCES parent" without columns means the parent's primary key.
                    ref_cols = primary_key(rows[0][2])
                fks.append(ForeignKey(tuple(r[3] for r in rows), rows[0][2], ref_cols))
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
