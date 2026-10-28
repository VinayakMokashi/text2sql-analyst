"""Value linking: the database values a question refers to, as hints for the prompts.

The SQL model sees the schema and a few example rows, so it guesses the values of
categorical columns: asked for a net profit margin, it filters on a metric called
'netProfitMargin' when the column holds 'net_income' and 'revenue'. A linker supplies the
real values as hints, which ``Pipeline.ask`` shows to the table-selection and SQL steps:

- ``ColumnCatalogue``: every value of a short categorical column, with an optional label,
  as one hint: "annual_values.metric takes only these values: revenue (Revenue), ...".
- ``ValueMatcher``: the values of a column that the question mentions, by the value itself
  or by an alias, as hints: '"Coke" is companies.name = 'Coca-Cola'.'
- ``Linker``: anything with ``hints(question)``. ``StaticLinker`` combines catalogues,
  matchers and fixed notes; a pipeline built with ``linker=`` adds its hints to every
  question.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from text2sql.db.connection import connect_readonly
from text2sql.db.postgres import fetch_rows, is_postgres_url
from text2sql.db.schema import quote_ident

MAX_CATALOGUE = 200
MAX_MATCHED_VALUES = 10_000


class Linker(Protocol):
    """What a pipeline needs from a linker."""

    def hints(self, question: str) -> list[str]:
        """Notes about the database values the question refers to."""
        ...


def sql_literal(value: str) -> str:
    """A string as an SQL literal: McDonald's becomes 'McDonald''s'."""
    return "'" + value.replace("'", "''") + "'"


def _distinct(db_path: str | Path, sql: str) -> list[tuple[object, ...]]:
    if is_postgres_url(db_path):
        return fetch_rows(str(db_path), sql)
    conn = connect_readonly(db_path)
    try:
        return [tuple(row) for row in conn.execute(sql).fetchall()]
    finally:
        conn.close()


@dataclass(frozen=True)
class ColumnCatalogue:
    """The values a categorical column takes, for one hint."""

    name: str  # how the hint names the column, such as "annual_values.metric"
    values: tuple[tuple[str, str | None], ...]  # (value, label or None), in order

    @classmethod
    def read(
        cls,
        db_path: str | Path,
        table: str,
        column: str,
        label_column: str | None = None,
        *,
        name: str | None = None,
        max_values: int = MAX_CATALOGUE,
    ) -> ColumnCatalogue:
        """The distinct values of ``table.column``, with their labels if a column has them.

        ``name`` is the column the hint talks about when the values come from a lookup
        table (read ``metrics.metric`` but name ``annual_values.metric``). Raises
        ValueError for more than ``max_values`` values: a catalogue is for a short list,
        not for free text.
        """
        label = f", {quote_ident(label_column)}" if label_column else ""
        rows = _distinct(
            db_path,
            f"SELECT DISTINCT {quote_ident(column)}{label} FROM {quote_ident(table)} "
            f"WHERE {quote_ident(column)} IS NOT NULL ORDER BY 1 LIMIT {max_values + 1}",
        )
        if len(rows) > max_values:
            raise ValueError(
                f"{table}.{column} has more than {max_values} values, too many for a catalogue."
            )
        values = tuple(
            (str(row[0]), str(row[1]) if label_column and row[1] is not None else None)
            for row in rows
        )
        return cls(name or f"{table}.{column}", values)

    def hint(self) -> str:
        items = ", ".join(
            f"{value} ({label})" if label and label != value else value
            for value, label in self.values
        )
        return f"{self.name} takes only these values: {items}."


class ValueMatcher:
    """Finds the values of one column that a question mentions.

    A value matches as a whole word or phrase, ignoring case; ``aliases`` maps other names
    to values ("Coke" to "Coca-Cola"). Longer names are tried first and each stretch of the
    question matches once, so "American Express" doesn't also count as "America".
    """

    def __init__(
        self, name: str, values: Iterable[str], aliases: Mapping[str, str] | None = None
    ) -> None:
        self.name = name
        known = {value for value in values if value and value.strip()}
        names = {value: value for value in known}
        names.update({alias: value for alias, value in (aliases or {}).items() if value in known})
        ordered = sorted(names, key=lambda text: (-len(text), text))
        self._patterns = [
            (re.compile(rf"(?<!\w){re.escape(text)}(?!\w)", re.IGNORECASE), names[text])
            for text in ordered
        ]

    @classmethod
    def read(
        cls,
        db_path: str | Path,
        table: str,
        column: str,
        aliases: Mapping[str, str] | None = None,
        *,
        name: str | None = None,
        max_values: int = MAX_MATCHED_VALUES,
    ) -> ValueMatcher:
        """A matcher for the distinct values of ``table.column``."""
        rows = _distinct(
            db_path,
            f"SELECT DISTINCT {quote_ident(column)} FROM {quote_ident(table)} "
            f"WHERE {quote_ident(column)} IS NOT NULL LIMIT {max_values + 1}",
        )
        if len(rows) > max_values:
            raise ValueError(f"{table}.{column} has more than {max_values} values to match.")
        return cls(name or f"{table}.{column}", (str(row[0]) for row in rows), aliases)

    def matches(self, question: str) -> list[tuple[str, str]]:
        """(the mention as written, the value) for each value mentioned, in question order."""
        remaining = question
        found: list[tuple[int, str, str]] = []
        for pattern, value in self._patterns:
            for match in pattern.finditer(remaining):
                start, end = match.span()
                found.append((start, question[start:end], value))
                remaining = remaining[:start] + " " * (end - start) + remaining[end:]
        seen: set[str] = set()
        result = []
        for _, mention, value in sorted(found):
            if value not in seen:
                seen.add(value)
                result.append((mention, value))
        return result

    def hints(self, question: str) -> list[str]:
        return [
            f'"{mention}" is {self.name} = {sql_literal(value)}.'
            for mention, value in self.matches(question)
        ]


@dataclass(frozen=True)
class StaticLinker:
    """Hints from value matchers (first), column catalogues, then fixed notes."""

    catalogues: Sequence[ColumnCatalogue] = ()
    matchers: Sequence[ValueMatcher] = ()
    notes: Sequence[str] = ()

    def hints(self, question: str) -> list[str]:
        found = [hint for matcher in self.matchers for hint in matcher.hints(question)]
        return [
            *found,
            *(catalogue.hint() for catalogue in self.catalogues),
            *(note for note in self.notes if note.strip()),
        ]
