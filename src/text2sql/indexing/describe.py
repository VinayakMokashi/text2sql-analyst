"""Natural-language table descriptions (the "documents" we embed and search).

Raw column names like ``InvoiceLine.UnitPrice`` match questions such as "which genre
earns the most?" poorly. An LLM-written description ("each row is one purchased track
... used for revenue and sales questions") bridges that vocabulary gap.
"""

from __future__ import annotations

from text2sql.db.schema import TableSchema
from text2sql.llm.base import LLM
from text2sql.prompts import table_description_prompt


def fallback_description(table: TableSchema) -> str:
    """A template description, used when no LLM is available (e.g. offline tests)."""
    links = ", ".join(sorted({fk.ref_table for fk in table.foreign_keys}))
    text = f"Table {table.name} with columns {', '.join(table.column_names)}."
    return text + (f" It links to {links}." if links else "")


def describe_table(table: TableSchema, llm: LLM | None) -> str:
    if llm is None:
        return fallback_description(table)
    system, user = table_description_prompt(table)
    text = llm.complete(system, user, temperature=0.2, max_tokens=1000).text.strip()
    return text or fallback_description(table)


def index_document(table: TableSchema, description: str) -> str:
    """The exact text that gets embedded: name + columns + description.

    Column names are included so questions that mention a column directly
    ("customers by country") still match even if the description omits it.
    """
    return f"Table: {table.name}\nColumns: {', '.join(table.column_names)}\n{description}"
