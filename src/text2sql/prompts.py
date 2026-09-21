"""Every prompt the pipeline sends, in one place.

Keeping prompts together (instead of scattered across modules) makes them easy to
read, compare and tune; each builder returns a ``(system, user)`` pair.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from text2sql.db.schema import TableSchema

CANNOT_ANSWER = "CANNOT_ANSWER"


@dataclass(frozen=True)
class Candidate:
    """A table offered to the table-selection step."""

    name: str
    description: str
    columns: Sequence[str]


# --------------------------------------------------------------------------- indexing
def table_description_prompt(table: TableSchema) -> tuple[str, str]:
    system = (
        "You write short documentation for database tables. Be factual and concrete; "
        "only describe what the schema and sample rows show."
    )
    user = f"""Describe this table in 2-3 sentences for a data catalogue.
Say what one row represents, name the most important columns, and mention which
related tables it links to. Finish with the kinds of questions it helps answer.
Reply with the description only, no preamble.

{table.to_ddl()}

Row count: {table.row_count}
Sample rows:
{table.samples_as_text()}"""
    return system, user


# -------------------------------------------------------------------------- retrieval
def table_selection_prompt(
    question: str,
    candidates: Sequence[Candidate],
    max_tables: int,
    other_tables: Sequence[str] = (),
) -> tuple[str, str]:
    system = (
        "You are a database expert. You decide which tables are needed to answer a "
        "question with SQL. You answer with JSON only."
    )
    listing = "\n\n".join(
        f"Table: {c.name}\nColumns: {', '.join(c.columns)}\nDescription: {c.description}"
        for c in candidates
    )
    # Vector search can rank a needed table just below the cut-off. Listing the rest
    # of the schema by name lets the model still pick it, and stops it from declaring
    # a question unanswerable only because the right table was not shown in detail.
    others = (
        "\n\nOther tables in the database (names only; you may choose these too):\n"
        + ", ".join(other_tables)
        if other_tables
        else ""
    )
    user = f"""Question: {question}

Candidate tables:
{listing}{others}

Pick the smallest set of tables (at most {max_tables}) whose columns are needed to write
the SQL, including tables needed only to join others together.
If the question cannot be answered from this database at all (for example it asks about
data that is not stored here, or it is not a data question), set "answerable" to false.

Reply with JSON only, in exactly this shape:
{{"tables": ["TableA", "TableB"], "answerable": true, "reason": "one short sentence"}}"""
    return system, user


# ------------------------------------------------------------------------- generation
def _schema_block(tables: Sequence[TableSchema], sample_rows: int) -> str:
    parts = []
    for t in tables:
        block = t.to_ddl()
        if sample_rows and t.sample_rows:
            block += f"\n/* {min(sample_rows, len(t.sample_rows))} example rows from {t.name}:\n"
            block += t.samples_as_text(limit=sample_rows) + "\n*/"
        parts.append(block)
    return "\n\n".join(parts)


def _join_hints(tables: Sequence[TableSchema]) -> str:
    """Spell out the foreign keys among the chosen tables; this prevents wrong joins."""
    names = {t.name.lower() for t in tables}
    hints = [
        f'"{t.name}"."{fk.column}" = "{fk.ref_table}"."{fk.ref_column}"'
        for t in tables
        for fk in t.foreign_keys
        if fk.ref_table.lower() in names
    ]
    return "\n".join(hints) if hints else "(none)"


def sql_system_prompt(dialect: str = "sqlite") -> str:
    return f"""You are an expert data analyst who writes correct {dialect.upper()} SQL.
Rules:
- Write exactly ONE read-only query: a SELECT, optionally with CTEs (WITH ...).
- Use only the tables and columns in the schema. Use the join conditions provided.
- Return human-readable columns (e.g. names, titles) rather than bare IDs, and give
  computed columns clear aliases (e.g. AS total_revenue).
- Use ORDER BY + LIMIT for "top N" / "most" / "least" questions. Do not add a LIMIT
  otherwise; the application caps the number of rows.
- Round money and averages to 2 decimals with ROUND(x, 2).
- {dialect} specifics: use strftime('%Y', col) for years, || for string concatenation.
- If a needed attribute is missing but a column is a close, reasonable equivalent
  (e.g. an invoice's billing country for where customers are), use that column.
- Only if no reasonable query over this schema can answer the question, reply with
  exactly {CANNOT_ANSWER}: <short reason>
Reply with the SQL inside a ```sql code block and nothing else."""


def sql_generation_prompt(
    question: str,
    tables: Sequence[TableSchema],
    dialect: str = "sqlite",
    sample_rows: int = 3,
) -> tuple[str, str]:
    user = f"""### Database schema ({dialect})
{_schema_block(tables, sample_rows)}

### Join conditions
{_join_hints(tables)}

### Question
{question}"""
    return sql_system_prompt(dialect), user


def sql_repair_prompt(
    question: str,
    tables: Sequence[TableSchema],
    failed_sql: str,
    error: str,
    dialect: str = "sqlite",
    sample_rows: int = 3,
) -> tuple[str, str]:
    """Self-correction: show the model its own query and the database error."""
    system, user = sql_generation_prompt(question, tables, dialect, sample_rows)
    user += f"""

### Previous attempt
```sql
{failed_sql}
```
It failed with this error:
{error}

Write a corrected query that fixes the error and still answers the question."""
    return system, user


# --------------------------------------------------------------------------- analysis
def analysis_prompt(
    question: str,
    sql: str,
    row_count: int,
    truncated: bool,
    table_text: str,
    stats_text: str,
) -> tuple[str, str]:
    system = (
        "You are a data analyst explaining query results to a non-technical reader. "
        "You only state facts supported by the data you are given, and answer with JSON."
    )
    shown_note = " (the result was cut off at the row limit)" if truncated else ""
    user = f"""Question: {question}

SQL that was run:
{sql}

Result: {row_count} row(s){shown_note}.
{table_text}

Summary statistics computed by the application (exact, prefer these for totals):
{stats_text}

Reply with JSON only, in exactly this shape:
{{"answer": "1-2 sentences that directly answer the question with the key numbers",
  "insights": ["up to 3 short observations: comparisons, concentration, outliers, trends"],
  "caveats": ["0-2 short notes on limitations, e.g. ties, truncated results, data range"]}}
Do not invent numbers. If the result is empty, say that no matching data was found."""
    return system, user
