"""Text-to-SQL generation and self-correction.

The generator only turns prompts into SQL text. Deciding *whether* a query is safe
and whether it ran is the job of ``execution``; the pipeline connects the two in a
generate -> validate -> execute -> repair loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from text2sql.db.schema import TableSchema
from text2sql.llm.base import LLM
from text2sql.prompts import CANNOT_ANSWER, sql_generation_prompt, sql_repair_prompt

_SQL_FENCE_RE = re.compile(r"```sql\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE_RE = re.compile(r"```\s*(.*?)```", re.DOTALL)
_SQL_START_RE = re.compile(r"\b(WITH|SELECT)\b", re.IGNORECASE)
_CANNOT_RE = re.compile(rf"{CANNOT_ANSWER}\s*:?\s*(.*)", re.IGNORECASE | re.DOTALL)


@dataclass
class GeneratedSQL:
    sql: str | None
    raw: str  # the full model reply, kept for debugging in the UI
    latency_s: float = 0.0
    cannot_answer: str | None = None  # the model's reason when it declines


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply.

    Order of preference: a ```sql fence, any fence, then everything from the first
    SELECT/WITH keyword. Models are asked for a fence but do not always comply.
    """
    match = _SQL_FENCE_RE.search(text) or _ANY_FENCE_RE.search(text)
    sql = match.group(1) if match else text
    if not match:
        start = _SQL_START_RE.search(sql)
        sql = sql[start.start() :] if start else sql
    return sql.strip().rstrip(";").strip()


def parse_reply(reply: str, latency_s: float = 0.0) -> GeneratedSQL:
    stripped = reply.strip()
    cannot = _CANNOT_RE.match(stripped.strip("`").strip())
    if cannot:
        reason = cannot.group(1).strip() or "The data does not contain this information."
        return GeneratedSQL(sql=None, raw=reply, latency_s=latency_s, cannot_answer=reason)
    return GeneratedSQL(sql=extract_sql(stripped), raw=reply, latency_s=latency_s)


class SQLGenerator:
    def __init__(self, llm: LLM, dialect: str = "sqlite", sample_rows: int = 3) -> None:
        self.llm = llm
        self.dialect = dialect
        self.sample_rows = sample_rows

    def generate(self, question: str, tables: list[TableSchema]) -> GeneratedSQL:
        system, user = sql_generation_prompt(question, tables, self.dialect, self.sample_rows)
        resp = self.llm.complete(system, user, max_tokens=800)
        return parse_reply(resp.text, resp.latency_s)

    def repair(
        self, question: str, tables: list[TableSchema], failed_sql: str, error: str
    ) -> GeneratedSQL:
        system, user = sql_repair_prompt(
            question, tables, failed_sql, error, self.dialect, self.sample_rows
        )
        resp = self.llm.complete(system, user, max_tokens=800)
        return parse_reply(resp.text, resp.latency_s)
