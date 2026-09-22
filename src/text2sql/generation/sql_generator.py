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

# A code fence with any (or no) language tag: ```sql, ```sqlite, ``` sql, ```.
_FENCE_RE = re.compile(r"```[ \t]*[\w+-]*[ \t]*\n(.*?)```", re.DOTALL)

# Where the query starts. "With the schema above..." is prose, so WITH only counts
# when it has the shape of a CTE ("WITH name AS (").
_CTE = (
    r"WITH\s+(?:RECURSIVE\s+)?[\w\"`\[\]]+\s*(?:\([^)]*\)\s*)?"
    r"AS\s*(?:NOT\s+)?(?:MATERIALIZED\s+)?\("
)
_KEYWORD = rf"(?:{_CTE}|SELECT\b)"
# Tried in order: an uppercase keyword at the start of a line, an uppercase keyword
# anywhere ("To select the artists, run: SELECT ..."), then the same ignoring case.
_SQL_START_PATTERNS = [
    re.compile(rf"^[ \t]*{_KEYWORD}", re.MULTILINE),
    re.compile(rf"\b{_KEYWORD}"),
    re.compile(rf"^[ \t]*{_KEYWORD}", re.MULTILINE | re.IGNORECASE),
    re.compile(rf"\b{_KEYWORD}", re.IGNORECASE),
]
# The refusal marker at the start of any line, even inside a code fence.
_CANNOT_RE = re.compile(
    rf"^[^\w\n]*{CANNOT_ANSWER}\b[*`_ \t]*:?[ \t]*(.*)$", re.MULTILINE | re.IGNORECASE
)
_DEFAULT_REASON = "The data does not contain this information."
# Any statement at all, safe or not: a reply starting with DELETE is SQL to reject and
# repair, not a prose refusal.
_ANY_STATEMENT_RE = re.compile(
    r"^[ \t]*(SELECT|WITH|INSERT|UPDATE|DELETE|REPLACE|DROP|CREATE|ALTER|PRAGMA|ATTACH"
    r"|DETACH|VACUUM|EXPLAIN|BEGIN)\b",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass
class GeneratedSQL:
    sql: str | None
    raw: str  # the full model reply, kept for debugging in the UI
    latency_s: float = 0.0
    cannot_answer: str | None = None  # the model's reason when it declines


def _sql_start(text: str) -> int | None:
    for pattern in _SQL_START_PATTERNS:
        match = pattern.search(text)
        if match:
            # The match may begin with indentation; start at the keyword itself.
            return match.start() + len(match.group(0)) - len(match.group(0).lstrip())
    return None


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply.

    Models are asked for one ```sql fence but do not always comply. Preference order:
    the last fence that contains a query (a model that drafts and then corrects itself
    puts the final version last), any fence, then the reply text from the first query
    keyword up to the first semicolon or blank line, so trailing prose is dropped.
    """
    fences = _FENCE_RE.findall(text)
    with_sql = [f for f in fences if _sql_start(f) is not None]
    if with_sql or fences:
        sql = (with_sql or fences)[-1]
        start = _sql_start(sql)
        sql = sql[start:] if start is not None else sql
    else:
        start = _sql_start(text)
        sql = text[start:] if start is not None else text
        sql = re.split(r";|\n[ \t]*\n", sql, maxsplit=1)[0]
    return sql.strip().rstrip(";").strip()


def parse_reply(reply: str, latency_s: float = 0.0) -> GeneratedSQL:
    cannot = _CANNOT_RE.search(reply)
    if cannot:
        reason = cannot.group(1).strip().strip("`*").strip() or _DEFAULT_REASON
        return GeneratedSQL(sql=None, raw=reply, latency_s=latency_s, cannot_answer=reason)
    is_prose = "```" not in reply and not _ANY_STATEMENT_RE.search(reply)
    if reply.strip() and is_prose and _sql_start(reply) is None:
        # Prose with no query at all ("The database doesn't store weather data.") is
        # a refusal in all but name; retrying it as SQL would only fail three times.
        reason = reply.strip().splitlines()[0][:300]
        return GeneratedSQL(sql=None, raw=reply, latency_s=latency_s, cannot_answer=reason)
    return GeneratedSQL(sql=extract_sql(reply), raw=reply, latency_s=latency_s)


class SQLGenerator:
    def __init__(self, llm: LLM, dialect: str = "sqlite", sample_rows: int = 3) -> None:
        self.llm = llm
        self.dialect = dialect
        self.sample_rows = sample_rows

    def generate(self, question: str, tables: list[TableSchema]) -> GeneratedSQL:
        system, user = sql_generation_prompt(question, tables, self.dialect, self.sample_rows)
        resp = self.llm.complete(system, user, max_tokens=2000)
        return parse_reply(resp.text, resp.latency_s)

    def repair(
        self, question: str, tables: list[TableSchema], failed_sql: str, error: str
    ) -> GeneratedSQL:
        system, user = sql_repair_prompt(
            question, tables, failed_sql, error, self.dialect, self.sample_rows
        )
        resp = self.llm.complete(system, user, max_tokens=2000)
        return parse_reply(resp.text, resp.latency_s)
