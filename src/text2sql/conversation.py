"""Follow-up questions: rewrite them so the rest of the pipeline sees one clear question.

"How many invoices were issued in 2010?" followed by "and in 2012?" only makes sense
together. Rather than teach every stage about conversations, a helper model rewrites
the follow-up into a standalone question ("How many invoices were issued in 2012?"),
and the pipeline answers that. The UI shows the rewrite, so a wrong reading is visible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from text2sql.llm.base import LLM
from text2sql.prompts import followup_rewrite_prompt
from text2sql.utils import parse_json_object

# Only the last few turns: enough to resolve "those" and "and for 2012?", while keeping
# the prompt short and stopping old topics from leaking into new questions.
MAX_TURNS = 3


@dataclass(frozen=True)
class Turn:
    """One answered question, as later follow-ups need to see it."""

    question: str  # the standalone question that was actually answered
    sql: str | None = None
    answer: str | None = None


class FollowUpRewriter:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def rewrite(self, history: Sequence[Turn], question: str) -> str:
        """Return a standalone version of ``question`` (unchanged without history)."""
        if not history:
            return question  # no LLM call for the first question
        recent = [(t.question, _short(t.sql, 600), _short(t.answer, 300)) for t in history]
        system, user = followup_rewrite_prompt(recent[-MAX_TURNS:], question)
        data = parse_json_object(self._llm.complete(system, user, max_tokens=600).text)
        standalone = str((data or {}).get("standalone") or "").strip()
        return standalone or question  # an unusable reply keeps the original question


def _short(text: str | None, limit: int) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
