"""Stage 2 of schema linking: the LLM picks the tables actually needed.

Vector similarity alone cannot tell that "best-selling artist" needs ``InvoiceLine``
rather than ``Playlist``; a language model reading the descriptions can. Keeping the
final schema small also makes the SQL prompt shorter and the SQL more accurate.
"""

from __future__ import annotations

from dataclasses import dataclass

from text2sql.db.schema import TableSchema
from text2sql.llm.base import LLM
from text2sql.prompts import Candidate, table_selection_prompt
from text2sql.retrieval.retriever import RetrievedTable
from text2sql.utils import parse_json_object

# Upper bound on the names-only list of non-candidate tables, so very large schemas
# cannot blow up the prompt; the candidates are always shown in full.
MAX_OTHER_TABLES = 200


@dataclass
class TableSelection:
    tables: list[str]
    answerable: bool = True
    reason: str = ""
    fallback: bool = False  # True when the LLM reply was unusable


class TableSelector:
    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def select(
        self,
        question: str,
        candidates: list[RetrievedTable],
        schemas: dict[str, TableSchema],
        max_tables: int,
    ) -> TableSelection:
        offered = [
            Candidate(c.name, c.description, schemas[c.name.lower()].column_names)
            for c in candidates
            if c.name.lower() in schemas
        ]
        offered_lower = {c.name.lower() for c in offered}
        others = sorted(s.name for key, s in schemas.items() if key not in offered_lower)
        system, user = table_selection_prompt(
            question, offered, max_tables, others[:MAX_OTHER_TABLES]
        )
        reply = self._llm.complete(system, user, max_tokens=1000).text
        data = parse_json_object(reply)

        def top_k_fallback(reason: str) -> TableSelection:
            # Never fail the whole question because of a formatting slip: the
            # retriever's top-k is a reasonable guess.
            return TableSelection(
                [c.name for c in offered[:max_tables]], reason=reason, fallback=True
            )

        if data is None:
            return top_k_fallback("Could not parse table selection; using top matches.")

        reason = str(data.get("reason", ""))
        if data.get("answerable") is False:
            return TableSelection([], answerable=False, reason=reason)

        # Keep only real table names (candidates or the names-only list), restoring
        # their original casing; anything else is a hallucination and is dropped.
        by_lower = {key: s.name for key, s in schemas.items()}
        raw = data.get("tables") or []
        chosen: list[str] = []
        for name in raw if isinstance(raw, list) else []:
            real = by_lower.get(str(name).strip().strip('"').lower())
            if real and real not in chosen:
                chosen.append(real)

        if not chosen:
            return top_k_fallback("Selection named no known tables; using top matches.")
        return TableSelection(chosen[:max_tables], answerable=True, reason=reason)
