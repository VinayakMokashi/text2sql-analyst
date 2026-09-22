"""Evaluation metrics: execution accuracy and table-retrieval recall.

Execution accuracy (EX) asks "did the predicted query return the same data as the
gold query?" rather than "is the SQL text identical?", because many different queries
are equally correct. Our comparison is a little more forgiving than the strict
Spider/BIRD versions, in ways a human grader would also accept:

* extra columns in the prediction are fine (returning ``Name, Revenue`` when the gold
  query returns only ``Name``), as long as every gold column is present;
* column order and row order do not matter;
* numbers are compared after rounding to 2 decimals, and numeric strings such as
  ``'2010'`` equal the number ``2010``.
Row counts must match exactly, so an extra or missing row is always an error.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

import sqlglot
from sqlglot import exp

_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")
_MAX_MAPPINGS = 1000
_CENT = Decimal("0.01")


def _round2(value: float | str) -> float:
    """Round to 2 decimals half away from zero, like SQLite's ROUND().

    Python's round() rounds half to even and works on the binary float, so it would
    turn 0.125 into 0.12 while SQLite's ROUND(0.125, 2) gives 0.13.
    """
    number = Decimal(value if isinstance(value, str) else repr(float(value)))
    return float(number.quantize(_CENT, rounding=ROUND_HALF_UP)) + 0.0  # -0.0 -> 0.0


def normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return _round2(value)
    text = str(value).strip()
    if _NUMERIC_RE.match(text):
        return _round2(text)
    return text


def _columns(rows: Sequence[Sequence[Any]]) -> list[tuple[Any, ...]]:
    return [tuple(normalize_value(v) for v in col) for col in zip(*rows, strict=True)]


def _multiset_key(col: tuple[Any, ...]) -> list[str]:
    return sorted(repr(v) for v in col)


def results_match(gold: Sequence[Sequence[Any]], pred: Sequence[Sequence[Any]]) -> bool:
    """True if ``pred`` contains the gold result (see module docstring for the rules)."""
    if len(gold) != len(pred):
        return False
    if not gold:
        return True
    gold_cols, pred_cols = _columns(gold), _columns(pred)

    # For each gold column, the prediction columns holding the same multiset of values.
    candidates = [
        [j for j, pc in enumerate(pred_cols) if _multiset_key(pc) == _multiset_key(gc)]
        for gc in gold_cols
    ]
    if any(not c for c in candidates):
        return False

    # Then check that the rows line up too (not just each column separately), trying
    # each way of assigning distinct prediction columns to the gold columns.
    gold_rows = Counter(zip(*gold_cols, strict=True))
    for n, mapping in enumerate(_injective_mappings(candidates)):
        if n >= _MAX_MAPPINGS:
            break
        projected = Counter(tuple(pred_cols[j][i] for j in mapping) for i in range(len(pred)))
        if projected == gold_rows:
            return True
    return False


def _injective_mappings(candidates: list[list[int]]) -> Iterator[tuple[int, ...]]:
    """Yield assignments of a distinct candidate to every position, depth first.

    Generating only valid assignments matters: with several identical columns (say,
    six all-NULL ones) most combinations reuse a column, and counting those against
    the search budget could reject a prediction identical to the gold result.
    """
    chosen: list[int] = []

    def extend(position: int) -> Iterator[tuple[int, ...]]:
        if position == len(candidates):
            yield tuple(chosen)
            return
        for j in candidates[position]:
            if j not in chosen:
                chosen.append(j)
                yield from extend(position + 1)
                chosen.pop()

    return extend(0)


def tables_in_sql(sql: str, dialect: str = "sqlite") -> set[str]:
    """Lower-cased names of the real tables a query reads (CTE names excluded)."""
    tree = sqlglot.parse_one(sql, read=dialect)
    ctes = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    return {t.name.lower() for t in tree.find_all(exp.Table)} - ctes


def recall(gold: set[str], found: Iterable[str]) -> float:
    """Share of the gold tables that were found (1.0 when nothing was needed)."""
    if not gold:
        return 1.0
    return len(gold & {f.lower() for f in found}) / len(gold)
