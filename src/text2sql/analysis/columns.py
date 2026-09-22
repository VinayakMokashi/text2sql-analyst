"""Classify result columns by role: identifier, time, measure or label.

Both the summary statistics and the chart picker need the same answer to "which
columns are numbers worth adding up?", so the rules live here once.
"""

from __future__ import annotations

import re

import pandas as pd

# Name words that mark a time column, matched as whole words: "invoice_year" and
# "InvoiceMonth" count, "days_rented" and "valid_until" do not.
_TIME_WORDS = {"date", "year", "month", "quarter", "week", "day", "hour", "period", "time"}
# For whole numbers (2009, 7, 3) only unit names count: "time_spent" is a measure.
_NUMERIC_TIME_WORDS = {"year", "month", "quarter", "week", "day", "hour"}
_DATE_VALUE_RE = re.compile(r"^\d{4}(-\d{2}){0,2}([ T].*)?$")  # 2024, 2024-01, 2024-01-31


def _name_words(name: str) -> set[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))  # InvoiceYear -> Invoice Year
    return set(re.findall(r"[a-z]+", spaced.lower()))


def is_id_column(name: str) -> bool:
    """Surrogate keys are numbers, but summing or charting them is meaningless.

    Matches ``id``, ``customer_id`` and ``CustomerId``, but not ``total_paid`` or
    ``valid``, which merely end in the letters "id".
    """
    text = str(name)
    return text.lower() == "id" or text.lower().endswith("_id") or text.endswith(("Id", "ID"))


def is_time_column(df: pd.DataFrame, col: str) -> bool:
    """A datetime column, or a time-named column whose values look like dates or periods."""
    series = df[col]
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    words = _name_words(col)
    if not words & _TIME_WORDS:
        return False
    values = series.dropna()
    if values.empty:
        return False
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        # 2009, 7 or 2009.0 (a year column with a NULL becomes float): a period number.
        whole = bool((values == values.round()).all())
        return whole and bool(words & _NUMERIC_TIME_WORDS)
    return bool(values.astype(str).map(lambda v: bool(_DATE_VALUE_RE.match(v))).all())


def measure_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns that are neither identifiers nor time (e.g. an integer year)."""
    return [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
        and not is_id_column(c)
        and not is_time_column(df, c)
    ]
