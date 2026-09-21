"""Classify result columns by role: identifier, time, measure or label.

Both the summary statistics and the chart picker need the same answer to "which
columns are numbers worth adding up?", so the rules live here once.
"""

from __future__ import annotations

import re

import pandas as pd

_TIME_NAME_RE = re.compile(r"(date|year|month|day|week|quarter|period|time)", re.IGNORECASE)
_DATE_VALUE_RE = re.compile(r"^\d{4}(-\d{2}){0,2}([ T].*)?$")  # 2024, 2024-01, 2024-01-31


def is_id_column(name: str) -> bool:
    """Surrogate keys are numbers, but summing or charting them is meaningless."""
    lowered = str(name).lower()
    return lowered == "id" or lowered.endswith("id")


def is_time_column(df: pd.DataFrame, col: str) -> bool:
    """Datetime dtype, or a time-sounding name whose values all look like dates/years."""
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return True
    if not _TIME_NAME_RE.search(str(col)):
        return False
    values = df[col].dropna().astype(str)
    return not values.empty and bool(values.map(lambda v: bool(_DATE_VALUE_RE.match(v))).all())


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
