"""Pick a simple, honest visual for a result - or none.

Rules (deliberately conservative; a wrong chart is worse than no chart):
  * one row with one number          -> a stat tile ("metric"), not a one-bar chart
  * a time-like column + a number     -> line chart, in time order
  * a category column + a number      -> bar chart sorted by value (2-25 rows)
  * anything else                     -> no chart; the table says it better
Only ONE measure is ever plotted, so there is never a second y-axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from text2sql.analysis.columns import is_id_column, is_time_column, measure_columns

MAX_BARS = 25


@dataclass(frozen=True)
class ChartSpec:
    kind: Literal["metric", "bar", "line"]
    y: str
    x: str | None = None
    title: str = ""


def suggest_chart(df: pd.DataFrame) -> ChartSpec | None:
    if df.empty:
        return None
    measures = measure_columns(df)
    if not measures:
        return None
    y = measures[0]

    if len(df) == 1 and len(df.columns) <= 2:
        return ChartSpec("metric", y=y, title=y)

    others = [c for c in df.columns if c not in measures and not is_id_column(c)]
    time_cols = [c for c in others if is_time_column(df, c)]
    if time_cols and len(df) >= 3 and df[time_cols[0]].is_unique:
        return ChartSpec("line", y=y, x=time_cols[0], title=f"{y} over {time_cols[0]}")

    labels = [c for c in others if c not in time_cols]
    if labels and 2 <= len(df) <= MAX_BARS and df[labels[0]].is_unique:
        return ChartSpec("bar", y=y, x=labels[0], title=f"{y} by {labels[0]}")
    return None
