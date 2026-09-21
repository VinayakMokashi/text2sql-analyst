"""Turn a query result into a plain-English answer with a short analysis.

LLMs are unreliable at arithmetic over many rows, so the application computes the
summary statistics itself (totals, averages, extremes) and hands them to the model;
the model's job is only to explain them clearly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from text2sql.analysis.columns import is_id_column, is_time_column, measure_columns
from text2sql.execution.executor import QueryResult
from text2sql.llm.base import LLM
from text2sql.prompts import analysis_prompt
from text2sql.utils import parse_json_object


@dataclass
class Analysis:
    answer: str
    insights: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    latency_s: float = 0.0


def _fmt(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def summary_stats(df: pd.DataFrame) -> str:
    """Exact totals/averages/extremes per measure column, with the row labels of the
    extremes (e.g. "max 826.65 (Rock)") so the model never has to compute them."""
    if df.empty:
        return "(no rows)"
    measures = measure_columns(df)
    labels = [c for c in df.columns if c not in measures and not is_id_column(c)]
    label_col = labels[0] if labels else None
    lines = []
    for col in measures:
        series = df[col].dropna()
        if series.empty:
            continue
        line = (
            f"{col}: total {_fmt(series.sum())}, average {_fmt(series.mean())}, "
            f"min {_fmt(series.min())}, max {_fmt(series.max())}"
        )
        if label_col is not None and len(df) > 1:
            top = df.loc[series.idxmax(), label_col]
            bottom = df.loc[series.idxmin(), label_col]
            share = series.max() / series.sum() * 100 if series.sum() else 0
            line += f"; max at {label_col}={top} ({share:.1f}% of total); min at {bottom}"
        lines.append(line)

    # For time series, state the overall direction explicitly: comparing the first and
    # last periods is exactly the kind of arithmetic models get wrong on their own.
    time_cols = [c for c in labels if is_time_column(df, c)]
    if time_cols and len(df) > 1:
        tcol = time_cols[0]
        ordered = df.sort_values(tcol)
        for col in measures:
            valid = ordered[[tcol, col]].dropna()
            if len(valid) < 2:
                continue
            # Index each column separately: a mixed-type row would upcast 2009 to 2009.0.
            t0, t1 = valid[tcol].iloc[0], valid[tcol].iloc[-1]
            v0, v1 = valid[col].iloc[0], valid[col].iloc[-1]
            change = v1 - v0
            pct = f", {change / v0 * 100:+.1f}%" if v0 else ""
            lines.append(
                f"{col} from {tcol}={t0} to {tcol}={t1}: {_fmt(v0)} -> {_fmt(v1)} "
                f"(overall change {change:+,.2f}{pct})"
            )
    return "\n".join(lines) if lines else "(no numeric columns)"


def result_table_text(df: pd.DataFrame, max_rows: int) -> str:
    """Compact CSV of the first ``max_rows`` rows (CSV costs the fewest tokens)."""
    shown = df.head(max_rows).to_csv(index=False).strip()
    if len(df) > max_rows:
        shown += f"\n... ({len(df) - max_rows} more rows not shown)"
    return shown


class Analyst:
    def __init__(self, llm: LLM, max_rows: int = 40) -> None:
        self._llm = llm
        self._max_rows = max_rows

    def analyze(self, question: str, result: QueryResult) -> Analysis:
        df = result.to_dataframe()
        if df.empty:
            # No LLM call needed; an empty result has only one honest answer.
            return Analysis(
                answer="No matching data was found for this question.",
                caveats=["The query ran successfully but returned no rows."],
            )

        system, user = analysis_prompt(
            question,
            result.sql,
            result.row_count,
            result.truncated,
            result_table_text(df, self._max_rows),
            summary_stats(df),
        )
        resp = self._llm.complete(system, user, temperature=0.2, max_tokens=1200)
        data = parse_json_object(resp.text)
        if data is None or not str(data.get("answer", "")).strip():
            # Still useful: show whatever the model wrote rather than nothing.
            return Analysis(answer=resp.text.strip(), latency_s=resp.latency_s)

        def as_list(value: object) -> list[str]:
            items = value if isinstance(value, list) else [value] if value else []
            return [str(v).strip() for v in items if str(v).strip()]

        caveats = as_list(data.get("caveats"))
        if result.truncated and not any("row" in c.lower() for c in caveats):
            caveats.append(f"Only the first {result.row_count} rows were returned.")
        return Analysis(
            answer=str(data["answer"]).strip(),
            insights=as_list(data.get("insights"))[:3],
            caveats=caveats[:3],
            latency_s=resp.latency_s,
        )
