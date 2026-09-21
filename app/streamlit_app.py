"""Streamlit UI: ask a question, get an answer, analysis, chart, table and (collapsed) SQL.

Run from the repository root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from text2sql.analysis import ChartSpec
from text2sql.config import get_settings
from text2sql.pipeline import Pipeline, PipelineResult

EXAMPLES = [
    "Which 5 genres generate the most revenue?",
    "How did total sales change from year to year?",
    "Who are the top 10 customers by total spending, and where are they from?",
    "Which sales support agent is responsible for the most revenue?",
    "Which artists have the most albums in the catalogue?",
    "What is the average invoice total per country?",
    "What is the salary of each employee?",
]

# Single-hue series colour, stepped separately for light and dark themes so the marks
# keep enough contrast on both surfaces.
SERIES_COLOR = {"light": "#2a78d6", "dark": "#3987e5"}

st.set_page_config(page_title="Text2SQL Analyst", page_icon=":bar_chart:", layout="centered")


@st.cache_resource(show_spinner="Loading models and index...")
def get_pipeline() -> Pipeline:
    return Pipeline.from_settings(get_settings())


def md(text: str) -> str:
    """Escape model-written text for st.markdown.

    Streamlit renders $...$ as LaTeX, so "between $449 and $481" would turn into a
    maths formula. Dollar signs are escaped so amounts display as written.
    """
    return text.replace("$", r"\$")


def theme() -> str:
    try:
        return st.context.theme.type or "light"
    except AttributeError:  # older Streamlit versions
        return "light"


# -------------------------------------------------------------------------- charts
def render_chart(df: pd.DataFrame, spec: ChartSpec) -> None:
    if spec.kind == "metric":
        value = df[spec.y].iloc[0]
        text = f"{value:,.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else f"{value:,}"
        st.metric(label=spec.y.replace("_", " "), value=text)
        return

    color = SERIES_COLOR[theme()]
    tooltip = [alt.Tooltip(spec.x), alt.Tooltip(spec.y, format=",.2f")]
    if spec.kind == "bar":
        # Horizontal bars sorted by value: long category names stay readable and the
        # ranking is visible at a glance.
        chart = (
            alt.Chart(df)
            .mark_bar(color=color, cornerRadiusEnd=4, height={"band": 0.7})
            .encode(
                x=alt.X(spec.y, type="quantitative", title=spec.y, axis=alt.Axis(tickCount=5)),
                y=alt.Y(spec.x, type="nominal", sort="-x", title=None),
                tooltip=tooltip,
            )
            # The height includes the axes (Streamlit fits the chart to it), so give
            # each bar ~32px plus room for the x-axis and its title.
            .properties(height=70 + 32 * len(df))
        )
    else:  # line
        data = df.sort_values(spec.x)
        chart = (
            alt.Chart(data)
            .mark_line(color=color, strokeWidth=2, point=alt.OverlayMarkDef(size=64, color=color))
            .encode(
                x=alt.X(spec.x, type="ordinal", title=None, axis=alt.Axis(labelAngle=0)),
                # Unlike bars, a line needs no zero baseline; starting the axis near the
                # data keeps small changes over time visible.
                y=alt.Y(spec.y, type="quantitative", title=spec.y, scale=alt.Scale(zero=False)),
                tooltip=tooltip,
            )
            .properties(height=280)
        )
    st.altair_chart(chart, width="stretch")


# ------------------------------------------------------------------------- results
def render_result(out: PipelineResult) -> None:
    if out.status == "unanswerable":
        st.warning(f"**I can't answer that from this database.** {md(out.message)}")
        render_details(out)
        return
    if out.status == "error":
        st.error(md(out.message))
        render_details(out)
        return

    a = out.analysis
    if a is not None:
        st.markdown(f"#### {md(a.answer)}")
        if a.insights:
            st.markdown("\n".join(f"- {md(i)}" for i in a.insights))
        if a.caveats:
            st.caption("Caveats: " + md(" ".join(a.caveats)))

    res = out.result
    df = res.to_dataframe()
    if out.chart is not None:
        render_chart(df, out.chart)
    label = f"Result table ({res.row_count} row{'s' if res.row_count != 1 else ''}"
    label += ", truncated)" if res.truncated else ")"
    with st.expander(label, expanded=out.chart is None or out.chart.kind == "metric"):
        st.dataframe(df, hide_index=True, width="stretch")
    render_details(out)


def render_details(out: PipelineResult) -> None:
    """SQL and pipeline internals, collapsed: useful for learning and debugging."""
    if not out.candidates:
        return
    with st.expander("SQL and how it was produced"):
        if out.sql:
            st.code(out.sql, language="sql")
        if len(out.attempts) > 1:
            st.markdown(f"**Self-correction:** {len(out.attempts) - 1} repair step(s)")
            for i, att in enumerate(out.attempts[:-1], 1):
                st.markdown(f"Attempt {i} failed with `{att.error}`")
                st.code(att.sql or "", language="sql")
        st.markdown("**Schema linking**")
        cand = ", ".join(f"{c.name} ({c.score:.2f})" for c in out.candidates)
        st.markdown(f"- Vector search candidates: {cand}")
        if out.selection is not None:
            st.markdown(
                f"- LLM selected: {', '.join(out.selection.tables) or 'none'}"
                + (f" - _{md(out.selection.reason)}_" if out.selection.reason else "")
            )
        extra = [t for t in out.tables_used if out.selection and t not in out.selection.tables]
        if extra:
            st.markdown(f"- Added to connect joins: {', '.join(extra)}")
        timing = " | ".join(f"{k.replace('_', ' ')} {v:.1f}s" for k, v in out.timings.items())
        st.caption(f"Timings: {timing} | total {out.total_s:.1f}s")


# ---------------------------------------------------------------------------- page
settings = get_settings()

with st.sidebar:
    st.header("Text2SQL Analyst")
    st.write(
        "Ask a question about the database in plain English. The app finds the relevant "
        "tables, writes SQL, runs it read-only and explains the result."
    )
    st.subheader("Try an example")
    for example in EXAMPLES:
        if st.button(example, width="stretch"):
            st.session_state.pending = example
    st.subheader("Configuration")
    st.markdown(
        f"- Database: `{settings.db_path.name}`\n"
        f"- Provider: `{settings.llm_provider}`\n"
        f"- SQL model: `{settings.sql_model}`\n"
        f"- Helper model: `{settings.helper_model}`\n"
        f"- Embeddings: `{settings.embedding_model}`"
    )
    if st.button("Clear conversation", width="stretch"):
        st.session_state.history = []

st.title("Ask your data")
st.caption(
    f"Connected to **{settings.db_path.name}**. Answers come from the data, not from "
    "the model's memory; open *SQL and how it was produced* to check the work."
)

try:
    pipeline = get_pipeline()
except Exception as exc:  # noqa: BLE001 - show setup problems in the UI, not a traceback
    st.error(f"**Setup problem:** {exc}")
    st.info(
        "See the README's *Setup* section: download the database, add your API key to "
        "`.env`, and run `python -m text2sql index`."
    )
    st.stop()

history: list[PipelineResult] = st.session_state.setdefault("history", [])
for past in history:
    with st.chat_message("user"):
        st.write(past.question)
    with st.chat_message("assistant"):
        render_result(past)

question = st.chat_input("e.g. Which country has the most customers?")
question = question or st.session_state.pop("pending", None)
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Finding tables, writing SQL, analysing..."):
            result = pipeline.ask(question)
        render_result(result)
    history.append(result)
