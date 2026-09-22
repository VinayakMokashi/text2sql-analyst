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

# Sidebar examples for the bundled sample databases (keyed by file name); any other
# database simply shows no examples.
EXAMPLES = {
    "chinook": [
        "Which 5 genres generate the most revenue?",
        "How did total sales change from year to year?",
        "Who are the top 10 customers by total spending, and where are they from?",
        "Which sales support agent is responsible for the most revenue?",
        "Which artists have the most albums in the catalogue?",
        "What is the average invoice total per country?",
        "What is the salary of each employee?",
    ],
    "sakila": [
        "Which film categories generate the most rental revenue?",
        "How did the total payment amount change from month to month?",
        "Which 10 actors appear in the most films?",
        "Which countries have the most customers?",
        "How many films are there for each rating?",
        "Which films won an Academy Award?",
    ],
}

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
def format_number(value: object) -> str:
    """Readable number for the metric tile: 1,234.5, 0.004 or 42."""
    if isinstance(value, float):
        if value != 0 and abs(value) < 1:
            return f"{value:.3g}"  # 0.004 stays 0.004, not "0"
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    try:
        return f"{value:,}"
    except (TypeError, ValueError):
        return str(value)


def render_chart(df: pd.DataFrame, spec: ChartSpec) -> None:
    if spec.kind == "metric":
        st.metric(label=spec.y.replace("_", " "), value=format_number(df[spec.y].iloc[0]))
        return

    # Altair reads field names as shorthand ("count(*)" looks like an aggregate, "a.b"
    # like a nested field), so plot copies under fixed names and show the originals as
    # titles.
    data = pd.DataFrame({"label": df[spec.x], "value": df[spec.y]})
    color = SERIES_COLOR[theme()]
    tooltip = [
        alt.Tooltip("label", title=spec.x),
        alt.Tooltip("value", title=spec.y, format=",.2f"),
    ]
    if spec.kind == "bar":
        # Horizontal bars sorted by value: long category names stay readable and the
        # ranking is visible at a glance.
        chart = (
            alt.Chart(data)
            .mark_bar(color=color, cornerRadiusEnd=4, height={"band": 0.7})
            .encode(
                x=alt.X("value", type="quantitative", title=spec.y, axis=alt.Axis(tickCount=5)),
                y=alt.Y("label", type="nominal", sort="-x", title=None),
                tooltip=tooltip,
            )
            # The height includes the axes (Streamlit fits the chart to it), so give
            # each bar ~32px plus room for the x-axis and its title.
            .properties(height=70 + 32 * len(data))
        )
    else:  # line
        chart = (
            alt.Chart(data.sort_values("label"))
            .mark_line(color=color, strokeWidth=2, point=alt.OverlayMarkDef(size=64, color=color))
            .encode(
                x=alt.X("label", type="ordinal", title=None, axis=alt.Axis(labelAngle=0)),
                # Unlike bars, a line needs no zero baseline; starting the axis near the
                # data keeps small changes over time visible.
                y=alt.Y("value", type="quantitative", title=spec.y, scale=alt.Scale(zero=False)),
                tooltip=tooltip,
            )
            .properties(height=280)
        )
    st.altair_chart(chart, width="stretch")


# ------------------------------------------------------------------------- results
def render_result(out: PipelineResult) -> None:
    if out.interpreted_as:
        # Show how a follow-up was understood, so a wrong reading is easy to spot.
        st.caption(f"Interpreted as: *{md(out.interpreted_as)}*")
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
def reload_everything() -> None:
    """Pick up a changed .env or a rebuilt index without restarting the server."""
    get_settings.cache_clear()
    get_pipeline.clear()


# Settings are read inside the try: an invalid T2S_* value should show the setup box
# below, not a traceback.
try:
    settings = get_settings()
    setup_error: Exception | None = None
except Exception as exc:  # noqa: BLE001
    settings, setup_error = None, exc

with st.sidebar:
    st.header("Text2SQL Analyst")
    st.write(
        "Ask a question about the database in plain English. The app finds the relevant "
        "tables, writes SQL, runs it read-only and explains the result."
    )
    examples = EXAMPLES.get(settings.db_path.stem.lower(), []) if settings else []
    if examples:
        st.subheader("Try an example")
    for example in examples:
        if st.button(example, width="stretch"):
            st.session_state.pending = example
    if settings:
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
    if st.button("Reload settings and index", width="stretch"):
        reload_everything()
        st.rerun()

st.title("Ask your data")
if settings:
    st.caption(
        f"Connected to **{settings.db_path.name}**. Answers come from the data, not from "
        "the model's memory; open *SQL and how it was produced* to check the work."
    )

try:
    if setup_error is not None:
        raise setup_error
    pipeline = get_pipeline()
except Exception as exc:  # noqa: BLE001 - show setup problems in the UI, not a traceback
    st.error(f"**Setup problem:** {md(str(exc))}")
    st.info(
        "See the README's *Setup* section: download the database, add your API key to "
        "`.env`, and run `python -m text2sql index`. Then click *Reload settings and "
        "index* in the sidebar."
    )
    st.stop()

history: list[PipelineResult] = st.session_state.setdefault("history", [])
for past in history:
    with st.chat_message("user"):
        st.markdown(md(past.question))
    with st.chat_message("assistant"):
        render_result(past)

question = st.chat_input('Ask a question, or a follow-up like "and for 2012?"')
question = question or st.session_state.pop("pending", None)
if question:
    with st.chat_message("user"):
        st.markdown(md(question))
    with st.chat_message("assistant"):
        with st.spinner("Finding tables, writing SQL, analysing..."):
            # Earlier answers give follow-up questions their context.
            context = [past.as_turn() for past in history if past.status != "error"]
            result = pipeline.ask(question, history=context)
        render_result(result)
    history.append(result)
