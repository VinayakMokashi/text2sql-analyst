import pandas as pd
import pytest

from text2sql.analysis import Analyst, measure_columns, suggest_chart, summary_stats
from text2sql.execution import QueryResult
from text2sql.llm import FakeLLM


def make_result(columns, rows, truncated=False):
    return QueryResult("SELECT ...", columns, rows, truncated, 0.01)


GENRES = make_result(
    ["genre", "total_revenue"], [("Rock", 826.65), ("Latin", 382.14), ("Metal", 261.36)]
)


# ---------------------------------------------------------------------- column roles
def test_measures_skip_ids_and_years():
    df = pd.DataFrame({"CustomerId": [1, 2], "year": [2021, 2022], "total": [5.0, 7.5]})
    assert measure_columns(df) == ["total"]


def test_summary_stats_are_exact_and_labelled():
    stats = summary_stats(GENRES.to_dataframe())
    assert "total 1,470.15" in stats
    assert "max at genre=Rock (56.2% of total)" in stats
    assert "min at Metal" in stats


def test_summary_stats_state_first_to_last_change_for_time_series():
    df = pd.DataFrame(
        {"year": [2011, 2009, 2010], "sales": [450.58, 449.46, 481.45]}  # unsorted on purpose
    )
    stats = summary_stats(df)
    assert "sales from year=2009 to year=2011: 449.46 -> 450.58 (overall change +1.12" in stats


# ---------------------------------------------------------------------------- charts
@pytest.mark.parametrize(
    "df, kind, x",
    [
        (pd.DataFrame({"n": [42]}), "metric", None),
        (GENRES.to_dataframe(), "bar", "genre"),
        (pd.DataFrame({"year": ["2021", "2022", "2023"], "sales": [1, 3, 2]}), "line", "year"),
        (
            pd.DataFrame({"month": ["2021-01", "2021-02", "2021-03"], "n": [1, 3, 2]}),
            "line",
            "month",
        ),
    ],
)
def test_chart_heuristics(df, kind, x):
    spec = suggest_chart(df)
    assert spec is not None
    assert (spec.kind, spec.x) == (kind, x)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame({"name": ["a", "b"], "country": ["x", "y"]}),  # nothing to measure
        pd.DataFrame({"name": [f"n{i}" for i in range(40)], "v": range(40)}),  # too many bars
        pd.DataFrame({"name": ["a", "a"], "v": [1, 2]}),  # duplicate labels
    ],
)
def test_no_chart_when_it_would_mislead(df):
    assert suggest_chart(df) is None


# --------------------------------------------------------------------------- analyst
def test_analyst_parses_json_answer():
    llm = FakeLLM(
        ['{"answer": "Rock leads with $826.65.", "insights": ["Rock is 56%"], "caveats": []}']
    )
    analysis = Analyst(llm).analyze("Top genres by revenue?", GENRES)
    assert analysis.answer == "Rock leads with $826.65."
    assert analysis.insights == ["Rock is 56%"]
    assert "max at genre=Rock" in llm.calls[0][1]  # stats were given to the model


def test_analyst_skips_llm_for_empty_results():
    llm = FakeLLM()
    analysis = Analyst(llm).analyze("q", make_result(["a"], []))
    assert "No matching data" in analysis.answer
    assert llm.calls == []


def test_analyst_falls_back_to_plain_text_and_flags_truncation():
    llm = FakeLLM(["Rock is the top genre."])
    analysis = Analyst(llm).analyze("q", make_result(GENRES.columns, GENRES.rows, True))
    assert analysis.answer == "Rock is the top genre."

    llm = FakeLLM(['{"answer": "Rock.", "caveats": []}'])
    analysis = Analyst(llm).analyze("q", make_result(GENRES.columns, GENRES.rows, True))
    assert analysis.caveats == ["Only the first 3 rows were returned."]
