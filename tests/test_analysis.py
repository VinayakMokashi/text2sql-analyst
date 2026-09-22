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


def test_summary_stats_give_the_gap_and_the_top_three_share():
    # Sakila's category revenue: a model once wrote the 1,896.49 gap as "896.49".
    df = pd.DataFrame(
        {
            "category": ["Sports", "Sci-Fi", "Animation", "Drama", "Music"],
            "revenue": [5314.21, 4756.98, 4656.30, 4587.39, 3417.72],
        }
    )
    stats = summary_stats(df)
    assert "max minus min 1,896.49" in stats
    assert "top 3 together 64.8% of total" in stats
    assert "top 3 together" not in summary_stats(GENRES.to_dataframe())  # 3 rows: all of it


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


# ---------------------------------------------------------------- review fixes
def test_names_ending_in_id_letters_are_not_ids():
    df = pd.DataFrame({"customer": ["a", "b"], "total_paid": [5.0, 7.5], "CustomerId": [1, 2]})
    assert measure_columns(df) == ["total_paid"]


def test_integer_months_and_nullable_years_are_time_not_measures():
    df = pd.DataFrame({"month": [1, 2, 3], "revenue": [10.0, 12.0, 9.0]})
    assert measure_columns(df) == ["revenue"]
    assert suggest_chart(df).kind == "line"
    years = pd.DataFrame({"year": [2009.0, None, 2011.0], "n": [1, 2, 3]})
    assert measure_columns(years) == ["n"]


def test_no_share_of_total_with_negative_values():
    df = pd.DataFrame({"region": ["North", "South", "East"], "profit": [120, -100, -15]})
    stats = summary_stats(df)
    assert "% of total" not in stats
    assert "max at region=North" in stats


def test_no_trend_line_when_periods_repeat():
    df = pd.DataFrame(
        {
            "year": [2009, 2009, 2010, 2010],
            "genre": ["A", "B", "A", "B"],
            "sales": [10.0, 90.0, 60.0, 20.0],
        }
    )
    assert "overall change" not in summary_stats(df)


def test_trend_percentage_uses_the_size_of_the_start_value():
    df = pd.DataFrame({"year": [2020, 2021], "profit": [-200.0, -100.0]})
    assert "overall change +100.00, +50.0%" in summary_stats(df)


def test_truncation_caveat_always_comes_first():
    llm = FakeLLM(['{"answer": "Rock.", "caveats": ["a", "b", "c row"]}'])
    analysis = Analyst(llm).analyze("q", make_result(GENRES.columns, GENRES.rows, True))
    assert analysis.caveats[0] == "Only the first 3 rows were returned."
    assert len(analysis.caveats) == 3
