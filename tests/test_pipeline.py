"""End-to-end pipeline tests on the shop database, with scripted (fake) LLMs."""

import pytest

from text2sql.analysis import Analyst
from text2sql.config import Settings
from text2sql.db import read_schema
from text2sql.generation import SQLGenerator
from text2sql.indexing import build_index, open_collection
from text2sql.llm import FakeLLM, LLMError
from text2sql.pipeline import Pipeline
from text2sql.retrieval import TableRetriever, TableSelector

REVENUE_SQL = """```sql
SELECT c.country, ROUND(SUM(p.price * oi.quantity), 2) AS revenue
FROM customers c JOIN orders o ON o.customer_id = c.customer_id
JOIN order_items oi ON oi.order_id = o.order_id
JOIN products p ON p.product_id = oi.product_id
GROUP BY c.country ORDER BY revenue DESC
```"""


def helper_replies(selection_json: str):
    """Helper model: answers table selection and analysis prompts differently."""

    def reply(system: str, _user: str) -> str:
        if "which tables are needed" in system:
            return selection_json
        return '{"answer": "Brazil leads with 375.0.", "insights": ["Two countries"], ' \
               '"caveats": []}'

    return reply


@pytest.fixture()
def make_pipeline(shop_db, tmp_path, embedder):
    tables = read_schema(shop_db)
    index_dir = tmp_path / "index"
    build_index(tables, index_dir, embedder, llm=None)

    def factory(sql_replies, selection='{"tables": ["customers", "products"], '
                '"answerable": true}', max_retries=2):
        settings = Settings(db_path=shop_db, max_retries=max_retries, llm_provider="fake")
        helper = FakeLLM(helper_replies(selection))
        sql_llm = FakeLLM(sql_replies)
        pipe = Pipeline(
            settings,
            tables,
            TableRetriever(open_collection(index_dir), embedder),
            TableSelector(helper),
            SQLGenerator(sql_llm),
            Analyst(helper),
        )
        return pipe, sql_llm

    return factory


def test_happy_path_returns_answer_table_chart_and_sql(make_pipeline):
    pipe, _ = make_pipeline([REVENUE_SQL])
    out = pipe.ask("Which countries bring in the most revenue?")

    assert out.ok, out.message
    # The two bridge tables were added automatically from the foreign keys.
    assert out.tables_used == ["customers", "products", "order_items", "orders"]
    assert out.result.rows == [("Brazil", 375.0), ("USA", 150.0)]
    assert out.analysis.answer == "Brazil leads with 375.0."
    assert (out.chart.kind, out.chart.x, out.chart.y) == ("bar", "country", "revenue")
    assert out.sql.startswith("SELECT c.country")
    assert len(out.attempts) == 1 and out.attempts[0].error is None
    assert {"retrieval", "table_selection", "sql_generation", "execution", "analysis"} <= set(
        out.timings
    )


def test_failed_sql_is_repaired_using_the_error_message(make_pipeline):
    pipe, sql_llm = make_pipeline(["SELECT contry FROM customers", "SELECT country FROM customers"])
    out = pipe.ask("List customer countries")

    assert out.ok
    assert [a.error is None for a in out.attempts] == [False, True]
    assert "no such column: contry" in out.attempts[0].error
    assert "no such column: contry" in sql_llm.calls[1][1]  # error fed back to the model


def test_unsafe_sql_is_rejected_and_repaired(make_pipeline):
    pipe, _ = make_pipeline(["DELETE FROM customers", "SELECT COUNT(*) AS n FROM customers"])
    out = pipe.ask("How many customers?")
    assert out.ok
    assert "Only SELECT" in out.attempts[0].error
    assert out.result.rows == [(3,)]
    assert out.chart.kind == "metric"


def test_gives_up_after_max_retries(make_pipeline):
    pipe, sql_llm = make_pipeline(["SELECT nope FROM customers"], max_retries=2)
    out = pipe.ask("q")
    assert out.status == "error"
    assert len(out.attempts) == 3
    assert "after 3 attempt(s)" in out.message
    assert len(sql_llm.calls) == 3


def test_unanswerable_from_table_selection(make_pipeline):
    pipe, sql_llm = make_pipeline(
        ["SELECT 1"], selection='{"tables": [], "answerable": false, "reason": "No weather data."}'
    )
    out = pipe.ask("Will it rain tomorrow?")
    assert out.status == "unanswerable"
    assert out.message == "No weather data."
    assert sql_llm.calls == []  # we stopped before generating SQL


def test_unanswerable_from_sql_model(make_pipeline):
    pipe, _ = make_pipeline(["CANNOT_ANSWER: the data has no employee salaries"])
    out = pipe.ask("What is the average salary?")
    assert out.status == "unanswerable"
    assert "salaries" in out.message


def test_llm_failure_becomes_readable_error(make_pipeline):
    pipe, _ = make_pipeline(["SELECT 1"])

    def boom(*_args, **_kwargs):
        raise LLMError("rate limited")

    pipe.generator.llm.complete = boom
    out = pipe.ask("anything")
    assert out.status == "error"
    assert "rate limited" in out.message


def test_empty_question(make_pipeline):
    pipe, _ = make_pipeline(["SELECT 1"])
    assert pipe.ask("   ").status == "error"
