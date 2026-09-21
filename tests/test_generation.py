import pytest

from text2sql.db import read_schema, schema_by_name
from text2sql.generation import SQLGenerator, extract_sql, parse_reply
from text2sql.llm import FakeLLM
from text2sql.prompts import (
    analysis_prompt,
    sql_generation_prompt,
    sql_repair_prompt,
    table_description_prompt,
)


@pytest.fixture()
def tables(shop_db):
    by_name = schema_by_name(read_schema(shop_db))
    return [by_name["orders"], by_name["customers"]]


# ------------------------------------------------------------------- prompt building
def test_generation_prompt_contains_schema_samples_joins_and_question(tables):
    system, user = sql_generation_prompt("How many orders per country?", tables, "sqlite", 2)
    assert "SQLITE" in system and "CANNOT_ANSWER" in system
    assert 'CREATE TABLE "orders"' in user and 'CREATE TABLE "customers"' in user
    assert "2 example rows from orders" in user
    assert '"orders"."customer_id" = "customers"."customer_id"' in user
    assert user.rstrip().endswith("How many orders per country?")


def test_join_hints_only_cover_selected_tables(tables):
    _, user = sql_generation_prompt("q", tables[1:], "sqlite", 0)  # customers only
    assert "### Join conditions\n(none)" in user
    assert "example rows" not in user  # sample_rows=0 hides samples


def test_repair_prompt_includes_failed_sql_and_error(tables):
    _, user = sql_repair_prompt("q", tables, "SELECT nme FROM customers", "no such column: nme")
    assert "SELECT nme FROM customers" in user
    assert "no such column: nme" in user


def test_description_and_analysis_prompts(tables):
    _, user = table_description_prompt(tables[0])
    assert "Row count: 3" in user
    _, user = analysis_prompt("q?", "SELECT 1", 1, True, "a\n1", "stats")
    assert "cut off at the row limit" in user and '"answer"' in user


# --------------------------------------------------------------------- SQL extraction
@pytest.mark.parametrize(
    "reply, expected",
    [
        ("```sql\nSELECT * FROM t;\n```", "SELECT * FROM t"),
        ("Here you go:\n```\nSELECT a FROM t\n```\nThis counts rows.", "SELECT a FROM t"),
        ("The query is: SELECT count(*) FROM t;", "SELECT count(*) FROM t"),
        ("WITH x AS (SELECT 1) SELECT * FROM x", "WITH x AS (SELECT 1) SELECT * FROM x"),
    ],
)
def test_extract_sql(reply, expected):
    assert extract_sql(reply) == expected


def test_cannot_answer_is_detected():
    result = parse_reply("CANNOT_ANSWER: there is no weather data")
    assert result.sql is None
    assert result.cannot_answer == "there is no weather data"


def test_generator_and_repair_call_the_llm(tables):
    llm = FakeLLM(["```sql\nSELECT nme FROM customers\n```", "```sql\nSELECT name FROM customers\n```"])
    gen = SQLGenerator(llm)
    first = gen.generate("names?", tables)
    fixed = gen.repair("names?", tables, first.sql, "no such column: nme")
    assert (first.sql, fixed.sql) == ("SELECT nme FROM customers", "SELECT name FROM customers")
    assert "no such column: nme" in llm.calls[1][1]
