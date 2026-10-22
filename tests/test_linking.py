"""Value linking: column catalogues, value matching and the pipeline's linker hook."""

from __future__ import annotations

import pytest
from conftest import helper_replies

from text2sql.analysis import Analyst
from text2sql.config import Settings
from text2sql.db import read_schema
from text2sql.generation import SQLGenerator
from text2sql.linking import ColumnCatalogue, StaticLinker, ValueMatcher, sql_literal
from text2sql.llm import FakeLLM
from text2sql.pipeline import Pipeline
from text2sql.retrieval import AllTablesRetriever, TableSelector


def test_a_catalogue_lists_a_column_s_values_with_labels(shop_db):
    plain = ColumnCatalogue.read(shop_db, "products", "category")
    labelled = ColumnCatalogue.read(shop_db, "products", "title", "category", name="item")

    assert plain.hint() == "products.category takes only these values: Books, Music."
    assert (
        labelled.hint()
        == "item takes only these values: Drum (Music), Guitar (Music), Novel (Books)."
    )
    with pytest.raises(ValueError, match="too many for a catalogue"):
        ColumnCatalogue.read(shop_db, "customers", "name", max_values=2)


def test_a_matcher_finds_values_and_aliases_as_whole_words(shop_db):
    matcher = ValueMatcher.read(shop_db, "customers", "country", {"Brasil": "Brazil", "Mars": "?"})

    assert matcher.matches("Orders from brasil and the usa, not Usage?") == [
        ("brasil", "Brazil"),
        ("usa", "USA"),
    ]
    assert matcher.hints("Ana in Brazil") == ["\"Brazil\" is customers.country = 'Brazil'."]
    assert matcher.matches("Mars") == []


def test_longer_names_win_and_quotes_are_escaped():
    matcher = ValueMatcher("t.name", ["America", "American Express", "McDonald's"])

    assert matcher.matches("Did American Express beat McDonald's?") == [
        ("American Express", "American Express"),
        ("McDonald's", "McDonald's"),
    ]
    assert sql_literal("McDonald's") == "'McDonald''s'"
    assert matcher.hints("McDonald's") == ["\"McDonald's\" is t.name = 'McDonald''s'."]


def test_a_pipeline_linker_adds_hints_before_the_caller_s(shop_db):
    tables = read_schema(shop_db)
    helper = FakeLLM(helper_replies('{"tables": ["customers"], "answerable": true}'))
    sql_llm = FakeLLM(["```sql\nSELECT name FROM customers WHERE country = 'Brazil'\n```"])
    linker = StaticLinker(
        catalogues=[ColumnCatalogue.read(shop_db, "products", "category")],
        matchers=[ValueMatcher.read(shop_db, "customers", "country", {"Brasil": "Brazil"})],
        notes=["Prices are in dollars."],
    )
    pipe = Pipeline(
        Settings(db_path=shop_db, llm_provider="fake"),
        tables,
        AllTablesRetriever(tables),
        TableSelector(helper),
        SQLGenerator(sql_llm),
        Analyst(helper),
        linker=linker,
    )

    out = pipe.ask("Which customers live in Brasil?", hints=["Answer with names."])

    assert out.ok, out.message
    assert out.hints == [
        "\"Brasil\" is customers.country = 'Brazil'.",
        "products.category takes only these values: Books, Music.",
        "Prices are in dollars.",
        "Answer with names.",
    ]
    assert "- \"Brasil\" is customers.country = 'Brazil'." in sql_llm.calls[0][1]
    assert "value_linking" in out.timings
