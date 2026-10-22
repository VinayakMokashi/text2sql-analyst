"""Using text2sql as a library: no vector index, no optional extras, caller-supplied hints."""

from __future__ import annotations

import subprocess
import sys

from conftest import helper_replies

from text2sql.analysis import Analyst
from text2sql.config import Settings
from text2sql.db import read_schema
from text2sql.generation import SQLGenerator
from text2sql.llm import FakeLLM
from text2sql.pipeline import Pipeline
from text2sql.prompts import sql_generation_prompt, table_selection_prompt
from text2sql.retrieval import AllTablesRetriever, TableSelector

BLOCK_EXTRAS = """
import importlib.abc
import sys


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"chromadb", "streamlit", "altair"}:
            raise ImportError(f"{name} is not installed")
        return None


sys.meta_path.insert(0, Block())
import text2sql.pipeline  # noqa: E402,F401
import text2sql.retrieval  # noqa: E402,F401
print("ok")
"""


def test_the_pipeline_imports_without_the_optional_extras():
    result = subprocess.run(
        [sys.executable, "-c", BLOCK_EXTRAS], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_all_tables_retriever_offers_every_table(shop_db):
    tables = read_schema(shop_db)

    retriever = AllTablesRetriever(tables, {"Customers": "One row per customer."})

    found = retriever.search("anything", n=2)
    assert retriever.count() == 4
    assert [t.name for t in found] == [t.name for t in tables]
    by_name = {t.name: t for t in found}
    assert by_name["customers"].description == "One row per customer."
    assert by_name["orders"].description.startswith("Table orders with columns")


def test_a_pipeline_without_a_vector_index_answers_and_passes_hints_on(shop_db):
    tables = read_schema(shop_db)
    helper = FakeLLM(helper_replies('{"tables": ["customers"], "answerable": true}'))
    sql_llm = FakeLLM(["```sql\nSELECT country, COUNT(*) AS n FROM customers GROUP BY 1\n```"])
    pipe = Pipeline(
        Settings(db_path=shop_db, llm_provider="fake"),
        tables,
        AllTablesRetriever(tables),
        TableSelector(helper),
        SQLGenerator(sql_llm),
        Analyst(helper),
    )

    out = pipe.ask(
        "How many customers per country?", hints=["'Brasil' means country = 'Brazil'", " "]
    )

    assert out.ok, out.message
    assert out.result is not None and out.result.row_count == 3
    assert out.hints == ["'Brasil' means country = 'Brazil'"]
    selection_prompt = helper.calls[0][1]
    sql_prompt = sql_llm.calls[0][1]
    assert "Hints:\n- 'Brasil' means country = 'Brazil'" in selection_prompt
    assert "### Hints\n- 'Brasil' means country = 'Brazil'\n\n### Question" in sql_prompt


def test_prompts_are_unchanged_without_hints(shop_db):
    tables = read_schema(shop_db)

    with_none = sql_generation_prompt("q", tables)
    with_empty = sql_generation_prompt("q", tables, hints=["", "  "])

    assert with_none == with_empty
    assert "### Hints" not in with_none[1]
    assert "Hints:" not in table_selection_prompt("q", [], 3)[1]
