import json

import pytest

from text2sql.db import read_schema, schema_by_name
from text2sql.indexing import build_index, load_descriptions, open_collection
from text2sql.llm import FakeLLM
from text2sql.retrieval import RetrievedTable, TableRetriever, TableSelector, add_join_tables


@pytest.fixture()
def shop_tables(shop_db):
    return read_schema(shop_db)


@pytest.fixture()
def shop_index(tmp_path, shop_tables, embedder):
    index_dir = tmp_path / "index"
    descriptions = {
        "customers": "People who buy things, with their name and home country.",
        "products": "Items for sale with a title, category and unit price.",
        "orders": "One purchase made by a customer on a date.",
        "order_items": "Line items: which product and what quantity each order contains.",
    }
    llm = FakeLLM(lambda _s, user: next(d for n, d in descriptions.items() if f'"{n}"' in user))
    build_index(shop_tables, index_dir, embedder, llm)
    return index_dir


# ------------------------------------------------------------------------ indexing
def test_index_stores_descriptions_and_vectors(shop_index):
    docs = load_descriptions(shop_index)
    assert docs["products"].startswith("Items for sale")
    assert open_collection(shop_index).count() == 4


def test_reindex_reuses_cached_descriptions(shop_index, shop_tables, embedder):
    llm = FakeLLM(["should not be called"])
    build_index(shop_tables, shop_index, embedder, llm)
    assert llm.calls == []


def test_index_without_llm_uses_template_descriptions(tmp_path, shop_tables, embedder):
    build_index(shop_tables, tmp_path / "idx", embedder, llm=None)
    docs = json.loads((tmp_path / "idx" / "table_docs.json").read_text())
    assert docs["orders"] == (
        "Table orders with columns order_id, customer_id, order_date. It links to customers."
    )


# ----------------------------------------------------------------------- retrieval
def test_retriever_ranks_relevant_table_first(shop_index, embedder):
    retriever = TableRetriever(open_collection(shop_index), embedder)
    hits = retriever.search("Which country has the most customers?", n=2)
    assert hits[0].name == "customers"
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


def test_retriever_on_empty_index_explains_what_to_do(tmp_path, embedder):
    retriever = TableRetriever(open_collection(tmp_path / "empty"), embedder)
    with pytest.raises(RuntimeError, match="text2sql index"):
        retriever.search("anything", n=3)


# ----------------------------------------------------------------------- selection
def _candidates(*names):
    return [RetrievedTable(n, f"about {n}", 0.5) for n in names]


def test_selector_parses_json_and_fixes_casing(shop_tables):
    llm = FakeLLM(['Sure! ```json\n{"tables": ["PRODUCTS", "unknown"], "answerable": true}\n```'])
    sel = TableSelector(llm).select(
        "price of guitars", _candidates("products", "orders"), schema_by_name(shop_tables), 3
    )
    assert sel.tables == ["products"]
    assert sel.answerable and not sel.fallback


def test_selector_reports_unanswerable(shop_tables):
    llm = FakeLLM(['{"tables": [], "answerable": false, "reason": "No weather data."}'])
    sel = TableSelector(llm).select(
        "weather tomorrow?", _candidates("products"), schema_by_name(shop_tables), 3
    )
    assert not sel.answerable
    assert sel.reason == "No weather data."


def test_selector_falls_back_to_top_k_on_garbage(shop_tables):
    llm = FakeLLM(["I think you need the products table."])
    sel = TableSelector(llm).select(
        "q", _candidates("products", "orders", "customers"), schema_by_name(shop_tables), 2
    )
    assert sel.fallback
    assert sel.tables == ["products", "orders"]


# --------------------------------------------------------------------------- joins
def test_join_expansion_adds_bridge_table(shop_tables):
    # customers -> orders -> order_items -> products: two bridges are needed.
    tables = add_join_tables(["customers", "products"], shop_tables)
    assert tables[:2] == ["customers", "products"]
    assert set(tables[2:]) == {"orders", "order_items"}


def test_join_expansion_keeps_directly_connected_tables(shop_tables):
    assert add_join_tables(["orders", "customers"], shop_tables) == ["orders", "customers"]
    assert add_join_tables(["products"], shop_tables) == ["products"]
