import sqlite3

import pytest

from text2sql.db import connect_readonly, read_schema, schema_by_name


def test_read_schema_finds_tables_columns_and_keys(shop_db):
    tables = schema_by_name(read_schema(shop_db, sample_rows=2))

    assert set(tables) == {"customers", "products", "orders", "order_items"}
    orders = tables["orders"]
    assert orders.column_names == ["order_id", "customer_id", "order_date"]
    assert orders.row_count == 3
    assert len(orders.sample_rows) == 2
    assert [(fk.column, fk.ref_table) for fk in orders.foreign_keys] == [
        ("customer_id", "customers")
    ]


def test_ddl_contains_keys(shop_db):
    items = schema_by_name(read_schema(shop_db))["order_items"]
    ddl = items.to_ddl()
    assert ddl.startswith('CREATE TABLE "order_items"')
    assert 'FOREIGN KEY ("product_id") REFERENCES "products" ("product_id")' in ddl


def test_samples_are_truncated(shop_db):
    customers = schema_by_name(read_schema(shop_db))["customers"]
    text = customers.samples_as_text(max_chars=4)
    assert text.splitlines()[0] == "customer_id | name | country"
    assert "B..." in text  # "Brazil" truncated to 4 chars


def test_connection_is_read_only(shop_db):
    conn = connect_readonly(shop_db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM customers")
    conn.close()


def test_missing_database_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_sample_db"):
        connect_readonly(tmp_path / "nope.db")
