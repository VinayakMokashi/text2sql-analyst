import sqlite3

import pytest

from text2sql.db import connect_readonly, read_schema, schema_by_name
from text2sql.prompts import sql_generation_prompt


def make_db(path, script):
    with sqlite3.connect(path) as conn:
        conn.executescript(script)
    return path


def test_read_schema_finds_tables_columns_and_keys(shop_db):
    tables = schema_by_name(read_schema(shop_db, sample_rows=2))

    assert set(tables) == {"customers", "products", "orders", "order_items"}
    orders = tables["orders"]
    assert orders.column_names == ["order_id", "customer_id", "order_date"]
    assert orders.row_count == 3
    assert len(orders.sample_rows) == 2
    assert [(fk.columns, fk.ref_table, fk.ref_columns) for fk in orders.foreign_keys] == [
        (("customer_id",), "customers", ("customer_id",))
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


def test_path_with_spaces_and_hash_opens(tmp_path):
    db = make_db(tmp_path / "my data #1.db", "CREATE TABLE t (x INTEGER);")
    assert connect_readonly(db).execute("SELECT COUNT(*) FROM t").fetchone() == (0,)


# ----------------------------------------------------------------- unusual schemas
def test_foreign_key_without_parent_column_uses_the_parent_primary_key(tmp_path):
    db = make_db(
        tmp_path / "a.db",
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT);"
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers);",
    )
    tables = schema_by_name(read_schema(db))
    fk = tables["orders"].foreign_keys[0]
    assert fk.ref_columns == ("id",)
    _, user = sql_generation_prompt("q", [tables["orders"], tables["customers"]])
    assert '"orders"."customer_id" = "customers"."id"' in user
    assert "None" not in user


def test_composite_foreign_key_is_one_key(tmp_path):
    db = make_db(
        tmp_path / "a.db",
        "CREATE TABLE parent (a INTEGER, b INTEGER, PRIMARY KEY (a, b));"
        "CREATE TABLE child (pa INTEGER, pb INTEGER, "
        "FOREIGN KEY (pa, pb) REFERENCES parent (a, b));",
    )
    tables = schema_by_name(read_schema(db))
    assert len(tables["child"].foreign_keys) == 1
    assert 'FOREIGN KEY ("pa", "pb") REFERENCES "parent" ("a", "b")' in tables["child"].to_ddl()
    _, user = sql_generation_prompt("q", [tables["child"], tables["parent"]])
    assert '"child"."pa" = "parent"."a" AND "child"."pb" = "parent"."b"' in user


def test_generated_columns_line_up_with_sample_rows(tmp_path):
    db = make_db(
        tmp_path / "a.db",
        "CREATE TABLE prices (net REAL, gross REAL GENERATED ALWAYS AS (net * 1.2), cur TEXT);"
        "INSERT INTO prices (net, cur) VALUES (100, 'EUR');",
    )
    prices = read_schema(db)[0]
    assert prices.column_names == ["net", "gross", "cur"]
    assert prices.samples_as_text().splitlines()[1] == "100.0 | 120.0 | EUR"


def test_tables_starting_with_sqlite_letters_are_kept_and_quotes_are_escaped(tmp_path):
    db = make_db(
        tmp_path / "a.db", 'CREATE TABLE SQLiteVersions (v TEXT); CREATE TABLE "my""table" (x);'
    )
    tables = read_schema(db)
    assert {t.name for t in tables} == {"SQLiteVersions", 'my"table'}
    quoted = next(t for t in tables if t.name == 'my"table')
    assert quoted.to_ddl().startswith('CREATE TABLE "my""table" (')
