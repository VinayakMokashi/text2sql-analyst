"""PostgreSQL support: translation rules, guardrails and URL handling run anywhere; the
tests marked ``postgres`` need a server, named by T2S_TEST_POSTGRES_URL (CI starts one).

The integration tests load the same shop data into SQLite and PostgreSQL and require
every query to give the same rows on both.
"""

from __future__ import annotations

import math
import os
import sqlite3
import time
import uuid
from functools import partial

import pytest
from conftest import SHOP_DDL, SHOP_ROWS

from text2sql.config import Settings
from text2sql.db import (
    DatabaseUnavailableError,
    connect_postgres,
    database_id,
    is_postgres_url,
    read_postgres_schema,
    redact_url,
    schema_by_name,
)
from text2sql.execution import (
    QueryTimeoutError,
    SQLExecutionError,
    UnsafeSQLError,
    execute_query,
    to_postgres,
    validate_sql,
)
from text2sql.execution.postgres import sqlite_column_names
from text2sql.linking import ColumnCatalogue

psycopg = pytest.importorskip("psycopg")

SERVER_URL = os.environ.get("T2S_TEST_POSTGRES_URL")
needs_server = pytest.mark.skipif(not SERVER_URL, reason="T2S_TEST_POSTGRES_URL is not set")

# A customer with no country, so NULL ordering is tested too.
EXTRA_ROWS = "INSERT INTO customers VALUES (4, 'dana', NULL);"


# ------------------------------------------------------------------ translation rules
# Columns are aliased here, so the rules are seen apart from the column naming.
def test_like_becomes_ilike_and_keeps_not():
    assert to_postgres("SELECT * FROM t WHERE name LIKE 'a%'") == (
        "SELECT * FROM t WHERE name ILIKE 'a%'"
    )
    assert to_postgres("SELECT * FROM t WHERE name NOT LIKE 'a%'") == (
        "SELECT * FROM t WHERE name NOT ILIKE 'a%'"
    )


def test_real_casts_keep_eight_bytes():
    assert to_postgres("SELECT CAST(v AS REAL) AS x FROM t") == (
        "SELECT CAST(v AS DOUBLE PRECISION) AS x FROM t"
    )


def test_round_to_places_rounds_a_numeric():
    assert to_postgres("SELECT ROUND(v, 2) AS x, ROUND(v) AS y FROM t") == (
        "SELECT ROUND(CAST(v AS DECIMAL), 2) AS x, ROUND(v) AS y FROM t"
    )


def test_comparisons_count_as_integers_in_sums():
    assert to_postgres("SELECT SUM(a > b) AS x, AVG(a = 1 AND b = 2) AS y, SUM(a) AS z FROM t") == (
        "SELECT SUM(CAST(a > b AS INT)) AS x, AVG(CAST(a = 1 AND b = 2 AS INT)) AS y, "
        "SUM(a) AS z FROM t"
    )


def test_total_is_a_sum_that_never_returns_null():
    assert to_postgres("SELECT TOTAL(v) AS x FROM t") == (
        "SELECT CAST(COALESCE(SUM(v), 0) AS DOUBLE PRECISION) AS x FROM t"
    )


def test_sqlglot_keeps_sqlite_null_order_and_division():
    assert to_postgres("SELECT a FROM t ORDER BY a") == "SELECT a FROM t ORDER BY a NULLS FIRST"
    assert "NULLIF" in to_postgres("SELECT a / b AS x FROM t")


def test_column_names_follow_sqlite():
    sql = "SELECT ROUND( price/7 ,2 ), c.NAME, (price), COUNT(*) AS nOrders, 1 FROM t c"
    assert sqlite_column_names(sql) == ["ROUND( price/7 ,2 )", None, None, "nOrders", "1"]
    assert sqlite_column_names("SELECT a + 1 FROM t UNION SELECT b FROM u") == ["a + 1"]
    assert sqlite_column_names("WITH x AS (SELECT 1 AS v) SELECT DISTINCT v * 2 FROM x") == [
        "v * 2"
    ]
    assert sqlite_column_names("SELECT *, price FROM t") is None
    assert sqlite_column_names("SELECT t.* FROM t") is None


def test_alias_references_in_where_and_having_are_expanded():
    sql = (
        "SELECT name, SUM(v) AS total FROM t WHERE name <> 'x' "
        "GROUP BY name HAVING total > 10 ORDER BY total"
    )
    out = to_postgres(sql, columns={"name", "v"})
    assert "HAVING SUM(v) > 10" in out
    assert out.endswith("ORDER BY total NULLS FIRST")  # PostgreSQL accepts aliases there
    where = to_postgres("SELECT v * 2 AS dbl FROM t WHERE dbl > 1", columns={"v"})
    assert "WHERE v * 2 > 1" in where


def test_real_columns_and_cte_outputs_are_not_expanded():
    # "v" is a column of the database: SQLite reads it as the column, not the alias.
    assert "WHERE v > 1" in to_postgres("SELECT v * 2 AS v FROM t WHERE v > 1", columns={"v"})
    # "w" is an output of the CTE, so the outer WHERE means the CTE's column.
    cte = "WITH x AS (SELECT v AS w FROM t) SELECT w * 2 AS w FROM x WHERE w > 1"
    assert "WHERE w > 1" in to_postgres(cte, columns={"v"})
    # Without the database's columns nothing is expanded.
    assert "HAVING n > 1" in to_postgres("SELECT COUNT(*) AS n FROM t HAVING n > 1")


def test_postgres_sql_passes_through_unchanged():
    sql = "SELECT name FROM t WHERE name LIKE 'A%'"
    assert to_postgres(sql, read="postgres") == sql


# ------------------------------------------------------------------ guardrails
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT pg_sleep(10)",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT repeat('x', 1000000000)",
        "SELECT set_config('statement_timeout', '0', false)",
        "SELECT * FROM dblink('host=x', 'select 1') AS t(a int)",
    ],
)
def test_postgres_functions_that_escape_a_read_only_query_are_refused(sql):
    with pytest.raises(UnsafeSQLError, match="not allowed"):
        validate_sql(sql, "postgres")


def test_writes_are_refused_in_postgres_dialect():
    with pytest.raises(UnsafeSQLError):
        validate_sql("DELETE FROM customers", "postgres")
    with pytest.raises(UnsafeSQLError):
        validate_sql("SELECT 1; DROP TABLE customers", "postgres")


def test_translated_sql_is_checked_again():
    # Written for SQLite, where pg_sleep means nothing; refused once it is PostgreSQL.
    with pytest.raises(UnsafeSQLError, match="(?i)pg_sleep"):
        execute_query("postgresql://u:p@127.0.0.1:1/db", "SELECT pg_sleep(5)")


# ------------------------------------------------------------------ URLs and settings
def test_urls_are_recognised_and_redacted():
    url = "postgresql://reader:s3cret@db.internal:5432/facts"
    assert is_postgres_url(url) and is_postgres_url("postgres://x@y/z")
    assert not is_postgres_url("data/chinook.db")
    assert redact_url(url) == "postgresql://reader:***@db.internal:5432/facts"
    assert "s3cret" not in database_id(url)


def test_settings_prefer_the_url(tmp_path):
    url = "postgresql://reader:pw@localhost/facts"
    s = Settings(db_path=tmp_path / "x.db", db_url=url, index_dir=tmp_path / "index")
    assert s.db_target == url
    assert s.db_index_dir == tmp_path / "index" / "facts"
    assert Settings(db_path=tmp_path / "x.db").db_target == tmp_path / "x.db"


def test_pipeline_runs_queries_through_its_executor(make_pipeline):
    pipe, _ = make_pipeline(["SELECT name FROM customers ORDER BY name"])
    calls = []

    def fake_executor(sql, max_rows, timeout_s, dialect):
        calls.append((sql, dialect))
        return execute_query(pipe.settings.db_path, sql, max_rows, timeout_s, dialect)

    pipe.executor = fake_executor
    result = pipe.ask("List the customers", analyze=False)
    assert result.status == "ok", result.message
    assert calls == [("SELECT name FROM customers ORDER BY name", "sqlite")]


def test_an_unreachable_server_is_reported_without_its_address():
    with pytest.raises(DatabaseUnavailableError) as info:
        connect_postgres("postgresql://u:secret@127.0.0.1:1/db")
    assert "127.0.0.1" not in str(info.value) and "secret" not in str(info.value)


def test_a_url_can_shorten_the_connect_timeout():
    # A blackholed address: without the URL's own timeout this would wait ten seconds.
    start = time.monotonic()
    with pytest.raises(DatabaseUnavailableError):
        connect_postgres("postgresql://u:p@10.255.255.1:5432/db?connect_timeout=1")
    assert time.monotonic() - start < 5


def test_pipeline_stops_without_repairs_when_the_server_is_down(make_pipeline):
    pipe, sql_llm = make_pipeline(["SELECT name FROM customers"])
    pipe.executor = partial(execute_query, "postgresql://u:p@127.0.0.1:1/db")
    result = pipe.ask("List the customers", analyze=False)
    assert result.status == "error"
    assert result.message == "The database could not be reached."
    assert len(sql_llm.calls) == 1  # no repair attempts


# ------------------------------------------------------------------ against a server
@pytest.fixture(scope="module")
def shop_pg():
    """The shop data in a fresh schema; the URL points its search_path there."""
    if not SERVER_URL:
        pytest.skip("T2S_TEST_POSTGRES_URL is not set")
    schema = f"t2s_{uuid.uuid4().hex[:8]}"
    ddl = SHOP_DDL.replace(" REAL", " DOUBLE PRECISION")
    with psycopg.connect(SERVER_URL, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA {schema}".encode())
        conn.execute(f"SET search_path TO {schema}".encode())
        for statement in (ddl + SHOP_ROWS + EXTRA_ROWS).split(";"):
            if statement.strip():
                conn.execute(statement.encode())
    sep = "&" if "?" in SERVER_URL else "?"
    yield f"{SERVER_URL}{sep}options=-csearch_path%3D{schema}"
    with psycopg.connect(SERVER_URL, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA {schema} CASCADE".encode())


@pytest.fixture()
def shop_sqlite(shop_db):
    with sqlite3.connect(shop_db) as conn:
        conn.executescript(EXTRA_ROWS)
    return shop_db


SAME_RESULTS = [
    "SELECT name FROM customers WHERE name LIKE 'a%' ORDER BY name",
    "SELECT country, COUNT(*) FROM customers GROUP BY country ORDER BY country",
    "SELECT name FROM customers ORDER BY country DESC, name",
    "SELECT title, ROUND(price / 7, 2) FROM products ORDER BY title",
    "SELECT SUM(price > 100) FROM products",
    "SELECT CAST(price AS REAL) / 3 FROM products ORDER BY product_id",
    "SELECT strftime('%Y', order_date) AS y, COUNT(*) FROM orders GROUP BY y ORDER BY y",
    "SELECT price / 0 FROM products ORDER BY product_id",
    "SELECT IFNULL(country, 'unknown') FROM customers ORDER BY customer_id",
    "SELECT AVG(quantity), TOTAL(quantity) FROM order_items",
    "SELECT TOTAL(quantity) FROM order_items WHERE quantity > 100",
    "SELECT MAX(price, 100) FROM products ORDER BY product_id",
    "SELECT c.name, SUM(p.price * oi.quantity) AS spent FROM customers c "
    "JOIN orders o USING (customer_id) JOIN order_items oi USING (order_id) "
    "JOIN products p USING (product_id) GROUP BY c.name ORDER BY spent DESC",
    # Mixed-case aliases, a column named in another case, and aliases in HAVING and WHERE.
    "SELECT c.NAME, COUNT(*) AS nOrders FROM customers c JOIN orders o USING (customer_id) "
    "GROUP BY c.name HAVING nOrders > 0 ORDER BY nOrders DESC, c.name",
    "SELECT title, price * 2 AS dblPrice FROM products WHERE dblPrice > 100 ORDER BY title",
    "SELECT (price), price AS Price2 FROM products ORDER BY product_id",
]


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-12)
    return a == b


@needs_server
@pytest.mark.parametrize("sql", SAME_RESULTS)
def test_sqlite_queries_give_the_same_rows_on_postgres(sql, shop_sqlite, shop_pg):
    lite = execute_query(shop_sqlite, sql)
    pg = execute_query(shop_pg, sql)
    assert pg.columns == lite.columns, pg.executed_sql
    assert len(pg.rows) == len(lite.rows), (pg.executed_sql, pg.rows, lite.rows)
    for pg_row, lite_row in zip(pg.rows, lite.rows, strict=True):
        assert all(_same(x, y) for x, y in zip(pg_row, lite_row, strict=True)), (
            pg.executed_sql,
            pg_row,
            lite_row,
        )
    assert pg.sql == lite.sql  # the query as written; executed_sql is PostgreSQL's
    assert pg.executed_sql and lite.executed_sql is None


@needs_server
def test_schema_is_read_like_sqlites(shop_pg, shop_sqlite):
    pg = schema_by_name(read_postgres_schema(shop_pg, sample_rows=2))
    assert set(pg) == {"customers", "products", "orders", "order_items"}
    orders = pg["orders"]
    assert orders.column_names == ["order_id", "customer_id", "order_date"]
    assert [c.name for c in orders.columns if c.primary_key] == ["order_id"]
    assert [(fk.columns, fk.ref_table, fk.ref_columns) for fk in orders.foreign_keys] == [
        (("customer_id",), "customers", ("customer_id",))
    ]
    assert orders.row_count == 3 and len(orders.sample_rows) == 2
    assert pg["products"].columns[3].type == "double precision"


@needs_server
def test_connections_refuse_writes_even_past_the_guardrails(shop_pg):
    with connect_postgres(shop_pg) as conn, pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        conn.execute("DELETE FROM customers")


@needs_server
def test_slow_queries_are_cancelled(shop_pg):
    with pytest.raises(QueryTimeoutError):
        execute_query(
            shop_pg,
            "SELECT COUNT(*) FROM generate_series(1, 500000000)",
            timeout_s=0.2,
            dialect="postgres",
        )


@needs_server
def test_server_errors_come_back_for_repair(shop_pg):
    with pytest.raises(SQLExecutionError, match="no_such_column"):
        execute_query(shop_pg, "SELECT no_such_column FROM customers")


@needs_server
def test_row_cap_and_truncation(shop_pg):
    result = execute_query(shop_pg, "SELECT * FROM customers ORDER BY customer_id", max_rows=2)
    assert result.row_count == 2 and result.truncated


@needs_server
def test_value_linking_reads_postgres(shop_pg, shop_sqlite):
    pg = ColumnCatalogue.read(shop_pg, "products", "category")
    lite = ColumnCatalogue.read(shop_sqlite, "products", "category")
    assert pg.values == lite.values == (("Books", None), ("Music", None))


@needs_server
def test_pipeline_answers_from_postgres(make_pipeline, shop_pg):
    pipe, _ = make_pipeline(["SELECT name FROM customers WHERE country LIKE 'b%'"])
    pipe.executor = partial(execute_query, shop_pg)
    result = pipe.ask("Customers in Brazil", analyze=False)
    assert result.status == "ok", result.message
    assert result.result.rows == [("Ana",)]
    assert "ILIKE" in result.result.executed_sql
