import sqlite3

import pytest

from text2sql.execution import (
    QueryTimeoutError,
    SQLExecutionError,
    UnsafeSQLError,
    execute_query,
    validate_sql,
)

# ------------------------------------------------------------------ static validation
SAFE = [
    "SELECT * FROM customers",
    "select name from customers where country = 'Brazil';",
    "WITH totals AS (SELECT customer_id, COUNT(*) n FROM orders GROUP BY 1) SELECT * FROM totals",
    "SELECT name FROM customers UNION SELECT title FROM products",
    "(SELECT 1)",
    "SELECT strftime('%Y', order_date) AS y, COUNT(*) FROM orders GROUP BY y",
]


@pytest.mark.parametrize("sql", SAFE)
def test_safe_queries_pass(sql):
    assert validate_sql(sql) == sql.strip().rstrip(";")


UNSAFE = [
    ("DELETE FROM customers", "Only SELECT"),
    ("DROP TABLE customers", "Only SELECT"),
    ("INSERT INTO customers VALUES (9, 'x', 'y')", "Only SELECT"),
    ("UPDATE customers SET name = 'x'", "Only SELECT"),
    ("REPLACE INTO customers VALUES (1, 'x', 'y')", "Only SELECT"),
    ("SELECT 1; DROP TABLE customers", "Exactly one statement"),
    ("PRAGMA table_info(customers)", "Only SELECT"),
    ("ATTACH DATABASE 'other.db' AS other", "Only SELECT"),
    ("VACUUM", "Only SELECT"),
    ("CREATE TABLE x AS SELECT * FROM customers", "Only SELECT"),
    ("SELECT load_extension('evil')", "not allowed"),
    ("WITH d AS (DELETE FROM customers RETURNING *) SELECT * FROM d", "Forbidden operation"),
    ("", "empty"),
    ("-- only a comment", "Exactly one statement"),
    ("SELEC name FROM customers", "could not be parsed"),
]


@pytest.mark.parametrize("sql, message", UNSAFE)
def test_unsafe_queries_are_rejected(sql, message):
    with pytest.raises(UnsafeSQLError, match=message):
        validate_sql(sql)


# ------------------------------------------------------------------------- execution
def test_execute_returns_rows_and_columns(shop_db):
    result = execute_query(shop_db, "SELECT name, country FROM customers ORDER BY name")
    assert result.columns == ["name", "country"]
    assert result.rows[0] == ("Ana", "Brazil")
    assert result.row_count == 3 and not result.truncated
    assert list(result.to_dataframe().columns) == ["name", "country"]


def test_row_limit_truncates_without_rewriting_sql(shop_db):
    sql = "SELECT * FROM order_items"
    result = execute_query(shop_db, sql, max_rows=2)
    assert result.row_count == 2
    assert result.truncated
    assert result.sql == sql


def test_database_errors_are_wrapped(shop_db):
    with pytest.raises(SQLExecutionError, match="no such column"):
        execute_query(shop_db, "SELECT nme FROM customers")


def test_unsafe_sql_never_reaches_the_database(shop_db):
    with pytest.raises(UnsafeSQLError):
        execute_query(shop_db, "DELETE FROM customers")
    assert execute_query(shop_db, "SELECT COUNT(*) FROM customers").rows == [(3,)]


def test_long_running_query_times_out(shop_db):
    endless = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    )
    with pytest.raises(QueryTimeoutError, match="longer than"):
        execute_query(shop_db, endless, timeout_s=0.2)


# ---------------------------------------------------------------- review fixes
def test_unbalanced_quotes_are_rejected_not_crashed():
    # sqlglot raises TokenError (not ParseError) here, e.g. for 'Guns N' Roses'.
    with pytest.raises(UnsafeSQLError, match="could not be parsed"):
        validate_sql("SELECT name FROM customers WHERE name = 'Guns N' Roses'")


def test_authorizer_refuses_what_the_parser_lets_through(shop_db):
    # pragma_table_info() is a SELECT to sqlglot, but not a data read to SQLite.
    with pytest.raises(SQLExecutionError, match="not authorized"):
        execute_query(shop_db, "SELECT * FROM pragma_table_info('customers')")


def test_authorizer_refuses_denied_functions_on_its_own(shop_db):
    # The layer must hold even without the parser check in front of it, so this runs
    # the authorizer directly on a real connection, with SQLite's own argument order.
    from contextlib import closing

    from text2sql.execution.executor import _authorize

    with closing(sqlite3.connect(shop_db)) as conn:
        # edit() exists only in the sqlite3 shell. A harmless stand-in makes it callable,
        # so the authorizer is the only thing that can refuse it.
        conn.create_function("edit", 1, lambda value: value)
        assert conn.execute("SELECT edit('x')").fetchone() == ("x",)
        conn.set_authorizer(_authorize)
        with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
            conn.execute("SELECT edit('x')")
        assert conn.execute("SELECT abs(-2)").fetchone() == (2,)  # ordinary functions run


def test_one_huge_value_cannot_be_built(shop_db):
    with pytest.raises(SQLExecutionError, match="too big"):
        execute_query(shop_db, "SELECT hex(zeroblob(40000000))")


def test_duplicate_column_names_are_made_unique(shop_db):
    result = execute_query(
        shop_db,
        "SELECT c.name, p.title AS name FROM customers c JOIN orders o ON o.customer_id = "
        "c.customer_id JOIN order_items i ON i.order_id = o.order_id JOIN products p "
        "ON p.product_id = i.product_id",
    )
    assert result.columns == ["name", "name_2"]
    assert list(result.to_dataframe().columns) == ["name", "name_2"]
