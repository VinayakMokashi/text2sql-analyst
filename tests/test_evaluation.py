import pytest

from text2sql.evaluation import normalize_value, recall, results_match, tables_in_sql


@pytest.mark.parametrize(
    "gold, pred",
    [
        ([("Rock",)], [("Rock", 826.65)]),  # extra columns are fine
        ([("a", 1), ("b", 2)], [(2, "b"), (1, "a")]),  # column and row order ignored
        ([(826.65,)], [(826.6500000001,)]),  # float noise
        ([("2010", 481.45)], [(2010, 481.45)]),  # numeric string == number
        ([], []),
    ],
)
def test_matching_results(gold, pred):
    assert results_match(gold, pred)


@pytest.mark.parametrize(
    "gold, pred",
    [
        ([("Rock",)], [("Rock",), ("Latin",)]),  # extra row
        ([("Rock",)], [("Metal",)]),  # wrong value
        ([(5.65,)], [(5.7,)]),  # different rounding
        # Same values per column but paired up differently across rows:
        ([("a", 1), ("b", 2)], [("a", 2), ("b", 1)]),
        ([("Jane", "Peacock")], [("Jane Peacock",)]),  # needs its own gold variant
    ],
)
def test_mismatching_results(gold, pred):
    assert not results_match(gold, pred)


def test_normalize_value():
    assert normalize_value(-0.0) == 0.0
    assert normalize_value(" USA ") == "USA"
    assert normalize_value(None) is None


def test_tables_in_sql_ignores_ctes():
    sql = (
        "WITH t AS (SELECT * FROM Invoice) "
        "SELECT c.Country FROM Customer c JOIN t ON t.CustomerId = c.CustomerId"
    )
    assert tables_in_sql(sql) == {"invoice", "customer"}


def test_recall():
    assert recall({"a", "b"}, ["A", "c"]) == 0.5
    assert recall(set(), []) == 1.0


def test_identical_results_with_many_identical_columns_match():
    assert results_match([(None,) * 6], [(None,) * 6])
    assert results_match([(0,) * 6, (0,) * 6], [(0,) * 6, (0,) * 6])


def test_rounding_matches_sqlite_half_away_from_zero():
    assert normalize_value(0.125) == 0.13
    assert results_match([(0.13,)], [(0.125,)])
