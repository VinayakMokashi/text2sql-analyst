"""Public-demo helpers: self-setup of a sample database, and the daily question budget."""

import json
import shutil
from datetime import date

from text2sql.config import Settings
from text2sql.demo import DailyBudget, ensure_sample_ready, index_ready
from text2sql.indexing import load_descriptions, load_index_info


def test_budget_counts_down_and_resets_the_next_day():
    today = [date(2026, 9, 22)]
    budget = DailyBudget(2, today=lambda: today[0])
    assert budget.take() and budget.take()
    assert not budget.take()
    assert budget.remaining() == 0
    today[0] = date(2026, 9, 23)
    assert budget.remaining() == 2 and budget.take()


def test_no_limit_means_unlimited():
    budget = DailyBudget(None)
    assert all(budget.take() for _ in range(1000))
    assert budget.remaining() is None


def test_sample_database_is_indexed_from_bundled_descriptions(tmp_path, shop_db, embedder):
    # Stand the shop database in for a bundled sample (the name is what counts).
    db = tmp_path / "data" / "chinook.db"
    db.parent.mkdir()
    shutil.copy(shop_db, db)
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    bundled = {"customers": "People who buy things.", "orders": "Purchases."}
    (seeds / "chinook.json").write_text(json.dumps(bundled), encoding="utf-8")
    settings = Settings(db_path=db, index_dir=tmp_path / "index", llm_provider="fake")

    assert not index_ready(settings)
    assert ensure_sample_ready(settings, seeds, lambda: embedder) is True
    assert index_ready(settings)
    docs = load_descriptions(settings.db_index_dir)
    assert docs["customers"] == "People who buy things."  # bundled text, no LLM call
    assert docs["products"].startswith("Table products")  # template for the rest
    assert load_index_info(settings.db_index_dir)["db_path"] == str(db.resolve())
    assert ensure_sample_ready(settings, seeds, lambda: embedder) is False  # already done


def test_other_databases_are_never_set_up_automatically(tmp_path, shop_db, embedder):
    settings = Settings(db_path=shop_db, index_dir=tmp_path / "index", llm_provider="fake")
    assert ensure_sample_ready(settings, tmp_path, lambda: embedder) is False
    assert not index_ready(settings)
