"""Public-demo helpers: self-setup of a sample database, and the daily question budget."""

import json
import shutil
from datetime import date

from text2sql.config import Settings
from text2sql.demo import DailyBudget, ensure_sample_ready, index_ready, shared_budget
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
    # Even with a descriptions file for it, only a bundled sample is set up by itself.
    (tmp_path / "shop.json").write_text('{"customers": "People."}', encoding="utf-8")
    settings = Settings(db_path=shop_db, index_dir=tmp_path / "index", llm_provider="fake")
    assert ensure_sample_ready(settings, tmp_path, lambda: embedder) is False
    assert not index_ready(settings)


def _sample(tmp_path, shop_db, folder="data"):
    """The shop database standing in for the Chinook sample, plus bundled descriptions."""
    db = tmp_path / folder / "chinook.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(shop_db, db)
    seeds = tmp_path / "seeds"
    seeds.mkdir(exist_ok=True)
    bundled = {"customers": "People who buy things.", "orders": "Purchases."}
    (seeds / "chinook.json").write_text(json.dumps(bundled), encoding="utf-8")
    return Settings(db_path=db, index_dir=tmp_path / "index", llm_provider="fake"), seeds


def test_descriptions_already_there_are_kept_and_completed(tmp_path, shop_db, embedder):
    settings, seeds = _sample(tmp_path, shop_db)
    # Say, a hand edit, or an `index` run that stopped after one table.
    settings.db_index_dir.mkdir(parents=True)
    docs_file = settings.db_index_dir / "table_docs.json"
    docs_file.write_text('{"customers": "Edited by hand."}', encoding="utf-8")

    ensure_sample_ready(settings, seeds, lambda: embedder)
    docs = load_descriptions(settings.db_index_dir)
    assert docs["customers"] == "Edited by hand."
    assert docs["orders"] == "Purchases."  # filled in from the bundled file


def test_unreadable_descriptions_file_is_replaced(tmp_path, shop_db, embedder):
    settings, seeds = _sample(tmp_path, shop_db)
    settings.db_index_dir.mkdir(parents=True)
    (settings.db_index_dir / "table_docs.json").write_text('{"customers": "Peop', encoding="utf-8")

    assert ensure_sample_ready(settings, seeds, lambda: embedder) is True
    assert load_descriptions(settings.db_index_dir)["customers"] == "People who buy things."


def test_index_of_another_copy_of_the_file_is_rebuilt(tmp_path, shop_db, embedder):
    first, seeds = _sample(tmp_path, shop_db, folder="old")
    ensure_sample_ready(first, seeds, lambda: embedder)
    moved, _ = _sample(tmp_path, shop_db, folder="new")  # same index folder, new path

    assert not index_ready(moved)
    assert ensure_sample_ready(moved, seeds, lambda: embedder) is True
    assert load_index_info(moved.db_index_dir)["db_path"] == str(moved.db_path.resolve())
    assert index_ready(moved)


def test_broken_sample_file_is_downloaded_again(tmp_path, shop_db, embedder, monkeypatch):
    settings, seeds = _sample(tmp_path, shop_db)
    settings.db_path.write_bytes(b"<!doctype html><title>502 Bad Gateway</title>")
    calls = []

    def fake_download(name, out, force=False, log=print):
        calls.append((name, force))
        shutil.copy(shop_db, out)
        return out

    monkeypatch.setattr("text2sql.demo.download", fake_download)
    assert ensure_sample_ready(settings, seeds, lambda: embedder) is True
    assert calls == [("chinook", True)]
    assert index_ready(settings)


def test_a_refund_gives_one_question_back():
    budget = DailyBudget(1, today=lambda: date(2026, 9, 22))
    assert budget.take() and not budget.take()
    budget.refund()  # say, the provider's quota stopped that question
    assert budget.remaining() == 1 and budget.take()
    budget.refund()
    budget.refund()  # never more than was used
    assert budget.remaining() == 1


def test_every_visitor_shares_one_budget_outside_any_cache():
    assert shared_budget(7) is shared_budget(7)
    assert shared_budget(7) is not shared_budget(8)
