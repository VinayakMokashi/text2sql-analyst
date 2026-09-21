"""Shared fixtures: a tiny shop database that every test can use without downloads."""

from __future__ import annotations

import math
import re
import sqlite3
import zlib
from pathlib import Path

import pytest

SHOP_DDL = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT,
    price REAL
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date TEXT
);
CREATE TABLE order_items (
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity INTEGER
);
"""

SHOP_ROWS = """
INSERT INTO customers VALUES (1, 'Ana', 'Brazil'), (2, 'Ben', 'USA'), (3, 'Chloe', 'France');
INSERT INTO products VALUES (1, 'Guitar', 'Music', 300.0), (2, 'Drum', 'Music', 150.0),
                            (3, 'Novel', 'Books', 12.5);
INSERT INTO orders VALUES (1, 1, '2024-01-05'), (2, 2, '2024-02-10'), (3, 1, '2025-03-01');
INSERT INTO order_items VALUES (1, 1, 1), (1, 3, 2), (2, 2, 1), (3, 3, 4);
"""


class HashEmbedder:
    """Bag-of-words hashing embedder: deterministic, instant, no model download.

    Good enough for tests because questions that share words with a table's
    document land closest to it.
    """

    dim = 256

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in re.findall(r"[a-z]+", text.lower()):
            word = word.rstrip("s")  # crude singularisation: customers ~ customer
            vec[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture()
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture()
def shop_db(tmp_path: Path) -> Path:
    path = tmp_path / "shop.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(SHOP_DDL + SHOP_ROWS)
    return path
