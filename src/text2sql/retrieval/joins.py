"""Add the "bridge" tables needed to join the selected tables together.

Example (Chinook): "revenue per genre" needs ``Genre`` and ``InvoiceLine``, but they
only connect through ``Track``. Vector search and LLM selection often miss such
bridge tables because the question never mentions them, so we recover them
deterministically from the foreign-key graph.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from text2sql.db.schema import TableSchema


def fk_graph(tables: Iterable[TableSchema]) -> dict[str, set[str]]:
    """Undirected adjacency map (lower-cased names) built from foreign keys.

    SQLite accepts a foreign key to a table that does not exist (for example after a
    lookup table was dropped). Such edges are skipped, so a join path can never route
    through a table the SQL model would not be given.
    """
    tables = list(tables)
    graph: dict[str, set[str]] = {t.name.lower(): set() for t in tables}
    for t in tables:
        for fk in t.foreign_keys:
            a, b = t.name.lower(), fk.ref_table.lower()
            if b in graph and a != b:  # a self-reference adds no bridge table
                graph[a].add(b)
                graph[b].add(a)
    return graph


def _path_to_tree(graph: dict[str, set[str]], start: str, tree: set[str]) -> list[str]:
    """BFS from ``start`` to the nearest node already in ``tree``; returns the path."""
    parents: dict[str, str | None] = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node in tree:
            path = []
            cur: str | None = node
            while cur is not None:
                path.append(cur)
                cur = parents[cur]
            return path
        for nxt in sorted(graph.get(node, ())):  # sorted -> deterministic results
            if nxt not in parents:
                parents[nxt] = node
                queue.append(nxt)
    return []  # not connected: leave it to the LLM (e.g. a cross join or subquery)


def add_join_tables(selected: list[str], all_tables: list[TableSchema]) -> list[str]:
    """Return ``selected`` plus any intermediate tables on the shortest FK paths.

    Greedy Steiner-tree approximation: grow a connected set one selected table at a
    time, each time attaching it by its shortest path. Plenty for real schemas.
    """
    if len(selected) < 2:
        return list(selected)
    graph = fk_graph(all_tables)
    canonical = {t.name.lower(): t.name for t in all_tables}

    tree = {selected[0].lower()}
    for name in selected[1:]:
        tree.update(_path_to_tree(graph, name.lower(), tree) or [name.lower()])

    extra = sorted(tree - {s.lower() for s in selected})
    return list(selected) + [canonical.get(e, e) for e in extra]
