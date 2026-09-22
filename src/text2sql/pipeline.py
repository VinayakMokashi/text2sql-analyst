"""The online question-answering pipeline, end to end.

    question
      -> vector search (top-N tables)          retrieval.TableRetriever
      -> LLM table selection (top-K)           retrieval.TableSelector
      -> add FK bridge tables                  retrieval.add_join_tables
      -> LLM writes SQL                        generation.SQLGenerator
      -> guardrails + read-only execution      execution.execute_query
           on error: LLM repairs the SQL, up to ``max_retries`` times
      -> LLM answer + analysis, chart choice   analysis.Analyst / suggest_chart

The CLI, the Streamlit app and the evaluation script all call ``Pipeline.ask``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

from text2sql.analysis import Analysis, Analyst, ChartSpec, suggest_chart
from text2sql.config import Settings, get_settings
from text2sql.db.schema import TableSchema, read_schema, schema_by_name
from text2sql.execution import QueryError, QueryResult, execute_query
from text2sql.generation import GeneratedSQL, SQLGenerator
from text2sql.indexing import FastEmbedEmbedder, load_index_info, open_collection
from text2sql.indexing.embeddings import Embedder
from text2sql.llm import LLMError, create_llm
from text2sql.retrieval import (
    RetrievedTable,
    TableRetriever,
    TableSelection,
    TableSelector,
    add_join_tables,
)

log = logging.getLogger(__name__)

Status = Literal["ok", "unanswerable", "error"]


@dataclass
class Attempt:
    """One SQL attempt: the query and, if it failed, why."""

    sql: str | None
    error: str | None = None
    latency_s: float = 0.0


@dataclass
class PipelineResult:
    question: str
    status: Status = "ok"
    message: str = ""  # explanation when status is not "ok"
    candidates: list[RetrievedTable] = field(default_factory=list)
    selection: TableSelection | None = None
    tables_used: list[str] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    result: QueryResult | None = None
    analysis: Analysis | None = None
    chart: ChartSpec | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def sql(self) -> str | None:
        """The last SQL that was tried (the one that ran, if the question succeeded)."""
        return self.attempts[-1].sql if self.attempts else None

    @property
    def total_s(self) -> float:
        return sum(self.timings.values())


def load_embedder(settings: Settings) -> FastEmbedEmbedder:
    return FastEmbedEmbedder(settings.embedding_model, cache_dir=settings.model_cache_dir)


def check_index(settings: Settings) -> None:
    """Refuse to use an index built for another database file or embedding model.

    Index folders are named after the database file, so ``a/chinook.db`` and
    ``b/chinook.db`` would share one; and vectors from a different embedding model
    are not comparable with the query's. Both would give silently wrong retrieval.
    """
    info = load_index_info(settings.db_index_dir)
    if not info:
        return  # an index built before this check existed; nothing to compare
    built_for = info.get("db_path")
    if built_for and built_for != str(settings.db_path.resolve()):
        raise ValueError(
            f"The index in {settings.db_index_dir} was built for {built_for}, not "
            f"{settings.db_path.resolve()}. Set T2S_INDEX_DIR to a different folder, "
            "or run `python -m text2sql index` to rebuild it for this database."
        )
    model = info.get("embedding_model")
    if model and model != settings.embedding_model:
        raise ValueError(
            f"The index was built with the embedding model {model}, but "
            f"T2S_EMBEDDING_MODEL is {settings.embedding_model}. Run "
            "`python -m text2sql index` to rebuild it."
        )


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        schemas: list[TableSchema],
        retriever: TableRetriever,
        selector: TableSelector,
        generator: SQLGenerator,
        analyst: Analyst,
    ) -> None:
        self.settings = settings
        self.schemas = schemas
        self._by_name = schema_by_name(schemas)
        self.retriever = retriever
        self.selector = selector
        self.generator = generator
        self.analyst = analyst

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        sql_model: str | None = None,
        helper_model: str | None = None,
        embedder: Embedder | None = None,
    ) -> Pipeline:
        """Wire up real components. Model names can be overridden (used by the eval)."""
        s = settings or get_settings()
        check_index(s)
        helper = create_llm(s, "helper", helper_model)
        sql_llm = create_llm(s, "sql", sql_model)
        embedder = embedder or load_embedder(s)
        return cls(
            settings=s,
            schemas=read_schema(s.db_path, s.sample_rows),
            retriever=TableRetriever(open_collection(s.db_index_dir), embedder),
            selector=TableSelector(helper),
            generator=SQLGenerator(sql_llm, s.sql_dialect, s.sample_rows),
            analyst=Analyst(helper, s.analysis_rows),
        )

    # ------------------------------------------------------------------ public API
    def ask(self, question: str, analyze: bool = True) -> PipelineResult:
        """Answer one question. Never raises for model/SQL problems: failures come
        back as ``status="error"`` with a readable ``message``."""
        out = PipelineResult(question=question.strip())
        if not out.question:
            out.status, out.message = "error", "Please enter a question."
            return out
        try:
            self._run(out, analyze)
        except LLMError as exc:
            out.status, out.message = "error", f"The language model request failed: {exc}"
        except Exception as exc:  # noqa: BLE001 - an app should report, not crash
            # Anything unexpected is a bug; the log keeps the traceback for fixing it.
            log.exception("Unexpected error while answering %r", out.question)
            out.status, out.message = "error", f"Unexpected error ({type(exc).__name__}): {exc}"
        return out

    # ------------------------------------------------------------------- internals
    @contextmanager
    def _timed(self, out: PipelineResult, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            out.timings[stage] = out.timings.get(stage, 0.0) + time.perf_counter() - start

    def _run(self, out: PipelineResult, analyze: bool) -> None:
        s = self.settings

        # 1-2. Schema linking: recall-oriented vector search, then LLM precision.
        with self._timed(out, "retrieval"):
            out.candidates = self.retriever.search(out.question, s.top_n_tables)
        if not any(c.name.lower() in self._by_name for c in out.candidates):
            out.status = "error"
            out.message = (
                "The table index does not match this database (none of the indexed tables "
                "exist in it). Run `python -m text2sql index` for this database."
            )
            return
        with self._timed(out, "table_selection"):
            out.selection = self.selector.select(
                out.question, out.candidates, self._by_name, s.top_k_tables
            )
        if not out.selection.answerable:
            out.status = "unanswerable"
            out.message = out.selection.reason or "This question can't be answered from the data."
            return

        out.tables_used = add_join_tables(out.selection.tables, self.schemas)
        tables = [self._by_name[t.lower()] for t in out.tables_used]

        # 3-4. Generate SQL, run it, and let the model repair its own errors.
        with self._timed(out, "sql_generation"):
            generated = self.generator.generate(out.question, tables)
        for attempt_no in range(s.max_retries + 1):
            if self._declined(out, generated):
                return
            attempt = Attempt(sql=generated.sql, latency_s=generated.latency_s)
            out.attempts.append(attempt)
            try:
                with self._timed(out, "execution"):
                    out.result = execute_query(
                        s.db_path, generated.sql or "", s.max_rows, s.query_timeout_s, s.sql_dialect
                    )
                break
            except QueryError as exc:
                attempt.error = str(exc)
                if attempt_no == s.max_retries:
                    out.status = "error"
                    out.message = (
                        f"Couldn't produce a working query after {len(out.attempts)} "
                        f"attempt(s). Last error: {exc}"
                    )
                    return
                with self._timed(out, "sql_repair"):
                    generated = self.generator.repair(
                        out.question, tables, generated.sql or "", str(exc)
                    )

        # 5. Explain the result and pick a chart.
        assert out.result is not None
        out.chart = suggest_chart(out.result.to_dataframe())
        if analyze:
            with self._timed(out, "analysis"):
                try:
                    out.analysis = self.analyst.analyze(out.question, out.result)
                except LLMError as exc:
                    # The data is still worth showing even if the explanation failed.
                    out.analysis = Analysis(
                        answer=f"Here are the results ({out.result.row_count} rows).",
                        caveats=[f"The written analysis is unavailable: {exc}"],
                    )

    @staticmethod
    def _declined(out: PipelineResult, generated: GeneratedSQL) -> bool:
        if generated.cannot_answer is None:
            return False
        out.status = "unanswerable"
        out.message = generated.cannot_answer
        return True
