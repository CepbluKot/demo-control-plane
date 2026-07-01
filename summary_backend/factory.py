"""Default component wiring for the summary backend.

Keep concrete technology choices here so API/worker entrypoints can be thin and
tests can swap adapters without touching the pipeline core.
"""

from __future__ import annotations

from .audit import AuditWriter
from .config import Settings, get_settings
from .llm_client import StructuredLLMClient
from .pipeline import PipelineService
from .ports import Chunker, InputSegmenter, SummaryLLM, SummaryStore, TaskQueue
from .store import ClickHouseStore
from .input_segments import RowBudgetInputSegmenter
from .text import CharBudgetChunker


def create_store(settings: Settings | None = None) -> SummaryStore:
    return ClickHouseStore(settings or get_settings())


def create_chunker(settings: Settings | None = None) -> Chunker:
    _ = settings or get_settings()
    return CharBudgetChunker()


def create_input_segmenter(settings: Settings | None = None) -> InputSegmenter:
    _ = settings or get_settings()
    return RowBudgetInputSegmenter()


def create_llm(store: SummaryStore, settings: Settings | None = None) -> SummaryLLM:
    settings = settings or get_settings()
    return StructuredLLMClient(
        store=store,
        settings=settings,
        audit=AuditWriter(settings),
        pool_kind="jobs",
    )


def create_pipeline_service(
    *,
    queue: TaskQueue | None = None,
    settings: Settings | None = None,
    store: SummaryStore | None = None,
    llm: SummaryLLM | None = None,
    chunker: Chunker | None = None,
    input_segmenter: InputSegmenter | None = None,
) -> PipelineService:
    settings = settings or get_settings()
    store = store or create_store(settings)
    return PipelineService(
        store=store,
        queue=queue,
        llm=llm or create_llm(store, settings),
        chunker=chunker or create_chunker(settings),
        input_segmenter=input_segmenter or create_input_segmenter(settings),
        settings=settings,
    )
