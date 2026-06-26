"""ClickHouse query ingestion workflow."""

from __future__ import annotations

from typing import Any

from .config import Settings, get_settings
from .ids import new_job_id
from .ingestion_models import QueryIngestionResult
from .ingestion_support import insert_segments, with_first
from .input_segments import RowBudgetInputSegmenter
from .logging_setup import get_logger
from .ports import InputSegmenter, SummaryStore, TaskQueue
from .query_sources import QueryLogRecordSource, QuerySourceError
from .schemas import ArtifactType, JobStatus, Stage

logger = get_logger("ingestion.query")


class ClickHouseQueryIngestionService:
    def __init__(
        self,
        *,
        store: SummaryStore,
        queue: TaskQueue | None,
        query_source: QueryLogRecordSource,
        settings: Settings | None = None,
        segmenter: InputSegmenter | None = None,
        insert_batch_size: int = 500,
    ) -> None:
        self.store = store
        self.queue = queue
        self.query_source = query_source
        self.settings = settings or get_settings()
        self.segmenter = segmenter or RowBudgetInputSegmenter()
        self.insert_batch_size = insert_batch_size

    def create_job_from_query(
        self,
        *,
        query: str,
        title: str | None,
        metadata: dict[str, Any] | None,
        raw_line_column: str | None = None,
        auto_start: bool = True,
    ) -> QueryIngestionResult:
        records = self.query_source.iter_log_records(query, raw_line_column=raw_line_column)
        segments = self.segmenter.build_segments(
            records,
            source_type="clickhouse_query",
            source_format="clickhouse",
            target_estimated_tokens=self.settings.chunk_target_estimated_tokens,
        )
        try:
            first_segment = next(segments, None)
        except QuerySourceError:
            raise
        except Exception as exc:
            raise QuerySourceError(f"failed to read query result: {exc}") from exc
        if first_segment is None:
            raise QuerySourceError("query did not return parseable log rows")

        job_id = new_job_id()
        payload = {
            "title": title or "ClickHouse query",
            "metadata": metadata or {},
            "source": {
                "kind": "clickhouse_query",
                "format": "clickhouse",
                "raw_line_column": raw_line_column or "",
                "query": query,
            },
        }
        self.store.insert_job_event(
            job_id=job_id,
            event_type="JOB_CREATED",
            job_status=JobStatus.CREATED,
            actor="api",
            message=title or "ClickHouse query",
            payload=payload,
        )

        try:
            segments_count, rows_count = insert_segments(
                store=self.store,
                job_id=job_id,
                segments=with_first(first_segment, segments),
                batch_size=self.insert_batch_size,
            )
            manifest = {
                "source": payload["source"],
                "segments_count": segments_count,
                "rows_count": rows_count,
            }
            self.store.insert_artifact(
                job_id=job_id,
                node_id="",
                artifact_type=ArtifactType.INPUT,
                stage=Stage.INPUT,
                level=0,
                content=f"clickhouse query manifest: rows={rows_count}, segments={segments_count}",
                metadata={**payload, "manifest": manifest},
            )
            self.store.insert_job_event(
                job_id=job_id,
                event_type="INPUT_INGESTED",
                job_status=JobStatus.CREATED,
                actor="api",
                payload=manifest,
            )
        except Exception as exc:
            self.store.insert_job_event(
                job_id=job_id,
                event_type="JOB_FAILED",
                job_status=JobStatus.FAILED,
                actor="api",
                message=str(exc),
                payload={"source": payload["source"], "error": str(exc)},
            )
            raise

        queued = False
        if auto_start and self.queue is not None:
            self.queue.advance_job(job_id)
            queued = True

        logger.info(
            "query_ingested | job_id=%s rows=%s segments=%s queued=%s",
            job_id,
            rows_count,
            segments_count,
            queued,
        )
        return QueryIngestionResult(
            job_id=job_id,
            source_format="clickhouse",
            segments_count=segments_count,
            rows_count=rows_count,
            queued=queued,
        )
