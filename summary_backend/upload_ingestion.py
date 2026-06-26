"""Synchronous uploaded-file ingestion workflow."""

from __future__ import annotations

from typing import Any, BinaryIO

from .config import Settings, get_settings
from .ids import new_job_id
from .ingestion_models import UploadIngestionResult
from .ingestion_support import insert_segments, open_text_stream, with_first
from .input_parsers import InputParseError, ParserRegistry
from .input_segments import RowBudgetInputSegmenter
from .logging_setup import get_logger
from .ports import InputSegmenter, SummaryStore, TaskQueue
from .schemas import ArtifactType, JobStatus, Stage

logger = get_logger("ingestion.upload")


class UploadedFileIngestionService:
    def __init__(
        self,
        *,
        store: SummaryStore,
        queue: TaskQueue | None,
        settings: Settings | None = None,
        parser_registry: ParserRegistry | None = None,
        segmenter: InputSegmenter | None = None,
        insert_batch_size: int = 500,
    ) -> None:
        self.store = store
        self.queue = queue
        self.settings = settings or get_settings()
        self.parser_registry = parser_registry or ParserRegistry.default()
        self.segmenter = segmenter or RowBudgetInputSegmenter()
        self.insert_batch_size = insert_batch_size

    def create_job_from_upload(
        self,
        *,
        file: BinaryIO,
        filename: str,
        content_type: str,
        title: str | None,
        metadata: dict[str, Any] | None,
        requested_format: str = "auto",
        raw_line_column: str | None = None,
        auto_start: bool = True,
    ) -> UploadIngestionResult:
        source_format = self.parser_registry.detect_format(
            filename=filename,
            content_type=content_type,
            requested_format=requested_format,
        )
        parser = self.parser_registry.get(source_format)
        stream = open_text_stream(file)
        try:
            records = parser.parse_text_stream(stream, raw_line_column=raw_line_column)
            segments = self.segmenter.build_segments(
                records,
                source_type="upload",
                source_format=source_format,
                target_estimated_tokens=self.settings.chunk_target_estimated_tokens,
            )
            first_segment = next(segments, None)
        except InputParseError:
            raise
        except Exception as exc:
            raise InputParseError(f"failed to parse uploaded file: {exc}") from exc

        if first_segment is None:
            raise InputParseError("uploaded file does not contain parseable log rows")

        job_id = new_job_id()
        payload = {
            "title": title or filename,
            "metadata": metadata or {},
            "source": {
                "kind": "upload",
                "filename": filename,
                "content_type": content_type,
                "format": source_format,
                "raw_line_column": raw_line_column or "",
            },
        }
        self.store.insert_job_event(
            job_id=job_id,
            event_type="JOB_CREATED",
            job_status=JobStatus.CREATED,
            actor="api",
            message=title or filename,
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
                content=(
                    f"uploaded file manifest: {filename} ({source_format}), "
                    f"rows={rows_count}, segments={segments_count}"
                ),
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
            "upload_ingested | job_id=%s filename=%s format=%s rows=%s segments=%s queued=%s",
            job_id,
            filename,
            source_format,
            rows_count,
            segments_count,
            queued,
        )
        return UploadIngestionResult(
            job_id=job_id,
            source_format=source_format,
            filename=filename,
            segments_count=segments_count,
            rows_count=rows_count,
            queued=queued,
        )
