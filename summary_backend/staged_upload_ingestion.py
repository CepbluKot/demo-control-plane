"""Staged uploaded-file ingestion workflow."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, BinaryIO

from .config import Settings, get_settings
from .ids import new_job_id
from .ingestion_models import StagedUploadResult
from .ingestion_support import latest_event_payload, open_text_stream, safe_filename
from .input_models import InputSegment
from .input_parsers import InputParseError, ParserRegistry
from .input_segments import RowBudgetInputSegmenter
from .logging_setup import get_logger
from .ports import InputSegmenter, SummaryStore, TaskQueue
from .schemas import ArtifactType, JobStatus, Stage

logger = get_logger("ingestion.staged_upload")


class StagedUploadIngestionService:
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

    def create_staged_upload_job(
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
    ) -> StagedUploadResult:
        source_format = self.parser_registry.detect_format(
            filename=filename,
            content_type=content_type,
            requested_format=requested_format,
        )
        job_id = new_job_id()
        filename_on_disk = safe_filename(filename or f"upload.{source_format}")
        job_dir = self.settings.upload_staging_dir / job_id
        final_path = job_dir / filename_on_disk
        part_path = job_dir / f"{filename_on_disk}.part"
        source = {
            "kind": "upload",
            "filename": filename,
            "content_type": content_type,
            "format": source_format,
            "raw_line_column": raw_line_column or "",
        }
        payload = {
            "title": title or filename,
            "metadata": metadata or {},
            "source": source,
            "auto_start": auto_start,
            "staging": {"path": str(final_path), "part_path": str(part_path), "status": "receiving"},
        }
        self.store.insert_job_event(
            job_id=job_id,
            event_type="JOB_CREATED",
            job_status=JobStatus.INGESTING,
            actor="api",
            message=title or filename,
            payload=payload,
        )

        try:
            staged_size = self._stage_upload(file=file, job_dir=job_dir, part_path=part_path, final_path=final_path)
        except Exception as exc:
            self.store.insert_job_event(
                job_id=job_id,
                event_type="JOB_FAILED",
                job_status=JobStatus.FAILED,
                actor="api",
                message=str(exc),
                payload={**payload, "error": str(exc)},
            )
            raise

        staged_payload = {
            **payload,
            "staging": {
                "path": str(final_path),
                "size_bytes": staged_size,
                "status": "staged",
            },
        }
        self.store.insert_job_event(
            job_id=job_id,
            event_type="FILE_STAGED",
            job_status=JobStatus.INGESTING,
            actor="api",
            payload=staged_payload,
        )
        queued = False
        if self.queue is not None:
            self.queue.ingest_upload(job_id)
            queued = True

        logger.info(
            "upload_staged | job_id=%s filename=%s format=%s size_bytes=%s queued=%s",
            job_id,
            filename,
            source_format,
            staged_size,
            queued,
        )
        return StagedUploadResult(
            job_id=job_id,
            source_format=source_format,
            filename=filename,
            staged_size_bytes=staged_size,
            queued=queued,
        )

    def create_job_from_existing_upload(
        self,
        *,
        upload_id: str,
        title: str | None,
        metadata: dict[str, Any] | None,
        source_format: str | None = None,
        raw_line_column: str | None = None,
        auto_start: bool = True,
    ) -> StagedUploadResult:
        staged_payload = latest_event_payload(self.store, upload_id, "FILE_STAGED")
        if not staged_payload:
            raise InputParseError(f"uploaded file not found: {upload_id}")

        original_source = staged_payload.get("source") if isinstance(staged_payload.get("source"), dict) else {}
        original_staging = staged_payload.get("staging") if isinstance(staged_payload.get("staging"), dict) else {}
        path = Path(str(original_staging.get("path") or ""))
        if not path.exists():
            raise InputParseError(f"uploaded file is no longer available: {upload_id}")

        requested_format = source_format or str(original_source.get("format") or "auto")
        if requested_format == "auto":
            requested_format = "auto"
        filename = str(original_source.get("filename") or path.name)
        content_type = str(original_source.get("content_type") or "")
        resolved_format = self.parser_registry.detect_format(
            filename=filename,
            content_type=content_type,
            requested_format=requested_format,
        )
        self.parser_registry.get(resolved_format)
        resolved_raw_line_column = (
            raw_line_column if raw_line_column is not None else str(original_source.get("raw_line_column") or "")
        )
        staged_size = int(original_staging.get("size_bytes") or path.stat().st_size)
        job_id = new_job_id()
        source = {
            "kind": "upload",
            "filename": filename,
            "content_type": content_type,
            "format": resolved_format,
            "raw_line_column": resolved_raw_line_column,
        }
        payload = {
            "title": title or filename,
            "metadata": metadata or {},
            "source": source,
            "auto_start": auto_start,
            "reused_upload": {
                "upload_id": upload_id,
                "source_job_id": upload_id,
            },
            "staging": {
                "path": str(path),
                "size_bytes": staged_size,
                "status": "staged",
            },
        }
        self.store.insert_job_event(
            job_id=job_id,
            event_type="JOB_CREATED",
            job_status=JobStatus.INGESTING,
            actor="api",
            message=title or filename,
            payload=payload,
        )
        self.store.insert_job_event(
            job_id=job_id,
            event_type="FILE_STAGED",
            job_status=JobStatus.INGESTING,
            actor="api",
            message=f"reused upload {upload_id}",
            payload=payload,
        )
        queued = False
        if self.queue is not None:
            self.queue.ingest_upload(job_id)
            queued = True

        logger.info(
            "upload_reused | upload_id=%s job_id=%s filename=%s format=%s size_bytes=%s queued=%s",
            upload_id,
            job_id,
            filename,
            resolved_format,
            staged_size,
            queued,
        )
        return StagedUploadResult(
            job_id=job_id,
            source_format=resolved_format,
            filename=filename,
            staged_size_bytes=staged_size,
            queued=queued,
        )

    def ingest_staged_upload(self, job_id: str) -> None:
        current = self.store.get_job_current(job_id)
        if current is None:
            raise KeyError(job_id)
        if str(current["job_status"]) in {JobStatus.CANCELLED, JobStatus.DONE, JobStatus.FAILED}:
            return
        if str(current["job_status"]) == JobStatus.CANCEL_REQUESTED:
            self.store.insert_job_event(
                job_id=job_id,
                event_type="JOB_CANCELLED",
                job_status=JobStatus.CANCELLED,
                actor="ingest_upload",
            )
            return

        staged_payload = latest_event_payload(self.store, job_id, "FILE_STAGED")
        if not staged_payload:
            self._fail_job(job_id, "staged upload metadata not found", {})
            return

        source = staged_payload.get("source") or {}
        staging = staged_payload.get("staging") or {}
        path = Path(str(staging.get("path") or ""))
        if not path.exists():
            self._fail_job(job_id, f"staged upload file not found: {path}", staged_payload)
            return

        if self.store.latest_artifact(job_id=job_id, artifact_type=ArtifactType.INPUT) is not None:
            if staged_payload.get("auto_start", True) and self.queue is not None:
                self.queue.advance_job(job_id)
            return

        source_format = str(source.get("format") or "")
        raw_line_column = str(source.get("raw_line_column") or "") or None
        parser = self.parser_registry.get(source_format)
        self.store.insert_job_event(
            job_id=job_id,
            event_type="INPUT_INGEST_STARTED",
            job_status=JobStatus.INGESTING,
            actor="ingest_upload",
            payload=staged_payload,
        )

        existing_indexes = {
            int(row["segment_index"])
            for row in self.store.list_input_segments(job_id=job_id, include_content=False)
        }
        total_segments = 0
        total_rows = 0
        inserted_since_progress = 0
        batch: list[InputSegment] = []
        try:
            with path.open("rb") as raw_file:
                stream = open_text_stream(raw_file)
                records = parser.parse_text_stream(stream, raw_line_column=raw_line_column)
                segments = self.segmenter.build_segments(
                    records,
                    source_type="upload",
                    source_format=source_format,
                    target_estimated_tokens=self.settings.chunk_target_estimated_tokens,
                )
                for segment in segments:
                    total_segments += 1
                    total_rows += segment.rows_count
                    if segment.segment_index in existing_indexes:
                        continue
                    batch.append(segment)
                    if len(batch) >= self.insert_batch_size:
                        inserted_since_progress = self._flush_progress_batch(
                            job_id=job_id,
                            batch=batch,
                            total_segments=total_segments,
                            total_rows=total_rows,
                            inserted_since_progress=inserted_since_progress,
                            existing_indexes=existing_indexes,
                        )
                        batch = []
                if batch:
                    inserted_since_progress = self._flush_progress_batch(
                        job_id=job_id,
                        batch=batch,
                        total_segments=total_segments,
                        total_rows=total_rows,
                        inserted_since_progress=inserted_since_progress,
                        existing_indexes=existing_indexes,
                    )
        except Exception as exc:
            self._fail_job(job_id, str(exc), staged_payload)
            raise

        if total_segments == 0:
            self._fail_job(job_id, "uploaded file does not contain parseable log rows", staged_payload)
            return

        manifest = {
            "source": source,
            "staging": staging,
            "segments_count": total_segments,
            "rows_count": total_rows,
        }
        self.store.insert_artifact(
            job_id=job_id,
            node_id="",
            artifact_type=ArtifactType.INPUT,
            stage=Stage.INPUT,
            level=0,
            content=(
                f"staged upload manifest: {source.get('filename', '')} "
                f"({source_format}), rows={total_rows}, segments={total_segments}"
            ),
            metadata={**staged_payload, "manifest": manifest},
        )
        self.store.insert_job_event(
            job_id=job_id,
            event_type="INPUT_READY",
            job_status=JobStatus.INPUT_READY,
            actor="ingest_upload",
            payload=manifest,
        )
        logger.info(
            "staged_upload_ingested | job_id=%s rows=%s segments=%s existing_segments=%s",
            job_id,
            total_rows,
            total_segments,
            len(existing_indexes),
        )
        if staged_payload.get("auto_start", True) and self.queue is not None:
            self.queue.advance_job(job_id)

    @staticmethod
    def _stage_upload(*, file: BinaryIO, job_dir: Path, part_path: Path, final_path: Path) -> int:
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            file.seek(0)
        except (OSError, AttributeError):
            pass
        with part_path.open("wb") as target:
            shutil.copyfileobj(file, target, length=1024 * 1024)
        part_path.replace(final_path)
        return final_path.stat().st_size

    def _flush_progress_batch(
        self,
        *,
        job_id: str,
        batch: list[InputSegment],
        total_segments: int,
        total_rows: int,
        inserted_since_progress: int,
        existing_indexes: set[int],
    ) -> int:
        self.store.insert_input_segments(job_id=job_id, segments=batch)
        inserted_since_progress += len(batch)
        existing_indexes.update(segment.segment_index for segment in batch)
        self._write_progress(job_id, total_segments, total_rows, inserted_since_progress)
        return inserted_since_progress

    def _write_progress(self, job_id: str, segments_seen: int, rows_seen: int, inserted_segments: int) -> None:
        self.store.insert_job_event(
            job_id=job_id,
            event_type="INPUT_INGEST_PROGRESS",
            job_status=JobStatus.INGESTING,
            actor="ingest_upload",
            payload={
                "segments_seen": segments_seen,
                "rows_seen": rows_seen,
                "inserted_segments": inserted_segments,
            },
        )

    def _fail_job(self, job_id: str, message: str, payload: dict[str, Any]) -> None:
        self.store.insert_job_event(
            job_id=job_id,
            event_type="JOB_FAILED",
            job_status=JobStatus.FAILED,
            actor="ingest_upload",
            message=message,
            payload={**payload, "error": message},
        )
