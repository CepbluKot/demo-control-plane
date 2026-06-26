"""Result objects returned by summary input ingestion workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UploadIngestionResult:
    job_id: str
    source_format: str
    filename: str
    segments_count: int
    rows_count: int
    queued: bool


@dataclass(frozen=True)
class StagedUploadResult:
    job_id: str
    source_format: str
    filename: str
    staged_size_bytes: int
    queued: bool


@dataclass(frozen=True)
class QueryIngestionResult:
    job_id: str
    source_format: str
    segments_count: int
    rows_count: int
    queued: bool
