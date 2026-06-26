"""Pydantic schemas for the summary backend."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JobStatus(StrEnum):
    CREATED = "CREATED"
    INGESTING = "INGESTING"
    INPUT_READY = "INPUT_READY"
    RUNNING = "RUNNING"
    PAUSE_REQUESTED = "PAUSE_REQUESTED"
    PAUSED = "PAUSED"
    RESUMED = "RESUMED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    WAITING_RETRY = "WAITING_RETRY"
    WAITING_PROVIDER = "WAITING_PROVIDER"
    FAILED = "FAILED"
    DONE = "DONE"


class NodeStatus(StrEnum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    WAITING_RETRY = "WAITING_RETRY"
    DONE = "DONE"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    FAILED_FINAL = "FAILED_FINAL"
    SKIPPED_ALREADY_DONE = "SKIPPED_ALREADY_DONE"


class NodeType(StrEnum):
    MAP = "MAP"
    REDUCE = "REDUCE"
    FINAL = "FINAL"


class ArtifactType(StrEnum):
    INPUT = "INPUT"
    CHUNK = "CHUNK"
    MAP_SUMMARY = "MAP_SUMMARY"
    REDUCE_SUMMARY = "REDUCE_SUMMARY"
    FINAL_SUMMARY = "FINAL_SUMMARY"


class Stage(StrEnum):
    INPUT = "INPUT"
    CHUNK = "CHUNK"
    MAP = "MAP"
    REDUCE = "REDUCE"
    FINAL = "FINAL"


class SummaryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    summary: str
    key_points: list[str]
    warnings: list[str]
    source_count: int


class CreateSummaryJobRequest(BaseModel):
    input_text: str = Field(min_length=1)
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    auto_start: bool = True


class CreateSummaryJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    queued: bool


class RerunSummaryJobResponse(BaseModel):
    source_job_id: str
    job_id: str
    status: JobStatus
    queued: bool


class CreateSummaryJobUploadResponse(BaseModel):
    job_id: str
    status: JobStatus
    queued: bool
    filename: str
    source_format: str
    segments_count: int
    rows_count: int


class UploadedFileRecord(BaseModel):
    upload_id: str
    source_job_id: str
    filename: str
    source_format: str
    content_type: str = ""
    raw_line_column: str = ""
    size_bytes: int = 0
    available: bool = True
    job_status: str = ""
    staged_at: datetime | None = None


class CreateSummaryJobFromUploadRequest(BaseModel):
    upload_id: str = Field(min_length=1)
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    auto_start: bool = True
    source_format: str | None = None
    raw_line_column: str | None = None


class CreateSummaryJobQueryRequest(BaseModel):
    query: str = Field(min_length=1)
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    auto_start: bool = True
    raw_line_column: str | None = None


class CreateSummaryJobQueryResponse(BaseModel):
    job_id: str
    status: JobStatus
    queued: bool
    source_format: str
    segments_count: int
    rows_count: int


class JobCurrent(BaseModel):
    job_id: str
    job_status: JobStatus
    last_event_type: str
    updated_at: datetime | None = None
    events_count: int = 0


class NodeCurrent(BaseModel):
    job_id: str
    node_id: str
    node_type: str
    level: int
    node_index: int
    node_status: str
    last_event_type: str
    updated_at: datetime | None = None
    events_count: int = 0


class ArtifactRecord(BaseModel):
    artifact_id: str
    job_id: str
    node_id: str
    artifact_type: str
    stage: str
    level: int
    content_hash: str
    content: str | None = None
    metadata: str
    created_at: datetime | None = None


class InputSegmentRecord(BaseModel):
    job_id: str
    segment_index: int
    source_type: str
    source_format: str
    content_hash: str
    content: str | None = None
    rows_count: int
    chars: int
    metadata: str
    created_at: datetime | None = None


class JobStatusResponse(BaseModel):
    job: JobCurrent
    node_counts: dict[str, int]
    artifact_counts: dict[str, int]


class EventRecord(BaseModel):
    event_id: str
    job_id: str
    event_time: datetime | None = None
    event_type: str
    status: str
    actor: str
    message: str
    payload: str


class PauseResumeResponse(BaseModel):
    job_id: str
    status: JobStatus
