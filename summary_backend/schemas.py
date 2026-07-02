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
    CANCELLED = "CANCELLED"
    WAITING_RETRY = "WAITING_RETRY"
    WAITING_PROVIDER = "WAITING_PROVIDER"
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


class JsonObjectResult(BaseModel):
    model_config = ConfigDict(extra="allow")


class SummaryServiceRuntimeSettings(BaseModel):
    api_host: str
    api_port: int
    cors_origins: list[str]
    websocket_poll_interval_seconds: float
    log_level: str
    broker_url: str
    worker_processes: int
    worker_threads: int


class SummaryServiceStorageSettings(BaseModel):
    log_dir: str
    audit_dir: str
    upload_staging_dir: str


class SummaryServiceDatabaseSettings(BaseModel):
    host: str
    port: int
    username: str
    database: str
    secure: bool
    password_configured: bool


class SummaryServiceLlmSettings(BaseModel):
    api_base: str
    model: str
    available_models: list[str]
    available_model_options: list[dict[str, Any]] = Field(default_factory=list)
    timeout_seconds: float
    max_retries: int
    retry_backoff_seconds: float
    max_concurrency: int
    assistant_max_concurrency: int
    pool_acquire_timeout_seconds: float
    pool_poll_interval_seconds: float
    api_key_configured: bool
    dry_run: bool


class SummaryServicePipelineSettings(BaseModel):
    chunk_target_estimated_tokens: int
    reduce_target_estimated_tokens: int
    reduce_group_size: int
    max_enqueue_nodes_per_advance: int


class SummaryServiceSettingsResponse(BaseModel):
    service_name: str
    read_only: bool
    runtime: SummaryServiceRuntimeSettings
    storage: SummaryServiceStorageSettings
    clickhouse: SummaryServiceDatabaseSettings
    source_clickhouse: SummaryServiceDatabaseSettings
    llm: SummaryServiceLlmSettings
    pipeline: SummaryServicePipelineSettings


class SummaryLlmConnectivityCheckResult(BaseModel):
    value: str
    label: str
    profile_id: str = ""
    profile_label: str = ""
    model: str = ""
    api_base: str = ""
    ok: bool
    status: str
    detail: str = ""
    error_class: str = ""
    latency_ms: int | None = None


class SummaryLlmConnectivityCheckResponse(BaseModel):
    checked_at: datetime
    dry_run: bool
    total: int
    ok_count: int
    failed_count: int
    results: list[SummaryLlmConnectivityCheckResult]


class SummaryPromptStageDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system: str = Field(min_length=1)
    user: str = Field(min_length=1)


class SummaryPromptOverridesDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    map: SummaryPromptStageDraft
    reduce: SummaryPromptStageDraft
    final: SummaryPromptStageDraft


class GenerateSummaryPromptDraftRequest(BaseModel):
    request: str = Field(min_length=1, max_length=4000)
    llm_model: str | None = None
    use_custom_map_output_json: bool | None = None
    map_output_json_schema: dict[str, Any] | None = None
    use_custom_reduce_output_json: bool | None = None
    reduce_output_json_schema: dict[str, Any] | None = None
    use_custom_intermediate_output_json: bool | None = None
    intermediate_output_json_schema: dict[str, Any] | None = None
    use_custom_output_json: bool | None = None
    output_json_schema: dict[str, Any] | None = None


class GenerateSummaryPromptDraftResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_name: str = ""
    report_instruction: str = ""
    use_custom_map_output_json: bool = False
    map_output_json_schema: dict[str, Any] | None = None
    use_custom_reduce_output_json: bool = False
    reduce_output_json_schema: dict[str, Any] | None = None
    use_custom_intermediate_output_json: bool = False
    intermediate_output_json_schema: dict[str, Any] | None = None
    use_custom_output_json: bool = False
    output_json_schema: dict[str, Any] | None = None
    prompt_overrides: SummaryPromptOverridesDraft


class PromptDraftJobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    WAITING_PROVIDER = "WAITING_PROVIDER"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    DONE = "DONE"


class PromptDraftJobRecord(BaseModel):
    job_id: str
    status: PromptDraftJobStatus
    stage: str = ""
    llm_model: str | None = None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None = None
    error_detail: str = ""
    result: GenerateSummaryPromptDraftResponse | None = None


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


class RerunSummaryNodeResponse(BaseModel):
    job_id: str
    node_id: str
    node_type: str
    status: NodeStatus
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


class DatasourceQueryPreviewRequest(BaseModel):
    datasource: str | None = None
    datasourceConfig: dict[str, Any] | None = None
    query: str = Field(min_length=1)
    step: str | None = None
    resultMode: str | None = None
    limit: int = Field(default=100, ge=1, le=500)


class DatasourceQueryPreviewMetadata(BaseModel):
    columns: list[str]
    records_count: int
    execution_time_ms: int


class DatasourceQueryPreviewResponse(BaseModel):
    success: bool = True
    datasource: str
    query: str
    metadata: DatasourceQueryPreviewMetadata
    data: list[dict[str, Any]] = Field(default_factory=list)
    raw_rows: list[dict[str, Any]] = Field(default_factory=list)


class JobCurrent(BaseModel):
    job_id: str
    job_status: JobStatus
    last_event_type: str
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
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


class SummaryLlmChatMessage(BaseModel):
    role: str
    content: str


class SummaryNodeLlmCallResponse(BaseModel):
    call_id: str = ""
    created_at: datetime | None = None
    job_id: str
    node_id: str
    provider: str = ""
    model: str = ""
    status: str = ""
    error_class: str = ""
    http_status: int = 0
    latency_ms: int = 0
    pool_wait_ms: int = 0
    provider_latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error_message: str = ""
    messages: list[SummaryLlmChatMessage] = Field(default_factory=list)
    assistant_message: str = ""
    request_json: dict[str, Any] | None = None
    response_json: dict[str, Any] | None = None


class PauseResumeResponse(BaseModel):
    job_id: str
    status: JobStatus


class MonitoringRunStatus(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class MonitoringRunTrigger(StrEnum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"


class MonitoringScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    cron: str = ""
    timezone: str = "UTC"
    max_active_runs: int = Field(default=1, ge=1)


class CreateMonitoringProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    service: str = Field(min_length=1, max_length=200)
    description: str = ""
    workflow_id: str = Field(min_length=1, max_length=200)
    workflow_inputs: dict[str, Any] = Field(default_factory=dict)
    schedule: MonitoringScheduleConfig | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MonitoringProfileRecord(BaseModel):
    profile_id: str
    name: str
    service: str
    description: str = ""
    workflow_id: str
    workflow_inputs: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_archived: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None
    schedule: MonitoringScheduleConfig | None = None
    next_run_at: datetime | None = None
    last_run_at: datetime | None = None


class CreateMonitoringProfileResponse(BaseModel):
    profile: MonitoringProfileRecord


class MonitoringScheduleTickItem(BaseModel):
    profile_id: str
    action: str
    run_id: str = ""
    detail: str = ""
    next_run_at: datetime | None = None


class MonitoringSchedulerTickResponse(BaseModel):
    checked_at: datetime
    launched: int
    skipped: int
    items: list[MonitoringScheduleTickItem]


class CreateMonitoringRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_inputs_override: dict[str, Any] = Field(default_factory=dict)


class MonitoringRunRecord(BaseModel):
    run_id: str
    profile_id: str
    status: MonitoringRunStatus
    trigger_type: MonitoringRunTrigger
    workflow_id: str
    workflow_run_id: str = ""
    task_id: str = ""
    requested_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    scheduled_for: datetime | None = None
    workflow_inputs: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_message: str = ""
    updated_at: datetime | None = None


class CreateMonitoringRunResponse(BaseModel):
    run: MonitoringRunRecord
