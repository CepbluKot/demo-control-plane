"""FastAPI control plane for summary jobs."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

from .audit import AuditWriter
from .config import get_settings
from .factory import create_pipeline_service
from .ingestion import ClickHouseQueryIngestionService, StagedUploadIngestionService
from .input_parsers import InputParseError
from .llm_client import StructuredLLMClient
from .logging_setup import configure_logging, get_logger
from .pipeline import FINAL_SYSTEM, FINAL_USER_TEMPLATE, MAP_SYSTEM, MAP_USER_TEMPLATE, REDUCE_SYSTEM, REDUCE_USER_TEMPLATE
from .query_sources import ClickHouseQueryLogRecordSource, QuerySourceError
from .schemas import (
    ArtifactRecord,
    CreateSummaryJobFromUploadRequest,
    CreateSummaryJobQueryRequest,
    CreateSummaryJobQueryResponse,
    CreateSummaryJobRequest,
    CreateSummaryJobResponse,
    CreateSummaryJobUploadResponse,
    EventRecord,
    GenerateSummaryPromptDraftRequest,
    GenerateSummaryPromptDraftResponse,
    InputSegmentRecord,
    JobCurrent,
    JobStatus,
    JobStatusResponse,
    NodeCurrent,
    PauseResumeResponse,
    RerunSummaryJobResponse,
    RerunSummaryNodeResponse,
    SummaryServiceDatabaseSettings,
    SummaryServiceLlmSettings,
    SummaryServicePipelineSettings,
    SummaryServiceRuntimeSettings,
    SummaryServiceSettingsResponse,
    SummaryServiceStorageSettings,
    SummaryPromptOverridesDraft,
    SummaryPromptStageDraft,
    UploadedFileRecord,
)
from .snapshots import build_job_snapshot
from .store import ClickHouseStore
from .tasks import DramatiqTaskQueue

settings = get_settings()
configure_logging(settings)
logger = get_logger("api")

app = FastAPI(title="Summary Job Control Plane", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)


def _service():
    return create_pipeline_service(queue=DramatiqTaskQueue(), settings=settings)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "ok": True,
        "dry_run": settings.dry_run,
        "clickhouse_database": settings.clickhouse_database,
    }


@app.get("/settings", response_model=SummaryServiceSettingsResponse)
def service_settings() -> SummaryServiceSettingsResponse:
    return build_public_settings(settings)


@app.post("/prompt-drafts", response_model=GenerateSummaryPromptDraftResponse)
def generate_summary_prompt_draft(request: GenerateSummaryPromptDraftRequest) -> GenerateSummaryPromptDraftResponse:
    if settings.dry_run:
        return _fallback_prompt_draft(request.request, request.output_json_schema)

    store = ClickHouseStore(settings)
    llm = StructuredLLMClient(store=store, settings=settings, audit=AuditWriter(settings))
    system = (
        "You design prompts for a three-stage summarization pipeline. "
        "Return only JSON matching the requested schema. "
        "Each stage must include a system prompt and a user prompt template. "
        "The MAP and REDUCE stages must keep returning the internal SummaryResult JSON shape: "
        '{"ok": true, "summary": "string", "key_points": ["string"], "warnings": ["string"], "source_count": 1}.'
    )
    output_schema_text = json.dumps(request.output_json_schema or {}, ensure_ascii=False, indent=2)
    user = (
        "Create prompt overrides for a summary-generation job.\n\n"
        f"User request:\n{request.request.strip()}\n\n"
        f"Desired final output JSON structure, if any:\n{output_schema_text}\n\n"
        "Hard constraints:\n"
        "- prompt_overrides.map.user must include the literal placeholder {chunk}.\n"
        "- prompt_overrides.reduce.user must include the literal placeholder {summaries}.\n"
        "- prompt_overrides.final.user must include the literal placeholder {summaries}.\n"
        "- The final prompt may also use {report_format_instruction} and {output_json_schema}.\n"
        "- Keep prompts concise enough for production use.\n"
        "- Do not include Markdown fences."
    )
    try:
        draft = llm.call_structured(
            job_id="prompt_draft",
            node_id="prompt_draft",
            stage="PROMPT_DRAFT",
            system=system,
            user=user,
            response_model=GenerateSummaryPromptDraftResponse,
        )
    except Exception as exc:
        logger.exception("api.prompt_draft_failed")
        raise HTTPException(status_code=502, detail=f"could not generate prompt draft: {exc}") from exc
    return _normalize_prompt_draft(draft, request.request, request.output_json_schema)


def build_public_settings(current_settings=settings) -> SummaryServiceSettingsResponse:
    return SummaryServiceSettingsResponse(
        service_name="summary-generator",
        read_only=True,
        runtime=SummaryServiceRuntimeSettings(
            api_host=current_settings.api_host,
            api_port=current_settings.api_port,
            cors_origins=list(current_settings.cors_origins),
            websocket_poll_interval_seconds=current_settings.websocket_poll_interval_seconds,
            log_level=current_settings.log_level,
            broker_url=_mask_url_credentials(current_settings.broker_url),
            worker_processes=current_settings.worker_processes,
            worker_threads=current_settings.worker_threads,
        ),
        storage=SummaryServiceStorageSettings(
            log_dir=str(current_settings.log_dir),
            audit_dir=str(current_settings.audit_dir),
            upload_staging_dir=str(current_settings.upload_staging_dir),
        ),
        clickhouse=SummaryServiceDatabaseSettings(
            host=current_settings.clickhouse_host,
            port=current_settings.clickhouse_port,
            username=current_settings.clickhouse_username,
            database=current_settings.clickhouse_database,
            secure=current_settings.clickhouse_secure,
            password_configured=bool(current_settings.clickhouse_password),
        ),
        source_clickhouse=SummaryServiceDatabaseSettings(
            host=current_settings.source_clickhouse_host,
            port=current_settings.source_clickhouse_port,
            username=current_settings.source_clickhouse_username,
            database=current_settings.source_clickhouse_database,
            secure=current_settings.source_clickhouse_secure,
            password_configured=bool(current_settings.source_clickhouse_password),
        ),
        llm=SummaryServiceLlmSettings(
            api_base=current_settings.openai_api_base,
            model=current_settings.llm_model,
            timeout_seconds=current_settings.llm_timeout_seconds,
            max_retries=current_settings.llm_max_retries,
            retry_backoff_seconds=current_settings.llm_retry_backoff_seconds,
            max_concurrency=current_settings.llm_max_concurrency,
            pool_acquire_timeout_seconds=current_settings.llm_pool_acquire_timeout_seconds,
            pool_poll_interval_seconds=current_settings.llm_pool_poll_interval_seconds,
            api_key_configured=bool(current_settings.openai_api_key),
            dry_run=current_settings.dry_run,
        ),
        pipeline=SummaryServicePipelineSettings(
            chunk_target_estimated_tokens=current_settings.chunk_target_estimated_tokens,
            reduce_group_size=current_settings.reduce_group_size,
            max_enqueue_nodes_per_advance=current_settings.max_enqueue_nodes_per_advance,
        ),
    )


def _mask_url_credentials(value: str) -> str:
    if not value:
        return value
    parsed = urlsplit(value)
    if not parsed.username and not parsed.password:
        return value
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"***:***@{host}{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _fallback_prompt_draft(
    user_request: str,
    output_json_schema: dict[str, Any] | None,
) -> GenerateSummaryPromptDraftResponse:
    clean_request = " ".join(user_request.split())[:1200]
    request_line = f"\nUser-specific direction: {clean_request}" if clean_request else ""
    final_user = FINAL_USER_TEMPLATE
    if output_json_schema:
        final_user = (
            "Create the final JSON response from these summaries.\n\n"
            "Output JSON structure:\n{output_json_schema}\n\n"
            "Additional report/prompt instruction:\n{report_format_instruction}\n\n"
            "Summaries:\n\n{summaries}"
        )
    return GenerateSummaryPromptDraftResponse(
        prompt_overrides=SummaryPromptOverridesDraft(
            map=SummaryPromptStageDraft(
                system=f"{MAP_SYSTEM}{request_line}",
                user=MAP_USER_TEMPLATE,
            ),
            reduce=SummaryPromptStageDraft(
                system=f"{REDUCE_SYSTEM}{request_line}",
                user=REDUCE_USER_TEMPLATE,
            ),
            final=SummaryPromptStageDraft(
                system=f"{FINAL_SYSTEM}{request_line}",
                user=final_user,
            ),
        )
    )


def _normalize_prompt_draft(
    draft: GenerateSummaryPromptDraftResponse,
    user_request: str,
    output_json_schema: dict[str, Any] | None,
) -> GenerateSummaryPromptDraftResponse:
    fallback = _fallback_prompt_draft(user_request, output_json_schema).prompt_overrides
    raw = draft.prompt_overrides
    return GenerateSummaryPromptDraftResponse(
        prompt_overrides=SummaryPromptOverridesDraft(
            map=_normalize_prompt_stage(raw.map, fallback.map, "{chunk}"),
            reduce=_normalize_prompt_stage(raw.reduce, fallback.reduce, "{summaries}"),
            final=_normalize_prompt_stage(
                raw.final,
                fallback.final,
                "{summaries}",
                extra_placeholder="{output_json_schema}" if output_json_schema else None,
            ),
        )
    )


def _normalize_prompt_stage(
    draft: SummaryPromptStageDraft,
    fallback: SummaryPromptStageDraft,
    required_placeholder: str,
    *,
    extra_placeholder: str | None = None,
) -> SummaryPromptStageDraft:
    system = (draft.system or "").strip()[:4000] or fallback.system
    user = (draft.user or "").strip()[:8000] or fallback.user
    if required_placeholder not in user:
        user = f"{user.rstrip()}\n\nInput:\n{required_placeholder}"
    if extra_placeholder and extra_placeholder not in user:
        user = f"{user.rstrip()}\n\nOutput JSON structure:\n{extra_placeholder}"
    return SummaryPromptStageDraft(system=system, user=user)


@app.post("/summary-jobs", response_model=CreateSummaryJobResponse)
def create_summary_job(request: CreateSummaryJobRequest) -> CreateSummaryJobResponse:
    service = _service()
    job_id = service.create_job(input_text=request.input_text, title=request.title, metadata=request.metadata)
    queued = False
    if request.auto_start:
        service.queue.advance_job(job_id)  # type: ignore[union-attr]
        queued = True
    logger.info("api.create_summary_job | job_id=%s queued=%s", job_id, queued)
    return CreateSummaryJobResponse(job_id=job_id, status=JobStatus.CREATED, queued=queued)


@app.get("/summary-jobs", response_model=list[JobCurrent])
def list_summary_jobs(
    limit: int = Query(default=200, ge=1, le=1000),
    status: JobStatus | None = Query(default=None),
) -> list[JobCurrent]:
    rows = ClickHouseStore(settings).list_jobs(limit=limit, status=str(status) if status else None)
    return [JobCurrent.model_validate(row) for row in rows]


@app.post("/summary-jobs/upload", response_model=CreateSummaryJobUploadResponse)
def create_summary_job_from_upload(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    metadata: str = Form(default="{}"),
    auto_start: bool = Form(default=True),
    source_format: str = Form(default="auto"),
    raw_line_column: str | None = Form(default=None),
) -> CreateSummaryJobUploadResponse:
    try:
        metadata_payload = json.loads(metadata or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"metadata must be JSON object: {exc}") from exc
    if not isinstance(metadata_payload, dict):
        raise HTTPException(status_code=422, detail="metadata must be JSON object")

    ingestion = StagedUploadIngestionService(
        store=ClickHouseStore(settings),
        queue=DramatiqTaskQueue(),
        settings=settings,
    )
    try:
        result = ingestion.create_staged_upload_job(
            file=file.file,
            filename=file.filename or "upload",
            content_type=file.content_type or "",
            title=title,
            metadata=metadata_payload,
            requested_format=source_format,
            raw_line_column=raw_line_column,
            auto_start=auto_start,
        )
    except InputParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("api.upload_failed | filename=%s", file.filename)
        raise

    return CreateSummaryJobUploadResponse(
        job_id=result.job_id,
        status=JobStatus.INGESTING,
        queued=result.queued,
        filename=result.filename,
        source_format=result.source_format,
        segments_count=0,
        rows_count=0,
    )


@app.get("/summary-uploads", response_model=list[UploadedFileRecord])
def list_summary_uploads(limit: int = Query(default=200, ge=1, le=1000)) -> list[UploadedFileRecord]:
    rows = ClickHouseStore(settings).list_staged_uploads(limit=limit)
    return [UploadedFileRecord.model_validate(row) for row in rows]


@app.post("/summary-jobs/from-upload", response_model=CreateSummaryJobUploadResponse)
def create_summary_job_from_existing_upload(
    request: CreateSummaryJobFromUploadRequest,
) -> CreateSummaryJobUploadResponse:
    source_format = request.source_format
    if source_format == "auto":
        source_format = None
    ingestion = StagedUploadIngestionService(
        store=ClickHouseStore(settings),
        queue=DramatiqTaskQueue(),
        settings=settings,
    )
    try:
        result = ingestion.create_job_from_existing_upload(
            upload_id=request.upload_id,
            title=request.title,
            metadata=request.metadata,
            source_format=source_format,
            raw_line_column=request.raw_line_column,
            auto_start=request.auto_start,
        )
    except InputParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("api.reuse_upload_failed | upload_id=%s", request.upload_id)
        raise

    return CreateSummaryJobUploadResponse(
        job_id=result.job_id,
        status=JobStatus.INGESTING,
        queued=result.queued,
        filename=result.filename,
        source_format=result.source_format,
        segments_count=0,
        rows_count=0,
    )


@app.post("/summary-jobs/clickhouse-query", response_model=CreateSummaryJobQueryResponse)
def create_summary_job_from_clickhouse_query(request: CreateSummaryJobQueryRequest) -> CreateSummaryJobQueryResponse:
    ingestion = ClickHouseQueryIngestionService(
        store=ClickHouseStore(settings),
        queue=DramatiqTaskQueue(),
        query_source=ClickHouseQueryLogRecordSource(settings),
        settings=settings,
    )
    try:
        result = ingestion.create_job_from_query(
            query=request.query,
            title=request.title,
            metadata=request.metadata,
            raw_line_column=request.raw_line_column,
            auto_start=request.auto_start,
        )
    except QuerySourceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("api.clickhouse_query_failed")
        raise

    return CreateSummaryJobQueryResponse(
        job_id=result.job_id,
        status=JobStatus.CREATED,
        queued=result.queued,
        source_format=result.source_format,
        segments_count=result.segments_count,
        rows_count=result.rows_count,
    )


@app.get("/summary-jobs/{job_id}", response_model=JobStatusResponse)
def get_summary_job(job_id: str) -> JobStatusResponse:
    try:
        return JobStatusResponse.model_validate(_service().get_status(job_id))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}") from None


@app.get("/summary-jobs/{job_id}/snapshot")
def get_summary_job_snapshot(job_id: str) -> dict[str, object]:
    try:
        return build_job_snapshot(_service(), job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}") from None


@app.get("/summary-jobs/{job_id}/events", response_model=list[EventRecord])
def list_summary_job_events(job_id: str, limit: int = Query(default=500, ge=1, le=5000)) -> list[EventRecord]:
    rows = ClickHouseStore(settings).list_job_events(job_id, limit=limit)
    return [EventRecord.model_validate(row) for row in rows]


@app.get("/summary-jobs/{job_id}/node-events")
def list_summary_node_events(job_id: str, limit: int = Query(default=1000, ge=1, le=10000)) -> list[dict[str, object]]:
    return ClickHouseStore(settings).list_node_events(job_id, limit=limit)


@app.get("/events")
def list_summary_recent_events(limit: int = Query(default=200, ge=1, le=1000)) -> dict[str, object]:
    return {"events": ClickHouseStore(settings).list_recent_events(limit=limit)}


@app.get("/summary-jobs/{job_id}/nodes", response_model=list[NodeCurrent])
def list_summary_nodes(job_id: str) -> list[NodeCurrent]:
    rows = ClickHouseStore(settings).list_nodes_current(job_id)
    return [NodeCurrent.model_validate(row) for row in rows]


@app.get("/summary-jobs/{job_id}/artifacts", response_model=list[ArtifactRecord])
def list_summary_artifacts(
    job_id: str,
    include_content: bool = Query(default=False),
) -> list[ArtifactRecord]:
    rows = ClickHouseStore(settings).list_artifacts(job_id=job_id, include_content=include_content)
    return [ArtifactRecord.model_validate(row) for row in rows]


@app.get("/summary-jobs/{job_id}/input-segments", response_model=list[InputSegmentRecord])
def list_summary_input_segments(
    job_id: str,
    include_content: bool = Query(default=False),
) -> list[InputSegmentRecord]:
    rows = ClickHouseStore(settings).list_input_segments(job_id=job_id, include_content=include_content)
    return [InputSegmentRecord.model_validate(row) for row in rows]


@app.post("/summary-jobs/{job_id}/pause", response_model=PauseResumeResponse)
def pause_summary_job(job_id: str) -> PauseResumeResponse:
    _service().pause_job(job_id)
    return PauseResumeResponse(job_id=job_id, status=JobStatus.PAUSE_REQUESTED)


@app.post("/summary-jobs/{job_id}/resume", response_model=PauseResumeResponse)
def resume_summary_job(job_id: str) -> PauseResumeResponse:
    _service().resume_job(job_id)
    return PauseResumeResponse(job_id=job_id, status=JobStatus.RESUMED)


@app.post("/summary-jobs/{job_id}/rerun", response_model=RerunSummaryJobResponse)
def rerun_summary_job(job_id: str) -> RerunSummaryJobResponse:
    service = _service()
    try:
        new_job_id, queued = service.rerun_job(job_id, auto_start=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"summary job not found: {job_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RerunSummaryJobResponse(
        source_job_id=job_id,
        job_id=new_job_id,
        status=JobStatus.CREATED,
        queued=queued,
    )


@app.post("/summary-jobs/{job_id}/nodes/{node_id}/rerun", response_model=RerunSummaryNodeResponse)
def rerun_summary_node(job_id: str, node_id: str) -> RerunSummaryNodeResponse:
    service = _service()
    try:
        node_type, status, queued = service.rerun_node(job_id, node_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"summary job or node not found: {job_id}/{node_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RerunSummaryNodeResponse(
        job_id=job_id,
        node_id=node_id,
        node_type=str(node_type),
        status=status,
        queued=queued,
    )


@app.post("/summary-jobs/{job_id}/cancel", response_model=PauseResumeResponse)
def cancel_summary_job(job_id: str) -> PauseResumeResponse:
    _service().cancel_job(job_id)
    return PauseResumeResponse(job_id=job_id, status=JobStatus.CANCEL_REQUESTED)


@app.websocket("/ws/summary-jobs/{job_id}")
async def watch_summary_job(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    logger.info("ws.connected | job_id=%s client=%s", job_id, websocket.client)
    service = _service()
    last_hash = ""
    try:
        while True:
            try:
                payload = {
                    "type": "snapshot",
                    "snapshot": build_job_snapshot(service, job_id),
                }
            except KeyError:
                await websocket.send_json({"type": "error", "detail": f"job not found: {job_id}"})
                await websocket.close(code=1008)
                return

            encoded = jsonable_encoder(payload)
            current_hash = hashlib.sha256(
                json.dumps(encoded, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if current_hash != last_hash:
                await websocket.send_json(encoded)
                last_hash = current_hash

            try:
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=settings.websocket_poll_interval_seconds,
                )
            except asyncio.TimeoutError:
                continue
            if message == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        logger.info("ws.disconnected | job_id=%s", job_id)


@app.websocket("/ws/events")
async def watch_summary_events(websocket: WebSocket, limit: int = 200) -> None:
    await websocket.accept()
    logger.info("ws.events.connected | client=%s", websocket.client)
    store = ClickHouseStore(settings)
    last_hash = ""
    try:
        while True:
            payload = {
                "type": "events",
                "events": store.list_recent_events(limit=max(1, min(limit, 1000))),
            }
            encoded = jsonable_encoder(payload)
            current_hash = hashlib.sha256(
                json.dumps(encoded, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if current_hash != last_hash:
                await websocket.send_json(encoded)
                last_hash = current_hash

            try:
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=settings.websocket_poll_interval_seconds,
                )
            except asyncio.TimeoutError:
                continue
            if message == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        logger.info("ws.events.disconnected")
