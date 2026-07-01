"""FastAPI control plane for summary jobs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from .audit import AuditWriter
from .config import build_settings_llm_model_options, get_settings, resolve_llm_model_option
from .errors import LlmPoolBusyError
from .factory import create_pipeline_service
from .ids import new_job_id
from .ingestion import ClickHouseQueryIngestionService, StagedUploadIngestionService
from .input_parsers import InputParseError
from .llm_client import StructuredLLMClient
from .logging_setup import configure_logging, get_logger
from .pipeline import FINAL_SYSTEM, FINAL_USER_TEMPLATE, MAP_SYSTEM, MAP_USER_TEMPLATE, PipelineService, REDUCE_SYSTEM, REDUCE_USER_TEMPLATE
from .query_sources import ClickHouseQueryLogRecordSource, QuerySourceError
from .schemas import (
    ArtifactRecord,
    CreateSummaryJobFromUploadRequest,
    CreateSummaryJobQueryRequest,
    CreateSummaryJobQueryResponse,
    CreateSummaryJobRequest,
    CreateSummaryJobResponse,
    CreateSummaryJobUploadResponse,
    DatasourceQueryPreviewMetadata,
    DatasourceQueryPreviewRequest,
    DatasourceQueryPreviewResponse,
    EventRecord,
    GenerateSummaryPromptDraftRequest,
    GenerateSummaryPromptDraftResponse,
    InputSegmentRecord,
    JobCurrent,
    JobStatus,
    JobStatusResponse,
    NodeCurrent,
    PauseResumeResponse,
    PromptDraftJobRecord,
    PromptDraftJobStatus,
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
    SummaryLlmConnectivityCheckResponse,
    SummaryLlmConnectivityCheckResult,
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


class _NoopSummaryStore:
    def insert_llm_call(self, **kwargs: Any) -> None:
        return None


class PromptDraftConceptSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_name: str = ""
    report_instruction: str = ""
    objective: str = ""
    audience: str = ""
    tone: str = ""
    map_focus: Any = Field(default_factory=list)
    reduce_focus: Any = Field(default_factory=list)
    final_sections: Any = Field(default_factory=list)
    final_requirements: Any = Field(default_factory=list)
    use_custom_output_json: bool = False
    output_json_schema: dict[str, Any] | None = None


class PromptDraftCancelledError(RuntimeError):
    """Raised when a prompt-draft job is cancelled between LLM stages."""


class PromptDraftJobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, PromptDraftJobRecord] = {}

    def create_job(
        self,
        *,
        request: GenerateSummaryPromptDraftRequest,
        llm_model: str | None,
        runner: Callable[[GenerateSummaryPromptDraftRequest, str | None, Callable[[str], None], Callable[[], bool]], GenerateSummaryPromptDraftResponse] | None = None,
    ) -> PromptDraftJobRecord:
        self._prune_terminal_jobs()
        now = self._utcnow()
        job_id = f"prompt_draft_{new_job_id()}"
        job = PromptDraftJobRecord(
            job_id=job_id,
            status=PromptDraftJobStatus.QUEUED,
            stage="QUEUED",
            llm_model=llm_model or settings.llm_model,
            created_at=now,
            updated_at=now,
            finished_at=None,
            error_detail="",
            result=None,
        )
        with self._lock:
            self._jobs[job_id] = job
        worker = runner or self._run_job
        thread = threading.Thread(
            target=self._run_in_thread,
            args=(job_id, request, llm_model, worker),
            daemon=True,
            name=f"prompt-draft-{job_id[-12:]}",
        )
        thread.start()
        return self.get_job(job_id)

    def create_completed_job(self, *, result: GenerateSummaryPromptDraftResponse, llm_model: str | None) -> PromptDraftJobRecord:
        self._prune_terminal_jobs()
        now = self._utcnow()
        job = PromptDraftJobRecord(
            job_id=f"prompt_draft_{new_job_id()}",
            status=PromptDraftJobStatus.DONE,
            stage="DONE",
            llm_model=llm_model or settings.llm_model,
            created_at=now,
            updated_at=now,
            finished_at=now,
            error_detail="",
            result=result,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job.model_copy(deep=True)

    def get_job(self, job_id: str) -> PromptDraftJobRecord:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return job.model_copy(deep=True)

    def cancel_job(self, job_id: str) -> PromptDraftJobRecord:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status in {PromptDraftJobStatus.DONE, PromptDraftJobStatus.FAILED, PromptDraftJobStatus.CANCELLED}:
                return job.model_copy(deep=True)
            now = self._utcnow()
            job.status = PromptDraftJobStatus.CANCELLED
            job.stage = "CANCELLED"
            job.updated_at = now
            job.finished_at = now
            job.result = None
            return job.model_copy(deep=True)

    def _run_in_thread(
        self,
        job_id: str,
        request: GenerateSummaryPromptDraftRequest,
        llm_model: str | None,
        runner: Callable[[GenerateSummaryPromptDraftRequest, str | None, Callable[[str], None], Callable[[], bool]], GenerateSummaryPromptDraftResponse],
    ) -> None:
        if not self._mark_running(job_id):
            return
        while True:
            try:
                result = runner(request, llm_model, lambda stage: self._set_stage(job_id, stage), lambda: self._is_cancel_requested(job_id))
                break
            except PromptDraftCancelledError:
                self._mark_cancelled(job_id)
                return
            except LlmPoolBusyError as exc:
                if self._is_cancel_requested(job_id):
                    self._mark_cancelled(job_id)
                    return
                self._mark_waiting_provider(
                    job_id,
                    _build_prompt_draft_waiting_provider_detail(llm_model=llm_model, exc=exc),
                )
                time.sleep(max(1.0, float(settings.llm_pool_retry_delay_seconds)))
                continue
            except Exception as exc:
                if self._is_cancel_requested(job_id):
                    self._mark_cancelled(job_id)
                else:
                    self._mark_failed(job_id, str(exc))
                return

        if self._is_cancel_requested(job_id):
            self._mark_cancelled(job_id)
            return
        self._mark_done(job_id, result)

    def _run_job(
        self,
        request: GenerateSummaryPromptDraftRequest,
        llm_model: str | None,
        stage_observer: Callable[[str], None],
        cancel_checker: Callable[[], bool],
    ) -> GenerateSummaryPromptDraftResponse:
        store = ClickHouseStore(settings)
        llm = StructuredLLMClient(store=store, settings=settings, audit=AuditWriter(settings))
        return _generate_prompt_draft_with_llm(
            llm=llm,
            request=request,
            llm_model=llm_model,
            stage_observer=stage_observer,
            cancel_checker=cancel_checker,
        )

    def _mark_running(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status == PromptDraftJobStatus.CANCELLED:
                return False
            job.status = PromptDraftJobStatus.RUNNING
            job.stage = "RUNNING"
            job.updated_at = self._utcnow()
            job.error_detail = ""
            return True

    def _set_stage(self, job_id: str, stage: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in {PromptDraftJobStatus.DONE, PromptDraftJobStatus.FAILED, PromptDraftJobStatus.CANCELLED}:
                return
            job.stage = stage
            if job.status != PromptDraftJobStatus.CANCEL_REQUESTED:
                job.status = PromptDraftJobStatus.WAITING_PROVIDER if stage == "WAITING_PROVIDER" else PromptDraftJobStatus.RUNNING
            job.updated_at = self._utcnow()
            if stage != "WAITING_PROVIDER":
                job.error_detail = ""

    def _mark_waiting_provider(self, job_id: str, error_detail: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in {PromptDraftJobStatus.DONE, PromptDraftJobStatus.FAILED, PromptDraftJobStatus.CANCELLED}:
                return
            now = self._utcnow()
            if job.status != PromptDraftJobStatus.CANCEL_REQUESTED:
                job.status = PromptDraftJobStatus.WAITING_PROVIDER
            job.stage = "WAITING_PROVIDER"
            job.updated_at = now
            job.error_detail = error_detail[:4000]

    def _mark_done(self, job_id: str, result: GenerateSummaryPromptDraftResponse) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            now = self._utcnow()
            job.status = PromptDraftJobStatus.DONE
            job.stage = "DONE"
            job.updated_at = now
            job.finished_at = now
            job.error_detail = ""
            job.result = result

    def _mark_failed(self, job_id: str, error_detail: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            now = self._utcnow()
            job.status = PromptDraftJobStatus.FAILED
            job.stage = "FAILED"
            job.updated_at = now
            job.finished_at = now
            job.error_detail = error_detail[:4000]
            job.result = None

    def _mark_cancelled(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            now = self._utcnow()
            job.status = PromptDraftJobStatus.CANCELLED
            job.stage = "CANCELLED"
            job.updated_at = now
            job.finished_at = now
            job.result = None

    def _is_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            return bool(job and job.status in {PromptDraftJobStatus.CANCEL_REQUESTED, PromptDraftJobStatus.CANCELLED})

    def _prune_terminal_jobs(self, limit: int = 200) -> None:
        with self._lock:
            terminal_jobs = [
                job
                for job in self._jobs.values()
                if job.status in {PromptDraftJobStatus.DONE, PromptDraftJobStatus.FAILED, PromptDraftJobStatus.CANCELLED}
            ]
            if len(terminal_jobs) < limit:
                return
            terminal_jobs.sort(key=lambda item: item.updated_at)
            for job in terminal_jobs[: max(0, len(terminal_jobs) - limit + 1)]:
                self._jobs.pop(job.job_id, None)

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)


prompt_draft_jobs = PromptDraftJobManager()


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


@app.post("/settings/llm-connectivity-check", response_model=SummaryLlmConnectivityCheckResponse)
def run_settings_llm_connectivity_check(llm_model: str | None = Query(default=None)) -> SummaryLlmConnectivityCheckResponse:
    client = StructuredLLMClient(store=_NoopSummaryStore(), settings=settings, audit=AuditWriter(settings))
    results = _run_llm_connectivity_checks(client=client, current_settings=settings, llm_model=llm_model)
    ok_count = sum(1 for result in results if result.ok)
    failed_count = len(results) - ok_count
    return SummaryLlmConnectivityCheckResponse(
        checked_at=datetime.now(timezone.utc),
        dry_run=settings.dry_run,
        total=len(results),
        ok_count=ok_count,
        failed_count=failed_count,
        results=results,
    )


@app.post("/prompt-draft-jobs", response_model=PromptDraftJobRecord)
def create_prompt_draft_job(request: GenerateSummaryPromptDraftRequest) -> PromptDraftJobRecord:
    llm_model = _resolve_requested_prompt_draft_model(request.llm_model)
    normalized_request = request.model_copy(update={"llm_model": llm_model})
    if settings.dry_run:
        job = prompt_draft_jobs.create_completed_job(
            result=_fallback_prompt_draft(request.request, request.use_custom_output_json, request.output_json_schema),
            llm_model=llm_model,
        )
    else:
        job = prompt_draft_jobs.create_job(request=normalized_request, llm_model=llm_model)
    logger.info("api.prompt_draft_job_created | job_id=%s status=%s", job.job_id, job.status)
    return job


@app.get("/prompt-draft-jobs/{job_id}", response_model=PromptDraftJobRecord)
def get_prompt_draft_job(job_id: str) -> PromptDraftJobRecord:
    try:
        return prompt_draft_jobs.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"prompt draft job not found: {job_id}") from exc


@app.post("/prompt-draft-jobs/{job_id}/cancel", response_model=PromptDraftJobRecord)
def cancel_prompt_draft_job(job_id: str) -> PromptDraftJobRecord:
    try:
        job = prompt_draft_jobs.cancel_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"prompt draft job not found: {job_id}") from exc
    logger.info("api.prompt_draft_job_cancel | job_id=%s status=%s", job.job_id, job.status)
    return job


@app.post("/prompt-drafts", response_model=GenerateSummaryPromptDraftResponse)
def generate_summary_prompt_draft(request: GenerateSummaryPromptDraftRequest) -> GenerateSummaryPromptDraftResponse:
    if settings.dry_run:
        return _fallback_prompt_draft(request.request, request.use_custom_output_json, request.output_json_schema)

    store = ClickHouseStore(settings)
    llm = StructuredLLMClient(store=store, settings=settings, audit=AuditWriter(settings))
    requested_llm_model = _resolve_requested_prompt_draft_model(request.llm_model)
    try:
        return _generate_prompt_draft_with_llm(
            llm=llm,
            request=request,
            llm_model=requested_llm_model,
        )
    except Exception as exc:
        logger.exception("api.prompt_draft_failed")
        raise HTTPException(status_code=502, detail=f"could not generate prompt draft: {exc}") from exc


def build_public_settings(current_settings=settings) -> SummaryServiceSettingsResponse:
    llm_model_options = build_settings_llm_model_options(current_settings)
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
            available_models=list(current_settings.llm_models or ((current_settings.llm_model,) if current_settings.llm_model else ())),
            available_model_options=[
                {
                    "value": option.value,
                    "label": option.label,
                    "profile_id": option.profile_id,
                    "profile_label": option.profile_label,
                    "model": option.model,
                    "api_base": option.api_base,
                    "api_key_configured": bool(
                        next(
                            (
                                profile.api_key
                                for profile in current_settings.llm_profiles
                                if profile.profile_id == option.profile_id
                            ),
                            current_settings.openai_api_key,
                        )
                    ),
                    "is_default": option.value == current_settings.llm_model,
                }
                for option in llm_model_options
            ],
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


def _build_prompt_draft_waiting_provider_detail(
    *,
    llm_model: str | None,
    exc: LlmPoolBusyError,
) -> str:
    probe_detail = _probe_prompt_draft_llm_availability(llm_model=llm_model)
    retry_delay_seconds = max(1.0, float(settings.llm_pool_retry_delay_seconds))
    return f"{exc}. {probe_detail} Retrying in {retry_delay_seconds:.0f}s."


def _probe_prompt_draft_llm_availability(*, llm_model: str | None) -> str:
    try:
        probe_client = StructuredLLMClient(
            store=_NoopSummaryStore(),
            settings=settings,
            audit=AuditWriter(settings),
        )
        probe = probe_client.probe_connection(
            model=llm_model,
            timeout_seconds=min(5.0, max(2.0, float(settings.llm_timeout_seconds))),
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        return f"LLM availability ping failed: {exc}"

    status = str(probe.get("status") or "error")
    detail = str(probe.get("detail") or "").strip()
    latency_ms = probe.get("latency_ms")
    if status == "ok":
        latency_suffix = f" ({latency_ms} ms)" if latency_ms is not None else ""
        return f"LLM availability ping succeeded{latency_suffix}."
    if status == "dry_run":
        return detail or "Dry run mode is enabled."
    if detail:
        return f"LLM availability ping status={status}: {detail}"
    return f"LLM availability ping status={status}."


def _run_llm_connectivity_checks(
    *,
    client: StructuredLLMClient,
    current_settings=settings,
    llm_model: str | None = None,
) -> list[SummaryLlmConnectivityCheckResult]:
    results: list[SummaryLlmConnectivityCheckResult] = []
    selected_model = (llm_model or "").strip()
    for option in build_settings_llm_model_options(current_settings):
        if selected_model and option.value != selected_model:
            continue
        probe = client.probe_connection(model=option.value)
        results.append(
            SummaryLlmConnectivityCheckResult(
                value=option.value,
                label=option.label,
                profile_id=option.profile_id,
                profile_label=option.profile_label,
                model=probe.get("selected_model") or option.model,
                api_base=probe.get("api_base") or option.api_base,
                ok=bool(probe.get("ok")),
                status=str(probe.get("status") or "error"),
                detail=str(probe.get("detail") or ""),
                error_class=str(probe.get("error_class") or ""),
                latency_ms=int(probe["latency_ms"]) if probe.get("latency_ms") is not None else None,
            )
        )
    return results


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
    use_custom_output_json: bool | None,
    output_json_schema: dict[str, Any] | None,
) -> GenerateSummaryPromptDraftResponse:
    clean_request = " ".join(user_request.split())[:1200]
    request_line = f"\nUser-specific direction: {clean_request}" if clean_request else ""
    report_name = _fallback_report_name(clean_request)
    report_instruction = clean_request or "Create a concise report that follows the user request."
    normalized_output_json_schema = _normalize_output_json_schema(
        output_json_schema,
        provided_output_json_schema=output_json_schema,
        use_custom_output_json=bool(use_custom_output_json),
        user_request=user_request,
    )
    final_user = FINAL_USER_TEMPLATE
    if normalized_output_json_schema:
        final_user = (
            "Create the final JSON response from these summaries.\n\n"
            "Output JSON structure:\n{output_json_schema}\n\n"
            "Additional report/prompt instruction:\n{report_format_instruction}\n\n"
            "Summaries:\n\n{summaries}"
        )
    return GenerateSummaryPromptDraftResponse(
        report_name=report_name,
        report_instruction=report_instruction,
        use_custom_output_json=normalized_output_json_schema is not None,
        output_json_schema=normalized_output_json_schema,
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


def _resolve_requested_prompt_draft_model(requested_llm_model: str | None) -> str | None:
    requested_model = (requested_llm_model or "").strip() or None
    available_models = tuple(model.strip() for model in settings.llm_models if model.strip())
    if requested_model:
        resolved = resolve_llm_model_option(settings, requested_model)
        shorthand_match = any(option.model == requested_model for option in build_settings_llm_model_options(settings))
        if shorthand_match:
            return resolved.value
    if requested_model and available_models and requested_model not in available_models:
        logger.warning(
            "api.prompt_draft_model_unavailable | requested=%s fallback=%s available=%s",
            requested_model,
            settings.llm_model,
            ",".join(available_models),
        )
        return settings.llm_model
    return requested_model


def _generate_prompt_draft_with_llm(
    *,
    llm: StructuredLLMClient,
    request: GenerateSummaryPromptDraftRequest,
    llm_model: str | None,
    stage_observer: Callable[[str], None] | None = None,
    cancel_checker: Callable[[], bool] | None = None,
) -> GenerateSummaryPromptDraftResponse:
    _raise_if_prompt_draft_cancelled(cancel_checker)
    if stage_observer:
        stage_observer("PROMPT_DRAFT_SPEC")
    spec = _generate_prompt_draft_concept_spec(
        llm=llm,
        request=request,
        llm_model=llm_model,
    )
    _raise_if_prompt_draft_cancelled(cancel_checker)
    normalized_spec = _normalize_prompt_draft_spec(
        spec,
        request.output_json_schema,
        request.use_custom_output_json,
        request.request,
    )
    if stage_observer:
        stage_observer("PROMPT_DRAFT_RENDER")
    draft = _generate_prompt_draft_from_spec(
        llm=llm,
        request=request,
        llm_model=llm_model,
        spec=normalized_spec,
    )
    _raise_if_prompt_draft_cancelled(cancel_checker)
    if stage_observer:
        stage_observer("PROMPT_DRAFT_NORMALIZE")
    merged_draft = _merge_prompt_draft_spec_into_draft(draft, normalized_spec)
    return _normalize_prompt_draft(
        merged_draft,
        request.request,
        request.output_json_schema,
        request.use_custom_output_json,
    )


def _raise_if_prompt_draft_cancelled(cancel_checker: Callable[[], bool] | None) -> None:
    if cancel_checker and cancel_checker():
        raise PromptDraftCancelledError("prompt draft job cancelled")


def _generate_prompt_draft_concept_spec(
    *,
    llm: StructuredLLMClient,
    request: GenerateSummaryPromptDraftRequest,
    llm_model: str | None,
) -> PromptDraftConceptSpec:
    output_schema_text = json.dumps(request.output_json_schema or {}, ensure_ascii=False, indent=2)
    custom_output_requested = bool(request.use_custom_output_json or request.output_json_schema)
    system = (
        "You convert a user's conceptual request for a three-stage summarization pipeline into a structured design spec. "
        "Return only JSON matching the requested schema. "
        "Extract implementation-ready intent for MAP, REDUCE, and FINAL so another model can write concrete prompts without guessing. "
        "Keep the spec concise, practical, and internally consistent. "
        "If the user already provided a final output JSON structure, preserve it and set use_custom_output_json to true. "
        "If the request explicitly asks for a strict final JSON structure, create one that is directly usable as a JSON schema for an object response. "
        "If no strict final structure is clearly needed, keep use_custom_output_json false and output_json_schema null."
    )
    user = (
        "Create a structured prompt-design spec from this conceptual request.\n\n"
        f"User request:\n{request.request.strip()}\n\n"
        f"Custom final JSON schema requested: {'yes' if custom_output_requested else 'no'}\n\n"
        f"Desired final output JSON structure, if any:\n{output_schema_text}\n\n"
        "Fill the spec with implementation-ready detail:\n"
        "- objective: what the summarization/reporting pipeline should accomplish\n"
        "- audience: who will read the final report\n"
        "- tone: how the report should sound\n"
        "- map_focus: what to extract from each source chunk\n"
        "- reduce_focus: how partial summaries should be merged\n"
        "- final_sections: preferred report sections or layout\n"
        "- final_requirements: evidence rules, severity handling, uncertainty, required details, and other constraints\n"
        "- report_instruction: reusable instruction text for the report form\n"
        "- report_name: short human-friendly title when helpful\n"
        "- use_custom_output_json/output_json_schema: only when strict structured final output is genuinely useful, unless the request explicitly says custom JSON output is required\n"
        "- When output_json_schema is needed, return a practical object schema with stable keys and concrete field types\n"
        "Do not write MAP, REDUCE, or FINAL prompts yet."
    )
    return llm.call_structured(
        job_id="prompt_draft",
        node_id="prompt_draft_spec",
        stage="PROMPT_DRAFT_SPEC",
        system=system,
        user=user,
        model=llm_model,
        response_model=PromptDraftConceptSpec,
    )


def _generate_prompt_draft_from_spec(
    *,
    llm: StructuredLLMClient,
    request: GenerateSummaryPromptDraftRequest,
    llm_model: str | None,
    spec: PromptDraftConceptSpec,
) -> GenerateSummaryPromptDraftResponse:
    spec_text = json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2)
    output_schema_text = json.dumps(spec.output_json_schema or {}, ensure_ascii=False, indent=2)
    system = (
        "You generate concrete MAP, REDUCE, and FINAL prompt fields for a three-stage summarization pipeline from a structured design spec. "
        "Return only JSON matching the requested schema. "
        "All three stages must follow the same objective, audience, tone, and evidence rules from the spec. "
        "Prompt quality is the primary objective; report_name and report_instruction should stay aligned with the spec. "
        "Each stage must include a system prompt and a user prompt template. "
        "The MAP and REDUCE stages must keep returning the internal SummaryResult JSON shape: "
        '{"ok": true, "summary": "string", "key_points": ["string"], "warnings": ["string"], "source_count": 1}. '
        "MAP should explain what to extract from a single chunk. "
        "REDUCE should explain how to merge partial summaries without losing evidence. "
        "If a final output JSON structure is provided, the FINAL stage must return that structure. "
        "Otherwise FINAL must keep the SummaryResult transport JSON shape and put the full report in summary."
    )
    user = (
        "Create prompt field content for a summary-generation job from this structured design spec.\n\n"
        f"Original user request:\n{request.request.strip()}\n\n"
        f"Design spec:\n{spec_text}\n\n"
        f"Desired final output JSON structure, if any:\n{output_schema_text}\n\n"
        "Hard constraints:\n"
        "- report_name and report_instruction should stay consistent with the design spec.\n"
        "- Convert the conceptual request into practical production-ready prompts.\n"
        "- prompt_overrides.map.user must include the literal placeholder {chunk}.\n"
        "- prompt_overrides.reduce.user must include the literal placeholder {summaries}.\n"
        "- prompt_overrides.final.user must include the literal placeholder {summaries}.\n"
        "- The final prompt may also use {report_format_instruction} and {output_json_schema}.\n"
        "- Without a custom final output schema, the final prompt should let the LLM choose a useful report layout inside summary.\n"
        "- Keep prompts concise enough for production use.\n"
        "- Do not include Markdown fences."
    )
    return llm.call_structured(
        job_id="prompt_draft",
        node_id="prompt_draft_render",
        stage="PROMPT_DRAFT_RENDER",
        system=system,
        user=user,
        model=llm_model,
        response_model=GenerateSummaryPromptDraftResponse,
    )


def _normalize_prompt_draft_spec(
    spec: PromptDraftConceptSpec,
    provided_output_json_schema: dict[str, Any] | None,
    requested_use_custom_output_json: bool | None = None,
    user_request: str = "",
) -> PromptDraftConceptSpec:
    effective_use_custom_output_json = _resolve_prompt_draft_custom_output_json_requested(
        requested_use_custom_output_json,
        fallback=spec.use_custom_output_json,
        provided_output_json_schema=provided_output_json_schema,
    )
    normalized_output_json_schema = _normalize_output_json_schema(
        spec.output_json_schema,
        provided_output_json_schema=provided_output_json_schema,
        use_custom_output_json=effective_use_custom_output_json,
        user_request=user_request,
    )
    return PromptDraftConceptSpec(
        report_name=(spec.report_name or "").strip()[:200],
        report_instruction=(spec.report_instruction or "").strip()[:4000],
        objective=(spec.objective or "").strip()[:2000],
        audience=(spec.audience or "").strip()[:500],
        tone=(spec.tone or "").strip()[:500],
        map_focus=_normalize_prompt_spec_list(spec.map_focus),
        reduce_focus=_normalize_prompt_spec_list(spec.reduce_focus),
        final_sections=_normalize_prompt_spec_list(spec.final_sections),
        final_requirements=_normalize_prompt_spec_list(spec.final_requirements),
        use_custom_output_json=normalized_output_json_schema is not None,
        output_json_schema=normalized_output_json_schema,
    )


def _merge_prompt_draft_spec_into_draft(
    draft: GenerateSummaryPromptDraftResponse,
    spec: PromptDraftConceptSpec,
) -> GenerateSummaryPromptDraftResponse:
    return GenerateSummaryPromptDraftResponse(
        report_name=(draft.report_name or "").strip() or spec.report_name,
        report_instruction=(draft.report_instruction or "").strip() or spec.report_instruction,
        use_custom_output_json=bool(draft.use_custom_output_json or spec.use_custom_output_json),
        output_json_schema=draft.output_json_schema or spec.output_json_schema,
        prompt_overrides=draft.prompt_overrides,
    )


def _normalize_prompt_spec_list(items: Any) -> list[str]:
    normalized: list[str] = []
    iterable = items if isinstance(items, (list, tuple, set)) else [items]
    for item in iterable:
        clean = _stringify_prompt_spec_value(item)
        if clean and clean not in normalized:
            normalized.append(clean[:500])
    return normalized[:12]


def _stringify_prompt_spec_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            rendered = _stringify_prompt_spec_value(item)
            label = str(key or "").strip()
            if rendered and label:
                parts.append(f"{label}: {rendered}")
            elif rendered:
                parts.append(rendered)
            elif label:
                parts.append(label)
        return "; ".join(parts).strip()
    if isinstance(value, (list, tuple, set)):
        parts = [_stringify_prompt_spec_value(item) for item in value]
        return "; ".join(part for part in parts if part).strip()
    return str(value).strip()


def _normalize_prompt_draft(
    draft: GenerateSummaryPromptDraftResponse,
    user_request: str,
    output_json_schema: dict[str, Any] | None,
    requested_use_custom_output_json: bool | None,
) -> GenerateSummaryPromptDraftResponse:
    fallback = _fallback_prompt_draft(user_request, requested_use_custom_output_json, output_json_schema)
    raw = draft.prompt_overrides
    effective_use_custom_output_json = _resolve_prompt_draft_custom_output_json_requested(
        requested_use_custom_output_json,
        fallback=draft.use_custom_output_json,
        provided_output_json_schema=output_json_schema,
    )
    normalized_output_json_schema = _normalize_output_json_schema(
        draft.output_json_schema,
        provided_output_json_schema=output_json_schema,
        use_custom_output_json=effective_use_custom_output_json,
        user_request=user_request,
    )
    return GenerateSummaryPromptDraftResponse(
        report_name=(draft.report_name or "").strip()[:200] or fallback.report_name,
        report_instruction=(draft.report_instruction or "").strip()[:4000] or fallback.report_instruction,
        use_custom_output_json=normalized_output_json_schema is not None,
        output_json_schema=normalized_output_json_schema,
        prompt_overrides=SummaryPromptOverridesDraft(
            map=_normalize_prompt_stage(raw.map, fallback.prompt_overrides.map, "{chunk}"),
            reduce=_normalize_prompt_stage(raw.reduce, fallback.prompt_overrides.reduce, "{summaries}"),
            final=_normalize_prompt_stage(
                raw.final,
                fallback.prompt_overrides.final,
                "{summaries}",
                extra_placeholder="{output_json_schema}" if normalized_output_json_schema else None,
            ),
        )
    )


def _normalize_output_json_schema(
    draft_output_json_schema: dict[str, Any] | None,
    *,
    provided_output_json_schema: dict[str, Any] | None,
    use_custom_output_json: bool,
    user_request: str = "",
) -> dict[str, Any] | None:
    if provided_output_json_schema:
        return PipelineService._build_final_response_schema(provided_output_json_schema)
    if not use_custom_output_json:
        return None
    if isinstance(draft_output_json_schema, dict) and draft_output_json_schema:
        return PipelineService._build_final_response_schema(draft_output_json_schema)
    return _build_fallback_generated_output_json_schema(user_request)


def _build_fallback_generated_output_json_schema(user_request: str) -> dict[str, Any]:
    title_description = _fallback_report_name(user_request)
    shape = {
        "report_title": f"string: short title for {title_description.lower() or 'the report'}",
        "summary": "string: concise executive summary of the final report",
        "key_points": ["string: concrete evidence-backed finding"],
        "recommendations": ["string: specific next action or recommendation"],
        "warnings": ["string: uncertainty, limitation, or caveat"],
    }
    return PipelineService._build_final_response_schema(shape)


def _resolve_prompt_draft_custom_output_json_requested(
    requested_use_custom_output_json: bool | None,
    *,
    fallback: bool,
    provided_output_json_schema: dict[str, Any] | None,
) -> bool:
    if provided_output_json_schema:
        return True
    if requested_use_custom_output_json is True:
        return True
    if requested_use_custom_output_json is False:
        return False
    return bool(fallback)


def _fallback_report_name(user_request: str) -> str:
    if not user_request:
        return "Generated report"
    compact = " ".join(user_request.split())
    normalized = compact[:80].strip(" .,:;")
    if not normalized:
        return "Generated report"
    return normalized


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


@app.post("/datasources/query", response_model=DatasourceQueryPreviewResponse)
def preview_source_clickhouse_query(request: DatasourceQueryPreviewRequest) -> DatasourceQueryPreviewResponse:
    source = ClickHouseQueryLogRecordSource(settings)
    started_at = perf_counter()
    try:
        columns, rows = source.preview_rows(request.query, limit=request.limit)
    except QuerySourceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("api.datasource_query_preview_failed")
        raise HTTPException(status_code=502, detail=f"query preview failed: {exc}") from exc

    return DatasourceQueryPreviewResponse(
        datasource=request.datasource or "source_clickhouse",
        query=request.query,
        metadata=DatasourceQueryPreviewMetadata(
            columns=columns,
            records_count=len(rows),
            execution_time_ms=max(1, round((perf_counter() - started_at) * 1000)),
        ),
        data=[],
        raw_rows=jsonable_encoder(rows),
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


@app.websocket("/ws/prompt-draft-jobs/{job_id}")
async def watch_prompt_draft_job(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    logger.info("ws.prompt_draft.connected | job_id=%s client=%s", job_id, websocket.client)
    last_hash = ""
    try:
        while True:
            try:
                payload = {
                    "type": "snapshot",
                    "job": prompt_draft_jobs.get_job(job_id),
                }
            except KeyError:
                await websocket.send_json({"type": "error", "detail": f"prompt draft job not found: {job_id}"})
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
        logger.info("ws.prompt_draft.disconnected | job_id=%s", job_id)


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
