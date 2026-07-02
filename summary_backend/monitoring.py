"""Monitoring profiles executed via Dify workflows."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter

from .config import Settings, get_settings
from .dify_client import DifyWorkflowClient
from .ids import new_monitoring_profile_id, new_monitoring_run_id
from .logging_setup import get_logger
from .schemas import (
    CreateMonitoringProfileRequest,
    CreateMonitoringRunRequest,
    MonitoringProfileRecord,
    MonitoringRunRecord,
    MonitoringRunStatus,
    MonitoringRunTrigger,
    MonitoringScheduleConfig,
    MonitoringScheduleTickItem,
    MonitoringSchedulerTickResponse,
)

logger = get_logger("monitoring")

SUCCESSFUL_DIFY_STATUSES = {"succeeded", "success", "done", "completed"}
FAILED_DIFY_STATUSES = {"failed", "error", "cancelled", "stopped", "timeout"}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def validate_schedule(schedule: MonitoringScheduleConfig) -> None:
    if not schedule.enabled:
        return
    if not str(schedule.cron or "").strip():
        raise ValueError("schedule.cron is required when schedule is enabled")
    try:
        ZoneInfo(schedule.timezone or "UTC")
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown timezone: {schedule.timezone}") from exc
    try:
        croniter(schedule.cron, datetime.now())
    except Exception as exc:
        raise ValueError(f"invalid cron expression: {schedule.cron}") from exc


def compute_next_run_at(*, cron: str, timezone_name: str, after: datetime | None = None) -> datetime:
    base_utc = after.astimezone(timezone.utc) if after else _utcnow()
    tz = ZoneInfo(timezone_name or "UTC")
    base_local = base_utc.astimezone(tz)
    iterator = croniter(cron, base_local)
    next_local = iterator.get_next(datetime)
    if next_local.tzinfo is None:
        next_local = next_local.replace(tzinfo=tz)
    return next_local.astimezone(timezone.utc)


def _coerce_profile(row: dict[str, Any] | None, schedule_row: dict[str, Any] | None = None) -> MonitoringProfileRecord | None:
    if not row:
        return None
    schedule = None
    next_run_at = None
    last_run_at = None
    if schedule_row:
        schedule = MonitoringScheduleConfig(
            enabled=_as_bool(schedule_row.get("is_enabled")),
            cron=str(schedule_row.get("cron") or ""),
            timezone=str(schedule_row.get("timezone") or "UTC"),
            max_active_runs=max(1, int(schedule_row.get("max_active_runs") or 1)),
        )
        next_run_at = schedule_row.get("next_run_at")
        last_run_at = schedule_row.get("last_run_at")
    return MonitoringProfileRecord(
        profile_id=str(row.get("profile_id") or ""),
        name=str(row.get("name") or ""),
        service=str(row.get("service") or ""),
        description=str(row.get("description") or ""),
        workflow_id=str(row.get("workflow_id") or ""),
        workflow_inputs=_parse_object(row.get("workflow_inputs")),
        metadata=_parse_object(row.get("metadata")),
        is_archived=_as_bool(row.get("is_archived")),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        schedule=schedule,
        next_run_at=next_run_at,
        last_run_at=last_run_at,
    )


def _coerce_run(row: dict[str, Any] | None) -> MonitoringRunRecord | None:
    if not row:
        return None
    return MonitoringRunRecord(
        run_id=str(row.get("run_id") or ""),
        profile_id=str(row.get("profile_id") or ""),
        status=MonitoringRunStatus(str(row.get("status") or MonitoringRunStatus.CREATED)),
        trigger_type=MonitoringRunTrigger(str(row.get("trigger_type") or MonitoringRunTrigger.MANUAL)),
        workflow_id=str(row.get("workflow_id") or ""),
        workflow_run_id=str(row.get("workflow_run_id") or ""),
        task_id=str(row.get("task_id") or ""),
        requested_at=row.get("requested_at"),
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
        scheduled_for=row.get("scheduled_for"),
        workflow_inputs=_parse_object(row.get("inputs_json")),
        output=_parse_object(row.get("output_json")),
        metadata=_parse_object(row.get("metadata")),
        error_message=str(row.get("error_message") or ""),
        updated_at=row.get("updated_at"),
    )


@dataclass
class MonitoringService:
    store: Any
    workflow_client: Any
    settings: Settings

    def create_profile(self, request: CreateMonitoringProfileRequest) -> MonitoringProfileRecord:
        schedule = request.schedule or MonitoringScheduleConfig()
        validate_schedule(schedule)
        now = _utcnow()
        profile_id = new_monitoring_profile_id()
        self.store.insert_monitoring_profile_snapshot(
            profile_id=profile_id,
            name=request.name,
            service=request.service,
            description=request.description,
            workflow_id=request.workflow_id,
            workflow_inputs=request.workflow_inputs,
            metadata=request.metadata,
            is_archived=False,
            created_at=now,
            updated_at=now,
        )
        self.store.insert_monitoring_schedule_snapshot(
            profile_id=profile_id,
            cron=schedule.cron,
            timezone=schedule.timezone,
            is_enabled=schedule.enabled,
            max_active_runs=schedule.max_active_runs,
            next_run_at=compute_next_run_at(cron=schedule.cron, timezone_name=schedule.timezone, after=now) if schedule.enabled else None,
            last_run_at=None,
            created_at=now,
            updated_at=now,
        )
        profile = self.get_profile(profile_id)
        if profile is None:
            raise RuntimeError(f"failed to load created profile: {profile_id}")
        return profile

    def list_profiles(self, *, include_archived: bool = False) -> list[MonitoringProfileRecord]:
        profiles = []
        for row in self.store.list_monitoring_profiles(include_archived=include_archived):
            schedule_row = self.store.get_monitoring_schedule_current(str(row.get("profile_id") or ""))
            profile = _coerce_profile(row, schedule_row)
            if profile:
                profiles.append(profile)
        return profiles

    def get_profile(self, profile_id: str) -> MonitoringProfileRecord | None:
        return _coerce_profile(
            self.store.get_monitoring_profile_current(profile_id),
            self.store.get_monitoring_schedule_current(profile_id),
        )

    def archive_profile(self, profile_id: str) -> MonitoringProfileRecord:
        current = self.get_profile(profile_id)
        if current is None:
            raise KeyError(profile_id)
        now = _utcnow()
        self.store.insert_monitoring_profile_snapshot(
            profile_id=current.profile_id,
            name=current.name,
            service=current.service,
            description=current.description,
            workflow_id=current.workflow_id,
            workflow_inputs=current.workflow_inputs,
            metadata=current.metadata,
            is_archived=True,
            created_at=current.created_at or now,
            updated_at=now,
        )
        archived = self.get_profile(profile_id)
        if archived is None:
            raise RuntimeError(f"failed to load archived profile: {profile_id}")
        return archived

    def list_runs(self, *, profile_id: str | None = None, status: str | None = None, limit: int = 200) -> list[MonitoringRunRecord]:
        rows = self.store.list_monitoring_runs(profile_id=profile_id, status=status, limit=limit)
        return [item for item in (_coerce_run(row) for row in rows) if item]

    def get_run(self, run_id: str) -> MonitoringRunRecord | None:
        return _coerce_run(self.store.get_monitoring_run_current(run_id))

    def create_run(
        self,
        profile_id: str,
        request: CreateMonitoringRunRequest | None = None,
        *,
        trigger_type: MonitoringRunTrigger = MonitoringRunTrigger.MANUAL,
        scheduled_for: datetime | None = None,
        auto_start: bool = True,
    ) -> MonitoringRunRecord:
        profile = self.get_profile(profile_id)
        if profile is None:
            raise KeyError(profile_id)
        if profile.is_archived:
            raise ValueError(f"profile is archived: {profile_id}")

        override_inputs = (request.workflow_inputs_override if request else {}) or {}
        workflow_inputs = dict(profile.workflow_inputs)
        workflow_inputs.update(override_inputs)
        now = _utcnow()
        run_id = new_monitoring_run_id()
        self.store.insert_monitoring_run_snapshot(
            run_id=run_id,
            profile_id=profile.profile_id,
            status=MonitoringRunStatus.CREATED,
            trigger_type=trigger_type,
            workflow_id=profile.workflow_id,
            workflow_run_id="",
            task_id="",
            requested_at=now,
            started_at=None,
            finished_at=None,
            scheduled_for=scheduled_for,
            inputs_json=workflow_inputs,
            output_json={},
            error_message="",
            metadata={
                "profile_name": profile.name,
                "service": profile.service,
            },
            updated_at=now,
        )
        if auto_start:
            thread = threading.Thread(
                target=self._execute_run,
                args=(run_id,),
                daemon=True,
                name=f"monitoring-run-{run_id[-10:]}",
            )
            thread.start()
        run = self.get_run(run_id)
        if run is None:
            raise RuntimeError(f"failed to load created run: {run_id}")
        return run

    def _execute_run(self, run_id: str) -> None:
        current = self.get_run(run_id)
        if current is None:
            return
        started_at = _utcnow()
        self.store.insert_monitoring_run_snapshot(
            run_id=current.run_id,
            profile_id=current.profile_id,
            status=MonitoringRunStatus.RUNNING,
            trigger_type=current.trigger_type,
            workflow_id=current.workflow_id,
            workflow_run_id=current.workflow_run_id,
            task_id=current.task_id,
            requested_at=current.requested_at or started_at,
            started_at=started_at,
            finished_at=None,
            scheduled_for=current.scheduled_for,
            inputs_json=current.workflow_inputs,
            output_json=current.output,
            error_message="",
            metadata=current.metadata,
            updated_at=started_at,
        )
        try:
            result = self.workflow_client.run_workflow(
                workflow_id=current.workflow_id,
                inputs=current.workflow_inputs,
                user=f"monitoring-profile:{current.profile_id}",
                timeout_seconds=self.settings.dify_timeout_seconds,
            )
            final_status = MonitoringRunStatus.DONE
            if result.status and result.status.strip().lower() in FAILED_DIFY_STATUSES:
                final_status = MonitoringRunStatus.FAILED
            elif result.status and result.status.strip().lower() not in SUCCESSFUL_DIFY_STATUSES:
                final_status = MonitoringRunStatus.DONE
            finished_at = _utcnow()
            self.store.insert_monitoring_run_snapshot(
                run_id=current.run_id,
                profile_id=current.profile_id,
                status=final_status,
                trigger_type=current.trigger_type,
                workflow_id=current.workflow_id,
                workflow_run_id=result.workflow_run_id,
                task_id=result.task_id,
                requested_at=current.requested_at or started_at,
                started_at=started_at,
                finished_at=finished_at,
                scheduled_for=current.scheduled_for,
                inputs_json=current.workflow_inputs,
                output_json={
                    "status": result.status,
                    "outputs": result.outputs,
                    "raw": result.raw,
                },
                error_message="",
                metadata=current.metadata,
                updated_at=finished_at,
            )
        except Exception as exc:
            finished_at = _utcnow()
            logger.exception("monitoring_run_failed | run_id=%s profile_id=%s", current.run_id, current.profile_id)
            self.store.insert_monitoring_run_snapshot(
                run_id=current.run_id,
                profile_id=current.profile_id,
                status=MonitoringRunStatus.FAILED,
                trigger_type=current.trigger_type,
                workflow_id=current.workflow_id,
                workflow_run_id=current.workflow_run_id,
                task_id=current.task_id,
                requested_at=current.requested_at or started_at,
                started_at=started_at,
                finished_at=finished_at,
                scheduled_for=current.scheduled_for,
                inputs_json=current.workflow_inputs,
                output_json={},
                error_message=str(exc),
                metadata=current.metadata,
                updated_at=finished_at,
            )

    def run_due_schedules(self, *, now: datetime | None = None, limit: int = 50) -> MonitoringSchedulerTickResponse:
        checked_at = now or _utcnow()
        launched = 0
        skipped = 0
        items: list[MonitoringScheduleTickItem] = []
        for row in self.store.list_due_monitoring_schedules(now=checked_at, limit=limit):
            profile_id = str(row.get("profile_id") or "")
            profile = self.get_profile(profile_id)
            due_at = row.get("next_run_at") or checked_at
            next_run_at = compute_next_run_at(
                cron=str(row.get("cron") or ""),
                timezone_name=str(row.get("timezone") or "UTC"),
                after=checked_at,
            )
            if profile is None or profile.is_archived:
                skipped += 1
                self.store.insert_monitoring_schedule_snapshot(
                    profile_id=profile_id,
                    cron=str(row.get("cron") or ""),
                    timezone=str(row.get("timezone") or "UTC"),
                    is_enabled=_as_bool(row.get("is_enabled")),
                    max_active_runs=max(1, int(row.get("max_active_runs") or 1)),
                    next_run_at=next_run_at,
                    last_run_at=row.get("last_run_at"),
                    created_at=row.get("created_at") or checked_at,
                    updated_at=checked_at,
                )
                items.append(
                    MonitoringScheduleTickItem(
                        profile_id=profile_id,
                        action="skipped",
                        detail="profile_missing_or_archived",
                        next_run_at=next_run_at,
                    )
                )
                continue
            active_runs = self.store.count_monitoring_active_runs(profile_id)
            max_active_runs = max(1, int(row.get("max_active_runs") or 1))
            if active_runs >= max_active_runs:
                skipped += 1
                self.store.insert_monitoring_schedule_snapshot(
                    profile_id=profile_id,
                    cron=str(row.get("cron") or ""),
                    timezone=str(row.get("timezone") or "UTC"),
                    is_enabled=_as_bool(row.get("is_enabled")),
                    max_active_runs=max_active_runs,
                    next_run_at=next_run_at,
                    last_run_at=row.get("last_run_at"),
                    created_at=row.get("created_at") or checked_at,
                    updated_at=checked_at,
                )
                items.append(
                    MonitoringScheduleTickItem(
                        profile_id=profile_id,
                        action="skipped",
                        detail="max_active_runs_reached",
                        next_run_at=next_run_at,
                    )
                )
                continue
            run = self.create_run(
                profile_id,
                trigger_type=MonitoringRunTrigger.SCHEDULED,
                scheduled_for=due_at,
                auto_start=True,
            )
            self.store.insert_monitoring_schedule_snapshot(
                profile_id=profile_id,
                cron=str(row.get("cron") or ""),
                timezone=str(row.get("timezone") or "UTC"),
                is_enabled=_as_bool(row.get("is_enabled")),
                max_active_runs=max_active_runs,
                next_run_at=next_run_at,
                last_run_at=due_at,
                created_at=row.get("created_at") or checked_at,
                updated_at=checked_at,
            )
            launched += 1
            items.append(
                MonitoringScheduleTickItem(
                    profile_id=profile_id,
                    action="launched",
                    run_id=run.run_id,
                    next_run_at=next_run_at,
                )
            )
        return MonitoringSchedulerTickResponse(
            checked_at=checked_at,
            launched=launched,
            skipped=skipped,
            items=items,
        )


class MonitoringSchedulerLoop:
    def __init__(self, service: MonitoringService, poll_interval_seconds: float) -> None:
        self.service = service
        self.poll_interval_seconds = poll_interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_forever, daemon=True, name="monitoring-scheduler")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=max(1.0, self.poll_interval_seconds + 1.0))

    def _run_forever(self) -> None:
        while not self._stop_event.is_set():
            try:
                result = self.service.run_due_schedules()
                if result.launched or result.skipped:
                    logger.info(
                        "monitoring_scheduler_tick | launched=%s skipped=%s",
                        result.launched,
                        result.skipped,
                    )
            except Exception:
                logger.exception("monitoring_scheduler_tick_failed")
            self._stop_event.wait(self.poll_interval_seconds)


def create_monitoring_service(*, store: Any, settings: Settings | None = None, workflow_client: Any | None = None) -> MonitoringService:
    current_settings = settings or get_settings()
    return MonitoringService(
        store=store,
        workflow_client=workflow_client or DifyWorkflowClient(current_settings),
        settings=current_settings,
    )
