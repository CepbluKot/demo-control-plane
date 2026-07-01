"""Helpers for deriving summary job lifecycle timestamps from persisted events."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

JOB_START_EVENT_TYPES = {
    "INPUT_INGEST_STARTED",
    "JOB_RUNNING",
    "JOB_RESUMED",
}

JOB_FINISH_EVENT_TYPES = {
    "JOB_DONE",
    "JOB_FAILED",
    "JOB_CANCELLED",
}

TERMINAL_JOB_STATUSES = {
    "DONE",
    "FAILED",
    "CANCELLED",
}


def attach_job_lifecycle_timestamps(job: dict[str, Any], events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return a copy of the job row enriched with created/started/finished timestamps."""

    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    for event in _sort_events(events):
        event_time = event.get("event_time")
        if not isinstance(event_time, datetime):
            continue
        event_type = str(event.get("event_type") or "").strip().upper()
        if created_at is None:
            created_at = event_time
        if started_at is None and event_type in JOB_START_EVENT_TYPES:
            started_at = event_time
        if event_type in JOB_FINISH_EVENT_TYPES:
            finished_at = event_time

    job_status = str(job.get("job_status") or "").strip().upper()
    updated_at = job.get("updated_at")
    if finished_at is None and job_status in TERMINAL_JOB_STATUSES and isinstance(updated_at, datetime):
        finished_at = updated_at

    enriched = dict(job)
    enriched["created_at"] = created_at
    enriched["started_at"] = started_at
    enriched["finished_at"] = finished_at
    return enriched


def _sort_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(events, key=_event_sort_key)


def _event_sort_key(event: dict[str, Any]) -> tuple[float, str, str]:
    event_time = event.get("event_time")
    event_time_seconds = event_time.timestamp() if isinstance(event_time, datetime) else float("inf")
    return (
        event_time_seconds,
        str(event.get("event_id") or ""),
        str(event.get("event_type") or ""),
    )
