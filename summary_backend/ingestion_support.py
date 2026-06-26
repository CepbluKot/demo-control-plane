"""Shared helpers for summary input ingestion services."""

from __future__ import annotations

import io
import json
from collections.abc import Iterable
from typing import Any, BinaryIO

from .input_models import InputSegment
from .ports import SummaryStore


def insert_segments(
    *,
    store: SummaryStore,
    job_id: str,
    segments: Iterable[InputSegment],
    batch_size: int,
) -> tuple[int, int]:
    batch: list[InputSegment] = []
    segments_count = 0
    rows_count = 0
    for segment in segments:
        batch.append(segment)
        segments_count += 1
        rows_count += segment.rows_count
        if len(batch) >= batch_size:
            store.insert_input_segments(job_id=job_id, segments=batch)
            batch = []
    if batch:
        store.insert_input_segments(job_id=job_id, segments=batch)
    return segments_count, rows_count


def with_first(first: InputSegment, rest: Iterable[InputSegment]) -> Iterable[InputSegment]:
    yield first
    yield from rest


def open_text_stream(file: BinaryIO) -> io.TextIOWrapper:
    try:
        file.seek(0)
    except (OSError, AttributeError):
        pass
    return io.TextIOWrapper(file, encoding="utf-8-sig", errors="replace", newline="")


def latest_event_payload(store: SummaryStore, job_id: str, event_type: str) -> dict[str, Any]:
    for event in reversed(store.list_job_events(job_id, limit=5000)):
        if event.get("event_type") != event_type:
            continue
        try:
            payload = json.loads(str(event.get("payload") or "{}"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def safe_filename(filename: str) -> str:
    safe = "".join(char if char.isalnum() or char in {".", "-", "_"} else "_" for char in filename)
    safe = safe.strip("._")
    return safe or "upload"
