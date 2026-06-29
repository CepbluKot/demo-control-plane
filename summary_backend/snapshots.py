"""Read-model helpers for UI and WebSocket snapshots."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from .pipeline import PipelineService
from .schemas import ArtifactType


def build_job_snapshot(service: PipelineService, job_id: str) -> dict[str, Any]:
    status = service.get_status(job_id)
    final_artifact = service.store.latest_artifact(
        job_id=job_id,
        artifact_type=ArtifactType.FINAL_SUMMARY,
    )
    input_segments = service.store.list_input_segments(job_id=job_id, include_content=False)
    return {
        **status,
        "nodes": service.store.list_nodes_current(job_id),
        "artifacts": service.store.list_artifacts(job_id=job_id, include_content=False),
        "input_stats": build_input_stats(input_segments),
        "final_artifact": final_artifact,
        "job_events": service.store.list_job_events(job_id, limit=200),
        "node_events": service.store.list_node_events(job_id, limit=500),
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


def build_input_stats(input_segments: list[dict[str, Any]]) -> dict[str, int]:
    chars = sum(_safe_int(segment.get("chars")) for segment in input_segments)
    rows = sum(_safe_int(segment.get("rows_count")) for segment in input_segments)
    return {
        "segments_count": len(input_segments),
        "rows_count": rows,
        "chars": chars,
        "estimated_tokens": math.ceil(chars / 2.2) if chars > 0 else 0,
    }


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
