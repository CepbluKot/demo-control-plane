"""Read-model helpers for UI and WebSocket snapshots."""

from __future__ import annotations

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
    return {
        **status,
        "nodes": service.store.list_nodes_current(job_id),
        "artifacts": service.store.list_artifacts(job_id=job_id, include_content=False),
        "final_artifact": final_artifact,
        "job_events": service.store.list_job_events(job_id, limit=200),
        "node_events": service.store.list_node_events(job_id, limit=500),
        "server_time": datetime.now(timezone.utc).isoformat(),
    }
