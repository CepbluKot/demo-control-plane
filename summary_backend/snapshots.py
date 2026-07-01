"""Read-model helpers for UI and WebSocket snapshots."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from .pipeline import PipelineService
from .schemas import ArtifactType


def build_job_snapshot(service: PipelineService, job_id: str) -> dict[str, Any]:
    status = service.get_status(job_id)
    nodes = service.store.list_nodes_current(job_id)
    final_artifact = service.store.latest_artifact(
        job_id=job_id,
        artifact_type=ArtifactType.FINAL_SUMMARY,
    )
    input_segments = service.store.list_input_segments(job_id=job_id, include_content=False)
    node_links = build_node_links(service, job_id, nodes)
    return {
        **status,
        "nodes": nodes,
        "node_links": node_links,
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


def build_node_links(
    service: PipelineService,
    job_id: str,
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    node_by_id = {
        str(node.get("node_id") or ""): node
        for node in nodes
        if str(node.get("node_id") or "").strip()
    }
    links: list[dict[str, Any]] = []

    for node in sorted(nodes, key=_node_link_sort_key):
        node_id = str(node.get("node_id") or "")
        if not node_id:
            continue
        node_type = str(node.get("node_type") or "")
        if node_type == "MAP":
            links.append(
                {
                    "from_node_id": "summary-source",
                    "from_node_type": "SOURCE",
                    "to_node_id": node_id,
                    "to_node_type": node_type,
                    "input_index": 0,
                }
            )
            continue

        payload = service.store.get_node_payload(job_id, node_id)
        input_node_ids = _read_string_list(payload.get("input_node_ids"))
        for input_index, input_node_id in enumerate(input_node_ids):
            source_node = node_by_id.get(input_node_id)
            links.append(
                {
                    "from_node_id": input_node_id,
                    "from_node_type": str(source_node.get("node_type") or "") if source_node else "",
                    "to_node_id": node_id,
                    "to_node_type": node_type,
                    "input_index": input_index,
                }
            )

    return links


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _read_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _node_link_sort_key(node: dict[str, Any]) -> tuple[int, int, int]:
    node_type = str(node.get("node_type") or "")
    node_type_order = {
        "MAP": 0,
        "REDUCE": 1,
        "FINAL": 2,
    }.get(node_type, 9)
    return (
        node_type_order,
        _safe_int(node.get("level")),
        _safe_int(node.get("node_index")),
    )
