#!/usr/bin/env python3
"""Run the summary backend pipeline synchronously against local ClickHouse."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from summary_backend.config import get_settings
from summary_backend.factory import create_pipeline_service
from summary_backend.logging_setup import configure_logging
from summary_backend.queue import TaskQueue


class InlineQueue(TaskQueue):
    def __init__(self) -> None:
        self.items: deque[tuple[str, str, str | None]] = deque()

    def ingest_upload(self, job_id: str) -> None:
        self.items.append(("ingest", job_id, None))

    def advance_job(self, job_id: str) -> None:
        self.items.append(("advance", job_id, None))

    def map_node(self, job_id: str, node_id: str) -> None:
        self.items.append(("map", job_id, node_id))

    def reduce_node(self, job_id: str, node_id: str) -> None:
        self.items.append(("reduce", job_id, node_id))

    def finalize_job(self, job_id: str) -> None:
        self.items.append(("finalize", job_id, None))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="")
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings)

    text = args.text or "\n".join(
        f"{i:04d} service=payments level=INFO message='synthetic event {i}'"
        for i in range(args.repeat)
    )

    queue = InlineQueue()
    service = create_pipeline_service(queue=queue, settings=settings)
    job_id = service.create_job(input_text=text, title="smoke", metadata={"source": "summary_backend_smoke"})
    queue.advance_job(job_id)

    while queue.items:
        kind, item_job_id, node_id = queue.items.popleft()
        if kind == "ingest":
            raise RuntimeError("summary_backend_smoke does not create staged upload jobs")
        if kind == "advance":
            service.advance_job(item_job_id)
        elif kind == "map":
            assert node_id is not None
            service.map_node(item_job_id, node_id)
        elif kind == "reduce":
            assert node_id is not None
            service.reduce_node(item_job_id, node_id)
        elif kind == "finalize":
            service.finalize_job(item_job_id)
        else:
            raise RuntimeError(kind)

    status = service.get_status(job_id)
    artifacts = service.store.list_artifacts(job_id=job_id, include_content=True)
    final = [item for item in artifacts if item["artifact_type"] == "FINAL_SUMMARY"]

    print(json.dumps({"job_id": job_id, "status": status, "final": final[-1] if final else None}, default=str, indent=2))


if __name__ == "__main__":
    main()
