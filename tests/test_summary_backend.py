from __future__ import annotations

import json
import unittest
from collections import Counter, deque
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from summary_backend.config import get_settings
from summary_backend.ids import sha256_text
from summary_backend.input_models import InputSegment
from summary_backend.pipeline import PipelineService
from summary_backend.ports import TaskQueue
from summary_backend.schemas import ArtifactType, JobStatus, NodeStatus, NodeType, Stage, SummaryResult
from summary_backend.snapshots import build_job_snapshot


class InMemorySummaryStore:
    def __init__(self) -> None:
        self.job_events: list[dict[str, Any]] = []
        self.node_events: list[dict[str, Any]] = []
        self.artifacts: list[dict[str, Any]] = []
        self.input_segments: list[dict[str, Any]] = []
        self.llm_calls: list[dict[str, Any]] = []
        self._seq = 0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _payload(payload: dict[str, Any] | None) -> str:
        return json.dumps(payload or {}, ensure_ascii=False, sort_keys=True)

    def insert_job_event(
        self,
        *,
        job_id: str,
        event_type: str,
        job_status: JobStatus | str,
        actor: str = "",
        message: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        seq = self._next_seq()
        self.job_events.append(
            {
                "_seq": seq,
                "event_id": f"job-event-{seq}",
                "job_id": job_id,
                "event_time": self._now(),
                "event_type": event_type,
                "job_status": str(job_status),
                "status": str(job_status),
                "actor": actor,
                "message": message,
                "payload": self._payload(payload),
            }
        )

    def insert_node_event(
        self,
        *,
        job_id: str,
        node_id: str,
        event_type: str,
        node_status: NodeStatus | str,
        node_type: NodeType | str,
        level: int,
        node_index: int,
        attempt: int = 0,
        actor: str = "",
        message: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        seq = self._next_seq()
        self.node_events.append(
            {
                "_seq": seq,
                "event_id": f"node-event-{seq}",
                "job_id": job_id,
                "node_id": node_id,
                "event_time": self._now(),
                "event_type": event_type,
                "node_status": str(node_status),
                "status": str(node_status),
                "node_type": str(node_type),
                "level": level,
                "node_index": node_index,
                "attempt": attempt,
                "actor": actor,
                "message": message,
                "payload": self._payload(payload),
            }
        )

    def insert_artifact(
        self,
        *,
        job_id: str,
        node_id: str,
        artifact_type: ArtifactType | str,
        stage: Stage | str,
        level: int,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        seq = self._next_seq()
        content_hash = sha256_text(content)
        self.artifacts.append(
            {
                "_seq": seq,
                "artifact_id": f"artifact-{seq}",
                "job_id": job_id,
                "node_id": node_id,
                "artifact_type": str(artifact_type),
                "stage": str(stage),
                "level": level,
                "content_hash": content_hash,
                "content": content,
                "metadata": self._payload(metadata),
                "created_at": self._now(),
            }
        )
        return content_hash

    def insert_input_segments(self, *, job_id: str, segments: list[InputSegment]) -> None:
        for segment in segments:
            seq = self._next_seq()
            self.input_segments.append(
                {
                    "_seq": seq,
                    "job_id": job_id,
                    "segment_index": segment.segment_index,
                    "source_type": segment.source_type,
                    "source_format": segment.source_format,
                    "content_hash": segment.content_hash,
                    "content": segment.content,
                    "rows_count": segment.rows_count,
                    "chars": segment.chars,
                    "metadata": self._payload(dict(segment.metadata)),
                    "created_at": self._now(),
                }
            )

    def insert_llm_call(self, **kwargs: Any) -> None:
        self.llm_calls.append(dict(kwargs))

    def get_job_current(self, job_id: str) -> dict[str, Any] | None:
        events = [event for event in self.job_events if event["job_id"] == job_id]
        if not events:
            return None
        latest = max(events, key=lambda event: event["_seq"])
        return {
            "job_id": job_id,
            "job_status": latest["job_status"],
            "last_event_type": latest["event_type"],
            "updated_at": latest["event_time"],
            "events_count": len(events),
        }

    def list_job_events(self, job_id: str, limit: int = 500) -> list[dict[str, Any]]:
        return [event for event in self.job_events if event["job_id"] == job_id][:limit]

    def list_node_events(self, job_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        return [event for event in self.node_events if event["job_id"] == job_id][:limit]

    def list_nodes_current(self, job_id: str) -> list[dict[str, Any]]:
        latest_by_node: dict[str, dict[str, Any]] = {}
        for event in self.node_events:
            if event["job_id"] != job_id:
                continue
            current = latest_by_node.get(event["node_id"])
            if current is None or event["_seq"] > current["_seq"]:
                latest_by_node[event["node_id"]] = event
        return [
            {
                "job_id": job_id,
                "node_id": event["node_id"],
                "node_type": event["node_type"],
                "level": event["level"],
                "node_index": event["node_index"],
                "node_status": event["node_status"],
                "last_event_type": event["event_type"],
                "updated_at": event["event_time"],
                "events_count": sum(
                    1
                    for candidate in self.node_events
                    if candidate["job_id"] == job_id and candidate["node_id"] == event["node_id"]
                ),
            }
            for event in sorted(latest_by_node.values(), key=lambda item: (item["level"], item["node_type"], item["node_index"]))
        ]

    def get_node_payload(self, job_id: str, node_id: str) -> dict[str, Any]:
        events = [
            event
            for event in self.node_events
            if event["job_id"] == job_id and event["node_id"] == node_id
        ]
        if not events:
            return {}
        latest = max(events, key=lambda event: event["_seq"])
        return json.loads(latest["payload"] or "{}")

    def latest_artifact(
        self,
        *,
        job_id: str,
        artifact_type: ArtifactType | str | None = None,
        node_id: str | None = None,
        stage: Stage | str | None = None,
        level: int | None = None,
    ) -> dict[str, Any] | None:
        artifacts = self.list_artifacts(
            job_id=job_id,
            include_content=True,
            artifact_type=artifact_type,
            stage=stage,
            level=level,
        )
        if node_id is not None:
            artifacts = [artifact for artifact in artifacts if artifact["node_id"] == node_id]
        if not artifacts:
            return None
        return max(artifacts, key=lambda artifact: artifact["_seq"])

    def list_artifacts(
        self,
        *,
        job_id: str,
        include_content: bool = False,
        artifact_type: ArtifactType | str | None = None,
        stage: Stage | str | None = None,
        level: int | None = None,
    ) -> list[dict[str, Any]]:
        artifacts = [artifact for artifact in self.artifacts if artifact["job_id"] == job_id]
        if artifact_type is not None:
            artifacts = [artifact for artifact in artifacts if artifact["artifact_type"] == str(artifact_type)]
        if stage is not None:
            artifacts = [artifact for artifact in artifacts if artifact["stage"] == str(stage)]
        if level is not None:
            artifacts = [artifact for artifact in artifacts if artifact["level"] == level]
        result: list[dict[str, Any]] = []
        for artifact in sorted(artifacts, key=lambda item: (item["level"], item["stage"], item["node_id"], item["_seq"])):
            row = dict(artifact)
            if not include_content:
                row["content"] = None
            result.append(row)
        return result

    def list_input_segments(self, *, job_id: str, include_content: bool = False) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        latest_by_index: dict[int, dict[str, Any]] = {}
        for segment in self.input_segments:
            if segment["job_id"] != job_id:
                continue
            current = latest_by_index.get(int(segment["segment_index"]))
            if current is None or segment["_seq"] > current["_seq"]:
                latest_by_index[int(segment["segment_index"])] = segment
        for segment in sorted(latest_by_index.values(), key=lambda item: item["segment_index"]):
            row = dict(segment)
            if not include_content:
                row["content"] = None
            result.append(row)
        return result

    def count_input_segments(self, job_id: str) -> int:
        return len({segment["segment_index"] for segment in self.input_segments if segment["job_id"] == job_id})

    def list_staged_uploads(self, limit: int = 200) -> list[dict[str, Any]]:
        uploads: list[dict[str, Any]] = []
        for event in reversed(self.job_events):
            if event["event_type"] != "FILE_STAGED":
                continue
            payload = json.loads(event["payload"] or "{}")
            if "reused_upload" in payload:
                continue
            source = payload.get("source") or {}
            staging = payload.get("staging") or {}
            current = self.get_job_current(event["job_id"]) or {}
            uploads.append(
                {
                    "upload_id": event["job_id"],
                    "source_job_id": event["job_id"],
                    "filename": source.get("filename", ""),
                    "source_format": source.get("format", ""),
                    "content_type": source.get("content_type", ""),
                    "raw_line_column": source.get("raw_line_column", ""),
                    "size_bytes": int(staging.get("size_bytes") or 0),
                    "available": True,
                    "job_status": current.get("job_status", ""),
                    "staged_at": event["event_time"],
                }
            )
            if len(uploads) >= limit:
                break
        return uploads

    def list_jobs_for_recovery(self) -> list[dict[str, Any]]:
        jobs = []
        for job_id in sorted({event["job_id"] for event in self.job_events}):
            current = self.get_job_current(job_id)
            if current and current["job_status"] in {
                "CREATED",
                "INGESTING",
                "INPUT_READY",
                "RUNNING",
                "RESUMED",
                "WAITING_RETRY",
                "WAITING_PROVIDER",
            }:
                jobs.append(current)
        return jobs


class FakeChunker:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks

    def build_chunks(self, text: str, target_estimated_tokens: int) -> list[str]:
        return list(self.chunks)


class FakeInputSegmenter:
    def __init__(self, segments: list[str]) -> None:
        self.segments = segments

    def build_segments(self, records, *, source_type: str, source_format: str, target_estimated_tokens: int):
        _ = list(records)
        for index, content in enumerate(self.segments):
            yield InputSegment(
                segment_index=index,
                source_type=source_type,
                source_format=source_format,
                content=content,
                rows_count=content.count("\n") + 1,
                metadata={"source": "fake"},
            )


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def call_summary(
        self,
        *,
        job_id: str,
        node_id: str,
        stage: str,
        system: str,
        user: str,
    ) -> SummaryResult:
        self.calls.append(
            {
                "job_id": job_id,
                "node_id": node_id,
                "stage": stage,
                "system": system,
                "user": user,
            }
        )
        return SummaryResult(
            ok=True,
            summary=f"{stage}:{node_id}:{len(user)}",
            key_points=[user.splitlines()[0] if user.splitlines() else ""],
            warnings=[],
            source_count=max(1, user.count("Summary ")),
        )


class FailingLLM:
    def __init__(self, message: str = "429 too many failed authentication attempts") -> None:
        self.message = message
        self.calls: list[dict[str, str]] = []

    def call_summary(
        self,
        *,
        job_id: str,
        node_id: str,
        stage: str,
        system: str,
        user: str,
    ) -> SummaryResult:
        self.calls.append(
            {
                "job_id": job_id,
                "node_id": node_id,
                "stage": stage,
                "system": system,
                "user": user,
            }
        )
        raise RuntimeError(self.message)


class ManualQueue(TaskQueue):
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


class SummaryBackendPipelineTests(unittest.TestCase):
    def make_service(
        self,
        *,
        chunks: list[str] | None = None,
        reduce_group_size: int = 2,
        queue: ManualQueue | None = None,
    ) -> tuple[PipelineService, InMemorySummaryStore, FakeLLM, ManualQueue]:
        store = InMemorySummaryStore()
        llm = FakeLLM()
        manual_queue = queue or ManualQueue()
        settings = replace(
            get_settings(),
            reduce_group_size=reduce_group_size,
            max_enqueue_nodes_per_advance=100,
            chunk_target_estimated_tokens=100,
        )
        service = PipelineService(
            store=store,
            queue=manual_queue,
            llm=llm,
            chunker=FakeChunker(chunks or ["chunk-1", "chunk-2", "chunk-3"]),
            input_segmenter=FakeInputSegmenter(chunks or ["chunk-1", "chunk-2", "chunk-3"]),
            settings=settings,
        )
        return service, store, llm, manual_queue

    @staticmethod
    def drain(queue: ManualQueue, service: PipelineService, max_steps: int = 200) -> None:
        steps = 0
        while queue.items:
            steps += 1
            if steps > max_steps:
                raise AssertionError("manual queue did not drain")
            kind, job_id, node_id = queue.items.popleft()
            if kind == "ingest":
                raise AssertionError("ingest task cannot be drained by pipeline-only test helper")
            elif kind == "advance":
                service.advance_job(job_id)
            elif kind == "map":
                assert node_id is not None
                service.map_node(job_id, node_id)
            elif kind == "reduce":
                assert node_id is not None
                service.reduce_node(job_id, node_id)
            elif kind == "finalize":
                service.finalize_job(job_id)
            else:
                raise AssertionError(kind)

    def test_full_pipeline_creates_map_reduce_levels_and_final_artifact(self) -> None:
        service, store, llm, queue = self.make_service(
            chunks=["chunk-1", "chunk-2", "chunk-3", "chunk-4", "chunk-5"],
            reduce_group_size=2,
        )
        job_id = service.create_job(input_text="input", title="full", metadata={"case": "full"})

        queue.advance_job(job_id)
        self.drain(queue, service)

        self.assertEqual(store.get_job_current(job_id)["job_status"], JobStatus.DONE)
        artifact_counts = Counter(artifact["artifact_type"] for artifact in store.list_artifacts(job_id=job_id, include_content=False))
        self.assertEqual(artifact_counts[ArtifactType.INPUT], 1)
        self.assertEqual(artifact_counts[ArtifactType.CHUNK], 5)
        self.assertEqual(artifact_counts[ArtifactType.MAP_SUMMARY], 5)
        self.assertEqual(artifact_counts[ArtifactType.REDUCE_SUMMARY], 6)
        self.assertEqual(artifact_counts[ArtifactType.FINAL_SUMMARY], 1)
        reduce_levels = {
            artifact["level"]
            for artifact in store.list_artifacts(
                job_id=job_id,
                include_content=False,
                artifact_type=ArtifactType.REDUCE_SUMMARY,
            )
        }
        self.assertEqual(reduce_levels, {1, 2, 3})
        self.assertGreaterEqual(len(llm.calls), 12)

    def test_duplicate_node_delivery_is_idempotent(self) -> None:
        service, store, llm, queue = self.make_service(chunks=["chunk-1", "chunk-2"])
        job_id = service.create_job(input_text="input", title="idempotent", metadata={})

        queue.advance_job(job_id)
        service.advance_job(job_id)
        first_map = next(item for item in list(queue.items) if item[0] == "map")
        queue.items.remove(first_map)
        _, _, node_id = first_map
        assert node_id is not None

        service.map_node(job_id, node_id)
        before = len(
            [
                artifact
                for artifact in store.list_artifacts(job_id=job_id, include_content=False)
                if artifact["node_id"] == node_id and artifact["artifact_type"] == ArtifactType.MAP_SUMMARY
            ]
        )
        service.map_node(job_id, node_id)
        after = len(
            [
                artifact
                for artifact in store.list_artifacts(job_id=job_id, include_content=False)
                if artifact["node_id"] == node_id and artifact["artifact_type"] == ArtifactType.MAP_SUMMARY
            ]
        )

        self.assertEqual(before, 1)
        self.assertEqual(after, 1)
        self.assertEqual(
            store.latest_artifact(job_id=job_id, node_id=node_id, artifact_type=ArtifactType.MAP_SUMMARY)["artifact_type"],
            ArtifactType.MAP_SUMMARY,
        )
        self.assertEqual(len([call for call in llm.calls if call["node_id"] == node_id]), 1)

    def test_pause_blocks_work_and_resume_continues(self) -> None:
        service, store, _, queue = self.make_service(chunks=["chunk-1", "chunk-2"])
        job_id = service.create_job(input_text="input", title="pause", metadata={})

        service.pause_job(job_id)
        service.advance_job(job_id)

        self.assertEqual(store.get_job_current(job_id)["job_status"], JobStatus.PAUSED)
        self.assertEqual(len(store.list_nodes_current(job_id)), 0)

        service.resume_job(job_id)
        self.drain(queue, service)

        self.assertEqual(store.get_job_current(job_id)["job_status"], JobStatus.DONE)
        self.assertIsNotNone(store.latest_artifact(job_id=job_id, artifact_type=ArtifactType.FINAL_SUMMARY))

    def test_cancel_blocks_work(self) -> None:
        service, store, _, _ = self.make_service(chunks=["chunk-1", "chunk-2"])
        job_id = service.create_job(input_text="input", title="cancel", metadata={})

        service.cancel_job(job_id)
        service.advance_job(job_id)

        self.assertEqual(store.get_job_current(job_id)["job_status"], JobStatus.CANCELLED)
        self.assertEqual(len(store.list_nodes_current(job_id)), 0)
        artifact_counts = Counter(artifact["artifact_type"] for artifact in store.list_artifacts(job_id=job_id, include_content=False))
        self.assertEqual(artifact_counts, Counter({ArtifactType.INPUT: 1}))

    def test_llm_failure_marks_node_and_job_failed(self) -> None:
        service, store, _, queue = self.make_service(chunks=["chunk-1"])
        service.llm = FailingLLM()
        job_id = service.create_job(input_text="input", title="llm-fail", metadata={})

        queue.advance_job(job_id)
        self.drain(queue, service)

        self.assertEqual(store.get_job_current(job_id)["job_status"], JobStatus.FAILED)
        nodes = store.list_nodes_current(job_id)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["node_status"], NodeStatus.FAILED_FINAL)
        self.assertEqual(nodes[0]["last_event_type"], "NODE_FAILED")

        failed_event = store.list_node_events(job_id)[-1]
        payload = json.loads(failed_event["payload"])
        self.assertEqual(payload["error_class"], "rate_limit")
        self.assertIn("429", failed_event["message"])

    def test_final_prompt_uses_requested_report_format(self) -> None:
        service, store, llm, queue = self.make_service(chunks=["chunk-1"])
        job_id = service.create_job(
            input_text="input",
            title="format",
            metadata={
                "report_format": "technical_rca",
                "report_format_instruction": "Highlight remediation owners.",
            },
        )

        queue.advance_job(job_id)
        self.drain(queue, service)

        final_call = next(call for call in llm.calls if call["stage"] == Stage.FINAL)
        self.assertIn("Desired report format", final_call["user"])
        self.assertIn("technical RCA", final_call["user"])
        self.assertIn("Highlight remediation owners.", final_call["user"])
        self.assertIn("Keep the SummaryResult JSON schema unchanged", final_call["user"])

        final_artifact = store.latest_artifact(job_id=job_id, artifact_type=ArtifactType.FINAL_SUMMARY)
        self.assertIsNotNone(final_artifact)
        metadata = json.loads(final_artifact["metadata"])
        self.assertEqual(metadata["report_format"], "technical_rca")
        self.assertIn("Highlight remediation owners.", metadata["report_format_instruction"])

    def test_create_job_persists_text_as_input_segments_for_map_nodes(self) -> None:
        service, store, _, queue = self.make_service(chunks=["row 1", "row 2\nrow 3"])
        job_id = service.create_job(input_text="raw text", title="text", metadata={})

        input_artifact = store.latest_artifact(job_id=job_id, artifact_type=ArtifactType.INPUT)
        self.assertIsNotNone(input_artifact)
        self.assertIn("text input manifest", input_artifact["content"])
        self.assertEqual(store.count_input_segments(job_id), 2)

        queue.advance_job(job_id)
        service.advance_job(job_id)

        chunks = sorted(
            store.list_artifacts(job_id=job_id, include_content=True, artifact_type=ArtifactType.CHUNK),
            key=lambda chunk: json.loads(chunk["metadata"])["chunk_index"],
        )
        self.assertEqual([chunk["content"] for chunk in chunks], ["row 1", "row 2\nrow 3"])
        self.assertEqual([json.loads(chunk["metadata"])["rows_count"] for chunk in chunks], [1, 2])
        self.assertEqual(len([item for item in queue.items if item[0] == "map"]), 2)

    def test_rerun_job_clones_input_segments_into_new_job(self) -> None:
        service, store, _, queue = self.make_service(chunks=["row 1", "row 2\nrow 3"])
        source_job_id = service.create_job(input_text="raw text", title="source", metadata={"case": "rerun"})

        new_job_id, queued = service.rerun_job(source_job_id)

        self.assertTrue(queued)
        self.assertNotEqual(source_job_id, new_job_id)
        self.assertIn(("advance", new_job_id, None), queue.items)
        self.assertEqual(store.count_input_segments(new_job_id), 2)

        source_segments = store.list_input_segments(job_id=source_job_id, include_content=True)
        rerun_segments = store.list_input_segments(job_id=new_job_id, include_content=True)
        self.assertEqual([segment["content"] for segment in rerun_segments], [segment["content"] for segment in source_segments])
        self.assertTrue(
            all(
                json.loads(segment["metadata"])["rerun_source_job_id"] == source_job_id
                for segment in rerun_segments
            )
        )

        input_artifact = store.latest_artifact(job_id=new_job_id, artifact_type=ArtifactType.INPUT)
        self.assertIsNotNone(input_artifact)
        self.assertIn("rerun input manifest", input_artifact["content"])

        self.drain(queue, service)
        self.assertEqual(store.get_job_current(new_job_id)["job_status"], JobStatus.DONE)

    def test_recovery_requeues_runnable_jobs_only(self) -> None:
        service, store, _, queue = self.make_service(chunks=["chunk-1"])
        runnable_job_id = service.create_job(input_text="input", title="runnable", metadata={})
        cancelled_job_id = service.create_job(input_text="input", title="cancelled", metadata={})
        service.cancel_job(cancelled_job_id)
        service.advance_job(cancelled_job_id)

        recovered = service.recover_jobs()

        self.assertIn(runnable_job_id, recovered)
        self.assertNotIn(cancelled_job_id, recovered)
        self.assertIn(("advance", runnable_job_id, None), queue.items)
        self.assertEqual(store.get_job_current(cancelled_job_id)["job_status"], JobStatus.CANCELLED)

    def test_recovery_requeues_ingesting_jobs_to_ingest_actor(self) -> None:
        service, store, _, queue = self.make_service(chunks=["chunk-1"])
        store.insert_job_event(
            job_id="job_ingesting",
            event_type="JOB_CREATED",
            job_status=JobStatus.INGESTING,
            actor="api",
            payload={"source": {"kind": "upload"}},
        )

        recovered = service.recover_jobs()

        self.assertIn("job_ingesting", recovered)
        self.assertIn(("ingest", "job_ingesting", None), queue.items)

    def test_snapshot_contains_ui_read_model(self) -> None:
        service, store, _, queue = self.make_service(chunks=["chunk-1", "chunk-2"])
        job_id = service.create_job(input_text="input", title="snapshot", metadata={})

        queue.advance_job(job_id)
        self.drain(queue, service)

        snapshot = build_job_snapshot(service, job_id)

        self.assertEqual(snapshot["job"]["job_status"], JobStatus.DONE)
        self.assertIn("nodes", snapshot)
        self.assertIn("artifacts", snapshot)
        self.assertIn("job_events", snapshot)
        self.assertIn("node_events", snapshot)
        self.assertIsNotNone(snapshot["final_artifact"])
        self.assertEqual(snapshot["artifact_counts"][ArtifactType.FINAL_SUMMARY], 1)


if __name__ == "__main__":
    unittest.main()
