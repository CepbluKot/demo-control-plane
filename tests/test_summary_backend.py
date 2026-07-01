from __future__ import annotations

import json
import os
import threading
import time
import unittest
from collections import Counter, deque
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

from summary_backend.api import (
    PromptDraftConceptSpec,
    PromptDraftJobManager,
    _generate_prompt_draft_with_llm,
    _normalize_prompt_draft_spec,
    _resolve_requested_prompt_draft_model,
    _run_llm_connectivity_checks,
    build_public_settings,
)
from summary_backend.config import get_settings, reset_settings_cache, resolve_llm_model_option
from summary_backend.errors import LlmPoolBusyError
from summary_backend.ids import sha256_text
from summary_backend.input_models import InputSegment
from summary_backend.llm_client import StructuredLLMClient
from summary_backend.pipeline import PipelineService
from summary_backend.ports import TaskQueue
from summary_backend.schemas import (
    ArtifactType,
    GenerateSummaryPromptDraftRequest,
    GenerateSummaryPromptDraftResponse,
    JobStatus,
    JsonObjectResult,
    NodeStatus,
    NodeType,
    PromptDraftJobStatus,
    Stage,
    SummaryPromptOverridesDraft,
    SummaryPromptStageDraft,
    SummaryResult,
)
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
        self.llm_calls.append({**dict(kwargs), "created_at": self._now()})

    def latest_llm_call(self, *, job_id: str, node_id: str) -> dict[str, Any] | None:
        calls = [call for call in self.llm_calls if call.get("job_id") == job_id and call.get("node_id") == node_id]
        return calls[-1] if calls else None

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
        def payload_priority(event: dict[str, Any]) -> tuple[int, int]:
            payload = event.get("payload") or "{}"
            if not payload or payload == "{}":
                return (0, int(event["_seq"]))
            if '"input_node_ids"' in payload:
                return (3, int(event["_seq"]))
            if '"chunk_hash"' in payload:
                return (2, int(event["_seq"]))
            return (1, int(event["_seq"]))

        latest = max(events, key=payload_priority)
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


class LargeSummaryLLM(FakeLLM):
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
        # Keep each intermediate summary larger than the reduce token target floor
        # so the test exercises the forced-progress regrouping path.
        return SummaryResult(
            ok=True,
            summary=("dense-summary " * 96).strip(),
            key_points=["oversized"],
            warnings=[],
            source_count=max(1, user.count("Summary ")),
        )


class ModelAwareLLM(FakeLLM):
    def call_summary(
        self,
        *,
        job_id: str,
        node_id: str,
        stage: str,
        system: str,
        user: str,
        model: str,
    ) -> SummaryResult:
        self.calls.append(
            {
                "job_id": job_id,
                "node_id": node_id,
                "stage": stage,
                "system": system,
                "user": user,
                "model": model,
            }
        )
        return SummaryResult(
            ok=True,
            summary=f"{stage}:{node_id}:{len(user)}",
            key_points=[user.splitlines()[0] if user.splitlines() else ""],
            warnings=[],
            source_count=max(1, user.count("Summary ")),
        )


class JobConcurrencyAwareLLM(FakeLLM):
    def call_summary(
        self,
        *,
        job_id: str,
        node_id: str,
        stage: str,
        system: str,
        user: str,
        job_max_concurrency: int | None = None,
        model: str | None = None,
    ) -> SummaryResult:
        self.calls.append(
            {
                "job_id": job_id,
                "node_id": node_id,
                "stage": stage,
                "system": system,
                "user": user,
                "job_max_concurrency": job_max_concurrency,
                "model": model,
            }
        )
        return SummaryResult(
            ok=True,
            summary=f"{stage}:{node_id}:{len(user)}",
            key_points=[user.splitlines()[0] if user.splitlines() else ""],
            warnings=[],
            source_count=max(1, user.count("Summary ")),
        )


class CustomJsonLLM(FakeLLM):
    def call_structured(
        self,
        *,
        job_id: str,
        node_id: str,
        stage: str,
        system: str,
        user: str,
        response_model: type,
        response_schema: dict | None = None,
    ):
        self.calls.append(
            {
                "job_id": job_id,
                "node_id": node_id,
                "stage": stage,
                "system": system,
                "user": user,
                "response_schema": response_schema,
            }
        )
        if response_model is JsonObjectResult:
            properties = (
                response_schema.get("properties")
                if isinstance(response_schema, dict)
                else None
            )
            if isinstance(properties, dict) and "chunk_summary" in properties:
                return response_model.model_validate({
                    "chunk_summary": f"{stage}:{node_id}",
                    "key_points": ["point"],
                    "evidence": [{"title": "metric spike", "details": "cpu=97%"}],
                    "warnings": [],
                })
            return response_model.model_validate({
                "headline": "custom output",
                "items": [{"title": "first", "severity": "info"}],
            })
        return response_model.model_validate({
            "ok": True,
            "summary": f"{stage}:{node_id}:{len(user)}",
            "key_points": [],
            "warnings": [],
            "source_count": 1,
        })


class PromptDraftLLMStub:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = deque(responses)
        self.calls: list[dict[str, Any]] = []

    def call_structured(
        self,
        *,
        job_id: str,
        node_id: str,
        stage: str,
        system: str,
        user: str,
        model: str | None = None,
        response_model: type,
        response_schema: dict | None = None,
    ):
        self.calls.append(
            {
                "job_id": job_id,
                "node_id": node_id,
                "stage": stage,
                "system": system,
                "user": user,
                "model": model,
                "response_model": response_model,
                "response_schema": response_schema,
            }
        )
        response = self.responses.popleft()
        if isinstance(response, response_model):
            return response
        return response_model.model_validate(response)


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


class BusyPoolLLM:
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
        raise LlmPoolBusyError("LLM pool acquire timeout after 1.0s; max_concurrency=2")


class BlockingThenFastLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self._call_count = 0

    def call_summary(
        self,
        *,
        job_id: str,
        node_id: str,
        stage: str,
        system: str,
        user: str,
    ) -> SummaryResult:
        self._call_count += 1
        self.calls.append(
            {
                "job_id": job_id,
                "node_id": node_id,
                "stage": stage,
                "system": system,
                "user": user,
            }
        )
        if self._call_count == 1:
            self.started.set()
            if not self.release.wait(timeout=5):
                raise AssertionError("blocking llm was not released in time")
        return SummaryResult(
            ok=True,
            summary=f"{stage}:{node_id}:{self._call_count}",
            key_points=[user.splitlines()[0] if user.splitlines() else ""],
            warnings=[],
            source_count=max(1, user.count("Summary ")),
        )


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


class SummaryServiceSettingsTests(unittest.TestCase):
    def test_public_settings_do_not_expose_secrets(self) -> None:
        settings = replace(
            get_settings(),
            openai_api_base="http://llm-gateway/v1",
            openai_api_key="secret-api-key",
            clickhouse_password="clickhouse-secret",
            source_clickhouse_password="source-secret",
            broker_url="redis://broker-user:broker-pass@redis:6379/0",
            worker_processes=2,
            worker_threads=8,
            llm_max_concurrency=2,
            llm_assistant_max_concurrency=4,
            llm_pool_acquire_timeout_seconds=30.0,
            llm_pool_poll_interval_seconds=0.2,
        )

        payload = build_public_settings(settings).model_dump()

        self.assertTrue(payload["llm"]["api_key_configured"])
        self.assertTrue(payload["clickhouse"]["password_configured"])
        self.assertTrue(payload["source_clickhouse"]["password_configured"])
        self.assertEqual(payload["runtime"]["broker_url"], "redis://***:***@redis:6379/0")
        self.assertEqual(payload["runtime"]["worker_processes"], 2)
        self.assertEqual(payload["runtime"]["worker_threads"], 8)
        self.assertEqual(payload["llm"]["max_concurrency"], 2)
        self.assertEqual(payload["llm"]["assistant_max_concurrency"], 4)
        self.assertEqual(payload["llm"]["pool_acquire_timeout_seconds"], 30.0)
        self.assertEqual(payload["llm"]["pool_poll_interval_seconds"], 0.2)
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("secret-api-key", serialized)
        self.assertNotIn("clickhouse-secret", serialized)
        self.assertNotIn("source-secret", serialized)
        self.assertNotIn("broker-pass", serialized)

    def test_settings_fallback_to_local_llm_gateway_defaults(self) -> None:
        fallback_env = {
            "SUMMARY_BACKEND_OPENAI_API_BASE": "",
            "OPENAI_API_BASE_DB": "",
            "SUMMARY_BACKEND_OPENAI_API_KEY": "",
            "OPENAI_API_KEY_DB": "",
            "SUMMARY_BACKEND_LLM_MODEL": "",
            "LLM_MODEL_ID": "",
            "SUMMARY_BACKEND_LLM_MODELS": "",
            "SUMMARY_BACKEND_DRY_RUN": "",
        }
        try:
            with patch.dict(os.environ, fallback_env, clear=False), patch(
                "summary_backend.config._load_llm_gateway_defaults",
                return_value={
                    "api_base": "http://llm-gateway.local/v1",
                    "api_key": "test-local-key",
                    "default_model": "llmgateway/free",
                    "available_models": ["llmgateway/free", "llmgateway/analysis"],
                },
            ):
                reset_settings_cache()
                settings = get_settings()

            self.assertEqual(settings.openai_api_base, "http://llm-gateway.local/v1")
            self.assertEqual(settings.openai_api_key, "test-local-key")
            self.assertEqual(settings.llm_model, "llmgateway/free")
            self.assertEqual(settings.llm_models, ("llmgateway/free", "llmgateway/analysis"))
            self.assertFalse(settings.dry_run)
        finally:
            reset_settings_cache()

    def test_settings_support_multiple_llm_profiles_from_env(self) -> None:
        profile_env = {
            "SUMMARY_BACKEND_LLM_PROFILES": "gateway,openai",
            "SUMMARY_BACKEND_LLM_PROFILE_DEFAULT": "openai",
            "SUMMARY_BACKEND_LLM_PROFILE__GATEWAY__LABEL": "Gateway",
            "SUMMARY_BACKEND_LLM_PROFILE__GATEWAY__API_BASE": "http://gateway.local/v1",
            "SUMMARY_BACKEND_LLM_PROFILE__GATEWAY__API_KEY": "gateway-key",
            "SUMMARY_BACKEND_LLM_PROFILE__GATEWAY__DEFAULT_MODEL": "llmgateway/free",
            "SUMMARY_BACKEND_LLM_PROFILE__GATEWAY__AVAILABLE_MODELS": "llmgateway/free,llmgateway/analysis",
            "SUMMARY_BACKEND_LLM_PROFILE__OPENAI__LABEL": "OpenAI",
            "SUMMARY_BACKEND_LLM_PROFILE__OPENAI__API_BASE": "https://api.openai.com/v1",
            "SUMMARY_BACKEND_LLM_PROFILE__OPENAI__API_KEY": "openai-key",
            "SUMMARY_BACKEND_LLM_PROFILE__OPENAI__DEFAULT_MODEL": "gpt-4.1-mini",
            "SUMMARY_BACKEND_LLM_PROFILE__OPENAI__AVAILABLE_MODELS": "gpt-4.1-mini,gpt-4.1",
            "SUMMARY_BACKEND_LLM_MODEL": "openai/gpt-4.1-mini",
        }
        try:
            with patch.dict(os.environ, profile_env, clear=False), patch(
                "summary_backend.config._load_llm_gateway_defaults",
                return_value={},
            ):
                reset_settings_cache()
                settings = get_settings()

            self.assertEqual(settings.llm_default_profile, "openai")
            self.assertEqual(settings.llm_model, "openai/gpt-4.1-mini")
            self.assertEqual(
                settings.llm_models,
                (
                    "gateway/llmgateway/free",
                    "gateway/llmgateway/analysis",
                    "openai/gpt-4.1-mini",
                    "openai/gpt-4.1",
                ),
            )
            self.assertEqual(settings.openai_api_base, "https://api.openai.com/v1")
            self.assertEqual(resolve_llm_model_option(settings, "openai/gpt-4.1").model, "gpt-4.1")
            self.assertEqual(resolve_llm_model_option(settings, "llmgateway/free").value, "gateway/llmgateway/free")
        finally:
            reset_settings_cache()


class PromptDraftGenerationTests(unittest.TestCase):
    def test_prompt_draft_spec_normalization_accepts_string_and_dict_shapes(self) -> None:
        spec = PromptDraftConceptSpec(
            map_focus="extract anomalies with timestamps",
            reduce_focus={"merge_strategy": "dedupe by metric and time window"},
            final_sections=[{"name": "Executive summary", "purpose": "top findings"}],
            final_requirements={"evidence_rules": "cite concrete values"},
        )

        normalized = _normalize_prompt_draft_spec(spec, None)

        self.assertEqual(normalized.map_focus, ["extract anomalies with timestamps"])
        self.assertEqual(normalized.reduce_focus, ["merge_strategy: dedupe by metric and time window"])
        self.assertEqual(normalized.final_sections, ["name: Executive summary; purpose: top findings"])
        self.assertEqual(normalized.final_requirements, ["evidence_rules: cite concrete values"])

    def test_prompt_draft_generation_uses_two_stage_llm_flow(self) -> None:
        llm = PromptDraftLLMStub(
            [
                PromptDraftConceptSpec(
                    report_name="Metric outliers report",
                    report_instruction="Structured report for SREs.",
                    objective="Analyze metric outliers and summarize evidence.",
                    audience="SRE team",
                    tone="Technical and concise",
                    map_focus=["Extract anomalous metrics with timestamps"],
                    reduce_focus=["Merge duplicates and preserve strongest evidence"],
                    final_sections=["Executive summary", "Timeline", "Recommendations"],
                    final_requirements=["Call out uncertainty explicitly"],
                    use_custom_output_json=True,
                    output_json_schema={"type": "object", "properties": {"wrong": {"type": "string"}}},
                ),
                GenerateSummaryPromptDraftResponse(
                    report_name="",
                    report_instruction="",
                    use_custom_output_json=False,
                    output_json_schema=None,
                    prompt_overrides=SummaryPromptOverridesDraft(
                        map=SummaryPromptStageDraft(system="map system", user="map user"),
                        reduce=SummaryPromptStageDraft(system="reduce system", user="reduce user"),
                        final=SummaryPromptStageDraft(system="final system", user="final user"),
                    ),
                ),
            ]
        )
        request = GenerateSummaryPromptDraftRequest(
            request="Сделай отчет по выбросам в метриках для SRE-команды.",
            llm_model="llmgateway/free",
            use_custom_output_json=True,
            output_json_schema={"type": "object", "properties": {"summary": {"type": "string"}}},
        )

        result = _generate_prompt_draft_with_llm(
            llm=llm,
            request=request,
            llm_model=request.llm_model,
        )

        self.assertEqual([call["stage"] for call in llm.calls], ["PROMPT_DRAFT_SPEC", "PROMPT_DRAFT_RENDER"])
        self.assertEqual(result.report_name, "Metric outliers report")
        self.assertEqual(result.report_instruction, "Structured report for SREs.")
        self.assertTrue(result.use_custom_output_json)
        assert result.output_json_schema is not None
        self.assertEqual(result.output_json_schema["type"], "object")
        self.assertEqual(result.output_json_schema["properties"]["summary"]["type"], "string")
        self.assertEqual(result.output_json_schema["required"], ["summary"])
        self.assertFalse(result.output_json_schema["additionalProperties"])
        self.assertIn("{chunk}", result.prompt_overrides.map.user)
        self.assertIn("{summaries}", result.prompt_overrides.reduce.user)
        self.assertIn("{summaries}", result.prompt_overrides.final.user)
        self.assertIn("{output_json_schema}", result.prompt_overrides.final.user)
        self.assertIn("Metric outliers report", llm.calls[1]["user"])
        self.assertIn("SRE team", llm.calls[1]["user"])

    def test_requested_custom_output_json_generates_normalized_schema(self) -> None:
        llm = PromptDraftLLMStub(
            [
                PromptDraftConceptSpec(
                    report_name="Outlier report",
                    report_instruction="Instruction",
                    use_custom_output_json=True,
                    output_json_schema={"headline": "string", "items": [{"severity": "string", "impact": "string"}]},
                ),
                GenerateSummaryPromptDraftResponse(
                    report_name="Outlier report",
                    report_instruction="Instruction",
                    use_custom_output_json=True,
                    output_json_schema={"headline": "string", "items": [{"severity": "string", "impact": "string"}]},
                    prompt_overrides=SummaryPromptOverridesDraft(
                        map=SummaryPromptStageDraft(system="map system", user="map user"),
                        reduce=SummaryPromptStageDraft(system="reduce system", user="reduce user"),
                        final=SummaryPromptStageDraft(system="final system", user="final user"),
                    ),
                ),
            ]
        )
        request = GenerateSummaryPromptDraftRequest(
            request="Сделай отчет по аномалиям с четкой JSON-структурой.",
            llm_model="llmgateway/free",
            use_custom_output_json=True,
        )

        result = _generate_prompt_draft_with_llm(llm=llm, request=request, llm_model=request.llm_model)

        assert result.output_json_schema is not None
        self.assertTrue(result.use_custom_output_json)
        self.assertEqual(result.output_json_schema["type"], "object")
        self.assertEqual(result.output_json_schema["properties"]["headline"]["type"], "string")
        self.assertEqual(result.output_json_schema["properties"]["items"]["type"], "array")
        self.assertEqual(
            result.output_json_schema["properties"]["items"]["items"]["properties"]["severity"]["type"],
            "string",
        )
        self.assertFalse(result.output_json_schema["additionalProperties"])
        self.assertIn("{output_json_schema}", result.prompt_overrides.final.user)

    def test_requested_custom_intermediate_output_json_generates_normalized_schema(self) -> None:
        llm = PromptDraftLLMStub(
            [
                PromptDraftConceptSpec(
                    report_name="Structured intermediate report",
                    report_instruction="Instruction",
                    use_custom_intermediate_output_json=True,
                    intermediate_output_json_schema={
                        "chunk_summary": "string",
                        "evidence": [{"title": "string", "details": "string"}],
                    },
                ),
                GenerateSummaryPromptDraftResponse(
                    report_name="Structured intermediate report",
                    report_instruction="Instruction",
                    use_custom_intermediate_output_json=True,
                    intermediate_output_json_schema={
                        "chunk_summary": "string",
                        "evidence": [{"title": "string", "details": "string"}],
                    },
                    prompt_overrides=SummaryPromptOverridesDraft(
                        map=SummaryPromptStageDraft(system="map system", user="map user"),
                        reduce=SummaryPromptStageDraft(system="reduce system", user="reduce user"),
                        final=SummaryPromptStageDraft(system="final system", user="final user"),
                    ),
                ),
            ]
        )
        request = GenerateSummaryPromptDraftRequest(
            request="Сделай промежуточные JSON-объекты для MAP и REDUCE по логам.",
            llm_model="llmgateway/free",
            use_custom_intermediate_output_json=True,
        )

        result = _generate_prompt_draft_with_llm(llm=llm, request=request, llm_model=request.llm_model)

        assert result.intermediate_output_json_schema is not None
        self.assertTrue(result.use_custom_intermediate_output_json)
        self.assertEqual(result.intermediate_output_json_schema["type"], "object")
        self.assertEqual(result.intermediate_output_json_schema["properties"]["chunk_summary"]["type"], "string")
        self.assertEqual(result.intermediate_output_json_schema["properties"]["evidence"]["type"], "array")
        self.assertIn("{intermediate_output_json_schema}", result.prompt_overrides.map.user)
        self.assertIn("{intermediate_output_json_schema}", result.prompt_overrides.reduce.user)

    def test_prompt_draft_generation_keeps_stage_specific_schemas_in_one_draft(self) -> None:
        llm = PromptDraftLLMStub(
            [
                PromptDraftConceptSpec(
                    report_name="Structured multi-stage report",
                    report_instruction="Instruction",
                    use_custom_map_output_json=True,
                    map_output_json_schema={
                        "chunk_summary": "string",
                        "evidence": [{"metric": "string", "value": "number"}],
                    },
                    use_custom_reduce_output_json=True,
                    reduce_output_json_schema={
                        "merged_summary": "string",
                        "deduplicated_findings": ["string"],
                        "evidence_groups": [{"title": "string", "details": "string"}],
                    },
                    use_custom_output_json=True,
                    output_json_schema={
                        "executive_summary": "string",
                        "recommendations": ["string"],
                    },
                ),
                GenerateSummaryPromptDraftResponse(
                    report_name="Structured multi-stage report",
                    report_instruction="Instruction",
                    prompt_overrides=SummaryPromptOverridesDraft(
                        map=SummaryPromptStageDraft(system="map system", user="Map chunk:\n{chunk}"),
                        reduce=SummaryPromptStageDraft(system="reduce system", user="Reduce merges:\n{summaries}"),
                        final=SummaryPromptStageDraft(system="final system", user="Final report:\n{summaries}"),
                    ),
                ),
            ]
        )
        request = GenerateSummaryPromptDraftRequest(
            request="Сделай отдельные JSON для map chunks, reduce merges и final report.",
            llm_model="llmgateway/free",
        )

        result = _generate_prompt_draft_with_llm(llm=llm, request=request, llm_model=request.llm_model)

        self.assertTrue(result.use_custom_map_output_json)
        self.assertTrue(result.use_custom_reduce_output_json)
        self.assertTrue(result.use_custom_output_json)
        assert result.map_output_json_schema is not None
        assert result.reduce_output_json_schema is not None
        assert result.output_json_schema is not None
        self.assertEqual(result.map_output_json_schema["properties"]["chunk_summary"]["type"], "string")
        self.assertEqual(result.reduce_output_json_schema["properties"]["merged_summary"]["type"], "string")
        self.assertEqual(result.output_json_schema["properties"]["executive_summary"]["type"], "string")
        self.assertIn("{chunk}", result.prompt_overrides.map.user)
        self.assertIn("{intermediate_output_json_schema}", result.prompt_overrides.map.user)
        self.assertIn("{summaries}", result.prompt_overrides.reduce.user)
        self.assertIn("{intermediate_output_json_schema}", result.prompt_overrides.reduce.user)
        self.assertIn("{summaries}", result.prompt_overrides.final.user)
        self.assertIn("{output_json_schema}", result.prompt_overrides.final.user)

    def test_disabled_custom_output_json_discards_generated_schema(self) -> None:
        llm = PromptDraftLLMStub(
            [
                PromptDraftConceptSpec(
                    report_name="Outlier report",
                    report_instruction="Instruction",
                    use_custom_output_json=True,
                    output_json_schema={"headline": "string"},
                ),
                GenerateSummaryPromptDraftResponse(
                    report_name="Outlier report",
                    report_instruction="Instruction",
                    use_custom_output_json=True,
                    output_json_schema={"headline": "string"},
                    prompt_overrides=SummaryPromptOverridesDraft(
                        map=SummaryPromptStageDraft(system="map system", user="map user"),
                        reduce=SummaryPromptStageDraft(system="reduce system", user="reduce user"),
                        final=SummaryPromptStageDraft(system="final system", user="final user"),
                    ),
                ),
            ]
        )
        request = GenerateSummaryPromptDraftRequest(
            request="Сделай отчет по аномалиям без кастомного JSON.",
            llm_model="llmgateway/free",
            use_custom_output_json=False,
        )

        result = _generate_prompt_draft_with_llm(llm=llm, request=request, llm_model=request.llm_model)

        self.assertFalse(result.use_custom_output_json)
        self.assertIsNone(result.output_json_schema)
        self.assertNotIn("{output_json_schema}", result.prompt_overrides.final.user)

    def test_invalid_prompt_draft_model_falls_back_to_default(self) -> None:
        self.assertEqual(_resolve_requested_prompt_draft_model("gpt-4.1-mini"), get_settings().llm_model)
        self.assertEqual(_resolve_requested_prompt_draft_model("llmgateway/free"), "llmgateway/free")
        self.assertIsNone(_resolve_requested_prompt_draft_model("   "))


class LlmConnectivityCheckTests(unittest.TestCase):
    def test_run_llm_connectivity_checks_returns_status_per_model(self) -> None:
        settings = replace(
            get_settings(),
            llm_model="llmgateway/free",
            llm_models=("llmgateway/free", "llmgateway/analysis"),
        )

        class StubClient:
            def probe_connection(self, *, model: str | None = None):
                if model == "llmgateway/free":
                    return {
                        "ok": True,
                        "status": "ok",
                        "detail": "free ok",
                        "error_class": "",
                        "latency_ms": 123,
                        "selected_model": "llmgateway/free",
                        "api_base": "http://gateway.local/v1",
                    }
                return {
                    "ok": False,
                    "status": "error",
                    "detail": "analysis failed",
                    "error_class": "provider_unavailable",
                    "latency_ms": 456,
                    "selected_model": "llmgateway/analysis",
                    "api_base": "http://gateway.local/v1",
                }

        results = _run_llm_connectivity_checks(client=StubClient(), current_settings=settings)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].value, "llmgateway/free")
        self.assertTrue(results[0].ok)
        self.assertEqual(results[0].status, "ok")
        self.assertEqual(results[0].latency_ms, 123)
        self.assertEqual(results[1].value, "llmgateway/analysis")
        self.assertFalse(results[1].ok)
        self.assertEqual(results[1].error_class, "provider_unavailable")

        filtered = _run_llm_connectivity_checks(
            client=StubClient(),
            current_settings=settings,
            llm_model="llmgateway/analysis",
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].value, "llmgateway/analysis")

    def test_llm_probe_connection_reports_dry_run_without_network_call(self) -> None:
        class NoopStore:
            def insert_llm_call(self, **kwargs: Any) -> None:
                return None

        client = StructuredLLMClient(
            store=NoopStore(),
            settings=replace(
                get_settings(),
                dry_run=True,
                llm_model="llmgateway/free",
                llm_models=("llmgateway/free",),
            ),
        )

        result = client.probe_connection(model="llmgateway/free")

        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "dry_run")
        self.assertIn("Dry run mode", result["detail"])

    def test_structured_llm_client_uses_assistant_pool_kind(self) -> None:
        class NoopStore:
            def insert_llm_call(self, **kwargs: Any) -> None:
                return None

        captured: dict[str, Any] = {}

        class FakeAcquire:
            def __init__(self, *_args: Any, **kwargs: Any) -> None:
                captured.update(kwargs)

            def __enter__(self) -> int:
                return 0

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

        class StubResponse:
            def __init__(self) -> None:
                self.choices = [type("Choice", (), {"message": type("Message", (), {"content": '{"ok": true, "summary": "done", "key_points": [], "source_count": 1}'})()})()]
                self.usage = type(
                    "Usage",
                    (),
                    {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                )()

            def model_dump(self, mode: str = "json") -> dict[str, Any]:
                del mode
                return {"choices": [{"message": {"content": '{"ok": true, "summary": "done", "key_points": [], "source_count": 1}'}}]}

        class FakeOpenAI:
            def __init__(self, **kwargs: Any) -> None:
                del kwargs
                self.chat = type(
                    "Chat",
                    (),
                    {"completions": type("Completions", (), {"create": staticmethod(lambda **_kwargs: StubResponse())})()},
                )()

        settings = replace(
            get_settings(),
            dry_run=False,
            openai_api_base="http://llm-gateway.local/v1",
            openai_api_key="secret",
            llm_model="llmgateway/free",
            llm_models=("llmgateway/free",),
            llm_max_concurrency=2,
            llm_assistant_max_concurrency=7,
        )
        client = StructuredLLMClient(
            store=NoopStore(),
            settings=settings,
            pool_kind="assistant",
        )

        with patch("summary_backend.llm_client.acquire_llm_pool_slot", FakeAcquire), patch(
            "openai.OpenAI",
            FakeOpenAI,
        ):
            result = client.call_summary(
                job_id="job-1",
                node_id="node-1",
                stage="MAP",
                system="system",
                user="user",
            )

        self.assertTrue(result.ok)
        self.assertEqual(captured["pool_kind"], "assistant")
        self.assertIsNone(captured.get("job_max_concurrency"))

    def test_structured_llm_client_passes_job_max_concurrency_to_pool(self) -> None:
        class NoopStore:
            def insert_llm_call(self, **kwargs: Any) -> None:
                return None

        captured: dict[str, Any] = {}

        class FakeAcquire:
            def __init__(self, *_args: Any, **kwargs: Any) -> None:
                captured.update(kwargs)

            def __enter__(self) -> int:
                return 0

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

        class StubResponse:
            def __init__(self) -> None:
                self.choices = [type("Choice", (), {"message": type("Message", (), {"content": '{"ok": true, "summary": "done", "key_points": [], "source_count": 1}'})()})()]
                self.usage = type(
                    "Usage",
                    (),
                    {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                )()

            def model_dump(self, mode: str = "json") -> dict[str, Any]:
                del mode
                return {"choices": [{"message": {"content": '{"ok": true, "summary": "done", "key_points": [], "source_count": 1}'}}]}

        class FakeOpenAI:
            def __init__(self, **kwargs: Any) -> None:
                del kwargs
                self.chat = type(
                    "Chat",
                    (),
                    {"completions": type("Completions", (), {"create": staticmethod(lambda **_kwargs: StubResponse())})()},
                )()

        settings = replace(
            get_settings(),
            dry_run=False,
            openai_api_base="http://llm-gateway.local/v1",
            openai_api_key="secret",
            llm_model="llmgateway/free",
            llm_models=("llmgateway/free",),
            llm_max_concurrency=4,
        )
        client = StructuredLLMClient(
            store=NoopStore(),
            settings=settings,
            pool_kind="jobs",
        )

        with patch("summary_backend.llm_client.acquire_llm_pool_slot", FakeAcquire), patch(
            "openai.OpenAI",
            FakeOpenAI,
        ):
            result = client.call_summary(
                job_id="job-1",
                node_id="node-1",
                stage="MAP",
                system="system",
                user="user",
                job_max_concurrency=1,
            )

        self.assertTrue(result.ok)
        self.assertEqual(captured["pool_kind"], "jobs")
        self.assertEqual(captured["job_max_concurrency"], 1)

    def test_prompt_draft_job_runner_builds_assistant_pool_client(self) -> None:
        manager = PromptDraftJobManager()
        captured: dict[str, Any] = {}
        expected = GenerateSummaryPromptDraftResponse(
            report_name="Draft",
            report_instruction="Instruction",
            use_custom_output_json=False,
            output_json_schema=None,
            prompt_overrides=SummaryPromptOverridesDraft(
                map=SummaryPromptStageDraft(system="map", user="{chunk}"),
                reduce=SummaryPromptStageDraft(system="reduce", user="{summaries}"),
                final=SummaryPromptStageDraft(system="final", user="{summaries}"),
            ),
        )

        class FakeStructuredLLMClient:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        def fake_generate_prompt_draft_with_llm(**kwargs: Any) -> GenerateSummaryPromptDraftResponse:
            self.assertIsInstance(kwargs["llm"], FakeStructuredLLMClient)
            return expected

        with patch("summary_backend.api.StructuredLLMClient", FakeStructuredLLMClient), patch(
            "summary_backend.api._generate_prompt_draft_with_llm",
            side_effect=fake_generate_prompt_draft_with_llm,
        ):
            result = manager._run_job(
                GenerateSummaryPromptDraftRequest(request="Draft report"),
                "llmgateway/free",
                lambda _stage: None,
                lambda: False,
            )

        self.assertEqual(result.report_name, "Draft")
        self.assertEqual(captured["pool_kind"], "assistant")


class PromptDraftJobManagerTests(unittest.TestCase):
    def test_prompt_draft_job_completes_in_background(self) -> None:
        manager = PromptDraftJobManager()
        expected = GenerateSummaryPromptDraftResponse(
            report_name="Async report",
            report_instruction="Instruction",
            use_custom_output_json=False,
            output_json_schema=None,
            prompt_overrides=SummaryPromptOverridesDraft(
                map=SummaryPromptStageDraft(system="map", user="{chunk}"),
                reduce=SummaryPromptStageDraft(system="reduce", user="{summaries}"),
                final=SummaryPromptStageDraft(system="final", user="{summaries}"),
            ),
        )

        def runner(request, llm_model, stage_observer, cancel_checker):
            stage_observer("PROMPT_DRAFT_SPEC")
            self.assertFalse(cancel_checker())
            stage_observer("PROMPT_DRAFT_RENDER")
            return expected

        job = manager.create_job(
            request=GenerateSummaryPromptDraftRequest(request="draft async report"),
            llm_model="llmgateway/free",
            runner=runner,
        )

        final = self._wait_for_prompt_draft_job(manager, job.job_id)

        self.assertEqual(final.status, PromptDraftJobStatus.DONE)
        assert final.result is not None
        self.assertEqual(final.result.report_name, "Async report")
        self.assertEqual(final.stage, "DONE")

    def test_prompt_draft_job_waits_for_provider_and_retries(self) -> None:
        manager = PromptDraftJobManager()
        expected = GenerateSummaryPromptDraftResponse(
            report_name="Recovered report",
            report_instruction="Instruction",
            use_custom_output_json=False,
            output_json_schema=None,
            prompt_overrides=SummaryPromptOverridesDraft(
                map=SummaryPromptStageDraft(system="map", user="{chunk}"),
                reduce=SummaryPromptStageDraft(system="reduce", user="{summaries}"),
                final=SummaryPromptStageDraft(system="final", user="{summaries}"),
            ),
        )
        call_counter = {"value": 0}
        waiting_seen = False
        release_retry = threading.Event()

        def runner(request, llm_model, stage_observer, cancel_checker):
            del request, llm_model, cancel_checker
            call_counter["value"] += 1
            if call_counter["value"] == 1:
                raise LlmPoolBusyError("LLM pool acquire timeout after 1.0s; max_concurrency=2")
            stage_observer("PROMPT_DRAFT_RENDER")
            return expected

        def block_retry_sleep(_seconds: float) -> None:
            release_retry.wait(0.5)

        with patch("summary_backend.api._probe_prompt_draft_llm_availability", return_value="LLM availability ping succeeded (42 ms)."), patch(
            "summary_backend.api.time.sleep",
            side_effect=block_retry_sleep,
        ):
            job = manager.create_job(
                request=GenerateSummaryPromptDraftRequest(request="recover after provider wait"),
                llm_model="llmgateway/free",
                runner=runner,
            )

            deadline = time.time() + 1.0
            while time.time() < deadline:
                current = manager.get_job(job.job_id)
                if current.status == PromptDraftJobStatus.WAITING_PROVIDER:
                    waiting_seen = True
                    self.assertEqual(current.stage, "WAITING_PROVIDER")
                    self.assertIn("LLM pool acquire timeout after 1.0s; max_concurrency=2", current.error_detail)
                    self.assertIn("LLM availability ping succeeded (42 ms).", current.error_detail)
                    release_retry.set()
                    break
                time.sleep(0.005)

            release_retry.set()
            final = self._wait_for_prompt_draft_job(manager, job.job_id)

        self.assertTrue(waiting_seen)
        self.assertEqual(call_counter["value"], 2)
        self.assertEqual(final.status, PromptDraftJobStatus.DONE)
        self.assertEqual(final.stage, "DONE")
        self.assertEqual(final.error_detail, "")
        assert final.result is not None
        self.assertEqual(final.result.report_name, "Recovered report")

    def test_prompt_draft_job_cancel_marks_job_cancelled(self) -> None:
        manager = PromptDraftJobManager()

        def runner(request, llm_model, stage_observer, cancel_checker):
            stage_observer("PROMPT_DRAFT_SPEC")
            deadline = time.time() + 1.0
            while time.time() < deadline:
                if cancel_checker():
                    raise RuntimeError("cancel requested")
                time.sleep(0.01)
            return GenerateSummaryPromptDraftResponse(
                report_name="Should not finish",
                report_instruction="",
                use_custom_output_json=False,
                output_json_schema=None,
                prompt_overrides=SummaryPromptOverridesDraft(
                    map=SummaryPromptStageDraft(system="map", user="{chunk}"),
                    reduce=SummaryPromptStageDraft(system="reduce", user="{summaries}"),
                    final=SummaryPromptStageDraft(system="final", user="{summaries}"),
                ),
            )

        job = manager.create_job(
            request=GenerateSummaryPromptDraftRequest(request="cancel async report"),
            llm_model="llmgateway/free",
            runner=runner,
        )
        manager.cancel_job(job.job_id)
        final = self._wait_for_prompt_draft_job(manager, job.job_id)

        self.assertEqual(final.status, PromptDraftJobStatus.CANCELLED)
        self.assertIsNone(final.result)
        self.assertEqual(final.stage, "CANCELLED")

    def _wait_for_prompt_draft_job(self, manager: PromptDraftJobManager, job_id: str, timeout: float = 2.0):
        deadline = time.time() + timeout
        last = manager.get_job(job_id)
        while time.time() < deadline:
            current = manager.get_job(job_id)
            if current.status in {PromptDraftJobStatus.DONE, PromptDraftJobStatus.FAILED, PromptDraftJobStatus.CANCELLED}:
                return current
            last = current
            time.sleep(0.01)
        self.fail(f"prompt draft job {job_id} did not finish in time; last status={last.status}")


class SummaryWorkerRecoveryTests(unittest.TestCase):
    def test_worker_boot_enqueues_recovery_once(self) -> None:
        from summary_backend import tasks as task_module

        previous = task_module._recovery_enqueued_on_worker_boot
        task_module._recovery_enqueued_on_worker_boot = False
        try:
            with patch.object(task_module.recover_jobs, "send") as send:
                task_module._RecoveryOnWorkerBootMiddleware().after_worker_boot(None, object())
                task_module._RecoveryOnWorkerBootMiddleware().after_worker_boot(None, object())

            send.assert_called_once_with()
        finally:
            task_module._recovery_enqueued_on_worker_boot = previous

    def test_recovery_poll_is_rescheduled_while_jobs_are_runnable(self) -> None:
        from summary_backend import tasks as task_module

        with patch.object(task_module.recover_jobs, "send_with_options") as send_with_options:
            task_module._schedule_recovery_poll_if_needed(["job_running"])
            task_module._schedule_recovery_poll_if_needed([])

        send_with_options.assert_called_once_with(delay=task_module._recovery_poll_delay_ms())


class StructuredLLMClientParsingTests(unittest.TestCase):
    def test_parse_summary_result_from_markdown_fenced_json(self) -> None:
        payload = StructuredLLMClient._parse_response_payload(
            content='```json\n{"summary":"ok","key_points":["a"],"warnings":[],"source_count":2}\n```',
            response_model=SummaryResult,
        )

        result = SummaryResult.model_validate(payload)

        self.assertTrue(result.ok)
        self.assertEqual(result.summary, "ok")
        self.assertEqual(result.key_points, ["a"])
        self.assertEqual(result.source_count, 2)

    def test_parse_summary_result_normalizes_object_summary(self) -> None:
        payload = StructuredLLMClient._parse_response_payload(
            content=json.dumps(
                {
                    "summary": {
                        "topic": "black holes",
                        "key_points": [{"content": "event horizon"}, "Schwarzschild"],
                    }
                }
            ),
            response_model=SummaryResult,
        )

        result = SummaryResult.model_validate(payload)

        self.assertTrue(result.ok)
        self.assertIn("black holes", result.summary)
        self.assertEqual(result.key_points, ["event horizon", "Schwarzschild"])
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.source_count, 1)

    def test_parse_summary_result_normalizes_alternate_gateway_shape(self) -> None:
        payload = StructuredLLMClient._parse_response_payload(
            content="""```json
{
  "incident_timeline": {
    "2026-06-26 17:00": "Service API latency increased after deployment v42",
    "2026-06-26 17:05": "Rollback initiated"
  },
  "root_cause": "Deployment v42",
  "impact": {"error_rate": "12%"},
  "next_actions": "Add rollback verification"
}
```""",
            response_model=SummaryResult,
        )

        result = SummaryResult.model_validate(payload)

        self.assertTrue(result.ok)
        self.assertIn("Incident timeline", result.summary)
        self.assertIn("Deployment v42", result.summary)
        self.assertIn("root cause: Deployment v42", result.key_points)
        self.assertEqual(result.source_count, 1)


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
            llm_max_concurrency=2,
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
        reduce_artifacts = store.list_artifacts(
            job_id=job_id,
            include_content=False,
            artifact_type=ArtifactType.REDUCE_SUMMARY,
        )
        for artifact in reduce_artifacts:
            metadata = json.loads(artifact["metadata"])
            self.assertGreater(len(metadata["input_node_ids"]), 0)

        nodes = store.list_nodes_current(job_id)
        map_node_indices = {
            str(node["node_id"]): int(node["node_index"])
            for node in nodes
            if node["node_type"] == NodeType.MAP
        }
        level1_reduce_nodes = [
            node
            for node in nodes
            if node["node_type"] == NodeType.REDUCE and int(node["level"]) == 1
        ]
        grouped_input_indices: list[list[int]] = []
        for node in level1_reduce_nodes:
            payload = store.get_node_payload(job_id, str(node["node_id"]))
            input_indices = [map_node_indices[str(input_node_id)] for input_node_id in payload["input_node_ids"]]
            grouped_input_indices.append(input_indices)
            self.assertEqual(payload["input_sequence_start"], input_indices[0])
            self.assertEqual(payload["input_sequence_end"], input_indices[-1])
        self.assertEqual(grouped_input_indices, [[0, 1], [2, 3], [4]])

        reduce_ranges_by_level = sorted(
            [
                (
                    int(artifact["level"]),
                    int(json.loads(artifact["metadata"])["sequence_start"]),
                    int(json.loads(artifact["metadata"])["sequence_end"]),
                )
                for artifact in reduce_artifacts
            ]
        )
        self.assertEqual(
            reduce_ranges_by_level,
            [
                (1, 0, 1),
                (1, 2, 3),
                (1, 4, 4),
                (2, 0, 3),
                (2, 4, 4),
                (3, 0, 4),
            ],
        )
        final_nodes = [node for node in nodes if node["node_type"] == NodeType.FINAL]
        self.assertEqual(len(final_nodes), 1)
        final_payload = store.get_node_payload(job_id, str(final_nodes[0]["node_id"]))
        self.assertEqual(final_payload["input_count"], 1)
        self.assertEqual(len(final_payload["input_node_ids"]), 1)
        self.assertGreaterEqual(len(llm.calls), 12)

    def test_reduce_grouping_forces_progress_when_token_budget_would_stall(self) -> None:
        store = InMemorySummaryStore()
        llm = LargeSummaryLLM()
        queue = ManualQueue()
        settings = replace(
            get_settings(),
            reduce_group_size=2,
            reduce_target_estimated_tokens=256,
            llm_max_concurrency=2,
            max_enqueue_nodes_per_advance=100,
            chunk_target_estimated_tokens=100,
        )
        service = PipelineService(
            store=store,
            queue=queue,
            llm=llm,
            chunker=FakeChunker(["chunk-1", "chunk-2", "chunk-3", "chunk-4"]),
            input_segmenter=FakeInputSegmenter(["chunk-1", "chunk-2", "chunk-3", "chunk-4"]),
            settings=settings,
        )
        job_id = service.create_job(input_text="input", title="reduce-force-progress", metadata={})

        queue.advance_job(job_id)
        self.drain(queue, service)

        self.assertEqual(store.get_job_current(job_id)["job_status"], JobStatus.DONE)
        reduce_nodes = [
            node
            for node in store.list_nodes_current(job_id)
            if node["node_type"] == NodeType.REDUCE
        ]
        self.assertEqual(Counter(int(node["level"]) for node in reduce_nodes), Counter({1: 2, 2: 1}))

        level1_payloads = [
            store.get_node_payload(job_id, str(node["node_id"]))
            for node in reduce_nodes
            if int(node["level"]) == 1
        ]
        self.assertTrue(level1_payloads)
        self.assertTrue(all(payload["grouping_strategy"] == "forced_progress" for payload in level1_payloads))
        self.assertEqual([payload["input_count"] for payload in level1_payloads], [2, 2])

        level2_payload = next(
            store.get_node_payload(job_id, str(node["node_id"]))
            for node in reduce_nodes
            if int(node["level"]) == 2
        )
        self.assertEqual(level2_payload["grouping_strategy"], "forced_progress")
        self.assertEqual(level2_payload["input_count"], 2)

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

    def test_pause_marks_running_node_paused_and_drops_stale_result_before_resume(self) -> None:
        service, store, _, queue = self.make_service(chunks=["chunk-1"])
        blocking_llm = BlockingThenFastLLM()
        service.llm = blocking_llm
        job_id = service.create_job(input_text="input", title="pause-running", metadata={})

        service.advance_job(job_id)
        first_map = next(item for item in list(queue.items) if item[0] == "map")
        queue.items.remove(first_map)
        _, _, node_id = first_map
        assert node_id is not None

        worker = threading.Thread(target=service.map_node, args=(job_id, node_id))
        worker.start()
        self.assertTrue(blocking_llm.started.wait(timeout=2))

        service.pause_job(job_id)
        self.assertEqual(store.get_job_current(job_id)["job_status"], JobStatus.PAUSED)
        paused_node = next(item for item in store.list_nodes_current(job_id) if item["node_id"] == node_id)
        self.assertEqual(paused_node["node_status"], NodeStatus.PAUSED)

        service.resume_job(job_id)
        resumed_node = next(item for item in store.list_nodes_current(job_id) if item["node_id"] == node_id)
        self.assertEqual(resumed_node["node_status"], NodeStatus.PENDING)

        blocking_llm.release.set()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(store.latest_artifact(job_id=job_id, node_id=node_id, artifact_type=ArtifactType.MAP_SUMMARY))

        self.drain(queue, service)

        self.assertEqual(store.get_job_current(job_id)["job_status"], JobStatus.DONE)
        self.assertIsNotNone(store.latest_artifact(job_id=job_id, node_id=node_id, artifact_type=ArtifactType.MAP_SUMMARY))
        self.assertEqual(len([call for call in blocking_llm.calls if call["node_id"] == node_id]), 2)

    def test_cancel_blocks_work(self) -> None:
        service, store, _, _ = self.make_service(chunks=["chunk-1", "chunk-2"])
        job_id = service.create_job(input_text="input", title="cancel", metadata={})

        service.cancel_job(job_id)
        service.advance_job(job_id)

        self.assertEqual(store.get_job_current(job_id)["job_status"], JobStatus.CANCELLED)
        self.assertEqual(len(store.list_nodes_current(job_id)), 0)
        artifact_counts = Counter(artifact["artifact_type"] for artifact in store.list_artifacts(job_id=job_id, include_content=False))
        self.assertEqual(artifact_counts, Counter({ArtifactType.INPUT: 1}))

    def test_cancel_marks_running_node_cancelled_and_discards_late_result(self) -> None:
        service, store, _, queue = self.make_service(chunks=["chunk-1"])
        blocking_llm = BlockingThenFastLLM()
        service.llm = blocking_llm
        job_id = service.create_job(input_text="input", title="cancel-running", metadata={})

        service.advance_job(job_id)
        first_map = next(item for item in list(queue.items) if item[0] == "map")
        queue.items.remove(first_map)
        _, _, node_id = first_map
        assert node_id is not None

        worker = threading.Thread(target=service.map_node, args=(job_id, node_id))
        worker.start()
        self.assertTrue(blocking_llm.started.wait(timeout=2))

        service.cancel_job(job_id)
        self.assertEqual(store.get_job_current(job_id)["job_status"], JobStatus.CANCELLED)
        cancelled_node = next(item for item in store.list_nodes_current(job_id) if item["node_id"] == node_id)
        self.assertEqual(cancelled_node["node_status"], NodeStatus.CANCELLED)

        blocking_llm.release.set()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())

        self.assertIsNone(store.latest_artifact(job_id=job_id, node_id=node_id, artifact_type=ArtifactType.MAP_SUMMARY))
        self.assertEqual(store.get_job_current(job_id)["job_status"], JobStatus.CANCELLED)
        current_node = next(item for item in store.list_nodes_current(job_id) if item["node_id"] == node_id)
        self.assertEqual(current_node["node_status"], NodeStatus.CANCELLED)

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

    def test_llm_pool_busy_defers_node_without_failing_job(self) -> None:
        service, store, _, queue = self.make_service(chunks=["chunk-1"])
        service.llm = BusyPoolLLM()
        job_id = service.create_job(input_text="input", title="pool-busy", metadata={})

        queue.advance_job(job_id)
        service.advance_job(job_id)
        first_map = next(item for item in list(queue.items) if item[0] == "map")
        queue.items.remove(first_map)
        _, _, node_id = first_map
        assert node_id is not None

        with self.assertRaises(LlmPoolBusyError):
            service.map_node(job_id, node_id)

        self.assertEqual(store.get_job_current(job_id)["job_status"], JobStatus.WAITING_PROVIDER)
        nodes = store.list_nodes_current(job_id)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["node_status"], NodeStatus.WAITING_PROVIDER)
        self.assertEqual(nodes[0]["last_event_type"], "NODE_WAITING_PROVIDER")
        payload = json.loads(store.list_node_events(job_id)[-1]["payload"])
        self.assertEqual(payload["error_class"], "llm_pool_busy")

    def test_advance_limits_enqueued_llm_nodes_to_pool_concurrency(self) -> None:
        service, store, _, queue = self.make_service(chunks=[f"chunk-{index}" for index in range(7)])
        job_id = service.create_job(input_text="input", title="dispatch-limit", metadata={})

        service.advance_job(job_id)

        queued_maps = [item for item in queue.items if item[0] == "map"]
        self.assertEqual(len(queued_maps), 2)
        nodes = store.list_nodes_current(job_id)
        self.assertEqual(
            Counter(str(node["node_status"]) for node in nodes),
            Counter({str(NodeStatus.QUEUED): 2, str(NodeStatus.PENDING): 5}),
        )

        service.advance_job(job_id)

        queued_maps_after_second_advance = [item for item in queue.items if item[0] == "map"]
        self.assertEqual(len(queued_maps_after_second_advance), 2)

    def test_job_llm_concurrency_can_reduce_dispatch_limit(self) -> None:
        service, store, _, queue = self.make_service(chunks=[f"chunk-{index}" for index in range(4)])
        job_id = service.create_job(input_text="input", title="job-dispatch-limit", metadata={"llm_concurrency": 1})

        service.advance_job(job_id)

        queued_maps = [item for item in queue.items if item[0] == "map"]
        self.assertEqual(len(queued_maps), 1)
        nodes = store.list_nodes_current(job_id)
        self.assertEqual(
            Counter(str(node["node_status"]) for node in nodes),
            Counter({str(NodeStatus.QUEUED): 1, str(NodeStatus.PENDING): 3}),
        )

    def test_job_llm_concurrency_is_capped_by_env_limit(self) -> None:
        service, _, _, queue = self.make_service(chunks=[f"chunk-{index}" for index in range(4)])
        job_id = service.create_job(input_text="input", title="job-dispatch-limit", metadata={"llm_concurrency": 99})

        service.advance_job(job_id)

        queued_maps = [item for item in queue.items if item[0] == "map"]
        self.assertEqual(len(queued_maps), 2)

    def test_map_node_passes_job_llm_concurrency_into_llm_call(self) -> None:
        service, store, _, _ = self.make_service(chunks=["chunk-1"])
        llm = JobConcurrencyAwareLLM()
        service.llm = llm
        job_id = service.create_job(input_text="input", title="job-runtime-limit", metadata={"llm_concurrency": 1})
        service.advance_job(job_id)

        node = next(item for item in store.list_nodes_current(job_id) if item["node_type"] == str(NodeType.MAP))
        service.map_node(job_id, str(node["node_id"]))

        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0]["job_max_concurrency"], 1)

    def test_advance_requeues_stale_running_nodes_after_worker_restart(self) -> None:
        service, store, _, queue = self.make_service(chunks=["chunk-1", "chunk-2", "chunk-3"])
        job_id = service.create_job(input_text="input", title="stale-running", metadata={})
        service.advance_job(job_id)
        first_map = next(item for item in list(queue.items) if item[0] == "map")
        queue.items.remove(first_map)
        _, _, node_id = first_map
        assert node_id is not None
        node = next(item for item in store.list_nodes_current(job_id) if item["node_id"] == node_id)
        store.insert_node_event(
            job_id=job_id,
            node_id=node_id,
            event_type="NODE_STARTED",
            node_status=NodeStatus.RUNNING,
            node_type=NodeType.MAP,
            level=int(node["level"]),
            node_index=int(node["node_index"]),
            actor="map_node",
        )
        store.node_events[-1]["event_time"] = datetime.now(timezone.utc) - timedelta(
            seconds=service.settings.llm_timeout_seconds + 180,
        )

        service.advance_job(job_id)

        requeued_maps = [item for item in queue.items if item[0] == "map" and item[2] == node_id]
        self.assertEqual(len(requeued_maps), 1)
        current_node = next(item for item in store.list_nodes_current(job_id) if item["node_id"] == node_id)
        self.assertEqual(current_node["node_status"], NodeStatus.QUEUED)

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
        self.assertIn("put the complete requested report in the summary string", final_call["user"])

        final_artifact = store.latest_artifact(job_id=job_id, artifact_type=ArtifactType.FINAL_SUMMARY)
        self.assertIsNotNone(final_artifact)
        metadata = json.loads(final_artifact["metadata"])
        self.assertEqual(metadata["report_format"], "technical_rca")
        self.assertIn("Highlight remediation owners.", metadata["report_format_instruction"])

    def test_prompt_overrides_are_used_for_pipeline_stages(self) -> None:
        service, _, llm, queue = self.make_service(chunks=[f"chunk-{index}" for index in range(9)])
        job_id = service.create_job(
            input_text="input",
            title="prompts",
            metadata={
                "prompt_overrides": {
                    "map": {
                        "system": "MAP SYS",
                        "user": "MAP CUSTOM {chunk}",
                    },
                    "reduce": {
                        "system": "REDUCE SYS",
                        "user": "REDUCE CUSTOM {summaries}",
                    },
                    "final": {
                        "system": "FINAL SYS",
                        "user": "FINAL CUSTOM {summaries} / {report_format_instruction}",
                    },
                },
                "report_format": "custom",
                "report_format_instruction": "Use owner fields.",
            },
        )

        queue.advance_job(job_id)
        self.drain(queue, service)

        map_call = next(call for call in llm.calls if call["stage"] == Stage.MAP)
        reduce_call = next(call for call in llm.calls if str(call["stage"]).startswith(str(Stage.REDUCE)))
        final_call = next(call for call in llm.calls if call["stage"] == Stage.FINAL)
        self.assertEqual(map_call["system"], "MAP SYS")
        self.assertIn("MAP CUSTOM chunk-", map_call["user"])
        self.assertEqual(reduce_call["system"], "REDUCE SYS")
        self.assertIn("REDUCE CUSTOM Summary", reduce_call["user"])
        self.assertEqual(final_call["system"], "FINAL SYS")
        self.assertIn("FINAL CUSTOM", final_call["user"])
        self.assertIn("Use owner fields.", final_call["user"])

    def test_selected_llm_model_is_forwarded_to_pipeline_calls(self) -> None:
        store = InMemorySummaryStore()
        llm = ModelAwareLLM()
        queue = ManualQueue()
        settings = replace(
            get_settings(),
            llm_model="llmgateway/free",
            llm_models=("llmgateway/free", "llmgateway/analysis"),
        )
        service = PipelineService(
            store=store,
            queue=queue,
            llm=llm,
            chunker=FakeChunker(["chunk-1"]),
            input_segmenter=FakeInputSegmenter(["input"]),
            settings=settings,
        )
        job_id = service.create_job(
            input_text="input",
            title="llm-model",
            metadata={"llm_model": "llmgateway/analysis"},
        )

        queue.advance_job(job_id)
        self.drain(queue, service)

        map_call = next(call for call in llm.calls if call["stage"] == Stage.MAP)
        final_call = next(call for call in llm.calls if call["stage"] == Stage.FINAL)
        self.assertEqual(map_call["model"], "llmgateway/analysis")
        self.assertEqual(final_call["model"], "llmgateway/analysis")

    def test_final_output_json_schema_allows_custom_final_json_object(self) -> None:
        store = InMemorySummaryStore()
        llm = CustomJsonLLM()
        queue = ManualQueue()
        service = PipelineService(
            store=store,
            queue=queue,
            llm=llm,
            chunker=FakeChunker(["chunk-1"]),
            input_segmenter=FakeInputSegmenter(["input"]),
        )
        job_id = service.create_job(
            input_text="input",
            title="custom-json",
            metadata={
                "output_json_schema": {
                    "headline": "string",
                    "items": [{"title": "string", "severity": "string"}],
                },
                "prompt_overrides": {
                    "final": {
                        "user": "CUSTOM JSON {output_json_schema} FROM {summaries}",
                    },
                },
            },
        )

        queue.advance_job(job_id)
        self.drain(queue, service)

        final_call = next(call for call in llm.calls if call["stage"] == Stage.FINAL)
        self.assertIn("CUSTOM JSON", final_call["user"])
        self.assertIn('"headline"', final_call["user"])
        self.assertIn('"type": "string"', final_call["user"])
        self.assertEqual(final_call["response_schema"]["type"], "object")
        self.assertEqual(final_call["response_schema"]["properties"]["headline"]["type"], "string")
        self.assertEqual(final_call["response_schema"]["properties"]["items"]["type"], "array")
        self.assertEqual(final_call["response_schema"]["properties"]["items"]["items"]["properties"]["severity"]["type"], "string")
        self.assertEqual(final_call["response_schema"]["required"], ["headline", "items"])
        self.assertFalse(final_call["response_schema"]["additionalProperties"])
        final_artifact = store.latest_artifact(job_id=job_id, artifact_type=ArtifactType.FINAL_SUMMARY)
        self.assertIsNotNone(final_artifact)
        payload = json.loads(final_artifact["content"])
        self.assertEqual(payload["headline"], "custom output")
        self.assertEqual(payload["items"][0]["severity"], "info")
        metadata = json.loads(final_artifact["metadata"])
        self.assertIn("output_json_schema", metadata)

    def test_intermediate_output_json_schema_applies_to_map_and_reduce_nodes(self) -> None:
        store = InMemorySummaryStore()
        llm = CustomJsonLLM()
        queue = ManualQueue()
        service = PipelineService(
            store=store,
            queue=queue,
            llm=llm,
            chunker=FakeChunker(["chunk-1", "chunk-2", "chunk-3"]),
            input_segmenter=FakeInputSegmenter(["input-1", "input-2", "input-3"]),
            settings=replace(
                get_settings(),
                reduce_group_size=2,
                llm_max_concurrency=2,
                max_enqueue_nodes_per_advance=100,
                chunk_target_estimated_tokens=100,
            ),
        )
        job_id = service.create_job(
            input_text="input",
            title="intermediate-json",
            metadata={
                "intermediate_output_json_schema": {
                    "chunk_summary": "string",
                    "key_points": ["string"],
                    "evidence": [{"title": "string", "details": "string"}],
                    "warnings": ["string"],
                },
            },
        )

        queue.advance_job(job_id)
        self.drain(queue, service)

        map_calls = [call for call in llm.calls if call["stage"] == Stage.MAP]
        reduce_calls = [call for call in llm.calls if str(call["stage"]).startswith(f"{Stage.REDUCE}_L")]
        self.assertTrue(map_calls)
        self.assertTrue(reduce_calls)
        self.assertTrue(all(call["response_schema"]["type"] == "object" for call in map_calls))
        self.assertTrue(all(call["response_schema"]["type"] == "object" for call in reduce_calls))
        self.assertIn('"chunk_summary"', map_calls[0]["user"])
        self.assertIn('"type": "string"', map_calls[0]["user"])
        self.assertIn('"chunk_summary"', reduce_calls[0]["user"])
        self.assertIn('"type": "string"', reduce_calls[0]["user"])

        map_artifact = store.latest_artifact(job_id=job_id, artifact_type=ArtifactType.MAP_SUMMARY)
        reduce_artifact = store.latest_artifact(job_id=job_id, artifact_type=ArtifactType.REDUCE_SUMMARY)
        assert map_artifact is not None
        assert reduce_artifact is not None
        map_metadata = json.loads(map_artifact["metadata"])
        reduce_metadata = json.loads(reduce_artifact["metadata"])
        self.assertIn("intermediate_output_json_schema", map_metadata)
        self.assertIn("intermediate_output_json_schema", reduce_metadata)
        self.assertEqual(json.loads(map_artifact["content"])["chunk_summary"].split(":")[0], str(Stage.MAP))

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

    def test_rerun_failed_node_requeues_selected_node(self) -> None:
        service, store, _, queue = self.make_service(chunks=["row 1", "row 2"])
        service.llm = FailingLLM("model returned invalid JSON")
        job_id = service.create_job(input_text="raw text", title="source", metadata={})

        queue.advance_job(job_id)
        service.advance_job(job_id)
        first_map = next(item for item in list(queue.items) if item[0] == "map")
        queue.items.remove(first_map)
        _, _, node_id = first_map
        assert node_id is not None

        service.map_node(job_id, node_id)
        self.assertEqual(store.get_job_current(job_id)["job_status"], JobStatus.FAILED)

        node_type, status, queued = service.rerun_node(job_id, node_id)

        self.assertEqual(node_type, NodeType.MAP)
        self.assertEqual(status, NodeStatus.QUEUED)
        self.assertTrue(queued)
        self.assertEqual(store.get_job_current(job_id)["job_status"], JobStatus.RUNNING)
        self.assertIn(("map", job_id, node_id), queue.items)

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

    def test_advance_fails_job_without_input_instead_of_retrying_forever(self) -> None:
        service, store, _, queue = self.make_service(chunks=["chunk-1"])
        store.insert_job_event(
            job_id="job_missing_input",
            event_type="JOB_CREATED",
            job_status=JobStatus.CREATED,
            actor="seed",
            payload={"source": "test"},
        )

        service.advance_job("job_missing_input")

        job = store.get_job_current("job_missing_input")
        self.assertEqual(job["job_status"], JobStatus.FAILED)
        self.assertEqual(job["last_event_type"], "JOB_FAILED")
        payload = json.loads(store.list_job_events("job_missing_input")[-1]["payload"])
        self.assertEqual(payload["error_class"], "missing_input")
        self.assertEqual(list(queue.items), [])

    def test_snapshot_contains_ui_read_model(self) -> None:
        service, store, _, queue = self.make_service(chunks=["chunk-1", "chunk-2"])
        job_id = service.create_job(input_text="input", title="snapshot", metadata={})

        queue.advance_job(job_id)
        self.drain(queue, service)

        snapshot = build_job_snapshot(service, job_id)

        self.assertEqual(snapshot["job"]["job_status"], JobStatus.DONE)
        self.assertIsNotNone(snapshot["job"]["created_at"])
        self.assertIsNotNone(snapshot["job"]["started_at"])
        self.assertIsNotNone(snapshot["job"]["finished_at"])
        self.assertLessEqual(snapshot["job"]["created_at"], snapshot["job"]["started_at"])
        self.assertLessEqual(snapshot["job"]["started_at"], snapshot["job"]["finished_at"])
        self.assertIn("nodes", snapshot)
        self.assertIn("node_links", snapshot)
        self.assertIn("artifacts", snapshot)
        self.assertIn("job_events", snapshot)
        self.assertIn("node_events", snapshot)
        self.assertIsNotNone(snapshot["final_artifact"])
        self.assertEqual(snapshot["artifact_counts"][ArtifactType.FINAL_SUMMARY], 1)
        self.assertEqual(snapshot["input_stats"]["segments_count"], 2)
        self.assertEqual(snapshot["input_stats"]["rows_count"], 2)
        self.assertGreater(snapshot["input_stats"]["chars"], 0)
        self.assertGreater(snapshot["input_stats"]["estimated_tokens"], 0)
        self.assertEqual(
            [(link["from_node_type"], link["to_node_type"]) for link in snapshot["node_links"]],
            [
                ("SOURCE", "MAP"),
                ("SOURCE", "MAP"),
                ("MAP", "REDUCE"),
                ("MAP", "REDUCE"),
                ("REDUCE", "FINAL"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
