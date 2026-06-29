"""Ports used by the summary backend core.

The pipeline depends on these protocols, not on ClickHouse, Dramatiq, or a
specific LLM provider. Concrete adapters live in separate modules.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Protocol

from .input_models import InputSegment, LogRecord
from .schemas import ArtifactType, JobStatus, NodeStatus, NodeType, Stage, SummaryResult


class TaskQueue(Protocol):
    def ingest_upload(self, job_id: str) -> None:
        ...

    def advance_job(self, job_id: str) -> None:
        ...

    def map_node(self, job_id: str, node_id: str) -> None:
        ...

    def reduce_node(self, job_id: str, node_id: str) -> None:
        ...

    def finalize_job(self, job_id: str) -> None:
        ...


class SummaryStore(Protocol):
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
        ...

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
        ...

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
        ...

    def insert_input_segments(self, *, job_id: str, segments: list[InputSegment]) -> None:
        ...

    def insert_llm_call(
        self,
        *,
        job_id: str,
        node_id: str,
        provider: str,
        model: str,
        status: str,
        error_class: str = "",
        http_status: int = 0,
        latency_ms: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        request_json: str = "{}",
        response_json: str = "{}",
        error_message: str = "",
    ) -> None:
        ...

    def latest_llm_call(self, *, job_id: str, node_id: str) -> dict[str, Any] | None:
        ...

    def get_job_current(self, job_id: str) -> dict[str, Any] | None:
        ...

    def list_job_events(self, job_id: str, limit: int = 500) -> list[dict[str, Any]]:
        ...

    def list_node_events(self, job_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        ...

    def list_nodes_current(self, job_id: str) -> list[dict[str, Any]]:
        ...

    def get_node_payload(self, job_id: str, node_id: str) -> dict[str, Any]:
        ...

    def latest_artifact(
        self,
        *,
        job_id: str,
        artifact_type: ArtifactType | str | None = None,
        node_id: str | None = None,
        stage: Stage | str | None = None,
        level: int | None = None,
    ) -> dict[str, Any] | None:
        ...

    def list_artifacts(
        self,
        *,
        job_id: str,
        include_content: bool = False,
        artifact_type: ArtifactType | str | None = None,
        stage: Stage | str | None = None,
        level: int | None = None,
    ) -> list[dict[str, Any]]:
        ...

    def list_input_segments(self, *, job_id: str, include_content: bool = False) -> list[dict[str, Any]]:
        ...

    def count_input_segments(self, job_id: str) -> int:
        ...

    def list_staged_uploads(self, limit: int = 200) -> list[dict[str, Any]]:
        ...

    def list_jobs_for_recovery(self) -> list[dict[str, Any]]:
        ...


class SummaryLLM(Protocol):
    def call_summary(
        self,
        *,
        job_id: str,
        node_id: str,
        stage: str,
        system: str,
        user: str,
        model: str | None = None,
    ) -> SummaryResult:
        ...


class Chunker(Protocol):
    def build_chunks(self, text: str, target_estimated_tokens: int) -> list[str]:
        ...


class InputSegmenter(Protocol):
    def build_segments(
        self,
        records: Iterable[LogRecord],
        *,
        source_type: str,
        source_format: str,
        target_estimated_tokens: int,
    ) -> Iterator[InputSegment]:
        ...


class AuditSink(Protocol):
    def write_llm_call(
        self,
        *,
        job_id: str,
        node_id: str,
        stage: str,
        system: str,
        user: str,
        request_json: dict[str, Any],
        response_json: dict[str, Any] | None,
        content: str | None,
        error: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        ...
