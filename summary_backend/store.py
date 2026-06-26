"""ClickHouse persistence for summary jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .ids import sha256_text
from .input_models import InputSegment
from .logging_setup import get_logger
from .schemas import ArtifactType, JobStatus, NodeStatus, NodeType, Stage

logger = get_logger("store")


class ClickHouseStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import clickhouse_connect

            self._client = clickhouse_connect.get_client(
                host=self.settings.clickhouse_host,
                port=self.settings.clickhouse_port,
                username=self.settings.clickhouse_username,
                password=self.settings.clickhouse_password,
                database=self.settings.clickhouse_database,
                secure=self.settings.clickhouse_secure,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None

    def _table(self, name: str) -> str:
        return f"{self.settings.clickhouse_database}.{name}"

    @staticmethod
    def _json(payload: dict[str, Any] | None) -> str:
        return json.dumps(payload or {}, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _rows(result) -> list[dict[str, Any]]:
        return [dict(zip(result.column_names, row)) for row in result.result_rows]

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
        self.client.insert(
            self._table("summary_job_events"),
            [[job_id, event_type, str(job_status), actor, message, self._json(payload)]],
            column_names=["job_id", "event_type", "job_status", "actor", "message", "payload"],
        )
        logger.info(
            "job_event | job_id=%s event_type=%s status=%s actor=%s",
            job_id,
            event_type,
            job_status,
            actor,
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
        self.client.insert(
            self._table("summary_node_events"),
            [
                [
                    job_id,
                    node_id,
                    event_type,
                    str(node_status),
                    str(node_type),
                    level,
                    node_index,
                    attempt,
                    actor,
                    message,
                    self._json(payload),
                ]
            ],
            column_names=[
                "job_id",
                "node_id",
                "event_type",
                "node_status",
                "node_type",
                "level",
                "node_index",
                "attempt",
                "actor",
                "message",
                "payload",
            ],
        )
        logger.info(
            "node_event | job_id=%s node_id=%s type=%s level=%s index=%s event=%s status=%s",
            job_id,
            node_id,
            node_type,
            level,
            node_index,
            event_type,
            node_status,
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
        content_hash = sha256_text(content)
        self.client.insert(
            self._table("summary_artifacts"),
            [[job_id, node_id, str(artifact_type), str(stage), level, content_hash, content, self._json(metadata)]],
            column_names=[
                "job_id",
                "node_id",
                "artifact_type",
                "stage",
                "level",
                "content_hash",
                "content",
                "metadata",
            ],
        )
        logger.info(
            "artifact_saved | job_id=%s node_id=%s type=%s stage=%s level=%s hash=%s",
            job_id,
            node_id,
            artifact_type,
            stage,
            level,
            content_hash[:16],
        )
        return content_hash

    def insert_input_segments(self, *, job_id: str, segments: list[InputSegment]) -> None:
        if not segments:
            return
        self.client.insert(
            self._table("summary_input_segments"),
            [
                [
                    job_id,
                    segment.segment_index,
                    segment.source_type,
                    segment.source_format,
                    segment.content_hash,
                    segment.content,
                    segment.rows_count,
                    segment.chars,
                    self._json(dict(segment.metadata)),
                ]
                for segment in segments
            ],
            column_names=[
                "job_id",
                "segment_index",
                "source_type",
                "source_format",
                "content_hash",
                "content",
                "rows_count",
                "chars",
                "metadata",
            ],
        )
        logger.info(
            "input_segments_saved | job_id=%s count=%s first_index=%s last_index=%s",
            job_id,
            len(segments),
            segments[0].segment_index,
            segments[-1].segment_index,
        )

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
        self.client.insert(
            self._table("summary_llm_calls"),
            [
                [
                    job_id,
                    node_id,
                    provider,
                    model,
                    status,
                    error_class,
                    http_status,
                    latency_ms,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    sha256_text(request_json) if request_json else "",
                    sha256_text(response_json) if response_json else "",
                    request_json,
                    response_json,
                    error_message,
                ]
            ],
            column_names=[
                "job_id",
                "node_id",
                "provider",
                "model",
                "status",
                "error_class",
                "http_status",
                "latency_ms",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "request_hash",
                "response_hash",
                "request_json",
                "response_json",
                "error_message",
            ],
        )

    def get_job_current(self, job_id: str) -> dict[str, Any] | None:
        result = self.client.query(
            f"""
            SELECT job_id, job_status, last_event_type, updated_at, events_count
            FROM {self._table("summary_job_current_v")}
            WHERE job_id = %(job_id)s
            LIMIT 1
            """,
            parameters={"job_id": job_id},
        )
        rows = self._rows(result)
        return rows[0] if rows else None

    def list_jobs(self, limit: int = 200, status: str | None = None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: dict[str, Any] = {"limit": limit}
        if status:
            clauses.append("job_status = %(status)s")
            params["status"] = status
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        result = self.client.query(
            f"""
            SELECT job_id, job_status, last_event_type, updated_at, events_count
            FROM {self._table("summary_job_current_v")}
            {where}
            ORDER BY updated_at DESC
            LIMIT %(limit)s
            """,
            parameters=params,
        )
        return self._rows(result)

    def list_job_events(self, job_id: str, limit: int = 500) -> list[dict[str, Any]]:
        result = self.client.query(
            f"""
            SELECT
                toString(event_id) AS event_id,
                job_id,
                event_time,
                event_type,
                job_status AS status,
                actor,
                message,
                payload
            FROM {self._table("summary_job_events")}
            WHERE job_id = %(job_id)s
            ORDER BY event_time ASC, event_id ASC
            LIMIT %(limit)s
            """,
            parameters={"job_id": job_id, "limit": limit},
        )
        return self._rows(result)

    def list_node_events(self, job_id: str, limit: int = 1000) -> list[dict[str, Any]]:
        result = self.client.query(
            f"""
            SELECT
                toString(event_id) AS event_id,
                job_id,
                event_time,
                event_type,
                node_status AS status,
                actor,
                message,
                payload,
                node_id,
                node_type,
                level,
                node_index
            FROM {self._table("summary_node_events")}
            WHERE job_id = %(job_id)s
            ORDER BY event_time ASC, event_id ASC
            LIMIT %(limit)s
            """,
            parameters={"job_id": job_id, "limit": limit},
        )
        return self._rows(result)

    def list_recent_events(self, limit: int = 200) -> list[dict[str, Any]]:
        result = self.client.query(
            f"""
            SELECT *
            FROM
            (
                SELECT
                    toString(event_id) AS event_id,
                    'summary' AS service,
                    'JOB' AS scope,
                    job_id,
                    '' AS node_id,
                    event_time,
                    event_type,
                    job_status AS status,
                    actor,
                    message,
                    payload,
                    '' AS node_type,
                    0 AS level,
                    0 AS node_index
                FROM {self._table("summary_job_events")}
                UNION ALL
                SELECT
                    toString(event_id) AS event_id,
                    'summary' AS service,
                    'NODE' AS scope,
                    job_id,
                    node_id,
                    event_time,
                    event_type,
                    node_status AS status,
                    actor,
                    message,
                    payload,
                    node_type,
                    level,
                    node_index
                FROM {self._table("summary_node_events")}
            )
            ORDER BY event_time DESC, event_id DESC
            LIMIT {{limit:UInt32}}
            """,
            parameters={"limit": limit},
        )
        return self._rows(result)

    def list_nodes_current(self, job_id: str) -> list[dict[str, Any]]:
        result = self.client.query(
            f"""
            SELECT job_id, node_id, node_type, level, node_index, node_status, last_event_type, updated_at, events_count
            FROM {self._table("summary_node_current_v")}
            WHERE job_id = %(job_id)s
            ORDER BY level ASC, node_type ASC, node_index ASC
            """,
            parameters={"job_id": job_id},
        )
        return self._rows(result)

    def get_node_payload(self, job_id: str, node_id: str) -> dict[str, Any]:
        result = self.client.query(
            f"""
            SELECT payload
            FROM {self._table("summary_node_events")}
            WHERE job_id = %(job_id)s AND node_id = %(node_id)s
              AND payload != ''
              AND payload != '{{}}'
            ORDER BY
                multiIf(position(payload, '"input_node_ids"') > 0, 3, position(payload, '"chunk_hash"') > 0, 2, 0) DESC,
                event_time DESC,
                event_id DESC
            LIMIT 1
            """,
            parameters={"job_id": job_id, "node_id": node_id},
        )
        rows = self._rows(result)
        if not rows:
            return {}
        try:
            return json.loads(rows[0]["payload"] or "{}")
        except json.JSONDecodeError:
            return {}

    def latest_artifact(
        self,
        *,
        job_id: str,
        artifact_type: ArtifactType | str | None = None,
        node_id: str | None = None,
        stage: Stage | str | None = None,
        level: int | None = None,
    ) -> dict[str, Any] | None:
        clauses = ["job_id = %(job_id)s"]
        params: dict[str, Any] = {"job_id": job_id}
        if artifact_type is not None:
            clauses.append("artifact_type = %(artifact_type)s")
            params["artifact_type"] = str(artifact_type)
        if node_id is not None:
            clauses.append("node_id = %(node_id)s")
            params["node_id"] = node_id
        if stage is not None:
            clauses.append("stage = %(stage)s")
            params["stage"] = str(stage)
        if level is not None:
            clauses.append("level = %(level)s")
            params["level"] = level
        where = " AND ".join(clauses)
        result = self.client.query(
            f"""
            SELECT
                toString(artifact_id) AS artifact_id,
                job_id,
                node_id,
                artifact_type,
                stage,
                level,
                content_hash,
                content,
                metadata,
                created_at
            FROM {self._table("summary_artifacts")}
            WHERE {where}
            ORDER BY created_at DESC, artifact_id DESC
            LIMIT 1
            """,
            parameters=params,
        )
        rows = self._rows(result)
        return rows[0] if rows else None

    def list_artifacts(
        self,
        *,
        job_id: str,
        include_content: bool = False,
        artifact_type: ArtifactType | str | None = None,
        stage: Stage | str | None = None,
        level: int | None = None,
    ) -> list[dict[str, Any]]:
        columns = "toString(artifact_id) AS artifact_id, job_id, node_id, artifact_type, stage, level, content_hash, metadata, created_at"
        if include_content:
            columns = "toString(artifact_id) AS artifact_id, job_id, node_id, artifact_type, stage, level, content_hash, content, metadata, created_at"
        clauses = ["job_id = %(job_id)s"]
        params: dict[str, Any] = {"job_id": job_id}
        if artifact_type is not None:
            clauses.append("artifact_type = %(artifact_type)s")
            params["artifact_type"] = str(artifact_type)
        if stage is not None:
            clauses.append("stage = %(stage)s")
            params["stage"] = str(stage)
        if level is not None:
            clauses.append("level = %(level)s")
            params["level"] = level
        result = self.client.query(
            f"""
            SELECT {columns}
            FROM {self._table("summary_artifacts")}
            WHERE {" AND ".join(clauses)}
            ORDER BY level ASC, stage ASC, node_id ASC, created_at ASC
            """,
            parameters=params,
        )
        rows = self._rows(result)
        if not include_content:
            for row in rows:
                row["content"] = None
        return rows

    def list_input_segments(self, *, job_id: str, include_content: bool = False) -> list[dict[str, Any]]:
        columns = """
            job_id,
            segment_index,
            argMax(source_type, created_at) AS source_type,
            argMax(source_format, created_at) AS source_format,
            argMax(content_hash, created_at) AS content_hash,
            argMax(rows_count, created_at) AS rows_count,
            argMax(chars, created_at) AS chars,
            argMax(metadata, created_at) AS metadata,
            max(created_at) AS created_at_latest
        """
        if include_content:
            columns = """
                job_id,
                segment_index,
                argMax(source_type, created_at) AS source_type,
                argMax(source_format, created_at) AS source_format,
                argMax(content_hash, created_at) AS content_hash,
                argMax(content, created_at) AS content,
                argMax(rows_count, created_at) AS rows_count,
                argMax(chars, created_at) AS chars,
                argMax(metadata, created_at) AS metadata,
                max(created_at) AS created_at_latest
            """
        result = self.client.query(
            f"""
            SELECT {columns}
            FROM {self._table("summary_input_segments")}
            WHERE job_id = %(job_id)s
            GROUP BY job_id, segment_index
            ORDER BY segment_index ASC
            """,
            parameters={"job_id": job_id},
        )
        rows = self._rows(result)
        for row in rows:
            if "created_at_latest" in row:
                row["created_at"] = row.pop("created_at_latest")
        if not include_content:
            for row in rows:
                row["content"] = None
        return rows

    def count_input_segments(self, job_id: str) -> int:
        result = self.client.query(
            f"""
            SELECT countDistinct(segment_index)
            FROM {self._table("summary_input_segments")}
            WHERE job_id = %(job_id)s
            """,
            parameters={"job_id": job_id},
        )
        return int(result.result_rows[0][0])

    def list_staged_uploads(self, limit: int = 200) -> list[dict[str, Any]]:
        result = self.client.query(
            f"""
            SELECT
                e.job_id AS source_job_id,
                e.event_time AS staged_at,
                e.payload AS payload,
                c.job_status AS job_status
            FROM {self._table("summary_job_events")} AS e
            LEFT JOIN {self._table("summary_job_current_v")} AS c ON c.job_id = e.job_id
            WHERE e.event_type = 'FILE_STAGED'
            ORDER BY e.event_time DESC, e.event_id DESC
            LIMIT %(limit)s
            """,
            parameters={"limit": limit},
        )
        uploads: list[dict[str, Any]] = []
        for row in self._rows(result):
            try:
                payload = json.loads(str(row.get("payload") or "{}"))
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            if "reused_upload" in payload:
                continue
            source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
            staging = payload.get("staging") if isinstance(payload.get("staging"), dict) else {}
            path_text = str(staging.get("path") or "")
            path = Path(path_text)
            uploads.append(
                {
                    "upload_id": str(row["source_job_id"]),
                    "source_job_id": str(row["source_job_id"]),
                    "filename": str(source.get("filename") or ""),
                    "source_format": str(source.get("format") or ""),
                    "content_type": str(source.get("content_type") or ""),
                    "raw_line_column": str(source.get("raw_line_column") or ""),
                    "size_bytes": int(staging.get("size_bytes") or 0),
                    "available": bool(path_text and path.exists()),
                    "job_status": str(row.get("job_status") or ""),
                    "staged_at": row.get("staged_at"),
                }
            )
        return uploads

    def list_jobs_for_recovery(self) -> list[dict[str, Any]]:
        result = self.client.query(
            f"""
            SELECT job_id, job_status, last_event_type, updated_at, events_count
            FROM {self._table("summary_job_current_v")}
            WHERE job_status IN ('CREATED', 'INGESTING', 'INPUT_READY', 'RUNNING', 'RESUMED', 'WAITING_RETRY', 'WAITING_PROVIDER')
            ORDER BY updated_at ASC
            """
        )
        return self._rows(result)
