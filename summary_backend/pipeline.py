"""Summary job pipeline service."""

from __future__ import annotations

import json
import inspect
from collections import Counter
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from .audit import AuditWriter
from .config import Settings, get_settings
from .errors import LlmPoolBusyError, classify_error
from .ids import make_node_id, new_job_id, sha256_text
from .input_models import InputSegment, LogRecord
from .input_segments import RowBudgetInputSegmenter
from .llm_client import StructuredLLMClient
from .logging_setup import get_logger, log_kv, log_stage, timed_stage
from .ports import Chunker, InputSegmenter, SummaryLLM, SummaryStore, TaskQueue
from .schemas import ArtifactType, JobStatus, JsonObjectResult, NodeStatus, NodeType, Stage, SummaryResult
from .store import ClickHouseStore
from .text import CharBudgetChunker, split_atomic_units

logger = get_logger("pipeline")


class MissingSummaryInputError(RuntimeError):
    pass


SUMMARY_RESULT_SCHEMA_INSTRUCTION = (
    "Return exactly one JSON object with this schema and no alternate top-level keys: "
    '{"ok": true, "summary": "string", "key_points": ["string"], "warnings": ["string"], "source_count": 1}. '
    "Put all narrative content inside summary and key_points."
)


MAP_SYSTEM = (
    "You summarize one part of a large context. "
    f"{SUMMARY_RESULT_SCHEMA_INSTRUCTION} "
    "Keep the summary concise, factual, and grounded in the provided text."
)

REDUCE_SYSTEM = (
    "You merge several partial summaries into one consolidated summary. "
    f"{SUMMARY_RESULT_SCHEMA_INSTRUCTION} "
    "Preserve important facts and remove duplicates."
)

FINAL_SYSTEM = (
    "You produce the final user-facing report from the consolidated context. "
    f"{SUMMARY_RESULT_SCHEMA_INSTRUCTION} "
    "Put the complete report in summary. Use key_points and warnings only when the user explicitly requested those fields. "
    "Choose clear headings, bullets, or prose inside summary when no specific report structure was requested. "
    "Be concise and explicit about uncertainty."
)

CUSTOM_JSON_FINAL_SYSTEM = (
    "You produce the final user-facing JSON object from the consolidated context. "
    "Return exactly one valid JSON object. Do not wrap it in Markdown and do not add prose outside JSON. "
    "Match the user-provided output JSON structure as closely as possible."
)

MAP_USER_TEMPLATE = "Summarize this chunk:\n\n{chunk}"
REDUCE_USER_TEMPLATE = "Merge these summaries:\n\n{summaries}"
FINAL_USER_TEMPLATE = (
    "Create the final user-facing report from these summaries. "
    "If the user did not request a specific structure, choose the most useful report structure yourself. "
    "Return the transport JSON schema, but put the whole report in the summary string.\n\n"
    "{summaries}"
)
FINAL_CUSTOM_JSON_USER_TEMPLATE = (
    "Create the final JSON response from these summaries.\n\n"
    "Output JSON structure to follow:\n{output_json_schema}\n\n"
    "Additional report/prompt instruction:\n{report_format_instruction}\n\n"
    "Summaries:\n\n{summaries}"
)

REPORT_FORMAT_INSTRUCTIONS = {
    "default": "Write a concise operational summary in clear prose.",
    "incident_report": (
        "Write the summary as an incident report. Include compact sections inside the summary text: "
        "Overview, impact, timeline, likely cause, mitigation, follow-ups, and open questions."
    ),
    "executive_summary": (
        "Write for leadership. Focus on business impact, current status, risk, decisions needed, "
        "and avoid low-level log detail unless it materially changes the conclusion."
    ),
    "technical_rca": (
        "Write as a technical RCA. Emphasize symptoms, evidence, causal chain, affected components, "
        "remediation, verification, and remaining uncertainty."
    ),
    "bullet_points": (
        "Write the summary as dense bullet-style findings. Keep each bullet factual and action-oriented."
    ),
    "markdown": (
        "Use Markdown headings and bullets inside the summary string. Keep the response JSON schema unchanged."
    ),
}


class PipelineService:
    def __init__(
        self,
        *,
        store: SummaryStore | None = None,
        queue: TaskQueue | None = None,
        llm: SummaryLLM | None = None,
        chunker: Chunker | None = None,
        input_segmenter: InputSegmenter | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or ClickHouseStore(self.settings)
        self.queue = queue
        self.chunker = chunker or CharBudgetChunker()
        self.input_segmenter = input_segmenter or RowBudgetInputSegmenter()
        self.audit = AuditWriter(self.settings)
        self.llm = llm or StructuredLLMClient(store=self.store, settings=self.settings, audit=self.audit)

    def create_job(self, *, input_text: str, title: str | None, metadata: dict[str, Any] | None) -> str:
        job_id = new_job_id()
        payload = {
            "title": title or "",
            "metadata": metadata or {},
            "source": {"kind": "text", "format": "plain_text"},
        }
        self.store.insert_job_event(
            job_id=job_id,
            event_type="JOB_CREATED",
            job_status=JobStatus.CREATED,
            actor="api",
            message=title or "",
            payload=payload,
        )
        records = (LogRecord(raw_line=unit) for unit in split_atomic_units(input_text))
        segments = list(
            self.input_segmenter.build_segments(
                records,
                source_type="text",
                source_format="plain_text",
                target_estimated_tokens=self.settings.chunk_target_estimated_tokens,
            )
        )
        if not segments:
            raise ValueError("input_text did not produce input segments")
        self.store.insert_input_segments(job_id=job_id, segments=segments)
        manifest = {
            "source": payload["source"],
            "segments_count": len(segments),
            "rows_count": sum(segment.rows_count for segment in segments),
            "chars": len(input_text),
        }
        self.store.insert_artifact(
            job_id=job_id,
            node_id="",
            artifact_type=ArtifactType.INPUT,
            stage=Stage.INPUT,
            level=0,
            content=(
                f"text input manifest: chars={len(input_text)}, "
                f"rows={manifest['rows_count']}, segments={manifest['segments_count']}"
            ),
            metadata={**payload, "manifest": manifest},
        )
        self.store.insert_job_event(
            job_id=job_id,
            event_type="INPUT_INGESTED",
            job_status=JobStatus.CREATED,
            actor="api",
            payload=manifest,
        )
        logger.info(
            "job_created | job_id=%s source=text chars=%s rows=%s segments=%s",
            job_id,
            len(input_text),
            manifest["rows_count"],
            manifest["segments_count"],
        )
        return job_id

    def rerun_job(self, source_job_id: str, *, auto_start: bool = True) -> tuple[str, bool]:
        source_job = self.store.get_job_current(source_job_id)
        if source_job is None:
            raise KeyError(source_job_id)

        source_segments = self.store.list_input_segments(job_id=source_job_id, include_content=True)
        if not source_segments:
            raise ValueError(f"source job has no persisted input segments: {source_job_id}")

        job_id = new_job_id()
        title = f"Rerun of {source_job_id}"
        source_metadata = self._job_metadata(source_job_id)
        payload = {
            "title": title,
            "metadata": {
                "rerun_source_job_id": source_job_id,
                "rerun_source_status": str(source_job.get("job_status") or ""),
                **self._report_format_metadata(source_metadata),
            },
            "source": {"kind": "rerun", "source_job_id": source_job_id},
        }

        segments: list[InputSegment] = []
        for row in source_segments:
            content = row.get("content")
            if content is None:
                raise ValueError(f"source job has an input segment without content: {source_job_id}")
            segment_index = int(row["segment_index"])
            metadata = self._safe_json_loads(str(row.get("metadata") or "{}"))
            segments.append(
                InputSegment(
                    segment_index=segment_index,
                    source_type=str(row.get("source_type") or "rerun"),
                    source_format=str(row.get("source_format") or ""),
                    content=str(content),
                    rows_count=int(row.get("rows_count") or 0),
                    metadata={
                        **metadata,
                        "rerun_source_job_id": source_job_id,
                        "rerun_source_segment_index": segment_index,
                    },
                )
            )

        self.store.insert_job_event(
            job_id=job_id,
            event_type="JOB_CREATED",
            job_status=JobStatus.CREATED,
            actor="api",
            message=title,
            payload=payload,
        )
        self.store.insert_input_segments(job_id=job_id, segments=segments)
        manifest = {
            "source": payload["source"],
            "source_job_id": source_job_id,
            "segments_count": len(segments),
            "rows_count": sum(segment.rows_count for segment in segments),
            "chars": sum(segment.chars for segment in segments),
        }
        self.store.insert_artifact(
            job_id=job_id,
            node_id="",
            artifact_type=ArtifactType.INPUT,
            stage=Stage.INPUT,
            level=0,
            content=(
                f"rerun input manifest: source_job_id={source_job_id}, "
                f"rows={manifest['rows_count']}, segments={manifest['segments_count']}"
            ),
            metadata={**payload, "manifest": manifest},
        )
        self.store.insert_job_event(
            job_id=job_id,
            event_type="INPUT_INGESTED",
            job_status=JobStatus.CREATED,
            actor="api",
            payload=manifest,
        )

        queued = False
        if auto_start and self.queue is not None:
            self.queue.advance_job(job_id)
            queued = True
        logger.info(
            "job_rerun_created | source_job_id=%s job_id=%s rows=%s segments=%s queued=%s",
            source_job_id,
            job_id,
            manifest["rows_count"],
            manifest["segments_count"],
            queued,
        )
        return job_id, queued

    def rerun_node(self, job_id: str, node_id: str) -> tuple[NodeType, NodeStatus, bool]:
        job = self.store.get_job_current(job_id)
        if job is None:
            raise KeyError(job_id)
        node = self._node_by_id(job_id, node_id)
        node_type = NodeType(str(node["node_type"]))
        if node_type not in {NodeType.MAP, NodeType.REDUCE, NodeType.FINAL}:
            raise ValueError(f"node cannot be rerun: {node_id}")

        existing_artifact_type = {
            NodeType.MAP: ArtifactType.MAP_SUMMARY,
            NodeType.REDUCE: ArtifactType.REDUCE_SUMMARY,
            NodeType.FINAL: ArtifactType.FINAL_SUMMARY,
        }[node_type]
        if self.store.latest_artifact(job_id=job_id, node_id=node_id, artifact_type=existing_artifact_type):
            raise ValueError(f"completed node already has an output artifact and cannot be rerun in place: {node_id}")

        payload = self.store.get_node_payload(job_id, node_id)
        request_payload = {
            "node_id": node_id,
            "node_type": str(node_type),
            "previous_status": str(node.get("node_status") or ""),
            "previous_event_type": str(node.get("last_event_type") or ""),
        }
        self.store.insert_job_event(
            job_id=job_id,
            event_type="JOB_NODE_RERUN_REQUESTED",
            job_status=JobStatus.RUNNING,
            actor="api",
            message=node_id,
            payload=request_payload,
        )
        self.store.insert_node_event(
            job_id=job_id,
            node_id=node_id,
            event_type="NODE_RERUN_REQUESTED",
            node_status=NodeStatus.PENDING,
            node_type=node_type,
            level=int(node["level"]),
            node_index=int(node["node_index"]),
            actor="api",
            message="Node rerun requested",
            payload={**payload, **request_payload},
        )

        queued = False
        if self.queue is None:
            if node_type == NodeType.MAP:
                self.map_node(job_id, node_id)
            elif node_type == NodeType.REDUCE:
                self.reduce_node(job_id, node_id)
            else:
                self.finalize_job(job_id)
        else:
            self.store.insert_node_event(
                job_id=job_id,
                node_id=node_id,
                event_type="NODE_ENQUEUED",
                node_status=NodeStatus.QUEUED,
                node_type=node_type,
                level=int(node["level"]),
                node_index=int(node["node_index"]),
                actor="api",
                message="Node rerun queued",
                payload={**payload, **request_payload},
            )
            if node_type == NodeType.MAP:
                self.queue.map_node(job_id, node_id)
            elif node_type == NodeType.REDUCE:
                self.queue.reduce_node(job_id, node_id)
            else:
                self.queue.finalize_job(job_id)
            queued = True

        log_kv(logger, "node_rerun_requested", job_id=job_id, node_id=node_id, node_type=node_type, queued=queued)
        return node_type, NodeStatus.QUEUED if queued else NodeStatus.PENDING, queued

    def pause_job(self, job_id: str) -> None:
        self.store.insert_job_event(
            job_id=job_id,
            event_type="JOB_PAUSE_REQUESTED",
            job_status=JobStatus.PAUSE_REQUESTED,
            actor="api",
        )

    def resume_job(self, job_id: str) -> None:
        self.store.insert_job_event(
            job_id=job_id,
            event_type="JOB_RESUMED",
            job_status=JobStatus.RESUMED,
            actor="api",
        )
        if self.queue is not None:
            self.queue.advance_job(job_id)

    def cancel_job(self, job_id: str) -> None:
        self.store.insert_job_event(
            job_id=job_id,
            event_type="JOB_CANCEL_REQUESTED",
            job_status=JobStatus.CANCEL_REQUESTED,
            actor="api",
        )

    def advance_job(self, job_id: str) -> None:
        with timed_stage(logger, "advance_job", job_id=job_id):
            if self._stop_if_not_runnable(job_id):
                return

            self.store.insert_job_event(
                job_id=job_id,
                event_type="JOB_RUNNING",
                job_status=JobStatus.RUNNING,
                actor="advance_job",
            )

            nodes = self.store.list_nodes_current(job_id)
            if not nodes:
                if not self._has_map_source_input(job_id):
                    self._fail_job(
                        job_id=job_id,
                        actor="advance_job",
                        exc=MissingSummaryInputError(f"input artifact or input segments not found for job_id={job_id}"),
                        payload={"error_class": "missing_input", "stage": "create_map_nodes"},
                    )
                    return
                self._create_map_nodes(job_id)
                nodes = self.store.list_nodes_current(job_id)

            if self._enqueue_pending(job_id, nodes, NodeType.MAP):
                return

            map_nodes = [node for node in nodes if node["node_type"] == NodeType.MAP]
            if not self._all_done(map_nodes):
                log_kv(logger, "advance_job.wait_map", job_id=job_id)
                return

            reduce_nodes = [node for node in nodes if node["node_type"] == NodeType.REDUCE]
            if not reduce_nodes:
                self._create_reduce_level_from_stage(job_id, source_type=ArtifactType.MAP_SUMMARY, level=1)
                nodes = self.store.list_nodes_current(job_id)
                if self._enqueue_pending(job_id, nodes, NodeType.REDUCE):
                    return

            latest_reduce_level = self._latest_reduce_level(nodes)
            if latest_reduce_level > 0:
                current_reduce = [
                    node
                    for node in nodes
                    if node["node_type"] == NodeType.REDUCE and int(node["level"]) == latest_reduce_level
                ]
                if self._enqueue_pending(job_id, current_reduce, NodeType.REDUCE):
                    return
                if not self._all_done(current_reduce):
                    log_kv(logger, "advance_job.wait_reduce", job_id=job_id, level=latest_reduce_level)
                    return

                outputs = self.store.list_artifacts(
                    job_id=job_id,
                    include_content=False,
                    artifact_type=ArtifactType.REDUCE_SUMMARY,
                    stage=Stage.REDUCE,
                    level=latest_reduce_level,
                )
                if len(outputs) > 1:
                    self._create_reduce_level_from_artifacts(job_id, outputs, latest_reduce_level + 1)
                    nodes = self.store.list_nodes_current(job_id)
                    self._enqueue_pending(job_id, nodes, NodeType.REDUCE)
                    return

            final = self.store.latest_artifact(job_id=job_id, artifact_type=ArtifactType.FINAL_SUMMARY)
            if final is not None:
                self.store.insert_job_event(
                    job_id=job_id,
                    event_type="JOB_DONE",
                    job_status=JobStatus.DONE,
                    actor="advance_job",
                    payload={"duration_ms": self._job_elapsed_ms(job_id)},
                )
                return

            if self.queue is None:
                self.finalize_job(job_id)
            else:
                self.queue.finalize_job(job_id)

    def map_node(self, job_id: str, node_id: str) -> None:
        with timed_stage(logger, "map_node", job_id=job_id, node_id=node_id):
            if self._stop_if_not_runnable(job_id):
                return
            if self.store.latest_artifact(job_id=job_id, node_id=node_id, artifact_type=ArtifactType.MAP_SUMMARY):
                self._skip_done(job_id, node_id, NodeType.MAP)
                return

            node = self._node_by_id(job_id, node_id)
            self.store.insert_node_event(
                job_id=job_id,
                node_id=node_id,
                event_type="NODE_STARTED",
                node_status=NodeStatus.RUNNING,
                node_type=NodeType.MAP,
                level=int(node["level"]),
                node_index=int(node["node_index"]),
                actor="map_node",
            )
            chunk = self.store.latest_artifact(job_id=job_id, node_id=node_id, artifact_type=ArtifactType.CHUNK)
            if chunk is None:
                raise RuntimeError(f"chunk artifact not found for node_id={node_id}")

            prompts = self._resolve_prompt_overrides(job_id)
            llm_model = self._resolve_llm_model(job_id)
            system = self._stage_system_prompt(prompts, "map", MAP_SYSTEM)
            user = self._stage_user_prompt(
                prompts,
                "map",
                MAP_USER_TEMPLATE,
                {"chunk": str(chunk["content"] or "")},
            )
            node_started = perf_counter()
            try:
                result = self._call_summary_llm(
                    job_id=job_id,
                    node_id=node_id,
                    stage=str(Stage.MAP),
                    system=system,
                    user=user,
                    model=llm_model,
                )
            except LlmPoolBusyError as exc:
                self._defer_node_for_llm_pool(
                    job_id=job_id,
                    node_id=node_id,
                    node_type=NodeType.MAP,
                    level=int(node["level"]),
                    node_index=int(node["node_index"]),
                    actor="map_node",
                    exc=exc,
                )
                raise
            except Exception as exc:
                duration_ms = int((perf_counter() - node_started) * 1000)
                self._fail_node_and_job(
                    job_id=job_id,
                    node_id=node_id,
                    node_type=NodeType.MAP,
                    level=int(node["level"]),
                    node_index=int(node["node_index"]),
                    actor="map_node",
                    exc=exc,
                    duration_ms=duration_ms,
                    llm_model=llm_model,
                )
                return
            duration_ms = int((perf_counter() - node_started) * 1000)
            self.store.insert_artifact(
                job_id=job_id,
                node_id=node_id,
                artifact_type=ArtifactType.MAP_SUMMARY,
                stage=Stage.MAP,
                level=0,
                content=result.model_dump_json(indent=2),
                metadata={"chunk_hash": chunk["content_hash"]},
            )
            self.store.insert_node_event(
                job_id=job_id,
                node_id=node_id,
                event_type="NODE_DONE",
                node_status=NodeStatus.DONE,
                node_type=NodeType.MAP,
                level=int(node["level"]),
                node_index=int(node["node_index"]),
                actor="map_node",
                payload=self._node_timing_payload(
                    job_id=job_id,
                    node_id=node_id,
                    node_type=NodeType.MAP,
                    duration_ms=duration_ms,
                    llm_model=llm_model,
                ),
            )
            self._advance_or_inline(job_id)

    def reduce_node(self, job_id: str, node_id: str) -> None:
        with timed_stage(logger, "reduce_node", job_id=job_id, node_id=node_id):
            if self._stop_if_not_runnable(job_id):
                return
            if self.store.latest_artifact(job_id=job_id, node_id=node_id, artifact_type=ArtifactType.REDUCE_SUMMARY):
                self._skip_done(job_id, node_id, NodeType.REDUCE)
                return

            node = self._node_by_id(job_id, node_id)
            payload = self.store.get_node_payload(job_id, node_id)
            input_node_ids = payload.get("input_node_ids") or []
            input_summaries = self._load_summary_artifacts(job_id, input_node_ids)

            self.store.insert_node_event(
                job_id=job_id,
                node_id=node_id,
                event_type="NODE_STARTED",
                node_status=NodeStatus.RUNNING,
                node_type=NodeType.REDUCE,
                level=int(node["level"]),
                node_index=int(node["node_index"]),
                actor="reduce_node",
                payload=payload,
            )
            prompts = self._resolve_prompt_overrides(job_id)
            llm_model = self._resolve_llm_model(job_id)
            system = self._stage_system_prompt(prompts, "reduce", REDUCE_SYSTEM)
            summaries_text = self._format_reduce_input_summaries(input_summaries)
            user = self._stage_user_prompt(
                prompts,
                "reduce",
                REDUCE_USER_TEMPLATE,
                {"summaries": summaries_text},
            )
            node_started = perf_counter()
            try:
                result = self._call_summary_llm(
                    job_id=job_id,
                    node_id=node_id,
                    stage=f"{Stage.REDUCE}_L{node['level']}",
                    system=system,
                    user=user,
                    model=llm_model,
                )
            except LlmPoolBusyError as exc:
                self._defer_node_for_llm_pool(
                    job_id=job_id,
                    node_id=node_id,
                    node_type=NodeType.REDUCE,
                    level=int(node["level"]),
                    node_index=int(node["node_index"]),
                    actor="reduce_node",
                    exc=exc,
                )
                raise
            except Exception as exc:
                duration_ms = int((perf_counter() - node_started) * 1000)
                self._fail_node_and_job(
                    job_id=job_id,
                    node_id=node_id,
                    node_type=NodeType.REDUCE,
                    level=int(node["level"]),
                    node_index=int(node["node_index"]),
                    actor="reduce_node",
                    exc=exc,
                    duration_ms=duration_ms,
                    llm_model=llm_model,
                )
                return
            duration_ms = int((perf_counter() - node_started) * 1000)
            self.store.insert_artifact(
                job_id=job_id,
                node_id=node_id,
                artifact_type=ArtifactType.REDUCE_SUMMARY,
                stage=Stage.REDUCE,
                level=int(node["level"]),
                content=result.model_dump_json(indent=2),
                metadata={"input_node_ids": input_node_ids},
            )
            self.store.insert_node_event(
                job_id=job_id,
                node_id=node_id,
                event_type="NODE_DONE",
                node_status=NodeStatus.DONE,
                node_type=NodeType.REDUCE,
                level=int(node["level"]),
                node_index=int(node["node_index"]),
                actor="reduce_node",
                payload=self._node_timing_payload(
                    job_id=job_id,
                    node_id=node_id,
                    node_type=NodeType.REDUCE,
                    duration_ms=duration_ms,
                    llm_model=llm_model,
                    payload=payload,
                ),
            )
            self._advance_or_inline(job_id)

    def finalize_job(self, job_id: str) -> None:
        with timed_stage(logger, "finalize_job", job_id=job_id):
            if self._stop_if_not_runnable(job_id):
                return
            if self.store.latest_artifact(job_id=job_id, artifact_type=ArtifactType.FINAL_SUMMARY):
                self.store.insert_job_event(
                    job_id=job_id,
                    event_type="JOB_DONE",
                    job_status=JobStatus.DONE,
                    actor="finalize_job",
                    payload={"duration_ms": self._job_elapsed_ms(job_id)},
                )
                return

            nodes = self.store.list_nodes_current(job_id)
            latest_reduce_level = self._latest_reduce_level(nodes)
            if latest_reduce_level > 0:
                inputs = self.store.list_artifacts(
                    job_id=job_id,
                    include_content=True,
                    artifact_type=ArtifactType.REDUCE_SUMMARY,
                    stage=Stage.REDUCE,
                    level=latest_reduce_level,
                )
            else:
                inputs = self.store.list_artifacts(
                    job_id=job_id,
                    include_content=True,
                    artifact_type=ArtifactType.MAP_SUMMARY,
                    stage=Stage.MAP,
                )
            if not inputs:
                raise RuntimeError(f"no summary artifacts found for finalization: job_id={job_id}")

            input_text = "\n\n".join(str(item["content"]) for item in inputs)
            final_node_id = make_node_id(
                job_id=job_id,
                node_type=NodeType.FINAL,
                level=0,
                index=0,
                input_hash=sha256_text(input_text),
            )
            report_format = self._resolve_report_format(job_id)
            prompt_overrides = self._resolve_prompt_overrides(job_id)
            output_json_schema = self._resolve_output_json_schema(job_id)
            final_payload = {
                "input_count": len(inputs),
                **report_format,
                **({"output_json_schema": output_json_schema} if output_json_schema else {}),
                **({"prompt_overrides": prompt_overrides} if prompt_overrides else {}),
            }
            self.store.insert_node_event(
                job_id=job_id,
                node_id=final_node_id,
                event_type="NODE_STARTED",
                node_status=NodeStatus.RUNNING,
                node_type=NodeType.FINAL,
                level=0,
                node_index=0,
                actor="finalize_job",
                payload=final_payload,
            )
            system = self._stage_system_prompt(
                prompt_overrides,
                "final",
                CUSTOM_JSON_FINAL_SYSTEM if output_json_schema else FINAL_SYSTEM,
            )
            llm_model = self._resolve_llm_model(job_id)
            user = self._build_final_user(
                input_text=input_text,
                report_format=report_format,
                prompt_overrides=prompt_overrides,
                output_json_schema=output_json_schema,
            )
            node_started = perf_counter()
            try:
                result = self._call_final_llm(
                    job_id=job_id,
                    node_id=final_node_id,
                    system=system,
                    user=user,
                    output_json_schema=output_json_schema,
                    model=llm_model,
                )
            except LlmPoolBusyError as exc:
                self._defer_node_for_llm_pool(
                    job_id=job_id,
                    node_id=final_node_id,
                    node_type=NodeType.FINAL,
                    level=0,
                    node_index=0,
                    actor="finalize_job",
                    exc=exc,
                )
                raise
            except Exception as exc:
                duration_ms = int((perf_counter() - node_started) * 1000)
                self._fail_node_and_job(
                    job_id=job_id,
                    node_id=final_node_id,
                    node_type=NodeType.FINAL,
                    level=0,
                    node_index=0,
                    actor="finalize_job",
                    exc=exc,
                    duration_ms=duration_ms,
                    llm_model=llm_model,
                )
                return
            duration_ms = int((perf_counter() - node_started) * 1000)
            self.store.insert_artifact(
                job_id=job_id,
                node_id=final_node_id,
                artifact_type=ArtifactType.FINAL_SUMMARY,
                stage=Stage.FINAL,
                level=0,
                content=result.model_dump_json(indent=2),
                metadata=final_payload,
            )
            self.store.insert_node_event(
                job_id=job_id,
                node_id=final_node_id,
                event_type="NODE_DONE",
                node_status=NodeStatus.DONE,
                node_type=NodeType.FINAL,
                level=0,
                node_index=0,
                actor="finalize_job",
                payload=self._node_timing_payload(
                    job_id=job_id,
                    node_id=final_node_id,
                    node_type=NodeType.FINAL,
                    duration_ms=duration_ms,
                    llm_model=llm_model,
                    payload=final_payload,
                ),
            )
            self.store.insert_job_event(
                job_id=job_id,
                event_type="JOB_DONE",
                job_status=JobStatus.DONE,
                actor="finalize_job",
                payload={"duration_ms": self._job_elapsed_ms(job_id)},
            )

    def recover_jobs(self) -> list[str]:
        jobs = self.store.list_jobs_for_recovery()
        job_ids = [str(job["job_id"]) for job in jobs]
        for job in jobs:
            job_id = str(job["job_id"])
            if self.queue is not None:
                if str(job["job_status"]) == JobStatus.INGESTING:
                    self.queue.ingest_upload(job_id)
                else:
                    self.queue.advance_job(job_id)
        logger.info("recover_jobs | queued=%s", len(job_ids))
        return job_ids

    def get_status(self, job_id: str) -> dict[str, Any]:
        job = self.store.get_job_current(job_id)
        if job is None:
            raise KeyError(job_id)
        nodes = self.store.list_nodes_current(job_id)
        artifacts = self.store.list_artifacts(job_id=job_id, include_content=False)
        return {
            "job": job,
            "node_counts": dict(Counter(str(node["node_status"]) for node in nodes)),
            "artifact_counts": dict(Counter(str(artifact["artifact_type"]) for artifact in artifacts)),
        }

    def _create_map_nodes(self, job_id: str) -> None:
        log_stage(logger, f"CREATE MAP NODES job_id={job_id}")
        input_segments = self.store.list_input_segments(job_id=job_id, include_content=True)
        if input_segments:
            self._create_map_nodes_from_input_segments(job_id, input_segments)
            return

        input_artifact = self.store.latest_artifact(job_id=job_id, artifact_type=ArtifactType.INPUT)
        if input_artifact is None:
            raise RuntimeError(f"input artifact not found for job_id={job_id}")

        chunks = self.chunker.build_chunks(
            str(input_artifact["content"]),
            target_estimated_tokens=self.settings.chunk_target_estimated_tokens,
        )
        for index, chunk in enumerate(chunks):
            chunk_hash = sha256_text(chunk)
            node_id = make_node_id(
                job_id=job_id,
                node_type=NodeType.MAP,
                level=0,
                index=index,
                input_hash=chunk_hash,
            )
            self.store.insert_artifact(
                job_id=job_id,
                node_id=node_id,
                artifact_type=ArtifactType.CHUNK,
                stage=Stage.CHUNK,
                level=0,
                content=chunk,
                metadata={"chunk_index": index, "chunks_total": len(chunks)},
            )
            self.store.insert_node_event(
                job_id=job_id,
                node_id=node_id,
                event_type="NODE_PENDING",
                node_status=NodeStatus.PENDING,
                node_type=NodeType.MAP,
                level=0,
                node_index=index,
                actor="advance_job",
                payload={"chunk_hash": chunk_hash},
            )

    def _create_map_nodes_from_input_segments(self, job_id: str, segments: list[dict[str, Any]]) -> None:
        total = len(segments)
        for row in segments:
            index = int(row["segment_index"])
            chunk = str(row["content"])
            chunk_hash = str(row["content_hash"])
            node_id = make_node_id(
                job_id=job_id,
                node_type=NodeType.MAP,
                level=0,
                index=index,
                input_hash=chunk_hash,
            )
            metadata = self._safe_json_loads(str(row.get("metadata") or "{}"))
            self.store.insert_artifact(
                job_id=job_id,
                node_id=node_id,
                artifact_type=ArtifactType.CHUNK,
                stage=Stage.CHUNK,
                level=0,
                content=chunk,
                metadata={
                    "chunk_index": index,
                    "chunks_total": total,
                    "source_type": row.get("source_type") or "",
                    "source_format": row.get("source_format") or "",
                    "segment_index": index,
                    "rows_count": int(row.get("rows_count") or 0),
                    "segment_metadata": metadata,
                },
            )
            self.store.insert_node_event(
                job_id=job_id,
                node_id=node_id,
                event_type="NODE_PENDING",
                node_status=NodeStatus.PENDING,
                node_type=NodeType.MAP,
                level=0,
                node_index=index,
                actor="advance_job",
                payload={
                    "chunk_hash": chunk_hash,
                    "segment_index": index,
                    "rows_count": int(row.get("rows_count") or 0),
                },
            )

    def _create_reduce_level_from_stage(self, job_id: str, *, source_type: ArtifactType, level: int) -> None:
        artifacts = self.store.list_artifacts(job_id=job_id, include_content=False, artifact_type=source_type)
        self._create_reduce_level_from_artifacts(job_id, artifacts, level)

    def _create_reduce_level_from_artifacts(self, job_id: str, artifacts: list[dict[str, Any]], level: int) -> None:
        if len(artifacts) <= 1:
            return
        group_size = max(2, self.settings.reduce_group_size)
        groups = [artifacts[i : i + group_size] for i in range(0, len(artifacts), group_size)]
        log_stage(logger, f"CREATE REDUCE LEVEL {level} job_id={job_id} groups={len(groups)}")
        for index, group in enumerate(groups):
            input_node_ids = [str(item["node_id"]) for item in group]
            input_hash = sha256_text("|".join(str(item["content_hash"]) for item in group))
            node_id = make_node_id(
                job_id=job_id,
                node_type=NodeType.REDUCE,
                level=level,
                index=index,
                input_hash=input_hash,
            )
            payload = {
                "input_node_ids": input_node_ids,
                "input_hash": input_hash,
                "input_count": len(input_node_ids),
            }
            self.store.insert_node_event(
                job_id=job_id,
                node_id=node_id,
                event_type="NODE_PENDING",
                node_status=NodeStatus.PENDING,
                node_type=NodeType.REDUCE,
                level=level,
                node_index=index,
                actor="advance_job",
                payload=payload,
            )

    def _enqueue_pending(self, job_id: str, nodes: list[dict[str, Any]], node_type: NodeType) -> bool:
        pending = [
            node
            for node in nodes
            if node["node_type"] == node_type
            and (
                node["node_status"] in {NodeStatus.PENDING, NodeStatus.WAITING_RETRY}
                or self._is_waiting_provider_ready_node(node)
                or self._is_stale_queued_node(node)
                or self._is_stale_running_node(node)
            )
        ]
        if not pending:
            return False
        active_count = sum(
            1
            for node in nodes
            if node["node_type"] == node_type and self._is_active_llm_dispatch_node(node)
        )
        dispatch_limit = self._llm_dispatch_limit()
        available_slots = max(0, dispatch_limit - active_count)
        if available_slots <= 0:
            log_kv(
                logger,
                "nodes_wait_dispatch_capacity",
                job_id=job_id,
                node_type=node_type,
                pending=len(pending),
                active=active_count,
                limit=dispatch_limit,
            )
            return True

        selected = pending[:available_slots]
        if self.queue is None:
            for node in selected:
                if node_type == NodeType.MAP:
                    self.map_node(job_id, str(node["node_id"]))
                elif node_type == NodeType.REDUCE:
                    self.reduce_node(job_id, str(node["node_id"]))
            return True

        for node in selected:
            payload = self.store.get_node_payload(job_id, str(node["node_id"]))
            self.store.insert_node_event(
                job_id=job_id,
                node_id=str(node["node_id"]),
                event_type="NODE_ENQUEUED",
                node_status=NodeStatus.QUEUED,
                node_type=node_type,
                level=int(node["level"]),
                node_index=int(node["node_index"]),
                actor="advance_job",
                payload=payload,
            )
            if node_type == NodeType.MAP:
                self.queue.map_node(job_id, str(node["node_id"]))
            elif node_type == NodeType.REDUCE:
                self.queue.reduce_node(job_id, str(node["node_id"]))
        log_kv(
            logger,
            "nodes_enqueued",
            job_id=job_id,
            node_type=node_type,
            count=len(selected),
            pending=len(pending),
            active=active_count,
            limit=dispatch_limit,
        )
        return True

    def _llm_dispatch_limit(self) -> int:
        return max(
            1,
            min(
                max(1, int(self.settings.max_enqueue_nodes_per_advance)),
                max(1, int(self.settings.llm_max_concurrency)),
            ),
        )

    def _is_active_llm_dispatch_node(self, node: dict[str, Any]) -> bool:
        status = node.get("node_status")
        if status == NodeStatus.RUNNING and not self._is_stale_running_node(node):
            return True
        if status == NodeStatus.QUEUED and not self._is_stale_queued_node(node):
            return True
        if status == NodeStatus.WAITING_PROVIDER and not self._is_waiting_provider_ready_node(node):
            return True
        return False

    def _is_stale_running_node(self, node: dict[str, Any]) -> bool:
        if node.get("node_status") != NodeStatus.RUNNING:
            return False
        age_seconds = self._node_age_seconds(node)
        if age_seconds is None:
            return True
        running_timeout_seconds = max(
            float(self.settings.queued_node_requeue_after_seconds),
            float(self.settings.llm_timeout_seconds) + 120.0,
        )
        return age_seconds >= running_timeout_seconds

    def _is_waiting_provider_ready_node(self, node: dict[str, Any]) -> bool:
        if node.get("node_status") != NodeStatus.WAITING_PROVIDER:
            return False
        age_seconds = self._node_age_seconds(node)
        if age_seconds is None:
            return True
        return age_seconds >= max(1.0, float(self.settings.llm_pool_retry_delay_seconds))

    def _is_stale_queued_node(self, node: dict[str, Any]) -> bool:
        if node.get("node_status") != NodeStatus.QUEUED:
            return False
        age_seconds = self._node_age_seconds(node)
        if age_seconds is None:
            return True
        return age_seconds >= self.settings.queued_node_requeue_after_seconds

    @staticmethod
    def _node_age_seconds(node: dict[str, Any]) -> float | None:
        updated_at = node.get("updated_at")
        if updated_at is None:
            return None
        if isinstance(updated_at, str):
            try:
                updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            except ValueError:
                return None
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - updated_at.astimezone(timezone.utc)).total_seconds()

    def _stop_if_not_runnable(self, job_id: str) -> bool:
        job = self.store.get_job_current(job_id)
        if job is None:
            raise KeyError(job_id)
        status = str(job["job_status"])
        if status == JobStatus.PAUSE_REQUESTED:
            self.store.insert_job_event(
                job_id=job_id,
                event_type="JOB_PAUSED",
                job_status=JobStatus.PAUSED,
                actor="worker",
            )
            return True
        if status == JobStatus.PAUSED:
            return True
        if status == JobStatus.INGESTING:
            return True
        if status == JobStatus.CANCEL_REQUESTED:
            self.store.insert_job_event(
                job_id=job_id,
                event_type="JOB_CANCELLED",
                job_status=JobStatus.CANCELLED,
                actor="worker",
            )
            return True
        if status in {JobStatus.CANCELLED, JobStatus.DONE, JobStatus.FAILED}:
            return True
        return False

    def _skip_done(self, job_id: str, node_id: str, node_type: NodeType) -> None:
        node = self._node_by_id(job_id, node_id)
        self.store.insert_node_event(
            job_id=job_id,
            node_id=node_id,
            event_type="NODE_SKIPPED_ALREADY_DONE",
            node_status=NodeStatus.DONE,
            node_type=node_type,
            level=int(node["level"]),
            node_index=int(node["node_index"]),
            actor="worker",
        )

    def _advance_or_inline(self, job_id: str) -> None:
        if self.queue is None:
            self.advance_job(job_id)
        else:
            self.queue.advance_job(job_id)

    def _has_map_source_input(self, job_id: str) -> bool:
        input_segments = self.store.list_input_segments(job_id=job_id, include_content=False)
        if input_segments:
            return True
        return self.store.latest_artifact(job_id=job_id, artifact_type=ArtifactType.INPUT) is not None

    def _fail_job(self, *, job_id: str, actor: str, exc: Exception, payload: dict[str, Any] | None = None) -> None:
        if self._stop_if_not_runnable(job_id):
            return

        error = str(exc)
        failure_payload = {
            "error_class": classify_error(exc),
            "error": error[:2000],
        }
        if payload:
            failure_payload.update(payload)
        self.store.insert_job_event(
            job_id=job_id,
            event_type="JOB_FAILED",
            job_status=JobStatus.FAILED,
            actor=actor,
            message=error[:500],
            payload=failure_payload,
        )

    def _fail_node_and_job(
        self,
        *,
        job_id: str,
        node_id: str,
        node_type: NodeType,
        level: int,
        node_index: int,
        actor: str,
        exc: Exception,
        duration_ms: int | None = None,
        llm_model: str | None = None,
    ) -> None:
        if self._stop_if_not_runnable(job_id):
            return

        error = str(exc)
        payload = {
            "error_class": classify_error(exc),
            "error": error[:2000],
            "node_type": str(node_type),
        }
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        if llm_model:
            payload["llm_model"] = llm_model
        llm_payload = self._latest_llm_payload(job_id=job_id, node_id=node_id)
        if llm_payload:
            payload["llm"] = llm_payload
        self.store.insert_node_event(
            job_id=job_id,
            node_id=node_id,
            event_type="NODE_FAILED",
            node_status=NodeStatus.FAILED_FINAL,
            node_type=node_type,
            level=level,
            node_index=node_index,
            actor=actor,
            message=error[:500],
            payload=payload,
        )
        self.store.insert_job_event(
            job_id=job_id,
            event_type="JOB_FAILED",
            job_status=JobStatus.FAILED,
            actor=actor,
            message=error[:500],
            payload=payload,
        )

    def _defer_node_for_llm_pool(
        self,
        *,
        job_id: str,
        node_id: str,
        node_type: NodeType,
        level: int,
        node_index: int,
        actor: str,
        exc: LlmPoolBusyError,
    ) -> None:
        if self._stop_if_not_runnable(job_id):
            return

        error = str(exc)
        payload = {
            "error_class": classify_error(exc),
            "error": error[:2000],
            "node_type": str(node_type),
            "retry_delay_seconds": self.settings.llm_pool_retry_delay_seconds,
        }
        self.store.insert_node_event(
            job_id=job_id,
            node_id=node_id,
            event_type="NODE_WAITING_PROVIDER",
            node_status=NodeStatus.WAITING_PROVIDER,
            node_type=node_type,
            level=level,
            node_index=node_index,
            actor=actor,
            message=error[:500],
            payload=payload,
        )
        self.store.insert_job_event(
            job_id=job_id,
            event_type="JOB_WAITING_PROVIDER",
            job_status=JobStatus.WAITING_PROVIDER,
            actor=actor,
            message=error[:500],
            payload=payload,
        )

    def _call_summary_llm(
        self,
        *,
        job_id: str,
        node_id: str,
        stage: str,
        system: str,
        user: str,
        model: str,
    ) -> SummaryResult:
        call_summary = self.llm.call_summary
        kwargs = {
            "job_id": job_id,
            "node_id": node_id,
            "stage": stage,
            "system": system,
            "user": user,
        }
        if self._call_accepts_keyword(call_summary, "model"):
            return call_summary(**kwargs, model=model)
        return call_summary(**kwargs)

    @staticmethod
    def _call_accepts_keyword(callable_obj: Any, keyword: str) -> bool:
        try:
            parameters = inspect.signature(callable_obj).parameters
        except (TypeError, ValueError):
            return False
        return keyword in parameters or any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())

    def _resolve_llm_model(self, job_id: str) -> str:
        metadata = self._job_metadata(job_id)
        requested = str(metadata.get("llm_model") or metadata.get("model") or "").strip()
        available_models = tuple(model for model in self.settings.llm_models if model)
        if requested and (not available_models or requested in available_models):
            return requested
        if requested and requested != self.settings.llm_model:
            log_kv(
                logger,
                "llm_model_unavailable",
                job_id=job_id,
                requested=requested,
                fallback=self.settings.llm_model,
                available=",".join(available_models),
            )
        return self.settings.llm_model

    def _node_timing_payload(
        self,
        *,
        job_id: str,
        node_id: str,
        node_type: NodeType,
        duration_ms: int,
        llm_model: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = dict(payload or {})
        result["duration_ms"] = duration_ms
        result["node_type"] = str(node_type)
        result["llm_model"] = llm_model
        llm_payload = self._latest_llm_payload(job_id=job_id, node_id=node_id)
        if llm_payload:
            result["llm"] = llm_payload
        log_kv(
            logger,
            "node_timing",
            job_id=job_id,
            node_id=node_id,
            node_type=node_type,
            duration_ms=duration_ms,
            llm_model=llm_model,
            llm_latency_ms=llm_payload.get("latency_ms", 0) if llm_payload else 0,
            llm_total_tokens=llm_payload.get("total_tokens", 0) if llm_payload else 0,
        )
        return result

    def _latest_llm_payload(self, *, job_id: str, node_id: str) -> dict[str, Any]:
        latest_llm_call = getattr(self.store, "latest_llm_call", None)
        if not callable(latest_llm_call):
            return {}
        row = latest_llm_call(job_id=job_id, node_id=node_id)
        if not row:
            return {}
        latency_ms = self._safe_positive_int(row.get("latency_ms"))
        prompt_tokens = self._safe_positive_int(row.get("prompt_tokens"))
        completion_tokens = self._safe_positive_int(row.get("completion_tokens"))
        total_tokens = self._safe_positive_int(row.get("total_tokens"))
        payload: dict[str, Any] = {
            "provider": str(row.get("provider") or ""),
            "model": str(row.get("model") or ""),
            "status": str(row.get("status") or ""),
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        if latency_ms > 0:
            seconds = latency_ms / 1000
            payload["completion_tokens_per_second"] = round(completion_tokens / seconds, 2) if completion_tokens > 0 else 0
            payload["total_tokens_per_second"] = round(total_tokens / seconds, 2) if total_tokens > 0 else 0
        error_class = str(row.get("error_class") or "")
        error_message = str(row.get("error_message") or "")
        if error_class:
            payload["error_class"] = error_class
        if error_message:
            payload["error_message"] = error_message[:2000]
        return payload

    @staticmethod
    def _safe_positive_int(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    def _job_elapsed_ms(self, job_id: str) -> int:
        events = self.store.list_job_events(job_id, limit=1000)
        created_events = [event for event in events if str(event.get("event_type") or "") == "JOB_CREATED"]
        start_event = created_events[0] if created_events else (events[0] if events else None)
        start_time = self._parse_event_time(start_event.get("event_time")) if start_event else None
        if start_time is None:
            return 0
        return max(0, int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000))

    @staticmethod
    def _parse_event_time(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str) and value.strip():
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                return None
        return None

    def _node_by_id(self, job_id: str, node_id: str) -> dict[str, Any]:
        for node in self.store.list_nodes_current(job_id):
            if node["node_id"] == node_id:
                return node
        raise KeyError(node_id)

    @staticmethod
    def _all_done(nodes: list[dict[str, Any]]) -> bool:
        return bool(nodes) and all(str(node["node_status"]) in {NodeStatus.DONE, NodeStatus.SKIPPED_ALREADY_DONE} for node in nodes)

    @staticmethod
    def _latest_reduce_level(nodes: list[dict[str, Any]]) -> int:
        levels = [int(node["level"]) for node in nodes if node["node_type"] == NodeType.REDUCE]
        return max(levels) if levels else 0

    def _load_summary_artifacts(self, job_id: str, node_ids: list[str]) -> list[dict[str, Any]]:
        artifacts: list[dict[str, Any]] = []
        for node_id in node_ids:
            artifact = self.store.latest_artifact(job_id=job_id, node_id=node_id, artifact_type=ArtifactType.MAP_SUMMARY)
            if artifact is None:
                artifact = self.store.latest_artifact(job_id=job_id, node_id=node_id, artifact_type=ArtifactType.REDUCE_SUMMARY)
            if artifact is None:
                raise RuntimeError(f"summary artifact not found: job_id={job_id} node_id={node_id}")
            artifacts.append(artifact)
        return artifacts

    def _job_metadata(self, job_id: str) -> dict[str, Any]:
        input_artifact = self.store.latest_artifact(job_id=job_id, artifact_type=ArtifactType.INPUT)
        if input_artifact is None:
            return {}
        artifact_metadata = self._safe_json_loads(str(input_artifact.get("metadata") or "{}"))
        job_metadata = artifact_metadata.get("metadata")
        return job_metadata if isinstance(job_metadata, dict) else {}

    def _resolve_report_format(self, job_id: str) -> dict[str, str]:
        metadata = self._job_metadata(job_id)
        report_format = str(
            metadata.get("report_format")
            or metadata.get("output_format")
            or metadata.get("desired_output_format")
            or ""
        ).strip()
        custom_instruction = str(
            metadata.get("report_format_instruction")
            or metadata.get("output_format_instruction")
            or ""
        ).strip()

        if not report_format:
            return {}
        if report_format not in REPORT_FORMAT_INSTRUCTIONS and report_format != "custom":
            custom_instruction = custom_instruction or report_format
            report_format = "custom"

        if report_format == "custom":
            instruction = custom_instruction[:1200]
        else:
            instruction = REPORT_FORMAT_INSTRUCTIONS[report_format]
            if custom_instruction:
                instruction = f"{instruction}\nAdditional user instruction: {custom_instruction[:1200]}"

        if not instruction:
            return {}
        return {
            "report_format": report_format,
            "report_format_instruction": instruction,
        }

    def _resolve_prompt_overrides(self, job_id: str) -> dict[str, dict[str, str]]:
        metadata = self._job_metadata(job_id)
        raw = metadata.get("prompt_overrides")
        if not isinstance(raw, dict):
            return {}

        prompts: dict[str, dict[str, str]] = {}
        for stage in ("map", "reduce", "final"):
            stage_raw = raw.get(stage)
            if not isinstance(stage_raw, dict):
                continue
            system = str(stage_raw.get("system") or "").strip()
            user = str(stage_raw.get("user") or stage_raw.get("user_template") or "").strip()
            clean: dict[str, str] = {}
            if system:
                clean["system"] = system[:4000]
            if user:
                clean["user"] = user[:8000]
            if clean:
                prompts[stage] = clean
        return prompts

    def _resolve_output_json_schema(self, job_id: str) -> dict[str, Any] | None:
        metadata = self._job_metadata(job_id)
        raw = (
            metadata.get("output_json_schema")
            or metadata.get("final_output_json_schema")
            or metadata.get("response_json_schema")
        )
        if isinstance(raw, dict) and raw:
            return raw
        if isinstance(raw, str) and raw.strip():
            parsed = self._safe_json_loads(raw)
            return parsed or None
        return None

    @staticmethod
    def _stage_system_prompt(prompts: dict[str, dict[str, str]], stage: str, default_system: str) -> str:
        return prompts.get(stage, {}).get("system") or default_system

    @staticmethod
    def _stage_user_prompt(
        prompts: dict[str, dict[str, str]],
        stage: str,
        default_template: str,
        values: dict[str, str],
    ) -> str:
        template = prompts.get(stage, {}).get("user") or default_template
        return PipelineService._render_prompt_template(template, values)

    @staticmethod
    def _render_prompt_template(template: str, values: dict[str, str]) -> str:
        rendered = template
        for key, value in values.items():
            rendered = rendered.replace("{" + key + "}", value)
        return rendered

    def _call_final_llm(
        self,
        *,
        job_id: str,
        node_id: str,
        system: str,
        user: str,
        output_json_schema: dict[str, Any] | None,
        model: str,
    ) -> SummaryResult | JsonObjectResult:
        if not output_json_schema:
            return self._call_summary_llm(
                job_id=job_id,
                node_id=node_id,
                stage=Stage.FINAL,
                system=system,
                user=user,
                model=model,
            )

        call_structured = getattr(self.llm, "call_structured", None)
        if callable(call_structured):
            kwargs = {
                "job_id": job_id,
                "node_id": node_id,
                "stage": Stage.FINAL,
                "system": system,
                "user": user,
                "response_model": JsonObjectResult,
            }
            if self._call_accepts_keyword(call_structured, "model"):
                kwargs["model"] = model
            return call_structured(
                **kwargs,
            )

        # Test doubles may only implement call_summary. Keep them compatible.
        return self.llm.call_summary(
            job_id=job_id,
            node_id=node_id,
            stage=Stage.FINAL,
            system=system,
            user=user,
        )

    @staticmethod
    def _report_format_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        preserved: dict[str, Any] = {}
        for key in (
            "report_format",
            "report_format_instruction",
            "output_format",
            "output_format_instruction",
            "desired_output_format",
            "output_json_schema",
            "final_output_json_schema",
            "response_json_schema",
            "prompt_overrides",
            "llm_model",
        ):
            if key in metadata:
                preserved[key] = metadata[key]
        return preserved

    @staticmethod
    def _build_final_user(
        *,
        input_text: str,
        report_format: dict[str, str],
        prompt_overrides: dict[str, dict[str, str]],
        output_json_schema: dict[str, Any] | None,
    ) -> str:
        instruction = report_format.get("report_format_instruction", "").strip()
        output_json_schema_text = json.dumps(output_json_schema, ensure_ascii=False, indent=2) if output_json_schema else ""
        if output_json_schema:
            return PipelineService._stage_user_prompt(
                prompt_overrides,
                "final",
                FINAL_CUSTOM_JSON_USER_TEMPLATE,
                {
                    "summaries": input_text,
                    "report_format_instruction": instruction or "Use the source evidence faithfully.",
                    "output_json_schema": output_json_schema_text,
                },
            )

        override = prompt_overrides.get("final", {}).get("user")
        if override:
            return PipelineService._render_prompt_template(
                override,
                {
                    "summaries": input_text,
                    "report_format_instruction": instruction,
                    "output_json_schema": output_json_schema_text,
                },
            )

        if not instruction:
            return PipelineService._stage_user_prompt(
                prompt_overrides,
                "final",
                FINAL_USER_TEMPLATE,
                {
                    "summaries": input_text,
                    "report_format_instruction": instruction,
                    "output_json_schema": output_json_schema_text,
                },
            )
        return (
            "Create the final user-facing report from these summaries.\n\n"
            "Desired report format:\n"
            f"{instruction}\n\n"
            "Return the SummaryResult transport JSON schema, but put the complete requested report in the summary string. "
            "Do not split the user-facing report into default key_points or warnings sections unless the user asked for those fields.\n\n"
            f"Summaries:\n\n{input_text}"
        )

    @staticmethod
    def _format_reduce_input_summaries(input_summaries: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for index, artifact in enumerate(input_summaries, start=1):
            try:
                payload = SummaryResult.model_validate_json(str(artifact["content"]))
                content = payload.summary
                key_points = "\n".join(f"- {point}" for point in payload.key_points)
            except Exception:
                content = str(artifact["content"])
                key_points = ""
            blocks.append(f"Summary {index}:\n{content}\n{key_points}".strip())
        return "\n\n".join(blocks)

    @staticmethod
    def _safe_json_loads(raw: str) -> dict[str, Any]:
        try:
            loaded = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
