"""Dramatiq actors for the summary backend."""

from __future__ import annotations

import dramatiq

from .broker import configure_broker
from .config import get_settings
from .errors import LlmPoolBusyError
from .factory import create_pipeline_service
from .ingestion import StagedUploadIngestionService
from .logging_setup import configure_logging, get_logger
from .queue import TaskQueue
from .redis_locks import run_with_redis_lock
from .store import ClickHouseStore

settings = get_settings()
configure_logging(settings)
configure_broker(settings)

logger = get_logger("tasks")


def _llm_pool_retry_delay_ms() -> int:
    return int(max(1.0, float(settings.llm_pool_retry_delay_seconds)) * 1000)


class DramatiqTaskQueue(TaskQueue):
    def ingest_upload(self, job_id: str) -> None:
        ingest_upload.send(job_id)

    def advance_job(self, job_id: str) -> None:
        advance_job.send(job_id)

    def map_node(self, job_id: str, node_id: str) -> None:
        map_node.send(job_id, node_id)

    def reduce_node(self, job_id: str, node_id: str) -> None:
        reduce_node.send(job_id, node_id)

    def finalize_job(self, job_id: str) -> None:
        finalize_job.send(job_id)


def _service():
    queue = DramatiqTaskQueue()
    return create_pipeline_service(queue=queue, settings=settings)


def _staged_upload_service():
    return StagedUploadIngestionService(
        store=ClickHouseStore(settings),
        queue=DramatiqTaskQueue(),
        settings=settings,
    )


def _llm_lock_timeout_seconds() -> float:
    return max(300.0, float(settings.llm_timeout_seconds) + 120.0)


@dramatiq.actor(max_retries=5, min_backoff=10000, max_backoff=600000)
def ingest_upload(job_id: str) -> None:
    logger.info("actor.ingest_upload | job_id=%s", job_id)
    _staged_upload_service().ingest_staged_upload(job_id)


@dramatiq.actor(max_retries=3, min_backoff=5000, max_backoff=300000)
def advance_job(job_id: str) -> None:
    logger.info("actor.advance_job | job_id=%s", job_id)
    acquired, _ = run_with_redis_lock(
        settings,
        f"summary:job:{job_id}:advance",
        timeout_seconds=60.0,
        blocking_timeout_seconds=30.0,
        action=lambda: _service().advance_job(job_id),
    )
    if not acquired:
        raise RuntimeError(f"advance lock busy for job_id={job_id}")


@dramatiq.actor(max_retries=5, min_backoff=10000, max_backoff=600000)
def map_node(job_id: str, node_id: str) -> None:
    logger.info("actor.map_node | job_id=%s node_id=%s", job_id, node_id)
    try:
        acquired, _ = run_with_redis_lock(
            settings,
            f"summary:job:{job_id}:node:{node_id}",
            timeout_seconds=_llm_lock_timeout_seconds(),
            blocking_timeout_seconds=0.0,
            action=lambda: _service().map_node(job_id, node_id),
        )
    except LlmPoolBusyError:
        delay_ms = _llm_pool_retry_delay_ms()
        logger.info("actor.map_node.llm_pool_busy_requeue | job_id=%s node_id=%s delay_ms=%s", job_id, node_id, delay_ms)
        map_node.send_with_options(args=(job_id, node_id), delay=delay_ms)
        return
    if not acquired:
        logger.info("actor.map_node.locked_skip | job_id=%s node_id=%s", job_id, node_id)


@dramatiq.actor(max_retries=5, min_backoff=10000, max_backoff=600000)
def reduce_node(job_id: str, node_id: str) -> None:
    logger.info("actor.reduce_node | job_id=%s node_id=%s", job_id, node_id)
    try:
        acquired, _ = run_with_redis_lock(
            settings,
            f"summary:job:{job_id}:node:{node_id}",
            timeout_seconds=_llm_lock_timeout_seconds(),
            blocking_timeout_seconds=0.0,
            action=lambda: _service().reduce_node(job_id, node_id),
        )
    except LlmPoolBusyError:
        delay_ms = _llm_pool_retry_delay_ms()
        logger.info("actor.reduce_node.llm_pool_busy_requeue | job_id=%s node_id=%s delay_ms=%s", job_id, node_id, delay_ms)
        reduce_node.send_with_options(args=(job_id, node_id), delay=delay_ms)
        return
    if not acquired:
        logger.info("actor.reduce_node.locked_skip | job_id=%s node_id=%s", job_id, node_id)


@dramatiq.actor(max_retries=3, min_backoff=10000, max_backoff=300000)
def finalize_job(job_id: str) -> None:
    logger.info("actor.finalize_job | job_id=%s", job_id)
    try:
        acquired, _ = run_with_redis_lock(
            settings,
            f"summary:job:{job_id}:finalize",
            timeout_seconds=_llm_lock_timeout_seconds(),
            blocking_timeout_seconds=0.0,
            action=lambda: _service().finalize_job(job_id),
        )
    except LlmPoolBusyError:
        delay_ms = _llm_pool_retry_delay_ms()
        logger.info("actor.finalize_job.llm_pool_busy_requeue | job_id=%s delay_ms=%s", job_id, delay_ms)
        finalize_job.send_with_options(args=(job_id,), delay=delay_ms)
        return
    if not acquired:
        logger.info("actor.finalize_job.locked_skip | job_id=%s", job_id)


@dramatiq.actor(max_retries=1)
def recover_jobs() -> None:
    logger.info("actor.recover_jobs")
    _service().recover_jobs()
