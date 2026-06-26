"""Dramatiq actors for the summary backend."""

from __future__ import annotations

import dramatiq

from .broker import configure_broker
from .config import get_settings
from .factory import create_pipeline_service
from .ingestion import StagedUploadIngestionService
from .logging_setup import configure_logging, get_logger
from .queue import TaskQueue
from .store import ClickHouseStore

settings = get_settings()
configure_logging(settings)
configure_broker(settings)

logger = get_logger("tasks")


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


@dramatiq.actor(max_retries=5, min_backoff=10000, max_backoff=600000)
def ingest_upload(job_id: str) -> None:
    logger.info("actor.ingest_upload | job_id=%s", job_id)
    _staged_upload_service().ingest_staged_upload(job_id)


@dramatiq.actor(max_retries=3, min_backoff=5000, max_backoff=300000)
def advance_job(job_id: str) -> None:
    logger.info("actor.advance_job | job_id=%s", job_id)
    _service().advance_job(job_id)


@dramatiq.actor(max_retries=5, min_backoff=10000, max_backoff=600000)
def map_node(job_id: str, node_id: str) -> None:
    logger.info("actor.map_node | job_id=%s node_id=%s", job_id, node_id)
    _service().map_node(job_id, node_id)


@dramatiq.actor(max_retries=5, min_backoff=10000, max_backoff=600000)
def reduce_node(job_id: str, node_id: str) -> None:
    logger.info("actor.reduce_node | job_id=%s node_id=%s", job_id, node_id)
    _service().reduce_node(job_id, node_id)


@dramatiq.actor(max_retries=3, min_backoff=10000, max_backoff=300000)
def finalize_job(job_id: str) -> None:
    logger.info("actor.finalize_job | job_id=%s", job_id)
    _service().finalize_job(job_id)


@dramatiq.actor(max_retries=1)
def recover_jobs() -> None:
    logger.info("actor.recover_jobs")
    _service().recover_jobs()
