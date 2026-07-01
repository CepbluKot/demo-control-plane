"""Distributed concurrency limiter for outbound LLM calls."""

from __future__ import annotations

import math
import socket
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

from .config import Settings
from .errors import LlmPoolBusyError
from .logging_setup import get_logger

logger = get_logger("llm_pool")

LlmPoolKind = Literal["jobs", "assistant"]

_POOL_KEYS: dict[LlmPoolKind, str] = {
    "jobs": "summary:llm:jobs_pool",
    "assistant": "summary:llm:assistant_pool",
}

_ACQUIRE_SCRIPT = """
for index = 1, #KEYS do
  redis.call("ZREMRANGEBYSCORE", KEYS[index], "-inf", ARGV[1])
end
if redis.call("ZCARD", KEYS[1]) >= tonumber(ARGV[5]) then
  return -1
end
if #KEYS > 1 then
  local job_limit = tonumber(ARGV[6])
  if job_limit and redis.call("ZCARD", KEYS[2]) >= job_limit then
    return -2
  end
end
for index = 1, #KEYS do
  redis.call("ZADD", KEYS[index], ARGV[2], ARGV[3])
  redis.call("EXPIRE", KEYS[index], ARGV[4])
end
return 1
"""

_RELEASE_SCRIPT = """
for index = 1, #KEYS do
  redis.call("ZREM", KEYS[index], ARGV[1])
  if redis.call("ZCARD", KEYS[index]) == 0 then
    redis.call("DEL", KEYS[index])
  end
end
return 1
"""


def _job_pool_key(base_key: str, job_id: str) -> str:
    return f"{base_key}:job:{job_id}"


@contextmanager
def acquire_llm_pool_slot(
    settings: Settings,
    *,
    job_id: str,
    node_id: str,
    stage: str,
    pool_kind: LlmPoolKind = "jobs",
    job_max_concurrency: int | None = None,
) -> Iterator[int]:
    """Acquire one process/pod-wide LLM slot backed by Redis.

    The slot has a lease so a crashed worker does not permanently reduce
    capacity. Keep the acquire timeout short so Dramatiq threads are not held
    while all outbound LLM slots are busy.
    """

    import redis

    client = redis.Redis.from_url(settings.broker_url)
    owner = f"{socket.gethostname()}:{job_id}:{node_id}:{stage}:{uuid.uuid4().hex}"
    lease_seconds = max(300.0, float(settings.llm_timeout_seconds) + 120.0)
    lease_ms = int(lease_seconds * 1000)
    key_ttl_seconds = int(math.ceil(lease_seconds + 60.0))
    pool_key = _POOL_KEYS[pool_kind]
    max_concurrency = max(
        1,
        int(
            settings.llm_assistant_max_concurrency
            if pool_kind == "assistant"
            else settings.llm_max_concurrency
        ),
    )
    normalized_job_max_concurrency = None
    if job_max_concurrency is not None:
        normalized_job_max_concurrency = max(1, int(job_max_concurrency))
        if normalized_job_max_concurrency >= max_concurrency:
            normalized_job_max_concurrency = None
    pool_keys = [pool_key]
    if normalized_job_max_concurrency is not None:
        pool_keys.append(_job_pool_key(pool_key, job_id))
    acquire_timeout = float(settings.llm_pool_acquire_timeout_seconds)
    poll_interval = max(0.05, float(settings.llm_pool_poll_interval_seconds))
    deadline = None if acquire_timeout <= 0 else time.monotonic() + acquire_timeout
    wait_started = time.monotonic()
    acquired = False

    try:
        while True:
            now_ms = int(time.time() * 1000)
            stale_before_ms = now_ms - lease_ms
            acquired = bool(
                client.eval(
                    _ACQUIRE_SCRIPT,
                    len(pool_keys),
                    *pool_keys,
                    stale_before_ms,
                    now_ms,
                    owner,
                    key_ttl_seconds,
                    max_concurrency,
                    normalized_job_max_concurrency or "",
                )
            )
            if acquired:
                wait_ms = int((time.monotonic() - wait_started) * 1000)
                if wait_ms > 0:
                    logger.info(
                        "llm_pool_acquired | job_id=%s node_id=%s stage=%s pool_kind=%s wait_ms=%s max_concurrency=%s job_max_concurrency=%s",
                        job_id,
                        node_id,
                        stage,
                        pool_kind,
                        wait_ms,
                        max_concurrency,
                        normalized_job_max_concurrency,
                    )
                yield wait_ms
                return

            if deadline is not None and time.monotonic() >= deadline:
                limit_details = f"max_concurrency={max_concurrency}"
                if normalized_job_max_concurrency is not None:
                    limit_details = f"{limit_details}; job_max_concurrency={normalized_job_max_concurrency}"
                raise LlmPoolBusyError(
                    f"LLM pool acquire timeout after {acquire_timeout:.1f}s; {limit_details}"
                )
            time.sleep(poll_interval)
    finally:
        if acquired:
            try:
                client.eval(_RELEASE_SCRIPT, len(pool_keys), *pool_keys, owner)
            except Exception as exc:  # pragma: no cover - best-effort cleanup log
                logger.warning(
                    "llm_pool_release_failed | job_id=%s node_id=%s stage=%s pool_kind=%s job_max_concurrency=%s error=%s",
                    job_id,
                    node_id,
                    stage,
                    pool_kind,
                    normalized_job_max_concurrency,
                    exc,
                )
        client.close()
