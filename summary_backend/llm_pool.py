"""Distributed concurrency limiter for outbound LLM calls."""

from __future__ import annotations

import math
import socket
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from .config import Settings
from .errors import LlmPoolBusyError
from .logging_setup import get_logger

logger = get_logger("llm_pool")

LLM_POOL_KEY = "summary:llm:global_pool"

_ACQUIRE_SCRIPT = """
redis.call("ZREMRANGEBYSCORE", KEYS[1], "-inf", ARGV[1])
local active = redis.call("ZCARD", KEYS[1])
if active < tonumber(ARGV[3]) then
  redis.call("ZADD", KEYS[1], ARGV[2], ARGV[4])
  redis.call("EXPIRE", KEYS[1], ARGV[5])
  return 1
end
return 0
"""

_RELEASE_SCRIPT = """
redis.call("ZREM", KEYS[1], ARGV[1])
if redis.call("ZCARD", KEYS[1]) == 0 then
  redis.call("DEL", KEYS[1])
end
return 1
"""


@contextmanager
def acquire_llm_pool_slot(
    settings: Settings,
    *,
    job_id: str,
    node_id: str,
    stage: str,
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
    max_concurrency = max(1, int(settings.llm_max_concurrency))
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
                    1,
                    LLM_POOL_KEY,
                    stale_before_ms,
                    now_ms,
                    max_concurrency,
                    owner,
                    key_ttl_seconds,
                )
            )
            if acquired:
                wait_ms = int((time.monotonic() - wait_started) * 1000)
                if wait_ms > 0:
                    logger.info(
                        "llm_pool_acquired | job_id=%s node_id=%s stage=%s wait_ms=%s max_concurrency=%s",
                        job_id,
                        node_id,
                        stage,
                        wait_ms,
                        max_concurrency,
                    )
                yield wait_ms
                return

            if deadline is not None and time.monotonic() >= deadline:
                raise LlmPoolBusyError(
                    f"LLM pool acquire timeout after {acquire_timeout:.1f}s; max_concurrency={max_concurrency}"
                )
            time.sleep(poll_interval)
    finally:
        if acquired:
            try:
                client.eval(_RELEASE_SCRIPT, 1, LLM_POOL_KEY, owner)
            except Exception as exc:  # pragma: no cover - best-effort cleanup log
                logger.warning(
                    "llm_pool_release_failed | job_id=%s node_id=%s stage=%s error=%s",
                    job_id,
                    node_id,
                    stage,
                    exc,
                )
        client.close()
