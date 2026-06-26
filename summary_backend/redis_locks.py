"""Small Redis lock helper for concurrent Dramatiq workers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from .config import Settings
from .logging_setup import get_logger

logger = get_logger("redis_locks")

T = TypeVar("T")


def run_with_redis_lock(
    settings: Settings,
    lock_name: str,
    *,
    timeout_seconds: float,
    blocking_timeout_seconds: float | None,
    action: Callable[[], T],
) -> tuple[bool, T | None]:
    import redis
    from redis.exceptions import LockError

    client = redis.Redis.from_url(settings.broker_url)
    lock = client.lock(lock_name, timeout=timeout_seconds)
    acquired = False
    try:
        if blocking_timeout_seconds is None:
            acquired = bool(lock.acquire(blocking=True))
        elif blocking_timeout_seconds <= 0:
            acquired = bool(lock.acquire(blocking=False))
        else:
            acquired = bool(lock.acquire(blocking=True, blocking_timeout=blocking_timeout_seconds))
        if not acquired:
            return False, None
        return True, action()
    finally:
        if acquired:
            try:
                lock.release()
            except LockError:
                logger.warning("redis_lock_release_failed | lock=%s", lock_name)
        client.close()
