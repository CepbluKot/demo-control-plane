"""Dramatiq broker configuration."""

from __future__ import annotations

from .config import Settings, get_settings
from .logging_setup import get_logger

logger = get_logger("broker")


def configure_broker(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    import dramatiq

    url = settings.broker_url
    if not url.startswith(("redis://", "rediss://")):
        raise ValueError(f"Only Redis broker URLs are supported, got: {url}")

    from dramatiq.brokers.redis import RedisBroker

    broker = RedisBroker(url=url)

    dramatiq.set_broker(broker)
    logger.info("Dramatiq broker configured: %s", url)
