"""Logging setup for API, workers, and local smoke runs."""

from __future__ import annotations

import logging
import sys
import threading
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterator

from .config import Settings, get_settings


LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)-7s | "
    "%(process)d:%(threadName)s | %(name)s:%(lineno)d | %(message)s"
)


def configure_logging(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    settings.log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    console.setLevel(level)

    main_file = RotatingFileHandler(
        settings.log_dir / "summary-backend.log",
        maxBytes=20 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    main_file.setFormatter(formatter)
    main_file.setLevel(level)

    error_file = RotatingFileHandler(
        settings.log_dir / "summary-backend.errors.log",
        maxBytes=20 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    error_file.setFormatter(formatter)
    error_file.setLevel(logging.ERROR)

    logger.setLevel(level)
    logger.addHandler(console)
    logger.addHandler(main_file)
    logger.addHandler(error_file)

    logging.captureWarnings(True)
    for noisy in ("httpcore", "httpx", "openai", "urllib3", "pika", "clickhouse_connect"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    def _handle_uncaught(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger("summary_backend.uncaught").exception(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    def _handle_thread_exception(args):
        logging.getLogger("summary_backend.uncaught.thread").exception(
            "Uncaught thread exception",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _handle_uncaught
    threading.excepthook = _handle_thread_exception

    logging.getLogger(__name__).info(
        "Logging configured: level=%s log_dir=%s",
        settings.log_level.upper(),
        settings.log_dir,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"summary_backend.{name}")


def log_kv(logger: logging.Logger, event: str, **fields: object) -> None:
    suffix = " ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))
    if suffix:
        logger.info("%s | %s", event, suffix)
    else:
        logger.info("%s", event)


def log_stage(logger: logging.Logger, title: str) -> None:
    sep = "=" * 88
    logger.info("")
    logger.info(sep)
    logger.info("%s", title)
    logger.info(sep)


@contextmanager
def timed_stage(logger: logging.Logger, name: str, **fields: object) -> Iterator[None]:
    started = time.monotonic()
    log_kv(logger, f"{name}.start", **fields)
    try:
        yield
    except Exception:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.exception("%s.error | elapsed_ms=%s", name, elapsed_ms)
        raise
    else:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        log_kv(logger, f"{name}.done", elapsed_ms=elapsed_ms, **fields)


def ensure_log_dirs(settings: Settings | None = None) -> tuple[Path, Path]:
    settings = settings or get_settings()
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    settings.audit_dir.mkdir(parents=True, exist_ok=True)
    return settings.log_dir, settings.audit_dir
