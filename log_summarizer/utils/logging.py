"""Настройка логирования для пайплайна Log Summarizer."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(name: str) -> logging.Logger:
    """Возвращает logger с именем внутри пространства log_summarizer."""
    return logging.getLogger(f"log_summarizer.{name}")


def setup_pipeline_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """Настраивает root-логгер пайплайна.

    Args:
        level: Уровень логирования (DEBUG / INFO / WARNING / ERROR).
        log_file: Опциональный путь к файлу для дублирования вывода.
    """
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger("log_summarizer")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Заглушаем шумные внешние библиотеки
    for noisy in ("httpcore", "httpx", "openai", "instructor", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
