from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Protocol

logger = logging.getLogger(__name__)


class PeriodSummarizer(Protocol):
    """
    Интерфейс суммаризации логов за период.
    Реализацию можно подложить через CONTROL_PLANE_SUMMARIZER_CALLABLE
    или через внешний пакет llm_log_summarizer.
    """

    def __call__(
        self,
        *,
        start_dt: datetime,
        end_dt: datetime,
        anomaly: Optional[Dict[str, Any]] = None,
    ) -> Any:
        ...


def do_summary(
    *,
    start_dt: datetime,
    end_dt: datetime,
    anomaly: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Дефолтный адаптер к внешнему llm_log_summarizer.
    Возвращает объект с полем summary или строку (как вернет внешний сервис).
    """
    del anomaly  # anomaly оставляем в интерфейсе для единообразия адаптеров
    try:
        from llm_log_summarizer import do_summary as do_summary_module
    except Exception as exc:
        raise ImportError(
            "Не удалось импортировать llm_log_summarizer. "
            "Установи пакет или задай CONTROL_PLANE_SUMMARIZER_CALLABLE."
        ) from exc

    # Поддерживаем оба варианта:
    # 1) do_summary_module — уже функция
    # 2) do_summary_module — модуль с функцией do_summary внутри
    if callable(do_summary_module):
        fn = do_summary_module
    else:
        fn = getattr(do_summary_module, "do_summary", None)
    if not callable(fn):
        raise ValueError(
            "llm_log_summarizer.do_summary.do_summary не найден или не является callable"
        )

    logger.info(
        "summarizer.do_summary.default_adapter: start=%s end=%s",
        start_dt.isoformat(),
        end_dt.isoformat(),
    )
    return fn(start_dt=start_dt, end_dt=end_dt)
