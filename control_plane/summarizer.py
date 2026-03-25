from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Protocol

logger = logging.getLogger(__name__)


class PeriodSummarizer(Protocol):
    """
    Интерфейс суммаризации логов за период.
    Реализацию можно подложить через CONTROL_PLANE_SUMMARIZER_CALLABLE
    или использовать дефолтный my_summarizer (map-reduce).
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
    Дефолтный адаптер: встроенный map-reduce summarizer (`my_summarizer.summarize_logs`).
    Возвращает dict с итоговым summary и промежуточными map summary.
    """
    try:
        from my_summarizer import summarize_logs
    except Exception as exc:
        raise ImportError(
            "Не удалось импортировать my_summarizer.summarize_logs. "
            "Проверь файл my_summarizer.py или задай CONTROL_PLANE_SUMMARIZER_CALLABLE."
        ) from exc

    logger.info(
        "summarizer.do_summary.default_adapter: start=%s end=%s",
        start_dt.isoformat(),
        end_dt.isoformat(),
    )
    return summarize_logs(start_dt=start_dt, end_dt=end_dt, anomaly=anomaly)
