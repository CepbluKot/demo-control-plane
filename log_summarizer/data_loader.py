"""DataLoader — постраничная выгрузка логов и метрик из ClickHouse.

Единственная ответственность: отдать данные. Ничего про батчи, LLM, токены.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Iterator, Optional

from log_summarizer.config import PipelineConfig
from log_summarizer.models import LogRow, MetricRow
from log_summarizer.utils.logging import get_logger

logger = get_logger("data_loader")


class DataLoader:
    """Постраничная выгрузка логов и метрик из ClickHouse.

    Args:
        clickhouse_client: Клиент clickhouse_connect (get_client(...)).
        config: Конфигурация пайплайна.
    """

    def __init__(self, clickhouse_client: Any, config: PipelineConfig) -> None:
        self.client = clickhouse_client
        self.config = config

    # ── Публичный API ─────────────────────────────────────────────────

    def iter_log_pages(self, page_size: int = 1000) -> Iterator[list[LogRow]]:
        """Постраничная выгрузка логов.

        Yield'ит страницы (list[LogRow]). Каждая страница не больше page_size.
        Пагинация через OFFSET или через timestamp-фильтр если шаблон содержит
        плейсхолдер {last_ts}.

        Args:
            page_size: Размер одной страницы (строк).
        """
        template = self.config.logs_sql_template
        start = self._fmt_dt(self.config.incident_start)
        end = self._fmt_dt(self.config.incident_end)

        uses_keyset = "{last_ts}" in template
        offset = 0
        last_ts = start

        while True:
            sql = self._render_logs_sql(
                template=template,
                start_time=start,
                end_time=end,
                limit=page_size,
                offset=offset,
                last_ts=last_ts,
            )
            logger.debug("Fetching log page: offset=%d, last_ts=%s", offset, last_ts)
            rows = self._execute_query(sql)
            if not rows:
                break

            page = [self._row_to_log(r) for r in rows]
            yield page
            logger.debug("Fetched %d log rows", len(page))

            if len(rows) < page_size:
                break  # последняя страница

            if uses_keyset:
                # keyset: следующая страница начинается после max(timestamp)
                last_ts = self._max_ts_from_rows(rows, last_ts)
            else:
                offset += page_size

    def fetch_metrics(
        self,
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> Optional[list[MetricRow]]:
        """Выгрузка всех метрик за период одним запросом.

        Метрики компактные — влезают в память целиком.
        Возвращает None если metrics_sql_template не задан.

        Args:
            start: Начало периода.
            end: Конец периода.
        """
        if not self.config.metrics_sql_template:
            return None

        sql = self._render_metrics_sql(
            template=self.config.metrics_sql_template,
            start_time=self._fmt_dt(start),
            end_time=self._fmt_dt(end),
        )
        logger.debug("Fetching metrics")
        rows = self._execute_query(sql)
        if not rows:
            return []

        result = []
        for r in rows:
            try:
                result.append(self._row_to_metric(r))
            except Exception as exc:
                logger.warning("Skipping invalid metric row: %s | row=%r", exc, r)
        logger.info("Fetched %d metric rows", len(result))
        return result

    # ── Вспомогательные методы ────────────────────────────────────────

    def _execute_query(self, sql: str) -> list[dict]:
        """Выполняет SQL и возвращает список строк как dict."""
        try:
            result = self.client.query(sql)
            if hasattr(result, "named_results"):
                return list(result.named_results())
            # clickhouse_connect может вернуть QueryResult
            if hasattr(result, "result_rows") and hasattr(result, "column_names"):
                cols = result.column_names
                return [dict(zip(cols, row)) for row in result.result_rows]
            return []
        except Exception as exc:
            logger.error("ClickHouse query error: %s\nSQL: %s", exc, sql[:200])
            raise

    def _render_logs_sql(
        self,
        template: str,
        start_time: str,
        end_time: str,
        limit: int,
        offset: int,
        last_ts: str,
    ) -> str:
        """Подставляет плейсхолдеры в SQL-шаблон логов."""
        replacements = {
            "start_time": start_time,
            "end_time": end_time,
            "limit": str(limit),
            "offset": str(offset),
            "last_ts": last_ts,
            # Совместимость с my_summarizer плейсхолдерами
            "period_start": start_time,
            "period_end": end_time,
        }
        sql = template
        for key, val in replacements.items():
            sql = sql.replace(f"{{{key}}}", val)
        return sql

    def _render_metrics_sql(
        self,
        template: str,
        start_time: str,
        end_time: str,
    ) -> str:
        """Подставляет плейсхолдеры в SQL-шаблон метрик."""
        replacements = {
            "start_time": start_time,
            "end_time": end_time,
            "period_start": start_time,
            "period_end": end_time,
        }
        sql = template
        for key, val in replacements.items():
            sql = sql.replace(f"{{{key}}}", val)
        return sql

    @staticmethod
    def _fmt_dt(dt: Optional[datetime]) -> str:
        """Форматируем datetime для подстановки в SQL."""
        if dt is None:
            return ""
        return dt.isoformat()

    @staticmethod
    def _row_to_log(row: dict) -> LogRow:
        """Конвертируем dict-строку ClickHouse в LogRow."""
        ts = row.get("timestamp") or row.get("time") or row.get("ts")
        message = (
            row.get("message")
            or row.get("msg")
            or row.get("log")
            or row.get("value")
            or ""
        )
        raw_line = row.get("raw_line") or str(row)
        return LogRow(
            timestamp=ts,
            level=row.get("level") or row.get("severity"),
            source=(
                row.get("source")
                or row.get("service")
                or row.get("kubernetes_container_name")
                or row.get("container")
                or row.get("pod")
            ),
            message=str(message),
            raw_line=str(raw_line),
        )

    @staticmethod
    def _row_to_metric(row: dict) -> MetricRow:
        """Конвертируем dict-строку ClickHouse в MetricRow."""
        ts = row.get("timestamp") or row.get("time") or row.get("ts")
        return MetricRow(
            timestamp=ts,
            service=str(row.get("service") or row.get("source") or "unknown"),
            metric_name=str(row.get("metric_name") or row.get("metric") or "value"),
            value=float(row.get("value") or 0),
        )

    @staticmethod
    def _max_ts_from_rows(rows: list[dict], fallback: str) -> str:
        """Извлекаем максимальный timestamp для keyset-пагинации."""
        ts_values = []
        for row in rows:
            ts = row.get("timestamp") or row.get("time") or row.get("ts")
            if ts is not None:
                ts_values.append(str(ts))
        if not ts_values:
            return fallback
        return max(ts_values)
