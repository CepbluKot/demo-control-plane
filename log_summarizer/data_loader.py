"""DataLoader — постраничная выгрузка логов и метрик из ClickHouse.

Единственная ответственность: отдать данные. Ничего про батчи, LLM, токены.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
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

        Логи грузятся за context-окно (context_start..context_end).
        Если context-окно не задано — используется incident-окно.
        Каждая строка получает zone-метку относительно incident-окна.

        Если config.query_time_slice_hours > 0 — период разбивается на слайсы
        и для каждого слайса выполняется отдельная серия запросов с пагинацией.
        Это предотвращает OOM в ClickHouse при SQL с оконными функциями.

        Args:
            page_size: Размер одной страницы (строк).
        """
        ctx_start = self.config.context_start_actual()
        ctx_end = self.config.context_end_actual()
        inc_start = self.config.incident_start
        inc_end = self.config.incident_end
        slice_hours = self.config.query_time_slice_hours

        if slice_hours and slice_hours > 0 and ctx_start and ctx_end:
            slice_start = ctx_start
            slice_num = 0
            while slice_start < ctx_end:
                slice_end = min(slice_start + timedelta(hours=slice_hours), ctx_end)
                slice_num += 1
                logger.debug(
                    "Time slice %d: %s → %s",
                    slice_num,
                    slice_start.strftime("%H:%M"),
                    slice_end.strftime("%H:%M"),
                )
                yield from self._iter_slice(slice_start, slice_end, inc_start, inc_end, page_size)
                slice_start = slice_end
        else:
            yield from self._iter_slice(ctx_start, ctx_end, inc_start, inc_end, page_size)

    def _iter_slice(
        self,
        slice_start: Optional[datetime],
        slice_end: Optional[datetime],
        inc_start: Optional[datetime],
        inc_end: Optional[datetime],
        page_size: int,
    ) -> Iterator[list[LogRow]]:
        """Постраничная выгрузка за один временно́й слайс."""
        template = self.config.logs_sql_template
        start = self._fmt_dt(slice_start)
        end = self._fmt_dt(slice_end)
        uses_keyset = "{last_ts}" in template
        offset = 0
        # Инициализируем last_ts на 1 мкс раньше начала слайса:
        # outer WHERE использует строгое >, поэтому без этого группы
        # с start_time == slice_start точно пропускались бы.
        last_ts = self._fmt_dt(slice_start - timedelta(microseconds=1)) if slice_start else start

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

            page = [
                self._row_to_log(r, inc_start=inc_start, inc_end=inc_end)
                for r in rows
            ]
            yield page
            logger.debug("Fetched %d log rows", len(page))

            if len(rows) < page_size:
                break  # последняя страница слайса

            if uses_keyset:
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
    def _row_to_log(
        row: dict,
        inc_start: Optional[datetime] = None,
        inc_end: Optional[datetime] = None,
    ) -> LogRow:
        """Конвертируем dict-строку ClickHouse в LogRow с zone-разметкой.

        Поддерживаемые форматы:
          - Классический: message/msg/log + source/service + level/severity
          - Kubernetes-агрегированный: namespace + container_name + raw_line
            (raw_line содержит весь текст лога, level вытаскивается парсером)
        """
        ts = row.get("timestamp") or row.get("time") or row.get("ts")
        raw_line = row.get("raw_line") or str(row)

        # source: явное поле → kubernetes_container_name → namespace/container_name → container/pod
        source = (
            row.get("source")
            or row.get("service")
            or row.get("kubernetes_container_name")
        )
        if not source:
            ns = row.get("namespace")
            cn = row.get("container_name") or row.get("container") or row.get("pod")
            if ns and cn:
                source = f"{ns}/{cn}"
            else:
                source = cn or ns

        # message: явное поле → fallback на raw_line (для агрегированного формата)
        message = (
            row.get("message")
            or row.get("msg")
            or row.get("log")
            or row.get("value")
            or raw_line
        )

        # level: явное поле → распознавание из raw_line
        level = row.get("level") or row.get("severity")
        if not level:
            level = DataLoader._extract_level(raw_line)

        # Определяем зону строки относительно окна инцидента.
        # ClickHouse возвращает DateTime как UTC-наивные объекты — нормализуем к UTC,
        # тогда сравнение с МСК-aware inc_start/inc_end работает корректно.
        zone = "incident"
        if ts is not None and inc_start is not None and inc_end is not None:
            if hasattr(ts, "replace"):
                ts_dt = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
            else:
                ts_dt = ts
            if ts_dt < inc_start:
                zone = "context_before"
            elif ts_dt > inc_end:
                zone = "context_after"

        return LogRow(
            timestamp=ts,
            level=level,
            source=source,
            message=str(message),
            raw_line=str(raw_line),
            zone=zone,
        )

    @staticmethod
    def _extract_level(raw_line: str) -> Optional[str]:
        """Пытается определить уровень лога из текста строки.

        Поддерживает два паттерна:
          1. structured: level=warning / level=error / level=info / level=debug
          2. klog (Kubernetes): E0317 / W0317 / I0317 / D0317 после пробелов или ']'
        """
        import re
        # structured logs: level=<value>
        m = re.search(r'\blevel=(\w+)', raw_line)
        if m:
            return m.group(1).lower()
        # klog prefix: E/W/I/D + MMDD
        # Варианты: в начале строки, после '] ×N  ' (агрегированный формат), после обычного ']'
        m = re.search(r'(?:]\s*(?:×\d+\s+)?|^\s*)([EWID])\d{4}\s', raw_line)
        if m:
            return {"E": "error", "W": "warning", "I": "info", "D": "debug"}.get(m.group(1))
        return None

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
