import logging
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from .config import (
    CLICKHOUSE_METRICS_HOST,
    CLICKHOUSE_METRICS_PASSWORD,
    CLICKHOUSE_METRICS_PORT,
    CLICKHOUSE_METRICS_QUERY,
    CLICKHOUSE_METRICS_SECURE,
    CLICKHOUSE_METRICS_USERNAME,
    METRICS_SOURCE,
)
from .prometheus_io import fetch_prometheus_df
from .trace import log_dataframe, log_event

logger = logging.getLogger(__name__)


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _render_clickhouse_query(
    query_template: str,
    start_time: datetime,
    end_time: datetime,
) -> str:
    params = _SafeFormatDict(
        start=start_time.isoformat(),
        end=end_time.isoformat(),
        start_iso=start_time.isoformat(),
        end_iso=end_time.isoformat(),
        start_ts=int(start_time.timestamp()),
        end_ts=int(end_time.timestamp()),
    )
    query_template = query_template.strip().rstrip(";")
    return query_template.format_map(params)


def _fetch_clickhouse_actuals(
    *,
    start_time: datetime,
    end_time: datetime,
    query_template: str,
) -> pd.DataFrame:
    try:
        import clickhouse_connect
    except Exception as exc:
        raise ImportError(
            "Для чтения фактических метрик из ClickHouse нужен пакет clickhouse-connect"
        ) from exc

    if not CLICKHOUSE_METRICS_HOST:
        raise ValueError(
            "Для METRICS_SOURCE=clickhouse укажи CONTROL_PLANE_CLICKHOUSE_METRICS_HOST в .env"
        )
    if not query_template:
        raise ValueError(
            "Для METRICS_SOURCE=clickhouse укажи CONTROL_PLANE_CLICKHOUSE_METRICS_QUERY в .env"
        )

    log_event(
        logger,
        "fetch_actuals.clickhouse.start",
        host=CLICKHOUSE_METRICS_HOST,
        port=CLICKHOUSE_METRICS_PORT,
        secure=CLICKHOUSE_METRICS_SECURE,
        start=start_time.isoformat(),
        end=end_time.isoformat(),
    )
    query = _render_clickhouse_query(
        query_template,
        start_time,
        end_time,
    )
    log_event(
        logger,
        "fetch_actuals.clickhouse.query",
        query_preview=query[:600],
    )
    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_METRICS_HOST,
        port=CLICKHOUSE_METRICS_PORT,
        username=CLICKHOUSE_METRICS_USERNAME or None,
        password=CLICKHOUSE_METRICS_PASSWORD or None,
        secure=CLICKHOUSE_METRICS_SECURE,
    )
    try:
        df = client.query_df(query)
    finally:
        try:
            client.close()
        except Exception:
            logger.warning("ClickHouse client close failed for actual metrics query")

    if df.empty:
        log_event(logger, "fetch_actuals.clickhouse.empty")
        return pd.DataFrame(columns=["timestamp", "value"])

    if "timestamp" not in df.columns or "value" not in df.columns:
        raise ValueError("ClickHouse query должен вернуть колонки timestamp и value")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    out = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp").reset_index(drop=True)
    log_dataframe(logger, "fetch_actuals.clickhouse.df", out)
    log_event(logger, "fetch_actuals.clickhouse.done", rows=len(out))
    return out


def fetch_actual_metrics_df(
    *,
    start_time: datetime,
    end_time: datetime,
    query: str,
    step: str,
    max_points: int,
    series_index: int,
    source: Optional[str] = None,
) -> pd.DataFrame:
    effective_source = (source or METRICS_SOURCE or "prometheus").strip().lower()
    log_event(
        logger,
        "fetch_actual_metrics_df.start",
        source=effective_source,
        start=start_time.isoformat(),
        end=end_time.isoformat(),
    )
    if effective_source == "prometheus":
        out = fetch_prometheus_df(
            query=query,
            start_time=start_time,
            end_time=end_time,
            step=step,
            max_points=max_points,
            series_index=series_index,
        )
        log_dataframe(logger, "fetch_actual_metrics_df.prometheus", out)
        return out
    if effective_source == "clickhouse":
        return _fetch_clickhouse_actuals(
            start_time=start_time,
            end_time=end_time,
            query_template=CLICKHOUSE_METRICS_QUERY,
        )

    raise ValueError(
        f"Неизвестный CONTROL_PLANE_METRICS_SOURCE={effective_source!r}. "
        "Поддерживается: prometheus, clickhouse"
    )
