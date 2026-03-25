import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from .config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_QUERY,
    CLICKHOUSE_SECURE,
    CLICKHOUSE_TIMESTAMP_COLUMN,
    CLICKHOUSE_TIMESTAMP_UNIT,
    CLICKHOUSE_USERNAME,
    CLICKHOUSE_VALUE_COLUMN,
    CLICKHOUSE_VALUE_SCALE,
    METRICS_SOURCE,
)
from .prometheus_io import fetch_prometheus_df
from .trace import log_dataframe, log_event

logger = logging.getLogger(__name__)


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_identifier(name: str) -> str:
    if not _IDENT_RE.match(name):
        raise ValueError(
            f"Некорректное имя колонки: {name!r}. Разрешены только [A-Za-z0-9_], начиная с буквы/_."
        )
    return name


def _render_clickhouse_query(
    query_template: str,
    start_time: datetime,
    end_time: datetime,
    *,
    limit: int,
    timestamp_column: str,
) -> tuple[str, str]:
    params = _SafeFormatDict(
        start=start_time.isoformat(),
        end=end_time.isoformat(),
        start_iso=start_time.isoformat(),
        end_iso=end_time.isoformat(),
        start_ts=int(start_time.timestamp()),
        end_ts=int(end_time.timestamp()),
    )
    query_template = query_template.strip().rstrip(";")
    # Backward compatibility: if template already has time placeholders,
    # treat it as full query and only format.
    if any(token in query_template for token in ("{start", "{end", "{start_ts}", "{end_ts}")):
        return query_template.format_map(params), "full"

    ts_col = _safe_identifier(timestamp_column)
    safe_limit = max(int(limit), 1)
    wrapped = (
        "SELECT *\n"
        f"FROM ({query_template}) AS src\n"
        f"WHERE src.{ts_col} >= parseDateTimeBestEffort('{params['start']}')\n"
        f"  AND src.{ts_col} < parseDateTimeBestEffort('{params['end']}')\n"
        f"ORDER BY src.{ts_col}\n"
        f"LIMIT {safe_limit}"
    )
    return wrapped, "base"


def _fetch_clickhouse_actuals(
    *,
    start_time: datetime,
    end_time: datetime,
    query_template: str,
    limit: int,
) -> pd.DataFrame:
    try:
        import clickhouse_connect
    except Exception as exc:
        raise ImportError(
            "Для чтения фактических метрик из ClickHouse нужен пакет clickhouse-connect"
        ) from exc

    if not CLICKHOUSE_HOST:
        raise ValueError(
            "Для METRICS_SOURCE=clickhouse укажи CONTROL_PLANE_CLICKHOUSE_HOST в .env"
        )
    if not query_template:
        raise ValueError(
            "Для METRICS_SOURCE=clickhouse укажи CONTROL_PLANE_CLICKHOUSE_QUERY в .env"
        )

    log_event(
        logger,
        "fetch_actuals.clickhouse.start",
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        database=CLICKHOUSE_DATABASE,
        secure=CLICKHOUSE_SECURE,
        start=start_time.isoformat(),
        end=end_time.isoformat(),
    )
    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USERNAME or None,
        password=CLICKHOUSE_PASSWORD or None,
        database=CLICKHOUSE_DATABASE or None,
        secure=CLICKHOUSE_SECURE,
    )
    query, query_mode = _render_clickhouse_query(
        query_template,
        start_time,
        end_time,
        limit=limit,
        timestamp_column=CLICKHOUSE_TIMESTAMP_COLUMN,
    )
    log_event(
        logger,
        "fetch_actuals.clickhouse.query",
        query_mode=query_mode,
        limit=limit,
        query_preview=query[:600],
    )
    df = client.query_df(query)

    if df.empty:
        log_event(logger, "fetch_actuals.clickhouse.empty")
        return pd.DataFrame(columns=["timestamp", "value"])

    if CLICKHOUSE_TIMESTAMP_COLUMN != "timestamp" and CLICKHOUSE_TIMESTAMP_COLUMN in df.columns:
        df = df.rename(columns={CLICKHOUSE_TIMESTAMP_COLUMN: "timestamp"})
    if CLICKHOUSE_VALUE_COLUMN != "value" and CLICKHOUSE_VALUE_COLUMN in df.columns:
        df = df.rename(columns={CLICKHOUSE_VALUE_COLUMN: "value"})

    if "timestamp" not in df.columns or "value" not in df.columns:
        raise ValueError(
            "ClickHouse query должен вернуть колонки timestamp и value "
            f"(или переименуй через env: timestamp={CLICKHOUSE_TIMESTAMP_COLUMN}, "
            f"value={CLICKHOUSE_VALUE_COLUMN})"
        )

    if CLICKHOUSE_TIMESTAMP_UNIT:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit=CLICKHOUSE_TIMESTAMP_UNIT,
            utc=True,
            errors="coerce",
        )
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if CLICKHOUSE_VALUE_SCALE != 1.0:
        df["value"] = df["value"] * float(CLICKHOUSE_VALUE_SCALE)

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
            query_template=CLICKHOUSE_QUERY,
            limit=max_points,
        )

    raise ValueError(
        f"Неизвестный CONTROL_PLANE_METRICS_SOURCE={effective_source!r}. "
        "Поддерживается: prometheus, clickhouse"
    )
