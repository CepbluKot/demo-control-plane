from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _normalize_period(
    *,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> Tuple[str, str]:
    if period_start and period_end:
        return period_start, period_end
    if start_dt is not None and end_dt is not None:
        return start_dt.isoformat(), end_dt.isoformat()
    raise ValueError("Provide either period_start+period_end or start_dt+end_dt")


def _safe_identifier(name: str) -> str:
    if not _IDENT_RE.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


def _resolve_logs_table(service: str) -> str:
    by_service_raw = os.getenv("CONTROL_PLANE_LOGS_TABLE_BY_SERVICE_JSON", "{}").strip() or "{}"
    try:
        by_service = json.loads(by_service_raw)
    except Exception as exc:
        raise ValueError("CONTROL_PLANE_LOGS_TABLE_BY_SERVICE_JSON must be valid JSON object") from exc
    if not isinstance(by_service, dict):
        raise ValueError("CONTROL_PLANE_LOGS_TABLE_BY_SERVICE_JSON must be JSON object")

    table = by_service.get(service) or os.getenv("CONTROL_PLANE_LOGS_DEFAULT_TABLE", "").strip()
    if not table:
        raise ValueError(
            f"Logs table for service={service!r} not found. "
            "Set CONTROL_PLANE_LOGS_TABLE_BY_SERVICE_JSON or CONTROL_PLANE_LOGS_DEFAULT_TABLE."
        )
    return _safe_identifier(table)


def _resolve_service(anomaly: Optional[Dict[str, Any]]) -> str:
    if anomaly and anomaly.get("service"):
        return str(anomaly["service"])
    service_env = os.getenv("CONTROL_PLANE_LOGS_DEFAULT_SERVICE", "").strip()
    if service_env:
        return service_env
    raise ValueError(
        "Service is required for logs fetch. "
        "Set anomaly['service'] or CONTROL_PLANE_LOGS_DEFAULT_SERVICE."
    )


def _resolve_columns() -> List[str]:
    raw = os.getenv("CONTROL_PLANE_LOGS_COLUMNS", "timestamp,level,message")
    cols = [c.strip() for c in raw.split(",") if c.strip()]
    if not cols:
        raise ValueError("CONTROL_PLANE_LOGS_COLUMNS must not be empty")
    for col in cols:
        _safe_identifier(col)
    return cols


def _build_base_select(table: str, columns: Sequence[str]) -> str:
    template = os.getenv(
        "CONTROL_PLANE_LOGS_BASE_SELECT",
        "SELECT {columns} FROM {table}",
    )
    return template.format(table=table, columns=", ".join(columns))


def _build_logs_page_query(
    *,
    base_select: str,
    period_start: str,
    period_end: str,
    timestamp_column: str,
    order_by: str,
    limit: int,
    offset: int,
) -> str:
    ts_col = _safe_identifier(timestamp_column)
    order_col = _safe_identifier(order_by)
    safe_limit = max(int(limit), 1)
    safe_offset = max(int(offset), 0)
    return (
        "SELECT *\n"
        f"FROM ({base_select}) AS src\n"
        f"WHERE src.{ts_col} >= parseDateTimeBestEffort('{period_start}')\n"
        f"  AND src.{ts_col} < parseDateTimeBestEffort('{period_end}')\n"
        f"ORDER BY src.{order_col}\n"
        f"LIMIT {safe_limit} OFFSET {safe_offset}"
    )


def _build_db_fetch_page(anomaly: Optional[Dict[str, Any]]) -> Callable[..., List[Dict[str, Any]]]:
    try:
        import clickhouse_connect
    except Exception as exc:
        raise ImportError("clickhouse-connect is required for logs batch fetch") from exc

    host = os.getenv("CONTROL_PLANE_LOGS_CH_HOST", "localhost").strip()
    port = int(os.getenv("CONTROL_PLANE_LOGS_CH_PORT", "8123"))
    username = os.getenv("CONTROL_PLANE_LOGS_CH_USERNAME", "").strip() or None
    password = os.getenv("CONTROL_PLANE_LOGS_CH_PASSWORD", "").strip() or None
    database = os.getenv("CONTROL_PLANE_LOGS_CH_DATABASE", "default").strip() or None

    service = _resolve_service(anomaly)
    table = _resolve_logs_table(service)
    timestamp_column = os.getenv("CONTROL_PLANE_LOGS_TIMESTAMP_COLUMN", "timestamp").strip() or "timestamp"
    order_by = os.getenv("CONTROL_PLANE_LOGS_ORDER_BY", timestamp_column).strip() or timestamp_column
    _safe_identifier(timestamp_column)
    _safe_identifier(order_by)

    client = clickhouse_connect.get_client(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
    )

    def _db_fetch_page(
        *,
        columns: Sequence[str],
        period_start: str,
        period_end: str,
        limit: int,
        offset: int,
    ) -> List[Dict[str, Any]]:
        for col in columns:
            _safe_identifier(col)
        base_select = _build_base_select(table, columns)
        query = _build_logs_page_query(
            base_select=base_select,
            period_start=period_start,
            period_end=period_end,
            timestamp_column=timestamp_column,
            order_by=order_by,
            limit=limit,
            offset=offset,
        )
        df = client.query_df(query)
        if df.empty:
            return []
        return df.to_dict(orient="records")

    return _db_fetch_page


def summarize_logs(
    *,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
    anomaly: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Batch logs fetch for a period using per-service table mapping.
    This adapter can be wired as CONTROL_PLANE_SUMMARIZER_CALLABLE.
    """
    start_iso, end_iso = _normalize_period(
        period_start=period_start,
        period_end=period_end,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    columns = _resolve_columns()
    page_limit = int(os.getenv("CONTROL_PLANE_LOGS_PAGE_LIMIT", "1000"))

    db_fetch_page = _build_db_fetch_page(anomaly)

    offset = 0
    total_rows = 0
    pages = 0
    batch_summaries: List[str] = []

    while True:
        page = db_fetch_page(
            columns=columns,
            period_start=start_iso,
            period_end=end_iso,
            limit=page_limit,
            offset=offset,
        )
        if not page:
            break
        pages += 1
        rows = len(page)
        total_rows += rows
        offset += rows

        page_df = pd.DataFrame(page)
        ts_col = os.getenv("CONTROL_PLANE_LOGS_TIMESTAMP_COLUMN", "timestamp")
        ts_min = str(page_df[ts_col].min()) if ts_col in page_df.columns and not page_df.empty else "n/a"
        ts_max = str(page_df[ts_col].max()) if ts_col in page_df.columns and not page_df.empty else "n/a"

        joined = " ".join(str(v).lower() for row in page for v in row.values())
        signal_hits = sum(1 for kw in ("error", "exception", "timeout", "failed", "fatal") if kw in joined)
        batch_summaries.append(
            f"[Batch {pages}] rows={rows}, ts=[{ts_min}..{ts_max}], problem_signals={signal_hits}"
        )
        if rows < page_limit:
            break

    if total_rows == 0:
        return {
            "summary": "Нет логов за указанный период.",
            "chunk_summaries": [],
            "pages_fetched": 0,
            "rows_processed": 0,
        }

    service = _resolve_service(anomaly)
    final_summary = (
        f"Сервис: {service}. Период: [{start_iso}, {end_iso}). "
        f"Прочитано {total_rows} строк логов батчами ({pages} стр.). "
        "Детальный LLM-анализ можно подключить поверх этого fetcher-а."
    )

    return {
        "summary": final_summary,
        "chunk_summaries": batch_summaries,
        "pages_fetched": pages,
        "rows_processed": total_rows,
        "source": "clickhouse_logs_batch_fetcher",
    }
