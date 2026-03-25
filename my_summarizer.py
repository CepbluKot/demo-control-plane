from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from settings import settings


LOGS_SQL_COLUMNS: tuple[str, str] = ("timestamp", "value")


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


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _resolve_service(anomaly: Optional[Dict[str, Any]]) -> str:
    if anomaly and anomaly.get("service"):
        return str(anomaly["service"])
    raise ValueError(
        "Missing anomaly['service'] for logs summarization. "
        "Pass service in anomaly payload."
    )


def _resolve_logs_query_template() -> str:
    template = str(settings.CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY).strip()
    if template:
        return template
    raise ValueError(
        "Set CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY in .env "
        "(full SQL string; optionally with placeholders)."
    )


def _render_logs_query(
    *,
    query_template: str,
    period_start: str,
    period_end: str,
    limit: int,
    offset: int,
    service: str,
) -> str:
    safe_limit = max(int(limit), 1)
    safe_offset = max(int(offset), 0)
    params = _SafeFormatDict(
        period_start=period_start,
        period_end=period_end,
        start=period_start,
        end=period_end,
        start_iso=period_start,
        end_iso=period_end,
        limit=safe_limit,
        page_limit=safe_limit,
        offset=safe_offset,
        service=service,
    )
    return query_template.strip().rstrip(";").format_map(params)


def _build_db_fetch_page(anomaly: Optional[Dict[str, Any]]) -> Callable[..., List[Dict[str, Any]]]:
    service = _resolve_service(anomaly)
    query_template = _resolve_logs_query_template()
    has_offset_placeholder = "{offset}" in query_template

    def _db_fetch_page(
        *,
        columns: Sequence[str],
        period_start: str,
        period_end: str,
        limit: int,
        offset: int,
    ) -> List[Dict[str, Any]]:
        del columns
        # Если offset-плейсхолдера нет, запрос считаем single-shot.
        if offset > 0 and not has_offset_placeholder:
            return []
        query = _render_logs_query(
            query_template=query_template,
            period_start=period_start,
            period_end=period_end,
            limit=limit,
            offset=offset,
            service=service,
        )
        try:
            from sqlalchemy import text
            from sqlalchemy_stuff.engine import LogsSession
        except Exception as exc:
            raise ImportError("sqlalchemy_stuff LogsSession is required for logs batch fetch") from exc

        session = LogsSession()
        try:
            result = session.execute(text(query))
            rows = result.mappings().all()
            if not rows:
                return []
            return [dict(row) for row in rows]
        finally:
            session.close()

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
    Batch logs fetch for a period using a full SQL query from env.
    Query should be configured in CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY.
    Supported placeholders:
      {period_start}, {period_end}, {start}, {end}, {limit}, {offset}, {service}

    If {offset} is missing in query template, request is treated as single-shot page.
    Query must return exactly two columns: timestamp, value.
    This adapter can be wired as CONTROL_PLANE_SUMMARIZER_CALLABLE.
    """
    start_iso, end_iso = _normalize_period(
        period_start=period_start,
        period_end=period_end,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    page_limit = int(settings.CONTROL_PLANE_LOGS_PAGE_LIMIT)

    db_fetch_page = _build_db_fetch_page(anomaly)

    offset = 0
    total_rows = 0
    pages = 0
    batch_summaries: List[str] = []

    while True:
        page = db_fetch_page(
            columns=LOGS_SQL_COLUMNS,
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
        missing_cols = [c for c in LOGS_SQL_COLUMNS if c not in page_df.columns]
        if missing_cols:
            raise ValueError(
                "CONTROL_PLANE_CLICKHOUSE_LOGS_QUERY must return columns timestamp and value "
                f"(missing: {', '.join(missing_cols)})"
            )
        ts_min = str(page_df["timestamp"].min()) if not page_df.empty else "n/a"
        ts_max = str(page_df["timestamp"].max()) if not page_df.empty else "n/a"

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
