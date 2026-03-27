from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

MSK = timezone(timedelta(hours=3))
import heapq
import json
import logging
import math
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

import pandas as pd
import streamlit as st


MAX_EVENT_LINES = 160
MAX_RENDERED_BATCHES = 10
MAX_LOG_ROWS_PREVIEW = 80
MAX_METRICS_ROWS_TOTAL = 50_000  # cap across all metric queries to avoid OOM
MAX_LLM_TIMELINE_ROWS = 400
LAST_STATE_SESSION_KEY = "logs_summary_last_result_state"
RUNNING_SESSION_KEY = "logs_summary_running"
RUN_PARAMS_SESSION_KEY = "logs_summary_run_params"
FORM_ERROR_SESSION_KEY = "logs_summary_form_error"
DEFAULT_SUMMARY_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "message",
    "level",
    "service",
    "pod",
    "container",
    "node",
    "cluster",
    "value",
)


def _timeline_row_color(status: str) -> str:
    normalized = status.strip().lower()
    if normalized == "ok":
        return "#e8f7ee"
    if normalized in ("error", "failed", "fail"):
        return "#fdecec"
    if normalized in ("retry", "retrying"):
        return "#fff4e5"
    if normalized == "started":
        return "#eaf2ff"
    return ""


def _style_llm_timeline(df: pd.DataFrame) -> Optional[Any]:
    if df.empty or "status" not in df.columns:
        return None

    def _row_style(row: pd.Series) -> List[str]:
        color = _timeline_row_color(str(row.get("status", "")))
        if not color:
            return ["" for _ in row]
        return [f"background-color: {color}" for _ in row]

    try:
        return df.style.apply(_row_style, axis=1)
    except Exception:
        return None


def _format_datetime_with_tz(value: Any) -> str:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return str(value)
    return ts.tz_convert(MSK).strftime("%Y-%m-%d %H:%M:%S MSK")


def _format_table_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    timestamp_like = {
        "timestamp",
        "ts",
        "time",
        "datetime",
        "period_start",
        "period_end",
        "start",
        "end",
    }
    for col in out.columns:
        col_name = str(col).strip().lower()
        if col_name not in timestamp_like and "timestamp" not in col_name:
            continue
        def _fmt(value: Any) -> str:
            ts = pd.to_datetime(value, utc=True, errors="coerce")
            if pd.isna(ts):
                return str(value)
            return ts.tz_convert(MSK).strftime("%Y-%m-%d %H:%M:%S.%f MSK")

        out[col] = out[col].apply(_fmt).astype(str)
    return out


def _format_eta_seconds(seconds: float) -> str:
    safe = max(int(seconds), 0)
    hours = safe // 3600
    minutes = (safe % 3600) // 60
    secs = safe % 60
    if hours > 0:
        return f"{hours}ч {minutes}м {secs}с"
    if minutes > 0:
        return f"{minutes}м {secs}с"
    return f"{secs}с"


def _format_attempts_total(total_attempts: Any) -> str:
    total = pd.to_numeric(total_attempts, errors="coerce")
    if pd.isna(total):
        return "?"
    if int(total) <= 0:
        return "∞"
    return str(int(total))


def _format_attempts_pair(attempt: Any, total_attempts: Any) -> str:
    current = pd.to_numeric(attempt, errors="coerce")
    current_text = "?" if pd.isna(current) else str(int(current))
    return f"{current_text}/{_format_attempts_total(total_attempts)}"


def _is_read_timeout_error(error_text: Any) -> bool:
    text = str(error_text or "").strip().lower()
    if not text:
        return False
    return (
        "read timed out" in text
        or "read timeout" in text
        or "readtimeout" in text
    )


def _normalize_logs_fetch_mode(value: Any) -> str:
    raw = str(value or "time_window").strip().lower()
    aliases = {
        "time_window": "time_window",
        "window": "time_window",
        "lookback": "time_window",
        "tail_n_logs": "tail_n_logs",
        "tail_n": "tail_n_logs",
        "last_n_logs": "tail_n_logs",
    }
    return aliases.get(raw, "time_window")


def _estimate_eta(state: Dict[str, Any], event: str, payload: Dict[str, Any]) -> None:
    started_mono = state.get("started_monotonic")
    if started_mono is None:
        return

    status = str(state.get("status", "queued"))
    now_mono = time.monotonic()
    elapsed = max(now_mono - float(started_mono), 0.001)
    state["elapsed_seconds"] = float(elapsed)

    ratio: Optional[float] = None
    remaining: Optional[float] = None

    # Primary: timestamp-based progress.  Works for both OFFSET and keyset pagination,
    # doesn't require pre-counting rows.  Rate is measured in "log-seconds per real second".
    period_start_ts = pd.to_datetime(state.get("period_start"), utc=True, errors="coerce")
    period_end_ts = pd.to_datetime(state.get("period_end"), utc=True, errors="coerce")
    last_batch_ts = pd.to_datetime(state.get("last_batch_ts"), utc=True, errors="coerce")

    if (
        not pd.isna(period_start_ts)
        and not pd.isna(period_end_ts)
        and not pd.isna(last_batch_ts)
    ):
        total_span = (period_end_ts - period_start_ts).total_seconds()
        done_span = max((last_batch_ts - period_start_ts).total_seconds(), 0.0)

        if total_span > 0:
            ratio = min(max(done_span / total_span, 0.0), 0.99)
            remaining_span = max(total_span - done_span, 0.0)

            # Track (real_time, log_seconds_done) samples for rate estimation
            samples = state.setdefault("progress_samples", [])
            if not samples or abs(float(samples[-1][1]) - done_span) >= 1.0:
                samples.append((now_mono, done_span))
            if len(samples) > 40:
                state["progress_samples"] = samples[-40:]
                samples = state["progress_samples"]

            if len(samples) >= 2:
                oldest_t, oldest_done = samples[0]
                newest_t, newest_done = samples[-1]
                delta_t = float(newest_t) - float(oldest_t)
                delta_done = float(newest_done) - float(oldest_done)
                if delta_t > 0 and delta_done > 0:
                    rate = delta_done / delta_t  # log-seconds per real-second
                    state["log_seconds_per_second"] = float(rate)
                    remaining = remaining_span / rate
            if remaining is None and state.get("log_seconds_per_second"):
                rate = float(state["log_seconds_per_second"])
                if rate > 0:
                    remaining = remaining_span / rate

    # Fallback: batch/reduce estimates when last_batch_ts is not yet available
    elif status == "map":
        batch_total = payload.get("batch_total")
        batch_index = payload.get("batch_index")
        if batch_total is not None and batch_index is not None and int(batch_total) > 0:
            ratio = 0.10 + 0.70 * ((int(batch_index) + 1) / int(batch_total))
        else:
            ratio = 0.20
    elif status == "reduce":
        group_total = payload.get("group_total")
        group_index = payload.get("group_index")
        if group_total is not None and group_index is not None and int(group_total) > 0:
            ratio = 0.80 + 0.18 * ((int(group_index) + 1) / int(group_total))
        else:
            ratio = 0.85
    elif status == "summarizing":
        ratio = 0.05

    if status in ("done", "error"):
        state["eta_seconds_left"] = 0
        state["eta_finish_at"] = datetime.now(timezone.utc).isoformat()
        return

    if remaining is not None and remaining >= 0:
        finish_dt = datetime.now(timezone.utc) + timedelta(seconds=remaining)
        state["eta_seconds_left"] = int(remaining)
        state["eta_finish_at"] = finish_dt.isoformat()
        return

    if ratio is None or ratio <= 0:
        state["eta_seconds_left"] = None
        state["eta_finish_at"] = None
        return

    ratio = min(max(ratio, 0.01), 0.99)
    total_estimated = elapsed / ratio
    remaining = max(total_estimated - elapsed, 0.0)
    finish_dt = datetime.now(timezone.utc) + timedelta(seconds=remaining)
    state["eta_seconds_left"] = int(remaining)
    state["eta_finish_at"] = finish_dt.isoformat()


@dataclass(frozen=True)
class LogsSummaryPageDeps:
    logger: logging.Logger
    db_batch_size: int
    llm_batch_size: int
    test_mode: bool
    loopback_minutes: int
    logs_tail_limit: int
    period_log_summarizer_cls: Any
    summarizer_config_cls: Any
    make_llm_call: Callable[..., Callable[[str], str]]
    query_logs_df: Callable[[str], pd.DataFrame]
    query_metrics_df: Callable[[str], pd.DataFrame]
    render_scrollable_text: Callable[..., None]
    render_pretty_summary_text: Callable[..., None]
    infer_batch_period: Callable[[Dict[str, Any]], tuple[Optional[str], Optional[str]]]
    summary_text_height: int
    final_text_height: int
    logs_batch_table_height: int
    sql_textarea_height: int
    default_sql_query: str
    default_metrics_query: str
    output_dir: Path
    logs_fetch_mode: str = "time_window"
    map_workers: int = 1
    max_retries: int = -1
    llm_timeout: int = 600


@dataclass
class _StreamingQueryCursor:
    query_index: int
    spec: Dict[str, Any]
    next_offset: int = 0
    page_records: List[Dict[str, Any]] = field(default_factory=list)
    page_pos: int = 0
    has_more: bool = True
    exhausted: bool = False
    failed: bool = False
    row_seq: int = 0
    # Keyset pagination state: max timestamp seen in the last fetched page.
    # None means "not yet fetched" — first page uses period_start as last_ts.
    last_ts: Optional[str] = None


class _StreamingLogsMerger:
    def __init__(
        self,
        *,
        query_specs: List[Dict[str, Any]],
        query_logs_df: Callable[[str], pd.DataFrame],
        register_query_error: Callable[[str], None],
        logger: logging.Logger,
    ) -> None:
        self._query_logs_df = query_logs_df
        self._register_query_error = register_query_error
        self._logger = logger
        self._cursors: List[_StreamingQueryCursor] = [
            _StreamingQueryCursor(query_index=idx, spec=spec)
            for idx, spec in enumerate(query_specs)
        ]
        self._heap: List[tuple[int, int, int, Dict[str, Any]]] = []
        self._started = False
        self._period_key: Optional[tuple[str, str]] = None
        self._global_emitted = 0

    def _reset(self, *, period_start: str, period_end: str) -> None:
        self._heap = []
        self._global_emitted = 0
        self._period_key = (period_start, period_end)
        self._started = True
        self._cursors = [
            _StreamingQueryCursor(query_index=idx, spec=cursor.spec)
            for idx, cursor in enumerate(self._cursors)
        ]

    def _load_next_page(
        self,
        cursor: _StreamingQueryCursor,
        *,
        period_start: str,
        period_end: str,
        fetch_limit: int,
    ) -> bool:
        if cursor.exhausted or cursor.failed:
            return False
        if not cursor.has_more:
            cursor.exhausted = True
            return False

        template = str(cursor.spec["template"])
        uses_keyset = "{last_ts}" in template.lower()

        # In rare cases page can contain only bad timestamps after normalization.
        # Continue fetching until we get at least one valid row or source is exhausted.
        for _ in range(20):
            query = _build_query_for_template(
                template=template,
                uses_template=bool(cursor.spec["uses_template"]),
                uses_paging_template=bool(cursor.spec["uses_paging_template"]),
                period_start_iso=period_start,
                period_end_iso=period_end,
                limit=fetch_limit,
                offset=cursor.next_offset,
                last_ts=cursor.last_ts,  # None on first page → defaults to period_start
            )
            try:
                df = self._query_logs_df(query)
            except Exception as exc:  # noqa: BLE001
                cursor.failed = True
                cursor.exhausted = True
                if uses_keyset:
                    self._register_query_error(
                        f"Запрос #{cursor.query_index + 1} page(last_ts={cursor.last_ts}, "
                        f"limit={fetch_limit}) упал: {exc}"
                    )
                else:
                    self._register_query_error(
                        f"Запрос #{cursor.query_index + 1} page(offset={cursor.next_offset}, "
                        f"limit={fetch_limit}) упал: {exc}"
                    )
                self._logger.warning(
                    "logs_summary_page.query_failed[stream:%s]: offset=%s limit=%s err=%s",
                    cursor.query_index + 1,
                    cursor.next_offset,
                    fetch_limit,
                    exc,
                )
                return False

            raw_rows = int(len(df))
            if uses_keyset:
                # In keyset mode offset tracking is not used for queries,
                # but we still increment it to detect empty pages / has_more correctly.
                cursor.next_offset += raw_rows
            else:
                cursor.next_offset += raw_rows
            cursor.has_more = raw_rows >= fetch_limit
            if raw_rows == 0:
                cursor.exhausted = True
                return False

            sorted_df = _sort_df_by_timestamp(df)
            if not sorted_df.empty:
                # Advance keyset cursor to last timestamp in the fetched page
                if uses_keyset and "timestamp" in sorted_df.columns:
                    max_ts = pd.to_datetime(
                        sorted_df["timestamp"], utc=True, errors="coerce"
                    ).max()
                    if not pd.isna(max_ts):
                        cursor.last_ts = max_ts.isoformat()
                cursor.page_records = [dict(row) for row in sorted_df.to_dict(orient="records")]
                cursor.page_pos = 0
                return True

            if not cursor.has_more:
                cursor.exhausted = True
                return False

        self._register_query_error(
            f"Запрос #{cursor.query_index + 1} пропущен: слишком много страниц без валидного timestamp"
        )
        cursor.exhausted = True
        return False

    def _push_next_row(
        self,
        cursor: _StreamingQueryCursor,
        *,
        period_start: str,
        period_end: str,
        fetch_limit: int,
    ) -> None:
        while True:
            if cursor.exhausted or cursor.failed:
                return

            if cursor.page_pos >= len(cursor.page_records):
                loaded = self._load_next_page(
                    cursor,
                    period_start=period_start,
                    period_end=period_end,
                    fetch_limit=fetch_limit,
                )
                if not loaded:
                    return

            row = cursor.page_records[cursor.page_pos]
            cursor.page_pos += 1
            ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
            if pd.isna(ts):
                continue
            cursor.row_seq += 1
            heapq.heappush(
                self._heap,
                (int(ts.value), cursor.query_index, cursor.row_seq, row),
            )
            return

    def _ensure_started(
        self,
        *,
        period_start: str,
        period_end: str,
        fetch_limit: int,
    ) -> None:
        if (not self._started) or self._period_key != (period_start, period_end):
            self._reset(period_start=period_start, period_end=period_end)
            for cursor in self._cursors:
                self._push_next_row(
                    cursor,
                    period_start=period_start,
                    period_end=period_end,
                    fetch_limit=fetch_limit,
                )

    def next_page(
        self,
        *,
        period_start: str,
        period_end: str,
        limit: int,
        offset: int,
    ) -> List[Dict[str, Any]]:
        safe_limit = max(int(limit), 1)
        safe_offset = max(int(offset), 0)
        self._ensure_started(
            period_start=period_start,
            period_end=period_end,
            fetch_limit=safe_limit,
        )

        if safe_offset != self._global_emitted:
            self._logger.warning(
                "logs_summary_page.streaming_offset_mismatch: expected=%s got=%s; resyncing stream",
                self._global_emitted,
                safe_offset,
            )
            if safe_offset < self._global_emitted:
                self._reset(period_start=period_start, period_end=period_end)
                for cursor in self._cursors:
                    self._push_next_row(
                        cursor,
                        period_start=period_start,
                        period_end=period_end,
                        fetch_limit=safe_limit,
                    )
            while self._global_emitted < safe_offset and self._heap:
                _, cursor_idx, _, _ = heapq.heappop(self._heap)
                self._global_emitted += 1
                self._push_next_row(
                    self._cursors[cursor_idx],
                    period_start=period_start,
                    period_end=period_end,
                    fetch_limit=safe_limit,
                )
            if self._global_emitted != safe_offset:
                return []

        multi_source = len(self._cursors) > 1
        out: List[Dict[str, Any]] = []
        while len(out) < safe_limit and self._heap:
            _, cursor_idx, _, row = heapq.heappop(self._heap)
            row_out = dict(row)
            if multi_source:
                row_out["_source"] = str(self._cursors[cursor_idx].spec.get("label", f"query_{cursor_idx + 1}"))
            out.append(row_out)
            self._global_emitted += 1
            self._push_next_row(
                self._cursors[cursor_idx],
                period_start=period_start,
                period_end=period_end,
                fetch_limit=safe_limit,
            )
        return out


def _escape_sql_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "''")


def _normalize_sql_query_text(query: str) -> str:
    normalized = str(query).replace("\r\n", "\n").replace("\r", "\n").strip()
    while normalized.endswith(";"):
        normalized = normalized[:-1].rstrip()
    return normalized


def _split_query_templates(raw_query: str) -> List[str]:
    normalized = str(raw_query).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    # Preferred explicit delimiter between multiple SQL templates.
    by_marker = re.split(r"(?im)^\s*(?:--\s*QUERY\s*--|/\*\s*QUERY\s*\*/|---)\s*$", normalized)
    marker_parts = [_normalize_sql_query_text(part) for part in by_marker if part and part.strip()]
    if len(marker_parts) > 1:
        return marker_parts

    # Fallback: split by semicolon terminators.
    by_semicolon = [p.strip() for p in re.split(r";\s*(?:\n|$)", normalized) if p.strip()]
    if len(by_semicolon) > 1:
        return [_normalize_sql_query_text(part) for part in by_semicolon]

    return [_normalize_sql_query_text(normalized)]


def _new_query_item(text: str = "") -> Dict[str, str]:
    return {"id": uuid4().hex, "text": str(text)}


def _normalize_query_items(
    raw_items: Any,
    *,
    default_value: str,
    min_items: int,
) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if isinstance(raw_items, list):
        for item in raw_items:
            if isinstance(item, dict):
                item_id = str(item.get("id") or uuid4().hex)
                item_text = str(item.get("text") or "")
                items.append({"id": item_id, "text": item_text})
            elif isinstance(item, str):
                items.append(_new_query_item(item))

    if min_items > 0:
        while len(items) < min_items:
            items.append(_new_query_item(default_value if not items else ""))
    return items


def _extract_queries_from_items(raw_items: Any) -> List[str]:
    if not isinstance(raw_items, list):
        return []
    queries: List[str] = []
    for item in raw_items:
        if isinstance(item, dict):
            text = str(item.get("text") or "")
        else:
            text = str(item or "")
        if text.strip():
            queries.append(text)
    return queries


def _query_uses_any_placeholders(base_query: str) -> bool:
    lowered = base_query.lower()
    return any(
        token in lowered
        for token in (
            "{period_start}",
            "{period_end}",
            "{last_ts}",
            "{start}",
            "{end}",
            "{start_iso}",
            "{end_iso}",
            "{limit}",
            "{offset}",
        )
    )


def _query_uses_paging_placeholders(base_query: str) -> bool:
    lowered = base_query.lower()
    # Keyset pagination via {last_ts} also means the template manages its own paging
    if "{last_ts}" in lowered:
        return True
    return "{limit}" in lowered and "{offset}" in lowered


def _choose_summary_columns(available_columns: List[str]) -> List[str]:
    if not available_columns:
        return list(DEFAULT_SUMMARY_COLUMNS)

    normalized = [str(col) for col in available_columns]
    lower_to_actual = {col.lower(): col for col in normalized}
    selected: List[str] = []

    for preferred in DEFAULT_SUMMARY_COLUMNS:
        actual = lower_to_actual.get(preferred.lower())
        if actual:
            selected.append(actual)

    if selected:
        return selected

    # If preferred columns are absent, keep a stable subset of source columns.
    return normalized[: min(len(normalized), 12)]


def _column_set(df: pd.DataFrame) -> set[str]:
    return {str(col).strip().lower() for col in df.columns if str(col).strip()}


def _validate_logs_merge_schema(previews: List[Dict[str, Any]]) -> List[str]:
    if not previews:
        return []

    errors: List[str] = []
    first_df = previews[0].get("df")
    if first_df is None or not isinstance(first_df, pd.DataFrame):
        return ["Не удалось валидировать логи: preview первого запроса отсутствует."]

    base_cols = _column_set(first_df)
    if "timestamp" not in base_cols:
        errors.append("Логи: в первом SQL отсутствует обязательная колонка `timestamp`.")

    for item in previews:
        idx = int(item.get("idx", 0)) + 1
        df = item.get("df")
        if df is None or not isinstance(df, pd.DataFrame):
            errors.append(f"Логи: запрос #{idx} не вернул preview-таблицу.")
            continue
        cols = _column_set(df)
        if "timestamp" not in cols:
            errors.append(f"Логи: запрос #{idx} не содержит колонку `timestamp`.")
        if cols != base_cols:
            missing = sorted(base_cols - cols)
            extra = sorted(cols - base_cols)
            errors.append(
                "Логи: запрос #{idx} имеет несовместимую схему. "
                "missing={missing}; extra={extra}".format(
                    idx=idx,
                    missing=missing,
                    extra=extra,
                )
            )
    return errors


def _has_metrics_value_column(df: pd.DataFrame) -> bool:
    cols_lower = {str(c).lower(): str(c) for c in df.columns}
    for candidate in ("value", "metric_value", "predicted"):
        if candidate in cols_lower:
            return True
    for col in df.columns:
        col_lower = str(col).lower()
        if col_lower in {"timestamp", "service", "metric_service"}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            return True
    return False


def _validate_metrics_merge_schema(previews: List[Dict[str, Any]]) -> List[str]:
    if not previews:
        return []

    errors: List[str] = []
    for item in previews:
        idx = int(item.get("idx", 0)) + 1
        df = item.get("df")
        if df is None or not isinstance(df, pd.DataFrame):
            errors.append(f"Метрики: запрос #{idx} не вернул preview-таблицу.")
            continue
        cols = _column_set(df)
        if "timestamp" not in cols:
            errors.append(f"Метрики: запрос #{idx} не содержит колонку `timestamp`.")
        if not _has_metrics_value_column(df):
            errors.append(
                f"Метрики: запрос #{idx} не содержит числовую value-колонку "
                "(ожидается `value`/`metric_value`/`predicted` или любой numeric столбец)."
            )
    return errors


def _sort_df_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "timestamp" not in df.columns:
        return df
    out = df.copy()
    out["__cp_ts"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["__cp_ts"]).sort_values("__cp_ts", kind="mergesort")
    return out.drop(columns=["__cp_ts"]).reset_index(drop=True)


def _normalize_metrics_df(chunk: pd.DataFrame, *, default_service: str) -> pd.DataFrame:
    if chunk is None or chunk.empty:
        return pd.DataFrame(columns=["timestamp", "value", "service"])
    if "timestamp" not in chunk.columns:
        return pd.DataFrame(columns=["timestamp", "value", "service"])

    out = chunk.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"])
    if out.empty:
        return pd.DataFrame(columns=["timestamp", "value", "service"])

    value_col: Optional[str] = None
    for candidate in ("value", "metric_value", "predicted"):
        if candidate in out.columns:
            value_col = candidate
            break
    if value_col is None:
        numeric_candidates = [
            c
            for c in out.columns
            if c != "timestamp" and pd.api.types.is_numeric_dtype(out[c])
        ]
        if numeric_candidates:
            value_col = numeric_candidates[0]
    if value_col is None:
        return pd.DataFrame(columns=["timestamp", "value", "service"])

    out["value"] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=["value"])
    if out.empty:
        return pd.DataFrame(columns=["timestamp", "value", "service"])

    if "service" in out.columns:
        out["service"] = out["service"].astype(str).replace("", default_service)
    elif "metric_service" in out.columns:
        out["service"] = out["metric_service"].astype(str).replace("", default_service)
    else:
        out["service"] = default_service

    out = out[["timestamp", "value", "service"]].copy()
    return _sort_df_by_timestamp(out)


def _build_metrics_context(metrics_df: pd.DataFrame, *, max_services: int = 12) -> str:
    if metrics_df is None or metrics_df.empty:
        return ""
    if "timestamp" not in metrics_df.columns or "value" not in metrics_df.columns:
        return ""

    normalized = metrics_df.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="coerce")
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    if "service" not in normalized.columns:
        normalized["service"] = "unknown-service"
    normalized["service"] = normalized["service"].astype(str)
    normalized = normalized.dropna(subset=["timestamp", "value"])
    if normalized.empty:
        return ""

    lines: List[str] = [
        "METRICS CONTEXT:",
        (
            f"Total metric rows: {len(normalized)}; "
            f"services: {normalized['service'].nunique()}."
        ),
    ]

    grouped = normalized.groupby("service", sort=True)
    for idx, (service, group) in enumerate(grouped):
        if idx >= max_services:
            lines.append(f"... truncated services after {max_services}")
            break
        g = group.sort_values("timestamp")
        first_ts = _format_datetime_with_tz(g["timestamp"].iloc[0])
        last_ts = _format_datetime_with_tz(g["timestamp"].iloc[-1])
        first_val = float(g["value"].iloc[0])
        last_val = float(g["value"].iloc[-1])
        delta = last_val - first_val
        lines.append(
            (
                f"[{service}] rows={len(g)} period={first_ts}->{last_ts} "
                f"min={g['value'].min():.3f} max={g['value'].max():.3f} "
                f"mean={g['value'].mean():.3f} last={last_val:.3f} delta={delta:+.3f}"
            )
        )

        diffs = g["value"].diff().abs().fillna(0.0)
        top_jumps = diffs.nlargest(min(2, len(diffs)))
        for j in top_jumps.index:
            jump_ts = _format_datetime_with_tz(g.loc[j, "timestamp"])
            jump_val = float(g.loc[j, "value"])
            jump_diff = float(diffs.loc[j])
            lines.append(
                f"  jump@{jump_ts}: value={jump_val:.3f}, abs_delta={jump_diff:.3f}"
            )

    return "\n".join(lines)


def _build_demo_metrics_for_window(
    *,
    period_start_dt: datetime,
    period_end_dt: datetime,
    service_label: str,
    total_points: int = 180,
) -> pd.DataFrame:
    safe_points = max(int(total_points), 20)
    window_seconds = max(int((period_end_dt - period_start_dt).total_seconds()), safe_points + 1)
    rows: List[Dict[str, Any]] = []
    seed_phase = (abs(hash(service_label)) % 17) / 7.0
    for i in range(safe_points):
        sec_offset = int((i + 1) * window_seconds / (safe_points + 1))
        ts = period_start_dt + timedelta(seconds=sec_offset)
        base = 100.0 + 5.0 * math.sin((i / 8.0) + seed_phase) + 0.08 * i
        if i % 57 == 0:
            base += 12.0
        rows.append(
            {
                "timestamp": ts.isoformat(),
                "value": float(base),
                "service": service_label,
            }
        )
    return pd.DataFrame(rows)


def _render_query_template(
    *,
    query_template: str,
    period_start_iso: str,
    period_end_iso: str,
    limit: int,
    offset: int,
    last_ts: Optional[str] = None,
) -> str:
    safe_limit = max(int(limit), 1)
    safe_offset = max(int(offset), 0)
    safe_start = _escape_sql_literal(period_start_iso)
    safe_end = _escape_sql_literal(period_end_iso)
    # For keyset pagination: last_ts defaults to period_start so the first page
    # returns rows from the beginning of the requested window.
    safe_last_ts = _escape_sql_literal(last_ts if last_ts is not None else period_start_iso)
    rendered = _normalize_sql_query_text(query_template)
    replacements = {
        "{period_start}": safe_start,
        "{period_end}": safe_end,
        "{start}": safe_start,
        "{end}": safe_end,
        "{start_iso}": safe_start,
        "{end_iso}": safe_end,
        "{limit}": str(safe_limit),
        "{offset}": str(safe_offset),
        "{page_limit}": str(safe_limit),
        "{last_ts}": safe_last_ts,
        "{PERIOD_START}": safe_start,
        "{PERIOD_END}": safe_end,
        "{START}": safe_start,
        "{END}": safe_end,
        "{START_ISO}": safe_start,
        "{END_ISO}": safe_end,
        "{LIMIT}": str(safe_limit),
        "{OFFSET}": str(safe_offset),
        "{PAGE_LIMIT}": str(safe_limit),
        "{LAST_TS}": safe_last_ts,
    }
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def _wrap_with_limit_offset(*, query: str, limit: int, offset: int) -> str:
    safe_limit = max(int(limit), 1)
    safe_offset = max(int(offset), 0)
    base = _normalize_sql_query_text(query)
    return (
        "SELECT * FROM ("
        f"{base}"
        ") AS cp_logs_page "
        "ORDER BY timestamp ASC "
        f"LIMIT {safe_limit} OFFSET {safe_offset}"
    )


def _build_window_query_for_plain_sql(
    *,
    base_query: str,
    period_start_iso: str,
    period_end_iso: str,
    limit: int,
    offset: int,
) -> str:
    safe_limit = max(int(limit), 1)
    safe_offset = max(int(offset), 0)
    base = _strip_trailing_limit_offset(_normalize_sql_query_text(base_query))
    start_escaped = _escape_sql_literal(period_start_iso)
    end_escaped = _escape_sql_literal(period_end_iso)
    # Only add ORDER BY to the inner subquery if the user's SQL doesn't already have one.
    # The outermost ORDER BY is always required for deterministic LIMIT/OFFSET in ClickHouse.
    inner_order = "" if "order by" in base.lower() else " ORDER BY timestamp ASC"
    return (
        "SELECT * FROM ("
        "SELECT * FROM ("
        f"{base}"
        ") AS cp_src "
        f"WHERE timestamp >= parseDateTimeBestEffort('{start_escaped}') "
        f"AND timestamp < parseDateTimeBestEffort('{end_escaped}')"
        f"{inner_order}"
        ") AS cp_window "
        "ORDER BY timestamp ASC "
        f"LIMIT {safe_limit} OFFSET {safe_offset}"
    )


def _build_query_for_template(
    *,
    template: str,
    uses_template: bool,
    uses_paging_template: bool,
    period_start_iso: str,
    period_end_iso: str,
    limit: int,
    offset: int,
    last_ts: Optional[str] = None,
) -> str:
    if uses_template:
        rendered_query = _render_query_template(
            query_template=template,
            period_start_iso=period_start_iso,
            period_end_iso=period_end_iso,
            limit=limit,
            offset=offset,
            last_ts=last_ts,
        )
        if uses_paging_template:
            return rendered_query
        rendered_without_limit = _strip_trailing_limit_offset(rendered_query)
        return _wrap_with_limit_offset(
            query=rendered_without_limit,
            limit=limit,
            offset=offset,
        )

    return _build_window_query_for_plain_sql(
        base_query=template,
        period_start_iso=period_start_iso,
        period_end_iso=period_end_iso,
        limit=limit,
        offset=offset,
    )


def _strip_trailing_limit_offset(query: str) -> str:
    stripped = query.strip().rstrip(";")
    stripped = re.sub(r"(?is)\s+OFFSET\s+\d+\s*$", "", stripped)
    stripped = re.sub(r"(?is)\s+LIMIT\s+\d+\s*$", "", stripped)
    return stripped


def _build_count_query_for_spec(
    *,
    spec: Dict[str, Any],
    period_start_iso: str,
    period_end_iso: str,
) -> str:
    template = str(spec["template"])
    uses_template = bool(spec["uses_template"])
    uses_paging_template = bool(spec["uses_paging_template"])

    if uses_template and uses_paging_template:
        # Template controls its own paging: render with dummy values, strip trailing LIMIT/OFFSET.
        # For keyset templates, pass period_start as last_ts so the rendered query covers
        # the full window (same as "start from the very beginning").
        rendered = _render_query_template(
            query_template=template,
            period_start_iso=period_start_iso,
            period_end_iso=period_end_iso,
            limit=1,
            offset=0,
            last_ts=period_start_iso,
        )
        base = _strip_trailing_limit_offset(rendered)
    elif uses_template:
        # Template has time placeholders but not paging
        base = _render_query_template(
            query_template=template,
            period_start_iso=period_start_iso,
            period_end_iso=period_end_iso,
            limit=1,
            offset=0,
            last_ts=period_start_iso,
        )
    else:
        # Plain SQL: apply time window filter without LIMIT/OFFSET
        base_sql = _normalize_sql_query_text(template)
        start_escaped = _escape_sql_literal(period_start_iso)
        end_escaped = _escape_sql_literal(period_end_iso)
        base = (
            "SELECT * FROM ("
            f"{base_sql}"
            ") AS cp_src "
            f"WHERE timestamp >= parseDateTimeBestEffort('{start_escaped}') "
            f"AND timestamp < parseDateTimeBestEffort('{end_escaped}')"
        )

    return f"SELECT count() AS total_rows FROM ({base}) AS cp_count"


def _precount_rows(
    *,
    query_specs: List[Dict[str, Any]],
    query_logs_df: Callable[[str], pd.DataFrame],
    period_start_iso: str,
    period_end_iso: str,
    logger: logging.Logger,
    on_error: Optional[Callable[[str], None]] = None,
) -> Optional[int]:
    total = 0
    for idx, spec in enumerate(query_specs):
        try:
            count_query = _build_count_query_for_spec(
                spec=spec,
                period_start_iso=period_start_iso,
                period_end_iso=period_end_iso,
            )
            df = query_logs_df(count_query)
            if df.empty:
                continue
            val = int(df["total_rows"].iloc[0] if "total_rows" in df.columns else df.iloc[0, 0])
            total += val
        except Exception as exc:
            msg = f"Подсчёт строк запроса #{idx + 1} не удался: {exc}"
            logger.warning("precount_rows[%s]: %s", idx + 1, exc)
            if on_error:
                on_error(msg)
            return None
    return total


def _build_demo_logs_for_window(
    *,
    period_start_dt: datetime,
    period_end_dt: datetime,
    total_logs: int,
) -> List[Dict[str, Any]]:
    safe_total = max(int(total_logs), 1)
    window_seconds = max(int((period_end_dt - period_start_dt).total_seconds()), safe_total + 1)
    rows: List[Dict[str, Any]] = []

    for idx in range(safe_total):
        sec_offset = int((idx + 1) * window_seconds / (safe_total + 1))
        ts = period_start_dt + timedelta(seconds=sec_offset)
        phase = (idx + 1) / safe_total
        if phase < 0.35:
            level = "INFO"
        elif phase < 0.65:
            level = "WARN" if idx % 3 == 0 else "INFO"
        elif phase < 0.9:
            level = "ERROR" if idx % 2 == 0 else "WARN"
        else:
            level = "CRITICAL" if idx % 3 == 0 else "ERROR"

        rows.append(
            {
                "timestamp": ts.isoformat().replace("+00:00", "Z"),
                "level": level,
                "message": f"{level.lower()} synthetic log #{idx + 1}",
                "service": "demo-service",
                "pod": f"demo-pod-{1 + (idx % 4)}",
                "container": "app",
                "node": f"node-{1 + (idx % 6):02d}",
                "cluster": "demo-cluster",
            }
        )
    return rows


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    return str(value)


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(payload), ensure_ascii=False) + "\n")


def _save_logs_summary_result(
    *,
    output_dir: Path,
    request_payload: Dict[str, Any],
    result_state: Dict[str, Any],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"logs_summary_result_{stamp}.json"
    summary_path = output_dir / f"logs_summary_result_{stamp}.md"

    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "request": _json_safe(request_payload),
        "result": _json_safe(result_state),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        "# Logs Summary Result",
        "",
        f"- saved_at: `{payload['saved_at']}`",
        f"- status: `{result_state.get('status')}`",
        f"- mode: `{result_state.get('mode')}`",
        f"- period: `{result_state.get('period_start')}` -> `{result_state.get('period_end')}`",
        "",
        "## Final Summary",
        "",
        str(result_state.get("final_summary") or "N/A"),
        "",
        "## Stats",
        "",
        f"- logs_processed: `{result_state.get('logs_processed')}`",
        f"- logs_total: `{result_state.get('logs_total')}`",
        f"- stats: `{result_state.get('stats')}`",
        f"- error: `{result_state.get('error')}`",
        "",
        f"JSON dump: `{json_path}`",
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {"json_path": str(json_path), "summary_path": str(summary_path)}


def _read_file_bytes(path: Optional[str]) -> Optional[bytes]:
    if not path:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        return p.read_bytes()
    except Exception:
        return None


def _build_freeform_summary_prompt(
    *,
    final_summary: str,
    user_goal: str,
    period_start: str,
    period_end: str,
    stats: Dict[str, Any],
    metrics_context: str,
) -> str:
    goal_block = user_goal.strip() or "Не указан"
    metrics_block = metrics_context.strip() or "Нет доп. метрик в контексте."
    return (
        "Ты пишешь итоговый отчёт об инциденте для дежурного SRE-инженера.\n\n"
        f"КОНТЕКСТ ИНЦИДЕНТА:\n{goal_block}\n\n"
        "Ниже — структурированный анализ логов. На его основе напиши отчёт строго по шаблону:\n\n"
        "## Что произошло\n"
        "2-3 предложения: суть инцидента простым языком.\n\n"
        "## Хронология\n"
        "Ключевые события с временными метками (только подтверждённые логами).\n\n"
        "## Объяснение алертов\n"
        "Для каждого алерта из контекста:\n"
        "- [Название алерта] — объяснение найдено / частично / не найдено\n"
        "  Что показывают логи: ...\n\n"
        "## Первопричина\n"
        "[ФАКТ] или [ГИПОТЕЗА]: конкретное утверждение с доказательством из логов.\n\n"
        "## Что делать прямо сейчас\n"
        "Конкретные шаги (не \"проверить ресурсы\", а \"проверить /var на node-X, т.к. логи показывают...\").\n\n"
        "Если данных недостаточно для какого-то раздела — прямо напиши об этом.\n\n"
        f"Период: [{period_start}, {period_end})\n"
        f"Метрики: {metrics_block}\n\n"
        "Структурированный анализ логов:\n"
        f"{final_summary}"
    )


def _render_final_report(container, state: Dict[str, Any], deps: LogsSummaryPageDeps) -> None:
    status = str(state.get("status", ""))
    if status not in ("done", "error"):
        return

    with container.container():
        st.markdown("2. Итоговый Отчет")
        final_height = max(
            int(st.session_state.get("logs_sum_llm_final_height", deps.final_text_height)),
            320,
        )

        stats = state.get("stats") or {}
        rows_processed = int(pd.to_numeric(state.get("logs_processed"), errors="coerce") or 0)
        rows_total = pd.to_numeric(state.get("logs_total"), errors="coerce")
        llm_calls = int(pd.to_numeric(stats.get("llm_calls", 0), errors="coerce") or 0)
        reduce_rounds = int(pd.to_numeric(stats.get("reduce_rounds", 0), errors="coerce") or 0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Статус", "Готово" if status == "done" else "Ошибка")
        col2.metric("Обработано логов", f"{rows_processed}")
        col3.metric("LLM вызовы", f"{llm_calls}")
        col4.metric("Reduce раунды", f"{reduce_rounds}")

        _last_ts = pd.to_datetime(state.get("last_batch_ts"), utc=True, errors="coerce")
        _p_start = pd.to_datetime(state.get("period_start"), utc=True, errors="coerce")
        _p_end = pd.to_datetime(state.get("period_end"), utc=True, errors="coerce")
        if (
            not pd.isna(_last_ts)
            and not pd.isna(_p_start)
            and not pd.isna(_p_end)
            and (_p_end - _p_start).total_seconds() > 0
        ):
            _span = (_p_end - _p_start).total_seconds()
            _done = max((_last_ts - _p_start).total_seconds(), 0.0)
            _pct = min(_done / _span, 1.0) * 100.0
            st.caption(
                f"Покрытие периода по времени: {_pct:.1f}% "
                f"(лог до {_last_ts.tz_convert(MSK).strftime('%H:%M:%S MSK')} | строк: {rows_processed:,})"
            )
        elif rows_processed > 0:
            st.caption(f"Обработано логов: {rows_processed:,}")
        st.caption(
            "Период отчета: "
            f"{_format_datetime_with_tz(state.get('period_start'))} -> "
            f"{_format_datetime_with_tz(state.get('period_end'))}"
        )
        metrics_rows = int(pd.to_numeric(state.get("metrics_rows"), errors="coerce") or 0)
        metrics_services = state.get("metrics_services") or []
        if metrics_rows > 0:
            st.caption(
                f"Метрики в контексте: rows={metrics_rows}, services={', '.join(map(str, metrics_services))}"
            )

        final_summary = str(state.get("final_summary") or "").strip()
        if final_summary:
            st.markdown("Итоговое расследование")
            deps.render_pretty_summary_text(final_summary, height=max(final_height, 280))

        freeform_summary = str(state.get("freeform_final_summary") or "").strip()
        if freeform_summary:
            st.markdown("Итоговое расследование в свободном формате")
            deps.render_pretty_summary_text(
                freeform_summary,
                height=max(final_height, 320),
            )

        query_errors = state.get("query_errors", [])
        if isinstance(query_errors, list) and query_errors:
            with st.expander("Ошибки ClickHouse (запросы, которые были пропущены)", expanded=False):
                for err in query_errors:
                    st.warning(str(err))

        artifacts_col, download_col = st.columns([2, 1])
        with artifacts_col:
            st.markdown("Артефакты")
            if state.get("result_json_path"):
                st.code(str(state.get("result_json_path")))
            if state.get("result_summary_path"):
                st.code(str(state.get("result_summary_path")))
            if state.get("live_events_path"):
                st.code(str(state.get("live_events_path")))
            if state.get("live_batches_path"):
                st.code(str(state.get("live_batches_path")))

        with download_col:
            st.markdown("Скачать")
            json_bytes = _read_file_bytes(state.get("result_json_path"))
            md_bytes = _read_file_bytes(state.get("result_summary_path"))
            if json_bytes is not None and state.get("result_json_path"):
                st.download_button(
                    label="JSON отчет",
                    data=json_bytes,
                    file_name=Path(str(state.get("result_json_path"))).name,
                    mime="application/json",
                    use_container_width=True,
                )
            if md_bytes is not None and state.get("result_summary_path"):
                st.download_button(
                    label="Markdown отчет",
                    data=md_bytes,
                    file_name=Path(str(state.get("result_summary_path"))).name,
                    mime="text/markdown",
                    use_container_width=True,
                )


def _render_logs_summary_chat(container, state: Dict[str, Any], deps: LogsSummaryPageDeps) -> None:
    status_titles = {
        "queued": "В очереди",
        "counting": "Подсчёт строк",
        "summarizing": "Подготовка summary",
        "map": "Map этап",
        "reduce": "Reduce этап",
        "summary_ready": "Summary готов",
        "done": "Завершено",
        "error": "Ошибка",
    }

    with container.container():
        st.markdown("1. Пошаговая Суммаризация Логов")
        summary_height = max(
            int(st.session_state.get("logs_sum_llm_summary_height", deps.summary_text_height)),
            220,
        )
        final_height = max(
            int(st.session_state.get("logs_sum_llm_final_height", deps.final_text_height)),
            320,
        )
        if not state:
            return

        with st.chat_message("user"):
            period_mode = str(state.get("period_mode", "window"))
            period_desc = (
                f"Окно: `+-{state.get('window_minutes')}` минут"
                if period_mode == "window"
                else "Режим: `start/end диапазон`"
            )
            logs_fetch_mode = _normalize_logs_fetch_mode(state.get("logs_fetch_mode", "time_window"))
            if str(state.get("mode", "db")) == "demo":
                logs_fetch_desc = (
                    f"Выборка логов: `по количеству` (demo, synthetic rows={state.get('demo_logs_count')})"
                )
            elif logs_fetch_mode == "tail_n_logs":
                logs_fetch_desc = (
                    f"Выборка логов: `по количеству` (tail_n_logs, limit={state.get('logs_tail_limit')})"
                )
            else:
                logs_fetch_desc = "Выборка логов: `по датам` (time_window)"
            period_start_text = _format_datetime_with_tz(state.get("period_start"))
            period_end_text = _format_datetime_with_tz(state.get("period_end"))
            retries_num_raw = pd.to_numeric(state.get("max_retries", -1), errors="coerce")
            retries_num = -1 if pd.isna(retries_num_raw) else int(retries_num_raw)
            st.markdown(
                "\n".join(
                    [
                        f"Режим: `{state.get('mode', 'db')}`",
                        f"Период: `{period_start_text}` -> `{period_end_text}`",
                        period_desc,
                        f"SQL запросов: `{state.get('queries_count', 1)}`",
                        f"SQL метрик: `{state.get('metrics_queries_count', 0)}`",
                        logs_fetch_desc,
                        f"DB batch: `{state.get('db_batch_size')}`",
                        f"LLM batch: `{state.get('llm_batch_size')}`",
                        f"MAP workers: `{state.get('map_workers', 1)}`",
                        ("Ретраи LLM: `∞`" if retries_num < 0 else f"Ретраи LLM: `{retries_num}`"),
                        f"Таймаут LLM: `{state.get('llm_timeout', 600)}s`",
                    ]
                )
            )
            deps.render_scrollable_text(state.get("query", ""), height=130)
            metrics_query_text = str(state.get("metrics_query", "")).strip()
            if metrics_query_text:
                st.markdown("SQL метрик")
                deps.render_scrollable_text(metrics_query_text, height=110)
            goal = str(state.get("user_goal", "")).strip()
            if goal:
                st.markdown("Контекст пользователя")
                deps.render_scrollable_text(goal, height=110)

        with st.chat_message("assistant"):
            status = str(state.get("status", "queued"))
            st.markdown(f"Статус: **{status_titles.get(status, status)}**")
            active_step = str(state.get("active_step", "")).strip()
            if active_step:
                st.caption(f"Сейчас: {active_step}")
            llm_started = int(pd.to_numeric(state.get("llm_calls_started"), errors="coerce") or 0)
            llm_ok = int(pd.to_numeric(state.get("llm_calls_succeeded"), errors="coerce") or 0)
            llm_fail = int(pd.to_numeric(state.get("llm_calls_failed"), errors="coerce") or 0)
            llm_last_dur = pd.to_numeric(state.get("llm_last_duration_sec"), errors="coerce")
            llm_active = bool(state.get("llm_active", False))
            llm_attempt = str(state.get("llm_last_attempt", "") or "")
            llm_error = str(state.get("llm_last_error", "") or "")
            if llm_started > 0:
                parts = [f"LLM вызовы: start={llm_started}, ok={llm_ok}, fail={llm_fail}"]
                if not pd.isna(llm_last_dur):
                    parts.append(f"last={float(llm_last_dur):.1f}s")
                st.caption(" | ".join(parts))
            if llm_active:
                wait_text = f"Ждём ответ LLM ({llm_attempt})..."
                st.info(wait_text)
            elif llm_error:
                st.warning(f"Последняя ошибка LLM: {llm_error}")
            read_timeout_active = bool(state.get("read_timeout_active", False))
            read_timeout_started_at = state.get("read_timeout_started_at")
            read_timeout_count = int(
                pd.to_numeric(state.get("read_timeout_count"), errors="coerce") or 0
            )
            if read_timeout_active and read_timeout_started_at:
                st.warning(
                    "ReadTimeout серия активна: "
                    f"с `{_format_datetime_with_tz(read_timeout_started_at)}`, "
                    f"ошибок `{read_timeout_count}`."
                )
            last_rt = state.get("read_timeout_last_episode")
            if isinstance(last_rt, dict):
                rt_start = last_rt.get("start")
                rt_end = last_rt.get("end")
                rt_count = last_rt.get("count")
                rt_resolution = str(last_rt.get("resolution") or "")
                rt_duration = pd.to_numeric(last_rt.get("duration_sec"), errors="coerce")
                if rt_start and rt_end:
                    duration_text = (
                        _format_eta_seconds(float(rt_duration))
                        if not pd.isna(rt_duration)
                        else "n/a"
                    )
                    st.caption(
                        "Последняя серия ReadTimeout: "
                        f"{_format_datetime_with_tz(rt_start)} -> "
                        f"{_format_datetime_with_tz(rt_end)} | "
                        f"ошибок={rt_count} | итог={rt_resolution} | длительность={duration_text}"
                    )
            llm_timeline = state.get("llm_timeline", [])
            if isinstance(llm_timeline, list) and llm_timeline:
                with st.expander("LLM Activity Timeline", expanded=False):
                    df_tl = _format_table_timestamps(pd.DataFrame(llm_timeline[-200:]))
                    if not df_tl.empty:
                        preferred_cols = [
                            "time",
                            "event",
                            "call_no",
                            "attempt",
                            "total_attempts",
                            "elapsed_sec",
                            "status",
                            "details",
                        ]
                        cols = [c for c in preferred_cols if c in df_tl.columns]
                        if cols:
                            df_tl = df_tl[cols]
                        styled_tl = _style_llm_timeline(df_tl)
                        st.dataframe(
                            styled_tl if styled_tl is not None else df_tl,
                            use_container_width=True,
                            hide_index=True,
                            height=220,
                        )
            # --- Timestamp-based progress bar ---
            last_batch_ts = pd.to_datetime(state.get("last_batch_ts"), utc=True, errors="coerce")
            period_start_ts = pd.to_datetime(state.get("period_start"), utc=True, errors="coerce")
            period_end_ts = pd.to_datetime(state.get("period_end"), utc=True, errors="coerce")

            ts_ratio: Optional[float] = None
            if (
                not pd.isna(last_batch_ts)
                and not pd.isna(period_start_ts)
                and not pd.isna(period_end_ts)
            ):
                total_span = (period_end_ts - period_start_ts).total_seconds()
                done_span = max((last_batch_ts - period_start_ts).total_seconds(), 0.0)
                if total_span > 0:
                    ts_ratio = min(max(done_span / total_span, 0.0), 1.0)
                    pct = ts_ratio * 100.0
                    last_ts_str = last_batch_ts.tz_convert(MSK).strftime("%H:%M:%S MSK")
                    st.progress(
                        ts_ratio,
                        text=f"Прогресс: {pct:.1f}% | лог до {last_ts_str}",
                    )
            elif state.get("logs_processed"):
                st.caption(f"Обработано строк: {state['logs_processed']}")

            eta_left = state.get("eta_seconds_left")
            eta_finish = state.get("eta_finish_at")
            elapsed_seconds = pd.to_numeric(state.get("elapsed_seconds"), errors="coerce")
            log_seconds_per_second = pd.to_numeric(state.get("log_seconds_per_second"), errors="coerce")
            details: List[str] = []
            if not pd.isna(elapsed_seconds):
                details.append(f"elapsed: {_format_eta_seconds(float(elapsed_seconds))}")
            if not pd.isna(log_seconds_per_second) and float(log_seconds_per_second) > 0:
                lsps = float(log_seconds_per_second)
                if lsps >= 60:
                    details.append(f"скорость: {lsps / 60:.1f} мин.лога/с")
                else:
                    details.append(f"скорость: {lsps:.1f} с.лога/с")
            if details:
                if eta_left is not None and status not in ("done", "error"):
                    details.append(f"(осталось ~{_format_eta_seconds(float(eta_left))})")
                st.caption(" | ".join(details))
            if eta_left is not None and status not in ("done", "error"):
                if eta_finish:
                    try:
                        eta_finish_dt = pd.to_datetime(eta_finish, utc=True, errors="coerce")
                        if not pd.isna(eta_finish_dt):
                            finish_text = eta_finish_dt.tz_convert(MSK).strftime("%H:%M:%S MSK")
                            st.caption(
                                f"Ориентировочно завершится в {finish_text} "
                                f"(через ~{_format_eta_seconds(float(eta_left))})"
                            )
                        else:
                            st.caption(
                                f"Ориентировочное время до завершения: ~{_format_eta_seconds(float(eta_left))}"
                            )
                    except Exception:
                        st.caption(
                            f"Ориентировочное время до завершения: ~{_format_eta_seconds(float(eta_left))}"
                        )
            for line in state.get("events", [])[-10:]:
                st.caption(str(line))

            query_errors = state.get("query_errors", [])
            if isinstance(query_errors, list) and query_errors:
                st.warning(f"Ошибок ClickHouse: {len(query_errors)} (часть запросов пропущена)")

        batches = state.get("map_batches", [])
        for batch in batches:
            idx = int(batch.get("batch_index", 0)) + 1
            total = batch.get("batch_total")
            title = f"Map summary {idx}/{total}" if total else f"Map summary {idx}"
            batch_logs = batch.get("batch_logs", [])
            if not isinstance(batch_logs, list):
                batch_logs = []

            with st.chat_message("assistant"):
                st.markdown(title)
                deps.render_pretty_summary_text(batch.get("batch_summary", ""), height=summary_height)
                batch_logs_count = batch.get("batch_logs_count")
                if batch_logs_count is None:
                    batch_logs_count = len(batch_logs)
                st.caption(f"Логов в батче: {batch_logs_count}")
                period_start, period_end = deps.infer_batch_period(batch)
                if period_start and period_end:
                    st.caption(f"Период логов батча: `{period_start}` -> `{period_end}`")
                if batch_logs:
                    st.dataframe(
                        _format_table_timestamps(pd.DataFrame(batch_logs)),
                        use_container_width=True,
                        hide_index=True,
                        height=deps.logs_batch_table_height,
                    )

        if state.get("final_summary"):
            with st.chat_message("assistant"):
                st.markdown("Итоговый Reduce summary")
                deps.render_pretty_summary_text(state["final_summary"], height=final_height)

        if state.get("stats"):
            stats = state["stats"]
            with st.chat_message("assistant"):
                st.caption(
                    "Статистика: "
                    f"pages={stats.get('pages_fetched', 0)}, "
                    f"rows={stats.get('rows_processed', 0)}, "
                    f"llm_calls={stats.get('llm_calls', 0)}, "
                    f"reduce_rounds={stats.get('reduce_rounds', 0)}"
                )

        if state.get("error"):
            with st.chat_message("assistant"):
                st.error(f"Ошибка: {state['error']}")

        if state.get("result_json_path") or state.get("result_summary_path"):
            with st.chat_message("assistant"):
                st.markdown("Результаты сохранены")
                if state.get("result_json_path"):
                    st.code(str(state.get("result_json_path")))
                if state.get("result_summary_path"):
                    st.code(str(state.get("result_summary_path")))
                if state.get("live_events_path"):
                    st.code(str(state.get("live_events_path")))
                if state.get("live_batches_path"):
                    st.code(str(state.get("live_batches_path")))


def _build_config(deps: LogsSummaryPageDeps, db_batch_size: int, llm_batch_size: int, map_workers: int = 1) -> Any:
    try:
        return deps.summarizer_config_cls(
            page_limit=db_batch_size,
            llm_chunk_rows=llm_batch_size,
            keep_map_batches_in_memory=False,
            keep_map_summaries_in_result=False,
            map_workers=max(map_workers, 1),
        )
    except TypeError:
        return deps.summarizer_config_cls(
            page_limit=db_batch_size,
            llm_chunk_rows=llm_batch_size,
        )


def render_logs_summary_page(deps: LogsSummaryPageDeps) -> None:
    st.title("Logs Summarizer")
    pending_form_error = st.session_state.pop(FORM_ERROR_SESSION_KEY, None)
    if pending_form_error:
        st.error(str(pending_form_error))
    if RUNNING_SESSION_KEY not in st.session_state:
        st.session_state[RUNNING_SESSION_KEY] = False
    is_running = bool(st.session_state.get(RUNNING_SESSION_KEY, False))
    run_params = st.session_state.get(RUN_PARAMS_SESSION_KEY)
    if is_running and not isinstance(run_params, dict):
        st.session_state[RUNNING_SESSION_KEY] = False
        is_running = False

    default_query = (deps.default_sql_query or "").strip() or (
        "SELECT timestamp, level, message FROM logs_demo_service "
        "WHERE timestamp >= parseDateTimeBestEffort('{period_start}') "
        "AND timestamp < parseDateTimeBestEffort('{period_end}') "
        "ORDER BY timestamp DESC LIMIT {limit} OFFSET {offset}"
    )

    now_msk = datetime.now(MSK).replace(microsecond=0)
    center_default = now_msk.isoformat()
    start_default = (now_msk - timedelta(minutes=max(int(deps.loopback_minutes), 1))).isoformat()
    end_default = now_msk.isoformat()
    default_metrics_query = str(deps.default_metrics_query or "").strip()

    widget_defaults: Dict[str, Any] = {
        "logs_sum_user_goal": "",
        "logs_sum_period_mode": "Явный диапазон (start/end)",
        "logs_sum_center_dt": center_default,
        "logs_sum_window_minutes": max(int(deps.loopback_minutes), 1),
        "logs_sum_start_dt": start_default,
        "logs_sum_end_dt": end_default,
        "logs_sum_db_batch": max(int(deps.db_batch_size), 1),
        "logs_sum_llm_batch": max(int(deps.llm_batch_size), 1),
        "logs_sum_parallel_map": False,
        "logs_sum_map_workers": max(int(deps.map_workers), 1),
        "logs_sum_max_retries": int(deps.max_retries),
        "logs_sum_llm_timeout": max(int(deps.llm_timeout), 10),
        "logs_sum_demo_mode": bool(deps.test_mode),
        "logs_sum_demo_logs_count": max(int(deps.logs_tail_limit), deps.db_batch_size * 4, 4000),
        "logs_sum_llm_summary_height": max(int(deps.summary_text_height), 220),
        "logs_sum_llm_final_height": max(int(deps.final_text_height), 320),
    }
    for key, value in widget_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    # Keep critical fields prefilled even if they were accidentally cleared.
    for key in ("logs_sum_center_dt", "logs_sum_start_dt", "logs_sum_end_dt"):
        current = st.session_state.get(key)
        if not isinstance(current, str) or not current.strip():
            st.session_state[key] = widget_defaults[key]

    # Query editors state (multiple boxes with + button).
    logs_queries_state_key = "logs_sum_log_queries_items"
    metrics_queries_state_key = "logs_sum_metrics_queries_items"
    if logs_queries_state_key not in st.session_state:
        legacy_query = str(st.session_state.get("logs_sum_sql_query", "")).strip()
        initial_logs_query = legacy_query or default_query
        st.session_state[logs_queries_state_key] = [_new_query_item(initial_logs_query)]
    st.session_state[logs_queries_state_key] = _normalize_query_items(
        st.session_state.get(logs_queries_state_key),
        default_value=default_query,
        min_items=1,
    )
    if metrics_queries_state_key not in st.session_state:
        legacy_metrics_query = str(st.session_state.get("logs_sum_metrics_query", "")).strip()
        initial_metrics_query = legacy_metrics_query or default_metrics_query
        st.session_state[metrics_queries_state_key] = (
            [_new_query_item(initial_metrics_query)] if initial_metrics_query else []
        )
    st.session_state[metrics_queries_state_key] = _normalize_query_items(
        st.session_state.get(metrics_queries_state_key),
        default_value=default_metrics_query,
        min_items=0,
    )

    with st.sidebar:
        st.markdown("SQL Запросы Логов")
        if st.button(
            "+ Добавить запрос логов",
            key="logs_sum_add_log_query",
            use_container_width=True,
            disabled=is_running,
        ):
            st.session_state[logs_queries_state_key].append(_new_query_item(""))
            st.rerun()

        remove_log_query_id: Optional[str] = None
        for idx, item in enumerate(st.session_state[logs_queries_state_key], start=1):
            item_id = str(item.get("id"))
            query_key = f"logs_sum_log_query_text_{item_id}"
            if query_key not in st.session_state:
                st.session_state[query_key] = str(item.get("text", ""))
            col_query, col_remove = st.columns([10, 2])
            with col_query:
                st.text_area(
                    f"SQL логов #{idx}",
                    key=query_key,
                    height=max(int(deps.sql_textarea_height), 180),
                    placeholder=(
                        "SELECT timestamp, message FROM logs_table\n"
                        "WHERE timestamp >= parseDateTimeBestEffort('{period_start}')\n"
                        "  AND timestamp < parseDateTimeBestEffort('{period_end}')\n"
                        "ORDER BY timestamp\n"
                        "LIMIT {limit} OFFSET {offset}"
                    ),
                    help=(
                        "Поддерживаются плейсхолдеры: "
                        "{period_start}, {period_end}, {start}, {end}, {limit}, {offset}. "
                        "Для keyset-пейджинга (без OFFSET) используйте {last_ts} вместо {offset}: "
                        "WHERE timestamp > '{last_ts}' AND timestamp < '{period_end}' ORDER BY timestamp LIMIT {limit}"
                    ),
                    disabled=is_running,
                )
            with col_remove:
                can_remove = len(st.session_state[logs_queries_state_key]) > 1
                if st.button(
                    "Убрать",
                    key=f"logs_sum_remove_log_query_{item_id}",
                    disabled=is_running or not can_remove,
                    help="Удалить этот SQL-блок",
                    use_container_width=True,
                ):
                    remove_log_query_id = item_id
            item["text"] = str(st.session_state.get(query_key, ""))

        if remove_log_query_id:
            st.session_state[logs_queries_state_key] = [
                item for item in st.session_state[logs_queries_state_key]
                if str(item.get("id")) != remove_log_query_id
            ]
            st.rerun()

        st.text_area(
            "Контекст по алертам/инциденту для LLM (опционально)",
            key="logs_sum_user_goal",
            height=140,
            disabled=is_running,
        )

        st.markdown("SQL Запросы Метрик (Опционально)")
        if st.button(
            "+ Добавить запрос метрик",
            key="logs_sum_add_metrics_query",
            use_container_width=True,
            disabled=is_running,
        ):
            st.session_state[metrics_queries_state_key].append(_new_query_item(""))
            st.rerun()

        remove_metrics_query_id: Optional[str] = None
        for idx, item in enumerate(st.session_state[metrics_queries_state_key], start=1):
            item_id = str(item.get("id"))
            query_key = f"logs_sum_metrics_query_text_{item_id}"
            if query_key not in st.session_state:
                st.session_state[query_key] = str(item.get("text", ""))
            col_query, col_remove = st.columns([10, 2])
            with col_query:
                st.text_area(
                    f"SQL метрик #{idx}",
                    key=query_key,
                    height=140,
                    placeholder=(
                        "SELECT timestamp, value, service FROM metrics_table\n"
                        "WHERE timestamp >= parseDateTimeBestEffort('{period_start}')\n"
                        "  AND timestamp < parseDateTimeBestEffort('{period_end}')\n"
                        "ORDER BY timestamp\n"
                        "LIMIT {limit} OFFSET {offset}"
                    ),
                    help=(
                        "Ожидаются колонки timestamp и числовая value "
                        "(service опционально)."
                    ),
                    disabled=is_running,
                )
            with col_remove:
                if st.button(
                    "Убрать",
                    key=f"logs_sum_remove_metrics_query_{item_id}",
                    disabled=is_running,
                    help="Удалить этот SQL-блок",
                    use_container_width=True,
                ):
                    remove_metrics_query_id = item_id
            item["text"] = str(st.session_state.get(query_key, ""))

        if remove_metrics_query_id:
            st.session_state[metrics_queries_state_key] = [
                item for item in st.session_state[metrics_queries_state_key]
                if str(item.get("id")) != remove_metrics_query_id
            ]
            st.rerun()

        period_mode_label = st.radio(
            "Режим периода",
            options=("Окно вокруг даты (±N минут)", "Явный диапазон (start/end)"),
            horizontal=False,
            key="logs_sum_period_mode",
            disabled=is_running,
        )
        is_window_mode = period_mode_label.startswith("Окно вокруг")

        if is_window_mode:
            st.text_input(
                "Целевая дата/время (ISO)",
                key="logs_sum_center_dt",
                placeholder="Например: 2026-03-27T14:30:00+03:00",
                help="Указывай дату/время с часовым поясом: `+03:00` или `Z`.",
                disabled=is_running,
            )
            st.number_input(
                "Окно анализа (+- N минут)",
                min_value=1,
                max_value=60 * 24 * 30,
                step=1,
                key="logs_sum_window_minutes",
                disabled=is_running,
            )
        else:
            st.text_input(
                "Дата/время начала (ISO)",
                key="logs_sum_start_dt",
                placeholder="Например: 2026-03-27T10:00:00+03:00",
                help="Формат: `YYYY-MM-DDTHH:MM:SS+03:00` (или `Z` для UTC).",
                disabled=is_running,
            )
            st.text_input(
                "Дата/время конца (ISO)",
                key="logs_sum_end_dt",
                placeholder="Например: 2026-03-27T12:00:00+03:00",
                help="Формат: `YYYY-MM-DDTHH:MM:SS+03:00` (или `Z` для UTC).",
                disabled=is_running,
            )

        st.number_input(
            "Размер DB batch (выгрузка из БД)",
            min_value=10,
            max_value=100_000,
            step=100,
            key="logs_sum_db_batch",
            disabled=is_running,
        )
        st.number_input(
            "Размер LLM batch (строк в один MAP prompt)",
            min_value=10,
            max_value=10_000,
            step=10,
            key="logs_sum_llm_batch",
            disabled=is_running,
        )
        st.checkbox(
            "Параллельные LLM вызовы (MAP workers)",
            key="logs_sum_parallel_map",
            disabled=is_running,
            help="Включить параллельную обработку MAP-батчей. По умолчанию выключено — один батч за раз (меньше нагрузки на LLM).",
        )
        if st.session_state.get("logs_sum_parallel_map"):
            st.number_input(
                "MAP workers",
                min_value=2,
                max_value=32,
                step=1,
                key="logs_sum_map_workers",
                disabled=is_running,
                help="Сколько MAP-батчей обрабатывается параллельно. 2–8 = ускоряет суммаризацию при медленном LLM.",
            )
        st.number_input(
            "Ретраи LLM (при ошибке)",
            min_value=-1,
            max_value=100,
            step=1,
            key="logs_sum_max_retries",
            disabled=is_running,
            help=(
                "Сколько раз повторить вызов LLM при ошибке. "
                "0 = без ретраев (сразу fallback). "
                "-1 = бесконечные ретраи. "
                "Пауза между попытками фиксированная: 10 секунд."
            ),
        )
        st.number_input(
            "Таймаут LLM (сек)",
            min_value=10,
            max_value=600,
            step=10,
            key="logs_sum_llm_timeout",
            disabled=is_running,
            help=(
                "Максимальное время ожидания одного LLM-ответа. "
                "При зависании запрос упадёт по таймауту и уйдёт на retry. "
                "Для ReadTimeout таймаут растёт прогрессивно (+base timeout на каждую ошибку), "
                "повторы идут бесконечно."
            ),
        )
        st.toggle(
            "Демо режим (без БД)",
            key="logs_sum_demo_mode",
            disabled=is_running,
        )
        st.number_input(
            "Количество логов в демо режиме",
            min_value=100,
            max_value=50_000,
            step=100,
            key="logs_sum_demo_logs_count",
            disabled=is_running or not bool(st.session_state.get("logs_sum_demo_mode", False)),
        )

        run_clicked = st.button(
            "Запустить Суммаризацию Логов",
            type="primary",
            use_container_width=True,
            disabled=is_running,
        )

    logs_queries = _extract_queries_from_items(st.session_state.get(logs_queries_state_key))
    metrics_queries = _extract_queries_from_items(st.session_state.get(metrics_queries_state_key))
    user_goal = str(st.session_state.get("logs_sum_user_goal", ""))
    window_minutes = int(st.session_state.get("logs_sum_window_minutes", max(int(deps.loopback_minutes), 1)))
    center_dt_text = str(st.session_state.get("logs_sum_center_dt", center_default))
    start_dt_text = str(st.session_state.get("logs_sum_start_dt", start_default))
    end_dt_text = str(st.session_state.get("logs_sum_end_dt", end_default))
    db_batch_size = int(st.session_state.get("logs_sum_db_batch", max(int(deps.db_batch_size), 1)))
    llm_batch_size = int(st.session_state.get("logs_sum_llm_batch", max(int(deps.llm_batch_size), 1)))
    _parallel_map = bool(st.session_state.get("logs_sum_parallel_map", False))
    map_workers = int(st.session_state.get("logs_sum_map_workers", max(int(deps.map_workers), 1))) if _parallel_map else 1
    max_retries = int(st.session_state.get("logs_sum_max_retries", int(deps.max_retries)))
    llm_timeout = max(int(st.session_state.get("logs_sum_llm_timeout", max(int(deps.llm_timeout), 10))), 10)
    demo_mode = bool(st.session_state.get("logs_sum_demo_mode", bool(deps.test_mode)))
    demo_logs_count = int(
        st.session_state.get(
            "logs_sum_demo_logs_count",
            max(int(deps.logs_tail_limit), deps.db_batch_size * 4, 4000),
        )
    )

    runtime_error_placeholder = st.empty()
    analysis_placeholder = st.empty()
    final_report_placeholder = st.empty()

    def _unlock_with_form_error(message: str) -> None:
        st.session_state[FORM_ERROR_SESSION_KEY] = str(message)
        st.session_state[RUNNING_SESSION_KEY] = False
        st.session_state.pop(RUN_PARAMS_SESSION_KEY, None)
        st.rerun()

    if run_clicked and not is_running:
        st.session_state[RUN_PARAMS_SESSION_KEY] = {
            "logs_queries": list(logs_queries),
            "metrics_queries": list(metrics_queries),
            "user_goal": user_goal,
            "period_mode": str(st.session_state.get("logs_sum_period_mode", "Явный диапазон (start/end)")),
            "window_minutes": window_minutes,
            "center_dt_text": center_dt_text,
            "start_dt_text": start_dt_text,
            "end_dt_text": end_dt_text,
            "db_batch_size": db_batch_size,
            "llm_batch_size": llm_batch_size,
            "map_workers": map_workers,
            "max_retries": max_retries,
            "llm_timeout": llm_timeout,
            "demo_mode": demo_mode,
            "demo_logs_count": demo_logs_count,
        }
        st.session_state[RUNNING_SESSION_KEY] = True
        st.rerun()

    active_params: Optional[Dict[str, Any]] = None
    if is_running and isinstance(st.session_state.get(RUN_PARAMS_SESSION_KEY), dict):
        active_params = dict(st.session_state[RUN_PARAMS_SESSION_KEY])
        with runtime_error_placeholder.container():
            st.info("Процесс выполняется: параметры зафиксированы до завершения.")
    elif not run_clicked:
        last_state = st.session_state.get(LAST_STATE_SESSION_KEY)
        if isinstance(last_state, dict) and last_state:
            _render_logs_summary_chat(analysis_placeholder, last_state, deps)
            _render_final_report(final_report_placeholder, last_state, deps)
        return
    else:
        active_params = {
            "logs_queries": list(logs_queries),
            "metrics_queries": list(metrics_queries),
            "user_goal": user_goal,
            "period_mode": str(st.session_state.get("logs_sum_period_mode", "Явный диапазон (start/end)")),
            "window_minutes": window_minutes,
            "center_dt_text": center_dt_text,
            "start_dt_text": start_dt_text,
            "end_dt_text": end_dt_text,
            "db_batch_size": db_batch_size,
            "llm_batch_size": llm_batch_size,
            "map_workers": map_workers,
            "max_retries": max_retries,
            "llm_timeout": llm_timeout,
            "demo_mode": demo_mode,
            "demo_logs_count": demo_logs_count,
        }

    logs_queries = list(active_params.get("logs_queries", []))
    metrics_queries = list(active_params.get("metrics_queries", []))
    user_goal = str(active_params.get("user_goal", ""))
    period_mode_label = str(active_params.get("period_mode", "Явный диапазон (start/end)"))
    period_mode = "window" if period_mode_label.startswith("Окно вокруг") else "start_end"
    window_minutes = int(active_params.get("window_minutes", window_minutes))
    center_dt_text = str(active_params.get("center_dt_text", center_dt_text))
    start_dt_text = str(active_params.get("start_dt_text", start_dt_text))
    end_dt_text = str(active_params.get("end_dt_text", end_dt_text))
    db_batch_size = int(active_params.get("db_batch_size", db_batch_size))
    llm_batch_size = int(active_params.get("llm_batch_size", llm_batch_size))
    map_workers = max(int(active_params.get("map_workers", map_workers)), 1)
    max_retries = int(active_params.get("max_retries", max_retries))
    llm_timeout = max(int(active_params.get("llm_timeout", llm_timeout)), 10)
    demo_mode = bool(active_params.get("demo_mode", demo_mode))
    demo_logs_count = int(active_params.get("demo_logs_count", demo_logs_count))

    logs_queries = [str(q) for q in logs_queries if str(q).strip()]
    metrics_queries = [str(q) for q in metrics_queries if str(q).strip()]

    if not logs_queries:
        _unlock_with_form_error("Добавь хотя бы один SQL запрос логов.")

    try:
        if period_mode == "window":
            parsed_center = pd.to_datetime(center_dt_text, utc=True, errors="coerce")
            if pd.isna(parsed_center):
                raise ValueError("Неверный формат даты/времени (ISO).")
            center_dt = parsed_center.to_pydatetime()
            period_start_dt = center_dt - timedelta(minutes=window_minutes)
            period_end_dt = center_dt + timedelta(minutes=window_minutes)
        else:
            parsed_start = pd.to_datetime(start_dt_text, utc=True, errors="coerce")
            parsed_end = pd.to_datetime(end_dt_text, utc=True, errors="coerce")
            if pd.isna(parsed_start) or pd.isna(parsed_end):
                raise ValueError("Неверный формат start/end (ISO).")
            period_start_dt = parsed_start.to_pydatetime()
            period_end_dt = parsed_end.to_pydatetime()
            if period_end_dt <= period_start_dt:
                raise ValueError("Дата конца должна быть больше даты начала.")
    except Exception as exc:  # noqa: BLE001
        _unlock_with_form_error(str(exc))

    period_start_iso = period_start_dt.isoformat()
    period_end_iso = period_end_dt.isoformat()
    query_templates: List[str] = []
    for raw_query in logs_queries:
        query_templates.extend(_split_query_templates(raw_query))
    if not query_templates:
        _unlock_with_form_error("Не удалось распознать SQL запросы логов.")

    metrics_templates: List[str] = []
    for raw_query in metrics_queries:
        metrics_templates.extend(_split_query_templates(raw_query))

    sql_query_clean = "\n\n-- QUERY --\n\n".join(query_templates)
    metrics_query_clean = "\n\n-- QUERY --\n\n".join(metrics_templates)
    query_specs: List[Dict[str, Any]] = [
        {
            "template": tpl,
            "uses_template": _query_uses_any_placeholders(tpl),
            "uses_paging_template": _query_uses_paging_placeholders(tpl),
            "label": f"query_{idx + 1}",
        }
        for idx, tpl in enumerate(query_templates)
    ]
    multi_query_mode = len(query_specs) > 1
    metrics_query_specs: List[Dict[str, Any]] = [
        {
            "template": tpl,
            "uses_template": _query_uses_any_placeholders(tpl),
            "uses_paging_template": _query_uses_paging_placeholders(tpl),
        }
        for tpl in metrics_templates
    ]
    preview_available_columns: Optional[List[str]] = None
    preview_query_errors: List[str] = []
    preview_metrics_errors: List[str] = []
    logs_preview_frames: List[Dict[str, Any]] = []
    metrics_preview_frames: List[Dict[str, Any]] = []
    schema_errors: List[str] = []

    # Fail fast: validate that formatted/multiline SQL compiles and runs in DB mode.
    if not demo_mode:
        all_columns: List[str] = []
        for idx, spec in enumerate(query_specs):
            try:
                preview_query = _build_query_for_template(
                    template=str(spec["template"]),
                    uses_template=bool(spec["uses_template"]),
                    uses_paging_template=bool(spec["uses_paging_template"]),
                    period_start_iso=period_start_iso,
                    period_end_iso=period_end_iso,
                    limit=1,
                    offset=0,
                )
                preview_df = deps.query_logs_df(preview_query)
                all_columns.extend([str(col) for col in preview_df.columns])
                logs_preview_frames.append({"idx": idx, "df": preview_df})
            except Exception as exc:  # noqa: BLE001
                err = f"Preview запроса #{idx + 1} завершился ошибкой: {exc}"
                preview_query_errors.append(err)
                deps.logger.warning("logs_summary_page.preview_query_failed[%s]: %s", idx + 1, exc)
        for idx, spec in enumerate(metrics_query_specs):
            try:
                preview_query = _build_query_for_template(
                    template=str(spec["template"]),
                    uses_template=bool(spec["uses_template"]),
                    uses_paging_template=bool(spec["uses_paging_template"]),
                    period_start_iso=period_start_iso,
                    period_end_iso=period_end_iso,
                    limit=1,
                    offset=0,
                )
                preview_df = deps.query_metrics_df(preview_query)
                metrics_preview_frames.append({"idx": idx, "df": preview_df})
            except Exception as exc:  # noqa: BLE001
                err = f"Preview метрик #{idx + 1} завершился ошибкой: {exc}"
                preview_metrics_errors.append(err)
                deps.logger.warning("logs_summary_page.preview_metrics_failed[%s]: %s", idx + 1, exc)
        # stable unique
        seen: set[str] = set()
        preview_available_columns = []
        for col in all_columns:
            key = col.lower()
            if key in seen:
                continue
            seen.add(key)
            preview_available_columns.append(col)

        schema_errors.extend(_validate_logs_merge_schema(logs_preview_frames))
        schema_errors.extend(_validate_metrics_merge_schema(metrics_preview_frames))
        if preview_query_errors:
            schema_errors.extend(
                [f"Логи: не удалось провалидировать запрос: {err}" for err in preview_query_errors]
            )
        if preview_metrics_errors:
            schema_errors.extend(
                [f"Метрики: не удалось провалидировать запрос: {err}" for err in preview_metrics_errors]
            )
        if not logs_preview_frames:
            schema_errors.append("Логи: ни один SQL не дал preview-результат для проверки merge-схемы.")

    if schema_errors:
        joined = "\n".join([str(err) for err in schema_errors[:12]])
        if len(schema_errors) > 12:
            joined += f"\n... и еще {len(schema_errors) - 12} ошибок."
        _unlock_with_form_error(
            "Формат SQL-результатов несовместим для merge в единые DataFrame.\n" + joined
        )

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = deps.output_dir / "logs_summary_live" / f"run_{run_stamp}"
    live_events_path = run_dir / "events.jsonl"
    live_batches_path = run_dir / "batches.jsonl"
    logs_fetch_mode = _normalize_logs_fetch_mode(deps.logs_fetch_mode)
    initial_events: List[str] = []
    if preview_query_errors or preview_metrics_errors:
        initial_events.append(
            f"Предварительных ошибок ClickHouse: {len(preview_query_errors) + len(preview_metrics_errors)}"
        )
    if demo_mode:
        initial_events.append(
            f"Старт: режим выборки логов по количеству (demo synthetic, rows={demo_logs_count})."
        )
    elif logs_fetch_mode == "tail_n_logs":
        initial_events.append(
            f"Старт: режим выборки логов по количеству (tail_n_logs, limit={deps.logs_tail_limit})."
        )
    else:
        initial_events.append(
            f"Старт: режим выборки логов по датам ({period_start_iso} -> {period_end_iso})."
        )

    state: Dict[str, Any] = {
        "status": "queued",
        "mode": "demo" if demo_mode else "db",
        "logs_fetch_mode": logs_fetch_mode,
        "logs_tail_limit": int(deps.logs_tail_limit),
        "demo_logs_count": int(demo_logs_count),
        "query": sql_query_clean,
        "metrics_query": metrics_query_clean,
        "queries_count": len(query_specs),
        "metrics_queries_count": len(metrics_query_specs),
        "user_goal": user_goal.strip(),
        "period_mode": period_mode,
        "period_start": period_start_iso,
        "period_end": period_end_iso,
        "window_minutes": window_minutes,
        "db_batch_size": db_batch_size,
        "llm_batch_size": llm_batch_size,
        "map_workers": map_workers,
        "max_retries": max_retries,
        "llm_timeout": llm_timeout,
        "logs_processed": 0,
        "logs_total": None,
        "events": initial_events,
        "query_errors": list(preview_query_errors) + list(preview_metrics_errors),
        "map_batches": [],
        "final_summary": None,
        "freeform_final_summary": None,
        "metrics_rows": 0,
        "metrics_services": [],
        "metrics_context_text": "",
        "stats": None,
        "error": None,
        "result_json_path": None,
        "result_summary_path": None,
        "live_events_path": str(live_events_path),
        "live_batches_path": str(live_batches_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "started_monotonic": time.monotonic(),
        "elapsed_seconds": 0.0,
        "active_step": "Инициализация пайплайна",
        "llm_active": False,
        "llm_calls_started": 0,
        "llm_calls_succeeded": 0,
        "llm_calls_failed": 0,
        "llm_last_duration_sec": None,
        "llm_last_error": None,
        "llm_last_attempt": None,
        "llm_timeline": [],
        "read_timeout_active": False,
        "read_timeout_started_at": None,
        "read_timeout_last_at": None,
        "read_timeout_count": 0,
        "read_timeout_last_episode": None,
        "rows_per_second": None,
        "progress_samples": [],
        "eta_seconds_left": None,
        "eta_finish_at": None,
    }
    _render_logs_summary_chat(analysis_placeholder, state, deps)

    request_payload = {
        "sql_query": sql_query_clean,
        "sql_queries_count": len(query_specs),
        "metrics_query": metrics_query_clean,
        "metrics_queries_count": len(metrics_query_specs),
        "logs_fetch_mode": logs_fetch_mode,
        "logs_tail_limit": int(deps.logs_tail_limit),
        "user_goal": user_goal.strip(),
        "period_mode": period_mode,
        "period_start": period_start_iso,
        "period_end": period_end_iso,
        "window_minutes": window_minutes,
        "demo_mode": demo_mode,
        "demo_logs_count": demo_logs_count,
        "db_batch_size": db_batch_size,
        "llm_batch_size": llm_batch_size,
        "live_events_path": str(live_events_path),
        "live_batches_path": str(live_batches_path),
    }

    try:
        demo_logs: List[Dict[str, Any]] = []
        total_rows_estimate: Optional[int] = None
        columns = list(DEFAULT_SUMMARY_COLUMNS)
        if demo_mode:
            demo_logs = _build_demo_logs_for_window(
                period_start_dt=period_start_dt,
                period_end_dt=period_end_dt,
                total_logs=demo_logs_count,
            )
            if demo_logs:
                columns = _choose_summary_columns(list(demo_logs[0].keys()))
            total_rows_estimate = len(demo_logs)
        elif preview_available_columns:
            columns = _choose_summary_columns(preview_available_columns)
        query_error_seen: set[str] = set(str(item) for item in state.get("query_errors", []))

        def _register_query_error(message: str) -> None:
            text = str(message)
            if text in query_error_seen:
                return
            query_error_seen.add(text)
            query_errors = state.setdefault("query_errors", [])
            if isinstance(query_errors, list):
                query_errors.append(text)
            events = state.setdefault("events", [])
            if isinstance(events, list):
                events.append(f"ClickHouse error: {text}")

        def _push_live_event(message: str, *, render_now: bool = False) -> None:
            events = state.setdefault("events", [])
            if isinstance(events, list):
                events.append(str(message))
                if len(events) > MAX_EVENT_LINES:
                    state["events"] = events[-MAX_EVENT_LINES:]
            if render_now and threading.current_thread() is threading.main_thread():
                _render_logs_summary_chat(analysis_placeholder, state, deps)
                time.sleep(0.02)

        def _append_llm_timeline(
            *,
            event: str,
            call_no: Optional[int] = None,
            attempt: Optional[int] = None,
            total_attempts: Optional[int] = None,
            elapsed_sec: Optional[float] = None,
            status: Optional[str] = None,
            details: Optional[str] = None,
        ) -> None:
            rows = state.setdefault("llm_timeline", [])
            if not isinstance(rows, list):
                rows = []
            row = {
                "time": datetime.now(MSK).strftime("%Y-%m-%d %H:%M:%S.%f MSK"),
                "event": str(event),
            }
            if call_no is not None:
                row["call_no"] = int(call_no)
            if attempt is not None:
                row["attempt"] = int(attempt)
            if total_attempts is not None:
                total_int = int(total_attempts)
                row["total_attempts"] = "∞" if total_int <= 0 else total_int
            if elapsed_sec is not None:
                row["elapsed_sec"] = round(float(elapsed_sec), 2)
            if status:
                row["status"] = str(status)
            if details:
                row["details"] = str(details)
            rows.append(row)
            if len(rows) > MAX_LLM_TIMELINE_ROWS:
                rows = rows[-MAX_LLM_TIMELINE_ROWS:]
            state["llm_timeline"] = rows

        def _start_read_timeout_episode(
            *,
            attempt: int,
            total_attempts: int,
            elapsed_sec: float,
            error_text: str,
        ) -> None:
            now_iso = datetime.now(timezone.utc).isoformat()
            if not bool(state.get("read_timeout_active", False)):
                state["read_timeout_active"] = True
                state["read_timeout_started_at"] = now_iso
                state["read_timeout_last_at"] = now_iso
                state["read_timeout_count"] = 1
                pair = _format_attempts_pair(attempt, total_attempts)
                state["active_step"] = f"Поймали ReadTimeout, запускаем ретраи ({pair})"
                _append_llm_timeline(
                    event="read_timeout_start",
                    call_no=int(state.get("llm_calls_started", 0)),
                    attempt=attempt,
                    total_attempts=total_attempts,
                    elapsed_sec=elapsed_sec,
                    status="error",
                    details=str(error_text or ""),
                )
                _push_live_event(
                    (
                        "ReadTimeout: начался период повторных ошибок. "
                        f"Старт: {_format_datetime_with_tz(now_iso)}."
                    ),
                    render_now=True,
                )
                return

            current_count = int(pd.to_numeric(state.get("read_timeout_count"), errors="coerce") or 0)
            state["read_timeout_count"] = current_count + 1
            state["read_timeout_last_at"] = now_iso

        def _finish_read_timeout_episode(*, resolution: str) -> None:
            if not bool(state.get("read_timeout_active", False)):
                return

            start_iso = state.get("read_timeout_started_at")
            end_iso = datetime.now(timezone.utc).isoformat()
            errors_count = int(pd.to_numeric(state.get("read_timeout_count"), errors="coerce") or 0)
            start_ts = pd.to_datetime(start_iso, utc=True, errors="coerce")
            end_ts = pd.to_datetime(end_iso, utc=True, errors="coerce")
            duration_sec: Optional[float] = None
            if not pd.isna(start_ts) and not pd.isna(end_ts):
                duration_sec = max((end_ts - start_ts).total_seconds(), 0.0)

            state["read_timeout_active"] = False
            state["read_timeout_started_at"] = None
            state["read_timeout_last_at"] = end_iso
            state["read_timeout_count"] = 0
            state["read_timeout_last_episode"] = {
                "start": start_iso,
                "end": end_iso,
                "count": errors_count,
                "resolution": resolution,
                "duration_sec": round(duration_sec, 2) if duration_sec is not None else None,
            }
            _append_llm_timeline(
                event="read_timeout_end",
                call_no=int(state.get("llm_calls_started", 0)),
                elapsed_sec=duration_sec,
                status="ok" if resolution == "success" else "error",
                details=f"resolution={resolution}, count={errors_count}",
            )
            duration_text = (
                _format_eta_seconds(duration_sec)
                if duration_sec is not None
                else "n/a"
            )
            _push_live_event(
                (
                    "ReadTimeout: серия завершилась. "
                    f"Финиш: {_format_datetime_with_tz(end_iso)}; "
                    f"ошибок: {errors_count}; итог: {resolution}; "
                    f"длительность: {duration_text}."
                ),
                render_now=True,
            )

        # Two-level summarization: multi-query mode uses per-source summarizers (no merged stream).
        streaming_logs_merger: Optional[_StreamingLogsMerger] = None

        metrics_df = pd.DataFrame(columns=["timestamp", "value", "service"])
        if metrics_query_specs:
            metrics_frames: List[pd.DataFrame] = []
            if demo_mode:
                for idx, _spec in enumerate(metrics_query_specs):
                    service_label = f"metrics_query_{idx + 1}"
                    demo_chunk = _build_demo_metrics_for_window(
                        period_start_dt=period_start_dt,
                        period_end_dt=period_end_dt,
                        service_label=service_label,
                        total_points=min(max(db_batch_size, 120), 2000),
                    )
                    normalized_chunk = _normalize_metrics_df(
                        demo_chunk,
                        default_service=service_label,
                    )
                    if not normalized_chunk.empty:
                        metrics_frames.append(normalized_chunk)
            else:
                safe_metrics_page_limit = max(int(db_batch_size), 10)
                total_metrics_fetched = 0
                for idx, spec in enumerate(metrics_query_specs):
                    if total_metrics_fetched >= MAX_METRICS_ROWS_TOTAL:
                        _register_query_error(
                            f"Metrics query #{idx + 1} пропущен: достигнут лимит "
                            f"{MAX_METRICS_ROWS_TOTAL:,} строк метрик"
                        )
                        break
                    service_label = f"metrics_query_{idx + 1}"
                    metrics_offset = 0
                    max_pages = 2_000
                    for _page_idx in range(max_pages):
                        remaining_cap = MAX_METRICS_ROWS_TOTAL - total_metrics_fetched
                        if remaining_cap <= 0:
                            _register_query_error(
                                f"Metrics query #{idx + 1} прерван: достигнут лимит "
                                f"{MAX_METRICS_ROWS_TOTAL:,} строк метрик"
                            )
                            break
                        page_limit = min(safe_metrics_page_limit, remaining_cap)
                        metrics_query = _build_query_for_template(
                            template=str(spec["template"]),
                            uses_template=bool(spec["uses_template"]),
                            uses_paging_template=bool(spec["uses_paging_template"]),
                            period_start_iso=period_start_iso,
                            period_end_iso=period_end_iso,
                            limit=page_limit,
                            offset=metrics_offset,
                        )
                        try:
                            chunk_df = deps.query_metrics_df(metrics_query)
                        except Exception as exc:  # noqa: BLE001
                            _register_query_error(
                                f"Metrics query #{idx + 1} page(offset={metrics_offset}, "
                                f"limit={page_limit}) упал: {exc}"
                            )
                            deps.logger.warning(
                                "logs_summary_page.metrics_query_failed[%s]: offset=%s limit=%s err=%s",
                                idx + 1,
                                metrics_offset,
                                page_limit,
                                exc,
                            )
                            break

                        if chunk_df is None or chunk_df.empty:
                            break

                        normalized_chunk = _normalize_metrics_df(
                            chunk_df,
                            default_service=service_label,
                        )
                        if not normalized_chunk.empty:
                            metrics_frames.append(normalized_chunk)

                        page_rows = len(chunk_df)
                        metrics_offset += page_rows
                        total_metrics_fetched += page_rows
                        if page_rows < page_limit:
                            break
                    else:
                        _register_query_error(
                            f"Metrics query #{idx + 1} прерван после {max_pages} страниц "
                            f"(возможен цикл из-за нестабильного SQL-пейджинга)"
                        )

            if metrics_frames:
                metrics_df = pd.concat(metrics_frames, ignore_index=True)
                metrics_df = _sort_df_by_timestamp(metrics_df)

        metrics_context_text = _build_metrics_context(metrics_df)
        state["metrics_rows"] = int(len(metrics_df))
        state["metrics_services"] = (
            sorted({str(s) for s in metrics_df.get("service", pd.Series(dtype=str)).dropna().tolist()})
            if not metrics_df.empty and "service" in metrics_df.columns
            else []
        )
        state["metrics_context_text"] = metrics_context_text
        if metrics_query_specs:
            state.setdefault("events", []).append(
                "Контекст метрик: "
                f"{state['metrics_rows']} строк, сервисов {len(state['metrics_services'])}"
            )
            _render_logs_summary_chat(analysis_placeholder, state, deps)

        # Pre-count total rows so the progress bar and ETA are accurate from the first batch
        if not demo_mode and total_rows_estimate is None:
            state["status"] = "counting"
            state.setdefault("events", []).append("Подсчёт строк в базе...")
            _render_logs_summary_chat(analysis_placeholder, state, deps)
            counted = _precount_rows(
                query_specs=query_specs,
                query_logs_df=deps.query_logs_df,
                period_start_iso=period_start_iso,
                period_end_iso=period_end_iso,
                logger=deps.logger,
                on_error=_register_query_error,
            )
            if counted is not None:
                total_rows_estimate = counted
                state["logs_total"] = counted
                state.setdefault("events", []).append(f"Найдено строк: {counted:,}")
                _render_logs_summary_chat(analysis_placeholder, state, deps)

        # Keyset pagination state for the single-query (non-streaming) path.
        # Mutable list so the closure can update the value between calls.
        _single_query_last_ts: List[Optional[str]] = [None]

        def _db_fetch_page(
            *,
            columns: List[str],
            period_start: str,
            period_end: str,
            limit: int,
            offset: int,
        ) -> List[Dict[str, Any]]:
            if demo_mode:
                rows = demo_logs[offset : offset + max(int(limit), 1)]
                return [dict(row) for row in rows]

            safe_limit = max(int(limit), 1)
            safe_offset = max(int(offset), 0)

            if not multi_query_mode:
                spec = query_specs[0]
                template = str(spec["template"])
                uses_keyset = "{last_ts}" in template.lower()
                query = _build_query_for_template(
                    template=template,
                    uses_template=bool(spec["uses_template"]),
                    uses_paging_template=bool(spec["uses_paging_template"]),
                    period_start_iso=period_start,
                    period_end_iso=period_end,
                    limit=safe_limit,
                    offset=safe_offset,
                    last_ts=_single_query_last_ts[0],  # None → period_start on first page
                )
                try:
                    df = deps.query_logs_df(query)
                except Exception as exc:  # noqa: BLE001
                    if uses_keyset:
                        _register_query_error(
                            f"Запрос #1 page(last_ts={_single_query_last_ts[0]}, "
                            f"limit={safe_limit}) упал: {exc}"
                        )
                    else:
                        _register_query_error(
                            f"Запрос #1 page(offset={safe_offset}, limit={safe_limit}) упал: {exc}"
                        )
                    deps.logger.warning(
                        "logs_summary_page.query_failed[single]: offset=%s limit=%s err=%s",
                        safe_offset,
                        safe_limit,
                        exc,
                    )
                    return []
                if df.empty:
                    return []
                df = _sort_df_by_timestamp(df)
                # Advance keyset cursor so the next call fetches the next page
                if uses_keyset and "timestamp" in df.columns:
                    max_ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").max()
                    if not pd.isna(max_ts):
                        _single_query_last_ts[0] = max_ts.isoformat()
                return [dict(row) for row in df.to_dict(orient="records")]

            # Multi-query mode:
            if streaming_logs_merger is None:
                return []
            return streaming_logs_merger.next_page(
                period_start=period_start,
                period_end=period_end,
                limit=safe_limit,
                offset=safe_offset,
            )

        def _on_progress(event: str, payload: Dict[str, Any]) -> None:
            if not isinstance(payload, dict):
                payload = {}

            _append_jsonl(
                live_events_path,
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "event": event,
                    "rows_processed": payload.get("rows_processed"),
                    "rows_total": payload.get("rows_total"),
                    "page_index": payload.get("page_index"),
                    "page_rows": payload.get("page_rows"),
                    "batch_index": payload.get("batch_index"),
                    "batch_total": payload.get("batch_total"),
                    "reduce_round": payload.get("reduce_round"),
                    "group_index": payload.get("group_index"),
                    "group_total": payload.get("group_total"),
                },
            )

            events = state.setdefault("events", [])
            if event == "map_start":
                state["status"] = "map"
                state["active_step"] = "Читаем логи и готовим MAP-батчи"
                events.append("Map этап запущен")
            elif event == "page_fetched":
                state["status"] = "map"
                state["active_step"] = "Выгружаем страницу логов из БД"
                events.append(
                    f"Страница #{payload.get('page_index')}: {payload.get('page_rows')} строк"
                )
            elif event == "map_batch_start":
                state["status"] = "map"
                idx = int(payload.get("batch_index", 0)) + 1
                total = payload.get("batch_total")
                retries_label = "∞" if int(max_retries) < 0 else str(max_retries)
                state["active_step"] = (
                    f"LLM анализирует MAP-batch {idx}/{total}"
                    if total else f"LLM анализирует MAP-batch {idx}"
                )
                events.append(
                    (
                        f"LLM анализирует batch {idx}/{total} "
                        f"(таймаут {llm_timeout}s, ретраи {retries_label})"
                    )
                    if total else
                    f"LLM анализирует batch {idx} (таймаут {llm_timeout}s, ретраи {retries_label})"
                )
            elif event == "map_batch":
                state["status"] = "map"
                state["active_step"] = "MAP-batch обработан, обновляем промежуточный результат"
                full_logs = payload.get("batch_logs", [])
                if not isinstance(full_logs, list):
                    full_logs = []

                # Track the timestamp of the last processed batch for timestamp-based
                # progress bar and ETA — no pre-counting of rows required.
                batch_period_end = payload.get("batch_period_end")
                if batch_period_end:
                    state["last_batch_ts"] = batch_period_end

                _append_jsonl(
                    live_batches_path,
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "event": "map_batch",
                        "batch_index": payload.get("batch_index"),
                        "batch_total": payload.get("batch_total"),
                        "batch_summary": payload.get("batch_summary"),
                        "batch_logs_count": payload.get("batch_logs_count"),
                        "batch_period_start": payload.get("batch_period_start"),
                        "batch_period_end": payload.get("batch_period_end"),
                        "batch_logs": full_logs,
                    },
                )

                preview_logs = full_logs[:MAX_LOG_ROWS_PREVIEW]
                batch_item = {
                    "batch_index": payload.get("batch_index"),
                    "batch_total": payload.get("batch_total"),
                    "batch_summary": payload.get("batch_summary"),
                    "batch_logs_count": payload.get("batch_logs_count"),
                    "batch_period_start": payload.get("batch_period_start"),
                    "batch_period_end": payload.get("batch_period_end"),
                    "batch_logs": preview_logs,
                }
                map_batches = state.setdefault("map_batches", [])
                map_batches.append(batch_item)
                if len(map_batches) > MAX_RENDERED_BATCHES:
                    state["map_batches"] = map_batches[-MAX_RENDERED_BATCHES:]

                idx = int(payload.get("batch_index", 0)) + 1
                total = payload.get("batch_total")
                events.append(f"Map summary {idx}/{total}" if total else f"Map summary {idx}")
            elif event == "map_done":
                state["status"] = "reduce"
                state["active_step"] = "MAP завершён, готовим REDUCE"
                events.append("Map этап завершен")
                # Warn if processing stopped early due to DB errors
                if state.get("query_errors"):
                    events.append(
                        "⚠️ ВНИМАНИЕ: часть данных не получена из БД — суммаризация неполная. "
                        "Проверьте ошибки ClickHouse ниже."
                    )
            elif event == "reduce_start":
                state["status"] = "reduce"
                state["active_step"] = "REDUCE: объединяем промежуточные summary"
                events.append("Reduce этап запущен")
            elif event == "reduce_group_start":
                state["status"] = "reduce"
                round_idx = payload.get("reduce_round")
                group_index = int(payload.get("group_index", 0)) + 1
                group_total = payload.get("group_total")
                state["active_step"] = f"REDUCE round {round_idx}, группа {group_index}/{group_total}"
                events.append(f"Reduce round {round_idx}: группа {group_index}/{group_total}")
            elif event == "reduce_group_done":
                state["status"] = "reduce"
            elif event == "reduce_done":
                state["status"] = "summary_ready"
                state["active_step"] = "REDUCE завершён, собираем финальный отчёт"
                if payload.get("summary") is not None:
                    state["final_summary"] = str(payload.get("summary"))
                events.append("Reduce этап завершен")
            elif event == "fetch_error":
                error_msg = str(payload.get("error", "ClickHouse query failed"))
                _register_query_error(error_msg)
                events.append(f"ClickHouse error: {error_msg}")
            elif event == "freeform_start":
                state["active_step"] = "LLM пишет финальный narrative-отчёт"
                events.append("Генерация финального нарратива...")
            elif event == "freeform_done":
                state["active_step"] = "Финальный narrative-отчёт готов"
                freeform = payload.get("freeform_summary")
                if freeform:
                    state["freeform_final_summary"] = str(freeform)
                events.append("Нарратив готов")
            else:
                events.append(f"Событие: {event}")

            if len(events) > MAX_EVENT_LINES:
                state["events"] = events[-MAX_EVENT_LINES:]

            progress_rows = payload.get("rows_processed")
            if progress_rows is None:
                progress_rows = payload.get("rows_fetched")
            if progress_rows is not None:
                state["logs_processed"] = int(progress_rows)
            if payload.get("rows_total") is not None:
                state["logs_total"] = int(payload.get("rows_total"))

            _estimate_eta(state, event, payload)
            _render_logs_summary_chat(analysis_placeholder, state, deps)
            time.sleep(0.01)

        state["status"] = "summarizing"
        state["active_step"] = "Подготовка LLM-контекста и запуск суммаризации"
        _render_logs_summary_chat(analysis_placeholder, state, deps)

        def _on_retry(attempt: int, total: int, exc: Exception) -> None:
            """Called by _make_llm_call before each retry. Thread-safe via GIL."""
            pair = _format_attempts_pair(attempt, total)
            msg = f"LLM retry {pair}: {type(exc).__name__}: {exc}"
            state["llm_last_error"] = str(exc)
            state["active_step"] = f"LLM retry {pair}"
            _append_llm_timeline(
                event="retry",
                attempt=attempt,
                total_attempts=total,
                status="retry",
                details=f"{type(exc).__name__}: {exc}",
            )
            _push_live_event(msg, render_now=True)

        def _on_llm_attempt(attempt: int, total_attempts: int, timeout_seconds: float) -> None:
            pair = _format_attempts_pair(attempt, total_attempts)
            state["llm_active"] = True
            state["llm_last_attempt"] = f"попытка {pair}, timeout={int(timeout_seconds)}s"
            if attempt == 1:
                state["llm_calls_started"] = int(state.get("llm_calls_started", 0)) + 1
                call_no = int(state.get("llm_calls_started", 0))
                state["active_step"] = f"LLM вызов #{call_no}: отправили prompt, ждём ответ"
                _append_llm_timeline(
                    event="call_start",
                    call_no=call_no,
                    attempt=attempt,
                    total_attempts=total_attempts,
                    status="started",
                    details=f"timeout={int(timeout_seconds)}s",
                )
                _push_live_event(
                    f"LLM вызов #{call_no}: старт (timeout {int(timeout_seconds)}s, attempts {_format_attempts_total(total_attempts)})",
                    render_now=True,
                )
            else:
                state["active_step"] = f"LLM повторная попытка {pair}"
                _append_llm_timeline(
                    event="attempt_start",
                    call_no=int(state.get("llm_calls_started", 0)),
                    attempt=attempt,
                    total_attempts=total_attempts,
                    status="retrying",
                    details=f"timeout={int(timeout_seconds)}s",
                )
                _push_live_event(
                    f"LLM: повторная попытка {pair} (timeout {int(timeout_seconds)}s)",
                    render_now=True,
                )

        def _on_llm_result(
            attempt: int,
            total_attempts: int,
            success: bool,
            elapsed_sec: float,
            error_text: Optional[str],
        ) -> None:
            state["llm_last_duration_sec"] = float(elapsed_sec)
            state["llm_active"] = False
            current_call_no = int(state.get("llm_calls_started", 0))
            if success:
                _finish_read_timeout_episode(resolution="success")
                state["llm_calls_succeeded"] = int(state.get("llm_calls_succeeded", 0)) + 1
                state["llm_last_error"] = None
                state["active_step"] = f"LLM ответ получен за {elapsed_sec:.1f}s"
                _append_llm_timeline(
                    event="call_done",
                    call_no=current_call_no,
                    attempt=attempt,
                    total_attempts=total_attempts,
                    elapsed_sec=elapsed_sec,
                    status="ok",
                    details="response received",
                )
                _push_live_event(f"LLM ответ получен ({elapsed_sec:.1f}s)", render_now=True)
            else:
                is_read_timeout = _is_read_timeout_error(error_text)
                if is_read_timeout:
                    _start_read_timeout_episode(
                        attempt=attempt,
                        total_attempts=total_attempts,
                        elapsed_sec=elapsed_sec,
                        error_text=str(error_text or ""),
                    )
                elif bool(state.get("read_timeout_active", False)):
                    _finish_read_timeout_episode(resolution="other_error")
                state["llm_calls_failed"] = int(state.get("llm_calls_failed", 0)) + 1
                state["llm_last_error"] = str(error_text or "")
                _append_llm_timeline(
                    event="call_failed",
                    call_no=current_call_no,
                    attempt=attempt,
                    total_attempts=total_attempts,
                    elapsed_sec=elapsed_sec,
                    status="error",
                    details=str(error_text or ""),
                )
                can_retry = total_attempts <= 0 or attempt < total_attempts
                next_pair = _format_attempts_pair(attempt + 1, total_attempts)
                if can_retry:
                    state["active_step"] = (
                        f"LLM ошибка за {elapsed_sec:.1f}s, готовим retry {next_pair}"
                    )
                    _push_live_event(
                        f"LLM ошибка ({elapsed_sec:.1f}s): {error_text}. Готовим retry {next_pair}",
                        render_now=True,
                    )
                else:
                    if is_read_timeout:
                        _finish_read_timeout_episode(resolution="fallback")
                    state["active_step"] = "LLM попытки исчерпаны, используем fallback"
                    _push_live_event(
                        f"LLM ошибка ({elapsed_sec:.1f}s): {error_text}. Попытки исчерпаны, fallback.",
                        render_now=True,
                    )

        base_llm_call = deps.make_llm_call(
            max_retries=max_retries,
            on_retry=_on_retry,
            on_attempt=_on_llm_attempt,
            on_result=_on_llm_result,
            llm_timeout=llm_timeout,
        )
        goal_text = user_goal.strip()
        llm_context_blocks: List[str] = []
        if goal_text:
            llm_context_blocks.append(
                "ИНЦИДЕНТ ДЛЯ РАССЛЕДОВАНИЯ:\n"
                f"{goal_text}\n\n"
                "ТВОЯ ЗАДАЧА: найти в логах конкретные события, которые объясняют перечисленные алерты.\n"
                "Для каждого алерта из контекста: было ли что-то в логах ДО него, что могло его вызвать?\n"
                "Привязывай каждый вывод к конкретному алерту (по имени/времени/ноде из контекста)."
            )
        if metrics_context_text.strip():
            llm_context_blocks.append(
                "МЕТРИКИ (агрегированный контекст по сервисам):\n"
                f"{metrics_context_text.strip()}"
            )

        context_prefix = "\n\n".join(llm_context_blocks).strip()

        def _llm_call_with_context(prompt: str) -> str:
            enriched_prompt = f"{context_prefix}\n\n{prompt}" if context_prefix else prompt
            return base_llm_call(enriched_prompt)

        llm_call = _llm_call_with_context

        if multi_query_mode and not demo_mode:
            # ---------------------------------------------------------------
            # Two-level summarization:
            #   1. For each source: independent MAP→REDUCE (own fetch, own summarizer).
            #   2. One cross-source REDUCE LLM call over per-source summaries.
            # MAP workers within each source still run in parallel (ThreadPoolExecutor
            # inside PeriodLogSummarizer). Sources are processed sequentially so there
            # are no concurrent thread pools running simultaneously.
            # ---------------------------------------------------------------
            from my_summarizer import build_cross_source_reduce_prompt  # noqa: PLC0415

            def _make_source_fetch_page(spec: Dict[str, Any], label: str) -> Callable:
                """Create an independent paged-fetch closure for a single query spec."""
                _last_ts: List[Optional[str]] = [None]
                _tmpl = str(spec["template"])
                _uses_keyset = "{last_ts}" in _tmpl.lower()
                _uses_tpl = bool(spec["uses_template"])
                _uses_paging_tpl = bool(spec["uses_paging_template"])

                def _fetch(
                    *,
                    columns: List[str],
                    period_start: str,
                    period_end: str,
                    limit: int,
                    offset: int,
                ) -> List[Dict[str, Any]]:
                    safe_limit = max(int(limit), 1)
                    safe_offset = max(int(offset), 0)
                    query = _build_query_for_template(
                        template=_tmpl,
                        uses_template=_uses_tpl,
                        uses_paging_template=_uses_paging_tpl,
                        period_start_iso=period_start,
                        period_end_iso=period_end,
                        limit=safe_limit,
                        offset=safe_offset,
                        last_ts=_last_ts[0],
                    )
                    try:
                        df = deps.query_logs_df(query)
                    except Exception as exc:  # noqa: BLE001
                        if _uses_keyset:
                            _register_query_error(
                                f"{label} page(last_ts={_last_ts[0]}, limit={safe_limit}) упал: {exc}"
                            )
                        else:
                            _register_query_error(
                                f"{label} page(offset={safe_offset}, limit={safe_limit}) упал: {exc}"
                            )
                        deps.logger.warning(
                            "logs_summary_page.per_source_fetch_failed[%s]: offset=%s limit=%s err=%s",
                            label, safe_offset, safe_limit, exc,
                        )
                        return []
                    if df is None or df.empty:
                        return []
                    df = _sort_df_by_timestamp(df)
                    if _uses_keyset and "timestamp" in df.columns:
                        max_ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").max()
                        if not pd.isna(max_ts):
                            _last_ts[0] = max_ts.isoformat()
                    return [dict(row) for row in df.to_dict(orient="records")]

                return _fetch

            per_source_summaries: Dict[str, str] = {}
            agg_pages = agg_rows = agg_llm = agg_reduce = 0

            for src_idx, spec in enumerate(query_specs):
                src_label = str(spec.get("label", f"query_{src_idx + 1}"))
                state.setdefault("events", []).append(
                    f"--- Источник: {src_label} ({src_idx + 1}/{len(query_specs)}) ---"
                )
                state.pop("last_batch_ts", None)  # reset timestamp progress for each source
                _render_logs_summary_chat(analysis_placeholder, state, deps)

                src_summarizer = deps.period_log_summarizer_cls(
                    db_fetch_page=_make_source_fetch_page(spec, src_label),
                    llm_call=llm_call,
                    config=_build_config(deps, db_batch_size, llm_batch_size, map_workers),
                    on_progress=_on_progress,
                )
                try:
                    src_result = src_summarizer.summarize_period(
                        period_start=period_start_iso,
                        period_end=period_end_iso,
                        columns=columns,
                        total_rows_estimate=None,
                    )
                except TypeError:
                    src_result = src_summarizer.summarize_period(
                        period_start=period_start_iso,
                        period_end=period_end_iso,
                        columns=columns,
                    )

                src_summary = (src_result.summary or "").strip()
                if src_summary and src_summary != "Нет логов за указанный период.":
                    per_source_summaries[src_label] = src_summary
                agg_pages += src_result.pages_fetched
                agg_rows += src_result.rows_processed
                agg_llm += src_result.llm_calls
                agg_reduce += src_result.reduce_rounds

            # Cross-source REDUCE: merge per-source summaries into a single report
            if len(per_source_summaries) > 1:
                state.setdefault("events", []).append("Кросс-источниковый анализ (финальный REDUCE)...")
                _render_logs_summary_chat(analysis_placeholder, state, deps)
                cross_prompt = build_cross_source_reduce_prompt(
                    summaries_by_source=per_source_summaries,
                    period_start=period_start_iso,
                    period_end=period_end_iso,
                )
                enriched_cross = f"{context_prefix}\n\n{cross_prompt}" if context_prefix else cross_prompt
                try:
                    cross_summary = llm_call(enriched_cross).strip()
                    agg_llm += 1
                    final_summary_text = cross_summary or "\n\n---\n\n".join(
                        f"=== {src} ===\n{s}" for src, s in per_source_summaries.items()
                    )
                except Exception as cross_exc:  # noqa: BLE001
                    deps.logger.warning("Cross-source reduce failed: %s", cross_exc)
                    final_summary_text = "\n\n---\n\n".join(
                        f"=== {src} ===\n{s}" for src, s in per_source_summaries.items()
                    )
                state.setdefault("events", []).append("Кросс-источниковый анализ завершён")
            elif per_source_summaries:
                final_summary_text = next(iter(per_source_summaries.values()))
            else:
                final_summary_text = "Нет логов за указанный период."

            state["status"] = "done"
            state["active_step"] = "Суммаризация завершена"
            state["final_summary"] = final_summary_text
            state["stats"] = {
                "pages_fetched": agg_pages,
                "rows_processed": agg_rows,
                "llm_calls": agg_llm,
                "reduce_rounds": agg_reduce,
            }
            state["logs_processed"] = agg_rows
        else:
            # Single-query or demo mode: use _db_fetch_page directly
            summarizer = deps.period_log_summarizer_cls(
                db_fetch_page=_db_fetch_page,
                llm_call=llm_call,
                config=_build_config(deps, db_batch_size, llm_batch_size, map_workers),
                on_progress=_on_progress,
            )
            try:
                result = summarizer.summarize_period(
                    period_start=period_start_iso,
                    period_end=period_end_iso,
                    columns=columns,
                    total_rows_estimate=total_rows_estimate,
                )
            except TypeError:
                result = summarizer.summarize_period(
                    period_start=period_start_iso,
                    period_end=period_end_iso,
                    columns=columns,
                )

            state["status"] = "done"
            state["active_step"] = "Суммаризация завершена"
            state["final_summary"] = str(result.summary)
            state["stats"] = {
                "pages_fetched": result.pages_fetched,
                "rows_processed": result.rows_processed,
                "llm_calls": result.llm_calls,
                "reduce_rounds": result.reduce_rounds,
            }
            state["logs_processed"] = int(result.rows_processed)
            if state.get("logs_total") is None and total_rows_estimate is not None:
                state["logs_total"] = int(total_rows_estimate)

        if state.get("final_summary"):
            try:
                events = state.setdefault("events", [])
                events.append("Готовим расширенный финальный отчет в свободном формате")
                _render_logs_summary_chat(analysis_placeholder, state, deps)
                freeform_prompt = _build_freeform_summary_prompt(
                    final_summary=str(state.get("final_summary", "")),
                    user_goal=goal_text,
                    period_start=period_start_iso,
                    period_end=period_end_iso,
                    stats=state.get("stats") or {},
                    metrics_context=metrics_context_text,
                )
                freeform_summary = str(base_llm_call(freeform_prompt)).strip()
                if freeform_summary:
                    state["freeform_final_summary"] = freeform_summary
                    events.append("Свободный финальный отчет готов")
                else:
                    events.append("Свободный финальный отчет пустой, используем основной")
            except Exception as freeform_exc:  # noqa: BLE001
                deps.logger.warning("freeform final report generation failed: %s", freeform_exc)
                state.setdefault("events", []).append(
                    "Не удалось сгенерировать свободный финальный отчет"
                )
        _estimate_eta(state, "done", {})

    except Exception as exc:  # noqa: BLE001
        state["status"] = "error"
        state["active_step"] = "Ошибка выполнения"
        state["error"] = str(exc)
        _estimate_eta(state, "error", {})
        deps.logger.exception("logs_summary_page.run_failed")
        with runtime_error_placeholder.container():
            st.error(f"Ошибка выполнения: {exc}")

    saved = _save_logs_summary_result(
        output_dir=deps.output_dir,
        request_payload=request_payload,
        result_state=state,
    )
    state["result_json_path"] = saved.get("json_path")
    state["result_summary_path"] = saved.get("summary_path")
    st.session_state[LAST_STATE_SESSION_KEY] = state
    st.session_state[RUNNING_SESSION_KEY] = False
    st.session_state.pop(RUN_PARAMS_SESSION_KEY, None)
    st.rerun()
