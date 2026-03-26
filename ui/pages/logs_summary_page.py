from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st


MAX_EVENT_LINES = 160
MAX_RENDERED_BATCHES = 10
MAX_LOG_ROWS_PREVIEW = 80
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


@dataclass(frozen=True)
class LogsSummaryPageDeps:
    logger: logging.Logger
    batch_size: int
    test_mode: bool
    loopback_minutes: int
    logs_tail_limit: int
    period_log_summarizer_cls: Any
    summarizer_config_cls: Any
    make_llm_call: Callable[[], Callable[[str], str]]
    query_logs_df: Callable[[str], pd.DataFrame]
    render_scrollable_text: Callable[..., None]
    render_pretty_summary_text: Callable[..., None]
    infer_batch_period: Callable[[Dict[str, Any]], tuple[Optional[str], Optional[str]]]
    summary_text_height: int
    final_text_height: int
    logs_batch_table_height: int
    sql_textarea_height: int
    default_sql_query: str
    output_dir: Path


def _escape_sql_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "''")


def _query_uses_any_placeholders(base_query: str) -> bool:
    lowered = base_query.lower()
    return any(
        token in lowered
        for token in (
            "{period_start}",
            "{period_end}",
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
    return "{limit}" in lowered and "{offset}" in lowered


def _render_query_template(
    *,
    query_template: str,
    period_start_iso: str,
    period_end_iso: str,
    limit: int,
    offset: int,
) -> str:
    safe_limit = max(int(limit), 1)
    safe_offset = max(int(offset), 0)
    safe_start = _escape_sql_literal(period_start_iso)
    safe_end = _escape_sql_literal(period_end_iso)
    rendered = query_template.strip().rstrip(";")
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
    }
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


def _wrap_with_limit_offset(*, query: str, limit: int, offset: int) -> str:
    safe_limit = max(int(limit), 1)
    safe_offset = max(int(offset), 0)
    base = query.strip().rstrip(";")
    return (
        "SELECT * FROM ("
        f"{base}"
        ") AS cp_logs_page "
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
    base = base_query.strip().rstrip(";")
    start_escaped = _escape_sql_literal(period_start_iso)
    end_escaped = _escape_sql_literal(period_end_iso)
    return (
        "SELECT * FROM ("
        "SELECT * FROM ("
        f"{base}"
        ") AS cp_src "
        f"WHERE timestamp >= parseDateTimeBestEffort('{start_escaped}') "
        f"AND timestamp < parseDateTimeBestEffort('{end_escaped}') "
        "ORDER BY timestamp"
        ") AS cp_window "
        f"LIMIT {safe_limit} OFFSET {safe_offset}"
    )


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


def _render_logs_summary_chat(container, state: Dict[str, Any], deps: LogsSummaryPageDeps) -> None:
    status_titles = {
        "queued": "В очереди",
        "summarizing": "Подготовка summary",
        "map": "Map этап",
        "reduce": "Reduce этап",
        "summary_ready": "Summary готов",
        "done": "Завершено",
        "error": "Ошибка",
    }

    with container.container():
        st.subheader("Logs Summarizer")
        if not state:
            return

        with st.chat_message("user"):
            period_mode = str(state.get("period_mode", "window"))
            period_desc = (
                f"Окно: `+-{state.get('window_minutes')}` минут"
                if period_mode == "window"
                else "Режим: `start/end диапазон`"
            )
            st.markdown(
                "\n".join(
                    [
                        f"Режим: `{state.get('mode', 'db')}`",
                        f"Период: `{state.get('period_start')}` -> `{state.get('period_end')}`",
                        period_desc,
                        f"Batch size: `{state.get('batch_size')}`",
                    ]
                )
            )
            deps.render_scrollable_text(state.get("query", ""), height=130)
            goal = str(state.get("user_goal", "")).strip()
            if goal:
                st.markdown("Контекст пользователя")
                deps.render_scrollable_text(goal, height=110)

        with st.chat_message("assistant"):
            status = str(state.get("status", "queued"))
            st.markdown(f"Статус: **{status_titles.get(status, status)}**")
            logs_processed = state.get("logs_processed")
            logs_total = state.get("logs_total")
            processed_num = pd.to_numeric(logs_processed, errors="coerce")
            total_num = pd.to_numeric(logs_total, errors="coerce")
            if not pd.isna(processed_num):
                if not pd.isna(total_num) and float(total_num) > 0:
                    ratio = min(max(float(processed_num) / float(total_num), 0.0), 1.0)
                    st.progress(
                        ratio,
                        text=f"Прогресс: {int(processed_num)}/{int(total_num)}",
                    )
                else:
                    st.caption(f"Прогресс: обработано {int(processed_num)} строк")
            for line in state.get("events", [])[-10:]:
                st.caption(str(line))

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
                deps.render_pretty_summary_text(batch.get("batch_summary", ""), height=deps.summary_text_height)
                batch_logs_count = batch.get("batch_logs_count")
                if batch_logs_count is None:
                    batch_logs_count = len(batch_logs)
                st.caption(f"Логов в батче: {batch_logs_count}")
                period_start, period_end = deps.infer_batch_period(batch)
                if period_start and period_end:
                    st.caption(f"Период логов батча: `{period_start}` -> `{period_end}`")
                if batch_logs:
                    st.dataframe(
                        pd.DataFrame(batch_logs),
                        use_container_width=True,
                        hide_index=True,
                        height=deps.logs_batch_table_height,
                    )

        if state.get("final_summary"):
            with st.chat_message("assistant"):
                st.markdown("Итоговый Reduce summary")
                deps.render_pretty_summary_text(state["final_summary"], height=deps.final_text_height)

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


def _build_config(deps: LogsSummaryPageDeps, batch_size: int) -> Any:
    try:
        return deps.summarizer_config_cls(
            page_limit=batch_size,
            llm_chunk_rows=batch_size,
            keep_map_batches_in_memory=False,
            keep_map_summaries_in_result=False,
        )
    except TypeError:
        return deps.summarizer_config_cls(
            page_limit=batch_size,
            llm_chunk_rows=batch_size,
        )


def render_logs_summary_page(deps: LogsSummaryPageDeps) -> None:
    st.title("Logs Summarizer")

    running_key = "logs_summary_running"
    if running_key not in st.session_state:
        st.session_state[running_key] = False
    is_running = bool(st.session_state[running_key])

    default_query = (deps.default_sql_query or "").strip() or (
        "SELECT timestamp, level, message FROM logs_demo_service "
        "WHERE timestamp >= parseDateTimeBestEffort('{period_start}') "
        "AND timestamp < parseDateTimeBestEffort('{period_end}') "
        "ORDER BY timestamp DESC LIMIT {limit} OFFSET {offset}"
    )

    sql_query = st.text_area(
        "SQL запрос логов",
        value=default_query,
        height=max(int(deps.sql_textarea_height), 180),
        help=(
            "Поддерживаются многострочные SQL. Можно использовать плейсхолдеры: "
            "{period_start}, {period_end}, {start}, {end}, {limit}, {offset}."
        ),
        disabled=is_running,
    )
    user_goal = st.text_area(
        "Дополнительный контекст для LLM (опционально)",
        value="",
        height=140,
        disabled=is_running,
    )

    period_mode_label = st.radio(
        "Режим периода",
        options=("Окно вокруг даты (±N минут)", "Явный диапазон (start/end)"),
        index=0,
        horizontal=True,
        disabled=is_running,
    )
    is_window_mode = period_mode_label.startswith("Окно вокруг")

    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    center_default = now_utc.isoformat().replace("+00:00", "Z")
    start_default = (now_utc - timedelta(minutes=max(int(deps.loopback_minutes), 1))).isoformat().replace(
        "+00:00", "Z"
    )
    end_default = now_utc.isoformat().replace("+00:00", "Z")

    center_dt_text = ""
    window_minutes = max(int(deps.loopback_minutes), 1)
    start_dt_text = ""
    end_dt_text = ""
    if is_window_mode:
        center_dt_text = st.text_input(
            "Целевая дата/время (ISO)",
            value=center_default,
            disabled=is_running,
        )
        window_minutes = int(
            st.number_input(
                "Окно анализа (+- N минут)",
                min_value=1,
                max_value=60 * 24 * 30,
                value=max(int(deps.loopback_minutes), 1),
                step=1,
                disabled=is_running,
            )
        )
    else:
        start_dt_text = st.text_input(
            "Дата/время начала (ISO)",
            value=start_default,
            disabled=is_running,
        )
        end_dt_text = st.text_input(
            "Дата/время конца (ISO)",
            value=end_default,
            disabled=is_running,
        )

    batch_size = int(
        st.number_input(
            "Общий размер batch (и для БД, и для LLM)",
            min_value=10,
            max_value=5_000,
            value=max(int(deps.batch_size), 1),
            step=10,
            disabled=is_running,
        )
    )
    demo_mode = st.toggle(
        "Демо режим (без БД)",
        value=bool(deps.test_mode),
        disabled=is_running,
    )
    demo_logs_count = int(
        st.number_input(
            "Количество логов в демо режиме",
            min_value=100,
            max_value=50_000,
            value=max(int(deps.logs_tail_limit), 1000),
            step=100,
            disabled=is_running or not demo_mode,
        )
    )

    run_clicked = st.button(
        "Запустить Суммаризацию Логов",
        type="primary",
        use_container_width=True,
        disabled=is_running,
    )

    analysis_placeholder = st.empty()
    if not run_clicked:
        return

    if not sql_query.strip():
        st.error("SQL запрос не должен быть пустым.")
        return

    period_mode = "window" if is_window_mode else "start_end"
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
        st.error(str(exc))
        return

    period_start_iso = period_start_dt.isoformat()
    period_end_iso = period_end_dt.isoformat()
    sql_query_clean = sql_query.strip()

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = deps.output_dir / "logs_summary_live" / f"run_{run_stamp}"
    live_events_path = run_dir / "events.jsonl"
    live_batches_path = run_dir / "batches.jsonl"

    state: Dict[str, Any] = {
        "status": "queued",
        "mode": "demo" if demo_mode else "db",
        "query": sql_query_clean,
        "user_goal": user_goal.strip(),
        "period_mode": period_mode,
        "period_start": period_start_iso,
        "period_end": period_end_iso,
        "window_minutes": window_minutes,
        "batch_size": batch_size,
        "logs_processed": 0,
        "logs_total": None,
        "events": [],
        "map_batches": [],
        "final_summary": None,
        "stats": None,
        "error": None,
        "result_json_path": None,
        "result_summary_path": None,
        "live_events_path": str(live_events_path),
        "live_batches_path": str(live_batches_path),
    }
    _render_logs_summary_chat(analysis_placeholder, state, deps)

    request_payload = {
        "sql_query": sql_query_clean,
        "user_goal": user_goal.strip(),
        "period_mode": period_mode,
        "period_start": period_start_iso,
        "period_end": period_end_iso,
        "window_minutes": window_minutes,
        "demo_mode": demo_mode,
        "demo_logs_count": demo_logs_count,
        "batch_size": batch_size,
        "live_events_path": str(live_events_path),
        "live_batches_path": str(live_batches_path),
    }

    st.session_state[running_key] = True
    try:
        uses_template = _query_uses_any_placeholders(sql_query_clean)
        uses_paging_template = _query_uses_paging_placeholders(sql_query_clean)

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
                columns = list(demo_logs[0].keys())
            total_rows_estimate = len(demo_logs)

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

            if uses_template:
                rendered_query = _render_query_template(
                    query_template=sql_query_clean,
                    period_start_iso=period_start,
                    period_end_iso=period_end,
                    limit=limit,
                    offset=offset,
                )
                query = (
                    rendered_query
                    if uses_paging_template
                    else _wrap_with_limit_offset(query=rendered_query, limit=limit, offset=offset)
                )
            else:
                query = _build_window_query_for_plain_sql(
                    base_query=sql_query_clean,
                    period_start_iso=period_start,
                    period_end_iso=period_end,
                    limit=limit,
                    offset=offset,
                )

            df = deps.query_logs_df(query)
            if df.empty:
                return []
            return [dict(row) for row in df.to_dict(orient="records")]

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
                events.append("Map этап запущен")
            elif event == "page_fetched":
                state["status"] = "map"
                events.append(
                    f"Страница #{payload.get('page_index')}: {payload.get('page_rows')} строк"
                )
            elif event == "map_batch_start":
                state["status"] = "map"
                idx = int(payload.get("batch_index", 0)) + 1
                total = payload.get("batch_total")
                events.append(f"LLM анализирует batch {idx}/{total}" if total else f"LLM анализирует batch {idx}")
            elif event == "map_batch":
                state["status"] = "map"
                full_logs = payload.get("batch_logs", [])
                if not isinstance(full_logs, list):
                    full_logs = []

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
                events.append("Map этап завершен")
            elif event == "reduce_start":
                state["status"] = "reduce"
                events.append("Reduce этап запущен")
            elif event == "reduce_group_start":
                state["status"] = "reduce"
                round_idx = payload.get("reduce_round")
                group_index = int(payload.get("group_index", 0)) + 1
                group_total = payload.get("group_total")
                events.append(f"Reduce round {round_idx}: группа {group_index}/{group_total}")
            elif event == "reduce_group_done":
                state["status"] = "reduce"
            elif event == "reduce_done":
                state["status"] = "summary_ready"
                if payload.get("summary") is not None:
                    state["final_summary"] = str(payload.get("summary"))
                events.append("Reduce этап завершен")
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

            _render_logs_summary_chat(analysis_placeholder, state, deps)
            time.sleep(0.01)

        state["status"] = "summarizing"
        _render_logs_summary_chat(analysis_placeholder, state, deps)

        base_llm_call = deps.make_llm_call()
        goal_text = user_goal.strip()
        if goal_text:

            def _llm_call_with_goal(prompt: str) -> str:
                enriched_prompt = (
                    "Контекст пользователя для этого summary:\n"
                    f"{goal_text}\n\n"
                    "Учитывай этот контекст при приоритизации проблем и рекомендаций.\n\n"
                    f"{prompt}"
                )
                return base_llm_call(enriched_prompt)

            llm_call = _llm_call_with_goal
        else:
            llm_call = base_llm_call

        summarizer = deps.period_log_summarizer_cls(
            db_fetch_page=_db_fetch_page,
            llm_call=llm_call,
            config=_build_config(deps, batch_size),
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

    except Exception as exc:  # noqa: BLE001
        state["status"] = "error"
        state["error"] = str(exc)
        deps.logger.exception("logs_summary_page.run_failed")
    finally:
        st.session_state[running_key] = False

    saved = _save_logs_summary_result(
        output_dir=deps.output_dir,
        request_payload=request_payload,
        result_state=state,
    )
    state["result_json_path"] = saved.get("json_path")
    state["result_summary_path"] = saved.get("summary_path")
    _render_logs_summary_chat(analysis_placeholder, state, deps)
