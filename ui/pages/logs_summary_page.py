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

LIVE_BATCHES_TO_RENDER = 6
FINAL_BATCHES_TO_RENDER = 24
MAX_BATCH_LOG_ROWS_PREVIEW = 80
MAX_EVENT_LINES = 200


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


def _query_uses_paging_placeholders(base_query: str) -> bool:
    lowered = base_query.lower()
    return any(token in lowered for token in ("{limit}", "{offset}"))


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
        "{page_limit}": str(safe_limit),
        "{offset}": str(safe_offset),
    }
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    return rendered


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

    return {
        "json_path": str(json_path),
        "summary_path": str(summary_path),
    }


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(payload), ensure_ascii=False) + "\n")


def _build_manual_logs_window_query(
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
        ") AS cp_manual_src "
        f"WHERE timestamp >= parseDateTimeBestEffort('{start_escaped}') "
        f"AND timestamp < parseDateTimeBestEffort('{end_escaped}') "
        "ORDER BY timestamp"
        ") AS cp_manual_window "
        f"LIMIT {safe_limit} OFFSET {safe_offset}"
    )


def _wrap_with_limit_offset(
    *,
    query: str,
    limit: int,
    offset: int,
) -> str:
    safe_limit = max(int(limit), 1)
    safe_offset = max(int(offset), 0)
    base = query.strip().rstrip(";")
    return f"SELECT * FROM ({base}) AS cp_manual_limited LIMIT {safe_limit} OFFSET {safe_offset}"


def _build_manual_logs_count_query(
    *,
    base_query: str,
    period_start_iso: str,
    period_end_iso: str,
) -> str:
    base = base_query.strip().rstrip(";")
    start_escaped = _escape_sql_literal(period_start_iso)
    end_escaped = _escape_sql_literal(period_end_iso)
    return (
        "SELECT count() AS total_rows FROM ("
        "SELECT * FROM ("
        f"{base}"
        ") AS cp_manual_src "
        f"WHERE timestamp >= parseDateTimeBestEffort('{start_escaped}') "
        f"AND timestamp < parseDateTimeBestEffort('{end_escaped}')"
        ") AS cp_manual_cnt"
    )


def _build_demo_logs_for_window(
    *,
    period_start_dt: datetime,
    period_end_dt: datetime,
    total_logs: int,
) -> List[Dict[str, Any]]:
    safe_total = max(int(total_logs), 1)
    window_seconds = max(int((period_end_dt - period_start_dt).total_seconds()), safe_total + 1)

    def _level_for(idx: int) -> str:
        phase = (idx + 1) / safe_total
        if phase < 0.35:
            levels = ("INFO", "INFO", "WARN")
        elif phase < 0.75:
            levels = ("INFO", "WARN", "ERROR")
        else:
            levels = ("WARN", "ERROR", "ERROR", "CRITICAL")
        return levels[idx % len(levels)]

    def _message_for(level: str, idx: int) -> str:
        if level == "CRITICAL":
            return f"critical incident #{idx}: request queue overflow and timeout storm"
        if level == "ERROR":
            return f"error #{idx}: upstream timeout while processing request"
        if level == "WARN":
            return f"warn #{idx}: retry rate increased above baseline"
        return f"info #{idx}: background processing in progress"

    rows: List[Dict[str, Any]] = []
    for idx in range(safe_total):
        sec_offset = int((idx + 1) * window_seconds / (safe_total + 1))
        ts = period_start_dt + timedelta(seconds=sec_offset)
        level = _level_for(idx)
        rows.append(
            {
                "timestamp": ts.isoformat().replace("+00:00", "Z"),
                "level": level,
                "message": _message_for(level, idx + 1),
                "service": "demo-service",
                "pod": f"demo-pod-{1 + (idx % 3)}",
                "container": "app",
                "node": f"node-{1 + (idx % 5):02d}",
                "cluster": "demo-cluster",
            }
        )
    return rows


def _render_logs_summary_chat(
    container,
    state: Dict[str, Any],
    deps: LogsSummaryPageDeps,
) -> None:
    with container.container():
        st.subheader("Суммаризация Логов")
        if not state:
            return

        with st.chat_message("user"):
            st.markdown("**Запуск Summarization Задачи**")
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
                    ]
                )
            )
            st.markdown("SQL-запрос")
            deps.render_scrollable_text(state.get("query", ""), height=130)
            user_goal = str(state.get("user_goal", "")).strip()
            if user_goal:
                st.markdown("Контекст инцидента / цель summary")
                deps.render_scrollable_text(user_goal, height=110)

        with st.chat_message("assistant"):
            status = str(state.get("status", "queued"))
            status_titles = {
                "queued": "В очереди",
                "summarizing": "Подготовка summary",
                "map": "Map этап",
                "reduce": "Reduce этап",
                "summary_ready": "Summary готов",
                "done": "Завершено",
                "error": "Ошибка",
            }
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
                        text=f"Прогресс суммаризации логов: {int(processed_num)}/{int(total_num)}",
                    )
                else:
                    st.caption(f"Прогресс суммаризации логов: обработано {int(processed_num)} строк")
            events = state.get("events", [])
            if isinstance(events, list) and events:
                for line in events[-8:]:
                    st.caption(str(line))

        batches = state.get("map_batches", [])
        if batches:
            status = str(state.get("status", ""))
            if status in {"map", "reduce", "summarizing"} and len(batches) > LIVE_BATCHES_TO_RENDER:
                shown_batches = batches[-LIVE_BATCHES_TO_RENDER:]
                st.caption(
                    f"Показаны последние {LIVE_BATCHES_TO_RENDER} map-batch из {len(batches)} "
                    "(полный результат сохранен в файл)."
                )
            elif len(batches) > FINAL_BATCHES_TO_RENDER:
                shown_batches = batches[-FINAL_BATCHES_TO_RENDER:]
                st.caption(
                    f"Показаны последние {FINAL_BATCHES_TO_RENDER} map-batch из {len(batches)} "
                    "(полный результат сохранен в файл)."
                )
            else:
                shown_batches = batches

            for batch in shown_batches:
                idx = int(batch.get("batch_index", 0)) + 1
                total = batch.get("batch_total", len(batches))
                batch_logs = batch.get("batch_logs", [])
                if not isinstance(batch_logs, list):
                    batch_logs = []
                batch_logs_count = batch.get("batch_logs_count")
                if batch_logs_count is None:
                    batch_logs_count = len(batch_logs)
                batch_period_start, batch_period_end = deps.infer_batch_period(batch)
                with st.chat_message("assistant"):
                    st.markdown(f"Map summary {idx}/{total}")
                    deps.render_pretty_summary_text(
                        batch.get("batch_summary", ""),
                        height=deps.summary_text_height,
                    )
                    st.caption(f"Логов в батче: {batch_logs_count}")
                    if batch_period_start and batch_period_end:
                        st.caption(f"Период логов батча: `{batch_period_start}` -> `{batch_period_end}`")
                    if batch_logs:
                        logs_df = pd.DataFrame(batch_logs)
                        st.dataframe(
                            logs_df,
                            use_container_width=True,
                            hide_index=True,
                            height=deps.logs_batch_table_height,
                        )
                    else:
                        st.caption("Логи батча не переданы.")

        if state.get("final_summary"):
            with st.chat_message("assistant"):
                st.markdown("Итоговый Reduce summary")
                deps.render_pretty_summary_text(
                    state["final_summary"],
                    height=deps.final_text_height,
                )

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
                st.error(f"Ошибка обработки: {state['error']}")

        result_json_path = state.get("result_json_path")
        result_summary_path = state.get("result_summary_path")
        live_events_path = state.get("live_events_path")
        live_batches_path = state.get("live_batches_path")
        if result_json_path or result_summary_path or live_events_path or live_batches_path:
            with st.chat_message("assistant"):
                st.markdown("Результаты сохранены в файлы")
                if result_json_path:
                    st.code(str(result_json_path))
                if result_summary_path:
                    st.code(str(result_summary_path))
                if live_events_path:
                    st.code(str(live_events_path))
                if live_batches_path:
                    st.code(str(live_batches_path))


def render_logs_summary_page(deps: LogsSummaryPageDeps) -> None:
    running_key = "logs_summary_running"
    pending_key = "logs_summary_pending_config"
    last_state_key = "logs_summary_last_state"

    if running_key not in st.session_state:
        st.session_state[running_key] = False
    if pending_key not in st.session_state:
        st.session_state[pending_key] = None
    if last_state_key not in st.session_state:
        st.session_state[last_state_key] = None

    running = bool(st.session_state.get(running_key))

    st.title("Logs Summarizer")
    st.caption(
        "Укажи SQL и период (±N минут или start/end); "
        "страница покажет map-reduce суммаризацию в live-режиме."
    )

    default_query = (deps.default_sql_query or "").strip() or (
        "SELECT timestamp, level, message, container, pod, node, cluster "
        "FROM logs_demo_service"
    )
    sql_query = st.text_area(
        "SQL запрос логов",
        value=default_query,
        height=max(int(deps.sql_textarea_height), 180),
        help=(
            "Запрос должен возвращать минимум колонку `timestamp`. "
            "Можно использовать плейсхолдеры: {period_start}, {period_end}, "
            "{start}, {end}, {limit}, {offset}. "
            "Если плейсхолдеров нет, пагинация и фильтр периода применяются автоматически."
        ),
        key="logs_summary_sql_query",
        disabled=running,
    )
    user_goal = st.text_area(
        "Дополнительный контекст для LLM (опционально)",
        value="",
        height=140,
        placeholder=(
            "Например: произошел инцидент, упала нода Kubernetes; "
            "нужно понять вероятную причину и что проверить в первую очередь."
        ),
        help=(
            "Этот текст будет добавлен к каждому LLM-промпту map/reduce и поможет "
            "сфокусировать summary под твою задачу."
        ),
        key="logs_summary_user_goal",
        disabled=running,
    )
    period_mode_label = st.radio(
        "Режим периода",
        options=("Окно вокруг даты (±N минут)", "Явный диапазон (start/end)"),
        index=0,
        horizontal=True,
        key="logs_summary_period_mode",
        disabled=running,
    )
    is_window_mode = period_mode_label.startswith("Окно вокруг")
    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    center_default = now_utc.isoformat().replace("+00:00", "Z")
    start_default = (now_utc - timedelta(minutes=max(int(deps.loopback_minutes), 1))).isoformat().replace(
        "+00:00",
        "Z",
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
            key="logs_summary_center_dt",
            disabled=running,
        )
        window_minutes = int(
            st.number_input(
                "Окно анализа (+- N минут)",
                min_value=1,
                max_value=60 * 24 * 30,
                value=max(int(deps.loopback_minutes), 1),
                step=1,
                key="logs_summary_window_minutes",
                disabled=running,
            )
        )
    else:
        start_dt_text = st.text_input(
            "Дата/время начала (ISO)",
            value=start_default,
            key="logs_summary_start_dt",
            disabled=running,
        )
        end_dt_text = st.text_input(
            "Дата/время конца (ISO)",
            value=end_default,
            key="logs_summary_end_dt",
            disabled=running,
        )

    st.markdown("Параметры батчинга")
    batch_size_ui = int(
        st.number_input(
            "Общий размер batch (и для БД, и для LLM)",
            min_value=10,
            max_value=5_000,
            value=max(int(deps.batch_size), 1),
            step=10,
            key="logs_summary_batch_size",
            disabled=running,
        )
    )

    demo_mode = st.toggle(
        "Демо режим (без запросов к БД)",
        value=bool(deps.test_mode),
        help=(
            "Если включено, логи берутся из синтетического генератора, "
            "а не из ClickHouse."
        ),
        key="logs_summary_demo_mode",
        disabled=running,
    )
    demo_logs_count = int(
        st.number_input(
            "Количество логов в демо режиме",
            min_value=100,
            max_value=50_000,
            value=max(int(deps.logs_tail_limit), 1000),
            step=100,
            key="logs_summary_demo_logs_count",
            disabled=(not demo_mode) or running,
        )
    )
    run_clicked = st.button(
        "Запустить Суммаризацию Логов",
        type="primary",
        use_container_width=False,
        disabled=running,
    )

    if running:
        st.info("Суммаризация выполняется. Параметры заблокированы до завершения процесса.")

    analysis_placeholder = st.empty()
    error_placeholder = st.empty()

    if run_clicked:
        st.session_state[pending_key] = {
            "sql_query": sql_query,
            "user_goal": user_goal,
            "period_mode": "window" if is_window_mode else "start_end",
            "center_dt_text": center_dt_text,
            "start_dt_text": start_dt_text,
            "end_dt_text": end_dt_text,
            "window_minutes": int(window_minutes),
            "demo_mode": bool(demo_mode),
            "demo_logs_count": int(demo_logs_count),
            "batch_size": int(batch_size_ui),
        }
        st.session_state[last_state_key] = None
        st.session_state[running_key] = True
        st.rerun()

    pending_cfg = st.session_state.get(pending_key)
    if not running or not isinstance(pending_cfg, dict):
        last_state = st.session_state.get(last_state_key)
        if isinstance(last_state, dict) and last_state:
            _render_logs_summary_chat(analysis_placeholder, last_state, deps)
        return

    sql_query = str(pending_cfg.get("sql_query", "")).strip()
    user_goal = str(pending_cfg.get("user_goal", "")).strip()
    period_mode = str(pending_cfg.get("period_mode", "window")).strip()
    center_dt_text = str(pending_cfg.get("center_dt_text", "")).strip()
    start_dt_text = str(pending_cfg.get("start_dt_text", "")).strip()
    end_dt_text = str(pending_cfg.get("end_dt_text", "")).strip()
    window_minutes = max(int(pending_cfg.get("window_minutes", max(int(deps.loopback_minutes), 1))), 1)
    demo_mode = bool(pending_cfg.get("demo_mode", False))
    demo_logs_count = max(int(pending_cfg.get("demo_logs_count", max(int(deps.logs_tail_limit), 1000))), 1)
    batch_size = max(int(pending_cfg.get("batch_size", deps.batch_size)), 1)
    page_limit = batch_size
    llm_chunk_rows = batch_size
    request_payload = {
        "sql_query": sql_query,
        "user_goal": user_goal,
        "period_mode": period_mode,
        "center_dt_text": center_dt_text,
        "start_dt_text": start_dt_text,
        "end_dt_text": end_dt_text,
        "window_minutes": window_minutes,
        "demo_mode": demo_mode,
        "demo_logs_count": demo_logs_count,
        "batch_size": batch_size,
    }
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = deps.output_dir / "logs_summary_live" / f"run_{run_stamp}"
    live_events_path = run_dir / "events.jsonl"
    live_batches_path = run_dir / "batches.jsonl"
    request_payload["live_events_path"] = str(live_events_path)
    request_payload["live_batches_path"] = str(live_batches_path)

    try:
        if not sql_query:
            raise ValueError("SQL запрос не должен быть пустым.")

        if period_mode == "window":
            parsed_center = pd.to_datetime(center_dt_text, utc=True, errors="coerce")
            if pd.isna(parsed_center):
                raise ValueError(
                    "Неверный формат даты/времени. Используй ISO, например: 2026-03-26T12:30:00Z"
                )
            center_dt = parsed_center.to_pydatetime()
            period_start_dt = center_dt - timedelta(minutes=window_minutes)
            period_end_dt = center_dt + timedelta(minutes=window_minutes)
        else:
            parsed_start = pd.to_datetime(start_dt_text, utc=True, errors="coerce")
            parsed_end = pd.to_datetime(end_dt_text, utc=True, errors="coerce")
            if pd.isna(parsed_start) or pd.isna(parsed_end):
                raise ValueError(
                    "Неверный формат start/end. Используй ISO, например: 2026-03-26T10:00:00Z"
                )
            period_start_dt = parsed_start.to_pydatetime()
            period_end_dt = parsed_end.to_pydatetime()
            if period_end_dt <= period_start_dt:
                raise ValueError("Дата конца должна быть больше даты начала.")

        period_start_iso = period_start_dt.isoformat()
        period_end_iso = period_end_dt.isoformat()

        state: Dict[str, Any] = {
            "status": "queued",
            "mode": "demo" if demo_mode else "db",
            "query": sql_query,
            "user_goal": user_goal,
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

        total_rows_estimate: Optional[int] = None
        demo_logs: List[Dict[str, Any]] = []
        uses_template = False
        uses_paging_template = False
        if demo_mode:
            demo_logs = _build_demo_logs_for_window(
                period_start_dt=period_start_dt,
                period_end_dt=period_end_dt,
                total_logs=demo_logs_count,
            )
            columns = list(demo_logs[0].keys()) if demo_logs else ["timestamp", "message"]
            total_rows_estimate = len(demo_logs)
        else:
            uses_template = _query_uses_any_placeholders(sql_query)
            uses_paging_template = _query_uses_paging_placeholders(sql_query)
            preview_limit = 1
            if uses_template:
                rendered_preview = _render_query_template(
                    query_template=sql_query,
                    period_start_iso=period_start_iso,
                    period_end_iso=period_end_iso,
                    limit=preview_limit,
                    offset=0,
                )
                if uses_paging_template:
                    preview_query = rendered_preview
                else:
                    preview_query = _wrap_with_limit_offset(
                        query=rendered_preview,
                        limit=preview_limit,
                        offset=0,
                    )
            else:
                preview_query = _build_manual_logs_window_query(
                    base_query=sql_query,
                    period_start_iso=period_start_iso,
                    period_end_iso=period_end_iso,
                    limit=preview_limit,
                    offset=0,
                )
            preview_df = deps.query_logs_df(preview_query)
            columns = list(preview_df.columns)
            if "timestamp" not in columns:
                raise ValueError("SQL должен возвращать колонку `timestamp`.")

            # Avoid expensive COUNT(*) over huge log tables; this can block UI for a long time.
            # Progress bar still works via processed rows counter.
            total_rows_estimate = None

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
                records = [dict(row) for row in rows]
            else:
                if _query_uses_any_placeholders(sql_query):
                    rendered_query = _render_query_template(
                        query_template=sql_query,
                        period_start_iso=period_start,
                        period_end_iso=period_end,
                        limit=limit,
                        offset=offset,
                    )
                    if uses_paging_template:
                        query = rendered_query
                    else:
                        query = _wrap_with_limit_offset(
                            query=rendered_query,
                            limit=limit,
                            offset=offset,
                        )
                else:
                    query = _build_manual_logs_window_query(
                        base_query=sql_query,
                        period_start_iso=period_start,
                        period_end_iso=period_end,
                        limit=limit,
                        offset=offset,
                    )
                df = deps.query_logs_df(query)
                if df.empty:
                    return []
                records = df.to_dict(orient="records")
            return [{col: row.get(col) for col in columns} for row in records]

        def _on_progress(event: str, payload: Dict[str, Any]) -> None:
            if not isinstance(payload, dict):
                payload = {}
            try:
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
            except Exception:  # noqa: BLE001
                deps.logger.exception("failed to append live event")
            events = state.setdefault("events", [])
            if event == "map_start":
                state["status"] = "map"
                state["map_batches"] = []
                events.append("Map этап запущен")
            elif event == "page_fetched":
                state["status"] = "map"
                page_index = payload.get("page_index")
                page_rows = payload.get("page_rows")
                events.append(
                    f"Получена страница #{page_index}: {page_rows} строк"
                )
            elif event == "map_batch_start":
                state["status"] = "map"
                batch_idx = int(payload.get("batch_index", 0)) + 1
                batch_total = payload.get("batch_total")
                logs_count = payload.get("batch_logs_count")
                if batch_total:
                    events.append(
                        f"LLM анализирует batch {batch_idx}/{batch_total} ({logs_count} строк)"
                    )
                else:
                    events.append(
                        f"LLM анализирует batch {batch_idx} ({logs_count} строк)"
                    )
            elif event == "map_batch":
                state["status"] = "map"
                try:
                    _append_jsonl(
                        live_batches_path,
                        {
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "event": "map_batch",
                            "batch_index": payload.get("batch_index"),
                            "batch_total": payload.get("batch_total"),
                            "batch_summary": payload.get("batch_summary"),
                            "batch_logs_count": payload.get("batch_logs_count"),
                            "batch_logs": payload.get("batch_logs", []),
                            "batch_period_start": payload.get("batch_period_start"),
                            "batch_period_end": payload.get("batch_period_end"),
                            "rows_processed": payload.get("rows_processed"),
                            "rows_total": payload.get("rows_total"),
                        },
                    )
                except Exception:  # noqa: BLE001
                    deps.logger.exception("failed to append map batch")
                batch_logs = payload.get("batch_logs", [])
                if not isinstance(batch_logs, list):
                    batch_logs = []
                if len(batch_logs) > MAX_BATCH_LOG_ROWS_PREVIEW:
                    batch_logs = batch_logs[:MAX_BATCH_LOG_ROWS_PREVIEW]
                map_batches = state.setdefault("map_batches", [])
                map_batches.append(
                    {
                        "batch_index": payload.get("batch_index"),
                        "batch_total": payload.get("batch_total"),
                        "batch_summary": payload.get("batch_summary"),
                        "batch_logs_count": payload.get("batch_logs_count"),
                        "batch_logs": batch_logs,
                        "batch_period_start": payload.get("batch_period_start"),
                        "batch_period_end": payload.get("batch_period_end"),
                    }
                )
                if len(map_batches) > FINAL_BATCHES_TO_RENDER:
                    for old_batch in map_batches[:-FINAL_BATCHES_TO_RENDER]:
                        old_batch["batch_logs"] = []
                batch_idx = int(payload.get("batch_index", 0)) + 1
                batch_total = payload.get("batch_total")
                if batch_total:
                    events.append(
                        f"Map summary {batch_idx}/{batch_total} готов"
                    )
                else:
                    events.append(
                        f"Map summary {batch_idx} готов"
                    )
            elif event in ("map_done", "reduce_start"):
                state["status"] = "reduce"
                if event == "map_done":
                    events.append("Map этап завершен")
                else:
                    events.append("Reduce этап запущен")
            elif event == "reduce_done":
                state["status"] = "summary_ready"
                if payload.get("summary") is not None:
                    state["final_summary"] = str(payload.get("summary"))
                events.append("Reduce этап завершен")
            elif event == "reduce_group_start":
                state["status"] = "reduce"
                round_idx = payload.get("reduce_round")
                group_idx = int(payload.get("group_index", 0)) + 1
                group_total = payload.get("group_total")
                events.append(
                    f"Reduce round {round_idx}: группа {group_idx}/{group_total} в работе"
                )
            elif event == "reduce_group_done":
                state["status"] = "reduce"
                round_idx = payload.get("reduce_round")
                group_idx = int(payload.get("group_index", 0)) + 1
                group_total = payload.get("group_total")
                events.append(
                    f"Reduce round {round_idx}: группа {group_idx}/{group_total} готова"
                )

            if len(events) > MAX_EVENT_LINES:
                state["events"] = events[-MAX_EVENT_LINES:]

            progress_rows = payload.get("rows_processed")
            if progress_rows is None:
                progress_rows = payload.get("rows_fetched")
            if progress_rows is not None:
                state["logs_processed"] = progress_rows
            if payload.get("rows_total") is not None:
                state["logs_total"] = payload.get("rows_total")
            _render_logs_summary_chat(analysis_placeholder, state, deps)
            # Gives Streamlit a tiny chance to flush UI deltas progressively.
            time.sleep(0.01)

        state["status"] = "summarizing"
        _render_logs_summary_chat(analysis_placeholder, state, deps)

        goal_text = user_goal.strip()
        base_llm_call = deps.make_llm_call()

        if goal_text:
            def _llm_call_with_goal(prompt: str) -> str:
                contextual_prompt = (
                    "Контекст пользователя для этого summary:\n"
                    f"{goal_text}\n\n"
                    "Учитывай этот контекст в приоритизации проблем и рекомендаций.\n\n"
                    f"{prompt}"
                )
                return base_llm_call(contextual_prompt)

            llm_call = _llm_call_with_goal
        else:
            llm_call = base_llm_call

        summarizer = deps.period_log_summarizer_cls(
            db_fetch_page=_db_fetch_page,
            llm_call=llm_call,
            config=deps.summarizer_config_cls(
                page_limit=page_limit,
                llm_chunk_rows=llm_chunk_rows,
                keep_map_batches_in_memory=False,
                keep_map_summaries_in_result=False,
            ),
            on_progress=_on_progress,
        )
        result = summarizer.summarize_period(
            period_start=period_start_iso,
            period_end=period_end_iso,
            columns=columns,
            total_rows_estimate=total_rows_estimate,
        )
        state["status"] = "done"
        state["final_summary"] = str(result.summary)
        state["stats"] = {
            "pages_fetched": result.pages_fetched,
            "rows_processed": result.rows_processed,
            "llm_calls": result.llm_calls,
            "reduce_rounds": result.reduce_rounds,
        }
        if state.get("logs_total") is None:
            state["logs_total"] = total_rows_estimate
        state["logs_processed"] = result.rows_processed
    except Exception as exc:  # noqa: BLE001
        state = {
            "status": "error",
            "mode": "demo" if demo_mode else "db",
            "query": sql_query,
            "user_goal": user_goal,
            "period_mode": period_mode,
            "period_start": None,
            "period_end": None,
            "window_minutes": window_minutes,
            "logs_processed": 0,
            "logs_total": None,
            "map_batches": [],
            "final_summary": None,
            "stats": None,
            "error": str(exc),
            "result_json_path": None,
            "result_summary_path": None,
            "live_events_path": str(live_events_path),
            "live_batches_path": str(live_batches_path),
        }
        error_placeholder.error(str(exc))
        deps.logger.exception("manual logs summary failed")

    try:
        saved_paths = _save_logs_summary_result(
            output_dir=deps.output_dir,
            request_payload=request_payload,
            result_state=state,
        )
        state["result_json_path"] = saved_paths.get("json_path")
        state["result_summary_path"] = saved_paths.get("summary_path")
    except Exception:  # noqa: BLE001
        deps.logger.exception("failed to save logs summary result artifacts")

    _render_logs_summary_chat(analysis_placeholder, state, deps)
    st.session_state[last_state_key] = state
    st.session_state[pending_key] = None
    st.session_state[running_key] = False
    st.rerun()
