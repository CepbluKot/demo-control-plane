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
LAST_STATE_SESSION_KEY = "logs_summary_last_result_state"
RUNNING_SESSION_KEY = "logs_summary_running"
PENDING_RUN_SESSION_KEY = "logs_summary_pending_run"
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


def _estimate_eta(state: Dict[str, Any], event: str, payload: Dict[str, Any]) -> None:
    started_mono = state.get("started_monotonic")
    if started_mono is None:
        return

    status = str(state.get("status", "queued"))
    now_mono = time.monotonic()
    elapsed = max(now_mono - float(started_mono), 0.001)

    rows_processed = pd.to_numeric(state.get("logs_processed"), errors="coerce")
    rows_total = pd.to_numeric(state.get("logs_total"), errors="coerce")
    ratio: Optional[float] = None

    if not pd.isna(rows_processed) and not pd.isna(rows_total) and float(rows_total) > 0:
        ratio = float(rows_processed) / float(rows_total)
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


def _normalize_sql_query_text(query: str) -> str:
    normalized = str(query).replace("\r\n", "\n").replace("\r", "\n").strip()
    while normalized.endswith(";"):
        normalized = normalized[:-1].rstrip()
    return normalized


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
        "{PERIOD_START}": safe_start,
        "{PERIOD_END}": safe_end,
        "{START}": safe_start,
        "{END}": safe_end,
        "{START_ISO}": safe_start,
        "{END_ISO}": safe_end,
        "{LIMIT}": str(safe_limit),
        "{OFFSET}": str(safe_offset),
        "{PAGE_LIMIT}": str(safe_limit),
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
    base = _normalize_sql_query_text(base_query)
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
) -> str:
    goal_block = user_goal.strip() or "Не указан"
    return (
        "Сделай финальный отчет по расследованию инцидента в СВОБОДНОМ формате (без фиксированных секций), "
        "в том виде, который ты считаешь максимально понятным для SRE-команды.\n\n"
        "Требования к содержанию:\n"
        "- Что произошло и когда (понятный таймлайн).\n"
        "- Что на что повлияло (причинно-следственные связи).\n"
        "- Наиболее вероятная первопричина и альтернативные гипотезы.\n"
        "- Подтверждающие сигналы из логов.\n"
        "- Приоритетный план действий (немедленно / далее).\n"
        "- Если есть неопределенность — явно укажи ее.\n\n"
        f"Период расследования: [{period_start}, {period_end})\n"
        f"Контекст пользователя: {goal_block}\n"
        f"Техническая статистика: {stats}\n\n"
        "Ниже структурированное summary, на основе которого нужно сделать улучшенную финальную версию:\n"
        f"{final_summary}"
    )


def _render_final_report(container, state: Dict[str, Any], deps: LogsSummaryPageDeps) -> None:
    status = str(state.get("status", ""))
    if status not in ("done", "error"):
        return

    with container.container():
        st.markdown("2. Итоговый Отчет")

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

        if not pd.isna(rows_total) and float(rows_total) > 0:
            pct = (rows_processed / float(rows_total)) * 100.0
            st.caption(f"Покрытие периода: {rows_processed}/{int(rows_total)} (~{pct:.1f}%)")
        elif rows_processed > 0:
            st.caption(f"Обработано логов: {rows_processed}")

        final_summary = str(state.get("final_summary") or "").strip()
        if final_summary:
            st.markdown("Итоговое расследование")
            deps.render_pretty_summary_text(final_summary, height=max(int(deps.final_text_height), 280))

        freeform_summary = str(state.get("freeform_final_summary") or "").strip()
        if freeform_summary:
            st.markdown("Итоговое расследование в свободном формате")
            deps.render_pretty_summary_text(
                freeform_summary,
                height=max(int(deps.final_text_height), 320),
            )

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
        "summarizing": "Подготовка summary",
        "map": "Map этап",
        "reduce": "Reduce этап",
        "summary_ready": "Summary готов",
        "done": "Завершено",
        "error": "Ошибка",
    }

    with container.container():
        st.markdown("1. Пошаговая Суммаризация Логов")
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
                        f"DB batch: `{state.get('db_batch_size')}`",
                        f"LLM batch: `{state.get('llm_batch_size')}`",
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
            eta_left = state.get("eta_seconds_left")
            eta_finish = state.get("eta_finish_at")
            if eta_left is not None and status not in ("done", "error"):
                if eta_finish:
                    try:
                        eta_finish_dt = pd.to_datetime(eta_finish, utc=True, errors="coerce")
                        if not pd.isna(eta_finish_dt):
                            finish_text = eta_finish_dt.strftime("%H:%M:%S UTC")
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


def _build_config(deps: LogsSummaryPageDeps, db_batch_size: int, llm_batch_size: int) -> Any:
    try:
        return deps.summarizer_config_cls(
            page_limit=db_batch_size,
            llm_chunk_rows=llm_batch_size,
            keep_map_batches_in_memory=False,
            keep_map_summaries_in_result=False,
        )
    except TypeError:
        return deps.summarizer_config_cls(
            page_limit=db_batch_size,
            llm_chunk_rows=llm_batch_size,
        )


def render_logs_summary_page(deps: LogsSummaryPageDeps) -> None:
    st.title("Logs Summarizer")
    if RUNNING_SESSION_KEY not in st.session_state:
        st.session_state[RUNNING_SESSION_KEY] = False
    is_running = bool(st.session_state.get(RUNNING_SESSION_KEY, False))

    default_query = (deps.default_sql_query or "").strip() or (
        "SELECT timestamp, level, message FROM logs_demo_service "
        "WHERE timestamp >= parseDateTimeBestEffort('{period_start}') "
        "AND timestamp < parseDateTimeBestEffort('{period_end}') "
        "ORDER BY timestamp DESC LIMIT {limit} OFFSET {offset}"
    )

    with st.sidebar:
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
            "Контекст по алертам/инциденту для LLM (опционально)",
            value="",
            height=140,
            disabled=is_running,
        )

        period_mode_label = st.radio(
            "Режим периода",
            options=("Окно вокруг даты (±N минут)", "Явный диапазон (start/end)"),
            index=0,
            horizontal=False,
            disabled=is_running,
        )
        is_window_mode = period_mode_label.startswith("Окно вокруг")

        now_utc = datetime.now(timezone.utc).replace(microsecond=0)
        center_default = now_utc.isoformat().replace("+00:00", "Z")
        start_default = (
            now_utc - timedelta(minutes=max(int(deps.loopback_minutes), 1))
        ).isoformat().replace("+00:00", "Z")
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

        db_batch_size = int(
            st.number_input(
                "Размер DB batch (выгрузка из БД)",
                min_value=10,
                max_value=100_000,
                value=max(int(deps.db_batch_size), 1),
                step=100,
                disabled=is_running,
            )
        )
        llm_batch_size = int(
            st.number_input(
                "Размер LLM batch (строк в один MAP prompt)",
                min_value=10,
                max_value=10_000,
                value=max(int(deps.llm_batch_size), 1),
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

    runtime_error_placeholder = st.empty()
    analysis_placeholder = st.empty()
    final_report_placeholder = st.empty()
    if run_clicked and not is_running:
        st.session_state[PENDING_RUN_SESSION_KEY] = {
            "sql_query": sql_query,
            "user_goal": user_goal,
            "period_mode": "window" if is_window_mode else "start_end",
            "center_dt_text": center_dt_text,
            "window_minutes": window_minutes,
            "start_dt_text": start_dt_text,
            "end_dt_text": end_dt_text,
            "db_batch_size": db_batch_size,
            "llm_batch_size": llm_batch_size,
            "demo_mode": demo_mode,
            "demo_logs_count": demo_logs_count,
        }
        st.session_state[RUNNING_SESSION_KEY] = True
        st.rerun()

    pending_run = st.session_state.get(PENDING_RUN_SESSION_KEY)
    if not isinstance(pending_run, dict):
        last_state = st.session_state.get(LAST_STATE_SESSION_KEY)
        if isinstance(last_state, dict) and last_state:
            _render_logs_summary_chat(analysis_placeholder, last_state, deps)
            _render_final_report(final_report_placeholder, last_state, deps)
        return

    sql_query = str(pending_run.get("sql_query", ""))
    user_goal = str(pending_run.get("user_goal", ""))
    period_mode = str(pending_run.get("period_mode", "window"))
    center_dt_text = str(pending_run.get("center_dt_text", ""))
    window_minutes = int(pending_run.get("window_minutes", deps.loopback_minutes))
    start_dt_text = str(pending_run.get("start_dt_text", ""))
    end_dt_text = str(pending_run.get("end_dt_text", ""))
    db_batch_size = int(pending_run.get("db_batch_size", deps.db_batch_size))
    llm_batch_size = int(pending_run.get("llm_batch_size", deps.llm_batch_size))
    demo_mode = bool(pending_run.get("demo_mode", deps.test_mode))
    demo_logs_count = int(pending_run.get("demo_logs_count", deps.logs_tail_limit))

    def _unlock_controls() -> None:
        st.session_state[RUNNING_SESSION_KEY] = False
        st.session_state.pop(PENDING_RUN_SESSION_KEY, None)

    if not sql_query.strip():
        st.error("SQL запрос не должен быть пустым.")
        _unlock_controls()
        return

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
        _unlock_controls()
        return

    period_start_iso = period_start_dt.isoformat()
    period_end_iso = period_end_dt.isoformat()
    sql_query_clean = _normalize_sql_query_text(sql_query)
    if not sql_query_clean:
        st.error("SQL запрос не должен быть пустым.")
        _unlock_controls()
        return

    uses_template = _query_uses_any_placeholders(sql_query_clean)
    uses_paging_template = _query_uses_paging_placeholders(sql_query_clean)
    preview_available_columns: Optional[List[str]] = None

    # Fail fast: validate that formatted/multiline SQL compiles and runs in DB mode.
    if not demo_mode:
        try:
            if uses_template:
                rendered_preview = _render_query_template(
                    query_template=sql_query_clean,
                    period_start_iso=period_start_iso,
                    period_end_iso=period_end_iso,
                    limit=1,
                    offset=0,
                )
                preview_query = (
                    rendered_preview
                    if uses_paging_template
                    else _wrap_with_limit_offset(query=rendered_preview, limit=1, offset=0)
                )
            else:
                preview_query = _build_window_query_for_plain_sql(
                    base_query=sql_query_clean,
                    period_start_iso=period_start_iso,
                    period_end_iso=period_end_iso,
                    limit=1,
                    offset=0,
                )
            preview_df = deps.query_logs_df(preview_query)
            preview_available_columns = [str(col) for col in preview_df.columns]
        except Exception as exc:  # noqa: BLE001
            st.error(f"SQL не прошел предварительную проверку: {exc}")
            _unlock_controls()
            return

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
        "db_batch_size": db_batch_size,
        "llm_batch_size": llm_batch_size,
        "logs_processed": 0,
        "logs_total": None,
        "events": [],
        "map_batches": [],
        "final_summary": None,
        "freeform_final_summary": None,
        "stats": None,
        "error": None,
        "result_json_path": None,
        "result_summary_path": None,
        "live_events_path": str(live_events_path),
        "live_batches_path": str(live_batches_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "started_monotonic": time.monotonic(),
        "eta_seconds_left": None,
        "eta_finish_at": None,
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

            _estimate_eta(state, event, payload)
            _render_logs_summary_chat(analysis_placeholder, state, deps)
            time.sleep(0.01)

        state["status"] = "summarizing"
        _render_logs_summary_chat(analysis_placeholder, state, deps)

        base_llm_call = deps.make_llm_call()
        goal_text = user_goal.strip()
        if goal_text:

            def _llm_call_with_goal(prompt: str) -> str:
                enriched_prompt = (
                    "Контекст по алертам/инциденту от пользователя:\n"
                    f"{goal_text}\n\n"
                    "Сфокусируйся на причинно-следственном разборе: почему алерты сработали, "
                    "какие события этому предшествовали и что стало триггером.\n\n"
                    f"{prompt}"
                )
                return base_llm_call(enriched_prompt)

            llm_call = _llm_call_with_goal
        else:
            llm_call = base_llm_call

        summarizer = deps.period_log_summarizer_cls(
            db_fetch_page=_db_fetch_page,
            llm_call=llm_call,
            config=_build_config(deps, db_batch_size, llm_batch_size),
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
    _unlock_controls()
    _render_logs_summary_chat(analysis_placeholder, state, deps)
    _render_final_report(final_report_placeholder, state, deps)
