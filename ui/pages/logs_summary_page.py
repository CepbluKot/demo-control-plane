from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import io

MSK = timezone(timedelta(hours=3))
from collections import deque
import heapq
import html
import json
import logging
import math
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4
import zipfile

import pandas as pd
import requests
import streamlit as st

from settings import settings


MAX_EVENT_LINES = 160
MAX_RENDERED_BATCHES = 10
MAX_LOG_ROWS_PREVIEW = 80
MAX_METRICS_ROWS_TOTAL = 50_000  # cap across all metric queries to avoid OOM
MAX_LLM_TIMELINE_ROWS = 400
LAST_STATE_SESSION_KEY = "logs_summary_last_result_state"
RUNNING_SESSION_KEY = "logs_summary_running"
RUN_PARAMS_SESSION_KEY = "logs_summary_run_params"
FORM_ERROR_SESSION_KEY = "logs_summary_form_error"
RESUME_SELECTED_SESSION_KEY = "logs_summary_resume_selected"
RESUME_BANNER_DISMISSED_SESSION_KEY = "logs_summary_resume_banner_dismissed_session"
PENDING_PREFILL_SESSION_KEY = "logs_summary_pending_prefill_params"
PORTABLE_BUNDLE_TYPE = "logs_summary_portable_report"
PORTABLE_BUNDLE_VERSION = 1
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
REPORT_CONTEXT_SECTION_TITLE = "1. Контекст Инцидента Из UI (Дословно)"
REPORT_METRICS_SECTION_TITLE = "7. Аномалии Метрик И Корреляции С Логами"
ALERT_STATUS_PRIORITY: Dict[str, int] = {
    "EXPLAINED": 4,
    "PARTIALLY": 3,
    "NOT_EXPLAINED": 2,
    "NOT_SEEN_IN_BATCH": 1,
}
ALERT_STATUS_VIEW: Dict[str, Dict[str, str]] = {
    "EXPLAINED": {"icon": "●", "color": "#15803d", "label": "EXPLAINED"},
    "PARTIALLY": {"icon": "◐", "color": "#a16207", "label": "PARTIALLY"},
    "NOT_EXPLAINED": {"icon": "✕", "color": "#b91c1c", "label": "NOT_EXPLAINED"},
    "NOT_SEEN_IN_BATCH": {"icon": "○", "color": "#6b7280", "label": "NOT_SEEN_IN_BATCH"},
}
STEP_STATUS_STYLE: Dict[str, Dict[str, str]] = {
    "done": {"icon": "✓", "color": "#15803d"},
    "active": {"icon": "●", "color": "#2563eb"},
    "future": {"icon": "○", "color": "#9ca3af"},
    "error": {"icon": "✕", "color": "#b91c1c"},
}
MODEL_CONTEXT_PRESETS: Dict[str, int] = {
    "claude-sonnet-4-20250514": 200_000,
    "gpt-4.1": 128_000,
    "gpt-4.1-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "PNX.QWEN3 235b a22b instruct": 200_000,
}
FINAL_REPORT_SECTIONS: tuple[tuple[str, str], ...] = (
    (
        REPORT_CONTEXT_SECTION_TITLE,
        "Вставь исходный контекст инцидента/алертов из UI дословно, без перефразирования и без интерпретаций.",
    ),
    (
        "2. Резюме Инцидента",
        "Дай 3-5 предложений: что произошло, когда, какие сервисы затронуты, наиболее вероятная первопричина, текущий статус.",
    ),
    (
        "3. Покрытие Данных",
        "Укажи период анализа, источники/SQL, покрытые сервисы, объёмы данных и что не покрыто.",
    ),
    (
        "4. Полная Хронология Событий",
        "Построй детальную timeline с точными timestamp до микросекунд и timezone. "
        "Для каждого события укажи источник, severity и маркировку [ФАКТ]/[ГИПОТЕЗА], для [ФАКТ] добавь цитату из лога.",
    ),
    (
        "5. Причинно-Следственные Цепочки",
        "Покажи цепочки что к чему привело, со стрелками причины -> следствия и явным механизмом связи.",
    ),
    (
        "6. Связь С Каждым Инцидентом/Алертом Из UI",
        "Для каждого пункта из UI дай статус [ОБЪЯСНЁН]/[ЧАСТИЧНО ОБЪЯСНЁН]/[НЕ ОБЪЯСНЁН] с доказательствами.",
    ),
    (
        REPORT_METRICS_SECTION_TITLE,
        "Если метрики есть: опиши аномалии и корреляции с логами. "
        "Если метрик нет: явно укажи это и какие метрики нужны для усиления анализа.",
    ),
    (
        "8. Гипотезы Первопричин",
        "По каждому инциденту/алерту из UI дай 2-5 гипотез с confidence, подтверждающими и противоречащими событиями.",
    ),
    (
        "9. Конфликтующие Версии",
        "Покажи конфликтующие интерпретации с аргументами сторон. Если конфликтов нет — так и напиши.",
    ),
    (
        "10. Разрывы В Цепочках",
        "Явно перечисли где цепочки рвутся и какие данные нужны для закрытия каждого разрыва.",
    ),
    (
        "11. Масштаб И Влияние",
        "Опиши затронутые сервисы/компоненты, пользовательские сценарии, количественные показатели и длительность инцидента.",
    ),
    (
        "12. Рекомендации Для SRE",
        "Дай конкретные действия с приоритетами P0/P1/P2, обоснованием и ожидаемым эффектом.",
    ),
    (
        "13. Уровень Уверенности И Ограничения Анализа",
        "Дай честную оценку уверенности, перечисли ограничения анализа и зоны низкой достоверности.",
    ),
)


def _build_models_url(api_base: str) -> str:
    base = str(api_base or "").strip().rstrip("/")
    if not base:
        return ""
    if base.endswith("/models"):
        return base
    if base.endswith("/chat/completions"):
        return f"{base[: -len('/chat/completions')]}/models"
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


def _extract_model_ids_from_payload(payload: Any) -> List[str]:
    rows: List[Any] = []
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        for key in ("data", "models", "result"):
            value = payload.get(key)
            if isinstance(value, list):
                rows = value
                break

    out: List[str] = []
    seen: set[str] = set()
    for row in rows:
        model_id = ""
        if isinstance(row, str):
            model_id = row.strip()
        elif isinstance(row, dict):
            for key in ("id", "model", "name"):
                value = row.get(key)
                if value is None:
                    continue
                candidate = str(value).strip()
                if candidate:
                    model_id = candidate
                    break
        if model_id and model_id not in seen:
            seen.add(model_id)
            out.append(model_id)
    return out


def _fetch_llm_model_candidates(timeout_seconds: float = 5.0) -> tuple[List[str], str]:
    api_base = str(getattr(settings, "OPENAI_API_BASE_DB", "") or "").strip()
    if not api_base:
        return [], "OPENAI_API_BASE_DB не задан."
    url = _build_models_url(api_base)
    if not url:
        return [], "Некорректный OPENAI_API_BASE_DB."
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    api_key = str(getattr(settings, "OPENAI_API_KEY_DB", "") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = requests.get(url, headers=headers, timeout=float(timeout_seconds))
        response.raise_for_status()
        model_ids = _extract_model_ids_from_payload(response.json())
        if not model_ids:
            return [], "API вернула пустой список моделей."
        return model_ids, ""
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"


def _normalize_timestamp_column_name(value: Any, *, default: str = "timestamp") -> str:
    raw = str(value or "").strip()
    return raw or default


def _normalize_summary_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"none", "null", "nan"}:
        return ""
    lowered = text.lower()
    if lowered.startswith("[llm недоступна") or "эвристический fallback" in lowered:
        error_line = ""
        for line in text.splitlines():
            if line.strip().lower().startswith("ошибка:"):
                error_line = line.strip()
                break
        if error_line:
            return (
                "[LLM ERROR]\n\n"
                f"{error_line}\n\n"
                "Summary по этому шагу не получен из LLM."
            )
        return "[LLM ERROR]\n\nSummary по этому шагу не получен из LLM."
    return text


def _ensure_report_topics_present(
    summary_text: str,
    *,
    topic_titles: List[str],
    preferred_sections: Optional[List[Dict[str, Any]]] = None,
) -> tuple[str, List[str]]:
    normalized = _normalize_summary_text(summary_text)
    preferred_by_title: Dict[str, str] = {}
    for item in preferred_sections or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        body = _normalize_summary_text(item.get("text"))
        if title and body:
            preferred_by_title[title.lower()] = body

    if not topic_titles:
        return normalized, []

    lowered = normalized.lower()
    missing: List[str] = []
    additions: List[str] = []
    for title in topic_titles:
        topic = str(title or "").strip()
        if not topic:
            continue
        if topic.lower() in lowered:
            continue
        missing.append(topic)
        body = preferred_by_title.get(topic.lower()) or "Данных недостаточно для уверенного вывода по этому разделу."
        additions.extend([f"## {topic}", "", body, ""])

    if not missing:
        return normalized, []

    combined_parts = [normalized] if normalized else []
    combined_parts.append("\n".join(additions).strip())
    combined = "\n\n".join(part for part in combined_parts if part).strip()
    return combined, missing


def _incident_verbatim_requirement_block(goal_text: str) -> str:
    raw_goal = str(goal_text or "").strip() or "Контекст инцидента в UI не задан."
    return (
        "ОБЯЗАТЕЛЬНО ПЕРЕД ГИПОТЕЗАМИ:\n"
        "Сначала выведи отдельный подраздел `ИНЦИДЕНТ ИЗ UI (ДОСЛОВНО)` и вставь текст ниже БЕЗ перефразирования.\n"
        "Сохрани название инцидента, формулировки, ноды, IP, статусы, время и орфографию как в исходном вводе.\n"
        "Только после этого пиши гипотезы первопричин.\n"
        f"Текст инцидента из UI:\n{raw_goal}"
    )


def _programmatic_section_text(
    *,
    section_title: str,
    user_goal: str,
    metrics_context: str,
) -> Optional[str]:
    normalized_title = str(section_title or "").strip()
    if normalized_title == REPORT_CONTEXT_SECTION_TITLE:
        raw_goal = str(user_goal or "").strip()
        if not raw_goal:
            raw_goal = "Контекст инцидента в UI не задан."
        return (
            "Ниже — исходный текст инцидента из UI (дословно, без изменений):\n\n"
            f"{raw_goal}"
        )
    if normalized_title == REPORT_METRICS_SECTION_TITLE and not str(metrics_context or "").strip():
        return (
            "Метрики не предоставлены. Для более полного анализа рекомендуется повторить запуск "
            "с метриками CPU, memory, latency, error rate и saturation по затронутым сервисам."
        )
    return None


def _summary_origin_label(value: Any) -> str:
    raw = str(value or "").strip()
    mapping = {
        "reduce_done_event": "Прямой REDUCE (событие прогресса)",
        "cross_reduce_llm": "Кросс-источниковый REDUCE (LLM)",
        "cross_reduce_fallback_join": "Fallback: склейка source-summary после сбоя cross-reduce",
        "per_source_reduce_direct": "Итог по единственному source-summary",
        "recovered_from_map_reduce": "Восстановлено: REDUCE из сохранённых MAP summary",
        "recovered_from_map_join": "Восстановлено: склейка сохранённых MAP summary",
        "single_reduce_direct": "Прямой REDUCE (single-source)",
        "single_recovered_from_map_reduce": "Восстановлено (single-source): REDUCE из MAP summary",
        "single_recovered_from_map_join": "Восстановлено (single-source): склейка MAP summary",
        "no_logs_hypothesis": "Режим без логов: гипотезы LLM",
        "resume_rereduce": "Resume: пересборка REDUCE из сохранённых MAP summary",
        "manual_rereduce": "Ручная пересборка REDUCE из сохранённых MAP summary",
        "final_recovery_reduce": "Финальное восстановление: REDUCE из MAP summary",
        "final_recovery_join": "Финальное восстановление: склейка MAP summary",
        "no_logs": "Логи не найдены",
    }
    return mapping.get(raw, raw)


def _md_fence_for_text(text: str) -> str:
    content = str(text or "")
    runs = re.findall(r"`+", content)
    longest = max((len(run) for run in runs), default=2)
    # At least triple backticks and strictly longer than any sequence inside content.
    return "`" * max(3, longest + 1)


def _looks_like_section_heading(line: str) -> bool:
    raw = str(line or "").strip()
    if not raw:
        return False
    if re.match(r"^#{1,6}\s+\S+", raw):
        return True
    if re.match(r"^\d+\s*[\)\.\-:]\s+\S+", raw):
        return True
    if raw.startswith(("-", "*", "•")):
        return False
    plain = re.sub(r"^[>\s]+", "", raw).strip()
    if plain.endswith(":") and len(plain) <= 120:
        return True
    if plain.isupper() and len(plain) <= 120 and len(plain.split()) <= 14:
        return True
    return False


def _extract_root_cause_hypotheses_block(value: Any) -> str:
    text = _normalize_summary_text(value)
    if not text:
        return ""
    lines = text.replace("\r\n", "\n").split("\n")
    markers = (
        "ПРЕДПОЛОЖЕНИЯ О ПЕРВОПРИЧИНЕ",
        "ГИПОТЕЗЫ ПЕРВОПРИЧИНЫ",
        "ПЕРВОПРИЧИНЫ ПО ЦЕПОЧКАМ",
        "ROOT_CAUSE_HYPOTHESES",
    )
    start_idx: Optional[int] = None
    for idx, line in enumerate(lines):
        upper = str(line).upper()
        if any(marker in upper for marker in markers):
            start_idx = idx
            break

    if start_idx is None:
        fallback = [
            line
            for line in lines
            if ("ГИПОТЕЗ" in str(line).upper()) and ("ПЕРВОПРИЧ" in str(line).upper())
        ]
        return "\n".join(fallback).strip()

    end_idx = len(lines)
    for idx in range(start_idx + 1, len(lines)):
        candidate = str(lines[idx] or "").strip()
        if not candidate:
            continue
        upper = candidate.upper()
        if any(marker in upper for marker in markers):
            continue
        if _looks_like_section_heading(candidate):
            end_idx = idx
            break
    return "\n".join(lines[start_idx:end_idx]).strip()


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _render_prompt_template(template: str, values: Dict[str, Any]) -> str:
    rendered = str(template)
    rendered = re.sub(
        r"\{\{#each\s+source_summaries\}\}[\s\S]*?\{\{\/each\}\}",
        "{source_summaries_text}",
        rendered,
        flags=re.IGNORECASE,
    )
    rendered = re.sub(
        r"\{\{#each\s+map_summaries\}\}[\s\S]*?\{\{\/each\}\}",
        "{summaries_text}",
        rendered,
        flags=re.IGNORECASE,
    )

    def _replace_var(match: re.Match[str]) -> str:
        key = str(match.group(1) or "").strip()
        if key.startswith("this."):
            key = key.split(".", 1)[1]
        return "{" + key + "}"

    rendered = re.sub(
        r"\{\{\s*([a-zA-Z0-9_\.]+)\s*\}\}",
        _replace_var,
        rendered,
    )
    safe_values = _SafeFormatDict({k: "" if v is None else str(v) for k, v in values.items()})
    return rendered.format_map(safe_values)


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
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return str(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(MSK)
    else:
        ts = ts.tz_convert(MSK)
    return ts.strftime("%Y-%m-%d %H:%M:%S.%f MSK")


def _to_msk_ts(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ts
    if ts.tzinfo is None:
        return ts.tz_localize(MSK)
    return ts.tz_convert(MSK)


def _parse_user_dt(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ts
    if ts.tzinfo is None:
        return ts.tz_localize(MSK)
    return ts.tz_convert(MSK)


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
        is_datetime_like = (
            col_name in timestamp_like
            or "timestamp" in col_name
            or "datetime" in col_name
            or col_name.endswith("_time")
            or col_name.endswith("_at")
        )
        if not is_datetime_like:
            continue
        def _fmt(value: Any) -> str:
            ts = pd.to_datetime(value, errors="coerce")
            if pd.isna(ts):
                return str(value)
            if ts.tzinfo is None:
                ts = ts.tz_localize(MSK)
            else:
                ts = ts.tz_convert(MSK)
            return ts.strftime("%Y-%m-%d %H:%M:%S.%f MSK")

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


def _resolve_elapsed_seconds(state: Dict[str, Any]) -> Optional[float]:
    elapsed = pd.to_numeric(state.get("elapsed_seconds"), errors="coerce")
    if not pd.isna(elapsed):
        return max(float(elapsed), 0.0)
    started_mono = state.get("started_monotonic")
    if started_mono is None:
        return None
    try:
        return max(time.monotonic() - float(started_mono), 0.0)
    except Exception:
        return None


def _enrich_stats_with_elapsed(state: Dict[str, Any]) -> None:
    stats = state.get("stats")
    if not isinstance(stats, dict):
        return
    elapsed = _resolve_elapsed_seconds(state)
    if elapsed is None:
        return
    stats["logs_processing_seconds"] = round(float(elapsed), 2)
    stats["logs_processing_human"] = _format_eta_seconds(float(elapsed))


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
    period_start_ts = _to_msk_ts(state.get("period_start"))
    period_end_ts = _to_msk_ts(state.get("period_end"))
    last_batch_ts = _to_msk_ts(state.get("last_batch_ts"))

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
        state["eta_finish_at"] = datetime.now(MSK).isoformat()
        return

    if remaining is not None and remaining >= 0:
        finish_dt = datetime.now(MSK) + timedelta(seconds=remaining)
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
    finish_dt = datetime.now(MSK) + timedelta(seconds=remaining)
    state["eta_seconds_left"] = int(remaining)
    state["eta_finish_at"] = finish_dt.isoformat()


def _build_stage1_progress_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    status = str(state.get("status", "queued")).strip().lower()
    if status not in {"queued", "map", "summarizing"}:
        return {"show": False, "ratio": 0.0, "label": "", "runtime_line": ""}

    def _as_int(value: Any) -> int:
        parsed = pd.to_numeric(value, errors="coerce")
        if pd.isna(parsed):
            return 0
        return int(parsed)

    ratio: Optional[float] = None
    label = ""

    logs_processed = _as_int(state.get("logs_processed"))
    logs_total = _as_int(state.get("logs_total"))
    if logs_total > 0:
        ratio = max(min(float(logs_processed) / float(logs_total), 1.0), 0.0)
        label = (
            f"Загрузка логов + MAP: {logs_processed:,}/{logs_total:,} строк "
            f"({ratio * 100:.1f}%)"
        )
    else:
        period_start_ts = _to_msk_ts(state.get("period_start"))
        period_end_ts = _to_msk_ts(state.get("period_end"))
        last_batch_ts = _to_msk_ts(state.get("last_batch_ts"))
        if (
            not pd.isna(period_start_ts)
            and not pd.isna(period_end_ts)
            and not pd.isna(last_batch_ts)
            and (period_end_ts - period_start_ts).total_seconds() > 0
        ):
            total_span = (period_end_ts - period_start_ts).total_seconds()
            done_span = max((last_batch_ts - period_start_ts).total_seconds(), 0.0)
            ratio = max(min(done_span / total_span, 1.0), 0.0)
            label = f"Загрузка логов + MAP: покрытие периода {ratio * 100:.1f}%"
        else:
            batch_done, batch_total = _resolve_map_batches_progress(state)
            if batch_total > 0:
                ratio = max(min(float(batch_done) / float(batch_total), 1.0), 0.0)
                label = f"Загрузка логов + MAP: батчи {batch_done}/{batch_total} ({ratio * 100:.1f}%)"
            else:
                ratio = 0.0
                label = "Загрузка логов + MAP: ожидание первых данных"

    elapsed = _resolve_elapsed_seconds(state)
    rate = None
    if elapsed is not None and elapsed > 0 and logs_processed > 0:
        rate = float(logs_processed) / float(elapsed)
    if rate is None:
        rate = pd.to_numeric(state.get("rows_per_second"), errors="coerce")
        if pd.isna(rate):
            rate = None
        else:
            rate = float(rate)

    runtime_parts: List[str] = []
    if elapsed is not None:
        runtime_parts.append(f"elapsed: {_format_eta_seconds(elapsed)}")
    if rate is not None and rate > 0:
        runtime_parts.append(f"rate: {rate:.1f} rows/s")

    eta_seconds = pd.to_numeric(state.get("eta_seconds_left"), errors="coerce")
    if not pd.isna(eta_seconds):
        runtime_parts.append(f"eta: {_format_eta_seconds(float(eta_seconds))}")

    eta_finish_at = state.get("eta_finish_at")
    if eta_finish_at:
        runtime_parts.append(f"finish: {_format_datetime_with_tz(eta_finish_at)}")

    runtime_line = " | ".join(runtime_parts)
    return {
        "show": True,
        "ratio": max(min(float(ratio or 0.0), 1.0), 0.0),
        "label": label,
        "runtime_line": runtime_line,
    }


def _build_report_generation_progress_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    total = max(_safe_int(state.get("report_progress_total"), 0), 0)
    done = max(_safe_int(state.get("report_progress_current"), 0), 0)
    label = str(state.get("report_progress_label") or "").strip()
    active = bool(state.get("report_progress_active", False))
    if total <= 0 and done <= 0 and not active:
        return {"show": False, "ratio": 0.0, "label": ""}
    if total <= 0:
        total = max(done, 1)
    done = min(max(done, 0), total)
    ratio = max(min(float(done) / float(total), 1.0), 0.0)
    if not label:
        label = f"Генерация итогового отчёта: {done}/{total}"
    else:
        label = f"{label} ({done}/{total})"
    return {
        "show": True,
        "ratio": ratio,
        "label": label,
    }


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
    logs_timestamp_column: str = "timestamp"
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
        timestamp_column: str = "timestamp",
    ) -> None:
        self._query_logs_df = query_logs_df
        self._register_query_error = register_query_error
        self._logger = logger
        self._timestamp_column = _normalize_timestamp_column_name(timestamp_column)
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
                timestamp_column=self._timestamp_column,
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

            sorted_df = _sort_df_by_timestamp(df, timestamp_column=self._timestamp_column)
            if not sorted_df.empty:
                # Advance keyset cursor to last timestamp in the fetched page
                if uses_keyset and self._timestamp_column in sorted_df.columns:
                    max_ts = sorted_df[self._timestamp_column].apply(_to_msk_ts).max()
                    if not pd.isna(max_ts):
                        cursor.last_ts = max_ts.isoformat()
                cursor.page_records = [dict(row) for row in sorted_df.to_dict(orient="records")]
                cursor.page_pos = 0
                return True

            if not cursor.has_more:
                cursor.exhausted = True
                return False

        self._register_query_error(
            f"Запрос #{cursor.query_index + 1} пропущен: слишком много страниц без валидного `{self._timestamp_column}`"
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
            ts = _to_msk_ts(row.get(self._timestamp_column))
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


def _new_alert_item(
    *,
    title: str = "",
    details: str = "",
    time_mode: str = "point",
    time_point: str = "",
    time_start: str = "",
    time_end: str = "",
    ) -> Dict[str, str]:
    return {
        "id": uuid4().hex,
        "title": str(title),
        "details": str(details),
        "time_mode": "range" if str(time_mode).strip().lower() == "range" else "point",
        "time_point": str(time_point),
        "time_start": str(time_start),
        "time_end": str(time_end),
    }


def _default_alert_time_values(
    *,
    center_dt_text: str,
    start_dt_text: str,
    end_dt_text: str,
    center_default: str,
    start_default: str,
    end_default: str,
) -> Dict[str, str]:
    return {
        "time_point": (str(center_dt_text or "").strip() or str(center_default or "").strip()),
        "time_start": (str(start_dt_text or "").strip() or str(start_default or "").strip()),
        "time_end": (str(end_dt_text or "").strip() or str(end_default or "").strip()),
    }


def _normalize_alert_items(
    raw_items: Any,
    *,
    min_items: int,
    legacy_user_goal: str = "",
) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if isinstance(raw_items, list):
        for item in raw_items:
            if isinstance(item, dict):
                items.append(
                    _new_alert_item(
                        title=str(item.get("title", "")),
                        details=str(item.get("details", "")),
                        time_mode=str(item.get("time_mode", "point")),
                        time_point=str(item.get("time_point", "")),
                        time_start=str(item.get("time_start", "")),
                        time_end=str(item.get("time_end", "")),
                    )
                )
                items[-1]["id"] = str(item.get("id") or uuid4().hex)
            elif isinstance(item, str):
                text = str(item).strip()
                if text:
                    items.append(_new_alert_item(title=f"alert_{len(items) + 1}", details=text))
    if not items and str(legacy_user_goal or "").strip():
        items.append(
            _new_alert_item(
                title="legacy_alert_context",
                details=str(legacy_user_goal).strip(),
            )
        )
    while len(items) < max(int(min_items), 0):
        items.append(_new_alert_item())
    return items


def _extract_alerts_from_items(raw_items: Any) -> List[Dict[str, str]]:
    if not isinstance(raw_items, list):
        return []
    alerts: List[Dict[str, str]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        details = str(item.get("details") or "").strip()
        time_mode = str(item.get("time_mode") or "point").strip().lower()
        if time_mode not in {"point", "range"}:
            time_mode = "point"
        time_point = str(item.get("time_point") or "").strip()
        time_start = str(item.get("time_start") or "").strip()
        time_end = str(item.get("time_end") or "").strip()
        if not any([title, details, time_point, time_start, time_end]):
            continue
        alerts.append(
            {
                "id": str(item.get("id") or uuid4().hex),
                "title": title,
                "details": details,
                "time_mode": time_mode,
                "time_point": time_point,
                "time_start": time_start,
                "time_end": time_end,
            }
        )
    return alerts


def _render_alerts_context(alerts: List[Dict[str, str]]) -> str:
    normalized_alerts = [item for item in alerts if isinstance(item, dict)]
    if not normalized_alerts:
        return ""
    lines = ["Структурированный список алертов/инцидентов из UI:"]
    for idx, alert in enumerate(normalized_alerts, start=1):
        title = str(alert.get("title") or "").strip() or f"alert_{idx}"
        details = str(alert.get("details") or "").strip()
        time_mode = str(alert.get("time_mode") or "point").strip().lower()
        time_point = str(alert.get("time_point") or "").strip()
        time_start = str(alert.get("time_start") or "").strip()
        time_end = str(alert.get("time_end") or "").strip()
        if time_mode == "range":
            time_line = (
                f"период: {time_start} -> {time_end}"
                if (time_start and time_end)
                else "период: не указан"
            )
        else:
            time_line = f"время: {time_point}" if time_point else "время: не указано"
        lines.append(f"{idx}. {title}")
        lines.append(f"   - {time_line}")
        if details:
            lines.append(f"   - описание: {details}")
    return "\n".join(lines).strip()


def _collect_alerts_and_goal(
    raw_items: Any,
    *,
    min_items: int,
    legacy_user_goal: str = "",
) -> tuple[List[Dict[str, str]], List[Dict[str, str]], str]:
    normalized_items = _normalize_alert_items(
        raw_items,
        min_items=min_items,
        legacy_user_goal=legacy_user_goal,
    )
    alerts = _extract_alerts_from_items(normalized_items)
    rendered_goal = _render_alerts_context(alerts) or str(legacy_user_goal or "").strip()
    return normalized_items, alerts, rendered_goal


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


def _validate_logs_merge_schema(
    previews: List[Dict[str, Any]],
    *,
    timestamp_column: str = "timestamp",
) -> List[str]:
    if not previews:
        return []

    ts_col = _normalize_timestamp_column_name(timestamp_column)
    ts_col_lc = ts_col.lower()
    errors: List[str] = []
    first_df = previews[0].get("df")
    if first_df is None or not isinstance(first_df, pd.DataFrame):
        return ["Не удалось валидировать логи: preview первого запроса отсутствует."]

    base_cols = _column_set(first_df)
    if ts_col_lc not in base_cols:
        errors.append(f"Логи: в первом SQL отсутствует обязательная колонка `{ts_col}`.")

    for item in previews:
        idx = int(item.get("idx", 0)) + 1
        df = item.get("df")
        if df is None or not isinstance(df, pd.DataFrame):
            errors.append(f"Логи: запрос #{idx} не вернул preview-таблицу.")
            continue
        cols = _column_set(df)
        if ts_col_lc not in cols:
            errors.append(f"Логи: запрос #{idx} не содержит колонку `{ts_col}`.")
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


def _sort_df_by_timestamp(df: pd.DataFrame, *, timestamp_column: str = "timestamp") -> pd.DataFrame:
    ts_col = _normalize_timestamp_column_name(timestamp_column)
    if df.empty or ts_col not in df.columns:
        return df
    out = df.copy()
    out["__cp_ts"] = out[ts_col].apply(_to_msk_ts)
    out = out.dropna(subset=["__cp_ts"]).sort_values("__cp_ts", kind="mergesort")
    return out.drop(columns=["__cp_ts"]).reset_index(drop=True)


def _normalize_metrics_df(chunk: pd.DataFrame, *, default_service: str) -> pd.DataFrame:
    if chunk is None or chunk.empty:
        return pd.DataFrame(columns=["timestamp", "value", "service"])
    if "timestamp" not in chunk.columns:
        return pd.DataFrame(columns=["timestamp", "value", "service"])

    out = chunk.copy()
    out["timestamp"] = out["timestamp"].apply(_to_msk_ts)
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


def _build_metrics_context(metrics_df: pd.DataFrame, *, max_services: int = 0) -> str:
    if metrics_df is None or metrics_df.empty:
        return ""
    if "timestamp" not in metrics_df.columns or "value" not in metrics_df.columns:
        return ""

    normalized = metrics_df.copy()
    normalized["timestamp"] = normalized["timestamp"].apply(_to_msk_ts)
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
        if max_services > 0 and idx >= max_services:
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
        f"LIMIT {safe_limit} OFFSET {safe_offset}"
    )


def _build_window_query_for_plain_sql(
    *,
    base_query: str,
    period_start_iso: str,
    period_end_iso: str,
    limit: int,
    offset: int,
    timestamp_column: str = "timestamp",
) -> str:
    safe_limit = max(int(limit), 1)
    safe_offset = max(int(offset), 0)
    ts_col = _normalize_timestamp_column_name(timestamp_column)
    base = _strip_trailing_limit_offset(_normalize_sql_query_text(base_query))
    start_escaped = _escape_sql_literal(period_start_iso)
    end_escaped = _escape_sql_literal(period_end_iso)
    # Only add ORDER BY to the inner subquery if the user's SQL doesn't already have one.
    # We intentionally avoid forced outer ORDER BY because custom queries may not
    # expose a `timestamp` column in the final SELECT (e.g. grouped queries with start_time).
    inner_order = "" if "order by" in base.lower() else f" ORDER BY {ts_col} ASC"
    return (
        "SELECT * FROM ("
        "SELECT * FROM ("
        f"{base}"
        ") AS cp_src "
        f"WHERE {ts_col} >= parseDateTimeBestEffort('{start_escaped}') "
        f"AND {ts_col} < parseDateTimeBestEffort('{end_escaped}')"
        f"{inner_order}"
        ") AS cp_window "
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
    timestamp_column: str = "timestamp",
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
        timestamp_column=timestamp_column,
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
    timestamp_column: str = "timestamp",
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
        ts_col = _normalize_timestamp_column_name(timestamp_column)
        base = (
            "SELECT * FROM ("
            f"{base_sql}"
            ") AS cp_src "
            f"WHERE {ts_col} >= parseDateTimeBestEffort('{start_escaped}') "
            f"AND {ts_col} < parseDateTimeBestEffort('{end_escaped}')"
        )

    return f"SELECT count() AS total_rows FROM ({base}) AS cp_count"


def _precount_rows(
    *,
    query_specs: List[Dict[str, Any]],
    query_logs_df: Callable[[str], pd.DataFrame],
    period_start_iso: str,
    period_end_iso: str,
    timestamp_column: str,
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
                timestamp_column=timestamp_column,
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
                "timestamp": ts.isoformat(),
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


def _write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text), encoding="utf-8")


def _safe_filename(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())
    normalized = normalized.strip("._")
    return normalized or "item"


def _load_map_summaries_from_jsonl(path: str) -> List[str]:
    p = Path(str(path or ""))
    if not p.exists() or not p.is_file():
        return []
    summaries: List[str] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            raw = str(line).strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            summary_text = _normalize_summary_text(payload.get("batch_summary"))
            if summary_text:
                summaries.append(summary_text)
    return summaries


def _load_map_summaries_from_jsonl_for_source(path: str, source_name: str) -> List[str]:
    p = Path(str(path or ""))
    if not p.exists() or not p.is_file():
        return []
    target = str(source_name or "").strip()
    summaries: List[str] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            raw = str(line).strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            row_source = str(payload.get("source_name") or "").strip()
            if target and row_source != target:
                continue
            summary_text = _normalize_summary_text(payload.get("batch_summary"))
            if summary_text:
                summaries.append(summary_text)
    return summaries


def _load_recent_batches_from_jsonl(
    path: str,
    *,
    max_items: int = MAX_RENDERED_BATCHES,
    max_logs_preview: int = MAX_LOG_ROWS_PREVIEW,
) -> List[Dict[str, Any]]:
    p = Path(str(path or ""))
    if not p.exists() or not p.is_file():
        return []
    recent: deque[Dict[str, Any]] = deque(maxlen=max(int(max_items), 1))
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                raw = str(line).strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except Exception:
                    continue
                if str(payload.get("event", "map_batch")) != "map_batch":
                    continue
                batch_logs = payload.get("batch_logs", [])
                if not isinstance(batch_logs, list):
                    batch_logs = []
                recent.append(
                    {
                        "batch_index": payload.get("batch_index"),
                        "batch_total": payload.get("batch_total"),
                        "batch_summary": payload.get("batch_summary"),
                        "batch_logs_count": payload.get("batch_logs_count"),
                        "batch_period_start": payload.get("batch_period_start"),
                        "batch_period_end": payload.get("batch_period_end"),
                        "batch_logs": batch_logs,
                    }
                )
    except Exception:
        return []
    return list(recent)


def _extract_last_batch_ts_from_run_dir(run_dir: Path) -> str:
    candidates = [
        run_dir / "summaries" / "map_summaries.jsonl",
        run_dir / "batches.jsonl",
    ]
    latest_ts: Optional[pd.Timestamp] = None
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    raw = str(line).strip()
                    if not raw:
                        continue
                    try:
                        payload = json.loads(raw)
                    except Exception:
                        continue
                    ts_value = payload.get("batch_period_end")
                    ts_parsed = _to_msk_ts(ts_value)
                    if pd.isna(ts_parsed):
                        continue
                    if latest_ts is None or ts_parsed > latest_ts:
                        latest_ts = ts_parsed
        except Exception:
            continue
    if latest_ts is None:
        return ""
    return latest_ts.to_pydatetime().isoformat()


def _form_values_from_saved_params(
    *,
    saved_params: Dict[str, Any],
    default_query: str,
) -> Dict[str, Any]:
    def _to_int(value: Any, default: int) -> int:
        parsed = pd.to_numeric(value, errors="coerce")
        if pd.isna(parsed):
            return int(default)
        return int(parsed)

    logs_queries = [str(q) for q in saved_params.get("logs_queries", []) if str(q).strip()]
    if not logs_queries:
        logs_queries = [default_query]
    metrics_queries = [
        str(q) for q in saved_params.get("metrics_queries", []) if str(q).strip()
    ]
    raw_alerts = saved_params.get("alerts")
    legacy_user_goal = str(saved_params.get("user_goal", ""))
    normalized_alerts = _normalize_alert_items(
        raw_alerts,
        min_items=1,
        legacy_user_goal=legacy_user_goal,
    )
    rendered_goal = _render_alerts_context(
        _extract_alerts_from_items(normalized_alerts)
    ) or legacy_user_goal
    return {
        "logs_queries": logs_queries,
        "metrics_queries": metrics_queries,
        "logs_sum_user_goal": rendered_goal,
        "alerts": normalized_alerts,
        "logs_sum_model_id": str(saved_params.get("llm_model_id", getattr(settings, "LLM_MODEL_ID", ""))).strip(),
        "logs_sum_period_mode": str(
            saved_params.get("period_mode", "Явный диапазон (start/end)")
        ),
        "logs_sum_window_minutes": _to_int(saved_params.get("window_minutes", 30), 30),
        "logs_sum_center_dt": str(saved_params.get("center_dt_text", "")),
        "logs_sum_start_dt": str(saved_params.get("start_dt_text", "")),
        "logs_sum_end_dt": str(saved_params.get("end_dt_text", "")),
        "logs_sum_db_batch": _to_int(saved_params.get("db_batch_size", 1000), 1000),
        # User-defined upper cap for rows per one LLM MAP call.
        "logs_sum_llm_batch": _to_int(
            saved_params.get("llm_batch_size", saved_params.get("db_batch_size", 1000)),
            _to_int(saved_params.get("db_batch_size", 1000), 1000),
        ),
        "logs_sum_parallel_map": False,
        "logs_sum_map_workers": 1,
        "logs_sum_max_retries": _to_int(saved_params.get("max_retries", -1), -1),
        "logs_sum_llm_timeout": _to_int(saved_params.get("llm_timeout", 600), 600),
        "logs_sum_demo_mode": bool(saved_params.get("demo_mode", False)),
        "logs_sum_demo_logs_count": _to_int(saved_params.get("demo_logs_count", 4000), 4000),
        "logs_sum_enable_no_logs_hypothesis": bool(
            saved_params.get("enable_no_logs_hypothesis", False)
        ),
    }


def _read_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)


def _checkpoint_payload_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "status",
        "mode",
        "period_mode",
        "period_start",
        "period_end",
        "window_minutes",
        "queries_count",
        "metrics_queries_count",
        "logs_processed",
        "logs_total",
        "last_batch_ts",
        "resume_rows_offset",
        "active_step",
        "error",
        "started_at",
        "elapsed_seconds",
        "log_seconds_per_second",
        "eta_seconds_left",
        "eta_finish_at",
        "progress_samples",
        "stats",
        "map_batches",
        "map_batches_done_total",
        "map_batches_total",
        "source_batch_offset",
        "resume_batch_offset",
        "resume_stats_offset",
        "final_summary",
        "final_summary_origin",
        "structured_sections",
        "freeform_final_summary",
        "freeform_sections",
        "result_json_path",
        "result_bundle_path",
        "result_summary_path",
        "result_html_path",
        "result_structured_md_path",
        "result_freeform_md_path",
        "result_structured_txt_path",
        "result_freeform_txt_path",
        "llm_calls_started",
        "llm_calls_succeeded",
        "llm_calls_failed",
        "llm_last_error",
        "llm_timeline",
        "report_progress_current",
        "report_progress_total",
        "report_progress_label",
        "report_progress_active",
        "events",
        "query_errors",
    )
    return {
        "saved_at": datetime.now(MSK).isoformat(),
        "state": {k: _json_safe(state.get(k)) for k in keys},
    }


def _persist_checkpoint(path: Path, state: Dict[str, Any]) -> None:
    _write_json_file(path, _checkpoint_payload_from_state(state))


def _discover_resume_sessions(output_dir: Path) -> List[Dict[str, Any]]:
    sessions_root = output_dir / "logs_summary_live"
    if not sessions_root.exists() or not sessions_root.is_dir():
        return []
    sessions: List[Dict[str, Any]] = []
    for run_dir in sorted(sessions_root.glob("run_*"), reverse=True):
        if not run_dir.is_dir():
            continue
        run_params_path = run_dir / "run_params.json"
        checkpoint_path = run_dir / "checkpoint.json"
        if not run_params_path.exists():
            continue
        checkpoint = _read_json_file(checkpoint_path) or {}
        checkpoint_state = checkpoint.get("state", {}) if isinstance(checkpoint, dict) else {}
        status = str(checkpoint_state.get("status", "unknown"))
        saved_at = str(checkpoint.get("saved_at", ""))
        try:
            parsed_saved = _to_msk_ts(saved_at)
            saved_text = (
                parsed_saved.strftime("%Y-%m-%d %H:%M:%S.%f MSK")
                if not pd.isna(parsed_saved)
                else "n/a"
            )
        except Exception:
            saved_text = "n/a"
        sessions.append(
            {
                "id": run_dir.name,
                "run_dir": str(run_dir),
                "run_params_path": str(run_params_path),
                "checkpoint_path": str(checkpoint_path),
                "request_path": str(run_dir / "request.json"),
                "status": status,
                "saved_at_text": saved_text,
                "label": f"{run_dir.name} | status={status} | saved={saved_text}",
            }
        )
    return sessions


def _build_portable_report_bundle(
    *,
    request_payload: Dict[str, Any],
    result_state: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "bundle_type": PORTABLE_BUNDLE_TYPE,
        "bundle_version": PORTABLE_BUNDLE_VERSION,
        "saved_at": datetime.now(MSK).isoformat(),
        "request": _json_safe(request_payload),
        "result": _json_safe(result_state),
    }


def _extract_request_result_from_bundle(payload: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(payload, dict):
        return {}, {}
    request = payload.get("request")
    result = payload.get("result")
    if isinstance(request, dict) and isinstance(result, dict):
        return request, result
    return {}, {}


def _state_from_imported_result(result_payload: Dict[str, Any]) -> Dict[str, Any]:
    state = dict(result_payload or {})
    status = str(state.get("status") or "").strip().lower()
    if status not in {"done", "error"}:
        state["status"] = "done"
    path_keys = (
        "result_json_path",
        "result_summary_path",
        "result_bundle_path",
        "result_html_path",
        "result_structured_md_path",
        "result_freeform_md_path",
        "result_structured_txt_path",
        "result_freeform_txt_path",
        "rebuild_reduce_path",
        "live_events_path",
        "live_batches_path",
        "run_params_path",
        "request_path",
        "checkpoint_path",
        "map_summaries_jsonl_path",
        "reduce_summaries_jsonl_path",
        "llm_calls_jsonl_path",
    )
    for key in path_keys:
        state[key] = None
    state.setdefault("events", [])
    state.setdefault("map_batches", [])
    state.setdefault("map_batches_done_total", len(state.get("map_batches") or []))
    state.setdefault("map_batches_total", _safe_int(state.get("map_batches_done_total"), 0))
    state.setdefault("source_batch_offset", _safe_int(state.get("map_batches_done_total"), 0))
    state.setdefault("reduce_nodes", [])
    state.setdefault("structured_sections", [])
    state.setdefault("freeform_sections", [])
    state.setdefault("report_progress_current", 0)
    state.setdefault("report_progress_total", 0)
    state.setdefault("report_progress_label", "")
    state.setdefault("report_progress_active", False)
    state.setdefault("stats", {})
    state.setdefault("final_summary_origin", "imported_bundle")
    return state


def _build_saved_params_from_import_request(
    *,
    request_payload: Dict[str, Any],
    center_default: str,
    start_default: str,
    end_default: str,
    default_query: str,
) -> Dict[str, Any]:
    request = dict(request_payload or {})
    logs_queries_raw = request.get("logs_queries")
    if isinstance(logs_queries_raw, list):
        logs_queries = [str(item) for item in logs_queries_raw if str(item).strip()]
    else:
        sql_query = str(request.get("sql_query") or "").strip()
        logs_queries = [sql_query] if sql_query else []
    if not logs_queries:
        logs_queries = [default_query]

    metrics_queries_raw = request.get("metrics_queries")
    if isinstance(metrics_queries_raw, list):
        metrics_queries = [str(item) for item in metrics_queries_raw if str(item).strip()]
    else:
        metrics_query = str(request.get("metrics_query") or "").strip()
        metrics_queries = [metrics_query] if metrics_query else []

    period_mode_raw = str(request.get("period_mode") or "").strip().lower()
    period_mode = (
        "Окно вокруг даты (±N минут)"
        if period_mode_raw in {"window", "окно вокруг даты (±n минут)"}
        else "Явный диапазон (start/end)"
    )
    start_text = str(request.get("period_start") or request.get("effective_period_start") or start_default)
    end_text = str(request.get("period_end") or end_default)
    center_text = center_default
    start_ts = _parse_user_dt(start_text)
    end_ts = _parse_user_dt(end_text)
    if not pd.isna(start_ts) and not pd.isna(end_ts):
        midpoint = start_ts + (end_ts - start_ts) / 2
        try:
            center_text = midpoint.to_pydatetime().isoformat()
        except Exception:
            center_text = center_default

    return {
        "logs_queries": logs_queries,
        "metrics_queries": metrics_queries,
        "alerts": request.get("alerts", []),
        "user_goal": str(request.get("user_goal") or ""),
        "period_mode": period_mode,
        "window_minutes": _safe_int(request.get("window_minutes"), 30),
        "center_dt_text": center_text,
        "start_dt_text": start_text,
        "end_dt_text": end_text,
        "db_batch_size": _safe_int(request.get("db_batch_size"), 1000),
        "llm_batch_size": _safe_int(
            request.get("llm_batch_size", request.get("db_batch_size")),
            1000,
        ),
        "llm_model_id": str(request.get("llm_model_id") or getattr(settings, "LLM_MODEL_ID", "")).strip(),
        "max_retries": _safe_int(request.get("max_retries"), -1),
        "llm_timeout": _safe_int(request.get("llm_timeout"), 600),
        "demo_mode": bool(request.get("demo_mode", False)),
        "demo_logs_count": _safe_int(request.get("demo_logs_count"), 4000),
        "enable_no_logs_hypothesis": bool(request.get("enable_no_logs_hypothesis", False)),
    }


def _html_cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(_json_safe(value), ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def _html_table_from_rows(
    rows: List[Dict[str, Any]],
    *,
    preferred_columns: Optional[List[str]] = None,
    empty_text: str = "Данные отсутствуют.",
) -> str:
    if not rows:
        return f"<div class='empty'>{html.escape(empty_text)}</div>"
    columns: List[str] = []
    if preferred_columns:
        for col in preferred_columns:
            if any(col in row for row in rows):
                columns.append(col)
    if not columns:
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                key_text = str(key)
                if key_text not in seen:
                    seen.add(key_text)
                    columns.append(key_text)
    head_html = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body_parts: List[str] = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape(_html_cell_text(row.get(col)))}</td>" for col in columns
        )
        body_parts.append(f"<tr>{cells}</tr>")
    body_html = "".join(body_parts)
    return (
        "<div class='table-wrap'>"
        "<table>"
        f"<thead><tr>{head_html}</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table>"
        "</div>"
    )


def _html_pre_block(text: str, *, empty_text: str = "Данные отсутствуют.") -> str:
    normalized = _normalize_summary_text(text)
    if not normalized:
        return f"<div class='empty'>{html.escape(empty_text)}</div>"
    return f"<pre class='report-text'>{html.escape(normalized)}</pre>"


def _build_interactive_final_report_html(state: Dict[str, Any]) -> str:
    final_summary = _normalize_summary_text(state.get("final_summary"))
    freeform_summary = _normalize_summary_text(state.get("freeform_final_summary"))
    report_sections_map = _get_report_sections_map(state)
    if not report_sections_map:
        report_sections_map = _extract_sections_from_text(final_summary)

    timeline_rows = _collect_timeline_events(state)
    timeline_rows_view: List[Dict[str, Any]] = []
    for row in timeline_rows:
        item = dict(row)
        if item.get("timestamp"):
            item["timestamp"] = _format_datetime_with_tz(item.get("timestamp"))
        tags_value = item.get("tags")
        if isinstance(tags_value, list):
            item["tags"] = ", ".join([str(tag) for tag in tags_value if str(tag).strip()])
        timeline_rows_view.append(item)

    hypotheses_rows = _collect_hypotheses(state)
    causal_links_rows = _collect_causal_links(state)
    gaps_rows = _collect_gaps(state)
    conflicts_rows = _collect_conflicts(state)
    recommendations_rows = _collect_recommendations(state)
    alert_rows = _build_alert_panel_state(state)

    grouped_hypotheses: Dict[str, List[Dict[str, Any]]] = {}
    for hyp in hypotheses_rows:
        related = hyp.get("related_alert_ids")
        if isinstance(related, list) and related:
            key = str(related[0])
        else:
            key = "unbound"
        grouped_hypotheses.setdefault(key, []).append(dict(hyp))

    grouped_recs = _group_recommendations_by_priority(recommendations_rows)
    stats = state.get("stats") or {}
    status_text = "Готово" if str(state.get("status", "")).strip().lower() == "done" else "Ошибка"
    rows_processed = _safe_int(state.get("logs_processed"), 0)
    llm_calls = _safe_int(stats.get("llm_calls"), 0)
    reduce_rounds = _safe_int(stats.get("reduce_rounds"), 0)
    processing_human = str(stats.get("logs_processing_human") or "").strip()
    processing_seconds = pd.to_numeric(stats.get("logs_processing_seconds"), errors="coerce")
    if not processing_human and not pd.isna(processing_seconds):
        processing_human = _format_eta_seconds(float(processing_seconds))

    section_1 = _find_section_text(report_sections_map, "1.")
    section_2 = _find_section_text(report_sections_map, "2.") or final_summary
    section_3 = _find_section_text(report_sections_map, "3.")
    section_4 = _find_section_text(report_sections_map, "4.")
    section_5 = _find_section_text(report_sections_map, "5.")
    section_6 = _find_section_text(report_sections_map, "6.")
    section_7 = _find_section_text(report_sections_map, "7.")
    section_8 = _find_section_text(report_sections_map, "8.")
    section_9 = _find_section_text(report_sections_map, "9.")
    section_10 = _find_section_text(report_sections_map, "10.")
    section_11 = _find_section_text(report_sections_map, "11.")
    section_12 = _find_section_text(report_sections_map, "12.")
    section_13 = _find_section_text(report_sections_map, "13.")

    alert_cards: List[str] = []
    for row in alert_rows:
        status = str(row.get("status") or "NOT_SEEN_IN_BATCH")
        view = ALERT_STATUS_VIEW.get(status, ALERT_STATUS_VIEW["NOT_SEEN_IN_BATCH"])
        details = str(row.get("details") or "").strip()
        explanation = str(row.get("explanation") or "").strip()
        details_html = (
            f"<div class='meta'>{html.escape(details)}</div>" if details else ""
        )
        explanation_html = (
            f"<div class='meta'>{html.escape(explanation)}</div>" if explanation else ""
        )
        related = row.get("related_events") or []
        related_html = ""
        if isinstance(related, list) and related:
            related_html = (
                "<div class='meta'>related_events: "
                + html.escape(", ".join([str(x) for x in related]))
                + "</div>"
            )
        alert_cards.append(
            (
                "<div class='alert-card' style='border-left-color:"
                + html.escape(view["color"])
                + ";'>"
                + f"<div class='alert-title'>{html.escape(view['icon'])} {html.escape(str(row.get('alert_id') or 'alert'))} — {html.escape(view['label'])}</div>"
                + details_html
                + related_html
                + explanation_html
                + "</div>"
            )
        )
    alerts_html = "".join(alert_cards) or "<div class='empty'>Алерты отсутствуют.</div>"

    hypotheses_html_blocks: List[str] = []
    if grouped_hypotheses:
        for alert_id, items in grouped_hypotheses.items():
            hypotheses_html_blocks.append(f"<h3>{html.escape(str(alert_id))}</h3>")
            ordered = sorted(
                items,
                key=lambda item: _safe_float(item.get("confidence"), 0.0),
                reverse=True,
            )
            for hyp in ordered:
                conf = min(max(_safe_float(hyp.get("confidence"), 0.0), 0.0), 1.0)
                title = str(hyp.get("title") or "Гипотеза")
                desc = _normalize_summary_text(hyp.get("description"))
                hypotheses_html_blocks.append(
                    "<div class='hypothesis-item'>"
                    f"<div class='hypothesis-title'>{html.escape(title)} ({conf:.2f})</div>"
                    "<div class='progress-track'>"
                    f"<div class='progress-fill' style='width:{conf * 100:.1f}%;'></div>"
                    "</div>"
                    + (f"<div class='meta'>{html.escape(desc)}</div>" if desc else "")
                    + "</div>"
                )
    hypotheses_html = "".join(hypotheses_html_blocks) or "<div class='empty'>Гипотезы отсутствуют.</div>"

    rec_blocks: List[str] = []
    priority_styles = {"P0": "#b91c1c", "P1": "#a16207", "P2": "#4b5563"}
    for priority in ("P0", "P1", "P2"):
        rec_blocks.append(
            f"<h3 style='color:{priority_styles[priority]};'>{priority}</h3>"
        )
        items = grouped_recs.get(priority) or []
        if not items:
            rec_blocks.append("<div class='empty'>—</div>")
            continue
        for item in items:
            action = str(item.get("action") or "Действие не указано")
            rationale = str(item.get("rationale") or "").strip()
            rec_blocks.append(
                "<div class='rec-item'>"
                f"<div><strong>{html.escape(action)}</strong></div>"
                + (f"<div class='meta'>{html.escape(rationale)}</div>" if rationale else "")
                + "</div>"
            )
    recs_html = "".join(rec_blocks)

    timeline_json = json.dumps(_json_safe(timeline_rows_view), ensure_ascii=False).replace(
        "</", "<\\/"
    )
    alert_rows_json = json.dumps(_json_safe(alert_rows), ensure_ascii=False).replace(
        "</", "<\\/"
    )
    causal_links_table = _html_table_from_rows(
        [dict(item) for item in causal_links_rows if isinstance(item, dict)],
        preferred_columns=["id", "cause_event_id", "effect_event_id", "mechanism", "confidence"],
        empty_text="Causal links отсутствуют.",
    )
    conflicts_table = _html_table_from_rows(
        [dict(item) for item in conflicts_rows if isinstance(item, dict)],
        empty_text="Конфликтующие интерпретации не обнаружены.",
    )
    gaps_table = _html_table_from_rows(
        [dict(item) for item in gaps_rows if isinstance(item, dict)],
        empty_text="Разрывы не обнаружены.",
    )

    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Logs Summary Interactive Report</title>
  <style>
    :root {{
      --bg: #f4f6fb;
      --panel: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --border: #e2e8f0;
      --primary: #2563eb;
      --ok: #15803d;
      --warn: #a16207;
      --danger: #b91c1c;
      --chip: #eef2ff;
    }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #eef2ff 0%, #f8fafc 100%);
      color: var(--text);
      font-family: "Segoe UI", "SF Pro Text", "Inter", Arial, sans-serif;
      line-height: 1.45;
    }}
    .wrap {{
      max-width: 1360px;
      margin: 0 auto;
      padding: 1.2rem;
    }}
    .title {{
      margin: 0 0 0.5rem 0;
      font-size: 1.4rem;
      font-weight: 700;
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.92rem;
      white-space: pre-wrap;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 0.65rem;
      margin: 0.9rem 0 1rem 0;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0.75rem;
    }}
    .metric-value {{
      font-size: 1.25rem;
      font-weight: 700;
      margin-top: 0.2rem;
    }}
    .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem;
      margin-bottom: 0.8rem;
    }}
    .tab-btn {{
      border: 1px solid var(--border);
      background: #fff;
      color: var(--text);
      border-radius: 999px;
      padding: 0.38rem 0.82rem;
      cursor: pointer;
      font-size: 0.92rem;
    }}
    .tab-btn.active {{
      background: var(--primary);
      color: #fff;
      border-color: var(--primary);
    }}
    .tab-pane {{
      display: none;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0.85rem;
      margin-bottom: 0.9rem;
    }}
    .tab-pane.active {{ display: block; }}
    .report-text {{
      margin: 0;
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.75rem;
      white-space: pre-wrap;
      font-size: 0.95rem;
      max-height: 540px;
      overflow: auto;
    }}
    .table-wrap {{
      overflow: auto;
      max-height: 520px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #fff;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 800px;
      font-size: 0.9rem;
    }}
    th, td {{
      border-bottom: 1px solid #edf2f7;
      text-align: left;
      padding: 0.42rem 0.5rem;
      vertical-align: top;
    }}
    thead th {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: #f8fafc;
      font-weight: 700;
    }}
    .filters {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 0.5rem;
      margin-bottom: 0.55rem;
    }}
    .filters label {{
      font-size: 0.84rem;
      color: var(--muted);
      display: block;
      margin-bottom: 0.2rem;
    }}
    .filters select, .filters input {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.4rem 0.45rem;
      font-size: 0.9rem;
      background: #fff;
    }}
    .alert-card {{
      background: #fff;
      border: 1px solid var(--border);
      border-left-width: 6px;
      border-radius: 10px;
      padding: 0.65rem;
      margin-bottom: 0.55rem;
    }}
    .alert-title {{
      font-weight: 700;
      margin-bottom: 0.2rem;
    }}
    .hypothesis-item, .rec-item {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.55rem;
      margin-bottom: 0.5rem;
      background: #fff;
    }}
    .hypothesis-title {{
      font-weight: 700;
      margin-bottom: 0.25rem;
    }}
    .progress-track {{
      width: 100%;
      height: 8px;
      background: #e2e8f0;
      border-radius: 999px;
      overflow: hidden;
      margin-bottom: 0.25rem;
    }}
    .progress-fill {{
      height: 100%;
      background: var(--primary);
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem;
      margin-bottom: 0.55rem;
    }}
    .chip {{
      border: 1px solid #c7d2fe;
      background: var(--chip);
      border-radius: 999px;
      padding: 0.22rem 0.6rem;
      font-size: 0.84rem;
    }}
    .empty {{
      color: var(--muted);
      border: 1px dashed #cbd5e1;
      border-radius: 10px;
      padding: 0.75rem;
      background: #f8fafc;
    }}
    details {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.5rem 0.65rem;
      margin-bottom: 0.55rem;
      background: #fff;
    }}
    summary {{
      cursor: pointer;
      font-weight: 600;
    }}
    @media (max-width: 1080px) {{
      .grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .filters {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
    @media (max-width: 680px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
      .filters {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1 class="title">Итоговый Отчёт По Саммаризации Логов</h1>
    <div class="meta">saved_at: {html.escape(datetime.now(MSK).strftime("%Y-%m-%d %H:%M:%S.%f MSK"))}</div>
    <div class="meta">period: {html.escape(_format_datetime_with_tz(state.get("period_start")))} -> {html.escape(_format_datetime_with_tz(state.get("period_end")))}</div>
    <div class="meta">summary_origin: {html.escape(_summary_origin_label(state.get("final_summary_origin")) or str(state.get("final_summary_origin") or ""))}</div>

    <div class="grid">
      <div class="card"><div class="meta">Статус</div><div class="metric-value">{html.escape(status_text)}</div></div>
      <div class="card"><div class="meta">Обработано логов</div><div class="metric-value">{rows_processed:,}</div></div>
      <div class="card"><div class="meta">LLM вызовы</div><div class="metric-value">{llm_calls}</div></div>
      <div class="card"><div class="meta">Reduce раунды</div><div class="metric-value">{reduce_rounds}</div></div>
    </div>
    <div class="meta">Время обработки логов: {html.escape(processing_human or "n/a")}</div>

    <div class="tabs">
      <button class="tab-btn active" data-tab="summary">Резюме</button>
      <button class="tab-btn" data-tab="timeline">Хронология</button>
      <button class="tab-btn" data-tab="chains">Цепочки</button>
      <button class="tab-btn" data-tab="alerts">Алерты</button>
      <button class="tab-btn" data-tab="hypotheses">Гипотезы</button>
      <button class="tab-btn" data-tab="recs">Рекомендации</button>
      <button class="tab-btn" data-tab="more">Ещё</button>
    </div>

    <section id="tab-summary" class="tab-pane active">
      {_html_pre_block(section_2, empty_text="Резюме пока недоступно.")}
    </section>

    <section id="tab-timeline" class="tab-pane">
      <div class="filters">
        <div>
          <label for="severityFilter">Severity</label>
          <select id="severityFilter"><option value="">Все</option></select>
        </div>
        <div>
          <label for="sourceFilter">Source</label>
          <select id="sourceFilter"><option value="">Все</option></select>
        </div>
        <div>
          <label for="evidenceFilter">Тип</label>
          <select id="evidenceFilter"><option value="">Все</option></select>
        </div>
        <div>
          <label for="tagFilter">Tags contains</label>
          <input id="tagFilter" type="text" placeholder="например: OOM" />
        </div>
      </div>
      <div class="meta" id="timelineCounter"></div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>timestamp</th><th>source</th><th>description</th><th>severity</th><th>evidence_type</th><th>importance</th><th>tags</th><th>id</th><th>batch_index</th>
            </tr>
          </thead>
          <tbody id="timelineBody"></tbody>
        </table>
      </div>
      <details style="margin-top:0.6rem;">
        <summary>Текст раздела 4 (Полная Хронология)</summary>
        {_html_pre_block(section_4, empty_text="Раздел 4 отсутствует.")}
      </details>
    </section>

    <section id="tab-chains" class="tab-pane">
      {causal_links_table}
      <details style="margin-top:0.6rem;">
        <summary>Текст раздела 5 (Причинно-следственные цепочки)</summary>
        {_html_pre_block(section_5, empty_text="Раздел 5 отсутствует.")}
      </details>
    </section>

    <section id="tab-alerts" class="tab-pane">
      <div class="chips" id="alertChips"></div>
      {alerts_html}
      <details style="margin-top:0.6rem;">
        <summary>Текст раздела 6 (Связь с алертами)</summary>
        {_html_pre_block(section_6, empty_text="Раздел 6 отсутствует.")}
      </details>
    </section>

    <section id="tab-hypotheses" class="tab-pane">
      {hypotheses_html}
      <details style="margin-top:0.6rem;">
        <summary>Текст раздела 8 (Гипотезы первопричин)</summary>
        {_html_pre_block(section_8, empty_text="Раздел 8 отсутствует.")}
      </details>
    </section>

    <section id="tab-recs" class="tab-pane">
      {recs_html}
      <details style="margin-top:0.6rem;">
        <summary>Текст раздела 12 (Рекомендации)</summary>
        {_html_pre_block(section_12, empty_text="Раздел 12 отсутствует.")}
      </details>
    </section>

    <section id="tab-more" class="tab-pane">
      <details open><summary>1. Контекст Инцидента</summary>{_html_pre_block(section_1, empty_text="Раздел 1 отсутствует.")}</details>
      <details><summary>3. Покрытие Данных</summary>{_html_pre_block(section_3, empty_text="Раздел 3 отсутствует.")}</details>
      <details><summary>7. Метрики И Корреляции</summary>{_html_pre_block(section_7, empty_text="Раздел 7 отсутствует.")}</details>
      <details><summary>9. Конфликтующие Версии</summary>{conflicts_table}{_html_pre_block(section_9, empty_text="Раздел 9 отсутствует.")}</details>
      <details><summary>10. Разрывы В Цепочках</summary>{gaps_table}{_html_pre_block(section_10, empty_text="Раздел 10 отсутствует.")}</details>
      <details><summary>11. Масштаб И Влияние</summary>{_html_pre_block(section_11, empty_text="Раздел 11 отсутствует.")}</details>
      <details><summary>13. Ограничения Анализа</summary>{_html_pre_block(section_13, empty_text="Раздел 13 отсутствует.")}</details>
      <details><summary>Итоговое расследование в свободном формате</summary>{_html_pre_block(freeform_summary, empty_text="Свободный отчёт отсутствует.")}</details>
    </section>
  </div>

  <script>
    const timelineRows = {timeline_json};
    const alertRows = {alert_rows_json};

    function uniqSorted(values) {{
      return Array.from(new Set(values.filter(Boolean).map(v => String(v)))).sort((a, b) => a.localeCompare(b));
    }}

    function populateSelect(selectId, values) {{
      const select = document.getElementById(selectId);
      if (!select) return;
      values.forEach(v => {{
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        select.appendChild(opt);
      }});
    }}

    function getSelected(id) {{
      const el = document.getElementById(id);
      return el ? String(el.value || "").trim() : "";
    }}

    function renderTimeline() {{
      const severity = getSelected("severityFilter");
      const source = getSelected("sourceFilter");
      const evidence = getSelected("evidenceFilter");
      const tagNeedle = getSelected("tagFilter").toLowerCase();

      const filtered = timelineRows.filter(row => {{
        const rowSeverity = String(row.severity || "");
        const rowSource = String(row.source || "");
        const rowEvidence = String(row.evidence_type || "");
        const rowTags = String(row.tags || "").toLowerCase();
        if (severity && rowSeverity !== severity) return false;
        if (source && rowSource !== source) return false;
        if (evidence && rowEvidence !== evidence) return false;
        if (tagNeedle && !rowTags.includes(tagNeedle)) return false;
        return true;
      }});

      const tbody = document.getElementById("timelineBody");
      if (!tbody) return;
      tbody.innerHTML = "";
      filtered.forEach(row => {{
        const tr = document.createElement("tr");
        const cols = ["timestamp", "source", "description", "severity", "evidence_type", "importance", "tags", "id", "batch_index"];
        cols.forEach(col => {{
          const td = document.createElement("td");
          td.textContent = row[col] === undefined || row[col] === null ? "" : String(row[col]);
          tr.appendChild(td);
        }});
        tbody.appendChild(tr);
      }});

      const counter = document.getElementById("timelineCounter");
      if (counter) {{
        counter.textContent = `Показано событий: ${{filtered.length}} / ${{timelineRows.length}}`;
      }}
    }}

    function renderAlertChips() {{
      const container = document.getElementById("alertChips");
      if (!container) return;
      container.innerHTML = "";
      alertRows.forEach(row => {{
        const chip = document.createElement("div");
        chip.className = "chip";
        chip.textContent = `${{row.alert_id}}: ${{row.status}}`;
        container.appendChild(chip);
      }});
    }}

    function initTabs() {{
      const buttons = document.querySelectorAll(".tab-btn");
      const panes = document.querySelectorAll(".tab-pane");
      buttons.forEach(btn => {{
        btn.addEventListener("click", () => {{
          const target = btn.getAttribute("data-tab");
          buttons.forEach(x => x.classList.remove("active"));
          panes.forEach(x => x.classList.remove("active"));
          btn.classList.add("active");
          const pane = document.getElementById(`tab-${{target}}`);
          if (pane) pane.classList.add("active");
        }});
      }});
    }}

    function init() {{
      initTabs();
      populateSelect("severityFilter", uniqSorted(timelineRows.map(r => r.severity)));
      populateSelect("sourceFilter", uniqSorted(timelineRows.map(r => r.source)));
      populateSelect("evidenceFilter", uniqSorted(timelineRows.map(r => r.evidence_type)));
      ["severityFilter", "sourceFilter", "evidenceFilter", "tagFilter"].forEach(id => {{
        const el = document.getElementById(id);
        if (!el) return;
        el.addEventListener("change", renderTimeline);
        el.addEventListener("input", renderTimeline);
      }});
      renderTimeline();
      renderAlertChips();
    }}

    document.addEventListener("DOMContentLoaded", init);
  </script>
</body>
</html>
"""


def _save_logs_summary_result(
    *,
    output_dir: Path,
    request_payload: Dict[str, Any],
    result_state: Dict[str, Any],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(MSK).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"logs_summary_result_{stamp}.json"
    bundle_path = output_dir / f"logs_summary_bundle_{stamp}.json"
    summary_path = output_dir / f"logs_summary_result_{stamp}.md"
    report_html_path = output_dir / f"logs_summary_report_{stamp}.html"
    structured_md_path = output_dir / f"logs_summary_structured_{stamp}.md"
    freeform_md_path = output_dir / f"logs_summary_freeform_{stamp}.md"
    structured_txt_path = output_dir / f"logs_summary_structured_{stamp}.txt"
    freeform_txt_path = output_dir / f"logs_summary_freeform_{stamp}.txt"

    payload = {
        "saved_at": datetime.now(MSK).isoformat(),
        "request": _json_safe(request_payload),
        "result": _json_safe(result_state),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(
            _build_portable_report_bundle(
                request_payload=request_payload,
                result_state=result_state,
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )

    structured_summary = str(result_state.get("final_summary") or "").strip()
    freeform_summary = str(result_state.get("freeform_final_summary") or "").strip()
    structured_fence = _md_fence_for_text(structured_summary)
    freeform_fence = _md_fence_for_text(freeform_summary)
    combined_structured_fence = _md_fence_for_text(structured_summary or "N/A")
    combined_freeform_fence = _md_fence_for_text(freeform_summary or "N/A")
    if structured_summary:
        _write_text_file(
            structured_md_path,
            "\n".join(
                [
                    "# Итоговое расследование",
                    "",
                    f"- saved_at: `{payload['saved_at']}`",
                    f"- period: `{result_state.get('period_start')}` -> `{result_state.get('period_end')}`",
                    f"- summary_origin: `{result_state.get('final_summary_origin')}`",
                    "",
                    f"{structured_fence}text",
                    structured_summary,
                    structured_fence,
                ]
            ),
        )
    if freeform_summary:
        _write_text_file(
            freeform_md_path,
            "\n".join(
                [
                    "# Итоговое расследование в свободном формате",
                    "",
                    f"- saved_at: `{payload['saved_at']}`",
                    f"- period: `{result_state.get('period_start')}` -> `{result_state.get('period_end')}`",
                    f"- summary_origin: `{result_state.get('final_summary_origin')}`",
                    "",
                    f"{freeform_fence}text",
                    freeform_summary,
                    freeform_fence,
                ]
            ),
        )
    if structured_summary:
        structured_txt_path.write_text(structured_summary, encoding="utf-8")
    if freeform_summary:
        freeform_txt_path.write_text(freeform_summary, encoding="utf-8")

    _write_text_file(
        report_html_path,
        _build_interactive_final_report_html(result_state),
    )

    lines = [
        "# Logs Summary Result",
        "",
        f"- saved_at: `{payload['saved_at']}`",
        f"- status: `{result_state.get('status')}`",
        f"- summary_origin: `{result_state.get('final_summary_origin')}`",
        f"- mode: `{result_state.get('mode')}`",
        f"- period: `{result_state.get('period_start')}` -> `{result_state.get('period_end')}`",
        "",
        "## Final Summary (Structured)",
        "",
        f"{combined_structured_fence}text",
        (structured_summary or "N/A"),
        combined_structured_fence,
        "",
        "## Final Summary (Freeform)",
        "",
        f"{combined_freeform_fence}text",
        (freeform_summary or "N/A"),
        combined_freeform_fence,
        "",
        "## Stats",
        "",
        f"- logs_processed: `{result_state.get('logs_processed')}`",
        f"- logs_total: `{result_state.get('logs_total')}`",
        f"- stats: `{result_state.get('stats')}`",
        f"- error: `{result_state.get('error')}`",
        f"- map_summaries_jsonl: `{result_state.get('map_summaries_jsonl_path')}`",
        f"- reduce_summaries_jsonl: `{result_state.get('reduce_summaries_jsonl_path')}`",
        f"- llm_calls_jsonl: `{result_state.get('llm_calls_jsonl_path')}`",
        f"- run_params_path: `{result_state.get('run_params_path')}`",
        f"- request_path: `{result_state.get('request_path')}`",
        f"- checkpoint_path: `{result_state.get('checkpoint_path')}`",
        "",
        f"JSON dump: `{json_path}`",
        f"- portable bundle: `{bundle_path}`",
        f"- report html: `{report_html_path}`",
        (f"- structured md: `{structured_md_path}`" if structured_summary else ""),
        (f"- freeform md: `{freeform_md_path}`" if freeform_summary else ""),
        (f"- structured txt: `{structured_txt_path}`" if structured_summary else ""),
        (f"- freeform txt: `{freeform_txt_path}`" if freeform_summary else ""),
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    result = {
        "json_path": str(json_path),
        "bundle_path": str(bundle_path),
        "summary_path": str(summary_path),
        "html_path": str(report_html_path),
    }
    if structured_summary:
        result["structured_md_path"] = str(structured_md_path)
    if freeform_summary:
        result["freeform_md_path"] = str(freeform_md_path)
    if structured_summary:
        result["structured_txt_path"] = str(structured_txt_path)
    if freeform_summary:
        result["freeform_txt_path"] = str(freeform_txt_path)
    return result


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


def _safe_int(value: Any, default: int = 0) -> int:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return int(default)
    return int(parsed)


def _safe_float(value: Any, default: float = 0.0) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return float(default)
    return float(parsed)


def _resolve_map_batches_progress(
    state: Dict[str, Any],
    map_batches: Optional[List[Dict[str, Any]]] = None,
) -> tuple[int, int]:
    batches = map_batches if isinstance(map_batches, list) else state.get("map_batches")
    if not isinstance(batches, list):
        batches = []

    done = _safe_int(state.get("map_batches_done_total"), 0)
    if done <= 0 and batches:
        done = max(
            max((_safe_int(item.get("batch_index"), -1) for item in batches), default=-1) + 1,
            len(batches),
        )

    total = max(
        _safe_int(state.get("map_batches_total"), 0),
        _safe_int(state.get("estimated_batch_total"), 0),
    )
    if total <= 0 and batches:
        total = max((_safe_int(item.get("batch_index"), -1) for item in batches), default=-1) + 1
    if total < done:
        total = done
    return max(done, 0), max(total, 0)


def _json_dict_from_text(raw: Any) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            payload = json.loads(candidate)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


def _summary_payload_from_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    structured = batch.get("batch_summary_structured")
    if isinstance(structured, dict) and structured:
        return structured
    return _json_dict_from_text(batch.get("batch_summary"))


def _get_report_sections_map(state: Dict[str, Any]) -> Dict[str, str]:
    sections_map: Dict[str, str] = {}
    raw_sections = state.get("structured_sections")
    if isinstance(raw_sections, list):
        for item in raw_sections:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            text = _normalize_summary_text(item.get("text"))
            if title:
                sections_map[title] = text
    if sections_map:
        return sections_map
    fallback_text = _normalize_summary_text(state.get("final_summary"))
    if not fallback_text:
        return sections_map
    # Very simple heading parser for markdown-style sections.
    current_title: Optional[str] = None
    current_lines: List[str] = []
    for line in fallback_text.splitlines():
        heading = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
        if heading:
            if current_title:
                sections_map[current_title] = "\n".join(current_lines).strip()
            current_title = str(heading.group(1)).strip()
            current_lines = []
            continue
        current_lines.append(line)
    if current_title:
        sections_map[current_title] = "\n".join(current_lines).strip()
    return sections_map


def _build_alert_panel_state(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    alerts = state.get("alerts")
    if not isinstance(alerts, list):
        return []
    map_batches = state.get("map_batches")
    if not isinstance(map_batches, list):
        map_batches = []
    panel_rows: List[Dict[str, Any]] = []
    for idx, alert in enumerate(alerts, start=1):
        if not isinstance(alert, dict):
            continue
        alert_id = str(alert.get("title") or alert.get("id") or f"alert_{idx}").strip() or f"alert_{idx}"
        details = str(alert.get("details") or "").strip()
        best_status = "NOT_SEEN_IN_BATCH"
        status_history: List[Dict[str, Any]] = []
        related_events: List[str] = []
        explanation_parts: List[str] = []
        for batch in map_batches:
            if not isinstance(batch, dict):
                continue
            payload = _summary_payload_from_batch(batch)
            if not payload:
                continue
            refs = payload.get("alert_refs")
            if not isinstance(refs, list):
                continue
            for ref in refs:
                if not isinstance(ref, dict):
                    continue
                ref_id = str(ref.get("alert_id") or "").strip()
                if ref_id and ref_id != alert_id:
                    continue
                status = str(ref.get("status") or "NOT_SEEN_IN_BATCH").strip().upper()
                if status not in ALERT_STATUS_PRIORITY:
                    status = "NOT_SEEN_IN_BATCH"
                if ALERT_STATUS_PRIORITY[status] >= ALERT_STATUS_PRIORITY.get(best_status, 1):
                    best_status = status
                batch_idx = _safe_int(batch.get("batch_index"), 0) + 1
                status_history.append(
                    {
                        "batch_index": batch_idx,
                        "status": status,
                        "explanation": str(ref.get("explanation") or "").strip(),
                        "related_events": ref.get("related_events") or [],
                    }
                )
                for event_id in ref.get("related_events") or []:
                    event_text = str(event_id).strip()
                    if event_text and event_text not in related_events:
                        related_events.append(event_text)
                explanation = str(ref.get("explanation") or "").strip()
                if explanation:
                    explanation_parts.append(explanation)
        panel_rows.append(
            {
                "alert_id": alert_id,
                "details": details,
                "status": best_status,
                "history": status_history,
                "related_events": related_events,
                "explanation": " ||| ".join(explanation_parts),
            }
        )
    return panel_rows


def _build_zip_artifacts_bytes(state: Dict[str, Any]) -> Optional[bytes]:
    artifact_candidates = [
        ("report/report.json", state.get("result_json_path")),
        ("report/report.bundle.json", state.get("result_bundle_path")),
        ("report/report.md", state.get("result_summary_path")),
        ("report/report.html", state.get("result_html_path")),
        ("report/structured.md", state.get("result_structured_md_path")),
        ("report/freeform.md", state.get("result_freeform_md_path")),
        ("report/structured.txt", state.get("result_structured_txt_path")),
        ("report/freeform.txt", state.get("result_freeform_txt_path")),
        ("summaries/map_summaries.jsonl", state.get("map_summaries_jsonl_path")),
        ("summaries/reduce_summaries.jsonl", state.get("reduce_summaries_jsonl_path")),
        ("summaries/llm_calls.jsonl", state.get("llm_calls_jsonl_path")),
        ("runtime/events.jsonl", state.get("live_events_path")),
        ("runtime/batches.jsonl", state.get("live_batches_path")),
        ("runtime/run_params.json", state.get("run_params_path")),
        ("runtime/request.json", state.get("request_path")),
        ("runtime/checkpoint.json", state.get("checkpoint_path")),
        ("runtime/rebuild_reduce.md", state.get("rebuild_reduce_path")),
    ]
    items: List[tuple[str, bytes]] = []
    for name, raw_path in artifact_candidates:
        path = Path(str(raw_path or "").strip())
        if not str(raw_path or "").strip() or not path.exists() or not path.is_file():
            continue
        try:
            items.append((name, path.read_bytes()))
        except Exception:
            continue
    if not items:
        return None
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in items:
            zf.writestr(name, content)
    return buffer.getvalue()


def _collect_map_payloads(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    map_batches = state.get("map_batches")
    if not isinstance(map_batches, list):
        return []
    payloads: List[Dict[str, Any]] = []
    for batch in map_batches:
        if not isinstance(batch, dict):
            continue
        payload = _summary_payload_from_batch(batch)
        if not isinstance(payload, dict) or not payload:
            continue
        payloads.append(payload)
    return payloads


def _collect_timeline_events(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    map_batches = state.get("map_batches")
    if not isinstance(map_batches, list):
        return []
    rows: List[Dict[str, Any]] = []
    for batch in map_batches:
        if not isinstance(batch, dict):
            continue
        payload = _summary_payload_from_batch(batch)
        timeline = payload.get("timeline") if isinstance(payload, dict) else None
        if not isinstance(timeline, list):
            continue
        batch_idx = _safe_int(batch.get("batch_index"), 0) + 1
        for event in timeline:
            if not isinstance(event, dict):
                continue
            row = dict(event)
            row.setdefault("batch_index", batch_idx)
            row.setdefault("batch_period_start", batch.get("batch_period_start"))
            row.setdefault("batch_period_end", batch.get("batch_period_end"))
            rows.append(row)
    rows.sort(key=lambda item: str(item.get("timestamp") or ""))
    return rows


def _collect_hypotheses(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    payloads = _collect_map_payloads(state)
    rows: List[Dict[str, Any]] = []
    for payload in payloads:
        hypotheses = payload.get("hypotheses")
        if not isinstance(hypotheses, list):
            continue
        for item in hypotheses:
            if isinstance(item, dict):
                rows.append(dict(item))
    return rows


def _collect_causal_links(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    payloads = _collect_map_payloads(state)
    rows: List[Dict[str, Any]] = []
    for payload in payloads:
        links = payload.get("causal_links")
        if not isinstance(links, list):
            continue
        for item in links:
            if isinstance(item, dict):
                rows.append(dict(item))
    return rows


def _collect_gaps(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    payloads = _collect_map_payloads(state)
    rows: List[Dict[str, Any]] = []
    for payload in payloads:
        gaps = payload.get("gaps")
        if not isinstance(gaps, list):
            continue
        for item in gaps:
            if isinstance(item, dict):
                rows.append(dict(item))
    return rows


def _collect_conflicts(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    payloads = _collect_map_payloads(state)
    rows: List[Dict[str, Any]] = []
    for payload in payloads:
        conflicts = payload.get("conflicts")
        if not isinstance(conflicts, list):
            continue
        for item in conflicts:
            if isinstance(item, dict):
                rows.append(dict(item))
    return rows


def _collect_recommendations(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    payloads = _collect_map_payloads(state)
    rows: List[Dict[str, Any]] = []
    for payload in payloads:
        recs = payload.get("preliminary_recommendations")
        if not isinstance(recs, list):
            continue
        for item in recs:
            if isinstance(item, dict):
                rows.append(dict(item))
    return rows


def _extract_sections_from_text(raw_text: str) -> Dict[str, str]:
    text = _normalize_summary_text(raw_text)
    if not text:
        return {}
    sections: Dict[str, str] = {}
    current_title: Optional[str] = None
    current_lines: List[str] = []
    for line in text.splitlines():
        if _looks_like_section_heading(line):
            if current_title:
                sections[current_title] = "\n".join(current_lines).strip()
            current_title = str(line).strip().lstrip("#").strip()
            current_lines = []
            continue
        current_lines.append(line)
    if current_title:
        sections[current_title] = "\n".join(current_lines).strip()
    return sections


def _find_section_text(sections_map: Dict[str, str], starts_with: str) -> str:
    prefix = str(starts_with or "").strip().lower()
    if not prefix:
        return ""
    for title, body in sections_map.items():
        if str(title).strip().lower().startswith(prefix):
            return _normalize_summary_text(body)
    return ""


def _build_causal_graph_dot(
    *,
    events: List[Dict[str, Any]],
    links: List[Dict[str, Any]],
) -> Optional[str]:
    if not links:
        return None
    by_id: Dict[str, Dict[str, Any]] = {}
    for event in events:
        event_id = str(event.get("id") or "").strip()
        if event_id:
            by_id[event_id] = event
    lines: List[str] = ["digraph CausalGraph {", "rankdir=LR;", 'node [shape=box, style="rounded,filled", fillcolor="#f8fafc"];']
    seen_nodes: set[str] = set()
    for link in links:
        cause = str(link.get("cause_event_id") or "").strip()
        effect = str(link.get("effect_event_id") or "").strip()
        if not cause or not effect:
            continue
        for node_id in (cause, effect):
            if node_id in seen_nodes:
                continue
            seen_nodes.add(node_id)
            event = by_id.get(node_id) or {}
            ts = str(event.get("timestamp") or "").strip()
            desc = str(event.get("description") or "").strip()
            short_desc = desc[:80] + ("..." if len(desc) > 80 else "")
            label = f"{node_id}\\n{ts}\\n{short_desc}".replace('"', '\\"')
            lines.append(f'"{node_id}" [label="{label}"];')
        confidence = _safe_float(link.get("confidence"), float("nan"))
        edge_label = ""
        if not pd.isna(confidence):
            edge_label = f" [label=\"conf={confidence:.2f}\"]"
        lines.append(f'"{cause}" -> "{effect}"{edge_label};')
    if len(lines) <= 3:
        return None
    lines.append("}")
    return "\n".join(lines)


def _group_recommendations_by_priority(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {"P0": [], "P1": [], "P2": []}
    for item in rows:
        priority = str(item.get("priority") or "").strip().upper()
        if priority not in grouped:
            priority = "P2"
        grouped[priority].append(item)
    return grouped


def _build_freeform_summary_prompt(
    *,
    final_summary: str,
    map_summaries_text: str = "",
    user_goal: str,
    period_start: str,
    period_end: str,
    stats: Dict[str, Any],
    metrics_context: str,
) -> str:
    goal_block = user_goal.strip() or "Не указан"
    incident_verbatim_block = _incident_verbatim_requirement_block(goal_block)
    metrics_block = metrics_context.strip() or "Нет доп. метрик в контексте."
    anti_rules = str(
        getattr(settings, "CONTROL_PLANE_LLM_ANTI_HALLUCINATION_RULES", "")
    ).strip()
    custom_template = str(
        getattr(settings, "CONTROL_PLANE_LLM_UI_FINAL_REPORT_PROMPT_TEMPLATE", "")
    ).strip()
    chain_block = (
        "\n\nОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК: ЦЕПОЧКА СОБЫТИЙ\n"
        "Оформи наглядно (Markdown-схема со стрелками):\n"
        "Для каждого узла обязательно укажи точную дату и время: YYYY-MM-DD HH:MM:SS.ffffff TZ.\n"
        "Не используй абстрактные t1/t2/t3 без реальных timestamp.\n"
        "[2026-03-31 12:34:56.123456 MSK] событие A [ФАКТ/ГИПОТЕЗА]\n"
        "    └─> механизм\n"
        "[2026-03-31 12:35:07.654321 MSK] событие B [ФАКТ/ГИПОТЕЗА]\n"
        "    └─> механизм\n"
        "[2026-03-31 12:35:10.000001 MSK] алерт/последствие [ФАКТ]\n"
        "Если есть разрывы — добавь: [РАЗРЫВ ЦЕПОЧКИ: чего не хватает].\n"
    )
    incident_link_block = (
        "\nОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК: СВЯЗЬ С ИНЦИДЕНТОМ ИЗ UI\n"
        "Возьми контекст инцидента/алертов из UI и для каждого пункта укажи:\n"
        "- статус: [ОБЪЯСНЁН]/[ЧАСТИЧНО ОБЪЯСНЁН]/[НЕ ОБЪЯСНЁН]\n"
        "- доказательства: конкретные timestamp/сообщения/метрики\n"
        "- цепочку причины->следствия (если подтверждается)\n"
        "- если не подтверждается: каких данных не хватает.\n"
    )
    root_cause_hypotheses_block = (
        "\nОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК: ПРЕДПОЛОЖЕНИЯ О ПЕРВОПРИЧИНЕ ПО КАЖДОМУ ИНЦИДЕНТУ\n"
        f"{incident_verbatim_block}\n"
        "Для КАЖДОГО выявленного инцидента/цепочки отдельно перечисли 2-5 гипотез первопричины:\n"
        "- Инцидент/цепочка: <название>\n"
        "- [ГИПОТЕЗА] первопричина: <кратко>\n"
        "- Почему это вероятно: <ссылка на timestamp/логи/метрики>\n"
        "- Что проверить для подтверждения/опровержения: <конкретные действия/данные>\n"
        "Если инцидент один — блок всё равно обязателен и минимум 2 гипотезы.\n"
    )
    if custom_template:
        rendered = _render_prompt_template(
            custom_template,
            {
                "final_summary": final_summary,
                "cross_source_summary": final_summary,
                "user_goal": goal_block,
                "incident_description": goal_block,
                "alerts_list": goal_block,
                "incident_start": period_start,
                "incident_end": period_end,
                "period_start": period_start,
                "period_end": period_end,
                "stats": json.dumps(_json_safe(stats), ensure_ascii=False, indent=2),
                "metrics_context": metrics_block,
                "map_summaries_text": str(map_summaries_text or ""),
                "anti_hallucination_rules": anti_rules,
            },
        )
        return f"{rendered.strip()}{chain_block}{incident_link_block}{root_cause_hypotheses_block}"
    anti_block = f"ПРАВИЛА АНТИГАЛЛЮЦИНАЦИИ:\n{anti_rules}\n\n" if anti_rules else ""
    return (
        "Напиши финальный narrative-отчёт для SRE на основе структурированного анализа.\n\n"
        f"КОНТЕКСТ ИНЦИДЕНТА:\n{goal_block}\n\n"
        "Нужен текст, который можно прочитать за 3–5 минут и понять полную картину.\n\n"
        "Структура:\n"
        "1) Краткое резюме (3-4 предложения).\n"
        "2) Ход событий: связный рассказ по хронологии, с явными переходами причины -> следствие.\n"
        "   Для каждого события обязательно указывай полную дату и время (до микросекунд) и timezone.\n"
        "3) Первопричина: что [ФАКТ], что [ГИПОТЕЗА].\n"
        "4) Что не удалось выяснить (разрывы цепочек и недостающие данные).\n"
        "5) Что делать дальше (конкретные приоритетные действия).\n\n"
        "6) ОТДЕЛЬНЫЙ БЛОК: ЦЕПОЧКА СОБЫТИЙ (обязательно, в виде схемы со стрелками).\n\n"
        "7) ОТДЕЛЬНЫЙ БЛОК: СВЯЗЬ С ИНЦИДЕНТОМ ИЗ UI (обязательно, с явным статусом по каждому алерту/пункту).\n\n"
        "8) ОТДЕЛЬНЫЙ БЛОК: ПРЕДПОЛОЖЕНИЯ О ПЕРВОПРИЧИНЕ ПО КАЖДОМУ ИНЦИДЕНТУ.\n\n"
        "Если видишь несколько независимых цепочек/инцидентов — скажи это явно.\n"
        "Если между событиями связь не доказана — помечай как гипотезу, не как факт.\n\n"
        "Если данных недостаточно для какого-то раздела — прямо напиши об этом.\n\n"
        f"{anti_block}"
        f"Период: [{period_start}, {period_end})\n"
        f"Метрики: {metrics_block}\n\n"
        "MAP summary по батчам логов:\n"
        f"{(map_summaries_text or 'Нет map-summary.')}\n\n"
        "Структурированный анализ логов:\n"
        f"{final_summary}"
        f"{chain_block}{incident_link_block}{root_cause_hypotheses_block}"
    )


def _build_sectional_freeform_prompt(
    *,
    section_index: int,
    section_total: int,
    section_title: str,
    section_requirement: str,
    previous_sections_text: str,
    final_summary: str,
    user_goal: str,
    period_start: str,
    period_end: str,
    stats: Dict[str, Any],
    metrics_context: str,
) -> str:
    goal_block = user_goal.strip() or "Не указан"
    metrics_block = metrics_context.strip() or "Нет доп. метрик в контексте."
    is_root_cause_section = "первоприч" in str(section_title).lower()
    incident_verbatim_block = (
        f"\nДОПОЛНИТЕЛЬНОЕ ТРЕБОВАНИЕ К ЭТОЙ СЕКЦИИ:\n"
        f"{_incident_verbatim_requirement_block(goal_block)}\n"
        if is_root_cause_section
        else ""
    )
    anti_rules = str(
        getattr(settings, "CONTROL_PLANE_LLM_ANTI_HALLUCINATION_RULES", "")
    ).strip()
    anti_block = f"\nПРАВИЛА АНТИГАЛЛЮЦИНАЦИИ:\n{anti_rules}\n" if anti_rules else ""
    custom_template = str(
        getattr(settings, "CONTROL_PLANE_LLM_UI_FINAL_REPORT_PROMPT_TEMPLATE", "")
    ).strip()
    custom_block = ""
    if custom_template:
        rendered_custom = _render_prompt_template(
            custom_template,
            {
                "final_summary": final_summary,
                "cross_source_summary": final_summary,
                "user_goal": goal_block,
                "incident_description": goal_block,
                "alerts_list": goal_block,
                "incident_start": period_start,
                "incident_end": period_end,
                "period_start": period_start,
                "period_end": period_end,
                "stats": json.dumps(_json_safe(stats), ensure_ascii=False, indent=2),
                "metrics_context": metrics_block,
                "previous_sections_text": previous_sections_text or "",
                "section_index": section_index,
                "section_total": section_total,
                "section_title": section_title,
                "section_requirement": section_requirement,
            },
        ).strip()
        if rendered_custom:
            custom_block = f"\nДОПОЛНИТЕЛЬНЫЕ ИНСТРУКЦИИ ИЗ ШАБЛОНА:\n{rendered_custom}\n"
    previous_block = previous_sections_text.strip() or "Пока секций нет."
    return (
        "Ты формируешь финальный отчёт по частям. За этот вызов напиши только одну секцию.\n\n"
        f"СЕКЦИЯ {section_index}/{section_total}: {section_title}\n"
        f"Требование секции: {section_requirement}\n\n"
        f"{incident_verbatim_block}"
        "Опирайся на уже написанные секции (если есть), не противоречь им, дополняй контекст.\n"
        "Не повторяй целиком старые секции, но используй их выводы как основу.\n"
        "Пиши максимально подробно и предметно.\n"
        "Если данных недостаточно — явно пометь это.\n\n"
        f"КОНТЕКСТ ИНЦИДЕНТА (UI):\n{goal_block}\n\n"
        f"Период: [{period_start}, {period_end})\n"
        f"Метрики: {metrics_block}\n"
        f"Статистика: {json.dumps(_json_safe(stats), ensure_ascii=False)}\n\n"
        "УЖЕ НАПИСАННЫЕ СЕКЦИИ:\n"
        f"{previous_block}\n\n"
        "СТРУКТУРИРОВАННЫЙ АНАЛИЗ (опорный контекст):\n"
        f"{final_summary}\n"
        f"{custom_block}"
        f"{anti_block}\n"
        "Верни только текст текущей секции, без префиксов вроде 'Секция N'."
    )


def _build_sectional_structured_prompt(
    *,
    section_index: int,
    section_total: int,
    section_title: str,
    section_requirement: str,
    previous_sections_text: str,
    base_summary: str,
    user_goal: str,
    period_start: str,
    period_end: str,
    stats: Dict[str, Any],
    metrics_context: str,
) -> str:
    goal_block = user_goal.strip() or "Не указан"
    metrics_block = metrics_context.strip() or "Нет доп. метрик в контексте."
    is_root_cause_section = "первоприч" in str(section_title).lower()
    incident_verbatim_block = (
        f"\nДОПОЛНИТЕЛЬНОЕ ТРЕБОВАНИЕ К ЭТОЙ СЕКЦИИ:\n"
        f"{_incident_verbatim_requirement_block(goal_block)}\n"
        if is_root_cause_section
        else ""
    )
    anti_rules = str(
        getattr(settings, "CONTROL_PLANE_LLM_ANTI_HALLUCINATION_RULES", "")
    ).strip()
    anti_block = f"\nПРАВИЛА АНТИГАЛЛЮЦИНАЦИИ:\n{anti_rules}\n" if anti_rules else ""
    previous_block = previous_sections_text.strip() or "Пока секций нет."
    return (
        "Ты формируешь СТРУКТУРИРОВАННЫЙ финальный отчёт по частям. За этот вызов напиши только одну секцию.\n\n"
        f"СЕКЦИЯ {section_index}/{section_total}: {section_title}\n"
        f"Требование секции: {section_requirement}\n\n"
        f"{incident_verbatim_block}"
        "Критично: опирайся на уже написанные секции, не противоречь им, наращивай контекст.\n"
        "Нужен технический стиль, конкретика, timestamps, факты/гипотезы.\n"
        "Если данных недостаточно — так и напиши.\n\n"
        f"КОНТЕКСТ ИНЦИДЕНТА (UI):\n{goal_block}\n\n"
        f"Период: [{period_start}, {period_end})\n"
        f"Метрики: {metrics_block}\n"
        f"Статистика: {json.dumps(_json_safe(stats), ensure_ascii=False)}\n\n"
        "УЖЕ НАПИСАННЫЕ СЕКЦИИ СТРУКТУРИРОВАННОГО ОТЧЁТА:\n"
        f"{previous_block}\n\n"
        "БАЗОВЫЙ REDUCE SUMMARY (опорный контекст):\n"
        f"{base_summary}\n"
        f"{anti_block}\n"
        "Верни только текст текущей секции, без префиксов вроде 'Секция N'."
    )


def _generate_sectional_structured_summary(
    *,
    llm_call: Callable[[str], str],
    base_summary: str,
    user_goal: str,
    period_start: str,
    period_end: str,
    stats: Dict[str, Any],
    metrics_context: str,
    on_section_start: Optional[Callable[[int, int, str], None]] = None,
    on_section_done: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[str, List[Dict[str, str]]]:
    sections: List[Dict[str, str]] = []
    total = len(FINAL_REPORT_SECTIONS)
    for idx, (title, requirement) in enumerate(FINAL_REPORT_SECTIONS, start=1):
        if on_section_start is not None:
            on_section_start(idx, total, title)
        previous_text = "\n\n".join(
            f"## {item['title']}\n{item['text']}" for item in sections if item.get("text")
        )
        section_text = _programmatic_section_text(
            section_title=title,
            user_goal=user_goal,
            metrics_context=metrics_context,
        )
        if section_text is None:
            prompt = _build_sectional_structured_prompt(
                section_index=idx,
                section_total=total,
                section_title=title,
                section_requirement=requirement,
                previous_sections_text=previous_text,
                base_summary=base_summary,
                user_goal=user_goal,
                period_start=period_start,
                period_end=period_end,
                stats=stats,
                metrics_context=metrics_context,
            )
            section_text = _normalize_summary_text(llm_call(prompt))
            if not section_text:
                section_text = "Данных недостаточно для уверенного вывода по этой секции."
        sections.append({"title": title, "text": section_text})
        if on_section_done is not None:
            on_section_done(idx, total, title)

    merged_lines: List[str] = []
    for item in sections:
        merged_lines.append(f"## {item['title']}")
        merged_lines.append("")
        merged_lines.append(item["text"])
        merged_lines.append("")
    return "\n".join(merged_lines).strip(), sections


def _generate_sectional_freeform_summary(
    *,
    llm_call: Callable[[str], str],
    final_summary: str,
    user_goal: str,
    period_start: str,
    period_end: str,
    stats: Dict[str, Any],
    metrics_context: str,
    on_section_start: Optional[Callable[[int, int, str], None]] = None,
    on_section_done: Optional[Callable[[int, int, str], None]] = None,
) -> tuple[str, List[Dict[str, str]]]:
    sections: List[Dict[str, str]] = []
    total = len(FINAL_REPORT_SECTIONS)
    for idx, (title, requirement) in enumerate(FINAL_REPORT_SECTIONS, start=1):
        if on_section_start is not None:
            on_section_start(idx, total, title)
        previous_text = "\n\n".join(
            f"## {item['title']}\n{item['text']}" for item in sections if item.get("text")
        )
        section_text = _programmatic_section_text(
            section_title=title,
            user_goal=user_goal,
            metrics_context=metrics_context,
        )
        if section_text is None:
            prompt = _build_sectional_freeform_prompt(
                section_index=idx,
                section_total=total,
                section_title=title,
                section_requirement=requirement,
                previous_sections_text=previous_text,
                final_summary=final_summary,
                user_goal=user_goal,
                period_start=period_start,
                period_end=period_end,
                stats=stats,
                metrics_context=metrics_context,
            )
            section_text = _normalize_summary_text(llm_call(prompt))
            if not section_text:
                section_text = "Данных недостаточно для уверенного вывода по этой секции."
        sections.append({"title": title, "text": section_text})
        if on_section_done is not None:
            on_section_done(idx, total, title)

    merged_lines: List[str] = []
    for item in sections:
        merged_lines.append(f"## {item['title']}")
        merged_lines.append("")
        merged_lines.append(item["text"])
        merged_lines.append("")
    return "\n".join(merged_lines).strip(), sections


def _build_no_logs_hypothesis_prompt(
    *,
    period_start: str,
    period_end: str,
    user_goal: str,
    metrics_context: str,
    logs_fetch_mode: str,
    logs_tail_limit: int,
    logs_queries_count: int,
) -> str:
    goal_block = user_goal.strip() or "Контекст инцидента не указан."
    metrics_block = metrics_context.strip() or "Метрики не предоставлены."
    anti_rules = str(
        getattr(settings, "CONTROL_PLANE_LLM_ANTI_HALLUCINATION_RULES", "")
    ).strip()
    anti_block = f"\nПРАВИЛА АНТИГАЛЛЮЦИНАЦИИ:\n{anti_rules}\n" if anti_rules else ""
    mode_hint = (
        f"режим tail_n_logs (limit={int(logs_tail_limit)})"
        if _normalize_logs_fetch_mode(logs_fetch_mode) == "tail_n_logs"
        else "режим выборки по временному окну"
    )
    return (
        "Ты senior SRE-аналитик.\n"
        "По указанному периоду логов в выборке нет, но нужно дать осторожное предварительное расследование.\n"
        "Не выдумывай факты. Явно разделяй [ФАКТ] и [ГИПОТЕЗА].\n"
        "Если данных мало — так и пиши.\n\n"
        f"Период анализа: [{period_start}, {period_end})\n"
        f"Режим получения логов: {mode_hint}\n"
        f"SQL источников логов: {int(logs_queries_count)}\n"
        f"Контекст инцидента/алертов:\n{goal_block}\n\n"
        f"Контекст метрик:\n{metrics_block}\n"
        f"{anti_block}\n"
        "Сформируй ответ строго в секциях:\n"
        "1) ЧТО ТОЧНО ИЗВЕСТНО [ФАКТ]\n"
        "2) ОСТОРОЖНЫЕ ГИПОТЕЗЫ (3-7 пунктов) [ГИПОТЕЗА]\n"
        "3) КАКИЕ ДАННЫЕ НУЖНЫ, ЧТОБЫ ПОДТВЕРДИТЬ/ОПРОВЕРГНУТЬ ГИПОТЕЗЫ\n"
        "4) ЧТО ПРОВЕРИТЬ SRE ПРЯМО СЕЙЧАС (приоритет P0/P1)\n"
        "5) СВЯЗЬ С ИНЦИДЕНТОМ ИЗ UI (по каждому пункту: [ОБЪЯСНЁН]/[ЧАСТИЧНО]/[НЕ ОБЪЯСНЁН])\n"
        "6) ПРЕДПОЛОЖЕНИЯ О ПЕРВОПРИЧИНЕ ПО КАЖДОМУ ИНЦИДЕНТУ (минимум 2 гипотезы на инцидент)\n"
        "7) КРАТКИЙ ВЫВОД ДЛЯ КОМАНДЫ\n"
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
        processing_seconds = pd.to_numeric(stats.get("logs_processing_seconds"), errors="coerce")
        processing_human = str(stats.get("logs_processing_human") or "").strip()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Статус", "Готово" if status == "done" else "Ошибка")
        col2.metric("Обработано логов", f"{rows_processed}")
        col3.metric("LLM вызовы", f"{llm_calls}")
        col4.metric("Reduce раунды", f"{reduce_rounds}")
        origin_label = _summary_origin_label(state.get("final_summary_origin"))
        if origin_label:
            st.caption(f"Источник итогового summary: `{origin_label}`")
        if processing_human:
            st.caption(f"Время обработки логов: {processing_human}")
        elif not pd.isna(processing_seconds):
            st.caption(f"Время обработки логов: {_format_eta_seconds(float(processing_seconds))}")

        _last_ts = _to_msk_ts(state.get("last_batch_ts"))
        _p_start = _to_msk_ts(state.get("period_start"))
        _p_end = _to_msk_ts(state.get("period_end"))
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
                f"(лог до {_last_ts.tz_convert(MSK).strftime('%H:%M:%S.%f MSK')} | строк: {rows_processed:,})"
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

        final_summary = _normalize_summary_text(state.get("final_summary"))
        freeform_summary = _normalize_summary_text(state.get("freeform_final_summary"))

        report_sections_map = _get_report_sections_map(state)
        if not report_sections_map:
            report_sections_map = _extract_sections_from_text(final_summary)
        timeline_rows = _collect_timeline_events(state)
        hypotheses_rows = _collect_hypotheses(state)
        causal_links_rows = _collect_causal_links(state)
        gaps_rows = _collect_gaps(state)
        conflicts_rows = _collect_conflicts(state)
        recommendations_rows = _collect_recommendations(state)
        alert_rows = _build_alert_panel_state(state)

        st.markdown("Интерактивный Отчёт")
        tab_summary, tab_timeline, tab_chains, tab_alerts, tab_hypotheses, tab_recs, tab_more = st.tabs(
            [
                "Резюме",
                "Хронология",
                "Цепочки",
                "Алерты",
                "Гипотезы",
                "Рекомендации",
                "Ещё",
            ]
        )

        with tab_summary:
            summary_text = (
                _find_section_text(report_sections_map, "2.")
                or _find_section_text(report_sections_map, "2 ")
                or final_summary
            )
            if summary_text:
                deps.render_pretty_summary_text(summary_text, height=max(260, min(final_height, 520)))
            else:
                st.info("Резюме пока недоступно.")

        with tab_timeline:
            if timeline_rows:
                timeline_df = _format_table_timestamps(pd.DataFrame(timeline_rows))
                severity_values = sorted(
                    {str(item.get("severity") or "").strip() for item in timeline_rows if str(item.get("severity") or "").strip()}
                )
                source_values = sorted(
                    {str(item.get("source") or "").strip() for item in timeline_rows if str(item.get("source") or "").strip()}
                )
                evidence_values = sorted(
                    {str(item.get("evidence_type") or "").strip() for item in timeline_rows if str(item.get("evidence_type") or "").strip()}
                )
                all_tags: List[str] = []
                for item in timeline_rows:
                    tags = item.get("tags")
                    if isinstance(tags, list):
                        for tag in tags:
                            text = str(tag).strip()
                            if text and text not in all_tags:
                                all_tags.append(text)
                filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
                selected_severity = filter_col1.multiselect(
                    "Severity",
                    severity_values,
                    default=severity_values,
                    key="logs_sum_report_filter_severity",
                )
                selected_sources = filter_col2.multiselect(
                    "Source",
                    source_values,
                    default=source_values,
                    key="logs_sum_report_filter_source",
                )
                selected_evidence = filter_col3.multiselect(
                    "Тип",
                    evidence_values,
                    default=evidence_values,
                    key="logs_sum_report_filter_evidence",
                )
                selected_tags = filter_col4.multiselect(
                    "Tags",
                    sorted(all_tags),
                    default=[],
                    key="logs_sum_report_filter_tags",
                )
                filtered = timeline_df.copy()
                if selected_severity and "severity" in filtered.columns:
                    filtered = filtered[filtered["severity"].astype(str).isin(selected_severity)]
                if selected_sources and "source" in filtered.columns:
                    filtered = filtered[filtered["source"].astype(str).isin(selected_sources)]
                if selected_evidence and "evidence_type" in filtered.columns:
                    filtered = filtered[filtered["evidence_type"].astype(str).isin(selected_evidence)]
                if selected_tags and "tags" in filtered.columns:
                    filtered = filtered[
                        filtered["tags"].apply(
                            lambda raw: any(
                                tag in selected_tags
                                for tag in (raw if isinstance(raw, list) else [raw])
                            )
                        )
                    ]
                keep_cols = [
                    col
                    for col in [
                        "timestamp",
                        "source",
                        "description",
                        "severity",
                        "evidence_type",
                        "importance",
                        "tags",
                        "id",
                        "batch_index",
                    ]
                    if col in filtered.columns
                ]
                st.dataframe(
                    filtered[keep_cols] if keep_cols else filtered,
                    use_container_width=True,
                    hide_index=True,
                    height=max(320, deps.logs_batch_table_height),
                )
            else:
                body = _find_section_text(report_sections_map, "4.")
                if body:
                    deps.render_pretty_summary_text(body, height=max(260, min(final_height, 520)))
                else:
                    st.info("Хронология пока недоступна.")

        with tab_chains:
            dot_graph = _build_causal_graph_dot(events=timeline_rows, links=causal_links_rows)
            mode = st.radio(
                "Режим отображения",
                options=("Граф", "Текст"),
                horizontal=True,
                key="logs_sum_chains_mode",
            )
            if mode == "Граф":
                if dot_graph:
                    st.graphviz_chart(dot_graph, use_container_width=True)
                else:
                    st.info("Граф причинно-следственных связей пока пуст.")
            else:
                body = _find_section_text(report_sections_map, "5.")
                if body:
                    deps.render_pretty_summary_text(body, height=max(260, min(final_height, 520)))
                else:
                    st.info("Текстовый блок цепочек пока недоступен.")
            if causal_links_rows:
                with st.expander("Детали causal links", expanded=False):
                    links_df = pd.DataFrame(causal_links_rows)
                    keep_cols = [c for c in ["id", "cause_event_id", "effect_event_id", "mechanism", "confidence"] if c in links_df.columns]
                    st.dataframe(
                        links_df[keep_cols] if keep_cols else links_df,
                        use_container_width=True,
                        hide_index=True,
                        height=220,
                    )

        with tab_alerts:
            if alert_rows:
                for row in alert_rows:
                    view = ALERT_STATUS_VIEW.get(row.get("status", ""), ALERT_STATUS_VIEW["NOT_SEEN_IN_BATCH"])
                    st.markdown(
                        (
                            "<div style='padding:0.6rem 0.8rem;border:1px solid #e5e7eb;border-radius:0.55rem;"
                            f"border-left:6px solid {view['color']};margin-bottom:0.5rem;'>"
                            f"<span style='color:{view['color']};font-weight:700;'>{view['icon']}</span> "
                            f"<strong>{row['alert_id']}</strong> — {view['label']}"
                            f"{' | ' + row['details'] if row.get('details') else ''}"
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )
                    with st.expander(f"Детали: {row['alert_id']}", expanded=False):
                        hist = row.get("history") or []
                        if hist:
                            hist_df = pd.DataFrame(hist)
                            keep_cols = [c for c in ["batch_index", "status", "related_events", "explanation"] if c in hist_df.columns]
                            st.dataframe(
                                hist_df[keep_cols] if keep_cols else hist_df,
                                use_container_width=True,
                                hide_index=True,
                                height=180,
                            )
                        if row.get("related_events"):
                            st.caption("Связанные события: " + ", ".join([str(x) for x in row["related_events"]]))
                        if row.get("explanation"):
                            deps.render_scrollable_text(str(row["explanation"]), height=140)
            else:
                body = _find_section_text(report_sections_map, "6.")
                if body:
                    deps.render_pretty_summary_text(body, height=max(260, min(final_height, 520)))
                else:
                    st.info("Данные по алертам пока недоступны.")

        with tab_hypotheses:
            if hypotheses_rows:
                grouped: Dict[str, List[Dict[str, Any]]] = {}
                for hyp in hypotheses_rows:
                    related = hyp.get("related_alert_ids")
                    if isinstance(related, list) and related:
                        key = str(related[0])
                    else:
                        key = "unbound"
                    grouped.setdefault(key, []).append(hyp)
                for alert_id, items in grouped.items():
                    st.markdown(f"**{alert_id}**")
                    items_sorted = sorted(
                        items,
                        key=lambda item: _safe_float(item.get("confidence"), 0.0),
                        reverse=True,
                    )
                    for hyp in items_sorted:
                        conf = min(max(_safe_float(hyp.get("confidence"), 0.0), 0.0), 1.0)
                        st.progress(conf, text=f"{hyp.get('title', 'Гипотеза')} ({conf:.2f})")
                        description = _normalize_summary_text(hyp.get("description"))
                        if description:
                            st.caption(description)
            else:
                body = _find_section_text(report_sections_map, "8.")
                if body:
                    deps.render_pretty_summary_text(body, height=max(260, min(final_height, 520)))
                else:
                    st.info("Гипотезы пока недоступны.")

        with tab_recs:
            grouped = _group_recommendations_by_priority(recommendations_rows)
            priority_styles = {
                "P0": "#b91c1c",
                "P1": "#a16207",
                "P2": "#4b5563",
            }
            has_structured_recs = any(grouped.values())
            if has_structured_recs:
                for priority in ("P0", "P1", "P2"):
                    items = grouped.get(priority) or []
                    st.markdown(
                        f"<h4 style='margin-bottom:0.4rem;color:{priority_styles[priority]};'>{priority}</h4>",
                        unsafe_allow_html=True,
                    )
                    if not items:
                        st.caption("—")
                        continue
                    for item in items:
                        action = str(item.get("action") or "Действие не указано").strip()
                        rationale = str(item.get("rationale") or "").strip()
                        st.markdown(f"- **{action}**")
                        if rationale:
                            st.caption(rationale)
            else:
                body = _find_section_text(report_sections_map, "12.")
                if body:
                    deps.render_pretty_summary_text(body, height=max(260, min(final_height, 520)))
                else:
                    st.info("Рекомендации пока недоступны.")

        with tab_more:
            subtab_metrics, subtab_conflicts, subtab_gaps, subtab_scale, subtab_limits, subtab_coverage, subtab_context = st.tabs(
                [
                    "Метрики",
                    "Конфликты",
                    "Разрывы",
                    "Масштаб",
                    "Ограничения",
                    "Покрытие",
                    "Контекст",
                ]
            )
            with subtab_metrics:
                body = _find_section_text(report_sections_map, "7.")
                if body:
                    deps.render_pretty_summary_text(body, height=max(240, min(final_height, 420)))
                else:
                    st.info("Раздел метрик пока недоступен.")
            with subtab_conflicts:
                if conflicts_rows:
                    st.dataframe(pd.DataFrame(conflicts_rows), use_container_width=True, hide_index=True, height=240)
                else:
                    body = _find_section_text(report_sections_map, "9.")
                    if body:
                        deps.render_pretty_summary_text(body, height=max(240, min(final_height, 420)))
                    else:
                        st.info("Конфликтующих интерпретаций не обнаружено.")
            with subtab_gaps:
                if gaps_rows:
                    st.dataframe(pd.DataFrame(gaps_rows), use_container_width=True, hide_index=True, height=240)
                else:
                    body = _find_section_text(report_sections_map, "10.")
                    if body:
                        deps.render_pretty_summary_text(body, height=max(240, min(final_height, 420)))
                    else:
                        st.info("Разрывы не зафиксированы.")
            with subtab_scale:
                body = _find_section_text(report_sections_map, "11.")
                if body:
                    deps.render_pretty_summary_text(body, height=max(240, min(final_height, 420)))
                else:
                    st.info("Раздел масштаба пока недоступен.")
            with subtab_limits:
                body = _find_section_text(report_sections_map, "13.")
                if body:
                    deps.render_pretty_summary_text(body, height=max(240, min(final_height, 420)))
                else:
                    st.info("Ограничения анализа пока недоступны.")
            with subtab_coverage:
                body = _find_section_text(report_sections_map, "3.")
                if body:
                    deps.render_pretty_summary_text(body, height=max(240, min(final_height, 420)))
                else:
                    st.info("Раздел покрытия данных пока недоступен.")
            with subtab_context:
                body = _find_section_text(report_sections_map, "1.")
                if body:
                    deps.render_pretty_summary_text(body, height=max(240, min(final_height, 420)))
                else:
                    st.info("Контекст инцидента не найден.")

        with st.expander("Полные тексты отчетов", expanded=False):
            if final_summary:
                st.markdown("Итоговое расследование (структурированное)")
                deps.render_scrollable_text(final_summary, height=max(final_height, 520))
            if freeform_summary:
                st.markdown("Итоговое расследование в свободном формате")
                deps.render_scrollable_text(freeform_summary, height=max(final_height, 620))

        structured_hypotheses = _extract_root_cause_hypotheses_block(final_summary)
        freeform_hypotheses = _extract_root_cause_hypotheses_block(freeform_summary)
        if structured_hypotheses or freeform_hypotheses:
            st.markdown("Гипотезы Первопричин По Инцидентам")
            if structured_hypotheses:
                st.caption("Источник: структурированный отчет")
                deps.render_pretty_summary_text(
                    structured_hypotheses,
                    height=max(260, min(final_height, 520)),
                )
            if freeform_hypotheses:
                st.caption("Источник: свободный отчет")
                deps.render_pretty_summary_text(
                    freeform_hypotheses,
                    height=max(260, min(final_height, 520)),
                )
        elif final_summary or freeform_summary:
            st.info(
                "Отдельный блок гипотез первопричин в тексте не найден. "
                "Попросите пересобрать отчет — блок добавится автоматически."
            )

        map_summaries_path = str(state.get("map_summaries_jsonl_path") or "").strip()
        if map_summaries_path:
            st.markdown("Переиспользование Summary")
            rebuild_cols = st.columns([1, 1, 1, 1])
            with rebuild_cols[0]:
                rebuild_report_only = st.button(
                    "Пересобрать Только Отчёт",
                    key=f"logs_sum_rebuild_report_only_{str(state.get('started_at') or 'na')}",
                    use_container_width=True,
                )
            with rebuild_cols[1]:
                rebuild_reduce_clicked = st.button(
                    "Пересобрать От Reduce",
                    key=f"logs_sum_rereduce_{str(state.get('started_at') or 'na')}",
                    use_container_width=True,
                )
            reduce_nodes_for_select = state.get("reduce_nodes")
            if not isinstance(reduce_nodes_for_select, list):
                reduce_nodes_for_select = []
            node_options = [
                f"R{_safe_int(node.get('round'), 0)}-G{_safe_int(node.get('group'), 0)}"
                for node in reduce_nodes_for_select
                if isinstance(node, dict)
            ] or ["n/a"]
            with rebuild_cols[2]:
                selected_node = st.selectbox(
                    "Узел",
                    options=node_options,
                    key=f"logs_sum_reduce_node_select_{str(state.get('started_at') or 'na')}",
                )
                rebuild_from_node_clicked = st.button(
                    "Пересобрать От Узла",
                    key=f"logs_sum_rereduce_node_{str(state.get('started_at') or 'na')}",
                    use_container_width=True,
                    disabled=(selected_node == "n/a"),
                )
            with rebuild_cols[3]:
                add_data_clicked = st.button(
                    "Добавить Данные",
                    key=f"logs_sum_add_data_mode_{str(state.get('started_at') or 'na')}",
                    use_container_width=True,
                )

            if add_data_clicked:
                st.info(
                    "Режим 'Добавить данные': обновите период/SQL в форме слева и запустите новый прогон. "
                    "Существующие артефакты сессии останутся доступными."
                )

            if rebuild_report_only:
                try:
                    retries_raw = pd.to_numeric(state.get("max_retries"), errors="coerce")
                    retries_value = -1 if pd.isna(retries_raw) else int(retries_raw)
                    timeout_raw = pd.to_numeric(state.get("llm_timeout"), errors="coerce")
                    timeout_value = int(deps.llm_timeout) if pd.isna(timeout_raw) else int(timeout_raw)
                    llm_call = deps.make_llm_call(
                        max_retries=retries_value,
                        llm_timeout=max(timeout_value, 10),
                    )
                    base_summary = _normalize_summary_text(state.get("final_summary"))
                    if not base_summary:
                        st.warning("Нет итогового summary для пересборки отчёта.")
                    else:
                        with st.spinner("Пересобираем structured/freeform отчёт из текущего summary..."):
                            rebuilt_structured, rebuilt_structured_sections = _generate_sectional_structured_summary(
                                llm_call=llm_call,
                                base_summary=base_summary,
                                user_goal=str(state.get("user_goal") or ""),
                                period_start=str(state.get("period_start") or ""),
                                period_end=str(state.get("period_end") or ""),
                                stats=state.get("stats") or {},
                                metrics_context=str(state.get("metrics_context_text") or ""),
                            )
                            rebuilt_freeform, rebuilt_freeform_sections = _generate_sectional_freeform_summary(
                                llm_call=llm_call,
                                final_summary=_normalize_summary_text(rebuilt_structured) or base_summary,
                                user_goal=str(state.get("user_goal") or ""),
                                period_start=str(state.get("period_start") or ""),
                                period_end=str(state.get("period_end") or ""),
                                stats=state.get("stats") or {},
                                metrics_context=str(state.get("metrics_context_text") or ""),
                            )
                        rebuilt_structured = _normalize_summary_text(rebuilt_structured) or base_summary
                        rebuilt_freeform = _normalize_summary_text(rebuilt_freeform)
                        state["final_summary"] = rebuilt_structured
                        state["structured_sections"] = rebuilt_structured_sections
                        if rebuilt_freeform:
                            state["freeform_final_summary"] = rebuilt_freeform
                            state["freeform_sections"] = rebuilt_freeform_sections
                        state.setdefault("events", []).append("Режим: пересборка только финального отчёта")
                        request_payload_for_save = {}
                        request_path = Path(str(state.get("request_path") or ""))
                        loaded_request = _read_json_file(request_path)
                        if isinstance(loaded_request, dict):
                            request_payload_for_save = dict(loaded_request)
                        saved_after_rebuild = _save_logs_summary_result(
                            output_dir=deps.output_dir,
                            request_payload=request_payload_for_save,
                            result_state=state,
                        )
                        state["result_json_path"] = saved_after_rebuild.get("json_path")
                        state["result_bundle_path"] = saved_after_rebuild.get("bundle_path")
                        state["result_summary_path"] = saved_after_rebuild.get("summary_path")
                        state["result_html_path"] = saved_after_rebuild.get("html_path")
                        state["result_structured_md_path"] = saved_after_rebuild.get("structured_md_path")
                        state["result_freeform_md_path"] = saved_after_rebuild.get("freeform_md_path")
                        state["result_structured_txt_path"] = saved_after_rebuild.get("structured_txt_path")
                        state["result_freeform_txt_path"] = saved_after_rebuild.get("freeform_txt_path")
                        checkpoint_raw = str(state.get("checkpoint_path") or "").strip()
                        if checkpoint_raw:
                            _persist_checkpoint(Path(checkpoint_raw), state)
                        st.session_state[LAST_STATE_SESSION_KEY] = state
                        st.success("Пересборка отчёта завершена.")
                        st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Не удалось пересобрать отчёт: {exc}")

            if rebuild_from_node_clicked and selected_node != "n/a":
                state.setdefault("events", []).append(
                    f"Ручной re-reduce запрошен от узла {selected_node} (пересборка с map-summary)."
                )
                rebuild_reduce_clicked = True

            if rebuild_reduce_clicked:
                try:
                    cached_map_summaries = _load_map_summaries_from_jsonl(map_summaries_path)
                    if not cached_map_summaries:
                        st.warning("В файле MAP summary нет данных для пересборки.")
                    else:
                        with st.spinner("Пересобираем финальный REDUCE из сохранённых MAP summary..."):
                            from my_summarizer import (  # noqa: PLC0415
                                regenerate_reduce_summary_from_map_summaries,
                            )

                            retries_raw = pd.to_numeric(
                                state.get("max_retries"), errors="coerce"
                            )
                            retries_value = -1 if pd.isna(retries_raw) else int(retries_raw)
                            timeout_raw = pd.to_numeric(
                                state.get("llm_timeout"), errors="coerce"
                            )
                            timeout_value = (
                                int(deps.llm_timeout)
                                if pd.isna(timeout_raw)
                                else int(timeout_raw)
                            )
                            llm_call = deps.make_llm_call(
                                max_retries=retries_value,
                                llm_timeout=max(timeout_value, 10),
                            )
                            rebuilt_summary = regenerate_reduce_summary_from_map_summaries(
                                map_summaries=cached_map_summaries,
                                period_start=str(state.get("period_start") or ""),
                                period_end=str(state.get("period_end") or ""),
                                llm_call=llm_call,
                                config=_build_config(
                                    deps,
                                    int(
                                        pd.to_numeric(
                                            state.get("db_batch_size"), errors="coerce"
                                        )
                                        or deps.db_batch_size
                                    ),
                                    int(
                                        pd.to_numeric(
                                            state.get("llm_batch_size"), errors="coerce"
                                        )
                                        or deps.llm_batch_size
                                    ),
                                    int(
                                        pd.to_numeric(
                                            state.get("map_workers"), errors="coerce"
                                        )
                                        or deps.map_workers
                                    ),
                                ),
                                prompt_context={
                                    "incident_start": str(state.get("period_start") or ""),
                                    "incident_end": str(state.get("period_end") or ""),
                                    "incident_description": str(state.get("user_goal") or ""),
                                    "alerts_list": str(state.get("user_goal") or ""),
                                    "metrics_context": str(state.get("metrics_context_text") or ""),
                                    "source_name": "cached_map_summaries",
                                    "sql_query": str(state.get("query") or ""),
                                    "time_column": str(state.get("logs_timestamp_column") or "timestamp"),
                                    "data_type": "",
                                },
                            )
                        rebuilt_summary = _normalize_summary_text(rebuilt_summary)
                        if rebuilt_summary:
                            state["final_summary"] = rebuilt_summary
                            state["final_summary_origin"] = "manual_rereduce"
                            state["structured_sections"] = []
                            state["freeform_sections"] = []
                            state.setdefault("events", []).append(
                                "Итоговый Reduce summary пересобран из сохранённых MAP summary"
                            )
                            try:
                                state.setdefault("events", []).append(
                                    "Пересборка structured: пишем структурированный отчет по секциям"
                                )
                                rebuilt_structured, rebuilt_structured_sections = _generate_sectional_structured_summary(
                                    llm_call=llm_call,
                                    base_summary=rebuilt_summary,
                                    user_goal=str(state.get("user_goal") or ""),
                                    period_start=str(state.get("period_start") or ""),
                                    period_end=str(state.get("period_end") or ""),
                                    stats=state.get("stats") or {},
                                    metrics_context=str(state.get("metrics_context_text") or ""),
                                )
                                rebuilt_structured = _normalize_summary_text(rebuilt_structured)
                                if rebuilt_structured:
                                    state["final_summary"] = rebuilt_structured
                                    state["structured_sections"] = rebuilt_structured_sections
                                    rebuilt_summary = rebuilt_structured
                                    state.setdefault("events", []).append(
                                        "Структурированный отчет пересобран из обновлённого итогового summary"
                                    )
                                else:
                                    state.setdefault("events", []).append(
                                        "Структурированный отчет при пересборке пустой, оставили предыдущую версию"
                                    )
                            except Exception as structured_exc:  # noqa: BLE001
                                deps.logger.warning(
                                    "manual re-reduce structured regeneration failed: %s",
                                    structured_exc,
                                )
                                state.setdefault("events", []).append(
                                    "Не удалось пересобрать структурированный отчет при manual re-reduce"
                                )
                            try:
                                state.setdefault("events", []).append(
                                    "Пересборка freeform: пишем финальный отчет по секциям"
                                )
                                rebuilt_freeform, rebuilt_sections = _generate_sectional_freeform_summary(
                                    llm_call=llm_call,
                                    final_summary=rebuilt_summary,
                                    user_goal=str(state.get("user_goal") or ""),
                                    period_start=str(state.get("period_start") or ""),
                                    period_end=str(state.get("period_end") or ""),
                                    stats=state.get("stats") or {},
                                    metrics_context=str(state.get("metrics_context_text") or ""),
                                )
                                rebuilt_freeform = _normalize_summary_text(rebuilt_freeform)
                                if rebuilt_freeform:
                                    state["freeform_final_summary"] = rebuilt_freeform
                                    state["freeform_sections"] = rebuilt_sections
                                    state.setdefault("events", []).append(
                                        "Свободный отчет пересобран из обновлённого итогового summary"
                                    )
                                else:
                                    state.setdefault("events", []).append(
                                        "Свободный отчет при пересборке пустой, оставили предыдущую версию"
                                    )
                            except Exception as freeform_exc:  # noqa: BLE001
                                deps.logger.warning(
                                    "manual re-reduce freeform regeneration failed: %s",
                                    freeform_exc,
                                )
                                state.setdefault("events", []).append(
                                    "Не удалось пересобрать свободный отчет при manual re-reduce"
                                )
                            topic_titles = [title for title, _ in FINAL_REPORT_SECTIONS]
                            preferred_structured_sections = (
                                state.get("structured_sections")
                                if isinstance(state.get("structured_sections"), list)
                                else (
                                    state.get("freeform_sections")
                                    if isinstance(state.get("freeform_sections"), list)
                                    else None
                                )
                            )
                            preferred_freeform_sections = (
                                state.get("freeform_sections")
                                if isinstance(state.get("freeform_sections"), list)
                                else (
                                    state.get("structured_sections")
                                    if isinstance(state.get("structured_sections"), list)
                                    else None
                                )
                            )
                            synced_structured, missing_structured = _ensure_report_topics_present(
                                str(state.get("final_summary") or ""),
                                topic_titles=topic_titles,
                                preferred_sections=preferred_structured_sections,
                            )
                            if synced_structured:
                                state["final_summary"] = synced_structured
                            synced_freeform, missing_freeform = _ensure_report_topics_present(
                                str(state.get("freeform_final_summary") or ""),
                                topic_titles=topic_titles,
                                preferred_sections=preferred_freeform_sections,
                            )
                            if synced_freeform:
                                state["freeform_final_summary"] = synced_freeform
                            if missing_structured:
                                state.setdefault("events", []).append(
                                    "Manual re-reduce: добавили недостающие топики в структурированный отчет"
                                )
                            if missing_freeform:
                                state.setdefault("events", []).append(
                                    "Manual re-reduce: добавили недостающие топики в свободный отчет"
                                )
                            rebuild_path = (
                                Path(map_summaries_path).parent
                                / f"rebuild_reduce_{datetime.now(MSK).strftime('%Y%m%d_%H%M%S')}.md"
                            )
                            _write_text_file(
                                rebuild_path,
                                "\n".join(
                                    [
                                        "# Rebuilt Reduce Summary",
                                        f"- rebuilt_at: `{datetime.now(MSK).isoformat()}`",
                                        f"- map_summaries: `{len(cached_map_summaries)}`",
                                        "",
                                        rebuilt_summary,
                                    ]
                                ),
                            )
                            state["rebuild_reduce_path"] = str(rebuild_path)
                            request_payload_for_save = {}
                            try:
                                request_path = Path(str(state.get("request_path") or ""))
                                loaded_request = _read_json_file(request_path)
                                if isinstance(loaded_request, dict):
                                    request_payload_for_save = dict(loaded_request)
                            except Exception:
                                request_payload_for_save = {}
                            saved_after_rebuild = _save_logs_summary_result(
                                output_dir=deps.output_dir,
                                request_payload=request_payload_for_save,
                                result_state=state,
                            )
                            state["result_json_path"] = saved_after_rebuild.get("json_path")
                            state["result_bundle_path"] = saved_after_rebuild.get("bundle_path")
                            state["result_summary_path"] = saved_after_rebuild.get("summary_path")
                            state["result_html_path"] = saved_after_rebuild.get("html_path")
                            state["result_structured_md_path"] = saved_after_rebuild.get("structured_md_path")
                            state["result_freeform_md_path"] = saved_after_rebuild.get("freeform_md_path")
                            state["result_structured_txt_path"] = saved_after_rebuild.get("structured_txt_path")
                            state["result_freeform_txt_path"] = saved_after_rebuild.get("freeform_txt_path")
                            checkpoint_raw = str(state.get("checkpoint_path") or "").strip()
                            if checkpoint_raw:
                                _persist_checkpoint(Path(checkpoint_raw), state)
                            st.session_state[LAST_STATE_SESSION_KEY] = state
                            st.success("Пересборка завершена. Обновили итоговый summary.")
                            st.rerun()
                        else:
                            st.warning("LLM вернул пустой результат при пересборке.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Не удалось пересобрать итог: {exc}")

        query_errors = state.get("query_errors", [])
        if isinstance(query_errors, list) and query_errors:
            with st.expander("Ошибки ClickHouse (запросы, которые были пропущены)", expanded=False):
                for err in query_errors:
                    st.warning(str(err))

        st.markdown("Артефакты")
        zip_bytes = _build_zip_artifacts_bytes(state)
        if zip_bytes is not None:
            st.download_button(
                label="Скачать Всё (ZIP)",
                data=zip_bytes,
                file_name=f"logs_summary_{datetime.now(MSK).strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
                use_container_width=True,
            )

        json_bytes = _read_file_bytes(state.get("result_json_path"))
        bundle_bytes = _read_file_bytes(state.get("result_bundle_path"))
        md_bytes = _read_file_bytes(state.get("result_summary_path"))
        html_bytes = _read_file_bytes(state.get("result_html_path"))
        structured_md_bytes = _read_file_bytes(state.get("result_structured_md_path"))
        freeform_md_bytes = _read_file_bytes(state.get("result_freeform_md_path"))
        structured_txt_bytes = _read_file_bytes(state.get("result_structured_txt_path"))
        freeform_txt_bytes = _read_file_bytes(state.get("result_freeform_txt_path"))
        map_jsonl_bytes = _read_file_bytes(state.get("map_summaries_jsonl_path"))
        reduce_jsonl_bytes = _read_file_bytes(state.get("reduce_summaries_jsonl_path"))
        llm_jsonl_bytes = _read_file_bytes(state.get("llm_calls_jsonl_path"))
        rebuild_md_bytes = _read_file_bytes(state.get("rebuild_reduce_path"))
        run_params_bytes = _read_file_bytes(state.get("run_params_path"))
        request_bytes = _read_file_bytes(state.get("request_path"))
        checkpoint_bytes = _read_file_bytes(state.get("checkpoint_path"))
        live_events_bytes = _read_file_bytes(state.get("live_events_path"))
        live_batches_bytes = _read_file_bytes(state.get("live_batches_path"))

        artifact_tabs = st.tabs(
            [
                "Финальный Отчёт",
                "Финальный Summary",
                "MAP Summaries",
                "Reduce Дерево",
                "Чекпоинты",
            ]
        )
        with artifact_tabs[0]:
            with st.expander("Что в этих файлах", expanded=False):
                st.markdown(
                    "\n".join(
                        [
                            "- `report.bundle.json (portable)` — единый переносимый файл: входные параметры + полный итоговый отчёт; можно импортировать в UI у коллег.",
                            "- `report.json` — полный технический dump запроса и результата запуска.",
                            "- `report.md (общий)` — markdown с обоими версиями отчёта и stats.",
                            "- `report.html (интерактивный)` — интерактивная страница, похожая на финальный UI.",
                            "- `structured.md` — только структурированный итоговый отчёт.",
                            "- `freeform.md` — только свободная narrative-версия отчёта.",
                        ]
                    )
                )
            if json_bytes is not None and state.get("result_json_path"):
                st.download_button(
                    label="Скачать report.json",
                    data=json_bytes,
                    file_name=Path(str(state.get("result_json_path"))).name,
                    mime="application/json",
                    use_container_width=True,
                )
            if bundle_bytes is not None and state.get("result_bundle_path"):
                st.download_button(
                    label="Скачать report.bundle.json (portable)",
                    data=bundle_bytes,
                    file_name=Path(str(state.get("result_bundle_path"))).name,
                    mime="application/json",
                    use_container_width=True,
                )
            if md_bytes is not None and state.get("result_summary_path"):
                st.download_button(
                    label="Скачать report.md (общий)",
                    data=md_bytes,
                    file_name=Path(str(state.get("result_summary_path"))).name,
                    mime="text/markdown",
                    use_container_width=True,
                )
            if html_bytes is not None and state.get("result_html_path"):
                st.download_button(
                    label="Скачать report.html (интерактивный)",
                    data=html_bytes,
                    file_name=Path(str(state.get("result_html_path"))).name,
                    mime="text/html",
                    use_container_width=True,
                )
            if structured_md_bytes is not None and state.get("result_structured_md_path"):
                st.download_button(
                    label="Скачать structured.md",
                    data=structured_md_bytes,
                    file_name=Path(str(state.get("result_structured_md_path"))).name,
                    mime="text/markdown",
                    use_container_width=True,
                )
            if freeform_md_bytes is not None and state.get("result_freeform_md_path"):
                st.download_button(
                    label="Скачать freeform.md",
                    data=freeform_md_bytes,
                    file_name=Path(str(state.get("result_freeform_md_path"))).name,
                    mime="text/markdown",
                    use_container_width=True,
                )
        with artifact_tabs[1]:
            with st.expander("Что в этих файлах", expanded=False):
                st.markdown(
                    "\n".join(
                        [
                            "- `structured.txt` — plain text структурированного отчёта (без markdown).",
                            "- `freeform.txt` — plain text свободного narrative-отчёта.",
                            "- `rebuild_reduce.md` — отчёт пересборки reduce (если запускалась пересборка).",
                        ]
                    )
                )
            if structured_txt_bytes is not None and state.get("result_structured_txt_path"):
                st.download_button(
                    label="Скачать structured.txt",
                    data=structured_txt_bytes,
                    file_name=Path(str(state.get("result_structured_txt_path"))).name,
                    mime="text/plain",
                    use_container_width=True,
                )
            if freeform_txt_bytes is not None and state.get("result_freeform_txt_path"):
                st.download_button(
                    label="Скачать freeform.txt",
                    data=freeform_txt_bytes,
                    file_name=Path(str(state.get("result_freeform_txt_path"))).name,
                    mime="text/plain",
                    use_container_width=True,
                )
            if rebuild_md_bytes is not None and state.get("rebuild_reduce_path"):
                st.download_button(
                    label="Скачать rebuild_reduce.md",
                    data=rebuild_md_bytes,
                    file_name=Path(str(state.get("rebuild_reduce_path"))).name,
                    mime="text/markdown",
                    use_container_width=True,
                )
        with artifact_tabs[2]:
            with st.expander("Что в этих файлах", expanded=False):
                st.markdown(
                    "\n".join(
                        [
                            "- `map_summaries.jsonl` — все MAP-summary по батчам (по одной записи на строку).",
                            "- `batches.jsonl` — live-лента обработанных батчей и их метаданных.",
                        ]
                    )
                )
            if map_jsonl_bytes is not None and state.get("map_summaries_jsonl_path"):
                st.download_button(
                    label="Скачать map_summaries.jsonl",
                    data=map_jsonl_bytes,
                    file_name=Path(str(state.get("map_summaries_jsonl_path"))).name,
                    mime="application/json",
                    use_container_width=True,
                )
            if live_batches_bytes is not None and state.get("live_batches_path"):
                st.download_button(
                    label="Скачать batches.jsonl",
                    data=live_batches_bytes,
                    file_name=Path(str(state.get("live_batches_path"))).name,
                    mime="application/json",
                    use_container_width=True,
                )
        with artifact_tabs[3]:
            with st.expander("Что в этих файлах", expanded=False):
                st.markdown(
                    "\n".join(
                        [
                            "- `reduce_summaries.jsonl` — промежуточные и финальные summary reduce-этапа.",
                            "- `llm_calls.jsonl` — журнал LLM-вызовов (попытки, ошибки, retry и т.д.).",
                        ]
                    )
                )
            if reduce_jsonl_bytes is not None and state.get("reduce_summaries_jsonl_path"):
                st.download_button(
                    label="Скачать reduce_summaries.jsonl",
                    data=reduce_jsonl_bytes,
                    file_name=Path(str(state.get("reduce_summaries_jsonl_path"))).name,
                    mime="application/json",
                    use_container_width=True,
                )
            if llm_jsonl_bytes is not None and state.get("llm_calls_jsonl_path"):
                st.download_button(
                    label="Скачать llm_calls.jsonl",
                    data=llm_jsonl_bytes,
                    file_name=Path(str(state.get("llm_calls_jsonl_path"))).name,
                    mime="application/json",
                    use_container_width=True,
                )
        with artifact_tabs[4]:
            with st.expander("Что в этих файлах", expanded=False):
                st.markdown(
                    "\n".join(
                        [
                            "- `run_params.json` — параметры запуска, зафиксированные перед стартом.",
                            "- `request.json` — собранный runtime-request (периоды, SQL, режимы и т.д.).",
                            "- `checkpoint.json` — состояние пайплайна для resume.",
                            "- `events.jsonl` — живой журнал событий выполнения.",
                        ]
                    )
                )
            if run_params_bytes is not None and state.get("run_params_path"):
                st.download_button(
                    label="Скачать run_params.json",
                    data=run_params_bytes,
                    file_name=Path(str(state.get("run_params_path"))).name,
                    mime="application/json",
                    use_container_width=True,
                )
            if request_bytes is not None and state.get("request_path"):
                st.download_button(
                    label="Скачать request.json",
                    data=request_bytes,
                    file_name=Path(str(state.get("request_path"))).name,
                    mime="application/json",
                    use_container_width=True,
                )
            if checkpoint_bytes is not None and state.get("checkpoint_path"):
                st.download_button(
                    label="Скачать checkpoint.json",
                    data=checkpoint_bytes,
                    file_name=Path(str(state.get("checkpoint_path"))).name,
                    mime="application/json",
                    use_container_width=True,
                )
            if live_events_bytes is not None and state.get("live_events_path"):
                st.download_button(
                    label="Скачать events.jsonl",
                    data=live_events_bytes,
                    file_name=Path(str(state.get("live_events_path"))).name,
                    mime="application/json",
                    use_container_width=True,
                )


def _render_logs_summary_chat(container, state: Dict[str, Any], deps: LogsSummaryPageDeps) -> None:
    with container.container():
        st.markdown("1. Pipeline")
        if not state:
            return

        map_batches = state.get("map_batches")
        if not isinstance(map_batches, list):
            map_batches = []
        status = str(state.get("status", "queued")).strip().lower()

        # --- Alerts panel (always on top) ---
        st.markdown("### Панель Статуса Алертов")
        alert_rows = _build_alert_panel_state(state)
        done_batches, batch_total = _resolve_map_batches_progress(state, map_batches)
        if batch_total > 0:
            st.caption(f"Батчи: {done_batches}/{batch_total} обработано")
        elif done_batches > 0:
            st.caption(f"Батчи: {done_batches} обработано")
        else:
            st.caption("Батчи: ожидание старта")
        if alert_rows:
            previous = state.setdefault("_ui_prev_alert_status", {})
            for row in alert_rows:
                view = ALERT_STATUS_VIEW.get(row["status"], ALERT_STATUS_VIEW["NOT_SEEN_IN_BATCH"])
                line = (
                    f"{view['icon']} {row['alert_id']}: {view['label']}"
                    + (f" — {row['details']}" if row["details"] else "")
                )
                st.markdown(
                    (
                        "<div style='padding:0.4rem 0.6rem;border:1px solid #e5e7eb;"
                        f"border-left:5px solid {view['color']};border-radius:0.5rem;"
                        "margin-bottom:0.35rem;'>"
                        f"<span style='color:{view['color']};font-weight:700;'>{line}</span>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
                with st.expander(f"Подробнее: {row['alert_id']}", expanded=False):
                    hist = row.get("history") or []
                    if hist:
                        hist_df = pd.DataFrame(hist)
                        if not hist_df.empty:
                            st.dataframe(
                                hist_df[["batch_index", "status", "related_events", "explanation"]],
                                use_container_width=True,
                                hide_index=True,
                                height=180,
                            )
                    if row.get("related_events"):
                        st.caption("related_events: " + ", ".join([str(x) for x in row["related_events"]]))
                    if row.get("explanation"):
                        deps.render_scrollable_text(str(row["explanation"]), height=120)
                prev_status = str(previous.get(row["alert_id"]) or "")
                if prev_status != "EXPLAINED" and row["status"] == "EXPLAINED":
                    st.success(f"Алерт `{row['alert_id']}` перешёл в EXPLAINED")
                previous[row["alert_id"]] = row["status"]
        else:
            st.info("Список алертов пока пуст.")

        # --- Stepper ---
        reduce_nodes = state.get("reduce_nodes")
        if not isinstance(reduce_nodes, list):
            reduce_nodes = []
        max_reduce_round = max((_safe_int(item.get("round"), 0) for item in reduce_nodes), default=0)
        stage_labels: List[str] = ["Получение Логов", "Саммаризация (MAP)"]
        if max_reduce_round <= 1:
            stage_labels.append("Reduce L1")
        else:
            for round_idx in range(1, max_reduce_round + 1):
                stage_labels.append(f"Reduce L{round_idx}")
        stage_labels += ["Верификация", "Отчёт"]

        active_step_text = str(state.get("active_step") or "").strip().lower()

        current_stage_idx = 0
        if status == "map":
            is_fetching_logs = any(
                marker in active_step_text
                for marker in (
                    "читаем логи",
                    "выгружаем страницу",
                    "страница логов",
                    "fetch",
                )
            )
            current_stage_idx = 0 if is_fetching_logs else 1
        elif status == "summarizing":
            current_stage_idx = 1
        elif status == "reduce":
            active_reduce_round = _safe_int(state.get("active_reduce_round"), 1)
            reduce_stage_base = 2
            current_stage_idx = min(reduce_stage_base + max(active_reduce_round - 1, 0), len(stage_labels) - 2)
        elif status == "summary_ready":
            current_stage_idx = max(len(stage_labels) - 2, 0)
        elif status in {"done", "error"}:
            current_stage_idx = len(stage_labels) - 1

        st.markdown("### Этапы")
        step_cols = st.columns(len(stage_labels))
        for idx, (col, label) in enumerate(zip(step_cols, stage_labels)):
            if status == "error" and idx == current_stage_idx:
                step_style = STEP_STATUS_STYLE["error"]
            elif idx < current_stage_idx:
                step_style = STEP_STATUS_STYLE["done"]
            elif idx == current_stage_idx:
                step_style = STEP_STATUS_STYLE["active"]
            else:
                step_style = STEP_STATUS_STYLE["future"]
            progress_suffix = ""
            if label == "Саммаризация (MAP)" and batch_total > 0:
                progress_suffix = f" ({done_batches}/{batch_total})"
            if label.startswith("Reduce L"):
                round_idx = _safe_int(label.replace("Reduce L", ""), 1)
                round_nodes = [x for x in reduce_nodes if _safe_int(x.get("round"), 0) == round_idx]
                done_round_nodes = [x for x in round_nodes if str(x.get("status")) == "done"]
                if round_nodes:
                    progress_suffix = f" ({len(done_round_nodes)}/{len(round_nodes)})"
            col.markdown(
                (
                    f"<div style='padding:0.5rem;border:1px solid #d1d5db;border-radius:0.5rem;"
                    f"background:#fff;'>"
                    f"<span style='color:{step_style['color']};font-weight:700;'>{step_style['icon']}</span> "
                    f"<span style='font-weight:600;'>{label}{progress_suffix}</span>"
                    f"</div>"
                ),
                unsafe_allow_html=True,
            )

        active_step = str(state.get("active_step") or "").strip()
        if active_step:
            st.caption(f"Текущий шаг: {active_step}")

        stage1_progress = _build_stage1_progress_snapshot(state)
        if bool(stage1_progress.get("show")):
            st.markdown("### Прогресс Этапа 1 (Загрузка Логов + MAP)")
            st.progress(
                float(stage1_progress.get("ratio", 0.0)),
                text=str(stage1_progress.get("label") or ""),
            )
            runtime_line = str(stage1_progress.get("runtime_line") or "").strip()
            if runtime_line:
                st.caption(runtime_line)

        report_progress = _build_report_generation_progress_snapshot(state)
        if bool(report_progress.get("show")):
            st.markdown("### Прогресс Генерации Итогового Отчёта")
            st.progress(
                float(report_progress.get("ratio", 0.0)),
                text=str(report_progress.get("label") or ""),
            )

        # --- Details area for active stage ---
        st.markdown("### Детали Текущего Этапа")
        if current_stage_idx == 0:
            st.markdown("#### Получение Логов")
            batch_plan = state.get("batch_plan")
            if not isinstance(batch_plan, list):
                batch_plan = []
            if not batch_plan and map_batches:
                # fallback plan from already processed map batches
                for item in map_batches:
                    logs_count = _safe_int(item.get("batch_logs_count"), 0)
                    batch_plan.append(
                        {
                            "batch_id": _safe_int(item.get("batch_index"), 0) + 1,
                            "period": f"{item.get('batch_period_start')} -> {item.get('batch_period_end')}",
                            "rows": logs_count,
                            "tokens": "disabled",
                            "status": "Готов",
                        }
                    )
            if batch_plan:
                plan_df = pd.DataFrame(batch_plan)
                st.dataframe(plan_df, use_container_width=True, hide_index=True, height=260)
                split_count = _safe_int(state.get("batch_split_count"), 0)
                if batch_total > 0:
                    st.caption(
                        f"{batch_total} батчей → {batch_total + split_count} LLM-вызовов "
                        f"(разбитых батчей: {split_count})"
                    )
                else:
                    total_batches = len(plan_df.index)
                    st.caption(
                        f"{total_batches} батчей → {total_batches + split_count} LLM-вызовов "
                        f"(разбитых батчей: {split_count})"
                    )
            else:
                st.info("План батчинга появится после старта map-фазы.")

        elif current_stage_idx == 1:
            st.markdown("#### Саммаризация (MAP)")
            if not map_batches:
                st.info("Map-батчи ещё не обработаны.")
            for item in map_batches:
                batch_idx = _safe_int(item.get("batch_index"), 0) + 1
                total = _safe_int(item.get("batch_total"), batch_total)
                payload = _summary_payload_from_batch(item)
                data_quality = payload.get("data_quality") if isinstance(payload, dict) else {}
                is_empty = bool((data_quality or {}).get("is_empty", False))
                noise_ratio = _safe_float((data_quality or {}).get("noise_ratio"), float("nan"))
                timeline = payload.get("timeline") if isinstance(payload, dict) else []
                hypotheses = payload.get("hypotheses") if isinstance(payload, dict) else []
                alert_refs = payload.get("alert_refs") if isinstance(payload, dict) else []
                status_icon = "✓"
                status_text = "готов"
                if is_empty:
                    status_icon = "—"
                    status_text = "пустой"
                elif (data_quality or {}).get("notes"):
                    notes = str((data_quality or {}).get("notes") or "").lower()
                    if "validation error" in notes or "raw llm" in notes:
                        status_icon = "⚠"
                        status_text = "degraded"
                period = f"{item.get('batch_period_start')} -> {item.get('batch_period_end')}"
                summary_line = (
                    f"{status_icon} Батч {batch_idx}/{total or '?'} | {period} | "
                    f"событий: {len(timeline) if isinstance(timeline, list) else 0}"
                )
                if not pd.isna(noise_ratio):
                    summary_line += f" | шум: {float(noise_ratio) * 100:.1f}%"
                duration_sec = _safe_float(item.get("processing_seconds"), float("nan"))
                if not pd.isna(duration_sec) and duration_sec > 0:
                    summary_line += f" | {duration_sec:.1f}s"
                summary_line += f" | {status_text}"
                with st.expander(summary_line, expanded=False):
                    if not isinstance(payload, dict) or not payload:
                        deps.render_pretty_summary_text(item.get("batch_summary", ""), height=max(deps.summary_text_height, 220))
                        continue
                    tab_events, tab_hyp, tab_alerts = st.tabs(["События", "Гипотезы", "Алерты"])
                    with tab_events:
                        if isinstance(timeline, list) and timeline:
                            events_df = pd.DataFrame(timeline)
                            keep_cols = [
                                col for col in
                                ["id", "timestamp", "source", "description", "severity", "importance", "evidence_type", "tags"]
                                if col in events_df.columns
                            ]
                            st.dataframe(
                                _format_table_timestamps(events_df[keep_cols] if keep_cols else events_df),
                                use_container_width=True,
                                hide_index=True,
                                height=220,
                            )
                        else:
                            st.info("События в этом батче не найдены.")
                    with tab_hyp:
                        if isinstance(hypotheses, list) and hypotheses:
                            for hyp in hypotheses:
                                if not isinstance(hyp, dict):
                                    continue
                                st.markdown(
                                    f"- **{hyp.get('title', 'Гипотеза')}** "
                                    f"(confidence: `{_safe_float(hyp.get('confidence'), 0.0):.2f}`)"
                                )
                                if hyp.get("description"):
                                    st.caption(str(hyp.get("description")))
                        else:
                            st.info("Гипотезы в этом батче отсутствуют.")
                    with tab_alerts:
                        if isinstance(alert_refs, list) and alert_refs:
                            alert_df = pd.DataFrame(alert_refs)
                            keep_cols = [c for c in ["alert_id", "status", "related_events", "explanation"] if c in alert_df.columns]
                            st.dataframe(
                                alert_df[keep_cols] if keep_cols else alert_df,
                                use_container_width=True,
                                hide_index=True,
                                height=180,
                            )
                        else:
                            st.info("Статусы алертов в этом батче не указаны.")

        elif "Reduce" in stage_labels[current_stage_idx]:
            st.markdown("#### Reduce")
            if not reduce_nodes:
                st.info("Reduce-узлы появятся после завершения map-фазы.")
            else:
                rounds: Dict[int, List[Dict[str, Any]]] = {}
                for node in reduce_nodes:
                    round_idx = _safe_int(node.get("round"), 1)
                    rounds.setdefault(round_idx, []).append(node)
                for round_idx in sorted(rounds.keys()):
                    st.markdown(f"**Уровень {round_idx}**")
                    nodes = sorted(rounds[round_idx], key=lambda x: _safe_int(x.get("group"), 0))
                    for node in nodes:
                        icon = "✓" if str(node.get("status")) == "done" else ("●" if str(node.get("status")) == "active" else "○")
                        group_idx = _safe_int(node.get("group"), 0)
                        group_size = _safe_int(node.get("group_size"), 0)
                        summary = (
                            f"{icon} R{round_idx}-G{group_idx} | size={group_size} | status={node.get('status')}"
                        )
                        with st.expander(summary, expanded=False):
                            in_events = _safe_int(node.get("input_events"), 0)
                            out_events = _safe_int(node.get("output_events"), 0)
                            in_hyp = _safe_int(node.get("input_hypotheses"), 0)
                            out_hyp = _safe_int(node.get("output_hypotheses"), 0)
                            gaps_closed = _safe_int(node.get("gaps_closed"), 0)
                            new_links = _safe_int(node.get("new_causal_links"), 0)
                            in_tokens = _safe_int(node.get("input_tokens"), 0)
                            out_tokens = _safe_int(node.get("output_tokens"), 0)
                            compression_pct = _safe_float(node.get("compression_pct"), float("nan"))
                            st.markdown(
                                "\n".join(
                                    [
                                        f"- group_size: `{group_size}`",
                                        f"- Вход события/гипотезы: `{in_events}` / `{in_hyp}`",
                                        f"- Выход события/гипотезы: `{out_events}` / `{out_hyp}`",
                                        f"- Закрыто gaps: `{gaps_closed}`",
                                        f"- Новые causal links: `{new_links}`",
                                        f"- Токены вход/выход: `{in_tokens}` / `{out_tokens}`",
                                        (
                                            f"- Сжатие: `{compression_pct:.1f}%`"
                                            if not pd.isna(compression_pct)
                                            else "- Сжатие: `n/a`"
                                        ),
                                    ]
                                )
                            )
                            if node.get("split_reason"):
                                st.warning(f"split: {node.get('split_reason')}")
                            if node.get("error"):
                                st.error(str(node.get("error")))
                split_count = _safe_int(state.get("reduce_split_count"), 0)
                if split_count > 0:
                    st.warning(f"Split узлов из-за overflow/timeout: {split_count}")

        elif stage_labels[current_stage_idx] == "Верификация":
            st.markdown("#### Верификация")
            verification = state.get("verification")
            if isinstance(verification, dict):
                st.caption(str(verification.get("summary") or ""))
                selected_logs = verification.get("selected_logs") or []
                if isinstance(selected_logs, list) and selected_logs:
                    with st.expander("Логи, выбранные для контрольного прохода", expanded=False):
                        st.dataframe(
                            _format_table_timestamps(pd.DataFrame(selected_logs)),
                            use_container_width=True,
                            hide_index=True,
                            height=220,
                        )
                corrections = verification.get("corrections") or []
                if corrections:
                    st.dataframe(pd.DataFrame(corrections), use_container_width=True, hide_index=True)
                else:
                    st.success("Поправок не требуется.")
            else:
                st.info("Этап верификации будет отображаться здесь после его запуска.")

        else:
            st.markdown("#### Отчёт")
            if _normalize_summary_text(state.get("final_summary")):
                st.success("Итоговый отчёт сформирован. Ниже доступен интерактивный разбор.")
            else:
                st.info("Отчёт пока формируется.")

        # Compact runtime metrics for observability.
        stats = state.get("stats") or {}
        with st.expander("Метрики Pipeline", expanded=False):
            elapsed_human = str(stats.get("logs_processing_human") or "")
            elapsed_seconds = _safe_float(stats.get("logs_processing_seconds"), 0.0)
            llm_calls = _safe_int(stats.get("llm_calls"), 0)
            reduce_rounds = _safe_int(stats.get("reduce_rounds"), 0)
            retries_total = _safe_int(state.get("llm_calls_failed"), 0)
            degraded_batches = 0
            empty_batches = 0
            for item in map_batches:
                payload = _summary_payload_from_batch(item)
                dq = payload.get("data_quality") if isinstance(payload, dict) else {}
                notes = str((dq or {}).get("notes") or "").lower()
                if "validation error" in notes or "raw llm map summary" in notes:
                    degraded_batches += 1
                if bool((dq or {}).get("is_empty", False)):
                    empty_batches += 1
            st.markdown(
                "\n".join(
                    [
                        f"- Общее время: `{elapsed_human or _format_eta_seconds(elapsed_seconds)}`",
                        f"- LLM-вызовов: `{llm_calls}`",
                        f"- Retry: `{retries_total}`",
                        f"- Reduce раундов: `{reduce_rounds}`",
                        f"- Degraded батчей: `{degraded_batches}`",
                        f"- Пустых батчей: `{empty_batches}`",
                        f"- Split на reduce: `{_safe_int(state.get('reduce_split_count'), 0)}`",
                    ]
                )
            )
        if state.get("error"):
            st.error(f"Ошибка выполнения: {state['error']}")


def _build_config(deps: LogsSummaryPageDeps, db_batch_size: int, llm_batch_size: int, map_workers: int = 1) -> Any:
    max_cell_chars = int(
        getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_CELL_CHARS", 0)
    )
    max_summary_chars = int(
        getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_SUMMARY_CHARS", 0)
    )
    reduce_prompt_max_chars = int(
        getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_REDUCE_PROMPT_MAX_CHARS", 0)
    )
    auto_shrink_on_400 = bool(
        getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_AUTO_SHRINK_ON_400", True)
    )
    min_llm_batch_size_env = max(
        int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MIN_LLM_BATCH_SIZE", 20) or 20),
        1,
    )
    min_llm_batch_size = min(max(int(llm_batch_size), 1), min_llm_batch_size_env)
    max_shrink_rounds = max(
        int(getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_SHRINK_ROUNDS", 6) or 6),
        0,
    )
    use_new_algorithm = bool(
        getattr(
            settings,
            "CONTROL_PLANE_UI_LOGS_SUMMARY_USE_NEW_ALGORITHM",
            getattr(settings, "CONTROL_PLANE_LLM_USE_NEW_ALGORITHM", True),
        )
    )
    reduce_target_token_pct = max(
        min(int(getattr(settings, "CONTROL_PLANE_LLM_REDUCE_TARGET_TOKEN_PCT", 50) or 50), 95),
        10,
    )
    compression_target_pct = max(
        min(int(getattr(settings, "CONTROL_PLANE_LLM_COMPRESSION_TARGET_PCT", 50) or 50), 95),
        10,
    )
    compression_importance_threshold = float(
        getattr(settings, "CONTROL_PLANE_LLM_COMPRESSION_IMPORTANCE_THRESHOLD", 0.7) or 0.7
    )
    use_instructor = bool(getattr(settings, "CONTROL_PLANE_LLM_USE_INSTRUCTOR", True))
    try:
        return deps.summarizer_config_cls(
            page_limit=db_batch_size,
            llm_chunk_rows=llm_batch_size,
            min_llm_chunk_rows=min_llm_batch_size,
            auto_shrink_on_400=auto_shrink_on_400,
            max_shrink_rounds=max_shrink_rounds,
            max_cell_chars=max_cell_chars,
            max_summary_chars=max_summary_chars,
            reduce_prompt_max_chars=reduce_prompt_max_chars,
            keep_map_batches_in_memory=False,
            keep_map_summaries_in_result=False,
            map_workers=max(map_workers, 1),
            use_new_algorithm=use_new_algorithm,
            reduce_target_token_pct=reduce_target_token_pct,
            compression_target_pct=compression_target_pct,
            compression_importance_threshold=compression_importance_threshold,
            use_instructor=use_instructor,
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
    resume_sessions = _discover_resume_sessions(deps.output_dir)
    latest_unfinished_session = next(
        (item for item in resume_sessions if str(item.get("status", "")).strip().lower() != "done"),
        None,
    )

    def _queue_resume_session(selected_session: Dict[str, Any], *, continue_mode: bool) -> Optional[str]:
        loaded_params = _read_json_file(Path(str(selected_session["run_params_path"])))
        if not isinstance(loaded_params, dict):
            return "Не удалось загрузить параметры выбранной сессии."
        loaded_params = dict(loaded_params)
        if continue_mode:
            checkpoint_payload = _read_json_file(Path(str(selected_session["checkpoint_path"]))) or {}
            checkpoint_state = (
                checkpoint_payload.get("state", {})
                if isinstance(checkpoint_payload, dict)
                else {}
            )
            resume_ts = str(checkpoint_state.get("last_batch_ts") or "").strip()
            if not resume_ts:
                resume_ts = _extract_last_batch_ts_from_run_dir(Path(str(selected_session["run_dir"])))
            loaded_params["resume_mode"] = "continue"
            loaded_params["resume_session_dir"] = str(selected_session["run_dir"])
            loaded_params["resume_from_ts"] = resume_ts
            loaded_params["resume_ts_missing"] = not bool(resume_ts)
        else:
            loaded_params["resume_mode"] = "restart"
            loaded_params["resume_session_dir"] = ""
            loaded_params["resume_from_ts"] = ""
            loaded_params["resume_ts_missing"] = False
        # Keep the form in sync with the resumed session config on next rerun.
        st.session_state[PENDING_PREFILL_SESSION_KEY] = dict(loaded_params)
        st.session_state[RUN_PARAMS_SESSION_KEY] = loaded_params
        st.session_state[RUNNING_SESSION_KEY] = True
        return None

    dismissed_resume_id = str(
        st.session_state.get(RESUME_BANNER_DISMISSED_SESSION_KEY, "") or ""
    )
    if (
        not is_running
        and latest_unfinished_session is not None
        and dismissed_resume_id != str(latest_unfinished_session.get("id", ""))
    ):
        with st.container():
            st.info(
                "Найдена незавершённая сессия. "
                "Можно продолжить с последней точки или запустить заново."
            )
            st.caption(str(latest_unfinished_session.get("label", "")))
            auto_col_continue, auto_col_restart, auto_col_hide = st.columns(3)
            with auto_col_continue:
                auto_continue_clicked = st.button(
                    "Продолжить Последнюю",
                    key="logs_sum_auto_resume_continue",
                    use_container_width=True,
                )
            with auto_col_restart:
                auto_restart_clicked = st.button(
                    "Заново По Параметрам",
                    key="logs_sum_auto_resume_restart",
                    use_container_width=True,
                )
            with auto_col_hide:
                auto_hide_clicked = st.button(
                    "Скрыть Подсказку",
                    key="logs_sum_auto_resume_hide",
                    use_container_width=True,
                )
            if auto_continue_clicked:
                err = _queue_resume_session(latest_unfinished_session, continue_mode=True)
                if err:
                    st.error(err)
                else:
                    st.rerun()
            if auto_restart_clicked:
                err = _queue_resume_session(latest_unfinished_session, continue_mode=False)
                if err:
                    st.error(err)
                else:
                    st.rerun()
            if auto_hide_clicked:
                st.session_state[RESUME_BANNER_DISMISSED_SESSION_KEY] = str(
                    latest_unfinished_session.get("id", "")
                )
                st.rerun()

    widget_defaults: Dict[str, Any] = {
        "logs_sum_user_goal": "",
        "logs_sum_model_id": str(getattr(settings, "LLM_MODEL_ID", "") or ""),
        "logs_sum_period_mode": "Явный диапазон (start/end)",
        "logs_sum_center_dt": center_default,
        "logs_sum_window_minutes": max(int(deps.loopback_minutes), 1),
        "logs_sum_start_dt": start_default,
        "logs_sum_end_dt": end_default,
        "logs_sum_db_batch": max(int(deps.db_batch_size), 1),
        "logs_sum_llm_batch": max(int(deps.llm_batch_size), 1),
        "logs_sum_parallel_map": False,  # kept for backward compatibility with old sessions
        "logs_sum_map_workers": 1,  # sequential only
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
    alerts_state_key = "logs_sum_alert_items"
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
    if alerts_state_key not in st.session_state:
        legacy_goal = str(st.session_state.get("logs_sum_user_goal", "")).strip()
        st.session_state[alerts_state_key] = _normalize_alert_items(
            st.session_state.get(alerts_state_key),
            min_items=1,
            legacy_user_goal=legacy_goal,
        )
    st.session_state[alerts_state_key] = _normalize_alert_items(
        st.session_state.get(alerts_state_key),
        min_items=1,
        legacy_user_goal=str(st.session_state.get("logs_sum_user_goal", "")).strip(),
    )

    def _apply_saved_params_to_form(saved_params: Dict[str, Any]) -> None:
        mapped = _form_values_from_saved_params(
            saved_params=saved_params,
            default_query=default_query,
        )
        st.session_state["logs_sum_user_goal"] = mapped["logs_sum_user_goal"]
        st.session_state["logs_sum_model_id"] = str(mapped.get("logs_sum_model_id") or widget_defaults["logs_sum_model_id"])
        st.session_state["logs_sum_period_mode"] = mapped["logs_sum_period_mode"]
        st.session_state["logs_sum_window_minutes"] = max(int(mapped["logs_sum_window_minutes"]), 1)
        st.session_state["logs_sum_center_dt"] = mapped["logs_sum_center_dt"] or center_default
        st.session_state["logs_sum_start_dt"] = mapped["logs_sum_start_dt"] or start_default
        st.session_state["logs_sum_end_dt"] = mapped["logs_sum_end_dt"] or end_default
        st.session_state["logs_sum_db_batch"] = max(int(mapped["logs_sum_db_batch"]), 1)
        st.session_state["logs_sum_llm_batch"] = max(int(mapped["logs_sum_llm_batch"]), 1)
        st.session_state["logs_sum_parallel_map"] = False
        st.session_state["logs_sum_map_workers"] = 1
        st.session_state["logs_sum_max_retries"] = int(mapped["logs_sum_max_retries"])
        st.session_state["logs_sum_llm_timeout"] = max(int(mapped["logs_sum_llm_timeout"]), 10)
        st.session_state["logs_sum_demo_mode"] = bool(mapped["logs_sum_demo_mode"])
        st.session_state["logs_sum_demo_logs_count"] = max(
            int(mapped["logs_sum_demo_logs_count"]),
            100,
        )
        st.session_state["logs_sum_enable_no_logs_hypothesis"] = bool(
            mapped["logs_sum_enable_no_logs_hypothesis"]
        )

        st.session_state[logs_queries_state_key] = [
            _new_query_item(text) for text in mapped["logs_queries"]
        ]
        st.session_state[metrics_queries_state_key] = [
            _new_query_item(text) for text in mapped["metrics_queries"]
        ]
        st.session_state[alerts_state_key] = _normalize_alert_items(
            mapped.get("alerts"),
            min_items=1,
            legacy_user_goal=str(mapped.get("logs_sum_user_goal", "")),
        )

    pending_prefill_params = st.session_state.pop(PENDING_PREFILL_SESSION_KEY, None)
    if isinstance(pending_prefill_params, dict):
        _apply_saved_params_to_form(dict(pending_prefill_params))

    with st.sidebar:
        st.markdown("Восстановление Сессии")
        if resume_sessions:
            if RESUME_SELECTED_SESSION_KEY not in st.session_state:
                st.session_state[RESUME_SELECTED_SESSION_KEY] = 0
            selected_session_idx = st.selectbox(
                "Сохранённая сессия",
                options=list(range(len(resume_sessions))),
                format_func=lambda idx: str(resume_sessions[int(idx)]["label"]),
                key=RESUME_SELECTED_SESSION_KEY,
                disabled=is_running,
            )
            selected_session = resume_sessions[int(selected_session_idx)]
            col_resume, col_restart, col_fill = st.columns(3)
            with col_resume:
                resume_clicked = st.button(
                    "Продолжить",
                    key="logs_sum_resume_continue",
                    disabled=is_running,
                    use_container_width=True,
                )
            with col_restart:
                restart_clicked = st.button(
                    "Заново",
                    key="logs_sum_resume_restart",
                    disabled=is_running,
                    use_container_width=True,
                )
            with col_fill:
                fill_form_clicked = st.button(
                    "В Форму",
                    key="logs_sum_resume_fill_form",
                    disabled=is_running,
                    use_container_width=True,
                    help="Подставить параметры выбранной сессии в поля формы без запуска.",
                )
            if fill_form_clicked and not is_running:
                loaded_params = _read_json_file(Path(str(selected_session["run_params_path"])))
                if not isinstance(loaded_params, dict):
                    st.error("Не удалось загрузить параметры выбранной сессии.")
                else:
                    _apply_saved_params_to_form(dict(loaded_params))
                    st.success("Параметры сессии подставлены в форму.")
                    st.rerun()
            if (resume_clicked or restart_clicked) and not is_running:
                err = _queue_resume_session(selected_session, continue_mode=bool(resume_clicked))
                if err:
                    st.error(err)
                else:
                    st.rerun()
        else:
            st.caption("Сохранённых сессий пока нет.")

        st.markdown("Импорт Единого Файла Отчёта")
        bundle_file = st.file_uploader(
            "Файл report.bundle.json / report.json",
            type=["json"],
            key="logs_sum_bundle_uploader",
            disabled=is_running,
            help=(
                "Загрузи единый JSON-файл отчёта, и страница восстановит входные параметры "
                "и финальный отчёт для просмотра."
            ),
        )
        import_bundle_clicked = st.button(
            "Открыть Файл В Программе",
            key="logs_sum_import_bundle",
            use_container_width=True,
            disabled=is_running,
        )
        if import_bundle_clicked:
            if bundle_file is None:
                st.warning("Выбери JSON-файл отчёта для импорта.")
            else:
                try:
                    raw_bytes = bundle_file.getvalue()
                    parsed_payload = json.loads(raw_bytes.decode("utf-8"))
                    imported_request, imported_result = _extract_request_result_from_bundle(parsed_payload)
                    if not imported_request and not imported_result:
                        raise ValueError(
                            "Формат файла не поддерживается. Нужен report.bundle.json или report.json."
                        )
                    imported_state = _state_from_imported_result(imported_result)
                    imported_state["events"] = list(
                        imported_state.get("events") or []
                    ) + ["Импортировано из единого файла отчёта."]
                    prefill_params = _build_saved_params_from_import_request(
                        request_payload=imported_request,
                        center_default=center_default,
                        start_default=start_default,
                        end_default=end_default,
                        default_query=default_query,
                    )
                    st.session_state[PENDING_PREFILL_SESSION_KEY] = prefill_params
                    st.session_state[LAST_STATE_SESSION_KEY] = imported_state
                    st.session_state[RUNNING_SESSION_KEY] = False
                    st.session_state.pop(RUN_PARAMS_SESSION_KEY, None)
                    st.success("Файл загружен. Показываю параметры и итоговый отчёт.")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Не удалось импортировать файл отчёта: {exc}")

        st.markdown("---")
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

        st.markdown("Алерты/Инциденты")
        alert_time_defaults = _default_alert_time_values(
            center_dt_text=str(st.session_state.get("logs_sum_center_dt", "")),
            start_dt_text=str(st.session_state.get("logs_sum_start_dt", "")),
            end_dt_text=str(st.session_state.get("logs_sum_end_dt", "")),
            center_default=center_default,
            start_default=start_default,
            end_default=end_default,
        )
        if st.button(
            "+ Добавить алерт",
            key="logs_sum_add_alert",
            use_container_width=True,
            disabled=is_running,
        ):
            st.session_state[alerts_state_key].append(
                _new_alert_item(
                    time_point=alert_time_defaults["time_point"],
                    time_start=alert_time_defaults["time_start"],
                    time_end=alert_time_defaults["time_end"],
                )
            )
            st.rerun()

        remove_alert_id: Optional[str] = None
        for idx, item in enumerate(st.session_state[alerts_state_key], start=1):
            item_id = str(item.get("id") or uuid4().hex)
            item["id"] = item_id
            title_key = f"logs_sum_alert_title_{item_id}"
            details_key = f"logs_sum_alert_details_{item_id}"
            mode_key = f"logs_sum_alert_time_mode_{item_id}"
            point_key = f"logs_sum_alert_time_point_{item_id}"
            start_key = f"logs_sum_alert_time_start_{item_id}"
            end_key = f"logs_sum_alert_time_end_{item_id}"
            if title_key not in st.session_state:
                st.session_state[title_key] = str(item.get("title", ""))
            if details_key not in st.session_state:
                st.session_state[details_key] = str(item.get("details", ""))
            if mode_key not in st.session_state:
                st.session_state[mode_key] = (
                    "Промежуток"
                    if str(item.get("time_mode", "point")).strip().lower() == "range"
                    else "Один момент"
                )
            if point_key not in st.session_state:
                st.session_state[point_key] = (
                    str(item.get("time_point", "")).strip() or alert_time_defaults["time_point"]
                )
            if start_key not in st.session_state:
                st.session_state[start_key] = (
                    str(item.get("time_start", "")).strip() or alert_time_defaults["time_start"]
                )
            if end_key not in st.session_state:
                st.session_state[end_key] = (
                    str(item.get("time_end", "")).strip() or alert_time_defaults["time_end"]
                )

            col_alert, col_remove = st.columns([10, 2])
            with col_alert:
                st.text_input(
                    f"Алерт #{idx} — Название",
                    key=title_key,
                    placeholder=f"Например: alert_{idx}",
                    disabled=is_running,
                )
                selected_mode = st.radio(
                    f"Алерт #{idx} — Время",
                    options=("Один момент", "Промежуток"),
                    key=mode_key,
                    horizontal=True,
                    disabled=is_running,
                )
                if selected_mode == "Промежуток":
                    st.text_input(
                        f"Алерт #{idx} — Начало (ISO)",
                        key=start_key,
                        placeholder="2026-03-18T18:42:00+03:00",
                        disabled=is_running,
                    )
                    st.text_input(
                        f"Алерт #{idx} — Конец (ISO)",
                        key=end_key,
                        placeholder="2026-03-18T19:10:00+03:00",
                        disabled=is_running,
                    )
                else:
                    st.text_input(
                        f"Алерт #{idx} — Время (ISO)",
                        key=point_key,
                        placeholder="2026-03-18T18:42:00+03:00",
                        disabled=is_running,
                    )
                st.text_area(
                    f"Алерт #{idx} — Описание (опционально)",
                    key=details_key,
                    height=100,
                    disabled=is_running,
                )
            with col_remove:
                can_remove_alert = len(st.session_state[alerts_state_key]) > 1
                if st.button(
                    "Убрать",
                    key=f"logs_sum_remove_alert_{item_id}",
                    disabled=is_running or not can_remove_alert,
                    use_container_width=True,
                    help="Удалить этот алерт",
                ):
                    remove_alert_id = item_id

            item["title"] = str(st.session_state.get(title_key, ""))
            item["details"] = str(st.session_state.get(details_key, ""))
            item["time_mode"] = (
                "range"
                if str(st.session_state.get(mode_key, "Один момент")) == "Промежуток"
                else "point"
            )
            item["time_point"] = str(st.session_state.get(point_key, ""))
            item["time_start"] = str(st.session_state.get(start_key, ""))
            item["time_end"] = str(st.session_state.get(end_key, ""))

        if remove_alert_id:
            st.session_state[alerts_state_key] = [
                item for item in st.session_state[alerts_state_key]
                if str(item.get("id")) != remove_alert_id
            ]
            st.rerun()

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
                help="Часовой пояс по умолчанию — MSK (`+03:00`). Можно явно указать `+03:00` или `Z`.",
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
                help="Формат: `YYYY-MM-DDTHH:MM:SS+03:00` (если без зоны — считаем как MSK).",
                disabled=is_running,
            )
            st.text_input(
                "Дата/время конца (ISO)",
                key="logs_sum_end_dt",
                placeholder="Например: 2026-03-27T12:00:00+03:00",
                help="Формат: `YYYY-MM-DDTHH:MM:SS+03:00` (если без зоны — считаем как MSK).",
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
            "Лимит строк на 1 LLM MAP batch",
            min_value=10,
            max_value=100_000,
            step=10,
            key="logs_sum_llm_batch",
            disabled=is_running,
            help=(
                "Верхняя граница строк для одного MAP-вызова LLM. "
                "Пайплайн никогда не отправит больше этого значения. "
                "Если LLM вернёт overflow/400, размер батча будет автоматически уменьшаться."
            ),
        )
        api_model_candidates, api_models_error = _fetch_llm_model_candidates()
        model_candidates = list(api_model_candidates)
        if not model_candidates:
            model_candidates = list(MODEL_CONTEXT_PRESETS.keys())
        current_model = str(st.session_state.get("logs_sum_model_id") or "").strip()
        if current_model and current_model not in model_candidates:
            model_candidates.insert(0, current_model)
        elif not current_model:
            fallback_model = str(getattr(settings, "LLM_MODEL_ID", "") or "").strip()
            if fallback_model and fallback_model not in model_candidates:
                model_candidates.insert(0, fallback_model)
        if not model_candidates:
            model_candidates = [str(getattr(settings, "LLM_MODEL_ID", "") or "default-model")]
        st.selectbox(
            "Модель LLM",
            options=model_candidates,
            key="logs_sum_model_id",
            disabled=is_running,
            help="Модель используется только на стороне API; локальный подсчёт токенов отключён.",
        )
        selected_model_for_budget = str(st.session_state.get("logs_sum_model_id") or "").strip()
        st.caption(
            f"Локальная оценка токенов отключена (model: `{selected_model_for_budget or 'n/a'}`)"
        )
        if api_model_candidates:
            st.caption(f"Список моделей загружен из LLM API (`{len(api_model_candidates)}` шт.).")
        elif api_models_error:
            st.caption(f"LLM API модели недоступны: {api_models_error} Используем fallback список.")
        st.caption("MAP обрабатывается последовательно (workers = 1).")
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
        st.checkbox(
            "Если логов нет — сделать осторожное предположение через LLM",
            key="logs_sum_enable_no_logs_hypothesis",
            disabled=is_running,
            help=(
                "Опционально: даже если в периоде нет логов, "
                "LLM сформирует гипотезы и список проверок для SRE."
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
    alert_items, alerts, user_goal = _collect_alerts_and_goal(
        st.session_state.get(alerts_state_key),
        min_items=1,
        legacy_user_goal=str(st.session_state.get("logs_sum_user_goal", "")),
    )
    st.session_state[alerts_state_key] = alert_items
    st.session_state["logs_sum_user_goal"] = user_goal
    window_minutes = int(st.session_state.get("logs_sum_window_minutes", max(int(deps.loopback_minutes), 1)))
    center_dt_text = str(st.session_state.get("logs_sum_center_dt", center_default))
    start_dt_text = str(st.session_state.get("logs_sum_start_dt", start_default))
    end_dt_text = str(st.session_state.get("logs_sum_end_dt", end_default))
    db_batch_size = int(st.session_state.get("logs_sum_db_batch", max(int(deps.db_batch_size), 1)))
    llm_batch_size = max(int(st.session_state.get("logs_sum_llm_batch", max(int(deps.llm_batch_size), 1))), 1)
    llm_model_id = str(st.session_state.get("logs_sum_model_id") or getattr(settings, "LLM_MODEL_ID", "")).strip()
    logs_timestamp_column = _normalize_timestamp_column_name(
        getattr(deps, "logs_timestamp_column", "timestamp")
    )
    map_workers = 1
    max_retries = int(st.session_state.get("logs_sum_max_retries", int(deps.max_retries)))
    llm_timeout = max(int(st.session_state.get("logs_sum_llm_timeout", max(int(deps.llm_timeout), 10))), 10)
    enable_no_logs_hypothesis = bool(
        st.session_state.get("logs_sum_enable_no_logs_hypothesis", False)
    )
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

    def _validate_alerts_for_run(alerts_for_run: List[Dict[str, str]]) -> Optional[str]:
        for idx, alert in enumerate(alerts_for_run, start=1):
            title = str(alert.get("title") or "").strip() or f"#{idx}"
            mode = str(alert.get("time_mode") or "point").strip().lower()
            if mode == "range":
                start_raw = str(alert.get("time_start") or "").strip()
                end_raw = str(alert.get("time_end") or "").strip()
                if not start_raw or not end_raw:
                    return (
                        f"Алерт {title}: для режима 'Промежуток' "
                        "нужно заполнить и начало, и конец."
                    )
                start_parsed = _parse_user_dt(start_raw)
                end_parsed = _parse_user_dt(end_raw)
                if pd.isna(start_parsed) or pd.isna(end_parsed):
                    return (
                        f"Алерт {title}: неверный ISO формат даты/времени "
                        "в диапазоне."
                    )
                if end_parsed <= start_parsed:
                    return (
                        f"Алерт {title}: дата конца должна быть больше даты начала."
                    )
            else:
                point_raw = str(alert.get("time_point") or "").strip()
                if not point_raw:
                    return (
                        f"Алерт {title}: для режима 'Один момент' "
                        "нужно заполнить время."
                    )
                point_parsed = _parse_user_dt(point_raw)
                if pd.isna(point_parsed):
                    return (
                        f"Алерт {title}: неверный ISO формат даты/времени."
                    )
        return None

    if run_clicked and not is_running:
        alerts_error = _validate_alerts_for_run(alerts)
        if alerts_error:
            _unlock_with_form_error(alerts_error)
        st.session_state[RUN_PARAMS_SESSION_KEY] = {
            "logs_queries": list(logs_queries),
            "metrics_queries": list(metrics_queries),
            "alerts": [dict(item) for item in alerts],
            "user_goal": user_goal,
            "period_mode": str(st.session_state.get("logs_sum_period_mode", "Явный диапазон (start/end)")),
            "window_minutes": window_minutes,
            "center_dt_text": center_dt_text,
            "start_dt_text": start_dt_text,
            "end_dt_text": end_dt_text,
            "db_batch_size": db_batch_size,
            "llm_batch_size": llm_batch_size,
            "llm_model_id": llm_model_id,
            "logs_timestamp_column": logs_timestamp_column,
            "map_workers": map_workers,
            "max_retries": max_retries,
            "llm_timeout": llm_timeout,
            "enable_no_logs_hypothesis": enable_no_logs_hypothesis,
            "demo_mode": demo_mode,
            "demo_logs_count": demo_logs_count,
            "resume_mode": "new",
            "resume_session_dir": "",
            "resume_from_ts": "",
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
            "alerts": [dict(item) for item in alerts],
            "user_goal": user_goal,
            "period_mode": str(st.session_state.get("logs_sum_period_mode", "Явный диапазон (start/end)")),
            "window_minutes": window_minutes,
            "center_dt_text": center_dt_text,
            "start_dt_text": start_dt_text,
            "end_dt_text": end_dt_text,
            "db_batch_size": db_batch_size,
            "llm_batch_size": llm_batch_size,
            "llm_model_id": llm_model_id,
            "logs_timestamp_column": logs_timestamp_column,
            "map_workers": map_workers,
            "max_retries": max_retries,
            "llm_timeout": llm_timeout,
            "enable_no_logs_hypothesis": enable_no_logs_hypothesis,
            "demo_mode": demo_mode,
            "demo_logs_count": demo_logs_count,
            "resume_mode": "new",
            "resume_session_dir": "",
            "resume_from_ts": "",
        }

    logs_queries = list(active_params.get("logs_queries", []))
    metrics_queries = list(active_params.get("metrics_queries", []))
    _, alerts, user_goal = _collect_alerts_and_goal(
        active_params.get("alerts"),
        min_items=0,
        legacy_user_goal=str(active_params.get("user_goal", "")),
    )
    active_params["alerts"] = [dict(item) for item in alerts]
    active_params["user_goal"] = user_goal
    period_mode_label = str(active_params.get("period_mode", "Явный диапазон (start/end)"))
    period_mode = "window" if period_mode_label.startswith("Окно вокруг") else "start_end"
    window_minutes = int(active_params.get("window_minutes", window_minutes))
    center_dt_text = str(active_params.get("center_dt_text", center_dt_text))
    start_dt_text = str(active_params.get("start_dt_text", start_dt_text))
    end_dt_text = str(active_params.get("end_dt_text", end_dt_text))
    db_batch_size = int(active_params.get("db_batch_size", db_batch_size))
    llm_batch_size = max(int(active_params.get("llm_batch_size", db_batch_size)), 1)
    llm_model_id = str(active_params.get("llm_model_id", llm_model_id)).strip() or llm_model_id
    logs_timestamp_column = _normalize_timestamp_column_name(
        active_params.get("logs_timestamp_column", logs_timestamp_column)
    )
    map_workers = 1
    max_retries = int(active_params.get("max_retries", max_retries))
    llm_timeout = max(int(active_params.get("llm_timeout", llm_timeout)), 10)
    enable_no_logs_hypothesis = bool(
        active_params.get("enable_no_logs_hypothesis", enable_no_logs_hypothesis)
    )
    demo_mode = bool(active_params.get("demo_mode", demo_mode))
    demo_logs_count = int(active_params.get("demo_logs_count", demo_logs_count))
    resume_mode = str(active_params.get("resume_mode", "new")).strip().lower()
    resume_session_dir = str(active_params.get("resume_session_dir", "")).strip()
    resume_from_ts = str(active_params.get("resume_from_ts", "")).strip()
    resume_ts_missing = bool(active_params.get("resume_ts_missing", False))
    active_params["llm_model_id"] = llm_model_id
    active_params["llm_batch_size"] = llm_batch_size
    active_params["map_workers"] = 1

    logs_queries = [str(q) for q in logs_queries if str(q).strip()]
    metrics_queries = [str(q) for q in metrics_queries if str(q).strip()]

    if not logs_queries:
        _unlock_with_form_error("Добавь хотя бы один SQL запрос логов.")

    try:
        if period_mode == "window":
            parsed_center = _parse_user_dt(center_dt_text)
            if pd.isna(parsed_center):
                raise ValueError("Неверный формат даты/времени (ISO).")
            center_dt = parsed_center.to_pydatetime()
            period_start_dt = center_dt - timedelta(minutes=window_minutes)
            period_end_dt = center_dt + timedelta(minutes=window_minutes)
        else:
            parsed_start = _parse_user_dt(start_dt_text)
            parsed_end = _parse_user_dt(end_dt_text)
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
    effective_period_start_iso = period_start_iso
    if resume_mode == "continue" and resume_from_ts:
        parsed_resume_ts = _to_msk_ts(resume_from_ts)
        parsed_end_ts = _to_msk_ts(period_end_iso)
        if not pd.isna(parsed_resume_ts) and not pd.isna(parsed_end_ts) and parsed_resume_ts < parsed_end_ts:
            effective_period_start_iso = (
                parsed_resume_ts.to_pydatetime() + timedelta(microseconds=1)
            ).isoformat()
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
                    timestamp_column=logs_timestamp_column,
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
                    timestamp_column="timestamp",
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

        schema_errors.extend(
            _validate_logs_merge_schema(
                logs_preview_frames,
                timestamp_column=logs_timestamp_column,
            )
        )
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
        joined = "\n".join([str(err) for err in schema_errors])
        _unlock_with_form_error(
            "Формат SQL-результатов несовместим для merge в единые DataFrame.\n" + joined
        )

    run_stamp = datetime.now(MSK).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = deps.output_dir / "logs_summary_live" / f"run_{run_stamp}"
    is_resume_continue = False
    if resume_mode == "continue" and resume_session_dir:
        candidate = Path(resume_session_dir)
        if candidate.exists() and candidate.is_dir():
            run_dir = candidate
            is_resume_continue = True
    live_events_path = run_dir / "events.jsonl"
    live_batches_path = run_dir / "batches.jsonl"
    run_params_path = run_dir / "run_params.json"
    request_path = run_dir / "request.json"
    session_checkpoint_path = run_dir / "checkpoint.json"
    summaries_dir = run_dir / "summaries"
    map_summaries_dir = summaries_dir / "map"
    reduce_summaries_dir = summaries_dir / "reduce"
    final_summaries_dir = summaries_dir / "final"
    llm_calls_dir = summaries_dir / "llm_calls"
    map_summaries_jsonl_path = summaries_dir / "map_summaries.jsonl"
    reduce_summaries_jsonl_path = summaries_dir / "reduce_summaries.jsonl"
    llm_calls_jsonl_path = summaries_dir / "llm_calls.jsonl"
    logs_fetch_mode = _normalize_logs_fetch_mode(deps.logs_fetch_mode)
    initial_events: List[str] = []
    if preview_query_errors or preview_metrics_errors:
        initial_events.append(
            f"Предварительных ошибок ClickHouse: {len(preview_query_errors) + len(preview_metrics_errors)}"
        )
    if is_resume_continue:
        initial_events.append(
            "РЕЖИМ ВОССТАНОВЛЕНИЯ: продолжаем с сохранённой сессии "
            f"({run_dir.name})"
        )
        if resume_ts_missing:
            initial_events.append(
                "Resume warning: не нашли last_batch_ts в checkpoint, "
                "продолжение может начаться с начала периода."
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
            f"Старт: режим выборки логов по датам ({effective_period_start_iso} -> {period_end_iso})."
        )
    initial_events.append(f"Колонка времени логов: `{logs_timestamp_column}`")
    initial_events.append(
        "LLM MAP batch: "
        f"стартовый лимит `{llm_batch_size}` строк, "
        "при overflow/400 будет авто-уменьшение."
    )

    state: Dict[str, Any] = {
        "status": "queued",
        "mode": "demo" if demo_mode else "db",
        "logs_fetch_mode": logs_fetch_mode,
        "logs_tail_limit": int(deps.logs_tail_limit),
        "logs_timestamp_column": logs_timestamp_column,
        "demo_logs_count": int(demo_logs_count),
        "query": sql_query_clean,
        "metrics_query": metrics_query_clean,
        "queries_count": len(query_specs),
        "metrics_queries_count": len(metrics_query_specs),
        "active_source_label": "query_1",
        "alerts": [dict(item) for item in alerts],
        "user_goal": user_goal.strip(),
        "period_mode": period_mode,
        "period_start": period_start_iso,
        "effective_period_start": effective_period_start_iso,
        "period_end": period_end_iso,
        "resume_mode": "continue" if is_resume_continue else ("restart" if resume_mode == "restart" else "new"),
        "resume_from_ts": resume_from_ts,
        "resume_session_dir": str(run_dir) if is_resume_continue else "",
        "window_minutes": window_minutes,
        "db_batch_size": db_batch_size,
        "llm_batch_size": llm_batch_size,
        "llm_model_id": llm_model_id,
        "map_workers": map_workers,
        "max_retries": max_retries,
        "llm_timeout": llm_timeout,
        "enable_no_logs_hypothesis": enable_no_logs_hypothesis,
        "logs_processed": 0,
        "logs_total": None,
        "resume_rows_offset": 0,
        "resume_batch_offset": 0,
        "resume_stats_offset": {
            "pages_fetched": 0,
            "rows_processed": 0,
            "llm_calls": 0,
            "reduce_rounds": 0,
        },
        "events": initial_events,
        "query_errors": list(preview_query_errors) + list(preview_metrics_errors),
        "map_batches": [],
        "map_batches_done_total": 0,
        "map_batches_total": 0,
        "source_batch_offset": 0,
        "batch_plan": [],
        "batch_split_count": 0,
        "reduce_nodes": [],
        "reduce_split_count": 0,
        "verification": None,
        "final_summary": None,
        "final_summary_origin": None,
        "structured_sections": [],
        "freeform_final_summary": None,
        "freeform_sections": [],
        "metrics_rows": 0,
        "metrics_services": [],
        "metrics_context_text": "",
        "stats": None,
        "error": None,
        "result_json_path": None,
        "result_bundle_path": None,
        "result_summary_path": None,
        "result_html_path": None,
        "result_structured_md_path": None,
        "result_freeform_md_path": None,
        "result_structured_txt_path": None,
        "result_freeform_txt_path": None,
        "rebuild_reduce_path": None,
        "live_events_path": str(live_events_path),
        "live_batches_path": str(live_batches_path),
        "run_params_path": str(run_params_path),
        "request_path": str(request_path),
        "checkpoint_path": str(session_checkpoint_path),
        "map_summaries_jsonl_path": str(map_summaries_jsonl_path),
        "reduce_summaries_jsonl_path": str(reduce_summaries_jsonl_path),
        "llm_calls_jsonl_path": str(llm_calls_jsonl_path),
        "started_at": datetime.now(MSK).isoformat(),
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
        "llm_phase_hint": "map",
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
        "report_progress_current": 0,
        "report_progress_total": 0,
        "report_progress_label": "",
        "report_progress_active": False,
    }

    if is_resume_continue:
        checkpoint_payload = _read_json_file(session_checkpoint_path) or {}
        checkpoint_state = (
            checkpoint_payload.get("state", {})
            if isinstance(checkpoint_payload, dict)
            else {}
        )
        if isinstance(checkpoint_state, dict):
            for key in (
                "logs_processed",
                "logs_total",
                "last_batch_ts",
                "stats",
                "elapsed_seconds",
                "log_seconds_per_second",
                "eta_seconds_left",
                "eta_finish_at",
                "llm_calls_started",
                "llm_calls_succeeded",
                "llm_calls_failed",
                "llm_last_error",
                "llm_timeline",
                "batch_plan",
                "batch_split_count",
                "map_batches_done_total",
                "map_batches_total",
                "source_batch_offset",
                "reduce_nodes",
                "reduce_split_count",
                "verification",
                "final_summary",
                "final_summary_origin",
                "structured_sections",
                "freeform_final_summary",
                "freeform_sections",
                "result_json_path",
                "result_bundle_path",
                "result_summary_path",
                "result_html_path",
                "result_structured_md_path",
                "result_freeform_md_path",
                "result_structured_txt_path",
                "result_freeform_txt_path",
                "report_progress_current",
                "report_progress_total",
                "report_progress_label",
                "report_progress_active",
            ):
                if checkpoint_state.get(key) is not None:
                    state[key] = checkpoint_state.get(key)
            prev_events = checkpoint_state.get("events")
            if isinstance(prev_events, list) and prev_events:
                merged_events = [str(item) for item in prev_events[-MAX_EVENT_LINES:]]
                merged_events.extend(initial_events)
                state["events"] = merged_events[-MAX_EVENT_LINES:]
            prev_map_batches = checkpoint_state.get("map_batches")
            if isinstance(prev_map_batches, list) and prev_map_batches:
                state["map_batches"] = prev_map_batches[-MAX_RENDERED_BATCHES:]
            elif live_batches_path.exists():
                loaded_batches = _load_recent_batches_from_jsonl(str(live_batches_path))
                if loaded_batches:
                    state["map_batches"] = loaded_batches[-MAX_RENDERED_BATCHES:]
            saved_rows = int(pd.to_numeric(checkpoint_state.get("logs_processed"), errors="coerce") or 0)
            state["resume_rows_offset"] = max(saved_rows, 0)
            state["logs_processed"] = max(saved_rows, int(pd.to_numeric(state.get("logs_processed"), errors="coerce") or 0))
            saved_stats = checkpoint_state.get("stats")
            if isinstance(saved_stats, dict):
                state["resume_stats_offset"] = {
                    "pages_fetched": int(pd.to_numeric(saved_stats.get("pages_fetched"), errors="coerce") or 0),
                    "rows_processed": int(pd.to_numeric(saved_stats.get("rows_processed"), errors="coerce") or 0),
                    "llm_calls": int(pd.to_numeric(saved_stats.get("llm_calls"), errors="coerce") or 0),
                    "reduce_rounds": int(pd.to_numeric(saved_stats.get("reduce_rounds"), errors="coerce") or 0),
                }
            else:
                state["resume_stats_offset"] = {
                    "pages_fetched": 0,
                    "rows_processed": saved_rows,
                    "llm_calls": int(pd.to_numeric(checkpoint_state.get("llm_calls_started"), errors="coerce") or 0),
                    "reduce_rounds": 0,
                }
            saved_elapsed = pd.to_numeric(checkpoint_state.get("elapsed_seconds"), errors="coerce")
            if not pd.isna(saved_elapsed) and float(saved_elapsed) > 0:
                state["elapsed_seconds"] = float(saved_elapsed)
                state["started_monotonic"] = time.monotonic() - float(saved_elapsed)

        state["resume_batch_offset"] = len(
            _load_map_summaries_from_jsonl(str(map_summaries_jsonl_path))
        )
        if _safe_int(state.get("map_batches_done_total"), 0) < int(state["resume_batch_offset"]):
            state["map_batches_done_total"] = int(state["resume_batch_offset"])
        if _safe_int(state.get("map_batches_total"), 0) < _safe_int(state.get("map_batches_done_total"), 0):
            state["map_batches_total"] = _safe_int(state.get("map_batches_done_total"), 0)
        state["source_batch_offset"] = _safe_int(state.get("map_batches_done_total"), 0)
        state["active_step"] = (
            f"Восстановление с прогресса: last_ts={state.get('last_batch_ts') or 'n/a'}"
        )

    _render_logs_summary_chat(analysis_placeholder, state, deps)

    request_payload = {
        "sql_query": sql_query_clean,
        "logs_queries": list(logs_queries),
        "sql_queries_count": len(query_specs),
        "metrics_query": metrics_query_clean,
        "metrics_queries": list(metrics_queries),
        "metrics_queries_count": len(metrics_query_specs),
        "logs_fetch_mode": logs_fetch_mode,
        "logs_tail_limit": int(deps.logs_tail_limit),
        "logs_timestamp_column": logs_timestamp_column,
        "alerts": [dict(item) for item in alerts],
        "user_goal": user_goal.strip(),
        "period_mode": period_mode,
        "period_start": period_start_iso,
        "effective_period_start": effective_period_start_iso,
        "period_end": period_end_iso,
        "resume_mode": "continue" if is_resume_continue else ("restart" if resume_mode == "restart" else "new"),
        "resume_from_ts": resume_from_ts,
        "resume_session_dir": str(run_dir) if is_resume_continue else "",
        "window_minutes": window_minutes,
        "demo_mode": demo_mode,
        "demo_logs_count": demo_logs_count,
        "db_batch_size": db_batch_size,
        "llm_batch_size": llm_batch_size,
        "llm_model_id": llm_model_id,
        "enable_no_logs_hypothesis": enable_no_logs_hypothesis,
        "live_events_path": str(live_events_path),
        "live_batches_path": str(live_batches_path),
        "map_summaries_jsonl_path": str(map_summaries_jsonl_path),
        "reduce_summaries_jsonl_path": str(reduce_summaries_jsonl_path),
        "llm_calls_jsonl_path": str(llm_calls_jsonl_path),
    }

    # Persist run metadata immediately so the session can be resumed after crash.
    _write_json_file(run_params_path, dict(active_params))
    _write_json_file(request_path, dict(request_payload))
    _persist_checkpoint(session_checkpoint_path, state)

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
            now_iso = datetime.now(MSK).isoformat()
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
            end_iso = datetime.now(MSK).isoformat()
            errors_count = int(pd.to_numeric(state.get("read_timeout_count"), errors="coerce") or 0)
            start_ts = _to_msk_ts(start_iso)
            end_ts = _to_msk_ts(end_iso)
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
                            timestamp_column="timestamp",
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
                timestamp_column=logs_timestamp_column,
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
                    timestamp_column=logs_timestamp_column,
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
                df = _sort_df_by_timestamp(df, timestamp_column=logs_timestamp_column)
                # Advance keyset cursor so the next call fetches the next page
                if uses_keyset and logs_timestamp_column in df.columns:
                    max_ts = df[logs_timestamp_column].apply(_to_msk_ts).max()
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
                    "ts": datetime.now(MSK).isoformat(),
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

            def _upsert_reduce_node(
                *,
                round_idx: int,
                group_idx: int,
                updates: Dict[str, Any],
            ) -> None:
                nodes = state.setdefault("reduce_nodes", [])
                if not isinstance(nodes, list):
                    nodes = []
                for node in nodes:
                    if (
                        _safe_int(node.get("round"), -1) == int(round_idx)
                        and _safe_int(node.get("group"), -1) == int(group_idx)
                    ):
                        node.update(updates)
                        state["reduce_nodes"] = nodes
                        return
                row = {"round": int(round_idx), "group": int(group_idx)}
                row.update(updates)
                nodes.append(row)
                state["reduce_nodes"] = nodes

            if event == "map_start":
                state["status"] = "map"
                state["llm_phase_hint"] = "map"
                state["active_step"] = "Читаем логи и готовим MAP-батчи"
                current_done = _safe_int(
                    state.get("map_batches_done_total"),
                    _safe_int(state.get("resume_batch_offset"), 0),
                )
                state["map_batches_done_total"] = max(
                    current_done,
                    _safe_int(state.get("resume_batch_offset"), 0),
                )
                state["source_batch_offset"] = _safe_int(state.get("map_batches_done_total"), 0)
                incoming_total = _safe_int(payload.get("batch_total"), 0)
                if incoming_total > 0:
                    state["estimated_batch_total"] = incoming_total
                    total_candidate = _safe_int(state.get("source_batch_offset"), 0) + incoming_total
                    state["map_batches_total"] = max(
                        _safe_int(state.get("map_batches_total"), 0),
                        total_candidate,
                    )
                else:
                    state["estimated_batch_total"] = None
                state["use_new_algorithm"] = bool(payload.get("use_new_algorithm", False))
                state["batch_plan"] = []
                events.append("Map этап запущен")
            elif event == "page_fetched":
                state["status"] = "map"
                state["active_step"] = "Выгружаем страницу логов из БД"
                events.append(
                    f"Страница #{payload.get('page_index')}: {payload.get('page_rows')} строк"
                )
            elif event == "map_batch_start":
                state["status"] = "map"
                state["llm_phase_hint"] = "map"
                source_batch_offset = _safe_int(
                    state.get("source_batch_offset"),
                    _safe_int(state.get("resume_batch_offset"), 0),
                )
                local_batch_index = int(payload.get("batch_index", 0))
                idx = source_batch_offset + local_batch_index + 1
                total = payload.get("batch_total")
                total_int = _safe_int(total, 0)
                if total_int > 0:
                    state["map_batches_total"] = max(
                        _safe_int(state.get("map_batches_total"), 0),
                        source_batch_offset + total_int,
                    )
                retries_label = "∞" if int(max_retries) < 0 else str(max_retries)
                state["active_step"] = (
                    f"LLM анализирует MAP-batch {idx}/{total}"
                    if total else f"LLM анализирует MAP-batch {idx}"
                )
                batch_plan = state.setdefault("batch_plan", [])
                if isinstance(batch_plan, list):
                    plan_item = {
                        "batch_id": idx,
                        "period": f"{payload.get('batch_period_start')} -> {payload.get('batch_period_end')}",
                        "rows": _safe_int(payload.get("batch_logs_count"), 0),
                        "tokens": "disabled",
                        "status": "В Работе",
                        "reason": str(payload.get("split_reason") or ""),
                    }
                    existing_idx = next(
                        (
                            i
                            for i, item in enumerate(batch_plan)
                            if _safe_int((item or {}).get("batch_id"), -1) == idx
                        ),
                        -1,
                    )
                    if existing_idx >= 0:
                        batch_plan[existing_idx].update(plan_item)
                    else:
                        batch_plan.append(plan_item)
                    state["batch_plan"] = batch_plan
                events.append(
                    (
                        f"LLM анализирует batch {idx}/{total} "
                        f"(таймаут {llm_timeout}s, ретраи {retries_label})"
                    )
                    if total else
                    f"LLM анализирует batch {idx} (таймаут {llm_timeout}s, ретраи {retries_label})"
                )
            elif event == "map_batch_resize":
                old_size = int(pd.to_numeric(payload.get("old_chunk_size"), errors="coerce") or 0)
                new_size = int(pd.to_numeric(payload.get("new_chunk_size"), errors="coerce") or 0)
                state["batch_split_count"] = _safe_int(state.get("batch_split_count"), 0) + 1
                events.append(
                    "Batch auto-shrink: "
                    f"400 Bad Request, уменьшаем размер batch {old_size} -> {new_size}"
                )
            elif event == "map_parallel_disabled":
                workers = int(pd.to_numeric(payload.get("map_workers_requested"), errors="coerce") or 0)
                events.append(
                    "Параллельный MAP отключен для auto-shrink режима "
                    f"(запрошено workers={workers})."
                )
            elif event == "map_batch":
                state["status"] = "map"
                state["llm_phase_hint"] = "map"
                state["active_step"] = "MAP-batch обработан, обновляем промежуточный результат"
                full_logs = payload.get("batch_logs", [])
                if not isinstance(full_logs, list):
                    full_logs = []
                source_batch_offset = _safe_int(
                    state.get("source_batch_offset"),
                    _safe_int(state.get("resume_batch_offset"), 0),
                )
                local_batch_index = int(payload.get("batch_index", 0))
                global_batch_index_zero = source_batch_offset + local_batch_index
                global_batch_index_one = global_batch_index_zero + 1
                state["map_batches_done_total"] = max(
                    _safe_int(state.get("map_batches_done_total"), 0),
                    global_batch_index_one,
                )
                local_total_int = _safe_int(payload.get("batch_total"), 0)
                if local_total_int > 0:
                    state["map_batches_total"] = max(
                        _safe_int(state.get("map_batches_total"), 0),
                        source_batch_offset + local_total_int,
                    )

                # Track the timestamp of the last processed batch for timestamp-based
                # progress bar and ETA — no pre-counting of rows required.
                batch_period_end = payload.get("batch_period_end")
                if batch_period_end:
                    state["last_batch_ts"] = batch_period_end

                _append_jsonl(
                    live_batches_path,
                    {
                        "ts": datetime.now(MSK).isoformat(),
                        "event": "map_batch",
                        "batch_index": global_batch_index_zero,
                        "batch_total": payload.get("batch_total"),
                        "batch_index_local": local_batch_index,
                        "batch_summary": payload.get("batch_summary"),
                        "batch_logs_count": payload.get("batch_logs_count"),
                        "batch_period_start": payload.get("batch_period_start"),
                        "batch_period_end": payload.get("batch_period_end"),
                        "batch_logs": full_logs,
                    },
                )

                src_label = str(state.get("active_source_label") or "query_1")
                src_file_label = _safe_filename(src_label)
                batch_summary_text = _normalize_summary_text(payload.get("batch_summary"))
                if batch_summary_text:
                    _append_jsonl(
                        map_summaries_jsonl_path,
                        {
                            "ts": datetime.now(MSK).isoformat(),
                            "source_name": src_label,
                            "batch_index": global_batch_index_one,
                            "batch_total": payload.get("batch_total"),
                            "batch_index_local": local_batch_index + 1,
                            "batch_period_start": payload.get("batch_period_start"),
                            "batch_period_end": payload.get("batch_period_end"),
                            "batch_logs_count": payload.get("batch_logs_count"),
                            "batch_summary": batch_summary_text,
                        },
                    )
                    _write_text_file(
                        map_summaries_dir / f"{src_file_label}_map_{global_batch_index_one:04d}.md",
                        "\n".join(
                            [
                                f"# MAP Summary #{global_batch_index_one}",
                                f"- source: `{src_label}`",
                                f"- period: `{payload.get('batch_period_start')}` -> `{payload.get('batch_period_end')}`",
                                f"- logs_count: `{payload.get('batch_logs_count')}`",
                                "",
                                batch_summary_text,
                            ]
                        ),
                    )

                preview_logs = full_logs
                batch_item = {
                    "batch_index": global_batch_index_zero,
                    "batch_total": payload.get("batch_total"),
                    "batch_index_local": local_batch_index,
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

                batch_plan = state.setdefault("batch_plan", [])
                if isinstance(batch_plan, list):
                    plan_item = {
                        "batch_id": global_batch_index_one,
                        "period": f"{payload.get('batch_period_start')} -> {payload.get('batch_period_end')}",
                        "rows": _safe_int(payload.get("batch_logs_count"), len(full_logs)),
                        "tokens": "disabled",
                        "status": "Готов",
                        "reason": str(payload.get("split_reason") or ""),
                    }
                    existing_idx = next(
                        (
                            i
                            for i, item in enumerate(batch_plan)
                            if _safe_int((item or {}).get("batch_id"), -1) == global_batch_index_one
                        ),
                        -1,
                    )
                    if existing_idx >= 0:
                        batch_plan[existing_idx].update(plan_item)
                    else:
                        batch_plan.append(plan_item)
                    state["batch_plan"] = batch_plan

                total = payload.get("batch_total")
                events.append(
                    f"Map summary {global_batch_index_one}/{total}"
                    if total else f"Map summary {global_batch_index_one}"
                )
            elif event == "map_done":
                state["status"] = "reduce"
                state["llm_phase_hint"] = "reduce"
                state["active_step"] = "MAP завершён, готовим REDUCE"
                source_batch_offset = _safe_int(
                    state.get("source_batch_offset"),
                    _safe_int(state.get("resume_batch_offset"), 0),
                )
                local_total_int = _safe_int(payload.get("batch_total"), 0)
                if local_total_int > 0:
                    finished_total = source_batch_offset + local_total_int
                    state["map_batches_done_total"] = max(
                        _safe_int(state.get("map_batches_done_total"), 0),
                        finished_total,
                    )
                    state["map_batches_total"] = max(
                        _safe_int(state.get("map_batches_total"), 0),
                        finished_total,
                    )
                events.append("Map этап завершен")
                # Warn if processing stopped early due to DB errors
                if state.get("query_errors"):
                    events.append(
                        "⚠️ ВНИМАНИЕ: часть данных не получена из БД — суммаризация неполная. "
                        "Проверьте ошибки ClickHouse ниже."
                    )
            elif event == "reduce_start":
                state["status"] = "reduce"
                state["llm_phase_hint"] = "reduce"
                state["active_step"] = "REDUCE: объединяем промежуточные summary"
                # Build static round plan (L1/L2/...) progressively via reduce_group events.
                if not isinstance(state.get("reduce_nodes"), list):
                    state["reduce_nodes"] = []
                events.append("Reduce этап запущен")
            elif event == "reduce_group_start":
                state["status"] = "reduce"
                state["llm_phase_hint"] = "reduce"
                round_idx = payload.get("reduce_round")
                group_index = int(payload.get("group_index", 0)) + 1
                group_total = payload.get("group_total")
                state["active_step"] = f"REDUCE round {round_idx}, группа {group_index}/{group_total}"
                state["active_reduce_round"] = _safe_int(round_idx, 1)
                _upsert_reduce_node(
                    round_idx=_safe_int(round_idx, 1),
                    group_idx=group_index,
                    updates={
                        "status": "active",
                        "group_total": _safe_int(group_total, 0),
                        "group_size": _safe_int(payload.get("group_size"), 0),
                        "input_events": _safe_int(payload.get("input_events"), 0),
                        "input_hypotheses": _safe_int(payload.get("input_hypotheses"), 0),
                        "input_tokens": _safe_int(payload.get("input_tokens"), 0),
                        "split_reason": str(payload.get("split_reason") or ""),
                    },
                )
                events.append(f"Reduce round {round_idx}: группа {group_index}/{group_total}")
            elif event == "reduce_group_done":
                state["status"] = "reduce"
                round_idx = _safe_int(payload.get("reduce_round"), _safe_int(state.get("active_reduce_round"), 1))
                group_index = _safe_int(payload.get("group_index"), 0) + 1
                _upsert_reduce_node(
                    round_idx=round_idx,
                    group_idx=group_index,
                    updates={
                        "status": "done",
                        "group_total": _safe_int(payload.get("group_total"), 0),
                        "group_size": _safe_int(payload.get("group_size"), 0),
                        "output_events": _safe_int(payload.get("output_events"), 0),
                        "output_hypotheses": _safe_int(payload.get("output_hypotheses"), 0),
                        "output_tokens": _safe_int(payload.get("output_tokens"), 0),
                        "gaps_closed": _safe_int(payload.get("gaps_closed"), 0),
                        "new_causal_links": _safe_int(payload.get("new_causal_links"), 0),
                        "compression_pct": _safe_float(payload.get("compression_pct"), float("nan")),
                    },
                )
            elif event == "reduce_context_fallback":
                state["reduce_split_count"] = _safe_int(state.get("reduce_split_count"), 0) + 1
                reason = str(payload.get("reason") or "context overflow")
                round_idx = _safe_int(payload.get("reduce_round"), _safe_int(state.get("active_reduce_round"), 1))
                _upsert_reduce_node(
                    round_idx=round_idx,
                    group_idx=_safe_int(payload.get("group_index"), 0) + 1,
                    updates={
                        "status": "split",
                        "split_reason": reason,
                    },
                )
                events.append(f"Reduce split: {reason}")
            elif event == "reduce_done":
                state["status"] = "summary_ready"
                state["llm_phase_hint"] = "freeform"
                state["active_step"] = "REDUCE завершён, собираем финальный отчёт"
                final_from_payload = _normalize_summary_text(payload.get("summary"))
                if final_from_payload:
                    state["final_summary"] = final_from_payload
                    state["final_summary_origin"] = "reduce_done_event"
                    src_label = str(state.get("active_source_label") or "query_1")
                    src_file_label = _safe_filename(src_label)
                    _append_jsonl(
                        reduce_summaries_jsonl_path,
                        {
                            "ts": datetime.now(MSK).isoformat(),
                            "source_name": src_label,
                            "reduce_round": payload.get("reduce_round"),
                            "summary": final_from_payload,
                        },
                    )
                    _write_text_file(
                        reduce_summaries_dir / f"{src_file_label}_reduce_final.md",
                        "\n".join(
                            [
                                "# REDUCE Final Summary",
                                f"- source: `{src_label}`",
                                f"- period: `{state.get('period_start')}` -> `{state.get('period_end')}`",
                                "",
                                final_from_payload,
                            ]
                        ),
                    )
                events.append("Reduce этап завершен")
            elif event == "fetch_error":
                error_msg = str(payload.get("error", "ClickHouse query failed"))
                _register_query_error(error_msg)
                events.append(f"ClickHouse error: {error_msg}")
            elif event == "freeform_start":
                state["llm_phase_hint"] = "freeform"
                state["active_step"] = "LLM пишет финальный narrative-отчёт"
                events.append("Генерация финального нарратива...")
            elif event == "freeform_done":
                state["llm_phase_hint"] = "freeform"
                state["active_step"] = "Финальный narrative-отчёт готов"
                freeform = payload.get("freeform_summary")
                freeform_text = _normalize_summary_text(freeform)
                if freeform_text:
                    state["freeform_final_summary"] = freeform_text
                    _write_text_file(
                        final_summaries_dir / "final_freeform.md",
                        "\n".join(
                            [
                                "# Final Freeform Summary",
                                f"- period: `{state.get('period_start')}` -> `{state.get('period_end')}`",
                                "",
                                freeform_text,
                            ]
                        ),
                    )
                events.append("Нарратив готов")
            elif event in {"verification_start", "verify_start"}:
                state["status"] = "summary_ready"
                state["active_step"] = "Верификация итогового summary"
                state["verification"] = {
                    "summary": str(payload.get("summary") or "Запущен контрольный проход"),
                    "selected_logs": payload.get("selected_logs") or [],
                    "corrections": [],
                }
            elif event in {"verification_done", "verify_done"}:
                state["status"] = "summary_ready"
                state["verification"] = {
                    "summary": str(payload.get("summary") or "Верификация завершена"),
                    "selected_logs": payload.get("selected_logs") or [],
                    "corrections": payload.get("corrections") or [],
                }
            else:
                events.append(f"Событие: {event}")

            if len(events) > MAX_EVENT_LINES:
                state["events"] = events[-MAX_EVENT_LINES:]

            progress_rows = payload.get("rows_processed")
            if progress_rows is None:
                progress_rows = payload.get("rows_fetched")
            if progress_rows is not None:
                resume_rows_offset = int(pd.to_numeric(state.get("resume_rows_offset"), errors="coerce") or 0)
                state["logs_processed"] = max(int(progress_rows) + resume_rows_offset, 0)
            if payload.get("rows_total") is not None:
                state["logs_total"] = int(payload.get("rows_total"))

            _estimate_eta(state, event, payload)
            _render_logs_summary_chat(analysis_placeholder, state, deps)
            _persist_checkpoint(session_checkpoint_path, state)
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
                        _finish_read_timeout_episode(resolution="exhausted")
                    state["active_step"] = "LLM попытки исчерпаны, завершаем шаг с ошибкой"
                    _push_live_event(
                        f"LLM ошибка ({elapsed_sec:.1f}s): {error_text}. Попытки исчерпаны.",
                        render_now=True,
                    )
            _persist_checkpoint(session_checkpoint_path, state)

        selected_model = str(state.get("llm_model_id") or llm_model_id or "").strip()
        if selected_model:
            try:
                settings.LLM_MODEL_ID = selected_model
            except Exception:
                pass

        base_llm_call = deps.make_llm_call(
            max_retries=max_retries,
            on_retry=_on_retry,
            on_attempt=_on_llm_attempt,
            on_result=_on_llm_result,
            llm_timeout=llm_timeout,
        )
        llm_call_counter = {"value": 0}

        def _trace_llm_call(
            *,
            phase: str,
            prompt_text: str,
            response_text: Optional[str],
            error_text: Optional[str] = None,
        ) -> None:
            llm_call_counter["value"] = int(llm_call_counter["value"]) + 1
            call_idx = int(llm_call_counter["value"])
            safe_phase = _safe_filename(phase or "unknown")
            call_dir = llm_calls_dir / safe_phase
            prompt_path = call_dir / f"call_{call_idx:04d}_prompt.txt"
            response_path = call_dir / f"call_{call_idx:04d}_response.txt"
            _write_text_file(prompt_path, prompt_text)
            if response_text is not None:
                _write_text_file(response_path, str(response_text))
            _append_jsonl(
                llm_calls_jsonl_path,
                {
                    "ts": datetime.now(MSK).isoformat(),
                    "call_index": call_idx,
                    "phase": phase or "unknown",
                    "prompt_path": str(prompt_path),
                    "response_path": str(response_path) if response_text is not None else None,
                    "error": error_text,
                    "source_name": state.get("active_source_label"),
                    "status": "error" if error_text else "ok",
                },
            )
            _persist_checkpoint(session_checkpoint_path, state)

        def _infer_phase_hint() -> str:
            hint = str(state.get("llm_phase_hint") or "").strip().lower()
            if hint:
                return hint
            status_hint = str(state.get("status") or "").strip().lower()
            if status_hint in {"map", "reduce", "summary_ready"}:
                return status_hint
            return "unknown"

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
            phase = _infer_phase_hint()
            try:
                response = base_llm_call(enriched_prompt)
                _trace_llm_call(
                    phase=phase,
                    prompt_text=enriched_prompt,
                    response_text=response,
                    error_text=None,
                )
                return response
            except Exception as exc:  # noqa: BLE001
                _trace_llm_call(
                    phase=phase,
                    prompt_text=enriched_prompt,
                    response_text=None,
                    error_text=str(exc),
                )
                raise

        llm_call = _llm_call_with_context

        def _maybe_generate_no_logs_hypothesis() -> None:
            if not bool(enable_no_logs_hypothesis):
                return
            normalized_summary = _normalize_summary_text(state.get("final_summary"))
            if normalized_summary and normalized_summary != "Нет логов за указанный период.":
                return
            events = state.setdefault("events", [])
            if isinstance(events, list):
                events.append("Логи не найдены: запускаем LLM-гипотезы по контексту инцидента")
            state["active_step"] = "Нет логов: формируем осторожные гипотезы"
            state["llm_phase_hint"] = "no_logs_hypothesis"
            state["active_source_label"] = "no_logs_hypothesis"
            _render_logs_summary_chat(analysis_placeholder, state, deps)
            hypothesis_prompt = _build_no_logs_hypothesis_prompt(
                period_start=period_start_iso,
                period_end=period_end_iso,
                user_goal=goal_text,
                metrics_context=metrics_context_text,
                logs_fetch_mode=logs_fetch_mode,
                logs_tail_limit=int(deps.logs_tail_limit),
                logs_queries_count=max(int(len(query_specs)), 1),
            )
            try:
                hypothesis_text = _normalize_summary_text(llm_call(hypothesis_prompt))
                stats = state.get("stats")
                if isinstance(stats, dict):
                    stats["llm_calls"] = int(pd.to_numeric(stats.get("llm_calls", 0), errors="coerce") or 0) + 1
                if hypothesis_text:
                    state["final_summary"] = (
                        "Логи за выбранный период не найдены.\n\n"
                        "Ниже — осторожные гипотезы на основе доступного контекста "
                        "(это НЕ подтвержденные факты):\n\n"
                        f"{hypothesis_text}"
                    )
                    state["final_summary_origin"] = "no_logs_hypothesis"
                    if isinstance(events, list):
                        events.append("LLM-гипотезы при отсутствии логов готовы")
                else:
                    if isinstance(events, list):
                        events.append("LLM вернул пустой ответ для режима без логов")
            except Exception as no_logs_exc:  # noqa: BLE001
                deps.logger.warning("no-logs hypothesis generation failed: %s", no_logs_exc)
                if isinstance(events, list):
                    events.append(
                        "Не удалось получить гипотезы без логов: "
                        f"{str(no_logs_exc)}"
                    )

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
                        timestamp_column=logs_timestamp_column,
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
                    df = _sort_df_by_timestamp(df, timestamp_column=logs_timestamp_column)
                    if _uses_keyset and logs_timestamp_column in df.columns:
                        max_ts = df[logs_timestamp_column].apply(_to_msk_ts).max()
                        if not pd.isna(max_ts):
                            _last_ts[0] = max_ts.isoformat()
                    return [dict(row) for row in df.to_dict(orient="records")]

                return _fetch

            per_source_summaries: Dict[str, str] = {}
            agg_pages = agg_rows = agg_llm = agg_reduce = 0

            for src_idx, spec in enumerate(query_specs):
                src_label = str(spec.get("label", f"query_{src_idx + 1}"))
                state["active_source_label"] = src_label
                state.setdefault("events", []).append(
                    f"--- Источник: {src_label} ({src_idx + 1}/{len(query_specs)}) ---"
                )
                state.pop("last_batch_ts", None)  # reset timestamp progress for each source
                _render_logs_summary_chat(analysis_placeholder, state, deps)

                _src_prompt_context = {
                    "incident_start": period_start_iso,
                    "incident_end": period_end_iso,
                    "incident_description": goal_text,
                    "alerts_list": goal_text,
                    "metrics_context": metrics_context_text,
                    "source_name": src_label,
                    "sql_query": str(spec.get("template", "")),
                    "time_column": logs_timestamp_column,
                    "data_type": "",
                }
                try:
                    src_summarizer = deps.period_log_summarizer_cls(
                        db_fetch_page=_make_source_fetch_page(spec, src_label),
                        llm_call=llm_call,
                        config=_build_config(deps, db_batch_size, llm_batch_size, map_workers),
                        on_progress=_on_progress,
                        prompt_context=_src_prompt_context,
                    )
                except TypeError:
                    src_summarizer = deps.period_log_summarizer_cls(
                        db_fetch_page=_make_source_fetch_page(spec, src_label),
                        llm_call=llm_call,
                        config=_build_config(deps, db_batch_size, llm_batch_size, map_workers),
                        on_progress=_on_progress,
                    )
                try:
                    src_result = src_summarizer.summarize_period(
                        period_start=effective_period_start_iso,
                        period_end=period_end_iso,
                        columns=columns,
                        total_rows_estimate=None,
                    )
                except TypeError:
                    src_result = src_summarizer.summarize_period(
                        period_start=effective_period_start_iso,
                        period_end=period_end_iso,
                        columns=columns,
                    )

                src_summary = _normalize_summary_text(getattr(src_result, "summary", ""))
                if not src_summary or src_summary == "Нет логов за указанный период.":
                    source_map_summaries = _load_map_summaries_from_jsonl_for_source(
                        str(map_summaries_jsonl_path),
                        src_label,
                    )
                    if source_map_summaries:
                        state.setdefault("events", []).append(
                            f"{src_label}: source summary пустой/\"Нет логов\", пересобираем из MAP summary"
                        )
                        try:
                            from my_summarizer import (  # noqa: PLC0415
                                regenerate_reduce_summary_from_map_summaries,
                            )

                            rebuilt_src = _normalize_summary_text(
                                regenerate_reduce_summary_from_map_summaries(
                                    map_summaries=source_map_summaries,
                                    period_start=str(period_start_iso or ""),
                                    period_end=str(period_end_iso or ""),
                                    llm_call=llm_call,
                                    config=_build_config(
                                        deps, db_batch_size, llm_batch_size, map_workers
                                    ),
                                    prompt_context={
                                        "incident_start": period_start_iso,
                                        "incident_end": period_end_iso,
                                        "incident_description": goal_text,
                                        "alerts_list": goal_text,
                                        "metrics_context": metrics_context_text,
                                        "source_name": src_label,
                                        "sql_query": str(spec.get("template", "")),
                                        "time_column": logs_timestamp_column,
                                        "data_type": "",
                                    },
                                )
                            )
                            src_summary = rebuilt_src or "\n\n---\n\n".join(source_map_summaries)
                        except Exception as rebuild_exc:  # noqa: BLE001
                            deps.logger.warning(
                                "%s: failed to rebuild source summary from MAP summaries: %s",
                                src_label,
                                rebuild_exc,
                            )
                            src_summary = "\n\n---\n\n".join(source_map_summaries)
                if src_summary and src_summary != "Нет логов за указанный период.":
                    per_source_summaries[src_label] = src_summary
                agg_pages += src_result.pages_fetched
                agg_rows += src_result.rows_processed
                agg_llm += src_result.llm_calls
                agg_reduce += src_result.reduce_rounds

            # Cross-source REDUCE: merge per-source summaries into a single report
            final_summary_origin = "no_logs"
            if len(per_source_summaries) > 1:
                state.setdefault("events", []).append("Кросс-источниковый анализ (финальный REDUCE)...")
                state["llm_phase_hint"] = "cross_reduce"
                state["active_source_label"] = "cross_source"
                _render_logs_summary_chat(analysis_placeholder, state, deps)
                cross_prompt = build_cross_source_reduce_prompt(
                    summaries_by_source=per_source_summaries,
                    period_start=period_start_iso,
                    period_end=period_end_iso,
                    context={
                        "incident_start": period_start_iso,
                        "incident_end": period_end_iso,
                        "incident_description": goal_text,
                        "alerts_list": goal_text,
                        "metrics_context": metrics_context_text,
                        "source_name": "cross_source",
                        "sql_query": sql_query_clean,
                        "time_column": logs_timestamp_column,
                        "data_type": "",
                    },
                )
                try:
                    cross_summary = llm_call(cross_prompt).strip()
                    agg_llm += 1
                    final_summary_text = _normalize_summary_text(cross_summary) or "\n\n---\n\n".join(
                        f"=== {src} ===\n{s}" for src, s in per_source_summaries.items()
                    )
                    final_summary_origin = (
                        "cross_reduce_llm"
                        if _normalize_summary_text(cross_summary)
                        else "cross_reduce_fallback_join"
                    )
                except Exception as cross_exc:  # noqa: BLE001
                    deps.logger.warning("Cross-source reduce failed: %s", cross_exc)
                    final_summary_text = "\n\n---\n\n".join(
                        f"=== {src} ===\n{s}" for src, s in per_source_summaries.items()
                    )
                    final_summary_origin = "cross_reduce_fallback_join"
                state.setdefault("events", []).append("Кросс-источниковый анализ завершён")
            elif per_source_summaries:
                final_summary_text = next(iter(per_source_summaries.values()))
                final_summary_origin = "per_source_reduce_direct"
            else:
                cached_map_summaries = _load_map_summaries_from_jsonl(str(map_summaries_jsonl_path))
                if cached_map_summaries:
                    state.setdefault("events", []).append(
                        "Обнаружены MAP summary, но финальный итог пустой — пересобираем REDUCE из сохранённых MAP summary"
                    )
                    try:
                        from my_summarizer import (  # noqa: PLC0415
                            regenerate_reduce_summary_from_map_summaries,
                        )

                        rebuilt_multi = regenerate_reduce_summary_from_map_summaries(
                            map_summaries=cached_map_summaries,
                            period_start=str(period_start_iso or ""),
                            period_end=str(period_end_iso or ""),
                            llm_call=llm_call,
                            config=_build_config(deps, db_batch_size, llm_batch_size, map_workers),
                            prompt_context={
                                "incident_start": period_start_iso,
                                "incident_end": period_end_iso,
                                "incident_description": goal_text,
                                "alerts_list": goal_text,
                                "metrics_context": metrics_context_text,
                                "source_name": "cached_map_summaries",
                                "sql_query": sql_query_clean,
                                "time_column": logs_timestamp_column,
                                "data_type": "",
                            },
                        )
                        final_summary_text = _normalize_summary_text(rebuilt_multi)
                    except Exception as rebuild_exc:  # noqa: BLE001
                        deps.logger.warning(
                            "fallback reduce rebuild from cached MAP summaries failed: %s",
                            rebuild_exc,
                        )
                        final_summary_text = ""
                    if not final_summary_text:
                        final_summary_text = "\n\n---\n\n".join(cached_map_summaries)
                        final_summary_origin = "recovered_from_map_join"
                    else:
                        final_summary_origin = "recovered_from_map_reduce"
                else:
                    final_summary_text = "Нет логов за указанный период."
                    final_summary_origin = "no_logs"

            normalized_multi_final = _normalize_summary_text(final_summary_text)
            if normalized_multi_final:
                _append_jsonl(
                    reduce_summaries_jsonl_path,
                    {
                        "ts": datetime.now(MSK).isoformat(),
                        "source_name": "cross_source" if len(per_source_summaries) > 1 else "query_1",
                        "summary": normalized_multi_final,
                    },
                )
                _write_text_file(
                    reduce_summaries_dir / "cross_source_reduce_final.md",
                    "\n".join(
                        [
                            "# Cross-source REDUCE Final Summary",
                            f"- period: `{period_start_iso}` -> `{period_end_iso}`",
                            f"- sources: `{len(per_source_summaries)}`",
                            "",
                            normalized_multi_final,
                        ]
                    ),
                )

            state["status"] = "done"
            state["active_step"] = "Суммаризация завершена"
            state["final_summary"] = _normalize_summary_text(final_summary_text) or "Нет логов за указанный период."
            state["final_summary_origin"] = final_summary_origin
            stats_offset = state.get("resume_stats_offset") if isinstance(state.get("resume_stats_offset"), dict) else {}
            pages_offset = int(pd.to_numeric((stats_offset or {}).get("pages_fetched"), errors="coerce") or 0)
            rows_offset_stats = int(pd.to_numeric((stats_offset or {}).get("rows_processed"), errors="coerce") or 0)
            llm_offset = int(pd.to_numeric((stats_offset or {}).get("llm_calls"), errors="coerce") or 0)
            reduce_offset = int(pd.to_numeric((stats_offset or {}).get("reduce_rounds"), errors="coerce") or 0)
            state["stats"] = {
                "pages_fetched": pages_offset + agg_pages,
                "rows_processed": rows_offset_stats + agg_rows,
                "llm_calls": llm_offset + agg_llm,
                "reduce_rounds": reduce_offset + agg_reduce,
            }
            _enrich_stats_with_elapsed(state)
            resume_rows_offset = int(pd.to_numeric(state.get("resume_rows_offset"), errors="coerce") or 0)
            state["logs_processed"] = max(int(pd.to_numeric(state.get("logs_processed"), errors="coerce") or 0), resume_rows_offset + agg_rows)
        else:
            # Single-query or demo mode: use _db_fetch_page directly
            state["active_source_label"] = "query_1"
            _single_prompt_context = {
                "incident_start": period_start_iso,
                "incident_end": period_end_iso,
                "incident_description": goal_text,
                "alerts_list": goal_text,
                "metrics_context": metrics_context_text,
                "source_name": "query_1",
                "sql_query": sql_query_clean,
                "time_column": logs_timestamp_column,
                "data_type": "",
            }
            try:
                summarizer = deps.period_log_summarizer_cls(
                    db_fetch_page=_db_fetch_page,
                    llm_call=llm_call,
                    config=_build_config(deps, db_batch_size, llm_batch_size, map_workers),
                    on_progress=_on_progress,
                    prompt_context=_single_prompt_context,
                )
            except TypeError:
                summarizer = deps.period_log_summarizer_cls(
                    db_fetch_page=_db_fetch_page,
                    llm_call=llm_call,
                    config=_build_config(deps, db_batch_size, llm_batch_size, map_workers),
                    on_progress=_on_progress,
                )
            try:
                result = summarizer.summarize_period(
                    period_start=effective_period_start_iso,
                    period_end=period_end_iso,
                    columns=columns,
                    total_rows_estimate=total_rows_estimate,
                )
            except TypeError:
                result = summarizer.summarize_period(
                    period_start=effective_period_start_iso,
                    period_end=period_end_iso,
                    columns=columns,
                )

            state["status"] = "done"
            state["active_step"] = "Суммаризация завершена"
            normalized_final_summary = _normalize_summary_text(getattr(result, "summary", None))
            single_final_origin = (
                "single_reduce_direct"
                if normalized_final_summary and normalized_final_summary != "Нет логов за указанный период."
                else "no_logs"
            )
            if (
                not normalized_final_summary
                or normalized_final_summary == "Нет логов за указанный период."
            ):
                cached_map_summaries = _load_map_summaries_from_jsonl(str(map_summaries_jsonl_path))
                if cached_map_summaries:
                    state.setdefault("events", []).append(
                        "Обнаружены MAP summary, но итог пустой — пересобираем REDUCE из сохранённых MAP summary"
                    )
                    try:
                        from my_summarizer import (  # noqa: PLC0415
                            regenerate_reduce_summary_from_map_summaries,
                        )

                        rebuilt_single = regenerate_reduce_summary_from_map_summaries(
                            map_summaries=cached_map_summaries,
                            period_start=str(period_start_iso or ""),
                            period_end=str(period_end_iso or ""),
                            llm_call=llm_call,
                            config=_build_config(deps, db_batch_size, llm_batch_size, map_workers),
                            prompt_context={
                                "incident_start": period_start_iso,
                                "incident_end": period_end_iso,
                                "incident_description": goal_text,
                                "alerts_list": goal_text,
                                "metrics_context": metrics_context_text,
                                "source_name": "cached_map_summaries",
                                "sql_query": sql_query_clean,
                                "time_column": logs_timestamp_column,
                                "data_type": "",
                            },
                        )
                        rebuilt_single_norm = _normalize_summary_text(rebuilt_single)
                        if rebuilt_single_norm:
                            normalized_final_summary = rebuilt_single_norm
                            single_final_origin = "single_recovered_from_map_reduce"
                    except Exception as rebuild_exc:  # noqa: BLE001
                        deps.logger.warning(
                            "fallback single-source reduce rebuild from cached MAP summaries failed: %s",
                            rebuild_exc,
                        )
                    if (
                        not normalized_final_summary
                        or normalized_final_summary == "Нет логов за указанный период."
                    ):
                        normalized_final_summary = "\n\n---\n\n".join(cached_map_summaries)
                        single_final_origin = "single_recovered_from_map_join"
            state["final_summary"] = normalized_final_summary or "Нет логов за указанный период."
            state["final_summary_origin"] = single_final_origin
            stats_offset = state.get("resume_stats_offset") if isinstance(state.get("resume_stats_offset"), dict) else {}
            pages_offset = int(pd.to_numeric((stats_offset or {}).get("pages_fetched"), errors="coerce") or 0)
            rows_offset_stats = int(pd.to_numeric((stats_offset or {}).get("rows_processed"), errors="coerce") or 0)
            llm_offset = int(pd.to_numeric((stats_offset or {}).get("llm_calls"), errors="coerce") or 0)
            reduce_offset = int(pd.to_numeric((stats_offset or {}).get("reduce_rounds"), errors="coerce") or 0)
            state["stats"] = {
                "pages_fetched": pages_offset + int(result.pages_fetched),
                "rows_processed": rows_offset_stats + int(result.rows_processed),
                "llm_calls": llm_offset + int(result.llm_calls),
                "reduce_rounds": reduce_offset + int(result.reduce_rounds),
            }
            _enrich_stats_with_elapsed(state)
            resume_rows_offset = int(pd.to_numeric(state.get("resume_rows_offset"), errors="coerce") or 0)
            state["logs_processed"] = max(
                int(pd.to_numeric(state.get("logs_processed"), errors="coerce") or 0),
                resume_rows_offset + int(result.rows_processed),
            )
            if state.get("logs_total") is None and total_rows_estimate is not None:
                state["logs_total"] = int(total_rows_estimate)

        _maybe_generate_no_logs_hypothesis()

        if is_resume_continue:
            try:
                cached_map_summaries = _load_map_summaries_from_jsonl(str(map_summaries_jsonl_path))
                if cached_map_summaries:
                    state.setdefault("events", []).append(
                        f"Восстановление: пересобираем итог по {len(cached_map_summaries)} сохранённым MAP summary"
                    )
                    state["llm_phase_hint"] = "resume_rereduce"
                    state["active_source_label"] = "resume_rereduce"
                    _render_logs_summary_chat(analysis_placeholder, state, deps)
                    from my_summarizer import (  # noqa: PLC0415
                        regenerate_reduce_summary_from_map_summaries,
                    )

                    rebuilt_from_resume = _normalize_summary_text(
                        regenerate_reduce_summary_from_map_summaries(
                            map_summaries=cached_map_summaries,
                            period_start=period_start_iso,
                            period_end=period_end_iso,
                            llm_call=llm_call,
                            config=_build_config(deps, db_batch_size, llm_batch_size, map_workers),
                            prompt_context={
                                "incident_start": period_start_iso,
                                "incident_end": period_end_iso,
                                "incident_description": goal_text,
                                "alerts_list": goal_text,
                                "metrics_context": metrics_context_text,
                                "source_name": "resume_rereduce",
                                "sql_query": sql_query_clean,
                                "time_column": logs_timestamp_column,
                                "data_type": "",
                            },
                        )
                    )
                    if rebuilt_from_resume:
                        state["final_summary"] = rebuilt_from_resume
                        state["final_summary_origin"] = "resume_rereduce"
                        state.setdefault("events", []).append(
                            "Восстановление: итоговый Reduce summary пересобран успешно"
                        )
                        stats_payload = state.get("stats")
                        if isinstance(stats_payload, dict):
                            stats_payload["llm_calls"] = int(
                                pd.to_numeric(stats_payload.get("llm_calls", 0), errors="coerce") or 0
                            ) + 1
                        _write_text_file(
                            reduce_summaries_dir / "resume_rebuild_reduce_final.md",
                            "\n".join(
                                [
                                    "# Resume Rebuilt REDUCE Final Summary",
                                    f"- rebuilt_at: `{datetime.now(MSK).isoformat()}`",
                                    f"- map_summaries_count: `{len(cached_map_summaries)}`",
                                    "",
                                    rebuilt_from_resume,
                                ]
                            ),
                        )
                else:
                    state.setdefault("events", []).append(
                        "Восстановление: нет сохранённых MAP summary, пересборка пропущена"
                    )
            except Exception as resume_exc:  # noqa: BLE001
                deps.logger.warning("resume re-reduce failed: %s", resume_exc)
                state.setdefault("events", []).append(
                    f"Восстановление: пересборка итога не удалась ({str(resume_exc)})"
                )

        final_summary_for_report = _normalize_summary_text(state.get("final_summary"))
        if (
            not final_summary_for_report
            or final_summary_for_report == "Нет логов за указанный период."
        ):
            recovered_map_summaries = _load_map_summaries_from_jsonl(str(map_summaries_jsonl_path))
            if not recovered_map_summaries:
                recovered_map_summaries = [
                    _normalize_summary_text(item.get("batch_summary"))
                    for item in (state.get("map_batches") or [])
                    if isinstance(item, dict)
                ]
                recovered_map_summaries = [item for item in recovered_map_summaries if item]
            if recovered_map_summaries:
                state.setdefault("events", []).append(
                    "Финальный summary был пустой/\"Нет логов\", но MAP summary есть — восстанавливаем итог"
                )
                try:
                    from my_summarizer import (  # noqa: PLC0415
                        regenerate_reduce_summary_from_map_summaries,
                    )

                    rebuilt_summary = _normalize_summary_text(
                        regenerate_reduce_summary_from_map_summaries(
                            map_summaries=recovered_map_summaries,
                            period_start=period_start_iso,
                            period_end=period_end_iso,
                            llm_call=llm_call,
                            config=_build_config(deps, db_batch_size, llm_batch_size, map_workers),
                            prompt_context={
                                "incident_start": period_start_iso,
                                "incident_end": period_end_iso,
                                "incident_description": goal_text,
                                "alerts_list": goal_text,
                                "metrics_context": metrics_context_text,
                                "source_name": "final_recovery",
                                "sql_query": sql_query_clean,
                                "time_column": logs_timestamp_column,
                                "data_type": "",
                            },
                        )
                    )
                    if rebuilt_summary:
                        final_summary_for_report = rebuilt_summary
                        state["final_summary_origin"] = "final_recovery_reduce"
                    else:
                        final_summary_for_report = "\n\n---\n\n".join(recovered_map_summaries)
                        state["final_summary_origin"] = "final_recovery_join"
                except Exception as recover_exc:  # noqa: BLE001
                    deps.logger.warning("final summary recovery from MAP summaries failed: %s", recover_exc)
                    final_summary_for_report = "\n\n---\n\n".join(recovered_map_summaries)
                    state["final_summary_origin"] = "final_recovery_join"
                state["final_summary"] = final_summary_for_report

        report_steps_total = len(FINAL_REPORT_SECTIONS) * 2 + 4

        def _set_report_progress(
            label: str,
            *,
            step_inc: int = 0,
            mark_active: Optional[bool] = None,
            render_now: bool = False,
        ) -> None:
            total = max(
                _safe_int(state.get("report_progress_total"), 0),
                report_steps_total,
                1,
            )
            current = _safe_int(state.get("report_progress_current"), 0)
            current = min(max(current + max(step_inc, 0), 0), total)
            state["report_progress_total"] = total
            state["report_progress_current"] = current
            state["report_progress_label"] = str(label or "").strip()
            if mark_active is not None:
                state["report_progress_active"] = bool(mark_active)
            if render_now:
                _render_logs_summary_chat(analysis_placeholder, state, deps)

        if final_summary_for_report and final_summary_for_report != "Нет логов за указанный период.":
            state["report_progress_total"] = report_steps_total
            state["report_progress_current"] = 0
            state["report_progress_active"] = True
            _set_report_progress("Подготовка итогового отчёта", render_now=True)
            try:
                events = state.setdefault("events", [])
                state["structured_sections"] = []
                events.append("Готовим структурированный финальный отчет по секциям")
                state["llm_phase_hint"] = "final_structured"
                state["active_source_label"] = "final_structured"
                _set_report_progress("Структурированный отчёт: старт", render_now=True)
                _render_logs_summary_chat(analysis_placeholder, state, deps)

                def _on_structured_section_start(
                    section_idx: int,
                    section_total: int,
                    title: str,
                ) -> None:
                    state["active_step"] = (
                        f"Структурированный отчёт: секция {section_idx}/{section_total} — {title}"
                    )
                    _set_report_progress(
                        f"Структурированный отчёт: секция {section_idx}/{section_total}",
                        render_now=False,
                    )
                    _push_live_event(
                        f"LLM пишет structured секцию {section_idx}/{section_total}: {title}",
                        render_now=True,
                    )

                def _on_structured_section_done(
                    section_idx: int,
                    section_total: int,
                    title: str,
                ) -> None:
                    _set_report_progress(
                        f"Структурированный отчёт: секция {section_idx}/{section_total} готова",
                        step_inc=1,
                        render_now=False,
                    )
                    _push_live_event(
                        f"Structured секция готова {section_idx}/{section_total}: {title}",
                        render_now=True,
                    )

                structured_summary, structured_sections = _generate_sectional_structured_summary(
                    llm_call=llm_call,
                    base_summary=final_summary_for_report,
                    user_goal=goal_text,
                    period_start=period_start_iso,
                    period_end=period_end_iso,
                    stats=state.get("stats") or {},
                    metrics_context=metrics_context_text,
                    on_section_start=_on_structured_section_start,
                    on_section_done=_on_structured_section_done,
                )
                structured_summary = _normalize_summary_text(structured_summary)
                if structured_summary:
                    final_summary_for_report = structured_summary
                    state["final_summary"] = structured_summary
                    state["structured_sections"] = structured_sections
                    events.append("Структурированный финальный отчет готов")
                else:
                    events.append("Структурированный секционный отчет пустой, оставляем reduce summary")
                _set_report_progress("Структурированный отчёт: этап завершён", step_inc=1, render_now=True)
            except Exception as structured_exc:  # noqa: BLE001
                deps.logger.warning("structured sectional report generation failed: %s", structured_exc)
                state.setdefault("events", []).append(
                    "Не удалось сгенерировать структурированный секционный отчет, оставляем reduce summary"
                )
                _set_report_progress(
                    "Структурированный отчёт: ошибка, продолжаем дальше",
                    step_inc=1,
                    render_now=True,
                )

            try:
                events = state.setdefault("events", [])
                state["final_summary"] = final_summary_for_report
                state["freeform_sections"] = []
                events.append("Готовим расширенный финальный отчет в свободном формате")
                state["llm_phase_hint"] = "final_freeform"
                state["active_source_label"] = "final_freeform"
                _set_report_progress("Свободный отчёт: старт", render_now=True)
                _render_logs_summary_chat(analysis_placeholder, state, deps)
                def _on_section_start(section_idx: int, section_total: int, title: str) -> None:
                    state["active_step"] = (
                        f"Финальный отчёт: секция {section_idx}/{section_total} — {title}"
                    )
                    _set_report_progress(
                        f"Свободный отчёт: секция {section_idx}/{section_total}",
                        render_now=False,
                    )
                    _push_live_event(
                        f"LLM пишет секцию {section_idx}/{section_total}: {title}",
                        render_now=True,
                    )

                def _on_section_done(section_idx: int, section_total: int, title: str) -> None:
                    _set_report_progress(
                        f"Свободный отчёт: секция {section_idx}/{section_total} готова",
                        step_inc=1,
                        render_now=False,
                    )
                    _push_live_event(
                        f"Секция готова {section_idx}/{section_total}: {title}",
                        render_now=True,
                    )

                freeform_summary, freeform_sections = _generate_sectional_freeform_summary(
                    llm_call=llm_call,
                    final_summary=final_summary_for_report,
                    user_goal=goal_text,
                    period_start=period_start_iso,
                    period_end=period_end_iso,
                    stats=state.get("stats") or {},
                    metrics_context=metrics_context_text,
                    on_section_start=_on_section_start,
                    on_section_done=_on_section_done,
                )
                freeform_summary = _normalize_summary_text(freeform_summary)
                if freeform_summary:
                    state["freeform_final_summary"] = freeform_summary
                    state["freeform_sections"] = freeform_sections
                    events.append("Свободный финальный отчет готов")
                else:
                    events.append("Свободный финальный отчет пустой, используем основной")
                _set_report_progress("Свободный отчёт: этап завершён", step_inc=1, render_now=True)
            except Exception as freeform_exc:  # noqa: BLE001
                deps.logger.warning("freeform final report generation failed: %s", freeform_exc)
                state.setdefault("events", []).append(
                    "Секционная генерация отчета не удалась, пробуем резервный один запрос"
                )
                try:
                    map_summaries_for_final = _load_map_summaries_from_jsonl(
                        str(map_summaries_jsonl_path)
                    )
                    map_summaries_text_for_final = "\n\n".join(
                        f"[MAP SUMMARY #{idx + 1}]\n{text}"
                        for idx, text in enumerate(map_summaries_for_final)
                    )
                    freeform_prompt = _build_freeform_summary_prompt(
                        final_summary=final_summary_for_report,
                        map_summaries_text=map_summaries_text_for_final,
                        user_goal=goal_text,
                        period_start=period_start_iso,
                        period_end=period_end_iso,
                        stats=state.get("stats") or {},
                        metrics_context=metrics_context_text,
                    )
                    fallback_freeform = _normalize_summary_text(llm_call(freeform_prompt))
                    if fallback_freeform:
                        state["freeform_final_summary"] = fallback_freeform
                        state.setdefault("events", []).append(
                            "Резервный freeform-отчет сгенерирован"
                        )
                except Exception as fallback_exc:  # noqa: BLE001
                    deps.logger.warning("fallback freeform generation failed: %s", fallback_exc)
                    state.setdefault("events", []).append(
                        "Не удалось сгенерировать свободный финальный отчет"
                    )
                _set_report_progress(
                    "Свободный отчёт: ошибка, продолжаем сохранение артефактов",
                    step_inc=1,
                    render_now=True,
                )

        topic_titles = [title for title, _ in FINAL_REPORT_SECTIONS]
        preferred_structured_sections = (
            state.get("structured_sections")
            if isinstance(state.get("structured_sections"), list)
            else (
                state.get("freeform_sections")
                if isinstance(state.get("freeform_sections"), list)
                else None
            )
        )
        preferred_freeform_sections = (
            state.get("freeform_sections")
            if isinstance(state.get("freeform_sections"), list)
            else (
                state.get("structured_sections")
                if isinstance(state.get("structured_sections"), list)
                else None
            )
        )
        synced_structured, missing_in_structured = _ensure_report_topics_present(
            str(state.get("final_summary") or ""),
            topic_titles=topic_titles,
            preferred_sections=preferred_structured_sections,
        )
        if synced_structured:
            state["final_summary"] = synced_structured
        synced_freeform, missing_in_freeform = _ensure_report_topics_present(
            str(state.get("freeform_final_summary") or ""),
            topic_titles=topic_titles,
            preferred_sections=preferred_freeform_sections,
        )
        if synced_freeform:
            state["freeform_final_summary"] = synced_freeform
        if missing_in_structured:
            state.setdefault("events", []).append(
                "Синхронизация топиков: добавили недостающие разделы в структурированный отчет"
            )
        if missing_in_freeform:
            state.setdefault("events", []).append(
                "Синхронизация топиков: добавили недостающие разделы в свободный отчет"
            )
        if _safe_int(state.get("report_progress_total"), 0) > 0:
            _set_report_progress("Синхронизация топиков завершена", step_inc=1, render_now=True)

        final_structured_now = _normalize_summary_text(state.get("final_summary"))
        if final_structured_now:
            _write_text_file(
                final_summaries_dir / "final_structured.md",
                "\n".join(
                    [
                        "# Final Structured Summary",
                        f"- period: `{state.get('period_start')}` -> `{state.get('period_end')}`",
                        f"- queries_count: `{state.get('queries_count')}`",
                        "",
                        final_structured_now,
                    ]
                ),
            )
        final_structured_sections = state.get("structured_sections")
        if isinstance(final_structured_sections, list) and final_structured_sections:
            _write_json_file(
                final_summaries_dir / "final_structured_sections.json",
                {
                    "saved_at": datetime.now(MSK).isoformat(),
                    "sections": _json_safe(final_structured_sections),
                },
            )
            section_lines: List[str] = ["# Final Structured Sections", ""]
            for item in final_structured_sections:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip() or "Untitled Section"
                text = _normalize_summary_text(item.get("text"))
                section_lines.extend([f"## {title}", "", text or "N/A", ""])
            _write_text_file(
                final_summaries_dir / "final_structured_sections.md",
                "\n".join(section_lines).strip(),
            )
        final_freeform_now = _normalize_summary_text(state.get("freeform_final_summary"))
        if final_freeform_now:
            _write_text_file(
                final_summaries_dir / "final_freeform.md",
                "\n".join(
                    [
                        "# Final Freeform Summary",
                        f"- period: `{state.get('period_start')}` -> `{state.get('period_end')}`",
                        "",
                        final_freeform_now,
                    ]
                ),
            )
        final_freeform_sections = state.get("freeform_sections")
        if isinstance(final_freeform_sections, list) and final_freeform_sections:
            _write_json_file(
                final_summaries_dir / "final_freeform_sections.json",
                {
                    "saved_at": datetime.now(MSK).isoformat(),
                    "sections": _json_safe(final_freeform_sections),
                },
            )
            section_lines: List[str] = ["# Final Freeform Sections", ""]
            for item in final_freeform_sections:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip() or "Untitled Section"
                text = _normalize_summary_text(item.get("text"))
                section_lines.extend([f"## {title}", "", text or "N/A", ""])
            _write_text_file(
                final_summaries_dir / "final_freeform_sections.md",
                "\n".join(section_lines).strip(),
            )
        if _safe_int(state.get("report_progress_total"), 0) > 0:
            _set_report_progress(
                "Финальные артефакты сохранены",
                step_inc=1,
                mark_active=False,
                render_now=True,
            )
        _estimate_eta(state, "done", {})
        _enrich_stats_with_elapsed(state)
        _persist_checkpoint(session_checkpoint_path, state)

    except Exception as exc:  # noqa: BLE001
        state["status"] = "error"
        state["active_step"] = "Ошибка выполнения"
        state["error"] = str(exc)
        if _safe_int(state.get("report_progress_total"), 0) > 0:
            state["report_progress_active"] = False
            if not str(state.get("report_progress_label") or "").strip():
                state["report_progress_label"] = "Генерация итогового отчёта прервана"
        _estimate_eta(state, "error", {})
        _enrich_stats_with_elapsed(state)
        _persist_checkpoint(session_checkpoint_path, state)
        deps.logger.exception("logs_summary_page.run_failed")
        with runtime_error_placeholder.container():
            st.error(f"Ошибка выполнения: {exc}")

    saved = _save_logs_summary_result(
        output_dir=deps.output_dir,
        request_payload=request_payload,
        result_state=state,
    )
    state["result_json_path"] = saved.get("json_path")
    state["result_bundle_path"] = saved.get("bundle_path")
    state["result_summary_path"] = saved.get("summary_path")
    state["result_html_path"] = saved.get("html_path")
    state["result_structured_md_path"] = saved.get("structured_md_path")
    state["result_freeform_md_path"] = saved.get("freeform_md_path")
    state["result_structured_txt_path"] = saved.get("structured_txt_path")
    state["result_freeform_txt_path"] = saved.get("freeform_txt_path")
    _persist_checkpoint(session_checkpoint_path, state)
    st.session_state[LAST_STATE_SESSION_KEY] = state
    st.session_state[RUNNING_SESSION_KEY] = False
    st.session_state.pop(RUN_PARAMS_SESSION_KEY, None)
    st.rerun()
