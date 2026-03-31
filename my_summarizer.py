from __future__ import annotations

import json
import logging
import math
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import pandas as pd
import requests

from settings import settings


LOGS_SQL_COLUMNS: tuple[str, ...] = ("timestamp", "value")
DEFAULT_SUMMARY_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "start_time",
    "end_time",
    "cnt",
    "log",
    "message",
    "time",
    "logtag",
    "ext_ClusterEnv",
    "ext_ClusterEventType",
    "ext_ClusterName",
    "kubernetes_pod_name",
    "kubernetes_namespace_name",
    "kubernetes_container_name",
    "kubernetes_docker_id",
    "kubernetes_container_image",
    "container",
    "pod",
    "node",
    "cluster",
    "level",
    "status",
    "value",
)
logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, Dict[str, Any]], None]
_EARLIEST_PERIOD_START = "1970-01-01T00:00:00+00:00"


class DBPageFetcher(Protocol):
    def __call__(
        self,
        *,
        columns: Sequence[str],
        period_start: str,
        period_end: str,
        limit: int,
        offset: int,
    ) -> List[Dict[str, Any]]:
        ...


class LLMTextCaller(Protocol):
    def __call__(self, prompt: str) -> str:
        ...


@dataclass
class SummarizerConfig:
    page_limit: int = 1000
    llm_chunk_rows: int = 200
    reduce_group_size: int = 8
    max_reduce_rounds: int = 12
    # 0 or negative => no truncation.
    max_cell_chars: int = 0
    # 0 or negative => no truncation.
    max_summary_chars: int = 0
    # 0 or negative => no local reduce prompt-length cap.
    reduce_prompt_max_chars: int = 0
    adaptive_reduce_on_overflow: bool = True
    keep_map_batches_in_memory: bool = False
    keep_map_summaries_in_result: bool = False
    # Number of parallel LLM workers for the MAP phase.
    # 1 = sequential (safe default).  Set to 3-5 for a significant speedup when the
    # LLM endpoint supports concurrent requests — total MAP time becomes roughly
    # ceil(n_batches / map_workers) × avg_llm_latency instead of n_batches × avg_llm_latency.
    map_workers: int = 1


@dataclass
class SummarizationResult:
    summary: str
    pages_fetched: int
    rows_processed: int
    llm_calls: int
    chunk_summaries: int
    reduce_rounds: int
    map_summaries: List[str]
    map_batches: List[Dict[str, Any]]
    freeform_summary: str = ""


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


DEFAULT_ANTI_HALLUCINATION_RULES = (
    "1) Цитируй источник: timestamp/сообщение/cnt/поле.\n"
    "2) Не додумывай: если данных нет — \"данных недостаточно\".\n"
    "3) Разделяй [ФАКТ] и [ГИПОТЕЗА]; для гипотезы указывай, что проверить.\n"
    "4) Не обобщай сверх данных (не раздувай масштаб).\n"
    "5) Корреляция по времени != причинность.\n"
    "6) Для [ФАКТ]-причинности нужны: A и B в данных, A<=B по времени, механизм влияния, подтверждение механизма.\n"
    "7) В агрегированных логах argMin-поля — пример, cnt — масштаб.\n"
    "8) Не экстраполируй вне временного диапазона данных.\n"
    "9) При противоречиях показывай оба варианта.\n"
    "10) Отделяй [РЕЛЕВАНТНО] от [ФОН]/[НЕЯСНО].\n"
    "11) В хронологии КАЖДОЕ событие обязано содержать полную дату и время (до микросекунд) и timezone."
)


def _chain_section_requirement(stage: str) -> str:
    normalized = str(stage or "").strip().lower()
    if normalized == "map":
        title = "ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК: ЛОКАЛЬНАЯ ЦЕПОЧКА СОБЫТИЙ БАТЧА"
    elif normalized == "reduce":
        title = "ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК: СВОДНАЯ ЦЕПОЧКА СОБЫТИЙ ИСТОЧНИКА"
    elif normalized == "cross":
        title = "ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК: ЕДИНАЯ ЦЕПОЧКА СОБЫТИЙ ИНЦИДЕНТА"
    else:
        title = "ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК: ЦЕПОЧКА СОБЫТИЙ"
    return "\n".join(
        [
            title,
            "Оформи красиво и явно как схему в Markdown.",
            "Для КАЖДОГО узла обязательно укажи точный timestamp события: `YYYY-MM-DD HH:MM:SS.ffffff TZ`.",
            "Запрещены абстрактные метки без времени (например, просто t1/t2/t3).",
            "Формат (сохрани стрелки и отступы):",
            "[2026-03-31 12:34:56.123456 MSK] ТРИГГЕР (компонент) [ФАКТ/ГИПОТЕЗА]",
            "    └─> (механизм влияния)",
            "[2026-03-31 12:35:07.654321 MSK] СЛЕДСТВИЕ (компонент) [ФАКТ/ГИПОТЕЗА]",
            "    └─> (механизм влияния)",
            "[2026-03-31 12:35:10.000001 MSK] АЛЕРТ/ПОСЛЕДСТВИЕ [ФАКТ]",
            "Если цепочек несколько — перечисли ЦЕПОЧКА #1, ЦЕПОЧКА #2 и т.д.",
            "Если есть разрыв — вставь узел: [РАЗРЫВ ЦЕПОЧКИ: каких данных не хватает].",
        ]
    )


def _append_chain_requirement(prompt_text: str, stage: str) -> str:
    base = str(prompt_text or "").strip()
    chain_block = _chain_section_requirement(stage)
    incident_link_block = "\n".join(
        [
            "ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК: СВЯЗЬ С ИНЦИДЕНТОМ ИЗ UI",
            "Нужно явно связать выводы с контекстом, который пользователь ввёл в UI",
            "(incident_description / alerts_list / user goal).",
            "Для каждого пункта инцидента/алерта укажи:",
            "- статус: [ОБЪЯСНЁН] / [ЧАСТИЧНО ОБЪЯСНЁН] / [НЕ ОБЪЯСНЁН]",
            "- доказательства: конкретные timestamp/сообщения/метрики",
            "- причинно-следственная связь (если есть)",
            "- если связи нет: что нужно проверить дополнительно.",
            "Если контекст инцидента пустой — напиши это явно отдельной строкой.",
        ]
    )
    root_cause_hypotheses_block = "\n".join(
        [
            "ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК: ПРЕДПОЛОЖЕНИЯ О ПЕРВОПРИЧИНЕ ПО КАЖДОМУ ИНЦИДЕНТУ",
            "Для КАЖДОГО выявленного инцидента/цепочки отдельно перечисли 2-5 гипотез первопричины.",
            "Формат для каждого пункта:",
            "- Инцидент/цепочка: <название>",
            "- [ГИПОТЕЗА] первопричина: <кратко>",
            "- Почему это вероятно: <ссылка на timestamp/логи/метрики>",
            "- Что проверить для подтверждения/опровержения: <конкретные действия/данные>",
            "Если инцидент один — блок всё равно обязателен и должен содержать минимум 2 гипотезы.",
        ]
    )
    if not base:
        return f"{chain_block}\n\n{incident_link_block}\n\n{root_cause_hypotheses_block}"
    return (
        f"{base}\n\n{chain_block}\n\n{incident_link_block}\n\n{root_cause_hypotheses_block}"
    )


def _read_prompt_setting(name: str) -> str:
    return str(getattr(settings, name, "") or "").strip()


def _resolve_anti_hallucination_rules() -> str:
    custom = _read_prompt_setting("CONTROL_PLANE_LLM_ANTI_HALLUCINATION_RULES")
    return custom or DEFAULT_ANTI_HALLUCINATION_RULES


def _render_prompt_template(template: str, values: Dict[str, Any]) -> str:
    rendered = str(template)
    # Basic Handlebars compatibility for user-provided templates.
    rendered = re.sub(
        r"\{\{#each\s+map_summaries\}\}[\s\S]*?\{\{\/each\}\}",
        "{summaries_text}",
        rendered,
        flags=re.IGNORECASE,
    )
    rendered = re.sub(
        r"\{\{#each\s+source_summaries\}\}[\s\S]*?\{\{\/each\}\}",
        "{source_summaries_text}",
        rendered,
        flags=re.IGNORECASE,
    )

    def _replace_if_data_type(match: re.Match[str]) -> str:
        body = match.group(1)
        return body if str(values.get("data_type", "")).lower() == "aggregated" else ""

    rendered = re.sub(
        r"\{\{#if\s+data_type\s*==\s*\"aggregated\"\s*\}\}([\s\S]*?)\{\{\/if\}\}",
        _replace_if_data_type,
        rendered,
        flags=re.IGNORECASE,
    )

    def _replace_if_key(match: re.Match[str]) -> str:
        key = str(match.group(1) or "").strip()
        body = match.group(2)
        value = values.get(key)
        return body if bool(value) else ""

    rendered = re.sub(
        r"\{\{#if\s+([a-zA-Z0-9_]+)\s*\}\}([\s\S]*?)\{\{\/if\}\}",
        _replace_if_key,
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


def _ctx_value(ctx: Optional[Dict[str, Any]], key: str, default: Any = "") -> str:
    if not isinstance(ctx, dict):
        return "" if default is None else str(default)
    value = ctx.get(key, default)
    if value is None:
        return ""
    return str(value)


def _normalize_summary_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"none", "null", "nan"}:
        return ""
    return text


def has_required_env() -> bool:
    return bool(str(settings.OPENAI_API_BASE_DB).strip()) and bool(str(settings.OPENAI_API_KEY_DB).strip())


def _build_chat_completions_url(api_base: str) -> str:
    base = api_base.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _extract_message_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if text is None:
                    text = item.get("content")
                if text is None and "value" in item:
                    text = item.get("value")
                if text is not None:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "".join(parts)
    return str(value)


def _parse_chat_completion_response(data: Any) -> Tuple[str, str]:
    if isinstance(data, dict):
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0] if isinstance(choices[0], dict) else {}
            finish_reason = str(choice.get("finish_reason") or "").strip().lower()
            message = choice.get("message", {})
            content = ""
            if isinstance(message, dict):
                content = _extract_message_content(message.get("content"))
            if not content and "text" in choice:
                content = _extract_message_content(choice.get("text"))
            if content:
                return content, finish_reason
        output_text = data.get("output_text")
        if output_text is not None:
            return _extract_message_content(output_text), ""
    return str(data), ""


def communicate_with_llm(message: str, system_prompt: str = "", timeout: float = 600.0) -> str:
    if not has_required_env():
        raise RuntimeError("OPENAI_API_BASE_DB and OPENAI_API_KEY_DB are required")

    url = _build_chat_completions_url(str(settings.OPENAI_API_BASE_DB))
    headers = {
        "Authorization": f"Bearer {str(settings.OPENAI_API_KEY_DB)}",
        "Content-Type": "application/json",
    }
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})
    base_payload: Dict[str, Any] = {
        "model": str(settings.LLM_MODEL_ID),
        "temperature": 0.1,
    }
    configured_max_tokens = int(getattr(settings, "CONTROL_PLANE_LLM_MAX_TOKENS", 0) or 0)
    if configured_max_tokens > 0:
        base_payload["max_tokens"] = configured_max_tokens
    continue_on_length = bool(
        getattr(settings, "CONTROL_PLANE_LLM_CONTINUE_ON_LENGTH", True)
    )
    continue_round_limit = max(
        int(getattr(settings, "CONTROL_PLANE_LLM_CONTINUE_MAX_ROUNDS", 12) or 12),
        1,
    )

    dialog_messages: List[Dict[str, str]] = list(messages)
    collected_parts: List[str] = []
    for round_idx in range(1, continue_round_limit + 1):
        payload = dict(base_payload)
        payload["messages"] = list(dialog_messages)
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        chunk_text, finish_reason = _parse_chat_completion_response(data)
        collected_parts.append(str(chunk_text or ""))

        can_continue = continue_on_length and finish_reason in {"length", "max_tokens"}
        if not can_continue:
            break
        if round_idx >= continue_round_limit:
            logger.warning(
                "LLM output still marked as truncated after %s continuation rounds; "
                "returning accumulated text as-is.",
                continue_round_limit,
            )
            break

        logger.warning(
            "LLM finish_reason=%s: requesting continuation chunk %s/%s",
            finish_reason,
            round_idx + 1,
            continue_round_limit,
        )
        dialog_messages.append({"role": "assistant", "content": str(chunk_text or "")})
        dialog_messages.append(
            {
                "role": "user",
                "content": (
                    "Продолжи строго с того места, где остановился. "
                    "Не повторяй предыдущие абзацы, сохрани формат ответа."
                ),
            }
        )

    return "".join(collected_parts).strip()


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


def _extract_batch_period(rows: Sequence[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    timestamps: List[pd.Timestamp] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_ts = None
        for key in ("timestamp", "start_time", "end_time", "ts", "time", "datetime"):
            if row.get(key) is not None:
                raw_ts = row.get(key)
                break
        if raw_ts is None:
            continue
        ts = pd.to_datetime(raw_ts, utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        timestamps.append(ts)

    if not timestamps:
        return None, None

    start_ts = min(timestamps).isoformat().replace("+00:00", "Z")
    end_ts = max(timestamps).isoformat().replace("+00:00", "Z")
    return start_ts, end_ts


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


def _resolve_logs_fetch_mode() -> str:
    raw_mode = str(getattr(settings, "CONTROL_PLANE_LOGS_FETCH_MODE", "time_window")).strip().lower()
    aliases = {
        "time_window": "time_window",
        "window": "time_window",
        "lookback": "time_window",
        "tail_n_logs": "tail_n_logs",
        "tail_n": "tail_n_logs",
        "last_n_logs": "tail_n_logs",
    }
    resolved = aliases.get(raw_mode)
    if resolved is None:
        logger.warning(
            "Unknown CONTROL_PLANE_LOGS_FETCH_MODE=%s; fallback to time_window",
            raw_mode,
        )
        return "time_window"
    return resolved


def _strip_trailing_limit_offset(query: str) -> str:
    stripped = query.strip().rstrip(";")
    without_limit_offset = re.sub(
        r"(?is)\s+LIMIT\s+\d+\s+OFFSET\s+\d+\s*$",
        "",
        stripped,
    )
    without_offset_only = re.sub(
        r"(?is)\s+OFFSET\s+\d+\s*$",
        "",
        without_limit_offset,
    )
    without_limit_only = re.sub(
        r"(?is)\s+LIMIT\s+\d+\s*$",
        "",
        without_offset_only,
    )
    return without_limit_only


def _build_tail_paged_query(
    *,
    base_query: str,
    tail_limit: int,
    limit: int,
    offset: int,
) -> str:
    safe_tail_limit = max(int(tail_limit), 1)
    safe_limit = max(int(limit), 1)
    safe_offset = max(int(offset), 0)
    return (
        "SELECT * FROM ("
        "SELECT * FROM ("
        f"{base_query}"
        f") AS cp_src ORDER BY timestamp DESC LIMIT {safe_tail_limit}"
        ") AS cp_tail "
        "ORDER BY timestamp ASC "
        f"LIMIT {safe_limit} OFFSET {safe_offset}"
    )


def _wrap_with_limit_offset(
    *,
    base_query: str,
    limit: int,
    offset: int,
) -> str:
    safe_limit = max(int(limit), 1)
    safe_offset = max(int(offset), 0)
    return (
        "SELECT * FROM ("
        f"{base_query}"
        ") AS cp_logs_page "
        f"LIMIT {safe_limit} OFFSET {safe_offset}"
    )


def _render_logs_query(
    *,
    query_template: str,
    period_start: str,
    period_end: str,
    limit: int,
    offset: int,
    service: str,
    last_ts: Optional[str] = None,
) -> str:
    safe_limit = max(int(limit), 1)
    safe_offset = max(int(offset), 0)
    # last_ts defaults to period_start so keyset queries work correctly on the first page
    effective_last_ts = last_ts if last_ts is not None else period_start
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
        last_ts=effective_last_ts,
    )
    return query_template.strip().rstrip(";").format_map(params)


def _query_logs_df(query: str) -> pd.DataFrame:
    try:
        import clickhouse_connect
    except Exception as exc:
        raise ImportError("Для чтения логов нужен пакет clickhouse-connect") from exc

    host = str(settings.CONTROL_PLANE_LOGS_CLICKHOUSE_HOST).strip()
    if not host:
        raise ValueError(
            "Set CONTROL_PLANE_LOGS_CLICKHOUSE_HOST in .env for logs summarization"
        )

    client = clickhouse_connect.get_client(
        host=host,
        port=int(settings.CONTROL_PLANE_LOGS_CLICKHOUSE_PORT),
        username=str(settings.CONTROL_PLANE_LOGS_CLICKHOUSE_USERNAME).strip() or None,
        password=str(settings.CONTROL_PLANE_LOGS_CLICKHOUSE_PASSWORD).strip() or None,
    )
    try:
        return client.query_df(query)
    finally:
        try:
            client.close()
        except Exception:
            logger.warning("ClickHouse client close failed for logs query")


def _build_db_fetch_page(
    anomaly: Optional[Dict[str, Any]],
    *,
    fetch_mode: str,
    tail_limit: int,
    on_error: Optional[Callable[[str], None]] = None,
) -> Callable[..., List[Dict[str, Any]]]:
    service = _resolve_service(anomaly)
    query_template = _resolve_logs_query_template()
    query_template_lc = query_template.lower()
    has_offset_placeholder = "{offset}" in query_template_lc
    # Keyset pagination: if template contains {last_ts}, we use timestamp-based pagination
    # instead of LIMIT/OFFSET to avoid ClickHouse MEMORY_LIMIT_EXCEEDED on large offsets.
    has_last_ts_placeholder = "{last_ts}" in query_template_lc
    # Mutable single-element list so the inner closure can update state between pages
    _keyset_last_ts: List[Optional[str]] = [None]

    def _db_fetch_page(
        *,
        columns: Sequence[str],
        period_start: str,
        period_end: str,
        limit: int,
        offset: int,
    ) -> List[Dict[str, Any]]:
        if has_last_ts_placeholder:
            # Keyset mode: ignore `offset`, track last_ts across calls.
            # On the very first call _keyset_last_ts[0] is None, so _render_logs_query
            # defaults last_ts to period_start (= fetch everything from the start).
            rendered_query = _render_logs_query(
                query_template=query_template,
                period_start=period_start,
                period_end=period_end,
                limit=limit,
                offset=0,
                service=service,
                last_ts=_keyset_last_ts[0],
            )
            try:
                page_df = _query_logs_df(rendered_query)
            except Exception as exc:
                msg = (
                    f"ClickHouse keyset query failed "
                    f"(service={service}, last_ts={_keyset_last_ts[0]}): {exc}"
                )
                logger.warning(msg)
                if on_error:
                    on_error(msg)
                return []

            if page_df.empty:
                return []

            # Advance last_ts to the maximum timestamp in this page
            if "timestamp" in page_df.columns:
                max_ts = pd.to_datetime(page_df["timestamp"], utc=True, errors="coerce").max()
                if not pd.isna(max_ts):
                    _keyset_last_ts[0] = max_ts.isoformat()

            records = page_df.to_dict(orient="records")
            if not columns:
                return [dict(row) for row in records]
            projected_rows: List[Dict[str, Any]] = []
            for row in records:
                projected_rows.append({col: row.get(col) for col in columns})
            return projected_rows

        # --- Legacy path ---
        effective_period_start = (
            _EARLIEST_PERIOD_START if fetch_mode == "tail_n_logs" else period_start
        )
        rendered_query = _render_logs_query(
            query_template=query_template,
            period_start=effective_period_start,
            period_end=period_end,
            limit=limit,
            offset=offset,
            service=service,
        )
        if fetch_mode == "tail_n_logs":
            base_query = _strip_trailing_limit_offset(rendered_query)
            query = _build_tail_paged_query(
                base_query=base_query,
                tail_limit=tail_limit,
                limit=limit,
                offset=offset,
            )
        elif has_offset_placeholder:
            # SQL template already supports paging placeholders.
            query = rendered_query
        else:
            # Auto-paging fallback:
            # if template has no OFFSET placeholder, page it externally to avoid
            # "first 1000 rows only" behavior when SQL has a fixed LIMIT.
            base_query = _strip_trailing_limit_offset(rendered_query)
            query = _wrap_with_limit_offset(
                base_query=base_query,
                limit=limit,
                offset=offset,
            )

        try:
            page_df = _query_logs_df(query)
        except Exception as exc:
            msg = f"ClickHouse query failed (service={service}, offset={offset}): {exc}"
            logger.warning(msg)
            if on_error:
                on_error(msg)
            return []

        if page_df.empty:
            return []
        records = page_df.to_dict(orient="records")
        if not columns:
            return [dict(row) for row in records]
        projected_rows: List[Dict[str, Any]] = []
        for row in records:
            projected_rows.append({col: row.get(col) for col in columns})
        return projected_rows

    return _db_fetch_page


def _build_count_query(data_query: str) -> str:
    without_limit_only = _strip_trailing_limit_offset(data_query)
    return f"SELECT count() AS total_rows FROM ({without_limit_only}) AS cp_logs"


def _estimate_total_logs(
    *,
    anomaly: Optional[Dict[str, Any]],
    period_start: str,
    period_end: str,
    page_limit: int,
    fetch_mode: str,
    tail_limit: int,
) -> Optional[int]:
    try:
        service = _resolve_service(anomaly)
        query_template = _resolve_logs_query_template()
        sample_query = _render_logs_query(
            query_template=query_template,
            period_start=(
                _EARLIEST_PERIOD_START if fetch_mode == "tail_n_logs" else period_start
            ),
            period_end=period_end,
            limit=page_limit,
            offset=0,
            service=service,
            last_ts=period_start,  # keyset: count from period start
        )
        count_query = _build_count_query(sample_query)
        df = _query_logs_df(count_query)
        if df.empty:
            return None
        if "total_rows" in df.columns:
            total_rows = int(df.iloc[0]["total_rows"])
        else:
            total_rows = int(df.iloc[0, 0])
        if fetch_mode == "tail_n_logs":
            return min(total_rows, max(int(tail_limit), 1))
        return total_rows
    except Exception as exc:
        logger.warning("Failed to estimate total logs for progress: %s", exc)
        return None


def _heuristic_llm_call(prompt: str, error: Optional[str] = None) -> str:
    error_line = f"ОШИБКА: {error}" if error else "LLM не настроена (OPENAI_API_BASE_DB / OPENAI_API_KEY_DB)."
    return (
        f"[LLM НЕДОСТУПНА — эвристический fallback]\n\n"
        f"{error_line}\n\n"
        "ХРОНОЛОГИЯ: данных нет.\n"
        "ПЕРВОПРИЧИНА: [ГИПОТЕЗА] не определена.\n"
        "ОБЪЯСНЕНИЕ АЛЕРТОВ: анализ недоступен.\n"
        "ПРОБЕЛЫ: см. ошибку выше."
    )


def _is_read_timeout_exception(exc: Exception) -> bool:
    if isinstance(exc, requests.exceptions.ReadTimeout):
        return True
    text = str(exc).strip().lower()
    return (
        "read timed out" in text
        or "read timeout" in text
        or "readtimeout" in text
    )


def _make_llm_call(
    max_retries: int = -1,
    retry_delay: float = 10.0,
    on_retry: Optional[Callable[[int, int, Exception], None]] = None,
    on_attempt: Optional[Callable[[int, int, float], None]] = None,
    on_result: Optional[Callable[[int, int, bool, float, Optional[str]], None]] = None,
    llm_timeout: float = 600.0,
) -> LLMTextCaller:
    if not has_required_env():
        logger.warning(
            "OPENAI_API_BASE_DB/OPENAI_API_KEY_DB не заданы; использую fallback summarizer"
        )
        return _heuristic_llm_call

    default_system_prompt = (
        "Ты — senior SRE-аналитик инцидентов. Анализируй логи и метрики строго на основе данных.\n"
        "Принципы:\n"
        "1) Только факты: каждое утверждение подтверждай timestamp/сообщением/значением.\n"
        "2) Маркировка: [ФАКТ] — прямое подтверждение, [ГИПОТЕЗА] — предположение.\n"
        "3) Если данных недостаточно — пиши \"данных недостаточно\", не додумывай.\n"
        "4) Хронология обязательна: строй цепочку событий по времени.\n"
        "5) Главный результат: причинно-следственные цепочки (триггер→распространение→последствия→алерты).\n"
        "6) Если звенья не связаны данными — отмечай разрывы цепочки и нужные данные для закрытия.\n"
        "7) Возможны несколько независимых цепочек/инцидентов: не склеивай их без механизма связи.\n"
        "8) Отделяй [РЕЛЕВАНТНО] события от [ФОН]/[НЕЯСНО].\n"
        "9) Для агрегированных логов: строка=группа событий, cnt=масштаб, argMin-поля=пример.\n"
        "10) Корреляция по времени не равна причинности."
    )
    custom_system_prompt = str(getattr(settings, "CONTROL_PLANE_LLM_SYSTEM_PROMPT", "")).strip()
    _system_prompt = custom_system_prompt or default_system_prompt

    def _llm_call(prompt: str) -> str:
        last_exc: Optional[Exception] = None
        retries = int(max_retries)
        infinite_retries = retries < 0
        configured_total_attempts = -1 if infinite_retries else (max(retries, 0) + 1)
        effective_total_attempts = configured_total_attempts
        base_timeout = max(float(llm_timeout), 1.0)
        current_timeout = base_timeout
        attempt_no = 0
        while True:
            attempt_no += 1
            if on_attempt is not None:
                try:
                    on_attempt(attempt_no, effective_total_attempts, current_timeout)
                except Exception:
                    pass
            started = time.monotonic()
            try:
                response = communicate_with_llm(
                    message=prompt,
                    system_prompt=_system_prompt,
                    timeout=current_timeout,
                )
                elapsed = max(time.monotonic() - started, 0.0)
                if on_result is not None:
                    try:
                        on_result(attempt_no, effective_total_attempts, True, elapsed, None)
                    except Exception:
                        pass
                return str(response).strip()
            except Exception as exc:
                last_exc = exc
                elapsed = max(time.monotonic() - started, 0.0)
                is_read_timeout = _is_read_timeout_exception(exc)
                if is_read_timeout:
                    # For ReadTimeout we keep waiting indefinitely and progressively
                    # increase the request timeout on each occurrence.
                    effective_total_attempts = -1
                if on_result is not None:
                    try:
                        on_result(
                            attempt_no,
                            effective_total_attempts,
                            False,
                            elapsed,
                            str(exc),
                        )
                    except Exception:
                        pass
                can_retry = (
                    is_read_timeout
                    or infinite_retries
                    or attempt_no <= max(retries, 0)
                )
                if can_retry:
                    if is_read_timeout:
                        next_timeout = current_timeout + base_timeout
                        logger.warning(
                            "LLM ReadTimeout on attempt %d; next timeout %.1fs (prev %.1fs)",
                            attempt_no,
                            next_timeout,
                            current_timeout,
                        )
                        current_timeout = next_timeout
                    elif infinite_retries:
                        logger.warning(
                            "LLM retry %d/∞ after error: %s",
                            attempt_no + 1,
                            exc,
                        )
                    else:
                        logger.warning(
                            "LLM retry %d/%d after error: %s",
                            attempt_no + 1,
                            configured_total_attempts,
                            exc,
                        )
                    if on_retry is not None:
                        try:
                            on_retry(attempt_no, effective_total_attempts, exc)
                        except Exception:
                            pass
                    # Fixed wait between retries (no exponential backoff).
                    if retry_delay > 0:
                        time.sleep(retry_delay)
                else:
                    logger.exception(
                        "LLM все %d попытки исчерпаны; использую fallback",
                        configured_total_attempts,
                    )
                    break
        return _heuristic_llm_call(prompt, error=str(last_exc))

    return _llm_call


class PeriodLogSummarizer:
    PROBLEM_KEYWORDS = (
        "error",
        "exception",
        "timeout",
        "failed",
        "fail",
        "fatal",
        "critical",
        "panic",
        "denied",
        "refused",
        "unavailable",
    )

    def __init__(
        self,
        *,
        db_fetch_page: DBPageFetcher,
        llm_call: LLMTextCaller,
        config: SummarizerConfig | None = None,
        on_progress: Optional[ProgressCallback] = None,
        prompt_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.db_fetch_page = db_fetch_page
        self.llm_call = llm_call
        self.config = config or SummarizerConfig()
        self.on_progress = on_progress
        self.prompt_context = prompt_context or {}

    def _emit_progress(self, event: str, payload: Dict[str, Any]) -> None:
        if self.on_progress is None:
            return
        self.on_progress(event, payload)

    def summarize_period(
        self,
        *,
        period_start: str,
        period_end: str,
        columns: Sequence[str],
        total_rows_estimate: Optional[int] = None,
    ) -> SummarizationResult:
        self._validate_iso_datetime(period_start)
        self._validate_iso_datetime(period_end)
        if not columns:
            raise ValueError("columns must not be empty")

        offset = 0
        pages_fetched = 0
        rows_processed = 0
        llm_calls = 0
        map_summaries: List[str] = []
        map_batches: List[Dict[str, Any]] = []
        map_batch_index = 0
        rows_mapped = 0
        estimated_batch_total: Optional[int] = None
        if total_rows_estimate is not None and total_rows_estimate > 0:
            estimated_batch_total = max(
                int(math.ceil(float(total_rows_estimate) / float(max(self.config.llm_chunk_rows, 1)))),
                1,
            )
        seen_sources: List[str] = []  # ordered unique sources seen across all pages
        self._emit_progress(
            "map_start",
            {
                "rows_processed": 0,
                "rows_total": total_rows_estimate,
            },
        )

        # Each pending entry: (Future | None, batch_index, rows_count, bp_start, bp_end, rows_snapshot)
        # rows_snapshot is kept only for the progress callback; raw rows are not stored otherwise.
        _PendingItem = Tuple[
            Optional[Future],  # type: ignore[type-arg]
            int,               # batch_index
            int,               # rows_count
            Optional[str],     # batch_period_start
            Optional[str],     # batch_period_end
            List[Dict[str, Any]],  # rows snapshot for progress callback
        ]
        pending_items: List[_PendingItem] = []
        parallel = self.config.map_workers > 1
        executor = ThreadPoolExecutor(max_workers=self.config.map_workers) if parallel else None

        try:
            while True:
                page = self.db_fetch_page(
                    columns=columns,
                    period_start=period_start,
                    period_end=period_end,
                    limit=self.config.page_limit,
                    offset=offset,
                )
                if not page:
                    break

                pages_fetched += 1
                rows_processed += len(page)
                offset += len(page)
                for _row in page:
                    _src = str(_row.get("_source") or "")
                    if _src and _src not in seen_sources:
                        seen_sources.append(_src)
                self._emit_progress(
                    "page_fetched",
                    {
                        "page_index": pages_fetched,
                        "page_rows": len(page),
                        "rows_fetched": rows_processed,
                        "rows_total": total_rows_estimate,
                    },
                )

                for i in range(0, len(page), self.config.llm_chunk_rows):
                    rows_chunk = page[i : i + self.config.llm_chunk_rows]
                    next_batch_index = map_batch_index
                    batch_period_start, batch_period_end = _extract_batch_period(rows_chunk)
                    self._emit_progress(
                        "map_batch_start",
                        {
                            "batch_index": next_batch_index,
                            "batch_total": estimated_batch_total,
                            "batch_logs_count": len(rows_chunk),
                            "batch_period_start": batch_period_start,
                            "batch_period_end": batch_period_end,
                            "rows_processed": rows_mapped,
                            "rows_total": total_rows_estimate,
                        },
                    )
                    prompt = self._build_chunk_prompt(
                        period_start=period_start,
                        period_end=period_end,
                        columns=columns,
                        rows=rows_chunk,
                        batch_number=next_batch_index + 1,
                        total_batches=estimated_batch_total,
                    )
                    future: Optional[Future] = None  # type: ignore[type-arg]
                    if executor is not None:
                        future = executor.submit(self.llm_call, prompt)
                    else:
                        # Sequential: run inline and wrap result as a fake "future"
                        try:
                            _result = self.llm_call(prompt)
                        except Exception as exc:
                            logger.exception("LLM call failed on batch %s", next_batch_index)
                            _result = _heuristic_llm_call(prompt, error=str(exc))
                        future = None
                        # Store result directly — reuse the same pending tuple structure
                        pending_items.append((
                            None,
                            next_batch_index,
                            len(rows_chunk),
                            batch_period_start,
                            batch_period_end,
                            list(rows_chunk),
                        ))
                        # Process immediately in sequential mode (maintains existing progress behaviour)
                        chunk_summary = (_result or "").strip() or "Пустой ответ LLM на map-этапе."
                        chunk_summary = self._truncate(chunk_summary, self.config.max_summary_chars)
                        map_summaries.append(chunk_summary)
                        if self.config.keep_map_batches_in_memory:
                            map_batches.append({
                                "batch_index": next_batch_index,
                                "rows_count": len(rows_chunk),
                                "summary": chunk_summary,
                                "batch_period_start": batch_period_start,
                                "batch_period_end": batch_period_end,
                            })
                        rows_mapped += len(rows_chunk)
                        self._emit_progress(
                            "map_batch",
                            {
                                "batch_index": next_batch_index,
                                "batch_total": estimated_batch_total,
                                "batch_summary": chunk_summary,
                                "batch_logs_count": len(rows_chunk),
                                "batch_logs": list(rows_chunk),
                                "batch_period_start": batch_period_start,
                                "batch_period_end": batch_period_end,
                                "rows_processed": rows_mapped,
                                "rows_total": total_rows_estimate,
                            },
                        )
                        map_batch_index += 1
                        llm_calls += 1
                        continue  # skip appending to pending_items again

                    # Parallel mode: store future for later collection
                    pending_items.append((
                        future,
                        next_batch_index,
                        len(rows_chunk),
                        batch_period_start,
                        batch_period_end,
                        list(rows_chunk),
                    ))
                    map_batch_index += 1

                if len(page) < self.config.page_limit:
                    break

        except Exception:
            if executor is not None:
                executor.shutdown(wait=False)
            raise
        else:
            if executor is not None:
                executor.shutdown(wait=True)

        # --- Parallel mode: collect futures in submission order ---
        if parallel and pending_items:
            for fut, batch_idx, nrows, bp_start, bp_end, rows_snap in pending_items:
                if fut is None:
                    continue  # sequential items already handled above
                try:
                    result_text = fut.result()
                except Exception as exc:
                    logger.exception("Parallel LLM call failed on batch %s; using fallback", batch_idx)
                    result_text = _heuristic_llm_call("", error=str(exc))
                chunk_summary = (result_text or "").strip() or "Пустой ответ LLM на map-этапе."
                chunk_summary = self._truncate(chunk_summary, self.config.max_summary_chars)
                map_summaries.append(chunk_summary)
                if self.config.keep_map_batches_in_memory:
                    map_batches.append({
                        "batch_index": batch_idx,
                        "rows_count": nrows,
                        "summary": chunk_summary,
                        "batch_period_start": bp_start,
                        "batch_period_end": bp_end,
                    })
                rows_mapped += nrows
                llm_calls += 1
                self._emit_progress(
                    "map_batch",
                    {
                        "batch_index": batch_idx,
                        "batch_total": estimated_batch_total,
                        "batch_summary": chunk_summary,
                        "batch_logs_count": nrows,
                        "batch_logs": rows_snap,
                        "batch_period_start": bp_start,
                        "batch_period_end": bp_end,
                        "rows_processed": rows_mapped,
                        "rows_total": total_rows_estimate,
                    },
                )

        if not map_summaries:
            self._emit_progress(
                "map_done",
                {
                    "batch_total": 0,
                    "rows_processed": 0,
                    "rows_total": total_rows_estimate,
                },
            )
            return SummarizationResult(
                summary="Нет логов за указанный период.",
                pages_fetched=pages_fetched,
                rows_processed=rows_processed,
                llm_calls=llm_calls,
                chunk_summaries=0,
                reduce_rounds=0,
                map_summaries=[],
                map_batches=[],
            )

        self._emit_progress(
            "map_done",
            {
                "batch_total": map_batch_index,
                "rows_processed": rows_mapped,
                "rows_total": total_rows_estimate,
            },
        )
        self._emit_progress(
            "reduce_start",
            {
                "batch_total": map_batch_index,
                "rows_processed": rows_mapped,
                "rows_total": total_rows_estimate,
            },
        )
        final_summary, reduce_calls, reduce_rounds = self._reduce_summaries(
            chunk_summaries=map_summaries,
            period_start=period_start,
            period_end=period_end,
            sources=seen_sources or None,
        )
        self._emit_progress(
            "reduce_done",
            {
                "summary": final_summary,
                "rows_processed": rows_mapped,
                "rows_total": total_rows_estimate,
            },
        )
        llm_calls += reduce_calls

        # Freeform narrative: one more LLM pass to produce a human-readable story for SRE team
        freeform_summary = ""
        if final_summary and final_summary != "Нет логов за указанный период.":
            self._emit_progress("freeform_start", {"rows_processed": rows_mapped})
            try:
                freeform_prompt = self._build_freeform_prompt(
                    period_start=period_start,
                    period_end=period_end,
                    structured_summary=final_summary,
                    map_summaries=map_summaries,
                )
                freeform_summary = self.llm_call(freeform_prompt).strip()
                llm_calls += 1
            except Exception:
                logger.warning("Freeform narrative generation failed, skipping")
            self._emit_progress("freeform_done", {"freeform_summary": freeform_summary})

        result_map_summaries = map_summaries if self.config.keep_map_summaries_in_result else []
        return SummarizationResult(
            summary=final_summary,
            freeform_summary=freeform_summary,
            pages_fetched=pages_fetched,
            rows_processed=rows_processed,
            llm_calls=llm_calls,
            chunk_summaries=len(map_summaries),
            reduce_rounds=reduce_rounds,
            map_summaries=result_map_summaries,
            map_batches=map_batches,
        )

    def _reduce_summaries(
        self,
        *,
        chunk_summaries: List[str],
        period_start: str,
        period_end: str,
        sources: Optional[List[str]] = None,
    ) -> tuple[str, int, int]:
        if len(chunk_summaries) == 1:
            return chunk_summaries[0], 0, 0
        if not self.config.adaptive_reduce_on_overflow:
            return self._reduce_summaries_fixed_groups(
                chunk_summaries=chunk_summaries,
                period_start=period_start,
                period_end=period_end,
                sources=sources,
            )

        # 1) Try single-pass reduce over all map batches first.
        # 2) Only if it does not fit context (or context-overflow error), switch to adaptive shrinking.
        current = list(chunk_summaries)
        llm_calls = 0
        first_round = 1
        full_prompt = self._build_reduce_prompt(
            period_start=period_start,
            period_end=period_end,
            reduce_round=first_round,
            summaries=current,
            sources=sources,
        )
        full_fits = self._prompt_fits_budget(full_prompt)
        if full_fits:
            self._emit_progress(
                "reduce_group_start",
                {
                    "reduce_round": first_round,
                    "group_index": 0,
                    "group_total": 1,
                },
            )
            try:
                merged = self.llm_call(full_prompt).strip()
                llm_calls += 1
                if not merged:
                    merged = "Пустой ответ LLM на reduce-этапе."
                self._emit_progress(
                    "reduce_group_done",
                    {
                        "reduce_round": first_round,
                        "group_index": 0,
                        "group_total": 1,
                    },
                )
                return self._truncate(merged, self.config.max_summary_chars), llm_calls, 1
            except Exception as exc:  # noqa: BLE001
                if not self._is_context_overflow_error(exc):
                    raise
                logger.warning("reduce full-merge overflow, fallback to adaptive mode: %s", exc)
                self._emit_progress(
                    "reduce_context_fallback",
                    {
                        "reduce_round": first_round,
                        "reason": str(exc),
                    },
                )
        else:
            self._emit_progress(
                "reduce_context_fallback",
                {
                    "reduce_round": first_round,
                    "reason": (
                        "full reduce prompt exceeds reduce_prompt_max_chars="
                        f"{self.config.reduce_prompt_max_chars}"
                    ),
                },
            )

        # Adaptive mode: for each round try to merge the largest possible group.
        round_idx = 0
        while len(current) > 1:
            round_idx += 1
            if round_idx > self.config.max_reduce_rounds:
                raise RuntimeError("Exceeded max reduce rounds")

            next_level: List[str] = []
            cursor = 0
            group_index = 0
            previous_len = len(current)
            while cursor < len(current):
                remaining = len(current) - cursor
                used_size = remaining
                merged_text: Optional[str] = None

                while used_size >= 1:
                    group = current[cursor : cursor + used_size]
                    if used_size == 1:
                        # Single summary cannot be reduced further; pass through.
                        merged_text = self._truncate(
                            str(group[0]),
                            self.config.max_summary_chars,
                        )
                        break

                    prompt = self._build_reduce_prompt(
                        period_start=period_start,
                        period_end=period_end,
                        reduce_round=round_idx,
                        summaries=group,
                        sources=sources,
                    )
                    if not self._prompt_fits_budget(prompt):
                        used_size -= 1
                        continue

                    self._emit_progress(
                        "reduce_group_start",
                        {
                            "reduce_round": round_idx,
                            "group_index": group_index,
                            "group_total": previous_len,
                            "group_size": used_size,
                        },
                    )
                    try:
                        merged = self.llm_call(prompt).strip()
                        llm_calls += 1
                        if not merged:
                            merged = "Пустой ответ LLM на reduce-этапе."
                        merged_text = self._truncate(merged, self.config.max_summary_chars)
                        self._emit_progress(
                            "reduce_group_done",
                            {
                                "reduce_round": round_idx,
                                "group_index": group_index,
                                "group_total": previous_len,
                                "group_size": used_size,
                            },
                        )
                        break
                    except Exception as exc:  # noqa: BLE001
                        if self._is_context_overflow_error(exc) and used_size > 2:
                            used_size -= 1
                            continue
                        raise

                if merged_text is None:
                    # Safety net: force progress in pathological edge-cases.
                    used_size = min(2, remaining)
                    forced = "\n\n".join(str(x) for x in current[cursor : cursor + used_size])
                    merged_text = self._truncate(forced, self.config.max_summary_chars)

                next_level.append(merged_text)
                cursor += max(used_size, 1)
                group_index += 1

            if len(next_level) >= previous_len:
                # Guarantee convergence to a single summary.
                compressed: List[str] = []
                for i in range(0, len(next_level), 2):
                    pair = next_level[i : i + 2]
                    compressed.append(
                        self._truncate("\n\n".join(pair), self.config.max_summary_chars)
                    )
                next_level = compressed
            current = next_level

        return current[0], llm_calls, round_idx

    def _reduce_summaries_fixed_groups(
        self,
        *,
        chunk_summaries: List[str],
        period_start: str,
        period_end: str,
        sources: Optional[List[str]] = None,
    ) -> tuple[str, int, int]:
        round_idx = 0
        current = chunk_summaries
        llm_calls = 0
        while len(current) > 1:
            round_idx += 1
            if round_idx > self.config.max_reduce_rounds:
                raise RuntimeError("Exceeded max reduce rounds")
            next_level: List[str] = []
            groups_total = int(math.ceil(len(current) / max(self.config.reduce_group_size, 1)))
            for i in range(0, len(current), self.config.reduce_group_size):
                group = current[i : i + self.config.reduce_group_size]
                group_index = int(i / max(self.config.reduce_group_size, 1))
                self._emit_progress(
                    "reduce_group_start",
                    {
                        "reduce_round": round_idx,
                        "group_index": group_index,
                        "group_total": groups_total,
                    },
                )
                prompt = self._build_reduce_prompt(
                    period_start=period_start,
                    period_end=period_end,
                    reduce_round=round_idx,
                    summaries=group,
                    sources=sources,
                )
                merged = self.llm_call(prompt).strip()
                if not merged:
                    merged = "Пустой ответ LLM на reduce-этапе."
                next_level.append(self._truncate(merged, self.config.max_summary_chars))
                self._emit_progress(
                    "reduce_group_done",
                    {
                        "reduce_round": round_idx,
                        "group_index": group_index,
                        "group_total": groups_total,
                    },
                )
                llm_calls += 1
            current = next_level
        return current[0], llm_calls, round_idx

    def _prompt_fits_budget(self, prompt: str) -> bool:
        max_chars = max(int(getattr(self.config, "reduce_prompt_max_chars", 0)), 0)
        if max_chars <= 0:
            return True
        return len(prompt) <= max_chars

    @staticmethod
    def _is_context_overflow_error(exc: Exception) -> bool:
        text = str(exc).lower()
        markers = (
            "maximum context length",
            "context length",
            "too many tokens",
            "token limit",
            "prompt is too long",
            "request too large",
            "input is too long",
            "413",
        )
        return any(marker in text for marker in markers)

    def _build_chunk_prompt(
        self,
        *,
        period_start: str,
        period_end: str,
        columns: Sequence[str],
        rows: List[Dict[str, Any]],
        batch_number: Optional[int] = None,
        total_batches: Optional[int] = None,
    ) -> str:
        # Determine display columns: put _source first if present
        has_source = any(row.get("_source") for row in rows)
        base_columns = [c for c in columns if c != "_source"]
        display_columns: List[str] = (["_source"] + base_columns) if has_source else list(base_columns)

        critical_rows = [row for row in rows if self._row_problem_score(row, list(columns)) > 0]
        extra_prompt_context = _read_prompt_setting("CONTROL_PLANE_LLM_EXTRA_PROMPT_CONTEXT")
        anti_rules = _resolve_anti_hallucination_rules()
        lower_cols = {str(c).lower() for c in display_columns}
        data_type = "aggregated" if {"start_time", "end_time", "cnt"}.issubset(lower_cols) else "raw"
        preferred_time = ("start_time", "timestamp", "time", "ts", "datetime", "end_time")
        time_column = next((c for c in preferred_time if c in lower_cols), "timestamp")
        if not self.prompt_context.get("data_type"):
            self.prompt_context["data_type"] = data_type
        if not self.prompt_context.get("time_column"):
            self.prompt_context["time_column"] = time_column

        source_stat = ""
        if has_source:
            source_counts: Dict[str, int] = {}
            for row in rows:
                src = str(row.get("_source") or "unknown")
                source_counts[src] = source_counts.get(src, 0) + 1
            source_stat = ", ".join(
                f"{src}={cnt}" for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1])
            )

        log_lines: List[str] = []
        for idx, row in enumerate(rows, start=1):
            rendered_parts: List[str] = []
            for col in display_columns:
                value = row.get(col)
                if value is None or value == "":
                    continue
                text = self._truncate(str(value), self.config.max_cell_chars)
                rendered_parts.append(f"{col}={text}")
            log_lines.append(f"{idx}. " + " | ".join(rendered_parts))
        logs_text = "\n".join(log_lines) if log_lines else "Нет строк в батче."

        map_template = _read_prompt_setting("CONTROL_PLANE_LLM_MAP_PROMPT_TEMPLATE")
        if map_template:
            rendered = _render_prompt_template(
                map_template,
                {
                    "period_start": period_start,
                    "period_end": period_end,
                    "incident_start": _ctx_value(self.prompt_context, "incident_start", period_start),
                    "incident_end": _ctx_value(self.prompt_context, "incident_end", period_end),
                    "incident_description": _ctx_value(self.prompt_context, "incident_description", ""),
                    "alerts_list": _ctx_value(self.prompt_context, "alerts_list", ""),
                    "metrics_context": _ctx_value(self.prompt_context, "metrics_context", ""),
                    "source_name": _ctx_value(self.prompt_context, "source_name", source_stat or "query_1"),
                    "sql_query": _ctx_value(self.prompt_context, "sql_query", ""),
                    "batch_number": batch_number if batch_number is not None else "",
                    "total_batches": total_batches if total_batches is not None else "",
                    "batch_data": logs_text,
                    "rows_count": len(rows),
                    "problem_rows": len(critical_rows),
                    "columns": ", ".join(display_columns),
                    "time_column": _ctx_value(self.prompt_context, "time_column", time_column),
                    "data_type": _ctx_value(self.prompt_context, "data_type", data_type),
                    "source_distribution": source_stat,
                    "extra_prompt_context": extra_prompt_context,
                    "anti_hallucination_rules": anti_rules,
                    "logs_text": logs_text,
                },
            ).strip()
            return _append_chain_requirement(rendered, "map")

        lines = [
            "Это MAP-этап расследования инцидента. Анализируй только этот фрагмент логов.",
            "Если выше есть контекст алертов/инцидента — используй его как приоритет.",
            "",
            f"Источник: {_ctx_value(self.prompt_context, 'source_name', source_stat or 'query_1')}",
            f"SQL: {_ctx_value(self.prompt_context, 'sql_query', '')}",
            f"Батч: {batch_number if batch_number is not None else ''}/{total_batches if total_batches is not None else ''}",
            f"Период: [{period_start}, {period_end})",
            f"Поле времени: {time_column}",
            f"Тип данных: {data_type}",
            f"Строк в куске: {len(rows)}",
            f"Строк с problem-сигналами: {len(critical_rows)}",
            f"Колонки: {', '.join(display_columns)}",
        ]
        if source_stat:
            lines.append(f"Распределение по источникам: {source_stat}")
        if extra_prompt_context:
            lines += [
                "",
                "ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ДАННЫХ:",
                extra_prompt_context,
            ]
        lines += [
            "",
            "ПРАВИЛА АНТИГАЛЛЮЦИНАЦИИ:",
            anti_rules,
            "",
            "Верни обычный текст (не JSON) со строгими секциями:",
            "1) ВРЕМЕННОЙ ДИАПАЗОН БАТЧА",
            "2) КЛАССИФИКАЦИЯ ЗАПИСЕЙ: [РЕЛЕВАНТНО] / [ФОН] / [НЕЯСНО]",
            "3) КЛЮЧЕВЫЕ СОБЫТИЯ (только [РЕЛЕВАНТНО])",
            "4) ПАТТЕРНЫ И АНОМАЛИИ",
            "5) ЦЕПОЧКА СОБЫТИЙ БАТЧА (ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК, A -> B -> C)",
            "6) СВЯЗЬ С АЛЕРТАМИ ([НАЙДЕНО]/[ЧАСТИЧНО]/[НЕ НАЙДЕНО])",
            "7) СИГНАЛЫ ДЛЯ ДАЛЬНЕЙШЕГО АНАЛИЗА",
            "8) ФОРМАТ ВЫВОДА ЦЕПОЧКИ: используй стрелки и узлы в стиле Markdown-схемы.",
            "",
            "Логи (хронологический порядок):",
            logs_text,
        ]
        return _append_chain_requirement("\n".join(lines), "map")

    def _build_reduce_prompt(
        self,
        *,
        period_start: str,
        period_end: str,
        reduce_round: int,
        summaries: List[str],
        sources: Optional[List[str]] = None,
    ) -> str:
        sources_line = (
            f"Источники данных: {', '.join(sources)}. "
            "Ищи причинно-следственные связи МЕЖДУ источниками."
            if sources else ""
        )
        extra_prompt_context = _read_prompt_setting("CONTROL_PLANE_LLM_EXTRA_PROMPT_CONTEXT")
        anti_rules = _resolve_anti_hallucination_rules()
        summaries_text = []
        for idx, text in enumerate(summaries, start=1):
            summaries_text.append(f"[BATCH {idx}]")
            summaries_text.append(text)
            summaries_text.append("")
        rendered_summaries = "\n".join(summaries_text).strip()

        reduce_template = _read_prompt_setting("CONTROL_PLANE_LLM_REDUCE_PROMPT_TEMPLATE")
        if reduce_template:
            rendered = _render_prompt_template(
                reduce_template,
                {
                    "period_start": period_start,
                    "period_end": period_end,
                    "incident_start": _ctx_value(self.prompt_context, "incident_start", period_start),
                    "incident_end": _ctx_value(self.prompt_context, "incident_end", period_end),
                    "incident_description": _ctx_value(self.prompt_context, "incident_description", ""),
                    "alerts_list": _ctx_value(self.prompt_context, "alerts_list", ""),
                    "metrics_context": _ctx_value(self.prompt_context, "metrics_context", ""),
                    "source_name": _ctx_value(self.prompt_context, "source_name", ", ".join(sources or [])),
                    "sql_query": _ctx_value(self.prompt_context, "sql_query", ""),
                    "data_type": _ctx_value(self.prompt_context, "data_type", ""),
                    "reduce_round": reduce_round,
                    "source_names": ", ".join(sources or []),
                    "sources_line": sources_line,
                    "batch_count": len(summaries),
                    "extra_prompt_context": extra_prompt_context,
                    "anti_hallucination_rules": anti_rules,
                    "summaries_text": rendered_summaries,
                    "map_summaries": json.dumps(summaries, ensure_ascii=False),
                    "map_summaries_text": rendered_summaries,
                },
            ).strip()
            return _append_chain_requirement(rendered, "reduce")

        lines = [
            "Это REDUCE-этап расследования инцидента.",
            "Если выше есть контекст инцидента/алертов — привяжи выводы к нему.",
            *(([sources_line]) if sources_line else []),
            "",
            f"Источник: {_ctx_value(self.prompt_context, 'source_name', ', '.join(sources or []))}",
            f"SQL: {_ctx_value(self.prompt_context, 'sql_query', '')}",
            f"Период: [{period_start}, {period_end})",
            f"Reduce round: {reduce_round}",
            f"Количество map-summary: {len(summaries)}",
        ]
        if extra_prompt_context:
            lines += [
                "",
                "ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ДАННЫХ:",
                extra_prompt_context,
            ]
        lines += [
            "",
            "ПРАВИЛА АНТИГАЛЛЮЦИНАЦИИ:",
            anti_rules,
            "",
            "Объедини частичные summary в единый отчёт со строгими секциями:",
            "1) ОБЗОР ИСТОЧНИКА",
            "2) ХРОНОЛОГИЯ КЛЮЧЕВЫХ СОБЫТИЙ (только [РЕЛЕВАНТНО])",
            "3) ЦЕПОЧКА СОБЫТИЙ ИСТОЧНИКА (ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК, несколько цепочек допустимы)",
            "4) СВЯЗЬ МЕЖДУ ЦЕПОЧКАМИ: [СВЯЗАНЫ]/[ВОЗМОЖНО СВЯЗАНЫ]/[НЕЗАВИСИМЫ]",
            "5) ОБЪЯСНЕНИЕ АЛЕРТОВ",
            "6) ПЕРВОПРИЧИНЫ ПО ЦЕПОЧКАМ",
            "7) ПРОБЕЛЫ В ДАННЫХ И РАЗРЫВЫ ЦЕПОЧЕК",
            "8) ФОРМАТ ВЫВОДА ЦЕПОЧКИ: оформи красиво с узлами и стрелками.",
            "",
            "Частичные summary:",
            rendered_summaries,
        ]
        return _append_chain_requirement("\n".join(lines).strip(), "reduce")

    def _build_freeform_prompt(
        self,
        *,
        period_start: str,
        period_end: str,
        structured_summary: str,
        map_summaries: Optional[Sequence[str]] = None,
    ) -> str:
        extra_prompt_context = _read_prompt_setting("CONTROL_PLANE_LLM_EXTRA_PROMPT_CONTEXT")
        anti_rules = _resolve_anti_hallucination_rules()
        map_items = [_normalize_summary_text(item) for item in (map_summaries or [])]
        map_items = [item for item in map_items if item]
        map_summaries_text = "\n\n".join(
            f"[MAP SUMMARY #{idx + 1}]\n{item}" for idx, item in enumerate(map_items)
        )
        freeform_template = _read_prompt_setting("CONTROL_PLANE_LLM_FREEFORM_PROMPT_TEMPLATE")
        if freeform_template:
            rendered = _render_prompt_template(
                freeform_template,
                {
                    "period_start": period_start,
                    "period_end": period_end,
                    "incident_start": _ctx_value(self.prompt_context, "incident_start", period_start),
                    "incident_end": _ctx_value(self.prompt_context, "incident_end", period_end),
                    "incident_description": _ctx_value(self.prompt_context, "incident_description", ""),
                    "alerts_list": _ctx_value(self.prompt_context, "alerts_list", ""),
                    "metrics_context": _ctx_value(self.prompt_context, "metrics_context", ""),
                    "structured_summary": structured_summary,
                    "cross_source_summary": structured_summary,
                    "map_summaries": json.dumps(map_items, ensure_ascii=False),
                    "map_summaries_text": map_summaries_text,
                    "extra_prompt_context": extra_prompt_context,
                    "anti_hallucination_rules": anti_rules,
                },
            ).strip()
            return _append_chain_requirement(rendered, "freeform")
        return _append_chain_requirement("\n".join([
            "На основе структурированного анализа инцидента ниже напиши черновой нарратив.",
            "Это промежуточный результат для SRE-команды — 3-5 абзацев связным текстом.",
            "Включи: что произошло, в какой последовательности, вероятные причины с пометками [ФАКТ]/[ГИПОТЕЗА],",
            "что нужно проверить дополнительно.",
            "Пиши конкретно — ссылайся на реальные timestamp'ы и цитаты из логов, не генерируй абстракции.",
            "Если данных недостаточно для какого-то утверждения — прямо напиши об этом.",
            "Отдельным обязательным пунктом дай наглядную цепочку событий (узлы + стрелки).",
            "",
            "ПРАВИЛА АНТИГАЛЛЮЦИНАЦИИ:",
            anti_rules,
            *((["", "ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ДАННЫХ:", extra_prompt_context]) if extra_prompt_context else []),
            *((["", "MAP SUMMARY ПО БАТЧАМ ЛОГОВ:", map_summaries_text]) if map_summaries_text else []),
            "",
            f"Период: [{period_start}, {period_end})",
            "",
            "Структурированный анализ:",
            structured_summary,
        ]), "freeform")

    def _rank_rows_by_problem_signal(
        self,
        rows: List[Dict[str, Any]],
        columns: Sequence[str],
    ) -> List[Dict[str, Any]]:
        return sorted(rows, key=lambda row: self._row_problem_score(row, columns), reverse=True)

    def _row_problem_score(self, row: Dict[str, Any], columns: Sequence[str]) -> int:
        score = 0
        text_parts = []
        for col in columns:
            value = row.get(col, "")
            if value is None:
                continue
            text_parts.append(str(value).lower())
        joined = " ".join(text_parts)
        for keyword in self.PROBLEM_KEYWORDS:
            if keyword in joined:
                score += 1
        if "level=error" in joined or "level=fatal" in joined:
            score += 2
        if "status=5" in joined or "http 5" in joined:
            score += 1
        return score

    @staticmethod
    def _validate_iso_datetime(value: str) -> None:
        datetime.fromisoformat(value.replace("Z", "+00:00"))

    @staticmethod
    def _truncate(value: str, max_chars: int) -> str:
        if max_chars <= 0:
            return value
        if len(value) <= max_chars:
            return value
        if max_chars <= 3:
            return value[:max_chars]
        return value[: max_chars - 3] + "..."


def build_cross_source_reduce_prompt(
    summaries_by_source: Dict[str, str],
    period_start: str,
    period_end: str,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a cross-source REDUCE prompt that merges per-source MAP→REDUCE results.

    Used in the two-level summarization algorithm for multi-query mode:
      1. Per-source MAP→REDUCE (independent summaries per query).
      2. One cross-source REDUCE LLM call using this prompt.
    """
    ctx = context or {}
    sources = list(summaries_by_source.keys())
    source_blocks: List[str] = []
    for source, summary in summaries_by_source.items():
        source_blocks.append(f"=== {source} ===\n{summary}")
    source_summaries_text = "\n\n".join(source_blocks).strip()
    source_summaries_array = [
        {"source_name": str(source), "reduce_summary": str(summary)}
        for source, summary in summaries_by_source.items()
    ]
    anti_rules = _resolve_anti_hallucination_rules()
    extra_prompt_context = _read_prompt_setting("CONTROL_PLANE_LLM_EXTRA_PROMPT_CONTEXT")
    template = _read_prompt_setting("CONTROL_PLANE_LLM_CROSS_SOURCE_REDUCE_PROMPT_TEMPLATE")
    if template:
        rendered = _render_prompt_template(
            template,
            {
                "period_start": period_start,
                "period_end": period_end,
                "incident_start": _ctx_value(ctx, "incident_start", period_start),
                "incident_end": _ctx_value(ctx, "incident_end", period_end),
                "incident_description": _ctx_value(ctx, "incident_description", ""),
                "alerts_list": _ctx_value(ctx, "alerts_list", ""),
                "metrics_context": _ctx_value(ctx, "metrics_context", ""),
                "source_names": ", ".join(sources),
                "source_count": len(sources),
                "source_name": _ctx_value(ctx, "source_name", ", ".join(sources)),
                "sql_query": _ctx_value(ctx, "sql_query", ""),
                "time_column": _ctx_value(ctx, "time_column", ""),
                "data_type": _ctx_value(ctx, "data_type", ""),
                "source_summaries_text": source_summaries_text,
                "source_summaries": json.dumps(source_summaries_array, ensure_ascii=False),
                "extra_prompt_context": extra_prompt_context,
                "anti_hallucination_rules": anti_rules,
            },
        ).strip()
        return _append_chain_requirement(rendered, "cross")

    lines = [
        "Это финальный CROSS-SOURCE REDUCE: объедини результаты из нескольких источников.",
        "Привяжи выводы к контексту инцидента/алертов, если он указан выше.",
        f"Источники: {', '.join(sources)}.",
        "",
        f"Период: [{period_start}, {period_end})",
    ]
    if extra_prompt_context:
        lines += [
            "",
            "ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ДАННЫХ:",
            extra_prompt_context,
        ]
    lines += [
        "",
        "ПРАВИЛА АНТИГАЛЛЮЦИНАЦИИ:",
        anti_rules,
        "",
        "Верни структурированный отчёт со секциями:",
        "1) ЕДИНАЯ ХРОНОЛОГИЯ ИНЦИДЕНТА",
        "2) КРОСС-КОРРЕЛЯЦИИ МЕЖДУ ИСТОЧНИКАМИ ([ФАКТ]/[ГИПОТЕЗА])",
        "3) ОБЪЯСНЕНИЕ АЛЕРТОВ (финальный вердикт)",
        "4) ЦЕПОЧКА СОБЫТИЙ ИНЦИДЕНТА (ОБЯЗАТЕЛЬНЫЙ ОТДЕЛЬНЫЙ БЛОК)",
        "5) СВЯЗЬ МЕЖДУ ЦЕПОЧКАМИ: [ОДИН ИНЦИДЕНТ]/[ВОЗМОЖНО СВЯЗАНЫ]/[НЕЗАВИСИМЫ]",
        "6) ПЕРВОПРИЧИНЫ ПО ЦЕПОЧКАМ",
        "7) МАСШТАБ ВОЗДЕЙСТВИЯ",
        "8) РЕКОМЕНДАЦИИ ДЛЯ SRE (P0/P1/P2)",
        "9) ПРОБЕЛЫ И ОТКРЫТЫЕ ВОПРОСЫ",
        "",
        "Summary по источникам:",
        source_summaries_text,
    ]
    return _append_chain_requirement("\n".join(lines).strip(), "cross")


def regenerate_reduce_summary_from_map_summaries(
    *,
    map_summaries: Sequence[str],
    period_start: str,
    period_end: str,
    llm_call: LLMTextCaller,
    prompt_context: Optional[Dict[str, Any]] = None,
    on_progress: Optional[ProgressCallback] = None,
    config: Optional[SummarizerConfig] = None,
) -> str:
    """
    Rebuild final REDUCE summary from already prepared MAP summaries.
    Useful for "rerun final summary" without refetching logs.
    """
    prepared = [_normalize_summary_text(item) for item in map_summaries]
    prepared = [item for item in prepared if item]
    if not prepared:
        return "Нет map-summary для повторного REDUCE."

    reducer = PeriodLogSummarizer(
        db_fetch_page=lambda **_: [],
        llm_call=llm_call,
        config=config
        or SummarizerConfig(
            max_cell_chars=int(
                getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_CELL_CHARS", 0)
            ),
            max_summary_chars=int(
                getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_SUMMARY_CHARS", 0)
            ),
            reduce_prompt_max_chars=int(
                getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_REDUCE_PROMPT_MAX_CHARS", 0)
            ),
        ),
        on_progress=on_progress,
        prompt_context=prompt_context or {},
    )
    final_summary, _, _ = reducer._reduce_summaries(
        chunk_summaries=list(prepared),
        period_start=period_start,
        period_end=period_end,
        sources=None,
    )
    return _normalize_summary_text(final_summary) or "Пустой итог повторного REDUCE."


def summarize_logs(
    *,
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
    anomaly: Optional[Dict[str, Any]] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """
    LLM map-reduce summarizer over paged logs from ClickHouse.
    Can be wired as CONTROL_PLANE_SUMMARIZER_CALLABLE=my_summarizer.summarize_logs
    """
    start_iso, end_iso = _normalize_period(
        period_start=period_start,
        period_end=period_end,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    fetch_mode = _resolve_logs_fetch_mode()
    tail_limit = max(int(getattr(settings, "CONTROL_PLANE_LOGS_TAIL_LIMIT", 1000)), 1)
    service = _resolve_service(anomaly)
    page_limit = int(settings.CONTROL_PLANE_LOGS_PAGE_LIMIT)
    total_rows_estimate = _estimate_total_logs(
        anomaly=anomaly,
        period_start=start_iso,
        period_end=end_iso,
        page_limit=page_limit,
        fetch_mode=fetch_mode,
        tail_limit=tail_limit,
    )

    fetch_errors: List[str] = []

    def _on_fetch_error(msg: str) -> None:
        fetch_errors.append(msg)
        if on_progress:
            on_progress("fetch_error", {"error": msg})

    db_fetch_page = _build_db_fetch_page(
        anomaly,
        fetch_mode=fetch_mode,
        tail_limit=tail_limit,
        on_error=_on_fetch_error,
    )
    llm_call = _make_llm_call()
    summarizer = PeriodLogSummarizer(
        db_fetch_page=db_fetch_page,
        llm_call=llm_call,
        config=SummarizerConfig(
            page_limit=page_limit,
            max_cell_chars=int(
                getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_CELL_CHARS", 0)
            ),
            max_summary_chars=int(
                getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_MAX_SUMMARY_CHARS", 0)
            ),
            reduce_prompt_max_chars=int(
                getattr(settings, "CONTROL_PLANE_UI_LOGS_SUMMARY_REDUCE_PROMPT_MAX_CHARS", 0)
            ),
        ),
        on_progress=on_progress,
        prompt_context={
            "incident_start": start_iso,
            "incident_end": end_iso,
            "incident_description": "",
            "alerts_list": "",
            "metrics_context": "",
            "source_name": service,
            "sql_query": _resolve_logs_query_template(),
            "time_column": str(getattr(settings, "CONTROL_PLANE_LOGS_TIMESTAMP_COLUMN", "timestamp")),
            "data_type": "",
        },
    )
    result = summarizer.summarize_period(
        period_start=start_iso,
        period_end=end_iso,
        columns=list(DEFAULT_SUMMARY_COLUMNS),
        total_rows_estimate=total_rows_estimate,
    )
    summary_text = str(result.summary)
    if summary_text and not summary_text.startswith("Сервис:"):
        summary_text = f"Сервис: {service}. {summary_text}"

    return {
        "summary": summary_text,
        "freeform_summary": result.freeform_summary,
        "fetch_errors": fetch_errors,
        "chunk_summaries": result.map_summaries,
        "map_batches": result.map_batches,
        "pages_fetched": result.pages_fetched,
        "rows_processed": result.rows_processed,
        "llm_calls": result.llm_calls,
        "reduce_rounds": result.reduce_rounds,
        "rows_total_estimate": total_rows_estimate,
        "logs_fetch_mode": fetch_mode,
        "logs_tail_limit": tail_limit,
        "source": "llm_map_reduce",
    }
