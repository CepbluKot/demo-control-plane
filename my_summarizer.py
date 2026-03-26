from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import pandas as pd
import requests

from settings import settings


LOGS_SQL_COLUMNS: tuple[str, ...] = ("timestamp", "value")
DEFAULT_SUMMARY_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "message",
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
    max_cell_chars: int = 500
    max_summary_chars: int = 10_000
    keep_map_batches_in_memory: bool = True
    keep_map_summaries_in_result: bool = True


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


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def has_required_env() -> bool:
    return bool(str(settings.OPENAI_API_BASE_DB).strip()) and bool(str(settings.OPENAI_API_KEY_DB).strip())


def _build_chat_completions_url(api_base: str) -> str:
    base = api_base.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def communicate_with_llm(message: str, system_prompt: str = "") -> str:
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
    payload = {
        "model": str(settings.LLM_MODEL_ID),
        "messages": messages,
        "temperature": 0.1,
    }

    response = requests.post(url, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") if isinstance(data, dict) else None
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {})
        content = msg.get("content")
        if content is not None:
            return str(content)
    return str(data)


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
        for key in ("timestamp", "ts", "time", "datetime"):
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
    without_limit_only = re.sub(
        r"(?is)\s+LIMIT\s+\d+\s*$",
        "",
        without_limit_offset,
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
) -> Callable[..., List[Dict[str, Any]]]:
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
        if fetch_mode != "tail_n_logs" and offset > 0 and not has_offset_placeholder:
            return []

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
        else:
            query = rendered_query

        page_df = _query_logs_df(query)
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


def _heuristic_llm_call(prompt: str) -> str:
    lowered = prompt.lower()
    keywords = (
        "error",
        "exception",
        "timeout",
        "failed",
        "fatal",
        "critical",
    )
    hits = [kw for kw in keywords if kw in lowered]
    if not hits:
        return (
            "TOP_PROBLEMS:\n- Явных критичных ошибок в чанке не найдено.\n"
            "EVIDENCE:\n- Низкая плотность проблемных сигналов.\n"
            "HYPOTHESES:\n- Инцидент может быть локальным или вне этого чанка.\n"
            "ACTIONS:\n- Проверить соседние чанки и инфраструктурные события."
        )
    top_hits = ", ".join(sorted(set(hits)))
    return (
        "TOP_PROBLEMS:\n"
        f"- Найдены проблемные сигналы: {top_hits}.\n"
        "EVIDENCE:\n"
        f"- Количество ключевых сигналов в чанке: {len(hits)}.\n"
        "HYPOTHESES:\n"
        "- Ошибки связаны с деградацией зависимости или перегрузкой.\n"
        "ACTIONS:\n"
        "- Проверить ошибки 5xx/timeout и последние деплои."
    )


def _make_llm_call() -> LLMTextCaller:
    if not has_required_env():
        logger.warning(
            "OPENAI_API_BASE_DB/OPENAI_API_KEY_DB не заданы; использую fallback summarizer"
        )
        return _heuristic_llm_call

    def _llm_call(prompt: str) -> str:
        try:
            response = communicate_with_llm(
                message=prompt,
                system_prompt=(
                    "Ты SRE-аналитик. Фокусируйся на признаках деградации, "
                    "ошибках, таймаутах и приоритетных действиях."
                ),
            )
            return str(response).strip()
        except Exception:
            logger.exception("Ошибка вызова communicate_with_llm; использую fallback на этот чанк")
            return _heuristic_llm_call(prompt)

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
    ) -> None:
        self.db_fetch_page = db_fetch_page
        self.llm_call = llm_call
        self.config = config or SummarizerConfig()
        self.on_progress = on_progress

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
        if total_rows_estimate and total_rows_estimate > 0:
            estimated_batch_total = int(
                math.ceil(total_rows_estimate / max(self.config.llm_chunk_rows, 1))
            )
        self._emit_progress(
            "map_start",
            {
                "rows_processed": 0,
                "rows_total": total_rows_estimate,
            },
        )

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
                ranked_chunk = self._rank_rows_by_problem_signal(rows_chunk, columns)
                next_batch_index = map_batch_index
                batch_period_start, batch_period_end = _extract_batch_period(ranked_chunk)
                self._emit_progress(
                    "map_batch_start",
                    {
                        "batch_index": next_batch_index,
                        "batch_total": estimated_batch_total,
                        "batch_logs_count": len(ranked_chunk),
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
                    rows=ranked_chunk,
                )
                chunk_summary = self.llm_call(prompt).strip()
                if not chunk_summary:
                    chunk_summary = "Пустой ответ LLM на map-этапе."
                chunk_summary = self._truncate(chunk_summary, self.config.max_summary_chars)
                map_summaries.append(chunk_summary)
                if self.config.keep_map_batches_in_memory:
                    map_batches.append(
                        {
                            "batch_index": next_batch_index,
                            "rows_count": len(ranked_chunk),
                            "rows": [dict(row) for row in ranked_chunk],
                            "summary": chunk_summary,
                            "batch_period_start": batch_period_start,
                            "batch_period_end": batch_period_end,
                        }
                    )
                rows_mapped += len(ranked_chunk)
                self._emit_progress(
                    "map_batch",
                    {
                        "batch_index": next_batch_index,
                        "batch_total": estimated_batch_total,
                        "batch_summary": chunk_summary,
                        "batch_logs_count": len(ranked_chunk),
                        "batch_logs": [dict(row) for row in ranked_chunk],
                        "batch_period_start": batch_period_start,
                        "batch_period_end": batch_period_end,
                        "rows_processed": rows_mapped,
                        "rows_total": total_rows_estimate,
                    },
                )
                map_batch_index += 1
                llm_calls += 1

            if len(page) < self.config.page_limit:
                break

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
        result_map_summaries = map_summaries if self.config.keep_map_summaries_in_result else []
        return SummarizationResult(
            summary=final_summary,
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
    ) -> tuple[str, int, int]:
        if len(chunk_summaries) == 1:
            return chunk_summaries[0], 0, 0

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

    def _build_chunk_prompt(
        self,
        *,
        period_start: str,
        period_end: str,
        columns: Sequence[str],
        rows: List[Dict[str, Any]],
    ) -> str:
        problem_rows = sum(1 for row in rows if self._row_problem_score(row, columns) > 0)
        lines = [
            "Ты SRE-аналитик. Ищи проблемы, а не общий обзор.",
            "Это MAP-этап: нужно проанализировать только этот кусок логов.",
            "Верни обычный текст (не JSON) со строгими секциями:",
            "1) TOP_PROBLEMS (3-7 пунктов, сортировка по критичности)",
            "2) EVIDENCE (краткие факты из логов)",
            "3) HYPOTHESES (возможные причины)",
            "4) ACTIONS (что проверить/сделать)",
            "Игнорируй рутину и нормальные сообщения, фокус на ошибках и деградациях.",
            "",
            f"Период: [{period_start}, {period_end})",
            f"Строк в этом куске: {len(rows)}",
            f"Строк с problem-сигналами: {problem_rows}",
            f"Колонки: {', '.join(columns)}",
            "",
            "Логи:",
        ]
        for idx, row in enumerate(rows, start=1):
            rendered_parts: List[str] = []
            for col in columns:
                value = row.get(col, "")
                text = self._truncate(str(value), self.config.max_cell_chars)
                rendered_parts.append(f"{col}={text}")
            lines.append(f"{idx}. " + " | ".join(rendered_parts))
        return "\n".join(lines)

    def _build_reduce_prompt(
        self,
        *,
        period_start: str,
        period_end: str,
        reduce_round: int,
        summaries: List[str],
    ) -> str:
        lines = [
            "Ты SRE-аналитик. Это REDUCE-этап map-reduce по логам.",
            "Объедини частичные summary в один итог по проблемам.",
            "Верни обычный текст (не JSON) с секциями:",
            "1) TOP_PROBLEMS (ранжирование critical->high->medium->low)",
            "2) GLOBAL_PATTERNS (повторяющиеся симптомы)",
            "3) ROOT_CAUSE_HYPOTHESES",
            "4) PRIORITY_ACTIONS (сначала самое срочное)",
            "Не теряй критичные инциденты, убирай дубли.",
            "",
            f"Период: [{period_start}, {period_end})",
            f"Reduce round: {reduce_round}",
            "",
            "Частичные summary:",
        ]
        for idx, text in enumerate(summaries, start=1):
            lines.append(f"[SUMMARY {idx}]")
            lines.append(text)
            lines.append("")
        return "\n".join(lines).strip()

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
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 3] + "..."


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

    db_fetch_page = _build_db_fetch_page(
        anomaly,
        fetch_mode=fetch_mode,
        tail_limit=tail_limit,
    )
    llm_call = _make_llm_call()
    summarizer = PeriodLogSummarizer(
        db_fetch_page=db_fetch_page,
        llm_call=llm_call,
        config=SummarizerConfig(page_limit=page_limit),
        on_progress=on_progress,
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
