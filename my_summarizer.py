from __future__ import annotations

import logging
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
    ) -> None:
        self.db_fetch_page = db_fetch_page
        self.llm_call = llm_call
        self.config = config or SummarizerConfig()

    def summarize_period(
        self,
        *,
        period_start: str,
        period_end: str,
        columns: Sequence[str],
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

            for i in range(0, len(page), self.config.llm_chunk_rows):
                rows_chunk = page[i : i + self.config.llm_chunk_rows]
                ranked_chunk = self._rank_rows_by_problem_signal(rows_chunk, columns)
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
                map_batches.append(
                    {
                        "batch_index": len(map_batches),
                        "rows_count": len(ranked_chunk),
                        "rows": [dict(row) for row in ranked_chunk],
                        "summary": chunk_summary,
                    }
                )
                llm_calls += 1

            if len(page) < self.config.page_limit:
                break

        if not map_summaries:
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

        final_summary, reduce_calls, reduce_rounds = self._reduce_summaries(
            chunk_summaries=map_summaries,
            period_start=period_start,
            period_end=period_end,
        )
        llm_calls += reduce_calls
        return SummarizationResult(
            summary=final_summary,
            pages_fetched=pages_fetched,
            rows_processed=rows_processed,
            llm_calls=llm_calls,
            chunk_summaries=len(map_summaries),
            reduce_rounds=reduce_rounds,
            map_summaries=map_summaries,
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
            for i in range(0, len(current), self.config.reduce_group_size):
                group = current[i : i + self.config.reduce_group_size]
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
    service = _resolve_service(anomaly)
    page_limit = int(settings.CONTROL_PLANE_LOGS_PAGE_LIMIT)

    db_fetch_page = _build_db_fetch_page(anomaly)
    llm_call = _make_llm_call()
    summarizer = PeriodLogSummarizer(
        db_fetch_page=db_fetch_page,
        llm_call=llm_call,
        config=SummarizerConfig(page_limit=page_limit),
    )
    result = summarizer.summarize_period(
        period_start=start_iso,
        period_end=end_iso,
        columns=list(DEFAULT_SUMMARY_COLUMNS),
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
        "source": "llm_map_reduce",
    }
