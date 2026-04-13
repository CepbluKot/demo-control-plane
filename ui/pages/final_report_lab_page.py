from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import requests
import streamlit as st

from settings import settings
from .logs_summary_page import (
    FINAL_REPORT_SECTIONS,
    _generate_sectional_freeform_summary,
    _generate_sectional_structured_summary,
    _normalize_summary_text,
)


DEFAULT_LAB_API_BASE = "https://phoenix.scm-test.int.gazprombank.ru/api/v1/chat/completions"
DEFAULT_LAB_MODEL = "PNX.QWEN3 235b a22b instruct"
# Backward-compatible aliases for older tests/imports.
DEFAULT_GROQ_API_BASE = DEFAULT_LAB_API_BASE
DEFAULT_GROQ_MODEL = DEFAULT_LAB_MODEL
FINAL_REPORT_LAB_TOKENS_PER_MINUTE_LIMIT = 30_000
CHARS_PER_TOKEN_ESTIMATE = 3
# FAT preset for local stress-tests of final-stage merge/section generation.
DEFAULT_SYNTH_CHUNK_COUNT = 24
DEFAULT_SYNTH_EVENTS_PER_CHUNK = 240
DEFAULT_SYNTH_DETAILS_PER_EVENT = 3
DEFAULT_SYNTH_PARAGRAPHS_PER_CHUNK = 10
DEFAULT_SYNTH_SEED = 20260413
AUTO_SEED_PROFILE_VERSION = 2


@dataclass(frozen=True)
class FinalReportLabPageDeps:
    logger: logging.Logger


def _estimate_tokens(text: str) -> int:
    raw = str(text or "")
    if not raw:
        return 0
    return max((len(raw) + CHARS_PER_TOKEN_ESTIMATE - 1) // CHARS_PER_TOKEN_ESTIMATE, 1)


class _TokenPerMinuteLimiter:
    def __init__(self, *, tokens_per_minute: int) -> None:
        self._limit = max(int(tokens_per_minute), 1)
        self._events: List[tuple[float, int]] = []

    def _prune(self, now: float) -> None:
        if not self._events:
            return
        cutoff = now - 60.0
        self._events = [(ts, tokens) for ts, tokens in self._events if ts > cutoff]

    def acquire(self, tokens: int, *, now: Optional[float] = None) -> float:
        requested = max(int(tokens), 0)
        if requested <= 0:
            return 0.0
        waited = 0.0
        while True:
            current_now = float(now if now is not None else time.monotonic())
            self._prune(current_now)
            used = sum(tokens_used for _, tokens_used in self._events)
            if used + requested <= self._limit:
                self._events.append((current_now, requested))
                return waited
            if not self._events:
                # Requested alone is larger than limit; still allow after marking usage.
                self._events.append((current_now, requested))
                return waited
            oldest_ts = min(ts for ts, _ in self._events)
            sleep_for = max((oldest_ts + 60.0) - current_now, 0.01)
            time.sleep(sleep_for)
            waited += sleep_for
            now = None


def _safe_filename(text: str, *, fallback: str = "item") -> str:
    raw = str(text or "").strip().lower()
    if not raw:
        return fallback
    slug = re.sub(r"[^a-z0-9а-яё_-]+", "_", raw, flags=re.IGNORECASE)
    slug = slug.strip("_")
    return slug or fallback


def _write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text or ""), encoding="utf-8")


def _write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_jsonl_file(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(dict(row), ensure_ascii=False)
        for row in (rows or [])
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_file_link_rows(run_dir: Path, manifest: Dict[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    def _add(label: str, rel_path: str) -> None:
        rows.append(
            {
                "label": label,
                "path": str((run_dir / rel_path).resolve()),
                "relative_path": rel_path,
            }
        )

    _add("manifest", "manifest.json")
    _add("inputs/base_summary", "inputs/base_summary.txt")
    _add("inputs/map_summaries", "inputs/map_summaries.txt")
    for kind in ("structured", "freeform"):
        if kind in manifest.get("outputs", {}):
            _add(f"{kind}/merged_report", f"outputs/{kind}/merged_report.md")
            _add(f"{kind}/llm_calls_jsonl", f"outputs/{kind}/llm_calls.jsonl")
    return rows


def _persist_final_report_lab_artifacts(
    *,
    user_goal: str,
    metrics_context: str,
    period_start: str,
    period_end: str,
    chunks: Sequence[str],
    base_summary: str,
    map_summaries_text: str,
    results: Dict[str, Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"final_report_lab_{ts}_{int(time.time() * 1000) % 1000:03d}"
    run_dir = Path(str(getattr(settings, "CONTROL_PLANE_ARTIFACTS_DIR", "artifacts"))).resolve() / "final_report_lab" / run_id

    # Inputs and transformation seeds
    _write_text_file(run_dir / "inputs" / "user_goal.txt", user_goal)
    _write_text_file(run_dir / "inputs" / "metrics_context.txt", metrics_context)
    _write_text_file(run_dir / "inputs" / "period_start.txt", period_start)
    _write_text_file(run_dir / "inputs" / "period_end.txt", period_end)
    _write_text_file(run_dir / "inputs" / "base_summary.txt", base_summary)
    _write_text_file(run_dir / "inputs" / "map_summaries.txt", map_summaries_text)
    chunks_dir = run_dir / "inputs" / "chunks"
    for idx, chunk in enumerate(chunks, start=1):
        _write_text_file(chunks_dir / f"chunk_{idx:04d}.txt", str(chunk or ""))

    outputs_manifest: Dict[str, Any] = {}
    for kind, result in results.items():
        if not isinstance(result, dict):
            continue
        kind_dir = run_dir / "outputs" / str(kind)
        sections_dir = kind_dir / "sections"
        calls_dir = kind_dir / "llm_calls"
        merged_text = str(result.get("merged_text") or "")
        _write_text_file(kind_dir / "merged_report.md", merged_text)
        section_rows: List[Dict[str, Any]] = []
        for idx, section in enumerate(result.get("sections") or [], start=1):
            title = str(section.get("title") or f"section_{idx}")
            safe_title = _safe_filename(title, fallback=f"section_{idx}")
            text = str(section.get("text") or "")
            rel_path = f"outputs/{kind}/sections/{idx:02d}_{safe_title}.md"
            _write_text_file(run_dir / rel_path, text)
            section_rows.append(
                {
                    "index": idx,
                    "title": title,
                    "path": rel_path,
                    "chars": len(text),
                }
            )

        llm_rows: List[Dict[str, Any]] = []
        for call in result.get("llm_calls") or []:
            call_no = int(call.get("call", len(llm_rows) + 1))
            call_prefix = f"call_{call_no:04d}"
            prompt_path = f"outputs/{kind}/llm_calls/{call_prefix}_prompt.txt"
            response_path = f"outputs/{kind}/llm_calls/{call_prefix}_response.txt"
            merge_prev_path = f"outputs/{kind}/llm_calls/{call_prefix}_merge_previous.txt"
            merge_base_path = f"outputs/{kind}/llm_calls/{call_prefix}_merge_base.txt"
            merge_map_path = f"outputs/{kind}/llm_calls/{call_prefix}_merge_map.txt"
            _write_text_file(run_dir / prompt_path, str(call.get("prompt_text") or ""))
            _write_text_file(run_dir / response_path, str(call.get("response_text") or ""))
            _write_text_file(run_dir / merge_prev_path, str(call.get("merge_previous_sections_text") or ""))
            _write_text_file(run_dir / merge_base_path, str(call.get("merge_base_summary_text") or ""))
            _write_text_file(run_dir / merge_map_path, str(call.get("merge_map_summaries_text") or ""))

            llm_rows.append(
                {
                    "call": call_no,
                    "section_index": int(call.get("section_index") or 0),
                    "section_title": str(call.get("section_title") or ""),
                    "merge_section_label": str(call.get("merge_section_label") or ""),
                    "status": str(call.get("status") or ""),
                    "error": str(call.get("error") or ""),
                    "prompt_chars": int(call.get("prompt_chars") or 0),
                    "response_chars": int(call.get("response_chars") or 0),
                    "elapsed_sec": float(call.get("elapsed_sec") or 0.0),
                    "prompt_path": prompt_path,
                    "response_path": response_path,
                    "merge_previous_path": merge_prev_path,
                    "merge_base_path": merge_base_path,
                    "merge_map_path": merge_map_path,
                }
            )

        _write_jsonl_file(kind_dir / "llm_calls.jsonl", llm_rows)
        _write_json_file(kind_dir / "sections_index.json", section_rows)
        outputs_manifest[str(kind)] = {
            "merged_report_path": f"outputs/{kind}/merged_report.md",
            "sections_index_path": f"outputs/{kind}/sections_index.json",
            "llm_calls_path": f"outputs/{kind}/llm_calls.jsonl",
            "sections_count": len(section_rows),
            "llm_calls_count": len(llm_rows),
            "elapsed_sec": float(result.get("elapsed_sec") or 0.0),
        }

    manifest = {
        "type": "final_report_lab_run",
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "inputs": {
            "chunks_count": len(chunks),
            "base_summary_chars": len(str(base_summary or "")),
            "map_summaries_chars": len(str(map_summaries_text or "")),
            "period_start": period_start,
            "period_end": period_end,
        },
        "outputs": outputs_manifest,
    }
    _write_json_file(run_dir / "manifest.json", manifest)

    link_rows = _build_file_link_rows(run_dir, manifest)
    for row in link_rows:
        logger.info("FINAL_REPORT_LAB artifact | label=%s | path=%s", row["label"], row["path"])
    logger.info("FINAL_REPORT_LAB artifact manifest | path=%s", str((run_dir / "manifest.json").resolve()))

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest_path": str((run_dir / "manifest.json").resolve()),
        "files": link_rows,
    }


def _build_chat_completions_url(api_base: str) -> str:
    base = str(api_base or "").strip()
    if not base:
        return str(getattr(settings, "OPENAI_API_BASE_DB", "") or "").strip() or DEFAULT_LAB_API_BASE
    normalized = base.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/chat/completions"


def _extract_openai_assistant_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Пустой ответ LLM: нет поля choices.")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("Пустой ответ LLM: нет message в choices[0].")
    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            piece = item.get("text")
            if isinstance(piece, str) and piece.strip():
                parts.append(piece.strip())
        text = "\n".join(parts).strip()
    else:
        text = str(content or "").strip()
    if not text:
        raise ValueError("Пустой ответ LLM: content пустой.")
    return text


def _safe_response_text(response: requests.Response, *, limit: int = 2000) -> str:
    text = ""
    try:
        text = str(response.text or "")
    except Exception:
        text = ""
    text = text.strip()
    if not text:
        return "<empty>"
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated]"


def _make_groq_chat_call(
    *,
    api_base: str,
    api_key: str,
    model: str,
    timeout_seconds: float,
    max_retries: int,
    temperature: float,
    max_tokens: int,
    logger: Optional[logging.Logger] = None,
) -> Callable[[str], str]:
    url = _build_chat_completions_url(api_base)
    retries = max(int(max_retries), 1)
    timeout = max(float(timeout_seconds), 5.0)
    model_name = str(model or "").strip() or str(getattr(settings, "LLM_MODEL_ID", "") or "").strip() or DEFAULT_LAB_MODEL
    token_limit = max(int(max_tokens), 64)
    temp = max(min(float(temperature), 2.0), 0.0)
    headers = {
        "Authorization": f"Bearer {str(api_key).strip()}",
        "Content-Type": "application/json",
    }
    token_limiter = _TokenPerMinuteLimiter(
        tokens_per_minute=FINAL_REPORT_LAB_TOKENS_PER_MINUTE_LIMIT
    )

    def _call(prompt: str) -> str:
        last_exc: Optional[Exception] = None
        prompt_text = str(prompt or "")
        for attempt in range(1, retries + 1):
            started = time.monotonic()
            try:
                token_budget = _estimate_tokens(prompt_text) + token_limit
                waited_for_budget = token_limiter.acquire(token_budget)
                if logger is not None and waited_for_budget > 0:
                    logger.info(
                        "FINAL_REPORT_LAB tpm_wait | waited=%.2fs | budget_tokens=%s | limit_per_min=%s",
                        waited_for_budget,
                        token_budget,
                        FINAL_REPORT_LAB_TOKENS_PER_MINUTE_LIMIT,
                    )
                payload = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "temperature": temp,
                    "max_tokens": token_limit,
                }
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if response.status_code >= 400:
                    body = _safe_response_text(response)
                    raise RuntimeError(
                        f"HTTP {response.status_code} при вызове LLM: {body}"
                    )
                data = response.json()
                text = _extract_openai_assistant_text(data)
                if logger is not None:
                    logger.info(
                        "FINAL_REPORT_LAB llm_call ok | provider=%s | model=%s | attempt=%s/%s | elapsed=%.2fs | prompt_chars=%s | response_chars=%s | budget_tokens=%s",
                        "openai_compatible",
                        model_name,
                        attempt,
                        retries,
                        time.monotonic() - started,
                        len(prompt_text),
                        len(text),
                        token_budget,
                    )
                return text
            except Exception as exc:  # noqa: BLE001
                last_exc = exc if isinstance(exc, Exception) else Exception(str(exc))
                if logger is not None:
                    logger.warning(
                        "FINAL_REPORT_LAB llm_call failed | attempt=%s/%s | error=%s",
                        attempt,
                        retries,
                        exc,
                    )
                if attempt < retries:
                    time.sleep(min(2.0 ** (attempt - 1), 8.0))
        raise RuntimeError(
            f"LLM вызов не удался после {retries} попыток: {last_exc}"
        ) from last_exc

    return _call


def _build_synthetic_report_chunks(
    *,
    chunk_count: int,
    events_per_chunk: int,
    details_per_event: int,
    paragraphs_per_chunk: int,
    seed: int,
) -> List[str]:
    rng = random.Random(int(seed))
    normalized_chunk_count = max(int(chunk_count), 1)
    normalized_events = max(int(events_per_chunk), 1)
    normalized_details = max(int(details_per_event), 1)
    normalized_paragraphs = max(int(paragraphs_per_chunk), 1)

    services = [
        "airflow-scheduler",
        "gateway",
        "payments",
        "billing",
        "auth",
        "reporting",
    ]
    severities = ["INFO", "WARN", "ERROR", "CRITICAL"]
    nodes = ["node-a1", "node-b2", "node-c3", "node-d4"]
    hypotheses = [
        "скачок задержки в downstream-зависимости",
        "деградация DNS/сетевой связности",
        "проблема с ротацией секретов",
        "конкурентная блокировка в БД",
        "каскад повторных ретраев после таймаута",
    ]
    base_dt = datetime(2026, 3, 18, 0, 0, 0, tzinfo=timezone.utc)
    global_index = 0
    chunks: List[str] = []

    for chunk_idx in range(normalized_chunk_count):
        lines: List[str] = []
        lines.append(f"### L1 summary chunk {chunk_idx + 1}")
        lines.append("#### Timeline событий")
        for event_idx in range(normalized_events):
            global_index += 1
            ts = base_dt + timedelta(seconds=global_index * 11)
            service = services[(global_index - 1) % len(services)]
            severity = severities[rng.randint(0, len(severities) - 1)]
            node = nodes[(chunk_idx + event_idx) % len(nodes)]
            lines.append(
                f"- [ФАКТ] {ts.isoformat()} service={service} severity={severity} node={node} "
                f"event_id=e{chunk_idx + 1:02d}_{event_idx + 1:03d}"
            )
            for detail_idx in range(normalized_details):
                hyp = hypotheses[rng.randint(0, len(hypotheses) - 1)]
                lines.append(
                    "  "
                    f"detail_{detail_idx + 1}: наблюдался сбой по цепочке ingress -> {service} -> db, "
                    f"error_signature=SIG{(global_index * 13 + detail_idx) % 997:03d}, "
                    f"предположение={hyp}."
                )
        lines.append("#### Гипотезы")
        for paragraph_idx in range(normalized_paragraphs):
            hyp = hypotheses[(chunk_idx + paragraph_idx) % len(hypotheses)]
            lines.append(
                f"- [ГИПОТЕЗА] {hyp}; confidence={0.40 + 0.08 * (paragraph_idx % 5):.2f}; "
                f"проверка=сопоставить timeline и метрики по сервисам {services[chunk_idx % len(services)]}/{services[(chunk_idx + 1) % len(services)]}."
            )
        lines.append("#### Рекомендации")
        lines.append("- Снять p95/p99 latency и ошибки по минутам.")
        lines.append("- Проверить ретраи upstream/downstream и saturation.")
        lines.append("- Сверить журнал деплоев с началом деградации.")
        chunks.append("\n".join(lines))
    return chunks


def _merge_synthetic_chunks(chunks: Sequence[str]) -> str:
    parts = [str(chunk or "").strip() for chunk in chunks if str(chunk or "").strip()]
    if not parts:
        return ""
    return "\n\n".join(parts)


def _build_map_summaries_text(chunks: Sequence[str]) -> str:
    lines: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        text = str(chunk or "").strip()
        if not text:
            continue
        lines.append(f"=== MAP BATCH {idx:04d} ===")
        lines.append(text)
    return "\n\n".join(lines)


def _extract_section_label_from_prompt(prompt_text: str) -> str:
    match = re.search(r"СЕКЦИЯ\s+(\d+/\d+):\s*(.+)", str(prompt_text or ""))
    if not match:
        return ""
    return f"{match.group(1)} — {match.group(2).strip()}"


def _extract_between_markers(
    text: str,
    start_marker: str,
    end_markers: Sequence[str],
) -> str:
    source = str(text or "")
    start_pos = source.find(start_marker)
    if start_pos < 0:
        return ""
    body_start = start_pos + len(start_marker)
    tail = source[body_start:]
    end_pos = len(tail)
    for marker in end_markers:
        pos = tail.find(marker)
        if pos >= 0 and pos < end_pos:
            end_pos = pos
    return tail[:end_pos].strip()


def _extract_merge_context_from_prompt(prompt_text: str, kind: str) -> Dict[str, str]:
    prompt = str(prompt_text or "")
    if kind == "structured":
        previous_marker = "УЖЕ НАПИСАННЫЕ СЕКЦИИ СТРУКТУРИРОВАННОГО ОТЧЁТА:\n"
        base_marker = "БАЗОВЫЙ REDUCE SUMMARY (опорный контекст):\n"
        map_marker = "MAP SUMMARY ПО ВСЕМ БАТЧАМ (опорный контекст):\n"
    else:
        previous_marker = "УЖЕ НАПИСАННЫЕ СЕКЦИИ:\n"
        base_marker = "СТРУКТУРИРОВАННЫЙ АНАЛИЗ (опорный контекст):\n"
        map_marker = "MAP SUMMARY ПО ВСЕМ БАТЧАМ (опорный контекст):\n"

    previous_text = _extract_between_markers(
        prompt,
        previous_marker,
        [base_marker, map_marker, "Верни только текст текущей секции"],
    )
    base_text = _extract_between_markers(
        prompt,
        base_marker,
        [map_marker, "Верни только текст текущей секции"],
    )
    map_text = _extract_between_markers(
        prompt,
        map_marker,
        ["Верни только текст текущей секции"],
    )
    section_label = _extract_section_label_from_prompt(prompt)
    recipe = (
        "merge("
        "already_written_sections + base_structured_summary + map_batches_summary"
        ") -> current_section"
    )
    return {
        "section_label": section_label,
        "merge_recipe": recipe,
        "previous_sections_text": previous_text,
        "base_summary_text": base_text,
        "map_summaries_text": map_text,
    }


def _run_sectional_generation(
    *,
    kind: str,
    llm_call: Callable[[str], str],
    base_summary: str,
    user_goal: str,
    period_start: str,
    period_end: str,
    stats: Dict[str, Any],
    metrics_context: str,
    map_summaries_text: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    total = len(FINAL_REPORT_SECTIONS)
    section_table: List[Dict[str, Any]] = []
    llm_calls_table: List[Dict[str, Any]] = []
    started_by_idx: Dict[int, float] = {}
    current_section_idx = 0
    current_section_title = ""
    progress_bar = st.progress(0.0, text=f"{kind}: 0/{total}")
    status_placeholder = st.empty()

    def _tracked_llm_call(prompt: str) -> str:
        started = time.monotonic()
        prompt_text = str(prompt or "")
        merge_ctx = _extract_merge_context_from_prompt(prompt_text, kind)
        row: Dict[str, Any] = {
            "call": len(llm_calls_table) + 1,
            "section_index": current_section_idx,
            "section_title": current_section_title,
            "prompt_chars": len(prompt_text),
            "status": "ok",
            "response_chars": 0,
            "response_text": "",
            "elapsed_sec": 0.0,
            "error": "",
            "prompt_text": prompt_text,
            "merge_recipe": str(merge_ctx.get("merge_recipe") or ""),
            "merge_section_label": str(merge_ctx.get("section_label") or ""),
            "merge_previous_sections_text": str(merge_ctx.get("previous_sections_text") or ""),
            "merge_base_summary_text": str(merge_ctx.get("base_summary_text") or ""),
            "merge_map_summaries_text": str(merge_ctx.get("map_summaries_text") or ""),
        }
        try:
            response = _normalize_summary_text(llm_call(prompt_text))
            row["response_chars"] = len(response)
            row["response_text"] = response
            row["elapsed_sec"] = round(time.monotonic() - started, 3)
            llm_calls_table.append(row)
            return response
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = str(exc)
            row["elapsed_sec"] = round(time.monotonic() - started, 3)
            llm_calls_table.append(row)
            raise

    def _on_start(section_idx: int, section_total: int, title: str) -> None:
        nonlocal current_section_idx, current_section_title
        current_section_idx = int(section_idx)
        current_section_title = str(title or "")
        started_by_idx[section_idx] = time.monotonic()
        progress_bar.progress(
            max((section_idx - 1) / max(section_total, 1), 0.0),
            text=f"{kind}: {section_idx - 1}/{section_total}",
        )
        status_placeholder.info(f"{kind}: старт секции {section_idx}/{section_total} — {title}")

    def _on_done(section_idx: int, section_total: int, title: str) -> None:
        elapsed = time.monotonic() - started_by_idx.pop(section_idx, time.monotonic())
        section_table.append(
            {
                "section": section_idx,
                "title": title,
                "elapsed_sec": round(elapsed, 3),
            }
        )
        progress_bar.progress(
            min(section_idx / max(section_total, 1), 1.0),
            text=f"{kind}: {section_idx}/{section_total}",
        )
        status_placeholder.success(f"{kind}: готово {section_idx}/{section_total} — {title}")

    run_started = time.monotonic()
    if kind == "structured":
        merged_text, sections = _generate_sectional_structured_summary(
            llm_call=_tracked_llm_call,
            base_summary=base_summary,
            user_goal=user_goal,
            period_start=period_start,
            period_end=period_end,
            stats=stats,
            metrics_context=metrics_context,
            map_summaries_text=map_summaries_text,
            on_section_start=_on_start,
            on_section_done=_on_done,
            logger=logger,
        )
    else:
        merged_text, sections = _generate_sectional_freeform_summary(
            llm_call=_tracked_llm_call,
            final_summary=base_summary,
            user_goal=user_goal,
            period_start=period_start,
            period_end=period_end,
            stats=stats,
            metrics_context=metrics_context,
            map_summaries_text=map_summaries_text,
            on_section_start=_on_start,
            on_section_done=_on_done,
            logger=logger,
        )
    progress_bar.progress(1.0, text=f"{kind}: завершено {total}/{total}")
    elapsed_total = round(time.monotonic() - run_started, 3)
    sections_with_lengths: List[Dict[str, Any]] = []
    for idx, item in enumerate(sections, start=1):
        title = str(item.get("title") or f"Секция {idx}")
        text = str(item.get("text") or "")
        event_row = next((row for row in section_table if int(row.get("section", -1)) == idx), None)
        sections_with_lengths.append(
            {
                "section": idx,
                "title": title,
                "chars": len(text),
                "elapsed_sec": float(event_row.get("elapsed_sec", 0.0)) if event_row else 0.0,
            }
        )
    return {
        "kind": kind,
        "merged_text": merged_text,
        "sections": sections,
        "section_stats": sections_with_lengths,
        "llm_calls": llm_calls_table,
        "elapsed_sec": elapsed_total,
    }


def _default_user_goal_text() -> str:
    return (
        "Инцидент: массовые ошибки 5xx на gateway и деградация latency в платежном контуре.\n"
        "Период: 2026-03-18 00:00:00+03:00 — 2026-03-18 06:00:00+03:00.\n"
        "Проверь связь с ретраями, возможными проблемами DNS и перегрузкой БД.\n"
        "Нужен вывод по первопричине, влиянию и конкретным действиям SRE."
    )


def _render_stage_result(
    *,
    kind: str,
    result: Dict[str, Any],
    show_merge_inputs: bool,
    show_prompts: bool,
) -> None:
    st.dataframe(result.get("section_stats") or [], use_container_width=True, hide_index=True)
    st.dataframe(
        [
            {
                "call": row.get("call"),
                "section": row.get("merge_section_label") or row.get("section_title") or "",
                "status": row.get("status"),
                "prompt_chars": row.get("prompt_chars"),
                "response_chars": row.get("response_chars"),
                "elapsed_sec": row.get("elapsed_sec"),
                "error": row.get("error", ""),
            }
            for row in (result.get("llm_calls") or [])
        ],
        use_container_width=True,
        hide_index=True,
    )
    if show_merge_inputs:
        st.markdown("##### Что С Чем Мержим (Полный Текст)")
        for call in result.get("llm_calls") or []:
            call_id = int(call.get("call", 0))
            section_label = str(call.get("merge_section_label") or call.get("section_title") or "")
            with st.expander(
                f"{kind} merge #{call_id} | {section_label or 'без секции'}",
                expanded=False,
            ):
                st.code(str(call.get("merge_recipe") or ""), language="text")
                st.markdown("`already_written_sections`")
                st.text_area(
                    f"{kind}_merge_prev_{call_id}",
                    value=str(call.get("merge_previous_sections_text") or ""),
                    height=220,
                    label_visibility="collapsed",
                )
                st.markdown("`base_structured_summary`")
                st.text_area(
                    f"{kind}_merge_base_{call_id}",
                    value=str(call.get("merge_base_summary_text") or ""),
                    height=220,
                    label_visibility="collapsed",
                )
                st.markdown("`map_batches_summary`")
                st.text_area(
                    f"{kind}_merge_map_{call_id}",
                    value=str(call.get("merge_map_summaries_text") or ""),
                    height=220,
                    label_visibility="collapsed",
                )
    merged_text = str(result.get("merged_text") or "")
    st.text_area(
        f"{kind.capitalize()} merged report",
        value=merged_text,
        height=320,
    )
    st.download_button(
        label=f"Скачать {kind}.md",
        data=merged_text.encode("utf-8"),
        file_name=f"final_report_lab_{kind}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    if show_prompts:
        for call in result.get("llm_calls") or []:
            call_id = int(call.get("call", 0))
            with st.expander(f"{kind} prompt #{call_id} ({call.get('status')})", expanded=False):
                st.text_area(
                    f"{kind}_prompt_{call_id}",
                    value=str(call.get("prompt_text") or ""),
                    height=260,
                )


def render_final_report_lab_page(deps: FinalReportLabPageDeps) -> None:
    st.title("Final Report Lab")
    st.caption(
        "Отдельная песочница для секционной финальной генерации отчёта: "
        "генерируем большие synthetic-summary и гоняем только финальные merge-вызовы."
    )
    st.info(
        "Ключ API вводится только для этого экрана. Не храните секрет в коде. "
        "Если ключ уже светился в чате/логах — лучше выпустить новый."
    )

    state = st.session_state
    state.setdefault("frl_chunks", [])
    state.setdefault("frl_base_summary", "")
    state.setdefault("frl_map_summaries_text", "")
    state.setdefault("frl_user_goal", _default_user_goal_text())
    state.setdefault("frl_metrics_context", "Метрики: всплеск error-rate до 12%, p99 latency вырос в 4.2 раза.")
    state.setdefault("frl_period_start", "2026-03-18T00:00:00+03:00")
    state.setdefault("frl_period_end", "2026-03-18T06:00:00+03:00")
    state.setdefault(
        "frl_api_base",
        str(getattr(settings, "OPENAI_API_BASE_DB", "") or "").strip()
        or str(os.getenv("OPENAI_API_BASE_DB", "") or "").strip()
        or DEFAULT_LAB_API_BASE,
    )
    state.setdefault(
        "frl_model",
        str(getattr(settings, "LLM_MODEL_ID", "") or "").strip()
        or str(os.getenv("LLM_MODEL_ID", "") or "").strip()
        or DEFAULT_LAB_MODEL,
    )
    state.setdefault(
        "frl_api_key",
        str(getattr(settings, "OPENAI_API_KEY_DB", "") or "").strip()
        or str(os.getenv("OPENAI_API_KEY_DB", "") or "").strip(),
    )
    state.setdefault("frl_last_results", {})
    state.setdefault("frl_last_artifacts", {})
    state.setdefault("frl_auto_seeded", False)
    state.setdefault("frl_auto_seed_profile_version", 0)

    # One-time migration to new default model for existing sessions
    # that still keep external providers defaults.
    if str(state.get("frl_model") or "").strip() in {"qwen/qwen3-32b", "llama-4-scout-17b", "Gemma 4 31B"}:
        state["frl_model"] = DEFAULT_LAB_MODEL
    current_api_base = str(state.get("frl_api_base") or "").strip()
    if (
        "generativelanguage.googleapis.com" in current_api_base.lower()
        or current_api_base == "https://api.groq.com/openai/v1/chat/completions"
    ):
        state["frl_api_base"] = DEFAULT_LAB_API_BASE

    # Auto-seed fat dataset so we can immediately test only final problematic stages.
    # Uses versioning to transparently refresh old/smaller presets in existing sessions.
    current_seed_version = int(state.get("frl_auto_seed_profile_version") or 0)
    needs_refresh = current_seed_version != AUTO_SEED_PROFILE_VERSION
    if needs_refresh:
        auto_chunks = _build_synthetic_report_chunks(
            chunk_count=DEFAULT_SYNTH_CHUNK_COUNT,
            events_per_chunk=DEFAULT_SYNTH_EVENTS_PER_CHUNK,
            details_per_event=DEFAULT_SYNTH_DETAILS_PER_EVENT,
            paragraphs_per_chunk=DEFAULT_SYNTH_PARAGRAPHS_PER_CHUNK,
            seed=DEFAULT_SYNTH_SEED,
        )
        state["frl_chunks"] = auto_chunks
        # Stage-2 transformation is manual, so only stage-1 output is pre-seeded.
        state["frl_base_summary"] = ""
        state["frl_map_summaries_text"] = ""
        state["frl_last_results"] = {}
        state["frl_last_artifacts"] = {}
        state["frl_auto_seeded"] = True
        state["frl_auto_seed_profile_version"] = AUTO_SEED_PROFILE_VERSION
        deps.logger.info(
            "FINAL_REPORT_LAB auto-seeded FAT stage-1 payload | version=%s | chunks=%s",
            AUTO_SEED_PROFILE_VERSION,
            len(auto_chunks),
        )

    state.setdefault("frl_stage_events", [])

    def _append_stage_event(stage: str, status: str, details: str = "") -> None:
        events = list(state.get("frl_stage_events") or [])
        event = {
            "ts_utc": datetime.now(tz=timezone.utc).isoformat(),
            "stage": stage,
            "status": status,
            "details": details,
        }
        events.append(event)
        state["frl_stage_events"] = events[-400:]
        deps.logger.info(
            "FINAL_REPORT_LAB stage_event | stage=%s | status=%s | details=%s",
            stage,
            status,
            details,
        )

    def _render_llm_controls(prefix: str) -> tuple[str, str, str, float, int, float, int]:
        st.caption(
            f"Ограничение страницы: до {FINAL_REPORT_LAB_TOKENS_PER_MINUTE_LIMIT:,} токенов/мин "
            "(оценка ~1 токен на 3 символа; считаем prompt + max_tokens)."
        )
        llm_cols = st.columns(3)
        with llm_cols[0]:
            api_base_local = st.text_input(
                "API base/url",
                value=str(state.get("frl_api_base") or DEFAULT_LAB_API_BASE),
                key=f"{prefix}_api_base",
                help="OpenAI-compatible endpoint рабочего стенда (обычно .../chat/completions).",
            )
        with llm_cols[1]:
            model_local = st.text_input(
                "Model",
                value=str(state.get("frl_model") or DEFAULT_LAB_MODEL),
                key=f"{prefix}_model",
            )
        with llm_cols[2]:
            api_key_local = st.text_input(
                "API key",
                value=str(state.get("frl_api_key") or ""),
                key=f"{prefix}_api_key",
                type="password",
                help="По умолчанию берётся из OPENAI_API_KEY_DB.",
            )
        cfg_cols = st.columns(4)
        with cfg_cols[0]:
            timeout_local = st.number_input(
                "Timeout (sec)",
                min_value=5.0,
                max_value=1800.0,
                value=float(state.get("frl_timeout", 120.0) or 120.0),
                step=5.0,
                key=f"{prefix}_timeout",
            )
        with cfg_cols[1]:
            retries_local = st.number_input(
                "Max retries",
                min_value=1,
                max_value=10,
                value=int(state.get("frl_retries", 3) or 3),
                step=1,
                key=f"{prefix}_retries",
            )
        with cfg_cols[2]:
            temperature_local = st.number_input(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=float(state.get("frl_temperature", 0.2) or 0.2),
                step=0.1,
                key=f"{prefix}_temperature",
            )
        with cfg_cols[3]:
            max_tokens_local = st.number_input(
                "Max tokens",
                min_value=64,
                max_value=65536,
                value=int(state.get("frl_max_tokens", 4096) or 4096),
                step=128,
                key=f"{prefix}_max_tokens",
            )

        state["frl_api_base"] = str(api_base_local or "")
        state["frl_model"] = str(model_local or "")
        state["frl_api_key"] = str(api_key_local or "")
        state["frl_timeout"] = float(timeout_local)
        state["frl_retries"] = int(retries_local)
        state["frl_temperature"] = float(temperature_local)
        state["frl_max_tokens"] = int(max_tokens_local)
        return (
            str(api_base_local or ""),
            str(model_local or ""),
            str(api_key_local or ""),
            float(timeout_local),
            int(retries_local),
            float(temperature_local),
            int(max_tokens_local),
        )

    last_results = state.get("frl_last_results") or {}
    has_chunks = bool(state.get("frl_chunks"))
    has_transform = bool(str(state.get("frl_base_summary") or "").strip()) and bool(
        str(state.get("frl_map_summaries_text") or "").strip()
    )
    has_structured = isinstance(last_results.get("structured"), dict)
    has_freeform = isinstance(last_results.get("freeform"), dict)
    has_audit = bool((state.get("frl_last_artifacts") or {}).get("run_dir"))

    st.markdown("### Стадии")
    status_cols = st.columns(5)
    status_cols[0].metric("1) Синтетика", "Готово" if has_chunks else "Ожидание")
    status_cols[1].metric("2) Transform", "Готово" if has_transform else "Ожидание")
    status_cols[2].metric("3) Structured", "Готово" if has_structured else "Ожидание")
    status_cols[3].metric("4) Freeform", "Готово" if has_freeform else "Ожидание")
    status_cols[4].metric("5) Audit", "Готово" if has_audit else "Ожидание")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "1. Синтетика",
            "2. Transform",
            "3. Structured",
            "4. Freeform",
            "5. Audit",
        ]
    )

    with tab1:
        st.caption(
            "Ручной запуск Stage-1: генерируем жирные synthetic chunks. "
            "Ни БД, ни лог-фетч тут не используются."
        )
        chunk_count = st.number_input(
            "L1 summaries",
            min_value=2,
            max_value=64,
            value=DEFAULT_SYNTH_CHUNK_COUNT,
            step=1,
            key="frl_stage1_chunk_count",
        )
        events_per_chunk = st.number_input(
            "Событий/summary",
            min_value=10,
            max_value=500,
            value=DEFAULT_SYNTH_EVENTS_PER_CHUNK,
            step=10,
            key="frl_stage1_events_per_chunk",
        )
        details_per_event = st.number_input(
            "Деталей/событие",
            min_value=1,
            max_value=8,
            value=DEFAULT_SYNTH_DETAILS_PER_EVENT,
            step=1,
            key="frl_stage1_details_per_event",
        )
        paragraphs_per_chunk = st.number_input(
            "Гипотез/summary",
            min_value=1,
            max_value=20,
            value=DEFAULT_SYNTH_PARAGRAPHS_PER_CHUNK,
            step=1,
            key="frl_stage1_paragraphs_per_chunk",
        )
        seed = st.number_input(
            "Seed",
            min_value=1,
            max_value=2_147_483_647,
            value=DEFAULT_SYNTH_SEED,
            step=1,
            key="frl_stage1_seed",
        )
        if st.button("Запустить этап 1: Сгенерировать chunks", key="frl_stage1_run", use_container_width=True):
            chunks = _build_synthetic_report_chunks(
                chunk_count=int(chunk_count),
                events_per_chunk=int(events_per_chunk),
                details_per_event=int(details_per_event),
                paragraphs_per_chunk=int(paragraphs_per_chunk),
                seed=int(seed),
            )
            state["frl_chunks"] = chunks
            state["frl_base_summary"] = ""
            state["frl_map_summaries_text"] = ""
            state["frl_last_results"] = {}
            state["frl_last_artifacts"] = {}
            _append_stage_event("stage1_synthetic", "done", f"chunks={len(chunks)}")
            st.success(f"Stage-1 готов: chunks={len(chunks)}")

        chunks_now = list(state.get("frl_chunks") or [])
        if chunks_now:
            st.dataframe(
                [
                    {
                        "chunk": idx + 1,
                        "chars": len(str(text or "")),
                        "lines": len(str(text or "").splitlines()),
                    }
                    for idx, text in enumerate(chunks_now)
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.text_area("Первый chunk (preview)", value=str(chunks_now[0] or ""), height=220)
            st.text_area("Последний chunk (preview)", value=str(chunks_now[-1] or ""), height=220)

    with tab2:
        st.caption("Ручной запуск Stage-2: преобразуем chunks -> base_summary + map_summaries_text.")
        st.write(f"Chunks на входе: `{len(list(state.get('frl_chunks') or []))}`")
        state["frl_user_goal"] = st.text_area(
            "Контекст инцидента (UI)",
            value=str(state.get("frl_user_goal") or ""),
            height=130,
            key="frl_stage2_user_goal",
        )
        state["frl_metrics_context"] = st.text_area(
            "Контекст метрик",
            value=str(state.get("frl_metrics_context") or ""),
            height=100,
            key="frl_stage2_metrics",
        )
        period_cols = st.columns(2)
        with period_cols[0]:
            state["frl_period_start"] = st.text_input(
                "Период start",
                value=str(state.get("frl_period_start") or ""),
                key="frl_stage2_period_start",
            )
        with period_cols[1]:
            state["frl_period_end"] = st.text_input(
                "Период end",
                value=str(state.get("frl_period_end") or ""),
                key="frl_stage2_period_end",
            )

        if st.button("Запустить этап 2: Собрать Transform", key="frl_stage2_run", use_container_width=True):
            chunks_in = list(state.get("frl_chunks") or [])
            if not chunks_in:
                _append_stage_event("stage2_transform", "error", "no_chunks")
                st.error("Нет chunks от Stage-1.")
            else:
                state["frl_base_summary"] = _merge_synthetic_chunks(chunks_in)
                state["frl_map_summaries_text"] = _build_map_summaries_text(chunks_in)
                state["frl_last_results"] = {}
                state["frl_last_artifacts"] = {}
                _append_stage_event(
                    "stage2_transform",
                    "done",
                    f"base_chars={len(str(state.get('frl_base_summary') or ''))}, map_chars={len(str(state.get('frl_map_summaries_text') or ''))}",
                )
                st.success("Stage-2 готов: base/map собраны.")

        st.write(
            f"base_summary chars: `{len(str(state.get('frl_base_summary') or ''))}` | "
            f"map_summaries chars: `{len(str(state.get('frl_map_summaries_text') or ''))}`"
        )
        st.text_area(
            "Base/Reduce summary",
            value=str(state.get("frl_base_summary") or ""),
            height=220,
            key="frl_stage2_base_preview",
        )
        st.text_area(
            "MAP summaries text",
            value=str(state.get("frl_map_summaries_text") or ""),
            height=220,
            key="frl_stage2_map_preview",
        )

    with tab3:
        st.caption("Ручной запуск Stage-3: секционная structured генерация.")
        show_merge_inputs_struct = st.checkbox("Показывать merge-входы", value=True, key="frl_stage3_show_merge")
        show_prompts_struct = st.checkbox("Показывать промпты", value=False, key="frl_stage3_show_prompts")
        api_base, model, api_key, timeout_seconds, max_retries, temperature, max_tokens = _render_llm_controls(
            "frl_stage3"
        )
        if st.button("Запустить этап 3: Structured", key="frl_stage3_run", use_container_width=True):
            base_summary = str(state.get("frl_base_summary") or "").strip()
            map_summaries_text = str(state.get("frl_map_summaries_text") or "")
            if not base_summary or not map_summaries_text.strip():
                _append_stage_event("stage3_structured", "error", "missing_transform")
                st.error("Сначала выполни Stage-2 (нужны base_summary и map_summaries_text).")
            elif not str(api_key or "").strip():
                _append_stage_event("stage3_structured", "error", "missing_api_key")
                st.error("Нужен API key.")
            else:
                llm_call = _make_groq_chat_call(
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    logger=deps.logger,
                )
                stats = {
                    "synthetic_chunks": len(state.get("frl_chunks") or []),
                    "base_summary_chars": len(base_summary),
                    "map_summaries_chars": len(map_summaries_text),
                    "final_sections_total": len(FINAL_REPORT_SECTIONS),
                }
                results = dict(state.get("frl_last_results") or {})
                results["structured"] = _run_sectional_generation(
                    kind="structured",
                    llm_call=llm_call,
                    base_summary=base_summary,
                    user_goal=str(state.get("frl_user_goal") or ""),
                    period_start=str(state.get("frl_period_start") or ""),
                    period_end=str(state.get("frl_period_end") or ""),
                    stats=stats,
                    metrics_context=str(state.get("frl_metrics_context") or ""),
                    map_summaries_text=map_summaries_text,
                    logger=deps.logger,
                )
                state["frl_last_results"] = results
                state["frl_last_artifacts"] = {}
                _append_stage_event(
                    "stage3_structured",
                    "done",
                    f"elapsed_sec={results['structured'].get('elapsed_sec')}",
                )
                st.success("Stage-3 готов.")

        structured_result = (state.get("frl_last_results") or {}).get("structured")
        if isinstance(structured_result, dict):
            _render_stage_result(
                kind="structured",
                result=structured_result,
                show_merge_inputs=bool(show_merge_inputs_struct),
                show_prompts=bool(show_prompts_struct),
            )

    with tab4:
        st.caption("Ручной запуск Stage-4: секционная freeform генерация.")
        show_merge_inputs_free = st.checkbox("Показывать merge-входы", value=True, key="frl_stage4_show_merge")
        show_prompts_free = st.checkbox("Показывать промпты", value=False, key="frl_stage4_show_prompts")
        api_base, model, api_key, timeout_seconds, max_retries, temperature, max_tokens = _render_llm_controls(
            "frl_stage4"
        )
        if st.button("Запустить этап 4: Freeform", key="frl_stage4_run", use_container_width=True):
            base_summary = str(state.get("frl_base_summary") or "").strip()
            map_summaries_text = str(state.get("frl_map_summaries_text") or "")
            if not base_summary or not map_summaries_text.strip():
                _append_stage_event("stage4_freeform", "error", "missing_transform")
                st.error("Сначала выполни Stage-2 (нужны base_summary и map_summaries_text).")
            elif not str(api_key or "").strip():
                _append_stage_event("stage4_freeform", "error", "missing_api_key")
                st.error("Нужен API key.")
            else:
                llm_call = _make_groq_chat_call(
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    logger=deps.logger,
                )
                stats = {
                    "synthetic_chunks": len(state.get("frl_chunks") or []),
                    "base_summary_chars": len(base_summary),
                    "map_summaries_chars": len(map_summaries_text),
                    "final_sections_total": len(FINAL_REPORT_SECTIONS),
                }
                results = dict(state.get("frl_last_results") or {})
                results["freeform"] = _run_sectional_generation(
                    kind="freeform",
                    llm_call=llm_call,
                    base_summary=base_summary,
                    user_goal=str(state.get("frl_user_goal") or ""),
                    period_start=str(state.get("frl_period_start") or ""),
                    period_end=str(state.get("frl_period_end") or ""),
                    stats=stats,
                    metrics_context=str(state.get("frl_metrics_context") or ""),
                    map_summaries_text=map_summaries_text,
                    logger=deps.logger,
                )
                state["frl_last_results"] = results
                state["frl_last_artifacts"] = {}
                _append_stage_event(
                    "stage4_freeform",
                    "done",
                    f"elapsed_sec={results['freeform'].get('elapsed_sec')}",
                )
                st.success("Stage-4 готов.")

        freeform_result = (state.get("frl_last_results") or {}).get("freeform")
        if isinstance(freeform_result, dict):
            _render_stage_result(
                kind="freeform",
                result=freeform_result,
                show_merge_inputs=bool(show_merge_inputs_free),
                show_prompts=bool(show_prompts_free),
            )

    with tab5:
        st.caption(
            "Ручной запуск Stage-5: сохраняем артефакты всех преобразований в файлы "
            "и показываем пути (дублируются в логах)."
        )
        if st.button("Запустить этап 5: Сохранить Audit", key="frl_stage5_run", use_container_width=True):
            results = dict(state.get("frl_last_results") or {})
            if not results:
                _append_stage_event("stage5_audit", "error", "no_results")
                st.error("Нет результатов Stage-3/4 для сохранения.")
            else:
                artifacts_payload = _persist_final_report_lab_artifacts(
                    user_goal=str(state.get("frl_user_goal") or ""),
                    metrics_context=str(state.get("frl_metrics_context") or ""),
                    period_start=str(state.get("frl_period_start") or ""),
                    period_end=str(state.get("frl_period_end") or ""),
                    chunks=list(state.get("frl_chunks") or []),
                    base_summary=str(state.get("frl_base_summary") or ""),
                    map_summaries_text=str(state.get("frl_map_summaries_text") or ""),
                    results=results,
                    logger=deps.logger,
                )
                state["frl_last_artifacts"] = artifacts_payload
                _append_stage_event("stage5_audit", "done", str(artifacts_payload.get("run_dir") or ""))
                st.success(f"Аудит сохранён: `{artifacts_payload.get('run_dir', '')}`")

        last_artifacts = state.get("frl_last_artifacts") or {}
        if isinstance(last_artifacts, dict) and last_artifacts.get("run_dir"):
            st.write(f"Каталог запуска: `{last_artifacts.get('run_dir')}`")
            file_rows = last_artifacts.get("files") or []
            if file_rows:
                st.dataframe(
                    [
                        {
                            "label": str(row.get("label") or ""),
                            "path": str(row.get("path") or ""),
                            "relative_path": str(row.get("relative_path") or ""),
                        }
                        for row in file_rows
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

        events_rows = list(state.get("frl_stage_events") or [])
        if events_rows:
            st.markdown("#### Журнал Этапов")
            st.dataframe(events_rows, use_container_width=True, hide_index=True)
